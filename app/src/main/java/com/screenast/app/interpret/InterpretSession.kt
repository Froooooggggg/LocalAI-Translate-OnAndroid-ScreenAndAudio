package com.screenast.app.interpret

import android.content.Context
import android.content.SharedPreferences
import android.os.Handler
import android.os.Looper
import android.util.Log
import androidx.compose.runtime.mutableStateOf
import com.screenast.app.asr.AsrFactory
import com.screenast.app.asr.DictationHub
import com.screenast.app.asr.SenseVoiceAsr
import com.screenast.app.core.Lang
import com.screenast.app.core.Settings
import com.screenast.app.overlay.SubtitleOverlay
import com.screenast.app.translate.ProviderFactory
import com.screenast.app.translate.TranslationProvider
import com.screenast.app.translate.llama.LlamaModels
import com.screenast.app.translate.llama.LlamaTranslationProvider
import com.screenast.app.translate.nllb.NllbModels
import com.screenast.app.translate.nllb.NllbTokenizer
import com.screenast.app.translate.nllb.NllbTranslationProvider
import com.screenast.app.translate.nllb.NllbTranslator
import java.io.File
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

/**
 * 同传会话：**引擎 + 浮窗 + 全部输出状态**的唯一持有者，活在进程里而不是 Activity 里。
 *
 * 为什么要有这个对象（线路图 ⑥ 的第一步）：
 * 以前 `InterpretEngine` 和 `SubtitleOverlay` 都归 `InterpretActivity` 管，`onDestroy`
 * 里一句 `engine.close()` 就把一场同传结束了 —— 用户切到别的 App 看视频，界面一被系统
 * 回收，字幕就断了。可他要的恰恰是「划掉界面，浮窗还在跑」。
 *
 * 现在的分工：
 *  - **本对象**：引擎、浮窗、当前句/历史/日志/状态这些输出状态，以及把引擎回调
 *    翻译成这些状态的逻辑。生命周期 = 进程（用户点「停止」或进程被杀才结束）。
 *  - **Activity**：只负责「读设置 → 把设置灌进来 → 画 UI → 收用户操作」。
 *    它销毁了也不影响这里继续跑，重建后直接读同一批 Compose 状态即可恢复画面。
 *
 * 线程：引擎回调来自采集线程 / 翻译线程，写 Compose 状态一律切主线程
 * （[post] 分流），调用方不必再操心。
 */
object InterpretSession {

    private const val TAG = "AST.Session"

    // ------------------------------------------------------------------ 输出状态
    // 用 Compose 的 State 而不是 Flow：Activity 里原本就是 `mutableStateOf` 直接读
    // `.value`，换成 Flow 要把所有读取点和 UI 都改一遍，收益只有「更正统」。

    val liveSource = mutableStateOf("")
    val liveTarget = mutableStateOf("")
    val history = mutableStateOf("")
    val interpState = mutableStateOf("翻译后端：未启用")
    val stats = mutableStateOf("")
    val error = mutableStateOf("")
    val log = mutableStateOf("")
    val speaking = mutableStateOf(false)

    /** 翻译开关。**默认开**：点「开始采集」的意图就是「翻给我看」。 */
    val translateOn = mutableStateOf(true)

    // ------------------------------------------------------------------ 设置快照

    /** 当前生效的设置（已合并 adb 覆盖）。改它就等于改「下一句的行为」。 */
    @Volatile
    var settings: Settings = Settings()
        private set

    /** 写盘钩子，由 Activity 装（DataStore 的写是挂起的，本对象不直接碰协程）。 */
    @Volatile
    var persist: ((Settings) -> Unit)? = null

    private var appContext: Context? = null
    private var prefs: SharedPreferences? = null

    val engine: InterpretEngine? get() = engineRef

    @Volatile
    private var engineRef: InterpretEngine? = null

    var overlay: SubtitleOverlay? = null
        private set

    private var detach: (() -> Unit)? = null

    private val nllbLocation: NllbModels.Location? by lazy {
        appContext?.let { NllbModels.resolve(it) }
    }

    // 历史配对用：译文迟到时要按 id 找回它自己的原文
    private val srcById = LinkedHashMap<Long, String>()
    @Volatile
    private var liveId = 0L

    /** 最近一次推进历史区的整段文本，用于断句合并时撤回来。 */
    private var lastPushedHead = ""

    private val timeFormat = SimpleDateFormat("HH:mm:ss", Locale.US)
    private val main = Handler(Looper.getMainLooper())

    private fun post(block: () -> Unit) {
        if (Looper.myLooper() == Looper.getMainLooper()) block() else main.post(block)
    }

    /** 由 Activity 在 `onCreate` 里调一次；重复调用无害。 */
    fun install(context: Context) {
        appContext = context.applicationContext
        prefs = appContext?.getSharedPreferences("asr_debug", Context.MODE_PRIVATE)
    }

    // ------------------------------------------------------------------ 文本输出

    fun append(line: String) {
        Log.i(TAG, "UI: $line")
        post { log.value = (line + "\n" + log.value).take(6000) }
    }

    fun pushHistory(src: String, tgt: String, late: Boolean = false) {
        val ts = timeFormat.format(Date())
        val mark = if (late) " ⏱迟到" else ""
        val head = "[$ts]$mark $src\n    → ${tgt.ifEmpty { "（未翻译）" }}\n"
        lastPushedHead = head
        post { history.value = (head + history.value).take(5000) }
    }

    fun clearHistory() {
        history.value = ""
        lastPushedHead = ""
    }

    // ------------------------------------------------------------------ 设置

    fun updateSettings(eff: Settings, persistIt: Boolean = false) {
        settings = eff
        overlay?.applySettings(eff)
        if (persistIt) persist?.invoke(eff)
    }

    fun updateGeometry(widthDp: Float, heightDp: Float, xDp: Float, yDp: Float) {
        settings = settings.copy(
            subtitleWidthDp = widthDp, subtitleHeightDp = heightDp, offsetXDp = xDp, offsetYDp = yDp,
        )
        persist?.invoke(settings)
    }

    // ------------------------------------------------------------------ 引擎装配

    /**
     * 建引擎并挂到采集上。已建过就直接返回 true（幂等：界面切回来再点「开始」不会重建）。
     *
     * 必须在启动采集之前调用：[DictationHub] 是单槽位，没人听时音频直接丢。
     *
     * @param whisperLang Whisper 解码语种覆盖（null = 跟着源语言走）
     */
    fun attach(eff: Settings, whisperLang: String? = null): Boolean {
        val ctx = appContext
        if (ctx == null) {
            append("❌ 会话没安装上下文（install 未调用）")
            return false
        }
        if (engineRef != null) return true
        settings = eff

        val kind = AsrFactory.normalize(eff.asrModel)
        // 按「选中的后端」找根目录，而不是「任意后端」：模型可能分散在
        // filesDir/models/asr（老布局）与 /data/local/tmp/snat-models/asr（新布局）两处。
        val root = AsrFactory.resolveRootFor(kind, ctx.filesDir, prefs().getString("asrDir", "") ?: "")
        if (root == null) {
            append("❌ 找不到「${AsrFactory.labelOf(kind)}」的模型")
            append("   查找位置：${File(ctx.filesDir, "models/asr")} → /data/local/tmp/snat-models/asr")
            append("   该后端需要子目录 ${AsrFactory.subdirOf(kind)}")
            return false
        }
        val vad = AsrFactory.resolveVad(root, prefs().getString("vadFile", "") ?: "")
        if (vad == null) {
            append("❌ 找不到 ${SenseVoiceAsr.VAD_FILE}（离线后端做端点检测必需）")
            return false
        }
        if (!AsrFactory.installed(kind, root)) {
            append("❌ 选中的识别后端「${AsrFactory.labelOf(kind)}」模型不全：$root")
            return false
        }
        append("识别后端：${AsrFactory.labelOf(kind)}（模型目录 $root）")

        val loc = nllbLocation
        if (loc == null) append("⚠ 没找到 NLLB 模型，只能出原文（翻译开关不会生效）") else append("NLLB：${loc.source}")

        val e = try {
            InterpretEngine(
                assets = ctx.assets,
                asrDir = root,
                vadFile = vad,
                // 这组 lambda 每个窗口/每个任务读一次 settings 字段，所以拖动滑条
                // 「下一句就生效」，不需要重建引擎（重建要重载 239MB 模型）。
                forceCutMs = { settings.forceSegmentMs },
                contextRounds = { settings.contextRounds },
                asrFactory = {
                    AsrFactory.create(
                        kind = kind,
                        asrRoot = root,
                        vadFile = vad,
                        threads = 4,
                        punctEnabled = eff.asrPunct,
                        denoise = eff.asrDenoise,
                        whisperLang = whisperLang ?: eff.sourceLang.code,
                    )
                },
                fallbackSource = { settings.sourceLang },
                targetLang = { settings.targetLang },
                customTargetLang = { settings.customTargetLang },
                customSourceLang = { settings.customSourceLang },
                callback = callback,
            )
        } catch (t: Throwable) {
            append("❌ 引擎建立失败：${t.javaClass.simpleName}: ${t.message}")
            Log.e(TAG, "引擎建立失败", t)
            return false
        }

        // 翻译后端供应器：这里只是一个 lambda，不做任何加载。真正的加载发生在翻译线程
        // 第一次使用时，否则点一下按钮界面会卡死 3.5 秒（NLLB）/ 3~4 秒（2B）。
        if (translateOn.value) {
            val f = translatorFactory(eff)
            if (f != null) {
                e.translatorFactory = f
                interpState.value = translatorStatusText()
            } else {
                interpState.value = "翻译后端：找不到本地模型（只能出原文）"
            }
        } else {
            interpState.value = "翻译后端：未开启（点「翻译」后下一句起出译文）"
        }

        if (!e.ready) append("⚠ 引擎建起来了但 ASR 模型没加载成功，见上方错误")
        append("✅ 同传引擎就绪（ASR: ${e.asrName}${if (e.asrStreaming) "·流式" else ""}，模型 $root）")

        engineRef = e
        detach = DictationHub.attach(e)
        return true
    }

    /** 只拆引擎（换识别后端 / 点停止用），浮窗保留。 */
    fun detachEngine() {
        val e = engineRef
        engineRef = null
        // 顺序：先摘（不再收音频）再关（释放 native）。反了会有一次回调打到已释放的 recognizer。
        runCatching { detach?.invoke() }
        detach = null
        runCatching { e?.close() }
        post {
            speaking.value = false
            liveId = 0
            liveSource.value = ""
            liveTarget.value = ""
            interpState.value = "翻译后端：未启用"
        }
    }

    /** 点「停止」：引擎、浮窗、状态一起收，回到「没在跑」的样子。 */
    fun stopAll() {
        detachEngine()
        hideOverlay()
    }

    fun hasEngine(): Boolean = engineRef != null

    fun srcOf(id: Long): String? = srcById[id]

    // ------------------------------------------------------------------ 引擎回调

    private val callback = object : InterpretEngine.Callback {

        override fun onUtterance(id: Long, text: String, lang: Lang, samples: Int, decodeMs: Long) {
            val secs = samples / SenseVoiceAsr.SAMPLE_RATE.toFloat()
            Log.i(TAG, "识别[$id][${lang.code}] ${"%.2f".format(secs)}s 解码${decodeMs}ms  $text")
            post {
                // 新的一句上位：先把上一条「当前句」落进历史，再清空译文
                // （留着旧译文会显示「新原文 + 旧译文」，比空着更误导）。
                val prevSrc = liveSource.value
                val prevTgt = liveTarget.value
                if (prevSrc.isNotEmpty()) pushHistory(prevSrc, prevTgt)
                srcById[id] = text
                while (srcById.size > 40) srcById.remove(srcById.keys.first())
                liveId = id
                liveSource.value = text
                liveTarget.value = ""
            }
            // 浮窗是独立一路出口，不经过上面的界面状态，互不干扰。
            overlay?.onUtterance(id, text)
        }

        override fun onUtteranceMerged(intoId: Long, fromId: Long, source: String) {
            Log.i(TAG, "合并[$fromId→$intoId]  $source")
            post {
                if (liveId == fromId || liveId == intoId) {
                    liveId = intoId
                    liveSource.value = source
                }
                srcById[intoId] = source
                srcById.remove(fromId)
                val h = history.value
                if (lastPushedHead.isNotEmpty() && h.startsWith(lastPushedHead)) {
                    history.value = h.substring(lastPushedHead.length)
                    lastPushedHead = ""
                }
            }
            overlay?.merge(intoId, fromId, source)
        }

        override fun onTranslation(id: Long, text: String, isFinal: Boolean, firstChunkMs: Long, totalMs: Long) {
            if (isFinal) Log.i(TAG, "译文[$id] 首字${firstChunkMs}ms 整句${totalMs}ms  $text")
            post {
                if (id == liveId) {
                    liveTarget.value = text
                } else {
                    // 迟到的译文绝不能写进当前句（会变成「正在说的话」配「上一句的译文」），
                    // 落进历史区，按 id 找回它自己的原文配对。
                    pushHistory(srcById[id] ?: "(已滚动)", text, late = true)
                }
            }
            overlay?.onTranslation(id, text)
        }

        override fun onTranslatorState(text: String) {
            Log.i(TAG, text)
            post { interpState.value = text }
        }

        override fun onNotice(text: String) {
            Log.i(TAG, "提示：$text")
            append("⚠ $text")
        }

        override fun onError(message: String) {
            Log.e(TAG, message)
            post { error.value = message }
        }

        override fun onSpeaking(speaking2: Boolean) {
            post { speaking.value = speaking2 }
        }
    }

    // ------------------------------------------------------------------ 翻译后端

    fun translatorStatusText(s: Settings = settings): String {
        if (s.netApiEnabled) {
            val model = s.netApiModel.ifBlank { "(模型名未填)" }
            return "翻译后端：网络API · $model（每句一次 HTTP，不加载本地模型）"
        }
        return when (s.providerId) {
            "llama" -> "翻译后端：端侧 Qwen3.5-2B @ ${s.llamaDevice}（说第一句时加载，约 3 秒）"
            "nllb" -> "翻译后端：端侧 NLLB（说第一句时加载，约 3.5 秒）"
            else -> "翻译后端：云端 ${s.model}"
        }
    }

    /** 造翻译后端供应器。返回 null = 选中的后端没有可用模型。 */
    fun translatorFactory(st: Settings): (() -> TranslationProvider)? {
        val ctx = appContext ?: return null

        // 「网络API调用」必须排在 providerId 之前：用户开它的动机就是「不加载端侧 2B」，
        // 让 providerId=llama 抢先生效的话开关就白开了。
        if (st.netApiEnabled) {
            if (st.netApiKey.isBlank()) {
                append("❌ 「网络API调用」已开启，但没填 API Key")
                return null
            }
            append(
                "网络API：${st.netApiModel.ifBlank { "(模型名未填)" }} @ " +
                    "${st.netApiBaseUrl.ifBlank { "(接口地址未填)" }}（思考：${st.netApiThinking}）",
            )
            return { ProviderFactory.create(st) }
        }

        val prefDir = st.llamaModelDir.ifBlank { null }
        return when (st.providerId) {
            "llama" -> {
                val model = LlamaModels.resolve(ctx, prefDir)
                if (model == null) {
                    append("❌ 没找到大模型文件（.gguf）")
                    append(LlamaModels.diagnostics(ctx, prefDir))
                    null
                } else {
                    append("大模型：${model.absolutePath}（${model.length() / 1048576} MB）")
                    val f: () -> TranslationProvider = { LlamaTranslationProvider.load(ctx, st) }
                    f
                }
            }

            "nllb" -> {
                val loc = nllbLocation
                if (loc == null) {
                    null
                } else {
                    val f: () -> TranslationProvider = {
                        val tok = NllbTokenizer.fromAssets(ctx, "nllb/tokenizer.bin")
                        val tr = NllbTranslator(tok, loc.encoder, loc.decoder, loc.decoderPast)
                        NllbTranslationProvider(tr, loc.source, loc)
                    }
                    f
                }
            }

            else -> {
                val f: () -> TranslationProvider = { ProviderFactory.create(st) }
                f
            }
        }
    }

    /** 翻译开关。关 = 下一句起不再出译文，但识别照常。 */
    fun setTranslateOn(on: Boolean) {
        translateOn.value = on
        val e = engineRef
        if (on) {
            val f = translatorFactory(settings)
            if (f != null && e != null) {
                e.translatorFactory = f
                post { interpState.value = translatorStatusText() }
            }
        } else {
            e?.translatorFactory = null
            post { interpState.value = "翻译后端：已关闭（只出原文）" }
        }
    }

    // ------------------------------------------------------------------ 浮窗

    /** 显示浮窗；没权限返回 false（调用方去拉授权）。 */
    fun showOverlay(): Boolean {
        val ctx = appContext ?: return false
        if (!SubtitleOverlay.canDraw(ctx)) return false
        val ov = overlay ?: SubtitleOverlay(ctx).also {
            it.onGeometryChanged = { w, h, x, y -> updateGeometry(w, h, x, y) }
            overlay = it
        }
        ov.show(settings)
        return ov.isShowing
    }

    fun hideOverlay() {
        val ov = overlay
        overlay = null
        runCatching { ov?.hide() }
    }

    fun clearOverlay() {
        overlay?.clear()
    }

    fun overlayShowing(): Boolean = overlay?.isShowing == true

    private fun prefs(): SharedPreferences =
        prefs ?: throw IllegalStateException("InterpretSession.install 未调用")
}
