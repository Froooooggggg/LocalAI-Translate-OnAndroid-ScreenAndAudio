package com.screenast.app.translate.llama

import android.content.Context
import com.screenast.app.core.Lang
import com.screenast.app.core.Settings
import com.screenast.app.translate.ProviderStatus
import com.screenast.app.translate.TranslateChunk
import com.screenast.app.translate.TranslateRequest
import com.screenast.app.translate.TranslationException
import com.screenast.app.translate.TranslationProvider
import com.screenast.app.translate.TranslationQuality
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.channels.Channel
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.buffer
import kotlinx.coroutines.flow.channelFlow
import kotlinx.coroutines.withContext

/**
 * 把端侧 Qwen3.5-2B（经由 llama.cpp JNI）包装成 [TranslationProvider]。
 *
 * ## 与 NLLB 的关键差异
 *
 * | | NLLB 600M | Qwen3.5-2B |
 * |---|---|---|
 * | 首字（真机） | ~100 ms | **60~92 ms（走 NPU）** |
 * | 整句（真机） | ~1300 ms | **183~501 ms（走 NPU）** |
 * | 长句 | 会提前收尾，要靠 `pickToken` 的概率判据救回 | **不会截断** |
 * | 俚语/粗口 | 会**吞掉**（`the hell`、`bullshit`、`Damn it` 消失） | 完整译出 |
 * | 需要源语种码 | 是（FLORES code + forced BOS） | **否**（提示词里写目标语种就够了） |
 * | 退化风险 | 复读 | **随机掉回思考模式** → 必须过 [LlamaOutput] 净化 |
 *
 * ## 为什么每句都 `memoryClear`
 *
 * 本 App 走**无上下文**策略（理由见 `docs/断点续接单.md` §0.5 第五之二节）：
 * 带上下文会在 40 轮里随机吐垃圾 1~3 次，而且实测**不存在安全长度阈值** ——
 * 把某句挪到第 4 轮它就在第 4 轮翻，说明是「内容触发的」不是「长度触发的」。
 * 换来的跨句指代收益，抵不过「每 20 句里有一句变成乱码字幕」。
 *
 * 代价：前缀缓存用不上，每句都要全量 prefill。但那只值几十 token、~10ms。
 *
 * ## 线程
 *
 * 所有 native 调用都在 [LlamaRuntime] 的锁里串行。这里的 `withContext(Dispatchers.Default)`
 * 是为了把**阻塞**的 native 调用挪出主线程（接口约定是「collect 时发起请求」，
 * 调用方可能在主线程 collect，留在主线程就是 ANR）。
 */
class LlamaTranslationProvider(
    private val modelPath: String,
    /** 期望的后端。`auto` = 先试 HTP0，失败退 CPU。 */
    private val device: String,
    private val nCtx: Int,
    private val nThreads: Int,
    private val nBatch: Int,
    private val nGpuLayers: Int,
    private val maxNewTokens: Int,
    /** 写进 `ADSP_LIBRARY_PATH` 的**单个**目录，供 FastRPC 找 HTP 的 skel。 */
    private val dspLibDirs: String,
) : TranslationProvider {

    override val displayName: String
        get() = "端侧 Qwen3.5-2B（${loadedDevice ?: device}）"

    /** 实际生效的后端，加载后才有值。前端展示与日志都用它。 */
    @Volatile
    var loadedDevice: String? = null
        private set

    /** 加载耗时，用于把「首句慢」归因到加载而不是推理。 */
    @Volatile
    var lastLoadMs: Long = 0L
        private set

    @Volatile
    private var loadFailure: String? = null

    /**
     * 加载模型。**耗时 2~4 秒、常驻约 1.4GB**，绝不能在主线程调用。
     */
    private fun ensureLoaded() {
        if (LlamaRuntime.isLoaded) return

        val failed = loadFailure
        if (failed != null) throw TranslationException("端侧大模型加载失败：$failed")

        LlamaRuntime.dspLibDirs = dspLibDirs

        // `auto` 的降级顺序：NPU → CPU。
        // 为什么不是「NPU → GPU → CPU」：实测 Adreno 830 的 OpenCL 后端 decode 只有
        // 26.1 tok/s，**比 CPU 的 30.8 还慢**（prefill 快但同传场景下 prefill 只占小头）。
        val order = when (device.trim().lowercase()) {
            "auto", "" -> listOf("HTP0", "CPU")
            else -> listOf(device.trim())
        }

        var lastError = ""
        for (dev in order) {
            val t0 = System.currentTimeMillis()
            val h = LlamaRuntime.acquire(
                path = modelPath,
                nCtx = nCtx,
                nThreads = nThreads,
                nBatch = nBatch,
                device = dev,
                nGpuLayers = nGpuLayers,
            )
            val cost = System.currentTimeMillis() - t0
            if (h != 0L) {
                loadedDevice = dev
                lastLoadMs = cost
                return
            }
            lastError = "$dev: ${LlamaNative.nativeLastError()}"
        }
        loadFailure = lastError.ifEmpty { "没有可用后端" }
        throw TranslationException("端侧大模型加载失败：$loadFailure")
    }

    override fun translate(request: TranslateRequest): Flow<TranslateChunk> = channelFlow {
        val text = request.text.trim()
        if (text.isEmpty()) {
            send(TranslateChunk("", isFinal = true))
            return@channelFlow
        }

        // 目标语种决定了系统提示词。AUTO 不是合法目标，兜到中文。
        // 自定义语种（设置里手填的）优先：它绕过枚举，能翻任何语言。
        val target = if (request.target == Lang.AUTO) Lang.ZH else request.target
        // 识图侧只传了 request.source（按块猜的语种），没传 sourceLabel；
        // 这里补一次兜底，否则日文会落到通用的「把下面的内容翻译成…」上。
        val derivedSourceLabel = request.sourceLabel.trim().ifBlank {
            if (request.source != Lang.AUTO && request.source != target) request.source.label else ""
        }
        // 日文源必须走专用提示词：当前通用提示词会把对话型日语改写成日文。
        val isJapanese = request.source == Lang.JA || derivedSourceLabel.contains("日")
        // 日文分支统一用「简体中文」；其它语种保持原来的 target.label（中文/英文/韩文），
        // 避免改变已经验证过的英文 15 条 / 韩文 4 条提示词。
        val targetLabel = request.targetLabel.trim().ifBlank {
            if (target == Lang.ZH) "简体中文" else target.label
        }
        val system = when {
            isJapanese -> LlamaPrompt.systemForJapanese(targetLabel)
            request.targetLabel.isNotBlank() || request.sourceLabel.isNotBlank() ->
                LlamaPrompt.systemForLabel(
                    request.targetLabel.ifBlank { target.label },
                    request.sourceLabel,
                )
            else -> LlamaPrompt.systemFor(target)
        }
        val strictSystem = if (isJapanese) LlamaPrompt.systemForJapaneseStrict(targetLabel) else system
        // 中文提示词两次都失败时的英文兜底。实测：片假名游戏术语 / ASR 听错专名的句子，
        // 中文提示词会原样回日文或中英日混杂，英文提示词能稳定译成中文。
        val englishSystem = if (isJapanese) {
            LlamaPrompt.systemForJapaneseEnglish(englishTargetName(target, request.targetLabel))
        } else {
            system
        }
        // 术语密集内容的最后一级整句兜底（片假名游戏/技术词）。
        val gameSystem = if (isJapanese) {
            LlamaPrompt.systemForJapaneseGame(englishTargetName(target, request.targetLabel))
        } else {
            system
        }

        /**
         * 跑一次生成。抽成局部函数是为了失败链：中文 → 严格中文 → 英文 → 拆句英文。
         * [emitChunks] 为 false 时不往 UI 推中间态（拆句时只在最后发一次合并结果）。
         */
        suspend fun runOnce(
            sys: String,
            source: String = text,
            useHistory: Boolean = true,
            emitChunks: Boolean = true,
        ): String {
            // 有上文就走多轮提示词；没有就走定稿的单轮形态（字节级不变，别动）。
            // 兜底链里的英文/拆句重试不带上下文：上文里可能已经混进坏译文，隔离掉更稳。
            val prompt = if (useHistory && request.history.isNotEmpty()) {
                LlamaPrompt.translateWithHistory(source, sys, request.history)
            } else {
                LlamaPrompt.translate(source, sys)
            }
            val raw = StringBuilder()
            withContext(Dispatchers.Default) {
                ensureLoaded()

                // 每句独立：把上一句的 KV 清掉。不清的话本句的提示词会被追加在
                // 上一句后面，模型看到的是两句话连在一起 —— 这就是「隐式上下文」，
                // 而且它的量随上一句长度变化，比显式窗口更难复现。
                // 注意：**带上文时也照清**。上文是显式写进提示词的多轮对话，
                // 不是靠 KV 残留，两者不能混用（混用就是 §0.5 那次随机吐垃圾的形态）。
                LlamaRuntime.memoryClear()

                if (useHistory && request.history.isNotEmpty()) {
                    // 打印轮数与提示词长度：上下文一变长，首字延迟上升是否来自 prefill，
                    // 只能靠这两个数字判断（别只看「感觉变慢了」）。
                    android.util.Log.i(
                        TAG,
                        "带上文 ${request.history.size} 轮，提示词 ${prompt.length} 字符 / 本句 ${source.length} 字符",
                    )
                }

                LlamaRuntime.generate(prompt, maxNew = maxNewTokens) { piece ->
                    raw.append(piece)
                    if (emitChunks) {
                        // 流式净化：每来一片就重新净化**整个**缓冲区。
                        // 触发截断后文本只会变短，而 TranslateChunk.text 本来就是覆盖语义，
                        // 所以「已经显示了一段、后来又收回」是允许的（也正是我们要的行为）。
                        val cleaned = LlamaOutput.clean(raw.toString())
                        if (cleaned.text.isNotEmpty()) {
                            trySend(TranslateChunk(cleaned.text, isFinal = false))
                        }
                    }
                }
            }
            return raw.toString()
        }

        /**
         * 最后一道兜底：按句/读点拆开，用英文提示词逐段翻。
         * 只保留真正翻成目标语的段；一段都没有就返回空串（让 UI 显示未翻译）。
         */
        suspend fun translateClauses(): LlamaOutput.Cleaned {
            val parts = splitJapanese(text)
            if (parts.size <= 1) return LlamaOutput.Cleaned("")
            val translated = ArrayList<String>(parts.size)
            for (part in parts) {
                var picked = ""
                // 每段依次试：中文专用 → 英文 → 术语兜底 → 严格中文。
                // 任何一段都失败就**整句放弃**（返回空串），不拼半句——
                // 半句中文比「未翻译」更误导（实测 `味方のコスト…` 只翻出后半句）。
                for (sys in listOf(system, englishSystem, gameSystem, strictSystem)) {
                    val c = LlamaOutput.clean(
                        runOnce(
                            sys,
                            source = part,
                            useHistory = false,
                            emitChunks = false,
                        ),
                    )
                    if (!TranslationQuality.looksUntranslated(Lang.JA, target, part, c.text)) {
                        picked = c.text
                        break
                    }
                }
                if (picked.isBlank()) return LlamaOutput.Cleaned("")
                translated += picked
            }
            return LlamaOutput.Cleaned(joinClauseTranslations(translated, target))
        }

        try {
            var final = LlamaOutput.clean(runOnce(system))
            // 日文专用提示词已经把普通日语压到 0/12；这里兜模型偶尔的日文改写。
            // 非日文不重试：英文 15 条 / 韩文 4 条已经验证过，不要额外改变行为。
            if (isJapanese && TranslationQuality.looksUntranslated(request.source, target, text, final.text)) {
                android.util.Log.w(
                    TAG,
                    "首次输出仍像日文，换严格提示词重试一次：${final.text.take(40)}",
                )
                val retry = LlamaOutput.clean(runOnce(strictSystem))
                if (retry.text.isNotBlank()) final = retry
                if (TranslationQuality.looksUntranslated(request.source, target, text, final.text)) {
                    android.util.Log.w(TAG, "严格中文仍失败，换英文提示词：${final.text.take(40)}")
                    val english = LlamaOutput.clean(runOnce(englishSystem, useHistory = false))
                    if (english.text.isNotBlank()) final = english
                    if (TranslationQuality.looksUntranslated(request.source, target, text, final.text)) {
                        android.util.Log.w(TAG, "英文提示词仍失败，换术语兜底：${final.text.take(40)}")
                        val game = LlamaOutput.clean(runOnce(gameSystem, useHistory = false))
                        if (game.text.isNotBlank()) final = game
                    }
                    if (TranslationQuality.looksUntranslated(request.source, target, text, final.text)) {
                        android.util.Log.w(TAG, "术语兜底仍失败，尝试按句拆开：${final.text.take(40)}")
                        final = translateClauses()
                        if (TranslationQuality.looksUntranslated(request.source, target, text, final.text)) {
                            // 不再把日文当译文显示：返回空串，UI 显示「（未翻译）」，
                            // 原文仍在上面；同时 InterpretEngine 不会把它写进上文。
                            android.util.Log.w(TAG, "所有兜底均失败，返回空译文（UI 显示未翻译）")
                            final = LlamaOutput.Cleaned("")
                        }
                    }
                }
            }
            if (final.truncated) {
                android.util.Log.i(
                    TAG,
                    "输出已净化（${final.reason}）：${final.text.length} 字符",
                )
            }
            send(TranslateChunk(final.text, isFinal = true))
        } catch (t: Throwable) {
            if (t is CancellationException) throw t
            throw TranslationException("端侧大模型翻译失败：${t.message ?: t.javaClass.simpleName}", t)
        }
    }.buffer(Channel.CONFLATED)

    /**
     * 英文兜底提示词里的目标语名。中文目标必须写 `Simplified Chinese`——
     * 只写 `Chinese` 时 2B 偶尔会输出繁体。
     */
    private fun englishTargetName(target: Lang, customLabel: String): String {
        val label = customLabel.trim()
        if (label.isNotEmpty()) {
            return when {
                label.contains("繁") -> "Traditional Chinese"
                label.contains("简") || label.contains("中") -> "Simplified Chinese"
                label.contains("英") -> "English"
                label.contains("日") -> "Japanese"
                label.contains("韩") || label.contains("韓") -> "Korean"
                label.contains("泰") -> "Thai"
                label.contains("越") -> "Vietnamese"
                else -> label
            }
        }
        return when (target) {
            Lang.ZH -> "Simplified Chinese"
            Lang.EN -> "English"
            Lang.JA -> "Japanese"
            Lang.KO -> "Korean"
            else -> "Simplified Chinese"
        }
    }

    /**
     * 日文兜底拆句：先按句末标点，长句再按读点拆。
     *
     * 只用于「中文/英文整句都失败」的最后一道兜底；拆得过多会让译文碎，
     * 所以单句 ≤28 字不拆，最多 8 段。
     */
    private fun splitJapanese(text: String): List<String> {
        if (text.length <= 24) return listOf(text)
        val sentences = ArrayList<String>()
        val sb = StringBuilder()
        for (ch in text) {
            sb.append(ch)
            if (ch in "。！？!?") {
                val s = sb.toString().trim()
                if (s.isNotEmpty()) sentences += s
                sb.clear()
            }
        }
        val tail = sb.toString().trim()
        if (tail.isNotEmpty()) sentences += tail

        val out = ArrayList<String>()
        for (s in sentences) {
            if (s.length <= 28) {
                out += s
                continue
            }
            val pieces = s.split('、', '，').map { it.trim() }.filter { it.isNotEmpty() }
            if (pieces.size <= 1) {
                out += s
            } else {
                pieces.forEach { p ->
                    out += if (p.last() in "。！？!?") p else "$p。"
                }
            }
        }
        return out.filter { it.length >= 2 }.take(8)
    }

    /** 拼拆句译文；中文直接连，英文/其它加空格，缺标点时补一个句号。 */
    private fun joinClauseTranslations(parts: List<String>, target: Lang): String {
        val sb = StringBuilder()
        for (raw in parts) {
            val s = raw.trim()
            if (s.isEmpty()) continue
            if (sb.isNotEmpty()) {
                if (target == Lang.ZH || target == Lang.JA) {
                    if (!isSentenceEnd(sb.last()) && !isSentenceEnd(s.first())) sb.append('。')
                } else {
                    sb.append(' ')
                }
            }
            sb.append(s)
        }
        return sb.toString()
    }

    private fun isSentenceEnd(c: Char): Boolean =
        c in "。！？!?；;，,、"

    /**
     * 自检**只看模型文件在不在，不真跑一次翻译**（同 [com.screenast.app.translate.nllb.NllbTranslationProvider]）。
     */
    override suspend fun checkAvailability(): ProviderStatus {
        val f = java.io.File(modelPath)
        return when {
            !f.isFile -> ProviderStatus.Error("模型文件不存在：$modelPath")
            f.length() < 100L * 1024 * 1024 -> ProviderStatus.Error(
                "模型文件只有 ${f.length() / 1048576} MB，疑似损坏：$modelPath",
            )
            loadFailure != null -> ProviderStatus.Error(loadFailure!!)
            else -> ProviderStatus.Ok(
                "Qwen3.5-2B ${f.length() / 1048576} MB @ ${loadedDevice ?: device}",
            )
        }
    }

    /**
     * **故意不释放模型。**
     *
     * 模型是 1.4GB、加载 2~4 秒的进程级单例，被调试页和正式 provider 共享。
     * 每次切换 provider 都释放，等于让用户每切一次后端就等 3 秒重新读盘；
     * 更糟的是切程序内部件的瞬间（比如改设置触发重建）会让正在解码的 KV 直接消失。
     * 释放由 [LlamaRuntime.release] 显式调用（调试页的 `auto=release` / 内存紧张时）。
     *
     * 这与接口注释里「close 用来释放后端资源」并不冲突：那边说的是**持有**资源的实现。
     * 本类是**租用**资源，释放责任在单例持有者。
     */
    override fun close() {
        // 故意留空，理由见上。
    }

    companion object {
        private const val TAG = "LlamaProvider"

        /**
         * 从设置加载一个端侧大模型 provider。
         *
         * **这里就把模型读进来**（[ensureLoaded]），不是留到第一次 `translate()`。
         *
         * 为什么必须在这里：`InterpretEngine.ensureProvider` 用这行日志报加载耗时
         * `翻译后端：X 就绪（N ms）`。懒加载时 `f()` 只造了个空壳，日志会写成
         * **「就绪（1 ms）」**，而真正的 5 秒读盘发生在第一次 translate 里 ——
         * 于是第一句的「首字 7155 ms」看上去像推理慢，实际是加载。数据全被污染。
         * 更糟的是 `checkAvailability()` 在没加载时只能报「文件存在」，等于没有自检。
         *
         * 调用方是翻译线程（`InterpretEngine.runJob`），本来就是后台线程，阻塞得起。
         *
         * @throws TranslationException 找不到模型文件，或模型加载失败
         */
        fun load(context: Context, settings: Settings): LlamaTranslationProvider {
            val prefDir = settings.llamaModelDir.ifBlank { null }
            val file = LlamaModels.resolve(context, prefDir)
                ?: throw TranslationException(
                    "找不到大模型文件（.gguf）。查找位置：\n" + LlamaModels.diagnostics(context, prefDir),
                )
            val provider = LlamaTranslationProvider(
                modelPath = file.absolutePath,
                device = settings.llamaDevice,
                nCtx = settings.llamaNCtx,
                nThreads = settings.llamaNThreads,
                nBatch = settings.llamaNBatch,
                nGpuLayers = settings.llamaNGpuLayers,
                maxNewTokens = settings.llamaMaxNewTokens,
                dspLibDirs = LlamaModels.dspLibDirs(context),
            )
            provider.ensureLoaded()
            return provider
        }
    }
}
