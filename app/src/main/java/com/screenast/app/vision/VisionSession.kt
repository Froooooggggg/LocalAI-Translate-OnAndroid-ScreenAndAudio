package com.screenast.app.vision

import android.content.Context
import android.graphics.Bitmap
import android.graphics.RectF
import android.os.Handler
import android.os.Looper
import android.os.SystemClock
import android.util.Log
import androidx.compose.runtime.mutableStateOf
import com.screenast.app.core.Settings
import com.screenast.app.core.SettingsStore
import com.screenast.app.interpret.InterpretSession
import com.screenast.app.core.Lang
import com.screenast.app.translate.TranslateChunk
import com.screenast.app.translate.TranslateRequest
import com.screenast.app.translate.TranslationProvider
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.flow.first
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

/**
 * 识图（⑬）的进程级会话：OCR 引擎、译文缓存、界面状态都放这里。
 *
 * 和 [InterpretSession] 同一个思路 —— 抓屏是前台服务在跑，界面只是视图。
 * 用户在识图页看列表、切回同传页、甚至划掉界面，抓屏与识别都不受影响。
 *
 * 线程模型：
 * - [onFrame] 由前台服务的采集线程调用，**立刻返回**（不阻塞 ImageReader 的队列）；
 * - 真正的 OCR 在 [scope] 的 Default 线程跑，一帧没跑完就丢掉下一帧（[busy]），
 *   屏幕静止时靠服务侧的画面指纹去重，不会白烧 CPU；
 * - 状态写回统一走主线程 Handler，避免后台改 Compose 快照引起重排抖动。
 */
object VisionSession {

    private const val TAG = "AST.Vision"

    /** 一屏里的一块文字（含它的译文）。坐标是**原图像素**，V3 贴回直接用它。 */
    data class Line(
        val text: String,
        val translation: String,
        val left: Float,
        val top: Float,
        val width: Float,
        val height: Float,
        val vertical: Boolean,
        val score: Float,
    )

    /** 界面读的四个状态。 */
    val lines = mutableStateOf<List<Line>>(emptyList())
    val capturing = mutableStateOf(false)
    val status = mutableStateOf("")
    val error = mutableStateOf("")

    /** 当前帧的源尺寸，界面拿来标「坐标基于 WxH」。 */
    val frameSize = mutableStateOf(0 to 0)

    private val scope = CoroutineScope(SupervisorJob() + Dispatchers.Default)
    private val main = Handler(Looper.getMainLooper())

    @Volatile private var busy = false

    /** 这一帧跑完要不要贴回屏幕（[captureOnce] 置位，一帧一清）。 */
    @Volatile private var pasteAfterFrame = false

    /**
     * 「单次识别」忙碌状态回调：true = 开始跑（悬浮键换成三角），false = 跑完/失败（换回暂停）。
     * 由 [VisionCaptureService] 挂上去驱动悬浮按钮的图标。
     */
    @Volatile var onBusy: ((Boolean) -> Unit)? = null
    @Volatile private var engine: OcrEngine? = null

    /** [engine] 装的是哪套权重；设置里切换 mobile/server 时靠它判断要不要重装。 */
    @Volatile private var engineVariant: VisionModels.Variant? = null
    @Volatile private var translator: TranslationProvider? = null
    @Volatile private var translatorKey = ""

    /**
     * 译文缓存：同一页反复抓帧时，文字几乎不变，靠它避免重复请求。
     *
     * 上限 300 条足够放下十几页漫画；超出按插入顺序淘汰最旧的。
     */
    private val cache = object : LinkedHashMap<String, String>(64, 0.75f, true) {
        override fun removeEldestEntry(eldest: MutableMap.MutableEntry<String, String>?): Boolean = size > 300
    }

    /** OCR 与翻译各自的累计耗时，界面一行显示。 */
    @Volatile private var lastOcrMs = 0L
    @Volatile private var lastTransMs = 0L
    @Volatile private var lastBlockCount = 0
    @Volatile private var lastTranslatedCount = 0

    fun log(line: String) {
        Log.i(TAG, line)
    }

    private fun publish(block: () -> Unit) {
        if (Looper.myLooper() == Looper.getMainLooper()) block() else main.post(block)
    }

    /** 抓屏开始/结束时调，负责清干净上一轮的状态。 */
    fun setCapturing(on: Boolean, reason: String = "") {
        publish {
            capturing.value = on
            if (!on) {
                status.value = reason
            }
        }
        log(if (on) "抓屏开始" else "抓屏结束：$reason")
    }

    fun clear() {
        publish {
            lines.value = emptyList()
            status.value = ""
            error.value = ""
            frameSize.value = 0 to 0
        }
        lastBlockCount = 0
        lastTranslatedCount = 0
        lastOcrMs = 0
        lastTransMs = 0
    }

    /**
     * 一帧进来了。
     *
     * 调用方（[VisionCaptureService]）已经做过「按间隔跳帧」和「画面没变就丢」，
     * 这里只负责排队：**上一帧还没跑完就直接扔掉这一帧**（OCR 150~1200ms，
     * 排队只会越积越多、界面显示的永远是旧画面）。
     */
    /**
     * 单次识别（悬浮「暂停键」用）：抓一帧 → OCR → 翻译 → 把译文贴回屏幕。
     *
     * 和 [onFrame] 的差别只有末尾那一步：贴回是**一次性**动作（画完 8 秒自动消失），
     * 自动抓屏那条路过不停贴，否则每隔 2 秒盖一层，屏幕没法看。
     */
    fun captureOnce(ctx: Context, bitmap: Bitmap) {
        if (busy) {
            Log.i(TAG, "上一次识别还没跑完，这次点击先忽略")
            publish { onBusy?.invoke(false) }
            bitmap.recycle()
            return
        }
        pasteAfterFrame = true
        onFrame(ctx, bitmap)
    }

    /** 上一帧还没跑完（界面上按钮点一片没反应时要能说清原因）。 */
    fun isBusy(): Boolean = busy

    fun onFrame(ctx: Context, bitmap: Bitmap) {
        if (busy) {
            bitmap.recycle()
            return
        }
        busy = true
        publish { onBusy?.invoke(true) }
        scope.launch {
            try {
                // 先读设置再装引擎：装哪套权重（mobile / server）本来就是设置决定的。
                Log.i(TAG, "取设置…")
                val st = SettingsStore(ctx).settings.first()
                Log.i(
                    TAG,
                    "设置读到了：translate=${st.visionTranslate} " +
                        "target=${st.visionTargetLang.code} source=${st.visionSourceLang.code} " +
                        "provider=${st.providerId} ocrServer=${st.visionOcrServer}",
                )
                val eng = buildEngine(ctx, st)
                val t0 = SystemClock.elapsedRealtime()
                val blocks = eng.recognize(bitmap)
                val ocrMs = SystemClock.elapsedRealtime() - t0

                // 状态栏那一行（时间/信号/电量图标）总会被 OCR 成「門*C公i」这种噪声，
                // 贴回时更是糊在最上面一行的位置 —— 按高度比例划一刀，比按像素更好移植。
                val statusBarCut = bitmap.height * STATUS_BAR_RATIO
                val rawKept = blocks.filter { b ->
                    b.text.isNotBlank() &&
                        b.height >= MIN_HEIGHT_PX &&
                        b.width >= MIN_WIDTH_PX &&
                        b.score >= MIN_SCORE &&
                        b.top >= statusBarCut
                }
                // 中文原文直接丢掉：本来就看不懂才要翻译/擦除，中文既不用翻也不用擦，
                // 留着还会白占「每帧最多翻 10 条」的配额（用户 m06585 的要求）。
                val noChinese = rawKept.filterNot { VisionInpaint.isChinese(it.text) }
                val zhDropped = rawKept.size - noChinese.size
                // 太短的块基本是噪声（`ln`/`SV`/`6175` 这种）：用户 m04363 要求
                // 「长度小于 n 的视为无效，识别后直接剔除，不参与翻译」，n 可在设置里 0~5 调。
                // 判据用 trim 之后的长度，标点也算一个字符。
                val minChars = st.visionMinChars.coerceIn(0, 5)
                val kept = if (minChars <= 0) noChinese else noChinese.filter { it.text.trim().length >= minChars }
                val shortDropped = noChinese.size - kept.size
                lastOcrMs = ocrMs
                lastBlockCount = blocks.size
                if (zhDropped > 0) Log.i(TAG, "跳过中文块 $zhDropped 个（共 ${rawKept.size} 个候选）")
                if (shortDropped > 0) Log.i(TAG, "跳过过短块 $shortDropped 个（< $minChars 字符）")

                // 同一区域的多行先拼成一段：漫画一页 30~40 个框，逐框翻译既慢又断句。
                // 拼接后译文是整段、贴回去也只有一个块，读起来才像人话（用户 m03318 要求）。
                val merged = mergeBlocks(kept)
                if (merged.size != kept.size) {
                    Log.i(TAG, "同区域自动拼接：${kept.size} 块 → ${merged.size} 段")
                }

                val w = bitmap.width
                val h = bitmap.height
                fun snapshotLines() = merged.map { b ->
                    Line(
                        text = b.text,
                        translation = cache[b.text].orEmpty(),
                        left = b.left,
                        top = b.top,
                        width = b.width,
                        height = b.height,
                        vertical = b.vertical,
                        score = b.score,
                    )
                }
                publish {
                    frameSize.value = w to h
                    lines.value = snapshotLines()
                    status.value = buildStatus(w, h)
                }

                // 翻译：只翻还没缓存的、且够长的块，一帧最多 [MAX_TRANSLATE_PER_FRAME] 条。
                if (st.visionTranslate && st.toolTargetIsTranslatable()) {
                    translateMissing(ctx, st, merged.map { it.text })
                    // ⚠️ 翻完必须**重新发一次**列表：上面那次 publish 发生在翻译之前，
                    // 里面只有缓存里的旧译文；不补这一次，界面看不到新译文，
                    // 贴片也会拿到一份「译文全空」的快照 —— 真机现象就是
                    // 日志里有「开始翻译/翻译完成」但屏幕上不贴字。
                    publish {
                        lines.value = snapshotLines()
                        status.value = buildStatus(w, h)
                    }
                }

                // 悬浮文字列表（用户 m04158）：屏幕上字太小、或贴片放不下时，
                // 同一份原文 + 译文在浮窗里可以随意放大/滚动着看。
                // 用户 m04287 改成「常驻 + 历史」：窗口在开始抓屏时就挂上（服务里），这里只负责追加，
                // 静止画面重复认到的同一批字由 VisionFloatList 自己去重。
                if (st.visionFloatList) {
                    if (VisionFloatList.show(ctx, st)) {
                        VisionFloatList.render(st, snapshotLines())
                    } else {
                        Log.w(TAG, "浮窗没加上：${VisionFloatList.lastError() ?: "未知原因"}")
                    }
                } else if (VisionFloatList.isShowing()) {
                    VisionFloatList.hide()
                }

                if (pasteAfterFrame) {
                    pasteAfterFrame = false
                    // 贴回用的是**这一帧刚算完的结果**，不是界面里可能已经换过的下一帧。
                    val snapshot = snapshotLines()
                    val paste = snapshot.filter { it.translation.isNotBlank() }
                    Log.i(TAG, "准备贴片：${paste.size}/${snapshot.size} 块有译文")
                    if (paste.isEmpty()) {
                        Log.i(TAG, "没有译文可贴（翻译关着或后端不可用）")
                        publish { status.value = buildStatus(w, h) + " · 贴回 0 块（无译文）" }
                    } else if (st.visionEraseMode == 1) {
                        // 模糊档：不加载模型，把原文那几块自己糊掉垫底再写译文。
                        // 用户：「之前在贴片翻译旁边不还有个模糊背景的嵌字方式，那个好用」。
                        val t0 = SystemClock.elapsedRealtime()
                        val img = VisionInpaint.renderBlur(ctx, bitmap, snapshot, st.visionTextColor)
                        Log.i(TAG, "模糊档整帧渲染完成 ${SystemClock.elapsedRealtime() - t0}ms")
                        val pasted = VisionInpaint.countTargets(snapshot)
                        VisionResultOverlay.showBitmap(ctx, img, w, h)
                        publish { status.value = buildStatus(w, h) + " · 模糊贴回 $pasted 块" }
                    } else if (st.visionEraseMode == 2) {
                        // 擦除档：补背景 → 写译文 → 整帧贴回。
                        // 补背景用 MI-GAN；用户把「补背景」关掉时走程序贴片（框四周平均色），
                        // 不加载那 27MB 模型，几毫秒出图。
                        val t0 = SystemClock.elapsedRealtime()
                        // 模型没导入时 ensureLoaded 返回 false —— 不能直接走 render，
                        // 那样会在 native 层拿到空 session。退回程序贴片，并在状态行说明原因。
                        val canInpaint = st.visionInpaint && VisionInpaint.ensureLoaded(ctx)
                        val img = if (canInpaint) {
                            VisionInpaint.render(ctx, bitmap, snapshot, st.visionTextColor)
                        } else {
                            VisionInpaint.renderFlat(ctx, bitmap, snapshot, st.visionTextColor)
                        }
                        Log.i(TAG, "擦除档整帧渲染完成 ${SystemClock.elapsedRealtime() - t0}ms")
                        val pasted = VisionInpaint.countTargets(snapshot)
                        VisionResultOverlay.showBitmap(ctx, img, w, h)
                        val how = if (canInpaint) "擦除贴回" else "程序贴回"
                        if (st.visionInpaint && !canInpaint) {
                            publish { error.value = "补背景模型没导入，这次用程序贴片（设置→离线模型）" }
                        }
                        publish { status.value = buildStatus(w, h) + " · $how $pasted 块" }
                    } else {
                        VisionResultOverlay.show(ctx, snapshot, w, h, st.visionTextColor)
                        publish { status.value = buildStatus(w, h) + " · 贴片 ${paste.size} 块" }
                    }
                }
            } catch (t: Throwable) {
                Log.e(TAG, "处理帧失败", t)
                publish { error.value = "❌ 处理帧失败：${t.javaClass.simpleName}: ${t.message}" }
            } finally {
                busy = false
                publish { onBusy?.invoke(false) }
                if (!bitmap.isRecycled) bitmap.recycle()
            }
        }
    }

    private fun buildStatus(w: Int, h: Int): String {
        val sb = StringBuilder()
        sb.append("${w}x${h} · 识别 $lastBlockCount 块").append(" · OCR ${lastOcrMs}ms")
        if (lastTransMs > 0) sb.append(" · 翻译 ${lastTranslatedCount} 条（${lastTransMs}ms）")
        return sb.toString()
    }

    /**
     * 把「同一区域」的若干块拼成一段（用户 m03318：同一区域的文本自动拼接）。
     *
     * 判定标准是**排版相邻**，不是语义：
     * - 横排两块：横向投影重叠 ≥ 较小宽度的 50%，且竖直间距在 [-0.35h, 0.75h] 之间（允许轻微重叠）；
     * - 竖排两块：竖直投影重叠 ≥ 较小高度的 50%，且横向间距在 [-0.35w, 0.75w] 之间。
     *
     * 拼接顺序：横排按 top 升序、竖排按 left 降序（日漫从右往左读）。
     * 拼接文本中日文之间不加空格（日文行尾换行不加空格才是正常排版），拉丁文之间加一个空格。
     * 合成块的框取并集，这样贴回去是一整片，不会一行一块碎成马赛克。
     */
    private fun mergeBlocks(blocks: List<OcrEngine.TextBlock>): List<OcrEngine.TextBlock> {
        if (blocks.size <= 1) return blocks
        val used = BooleanArray(blocks.size)
        val out = ArrayList<OcrEngine.TextBlock>(blocks.size)

        // 先横排、后竖排：两类互不影响，分开处理省得互相干扰。
        for (vertical in listOf(false, true)) {
            val order = blocks.indices
                .filter { blocks[it].vertical == vertical }
                .sortedWith(
                    if (vertical) compareByDescending { blocks[it].left } else compareBy { blocks[it].top },
                )
            for (i in order) {
                if (used[i]) continue
                used[i] = true
                val g = ArrayList<Int>()
                g.add(i)
                // 贪心：反复扫，直到这一组不再变长（一段文字通常 2~6 行，代价可忽略）。
                var grew = true
                while (grew) {
                    grew = false
                    val box = union(g, blocks)
                    for (j in order) {
                        if (used[j]) continue
                        val b = blocks[j]
                        if (b.vertical != vertical) continue
                        val adjacent =
                            if (!vertical) {
                                val overlap = minOf(box.right, b.right) - maxOf(box.left, b.left)
                                val gap = b.top - box.bottom
                                overlap >= minOf(box.width(), b.width) * 0.5f &&
                                    gap >= -box.height() * 0.35f &&
                                    gap <= box.height() * 0.75f
                            } else {
                                val overlap = minOf(box.bottom, b.bottom) - maxOf(box.top, b.top)
                                val gap = box.left - b.right
                                overlap >= minOf(box.height(), b.height) * 0.5f &&
                                    gap >= -box.width() * 0.35f &&
                                    gap <= box.width() * 0.75f
                            }
                        if (adjacent) {
                            g.add(j)
                            used[j] = true
                            grew = true
                            break
                        }
                    }
                }
                out.add(combine(g.map { blocks[it] }, vertical))
            }
        }
        return out
    }

    private fun union(idx: List<Int>, blocks: List<OcrEngine.TextBlock>): RectF {
        var l = Float.MAX_VALUE
        var t = Float.MAX_VALUE
        var r = -Float.MAX_VALUE
        var bo = -Float.MAX_VALUE
        idx.forEach { i ->
            val b = blocks[i]
            l = minOf(l, b.left); t = minOf(t, b.top)
            r = maxOf(r, b.right); bo = maxOf(bo, b.bottom)
        }
        return RectF(l, t, r, bo)
    }

    private fun combine(group: List<OcrEngine.TextBlock>, vertical: Boolean): OcrEngine.TextBlock {
        val sorted = if (vertical) group.sortedByDescending { it.left } else group.sortedBy { it.top }
        val sb = StringBuilder()
        sorted.forEach { b ->
            val piece = b.text.trim()
            if (piece.isEmpty()) return@forEach
            if (sb.isNotEmpty() && needsSpace(sb.last(), piece.first())) sb.append(' ')
            sb.append(piece)
        }
        val u = union(group.indices.toList(), group)
        val quad = floatArrayOf(
            u.left, u.top,
            u.right, u.top,
            u.right, u.bottom,
            u.left, u.bottom,
        )
        return OcrEngine.TextBlock(
            text = sb.toString(),
            quad = quad,
            score = group.minOf { it.score },
            vertical = vertical,
        )
    }

    /** 中日韩字符之间不加空格，拉丁文之间加。 */
    private fun needsSpace(prev: Char, next: Char): Boolean {
        fun cjk(c: Char) = c.code in 0x3040..0x30FF || c.code in 0x4E00..0x9FFF ||
            c.code in 0xAC00..0xD7AF || c.code in 0xFF00..0xFFEF
        return !cjk(prev) && !cjk(next)
    }

    /**
     * 按设置选权重并装载引擎。
     *
     * 缓存 key 带上变体：用户在设置里从 mobile 切到 server（或切回来），
     * 必须换一套会话重装；只缓存一个 `engine` 的话切了也不生效，看起来像开关失灵。
     * 想要的那套没装时退到已装的那套（[VisionModels.resolveOrFallback]），并在状态行里说清楚。
     */
    private fun buildEngine(ctx: Context, st: Settings): OcrEngine {
        val want = if (st.visionOcrServer) VisionModels.Variant.SERVER else VisionModels.Variant.MOBILE
        val (files, degraded) = VisionModels.resolveOrFallback(ctx, want)
            ?: error("OCR 模型不在 files/models/ocr/")
        if (degraded) log("想要 ${want.label}，只装了 ${files.variant.label}，先按它跑")
        engine?.takeIf { engineVariant == files.variant }?.let { return it }
        runCatching { engine?.close() }
        log(
            "装载 OCR：${files.variant.label} · det ${files.det.length() / 1024}KB / " +
                "rec ${files.rec.length() / 1024}KB · 最长边 ${files.variant.maxSide}",
        )
        return OcrEngine(ctx, files.det, files.rec, maxSide = files.variant.maxSide)
            .also {
                engine = it
                engineVariant = files.variant
            }
    }

    /**
     * 补齐这一帧里还没翻译过的文字。
     *
     * 为什么按块翻而不是整屏翻：贴回（V3）是**按块**写回原坐标，翻译就必须保持块级对应；
     * 整屏一起发给模型，行数一多它会自己合并/丢行，坐标就对不上了。
     */
    private suspend fun translateMissing(ctx: Context, st: Settings, texts: List<String>) {
        val todo = texts
            .filter { it.length >= 2 && it.length <= 200 }
            .filter { cache[it] == null }
            .distinct()
            .take(MAX_TRANSLATE_PER_FRAME)
        if (todo.isEmpty()) return

        val t = provider(ctx, st) ?: return
        Log.i(TAG, "开始翻译 ${todo.size} 条 → ${st.visionTargetLang.code}")
        val t0 = SystemClock.elapsedRealtime()
        var done = 0
        var echoSkipped = 0
        for (src in todo) {
            // 源语言：设置里指定了就用它，否则按块猜。
            // 必须猜 —— NLLB 是单语向模型，`Lang.AUTO` 会直接抛
            // 「端侧 NLLB 需要明确的源语言」（真机实测整页翻译全失败）。
            // 一页里日文 + 英文混排很常见，所以**按块**猜，不是整屏一个语种。
            val from = if (st.visionSourceLang != Lang.AUTO) st.visionSourceLang else guessLang(src)
            // ⚠️ 每个 TranslateChunk.text 都是**累计全文**，不是增量：三个后端
            // （NLLB 的 onProgress / llama 的 partial / 网络API 的 SSE accumulated）都发
            // 「到目前为完整的一句话」，末帧再发一次完整结果。这里必须 takeLast 覆盖，
            // 写成 sb.append(c.text) 会把每个前缀都拼进去，真机表现是一屏
            // 「【【网【网络【网络】【网络】今…」这种层层叠叠的垃圾。
            var out = ""
            try {
                out = runTranslate(t, src, from, st.visionTargetLang, nudge = false)
                // 模型偶尔直接吐回原文（尤其短句、专有名词、整句都是符号时）。
                // 这种「译文」贴上去等于没翻，还会把擦掉的原字原样写回。
                // 擦除档的块尤其不能省：原文已经被抹掉了，宁可多试一次
                // （第二次把原文包进「」，破坏复读的连续性），实在不行才跳过，
                // 跳过的那块保留原文可见，等于「不贴」而不是「贴回原文」。
                if (isEcho(src, out)) {
                    Log.w(TAG, "译文=原文，换写法重译一次：${src.take(30)}")
                    val retry = runTranslate(t, src, from, st.visionTargetLang, nudge = true)
                    out = retry
                    if (isEcho(src, out) || out.isBlank()) {
                        Log.w(TAG, "重译仍无有效译文，跳过该块：${src.take(30)}")
                        echoSkipped++
                        continue
                    }
                } else if (out.isBlank()) {
                    // 空译文（后端抽风 / 截断）也重试一次，再空才跳过。
                    Log.w(TAG, "译文为空，重试一次：${src.take(30)}")
                    out = runTranslate(
                        t, src, from, st.visionTargetLang, nudge = true,
                        targetLabel = st.customTargetLang.trim(),
                        sourceLabel = st.customSourceLang.trim(),
                    )
                    if (out.isBlank() || isEcho(src, out)) {
                        echoSkipped++
                        continue
                    }
                }
                cache[src] = out
                done++
                publish {
                    lines.value = lines.value.map { if (it.text == src) it.copy(translation = out) else it }
                }
            } catch (t2: Throwable) {
                Log.w(TAG, "翻译失败：$src", t2)
                publish { error.value = "❌ 翻译失败：${t2.message}" }
            }
        }
        lastTransMs = SystemClock.elapsedRealtime() - t0
        lastTranslatedCount = done
        if (echoSkipped > 0) Log.i(TAG, "共跳过 $echoSkipped 条（重译后仍无有效译文）")
    }

    /**
     * 译一条，返回**累计末帧**的全文。
     *
     * @param nudge true 时把原文包进「」，用来打断「原样复读」——模型看到引号
     *   通常会把括号内当成待译内容而不是照抄；返回值再把括号剥掉。
     */
    private suspend fun runTranslate(
        provider: TranslationProvider,
        src: String,
        from: Lang,
        target: Lang,
        nudge: Boolean,
        /** 设置里手填的目标语种；非空时进提示词（识图也吃同一套自定义语种）。 */
        targetLabel: String = "",
        sourceLabel: String = "",
    ): String {
        val payload = if (nudge) "「$src」" else src
        var out = ""
        provider.translate(
            TranslateRequest(
                payload, from, target,
                revision = if (nudge) 1 else 0,
                targetLabel = targetLabel,
                sourceLabel = sourceLabel,
            ),
        )
            .collect { c: TranslateChunk -> out = c.text }
        return out.trim().removePrefix("「").removeSuffix("」").trim()
    }

    /**
     * 译文是不是就是把原文抄了一遍。
     *
     * 比较前去掉空白和标点、统一大小写：模型常常只改了空格或标点，
     * 那也算复读（贴上去用户看到的是同一句话）。
     */
    private fun isEcho(src: String, out: String): Boolean {
        if (out.isBlank()) return false
        fun norm(s: String): String = s.filter { it.isLetterOrDigit() }.lowercase()
        val a = norm(src)
        val b = norm(out)
        if (a.isEmpty() || b.isEmpty()) return false
        return a == b
    }

    private fun provider(ctx: Context, st: Settings): TranslationProvider? {
        val key = "${st.providerId}|${st.netApiEnabled}|${st.netApiModel}|${st.visionTargetLang.code}|${st.llamaModelDir}"
        translator?.let { if (key == translatorKey) return it }
        // 配置变了才重建：端侧 2B 装载要几秒，不能每帧都来一次。
        runCatching { translator?.close() }
        val factory = InterpretSession.translatorFactory(st)
        if (factory == null) {
            Log.w(TAG, "翻译后端没有可用工厂（provider=${st.providerId}）——见同传页日志")
            publish { error.value = "❌ 翻译后端不可用（provider=${st.providerId}），见同传页日志" }
            return null
        }
        val p = runCatching { factory() }.getOrElse {
            Log.e(TAG, "翻译后端装载失败", it)
            publish { error.value = "❌ 翻译后端装载失败：${it.message}" }
            return null
        }
        translator = p
        translatorKey = key
        log("翻译后端：${p.displayName}")
        return p
    }

    /** 停止时释放：先关翻译后端（可能占着 1.4GB 的 2B），再关 OCR。 */
    fun release() {
        scope.launch {
            runCatching { translator?.close() }
            translator = null
            translatorKey = ""
            runCatching { engine?.close() }
            engine = null
            cache.clear()
            withContext(Dispatchers.Main) {
                lines.value = emptyList()
                status.value = ""
            }
        }
    }

    // ---- 过滤阈值：这些是「值不值得翻/贴」的界线，V3 贴回也复用同一套 ----
    private const val MIN_HEIGHT_PX = 14f
    private const val MIN_WIDTH_PX = 6f
    private const val MIN_SCORE = 0.5f

    /** 顶部这一条（状态栏）里的文字不算内容：它既不该翻译，也不该贴。 */
    private const val STATUS_BAR_RATIO = 0.025f
    /**
     * 一帧最多翻多少条。
     *
     * 原来只有 10 —— 用户 m03182 要求「翻译完之后全贴在屏幕上」，就不能只翻一页里的前十条：
     * 漫画一页常有 20~40 块。端侧 2B 约 0.3s/条，48 条约 15 秒，手动模式（点一次才抓）能接受；
     * 自动抓帧模式下超出的部分下次翻到同一张图会命中 300 条译文缓存。
     */
    private const val MAX_TRANSLATE_PER_FRAME = 48

    /**
     * 按字符集猜这一块的语种。
     *
     * 为什么不用模型判语种：识别（SenseVoice）那条路是拿音频判的，这里是纯文本，
     * 为了 0.1ms 的判断去加载一个语种模型不值得；漫画/截图的文字用「假名 > 谚文 > 汉字 > 拉丁」
     * 这套优先级已经够准（日文里必然含假名，纯汉字按中文处理 —— 这正是我们要的行为：
     * 汉字翻成中文基本无损，翻成别的语种反而会错）。
     */
    private fun guessLang(text: String): Lang {
        if (text.any { it in 'ぁ'..'ゖ' || it in 'ァ'..'ヺ' || it == 'ー' }) return Lang.JA
        if (text.any { it in '가'..'힣' || it in 'ㄱ'..'ㅎ' }) return Lang.KO
        if (text.any { it in '\u4e00'..'\u9fff' }) return Lang.ZH
        return Lang.EN
    }
}

/** `Settings` 里没有这个便利方法，写成扩展放在这里，避免动 core 包。 */
private fun Settings.toolTargetIsTranslatable(): Boolean = visionTargetLang != Lang.AUTO
