package com.screenast.app.vision

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.Rect
import android.graphics.RectF
import android.graphics.Typeface
import android.util.Log
import ai.onnxruntime.OnnxJavaType
import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder

/**
 * 「擦除档」：用 MI-GAN-512 把原文抹掉、按背景纹理补全，再把译文写回去。
 *
 * 模型是 Picsart 的 MI-GAN（ICCV 2023，专为手机设计）导出的**完整 pipeline** ONNX：
 * 输入 `image`（uint8 NHWC → 实际是 NCHW `[1,3,H,W]`，0~255）与 `mask`
 * （uint8 `[1,1,H,W]`，**255 = 保留，0 = 要补**），内部自带「按 mask 求外接框 → 裁到 512 →
 * 补全 → 缩回 → 与原图按 mask 羽化混合」，输出 `result`（uint8 `[1,3,H,W]`）。
 *
 * 手机上的取舍：
 * - 只把**有译文、且横排**的框挖洞：竖排块写不下译文，挖了就等于把原文也删了；
 * - 送模型前先自己裁到所有洞的外接框（+[PAD]），避免把整块 1440×3200 塞进去；
 * - 补完的字用「底色亮度」自动选黑字/白字，避免白底上写白字。
 */
object VisionInpaint {

    private const val TAG = "AST.Inpaint"
    /** 相对 `files/models` 的路径（与 ModelImporter 清单里的 rel 一致）。 */
    private const val ASSET = "inpaint/migan_pipeline_v2.onnx"

    /** 洞外接框四周再留一圈，给模型一点上下文。 */
    private const val PAD = 24

    /** 送去模型的图最长边上限（模型内部反正会缩到 512，超过这个只是白烧 CPU）。 */
    private const val MAX_SIDE = 1024

    /** 聚合出来的区域最大边长：超过就拆开，别让 512 的有效分辨率摊到一大片上。 */
    private const val REGION_MAX = 720

    @Volatile private var env: OrtEnvironment? = null
    @Volatile private var session: OrtSession? = null

    /** 「模型没导入」只提示一次，免得每帧刷日志。 */
    @Volatile private var warnedMissing = false

    fun isLoaded(): Boolean = session != null

    /**
     * 这块原文是不是「纯中文」—— 是的话不翻、不擦、不写。
     *
     * 用户 m03182 的规则：**无片假名的中文，且中文占文本 75% 以上**才算纯中文。
     * 日语原文里常夹汉字（`識別`、`先生`），所以必须同时满足：
     * 有汉字、**没有假名**、没有谚文，且汉字占「有效字符」的四分之三以上。
     * 混排行（`已安装 det + rec`、`V2 · 抓屏`）汉字占比不足 75%，会照常翻译 + 贴回。
     */
    fun isChinese(text: String): Boolean {
        var han = 0
        var other = 0
        var kana = 0
        var hangul = 0
        for (ch in text) {
            when (ch.code) {
                in 0x4E00..0x9FFF, in 0x3400..0x4DBF, in 0xF900..0xFAFF -> han++
                in 0x3040..0x30FF -> kana++
                in 0xAC00..0xD7AF, in 0x1100..0x11FF -> hangul++
                else -> if (ch.isLetterOrDigit()) other++
            }
        }
        // 有假名 / 谚文 = 日文或韩文，再多的汉字也是「日语里的汉字」，不能当中文丢掉。
        if (han == 0 || kana > 0 || hangul > 0) return false
        // 75% 阈值（用户明确要求）：han / (han + other) >= 0.75
        return han * 4 >= (han + other) * 3
    }

    /** 这一块要不要擦除重写：有译文、尺寸够大、原文不是纯中文、且译文确实和原文不同。 */
    fun needsReplace(line: VisionSession.Line): Boolean =
        line.translation.isNotBlank() &&
            line.width > 8f &&
            line.height > 8f &&
            !isChinese(line.text) &&
            line.translation.trim() != line.text.trim()

    /** 这一帧真正会被擦除重写的块数（界面状态行用它告诉用户「贴回了几块」）。 */
    fun countTargets(lines: List<VisionSession.Line>): Int = lines.count { needsReplace(it) }

    /**
     * 补背景模型的位置：`files/models/inpaint/migan_pipeline_v2.onnx`。
     *
     * 它以前内嵌在 APK assets 里（白送 28MB 体积，而擦除是可选功能），现在跟着别的模型
     * 一起由用户导入；没导入时 [ensureLoaded] 返回 false，调用方退回程序贴片。
     */
    fun modelFile(ctx: Context): File = File(ctx.filesDir, "models/$ASSET")

    /** 模型在不在（不加载）。界面用它决定「自动补背景」开关能不能点。 */
    fun available(ctx: Context): Boolean = modelFile(ctx).isFile

    /** 装载模型；重复调用只装一次。@return 装好了 true，模型没导入 false。 */
    @Synchronized
    fun ensureLoaded(ctx: Context): Boolean {
        if (session != null) return true
        val f = modelFile(ctx)
        if (!f.isFile) {
            if (!warnedMissing) {
                warnedMissing = true
                Log.i(TAG, "没装补背景模型（${f.absolutePath}），擦除档退回程序贴片")
            }
            return false
        }
        val t0 = System.currentTimeMillis()
        val bytes = f.readBytes()
        val e = OrtEnvironment.getEnvironment()
        val opts = OrtSession.SessionOptions().apply {
            setIntraOpNumThreads(4)
            setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT)
        }
        env = e
        session = e.createSession(bytes, opts)
        Log.i(TAG, "MI-GAN 装载完成：${bytes.size / 1024}KB，${System.currentTimeMillis() - t0}ms")
        return true
    }

    /** 把这一帧的「要擦块」排好版（两个渲染器共用）。 */
    private fun buildPlans(targets: List<VisionSession.Line>, frameW: Int, frameH: Int): List<Plan> =
        targets.mapNotNull { line ->
            val s = line.translation.trim()
            if (s.isEmpty()) return@mapNotNull null
            val box = Rect(
                line.left.toInt(),
                line.top.toInt(),
                (line.left + line.width).toInt(),
                (line.top + line.height).toInt(),
            )
            if (box.width() <= 2 || box.height() <= 2) return@mapNotNull null
            // 只有「又窄又高」才算竖排，按列逐字往下摞；其它一律横排并按框宽断行。
            // 之前一律信 OCR 的 vertical 标记，结果宽度够的框也被画成一列单字，
            // 看起来就是「译文不换行、显示不全」。判据用框的宽高比，不看标记。
            val narrowColumn = line.vertical && box.width() < box.height() * 0.45f
            if (narrowColumn) {
                Plan(line, s, box, (box.width() * 0.82f).coerceIn(10f, 48f), emptyList())
            } else {
                layoutHorizontal(line, s, box, frameW, frameH)
            }
        }

    /**
     * **程序贴片**：不加载任何模型，直接用「框四周的平均色」盖住原文再写译文。
     *
     * 用户 m03633 要求把 MI-GAN 做成可禁用，禁用后走这条路：成本几乎为零（几毫秒），
     * 纯色块在纯色/渐变背景上几乎无痕，只有照片背景会露馅 —— 显式取舍，不是降级 bug。
     */
    fun renderFlat(ctx: Context?, src: Bitmap, lines: List<VisionSession.Line>): Bitmap {
        val targets = lines.filter { needsReplace(it) }
        val out = src.copy(Bitmap.Config.ARGB_8888, true) ?: return src
        if (targets.isEmpty()) return out
        val t0 = System.currentTimeMillis()
        val plans = buildPlans(targets, src.width, src.height)
        val canvas = Canvas(out)
        val fill = Paint(Paint.ANTI_ALIAS_FLAG)
        val rad = 6f
        plans.forEach { plan ->
            fill.color = ringColor(src, plan.rect)
            canvas.drawRoundRect(
                RectF(plan.rect.left - 1f, plan.rect.top - 1f, plan.rect.right + 1f, plan.rect.bottom + 1f),
                rad,
                rad,
                fill,
            )
        }
        drawTranslations(out, plans)
        // 调试留档（与 MI-GAN 档同一份路径）：程序贴片也落一张，方便直接 pull 出来看
        // 换行/排版/底色到底长什么样，不用跟用户的点击抢时间。
        runCatching {
            val dir = ctx?.getExternalFilesDir(null) ?: ctx?.filesDir
            if (dir != null) {
                java.io.FileOutputStream(java.io.File(dir, "erase_last.png")).use {
                    out.compress(Bitmap.CompressFormat.PNG, 100, it)
                }
            }
        }.onFailure { Log.w(TAG, "留档失败：${it.message}") }
        plans.take(3).forEachIndexed { i, p ->
            Log.i(
                TAG,
                "plan[$i] ${p.text.take(12)} box=${p.rect} size=${p.size} rows=${p.rows.size}" +
                    if (p.rows.isEmpty()) "（竖排）" else " -> ${p.rows.joinToString("|").take(40)}",
            )
        }
        Log.i(TAG, "程序贴片（未加载 MI-GAN）：${plans.size} 块，${System.currentTimeMillis() - t0}ms")
        return out
    }

    /**
     * **模糊贴片**：把原文那几块自己糊掉垫在底下，再写译文。
     *
     * 不加载任何模型，成本只有「降采样 + 放大」（`FILTER_BITMAP_FLAG` 的近似 box blur），
     * 一屏几十块也就几十毫秒；比同色块自然（不会在花纹/渐变背景上留一块死板的颜色），
     * 比 MI-GAN 快两个数量级。
     */
    fun renderBlur(ctx: Context?, src: Bitmap, lines: List<VisionSession.Line>): Bitmap {
        val targets = lines.filter { needsReplace(it) }
        val out = src.copy(Bitmap.Config.ARGB_8888, true) ?: return src
        if (targets.isEmpty()) return out
        val t0 = System.currentTimeMillis()
        val plans = buildPlans(targets, src.width, src.height)
        val canvas = Canvas(out)
        val bmpPaint = Paint(Paint.FILTER_BITMAP_FLAG or Paint.ANTI_ALIAS_FLAG)
        val scrim = Paint(Paint.ANTI_ALIAS_FLAG)
        scrim.color = Color.argb(38, 0, 0, 0)
        val pad = 6
        plans.forEach { plan ->
            val r = plan.rect
            val l = (r.left.toInt() - pad).coerceAtLeast(0)
            val t = (r.top.toInt() - pad).coerceAtLeast(0)
            val rr = (r.right.toInt() + pad).coerceAtMost(src.width)
            val bb = (r.bottom.toInt() + pad).coerceAtMost(src.height)
            val w = rr - l
            val h = bb - t
            if (w <= 2 || h <= 2) return@forEach
            // 先按 1/8 缩小、再放大回原尺寸：缩小丢掉的是高频细节，放大就是模糊。
            val sw = (w / 8).coerceAtLeast(1)
            val sh = (h / 8).coerceAtLeast(1)
            val crop = Bitmap.createBitmap(src, l, t, w, h)
            val tiny = Bitmap.createScaledBitmap(crop, sw, sh, true)
            if (crop !== tiny && !crop.isRecycled) crop.recycle()
            val dst = RectF(
                l.toFloat() - 1f,
                t.toFloat() - 1f,
                rr.toFloat() + 1f,
                bb.toFloat() + 1f,
            )
            canvas.save()
            canvas.clipRect(dst)
            canvas.drawBitmap(tiny, null, dst, bmpPaint)
            // 压暗一丁点：浅色背景上白字才看得清（透明度很低，不改变整体观感）。
            canvas.drawRect(dst, scrim)
            canvas.restore()
            if (!tiny.isRecycled) tiny.recycle()
        }
        drawTranslations(out, plans)
        // 跟另外两档一样留一份调试帧（files/erase_last.png），方便 adb pull 看实际效果。
        runCatching {
            val dir = ctx?.getExternalFilesDir(null) ?: ctx?.filesDir
            if (dir != null) {
                java.io.FileOutputStream(java.io.File(dir, "erase_last.png")).use {
                    out.compress(Bitmap.CompressFormat.PNG, 100, it)
                }
            }
        }.onFailure { Log.w(TAG, "写调试帧失败：${it.message}") }
        plans.take(3).forEachIndexed { i, p ->
            Log.i(TAG, "blur[$i] box=${p.rect} size=${p.size} rows=${p.rows.size}")
        }
        Log.i(TAG, "模糊贴片：${plans.size} 块，${System.currentTimeMillis() - t0}ms")
        return out
    }

    /** 取框外一圈（上下各一条 3px 带）的平均色，避免采到框内原文的笔画。 */
    private fun ringColor(bmp: Bitmap, box: Rect): Int {
        val band = 3
        var r = 0L
        var g = 0L
        var b = 0L
        var n = 0L
        fun eat(x: Int, y: Int) {
            if (x < 0 || y < 0 || x >= bmp.width || y >= bmp.height) return
            val c = bmp.getPixel(x, y)
            r += Color.red(c); g += Color.green(c); b += Color.blue(c); n++
        }
        for (x in box.left until box.right step 3) {
            for (d in 1..band) {
                eat(x, box.top - d)
                eat(x, box.bottom + d - 1)
            }
        }
        if (n == 0L) return Color.rgb(24, 22, 32)
        return Color.rgb((r / n).toInt(), (g / n).toInt(), (b / n).toInt())
    }

    /**
     * 抹掉原文 + 写译文，返回**整帧**的新位图（调用方负责回收）。
     *
     * @param lines 这一帧的识别结果；只有 `translation` 非空且非竖排的块会被擦除重写。
     */
    fun render(ctx: Context, src: Bitmap, lines: List<VisionSession.Line>): Bitmap {
        // 只擦「需要替换」的块：中文原文、译文与原文相同的一律不碰。
        val targets = lines.filter { needsReplace(it) }
        val out = src.copy(Bitmap.Config.ARGB_8888, true) ?: return src
        if (targets.isEmpty()) {
            // 一屏认出来了、翻译也回来了，却一块都不擦 —— 用户看到的就是「不嵌字」。
            // 把每类原因数出来，下次一眼能定位是译文没回来、还是被判成中文/同文。
            val noTr = lines.count { it.translation.isBlank() }
            val zh = lines.filter { it.translation.isNotBlank() }.count { isChinese(it.text) }
            val same = lines.filter { it.translation.isNotBlank() }.count {
                it.translation.trim() == it.text.trim()
            }
            val small = lines.filter { it.translation.isNotBlank() }.count { it.width <= 8f || it.height <= 8f }
            Log.w(
                TAG,
                "无可擦块：共 ${lines.size} 块（无译文 $noTr / 判中文 $zh / 译文同原文 $same / 框太小 $small）",
            )
            return out
        }

        val t0 = System.currentTimeMillis()
        // 1) 先排版：译文往往比原文长（同区域拼接后更长），按框宽断行；
        //    行数放不下就把占位往下扩，扩出来的部分同样要擦，否则字会压在原文上。
        val plans = buildPlans(targets, src.width, src.height)
        // 2) 按「行」聚成若干区域，逐块送模型 —— 整屏送进去会被缩到 512，
        //    1440 宽的一行字只剩 1/3 分辨率，补出来是糊的（实测一屏幕灰斑）。
        val regions = clusterRegions(plans.map { it.rect }, src.width, src.height)
        var doneMs = 0L
        regions.forEach { reg ->
            val t1 = System.currentTimeMillis()
            val crop = Bitmap.createBitmap(src, reg.left, reg.top, reg.width(), reg.height())
            // 掩码：整张「保留=255」，本区域里的框挖成 0（外扩 3px 吃掉抗锯齿边）。
            val grow = 3
            val mask = ByteArray(reg.width() * reg.height()) { 255.toByte() }
            plans.forEach plan@{ plan ->
                val r = plan.rect
                if (!reg.intersects(r.left, r.top, r.right, r.bottom)) {
                    return@plan
                }
                val x0 = r.left - grow - reg.left
                val y0 = r.top - grow - reg.top
                val x1 = r.right + grow - reg.left
                val y1 = r.bottom + grow - reg.top
                for (y in y0.coerceAtLeast(0) until y1.coerceAtMost(reg.height())) {
                    val row = y * reg.width()
                    for (x in x0.coerceAtLeast(0) until x1.coerceAtMost(reg.width())) mask[row + x] = 0
                }
            }
            val patch = runCatching { infer(crop, mask, reg.width(), reg.height()) }
                .getOrElse {
                    Log.e(TAG, "补背景失败，保留原图", it)
                    crop
                }
            val scaled = if (patch.width != reg.width() || patch.height != reg.height()) {
                Bitmap.createScaledBitmap(patch, reg.width(), reg.height(), true)
            } else {
                patch
            }
            Canvas(out).drawBitmap(scaled, reg.left.toFloat(), reg.top.toFloat(), null)
            if (scaled !== crop) {
                crop.recycle()
                if (patch !== scaled) patch.recycle()
            }
            doneMs += System.currentTimeMillis() - t1
        }
        drawTranslations(out, plans)
        // 调试留档：整帧结果落一份 PNG（`files/erase_last.png`），方便 adb pull 出来看补背景质量，
        // 也免得为了截一张图跟用户的点击抢时间。每次覆盖，不占空间。
        runCatching {
            val dir = ctx.getExternalFilesDir(null) ?: ctx.filesDir
            java.io.FileOutputStream(java.io.File(dir, "erase_last.png")).use { out.compress(Bitmap.CompressFormat.PNG, 100, it) }
        }.onFailure { Log.w(TAG, "留档失败：${it.message}") }
        Log.i(
            TAG,
            "擦除补背景：${regions.size} 个区域（全帧 ${src.width}x${src.height}），模型共 ${doneMs}ms、" +
                "总 ${System.currentTimeMillis() - t0}ms，改写 ${targets.size} 块",
        )
        return out
    }

    /**
     * 把要擦的框按「同一行/相邻」聚成矩形区域。
     *
     * 每块单独跑模型最清晰，但一次点击可能十几块（十几倍耗时）；整屏跑一次又太糊。
     * 这里折中：竖向间距 < 1.6 倍行高、横向间距 < 40px 的框并成一个区域，
     * 再把区域裁到不超过 [REGION_MAX] 见方，保证模型 512 的有效分辨率够用。
     */
    private fun clusterRegions(rects0: List<Rect>, w: Int, h: Int): List<Rect> {
        val rects = rects0.map {
            Rect(
                (it.left - PAD).coerceAtLeast(0),
                (it.top - PAD).coerceAtLeast(0),
                (it.right + PAD).coerceAtMost(w),
                (it.bottom + PAD).coerceAtMost(h),
            )
        }.toMutableList()
        var merged = true
        while (merged) {
            merged = false
            outer@ for (i in rects.indices) {
                for (j in i + 1 until rects.size) {
                    val a = rects[i]
                    val b = rects[j]
                    val nearY = b.top <= a.bottom + 8 && b.bottom >= a.top - 8
                    val nearX = b.left <= a.right + 40 && b.right >= a.left - 40
                    if (nearY && nearX && (maxOf(a.right, b.right) - minOf(a.left, b.left)) <= REGION_MAX) {
                        a.union(b)
                        rects.removeAt(j)
                        merged = true
                        break@outer
                    }
                }
            }
        }
        return rects
    }

    /** 一块译文最终怎么画：字号、断行结果、真正要占的矩形（放不下时向下扩）。 */
    private class Plan(
        val line: VisionSession.Line,
        val text: String,
        val rect: Rect,
        val size: Float,
        val rows: List<String>,
    ) {
        val lineHeight: Float get() = size * 1.16f
    }

    /**
     * 横排排版：按框宽断行。
     *
     * 译文比原文长是常态（同区域拼接后更长），所以先按行数决定要不要把占位往下扩；
     * 扩到原文框 2.2 倍还放不下就缩小字号，避免盖住旁边的内容。
     */
    private fun layoutHorizontal(
        line: VisionSession.Line,
        text: String,
        box: Rect,
        frameW: Int,
        frameH: Int,
    ): Plan {
        val paint = Paint(Paint.ANTI_ALIAS_FLAG)
        // 允许把占位**横向撑宽**（最多 1.6 倍框宽、且不出屏）：译文常比原文长，
        // 只往下长会把下半屏都盖住；左右各松一点更接近原版排版。
        // 上限卡在 1.6 倍是因为这一档要擦掉原字，撑太宽会连带擦掉旁边的画面。
        val growW = minOf(box.width() * 1.6f, (frameW - 8f).coerceAtLeast(box.width().toFloat()))
        val maxW = (growW - 4f).coerceAtLeast(8f)
        var size = (box.height() * 0.78f).coerceIn(14f, 56f)
        var rows: List<String> = emptyList()
        var need = 0f
        while (true) {
            paint.textSize = size
            rows = wrap(paint, text, maxW)
            need = rows.size * size * 1.16f
            val fitsBox = need <= box.height() * 1.05f
            val fitsGrow = need <= box.height() * 2.2f
            if (fitsBox || fitsGrow || size <= 14f) break
            size *= 0.86f
        }
        val h = (need + 2f).toInt().coerceAtLeast(box.height())
        val cy = box.centerY()
        var top = cy - h / 2
        top = top.coerceIn(0, (frameH - h).coerceAtLeast(0))
        // 横向：新宽度以原框中心为准，两侧夹到屏幕内。
        val w = maxW.toInt().coerceAtLeast(box.width())
        var left = (box.centerX() - w / 2f).toInt()
        left = left.coerceIn(0, (frameW - w).coerceAtLeast(0))
        val rect = Rect(left, top, left + w, top + h)
        return Plan(line, text, rect, size, rows)
    }

    /** 按像素宽度断行：CJK 逐字断，拉丁词尽量整词换行。最多 10 行，尾部加省略号。 */
    private fun wrap(paint: Paint, s: String, maxW: Float): List<String> {
        val out = ArrayList<String>()
        var i = 0
        while (i < s.length && out.size < 10) {
            val n = paint.breakText(s, i, s.length, true, maxW, null)
            if (n <= 0) break
            var end = i + n
            if (end < s.length) {
                val chunk = s.substring(i, end)
                val sp = chunk.lastIndexOf(' ')
                if (sp > 0 && chunk.length - sp <= 14) end = i + sp + 1
            }
            val piece = s.substring(i, end).trim()
            if (piece.isNotEmpty()) out += piece
            i = end
        }
        if (i < s.length && out.isNotEmpty()) {
            out[out.size - 1] = out.last().trimEnd() + "…"
        }
        return out
    }

    /** 把译文写进擦干净的框：底色亮则黑字，暗则白字；放不下会断成多行。竖排逐字往下摞。 */
    private fun drawTranslations(dst: Bitmap, plans: List<Plan>) {
        val canvas = Canvas(dst)
        val paint = Paint(Paint.ANTI_ALIAS_FLAG).apply { typeface = Typeface.DEFAULT }
        plans.forEach { plan ->
            val lum = meanLuminance(dst, plan.rect)
            paint.color = if (lum > 140) Color.rgb(20, 18, 26) else Color.WHITE
            paint.textSize = plan.size
            val save = canvas.save()
            canvas.clipRect(plan.rect)
            if (plan.rows.isEmpty()) {
                drawVertical(canvas, paint, plan.rect, plan.text)
            } else {
                var y = plan.rect.centerY() - plan.rows.size * plan.lineHeight / 2f + plan.size
                plan.rows.forEach { row ->
                    canvas.drawText(row, plan.rect.left + 2f, y, paint)
                    y += plan.lineHeight
                }
            }
            canvas.restoreToCount(save)
        }
    }

    /** 竖排：字一个个往下摞，字号受列宽限制；字多到一列放不下就缩字号，尽量一个字不丢。 */
    private fun drawVertical(canvas: Canvas, paint: Paint, box: Rect, s: String) {
        var size = (box.width() * 0.82f).coerceIn(10f, 48f)
        var step = size * 1.05f
        var rows = (box.height() / step).toInt().coerceAtLeast(1)
        // 一列塞不下就缩字号（擦除档不能横向加列：加列要擦掉旁边画面）。
        while (rows < s.length && size > 8f) {
            size *= 0.9f
            step = size * 1.05f
            rows = (box.height() / step).toInt().coerceAtLeast(1)
        }
        paint.textSize = size
        val shown = if (rows >= s.length) s else s.take(rows - 1) + "…"
        var y = box.top + size
        shown.forEach { ch ->
            val w = paint.measureText(ch.toString())
            canvas.drawText(ch.toString(), box.centerX() - w / 2f, y, paint)
            y += step
        }
    }

    /** 取框内平均亮度（每 4 像素采一次，够用且不慢）。 */
    private fun meanLuminance(bmp: Bitmap, box: Rect): Int {
        val x0 = box.left.coerceIn(0, bmp.width - 1)
        val x1 = box.right.coerceIn(x0 + 1, bmp.width)
        val y0 = box.top.coerceIn(0, bmp.height - 1)
        val y1 = box.bottom.coerceIn(y0 + 1, bmp.height)
        var sum = 0L
        var n = 0L
        var y = y0
        while (y < y1) {
            var x = x0
            while (x < x1) {
                val c = bmp.getPixel(x, y)
                sum += ((Color.red(c) * 299 + Color.green(c) * 587 + Color.blue(c) * 114) / 1000)
                n++
                x += 4
            }
            y += 4
        }
        return if (n == 0L) 128 else (sum / n).toInt()
    }

    /** 真正喂模型：image `[1,3,H,W]` uint8、mask `[1,1,H,W]` uint8，输出同尺寸 uint8。 */
    private fun infer(crop: Bitmap, mask: ByteArray, w: Int, h: Int): Bitmap {
        val e = env ?: error("模型没装载")
        val s = session ?: error("模型没装载")

        val pixels = IntArray(w * h)
        crop.getPixels(pixels, 0, w, 0, 0, w, h)

        // 缩到 MAX_SIDE 以内，减少推理时间（模型内部还会再缩到 512）。
        val scale = minOf(1f, MAX_SIDE.toFloat() / maxOf(w, h))
        val iw = (w * scale).toInt().coerceAtLeast(8)
        val ih = (h * scale).toInt().coerceAtLeast(8)

        val srcPixels = if (scale < 1f) {
            val small = Bitmap.createScaledBitmap(crop, iw, ih, true)
            IntArray(iw * ih).also { small.getPixels(it, 0, iw, 0, 0, iw, ih) }
                .also { if (small !== crop) small.recycle() }
        } else {
            pixels
        }
        val srcMask = if (scale < 1f) {
            ByteArray(iw * ih) { i ->
                val sx = (i % iw) / scale
                val sy = (i / iw) / scale
                mask[(sy.toInt() * w + sx.toInt()).coerceIn(0, mask.size - 1)]
            }
        } else {
            mask
        }

        val n = iw * ih
        // ⚠️ 张量是 NCHW（三个完整平面），不是逐像素 RGBRGB ——
        // 按像素交错写会把 R/G/B 拆成三张错位的图，输出就是一屏红绿蓝鬼影（踩过）。
        val imgBuf = ByteBuffer.allocateDirect(3 * n).order(ByteOrder.nativeOrder())
        val maskBuf = ByteBuffer.allocateDirect(n).order(ByteOrder.nativeOrder())
        for (i in 0 until n) imgBuf.put(((srcPixels[i] shr 16) and 0xFF).toByte())
        for (i in 0 until n) imgBuf.put(((srcPixels[i] shr 8) and 0xFF).toByte())
        for (i in 0 until n) imgBuf.put((srcPixels[i] and 0xFF).toByte())
        for (i in 0 until n) maskBuf.put(srcMask[i])
        imgBuf.rewind()
        maskBuf.rewind()

        val imgTensor = OnnxTensor.createTensor(e, imgBuf, longArrayOf(1, 3, ih.toLong(), iw.toLong()), OnnxJavaType.UINT8)
        val maskTensor = OnnxTensor.createTensor(e, maskBuf, longArrayOf(1, 1, ih.toLong(), iw.toLong()), OnnxJavaType.UINT8)
        val inputs = mapOf("image" to imgTensor, "mask" to maskTensor)
        s.run(inputs).use { result ->
            val outTensor = result[0] as OnnxTensor
            val bb = outTensor.byteBuffer ?: error("输出不是 uint8")
            val outPixels = IntArray(n)
            for (i in 0 until n) {
                val rr = bb.get(i).toInt() and 0xFF
                val gg = bb.get(n + i).toInt() and 0xFF
                val bl = bb.get(2 * n + i).toInt() and 0xFF
                outPixels[i] = (0xFF shl 24) or (rr shl 16) or (gg shl 8) or bl
            }
            val outBmp = Bitmap.createBitmap(outPixels, iw, ih, Bitmap.Config.ARGB_8888)
            imgTensor.close()
            maskTensor.close()
            if (outBmp.width == w && outBmp.height == h) return outBmp
            val scaled = Bitmap.createScaledBitmap(outBmp, w, h, true)
            outBmp.recycle()
            return scaled
        }
    }
}
