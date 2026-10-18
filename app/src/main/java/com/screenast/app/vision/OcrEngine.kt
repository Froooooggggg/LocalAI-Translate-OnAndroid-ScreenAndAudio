package com.screenast.app.vision

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Matrix
import android.util.Log
import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import java.io.File
import java.nio.FloatBuffer

/**
 * PP-OCRv5 mobile（det + rec）的端侧推理。
 *
 * 两个模型都是从 PaddleOCR 官方权重导出的 ONNX（见 [VisionModels]）：
 * - det：输入 `x = [1,3,H,W]`（H/W 为 32 的倍数，最长边缩到 [maxSide]），输出 `[1,1,H,W]`
 *   的「文字概率图」（DB 算法的 probability map）；
 * - rec：输入 `x = [1,3,48,W]`，输出 `[1,T,18385]` 的 logits，CTC 解码后配 [charset] 得文字。
 *
 * 与 PaddleOCR 原版的差异（有意简化，屏幕截图用不到）：
 * - 不做 `unclip`（pyclipper 那套多边形外扩），改为把轴对齐外框按 [UNCLIP] 等比放大；
 * - 不做方向分类器（cls 模型），竖排文字用「高 > 宽 × 1.5 就转 90°」兜底。
 */
class OcrEngine(
    context: Context,
    private val detFile: File,
    private val recFile: File,
    private val threads: Int = 4,
    /**
     * det 的输入边长上限（像素）。mobile 权重按 960 训练，放大到 1600 只是变慢不明显变好；
     * server 权重容量大，放大到 1280~1600 能明显多检出一批小字（PC 侧实测：同一张 1440×3200
     * 截图，960 检出 28 框、1600 检出 41 框）。
     */
    private val maxSide: Int = 960,
) : AutoCloseable {

    /** 一块被识别出来的文字：文本框 + 文字 + 置信度（置信度取 det 概率图里框内的均值）。 */
    data class TextBlock(
        val text: String,
        /** 8 个浮点数，顺序为左上、右上、右下、左下，单位是 [Bitmap] 的像素。 */
        val quad: FloatArray,
        val score: Float,
        /** 原框是不是竖排（高 ≥ 宽 × [VERTICAL_RATIO]）。日漫竖排文字靠它决定阅读顺序。 */
        val vertical: Boolean = false,
    ) {
        val left: Float get() = minOf(quad[0], quad[6])
        val top: Float get() = minOf(quad[1], quad[3])
        val right: Float get() = maxOf(quad[2], quad[4])
        val bottom: Float get() = maxOf(quad[5], quad[7])
        val width: Float get() = right - left
        val height: Float get() = bottom - top
    }

    private val env: OrtEnvironment = OrtEnvironment.getEnvironment()
    private val det: OrtSession
    private val rec: OrtSession

    /** CTC 词表：下标 0 = blank，1..N = 字典，最后一个 = 空格。总长必须等于 rec 的输出维度。 */
    private val charset: List<String>

    /** 会话里第一个输入的名字。mobile 与 server 权重不一定都叫 `x`，所以运行时取而不是写死。 */
    private val detInput: String
    private val recInput: String

    init {
        val opts = OrtSession.SessionOptions().apply {
            setIntraOpNumThreads(threads)
            setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT)
        }
        det = env.createSession(detFile.absolutePath, opts)
        rec = env.createSession(recFile.absolutePath, opts)
        // 输入名别写死：mobile/server 两套权重的输入名不一定都叫 x，取会话里的第一个输入。
        detInput = det.inputNames.first()
        recInput = rec.inputNames.first()
        charset = loadCharset(context)
        Log.i(
            TAG,
            "OCR 就绪：det ${detFile.length() / 1024}KB / rec ${recFile.length() / 1024}KB，" +
                "词表 ${charset.size}，输入名 det=$detInput rec=$recInput，maxSide=$maxSide",
        )
    }

    /** 识别一整张图。返回按阅读顺序排好的文字块（竖排从右往左，横排从上往下）。 */
    fun recognize(bitmap: Bitmap): List<TextBlock> {
        val t0 = System.currentTimeMillis()
        val boxes = detect(bitmap)
        val t1 = System.currentTimeMillis()
        val blocks = ArrayList<TextBlock>(boxes.size)
        for (b in boxes) {
            val crop = cropOf(bitmap, b) ?: continue
            // 图标/分隔线这类 8×12 的小块不是文字，留着只会污染结果（真机见过把状态栏图标读成「E」）
            if (crop.width * crop.height < MIN_AREA || minOf(crop.width, crop.height) < MIN_SIDE) {
                crop.recycle()
                continue
            }
            val vertical = crop.height >= crop.width * VERTICAL_RATIO
            val text = recognizeLine(crop, vertical)
            crop.recycle()
            if (text.isNotBlank()) blocks.add(TextBlock(text, b.quad, b.score, vertical))
        }
        val t2 = System.currentTimeMillis()
        val nv = blocks.count { it.vertical }
        Log.i(TAG, "识别完成：${bitmap.width}x${bitmap.height} → ${blocks.size} 块（竖排 $nv），" +
            "det ${t1 - t0}ms / rec ${t2 - t1}ms / 共 ${t2 - t0}ms")
        return sortReading(blocks)
    }

    // ---------------------------------------------------------------- det

    private class Box(val quad: FloatArray, val score: Float)

    private fun detect(src: Bitmap): List<Box> {
        val (rw, rh) = detSize(src.width, src.height)
        val scaled = Bitmap.createScaledBitmap(src, rw, rh, true)
        val buf = FloatArray(3 * rh * rw)
        val row = IntArray(rw)
        val area = rw * rh
        for (y in 0 until rh) {
            scaled.getPixels(row, 0, rw, 0, y, rw, 1)
            for (x in 0 until rw) {
                val c = row[x]
                val r = ((c shr 16) and 0xFF) / 255f
                val g = ((c shr 8) and 0xFF) / 255f
                val b = (c and 0xFF) / 255f
                val i = y * rw + x
                buf[i] = (r - MEAN_R) / STD_R
                buf[area + i] = (g - MEAN_G) / STD_G
                buf[2 * area + i] = (b - MEAN_B) / STD_B
            }
        }
        if (scaled !== src) scaled.recycle()

        val probs: FloatArray
        val shape = longArrayOf(1, 3, rh.toLong(), rw.toLong())
        OnnxTensor.createTensor(env, FloatBuffer.wrap(buf), shape).use { input ->
            det.run(mapOf(detInput to input)).use { out ->
                @Suppress("UNCHECKED_CAST")
                val v = out[0].value as Array<Array<Array<FloatArray>>>
                probs = FloatArray(rw * rh)
                var k = 0
                for (yy in 0 until rh) for (xx in 0 until rw) probs[k++] = v[0][0][yy][xx]
            }
        }
        return components(probs, rw, rh, src.width, src.height)
    }

    /** 目标尺寸：最长边不超过 [MAX_SIDE]，且两边都取 32 的倍数（det 的 stride 要求）。 */
    private fun detSize(w: Int, h: Int): Pair<Int, Int> {
        val ratio = minOf(1.0, maxSide.toDouble() / maxOf(w, h))
        val rw = maxOf(32, (w * ratio / 32).toInt() * 32)
        val rh = maxOf(32, (h * ratio / 32).toInt() * 32)
        return rw to rh
    }

    /**
     * 在概率图上做连通域，得到文字框。
     *
     * 用「行优先扫描 + 栈式 4 邻域」而不是 opencv 的 findContours：这里只关心轴对齐外框，
     * 连通域足够，少一个 native 依赖。
     */
    private fun components(prob: FloatArray, rw: Int, rh: Int, srcW: Int, srcH: Int): List<Box> {
        val seen = BooleanArray(rw * rh)
        val stack = IntArray(rw * rh)
        val boxes = ArrayList<Box>()
        val sx = srcW.toFloat() / rw
        val sy = srcH.toFloat() / rh
        for (start in prob.indices) {
            if (seen[start] || prob[start] <= BIN_THRESH) continue
            var sp = 0
            stack[sp++] = start
            seen[start] = true
            var x0 = Int.MAX_VALUE
            var x1 = -1
            var y0 = Int.MAX_VALUE
            var y1 = -1
            var sum = 0f
            var n = 0
            while (sp > 0) {
                val p = stack[--sp]
                val py = p / rw
                val px = p - py * rw
                if (px < x0) x0 = px
                if (px > x1) x1 = px
                if (py < y0) y0 = py
                if (py > y1) y1 = py
                sum += prob[p]
                n++
                if (px > 0 && !seen[p - 1] && prob[p - 1] > BIN_THRESH) { seen[p - 1] = true; stack[sp++] = p - 1 }
                if (px < rw - 1 && !seen[p + 1] && prob[p + 1] > BIN_THRESH) { seen[p + 1] = true; stack[sp++] = p + 1 }
                if (py > 0 && !seen[p - rw] && prob[p - rw] > BIN_THRESH) { seen[p - rw] = true; stack[sp++] = p - rw }
                if (py < rh - 1 && !seen[p + rw] && prob[p + rw] > BIN_THRESH) { seen[p + rw] = true; stack[sp++] = p + rw }
            }
            val bw = x1 - x0 + 1
            val bh = y1 - y0 + 1
            if (bw < 3 || bh < 3) continue
            val score = sum / n
            if (score < BOX_THRESH) continue
            // 外扩：det 的概率图只覆盖笔画内侧，直接裁会切掉字的上下缘
            val cx = (x0 + x1) / 2f
            val cy = (y0 + y1) / 2f
            val ex = bw * UNCLIP / 2f
            val ey = bh * UNCLIP / 2f
            val l = ((cx - ex) * sx).coerceIn(0f, srcW - 1f)
            val r = ((cx + ex) * sx).coerceIn(0f, srcW - 1f)
            val t = ((cy - ey) * sy).coerceIn(0f, srcH - 1f)
            val b = ((cy + ey) * sy).coerceIn(0f, srcH - 1f)
            if (r - l < 4f || b - t < 4f) continue
            boxes.add(Box(floatArrayOf(l, t, r, t, r, b, l, b), score))
        }
        return boxes
    }

    // ---------------------------------------------------------------- rec

    /**
     * 识别一行文字。
     *
     * 竖排（日漫、竖排中文）必须**逆时针**转 90°：日文竖排每个字本身是正的，从上往下排，
     * 逆时针一转，第一个字就落到最左边，rec 才能按「从左到右」正常读。
     * 顺时针（`postRotate(90f)`）会把整行读成倒序 —— 这是实测出来的差别，别改回去。
     */
    private fun recognizeLine(src: Bitmap, vertical: Boolean): String {
        var bmp = src
        if (vertical) {
            val m = Matrix().apply { postRotate(-90f) }
            bmp = Bitmap.createBitmap(bmp, 0, 0, bmp.width, bmp.height, m, true)
        }
        val nw = maxOf(16, (REC_HEIGHT * bmp.width.toFloat() / bmp.height).toInt()).coerceAtMost(MAX_REC_WIDTH)
        val inBmp = Bitmap.createScaledBitmap(bmp, nw, REC_HEIGHT, true)
        if (inBmp !== bmp) bmp.recycle()

        val area = REC_HEIGHT * nw
        val buf = FloatArray(3 * area)
        val row = IntArray(nw)
        for (y in 0 until REC_HEIGHT) {
            inBmp.getPixels(row, 0, nw, 0, y, nw, 1)
            for (x in 0 until nw) {
                val c = row[x]
                val i = y * nw + x
                buf[i] = (((c shr 16) and 0xFF) / 255f - 0.5f) / 0.5f
                buf[area + i] = (((c shr 8) and 0xFF) / 255f - 0.5f) / 0.5f
                buf[2 * area + i] = ((c and 0xFF) / 255f - 0.5f) / 0.5f
            }
        }
        inBmp.recycle()

        val shape = longArrayOf(1, 3, REC_HEIGHT.toLong(), nw.toLong())
        return OnnxTensor.createTensor(env, FloatBuffer.wrap(buf), shape).use { input ->
            rec.run(mapOf(recInput to input)).use { out ->
                @Suppress("UNCHECKED_CAST")
                val v = out[0].value as Array<Array<FloatArray>>
                ctcDecode(v[0])
            }
        }
    }

    /** CTC 贪心解码：argmax → 去重 → 去 blank(0) → 查词表。 */
    private fun ctcDecode(logits: Array<FloatArray>): String {
        val sb = StringBuilder()
        var prev = -1
        for (t in logits.indices) {
            val l = logits[t]
            var best = 0
            var bestV = l[0]
            for (i in 1 until l.size) if (l[i] > bestV) { bestV = l[i]; best = i }
            if (best != 0 && best != prev && best < charset.size) sb.append(charset[best])
            prev = best
        }
        return sb.toString()
    }

    private fun cropOf(src: Bitmap, b: Box): Bitmap? {
        val l = b.quad[0].toInt().coerceIn(0, src.width - 1)
        val t = b.quad[1].toInt().coerceIn(0, src.height - 1)
        val r = b.quad[4].toInt().coerceIn(l + 1, src.width)
        val bo = b.quad[5].toInt().coerceIn(t + 1, src.height)
        val w = r - l
        val h = bo - t
        if (w < 2 || h < 2) return null
        return Bitmap.createBitmap(src, l, t, w, h)
    }

    /**
     * 阅读顺序。
     *
     * - **竖排**（日漫主流）：一页里从右往左一列列读，列内从上往下。按「列中心 x 降序、top 升序」。
     * - **横排**：先按行聚（top 差不超过行高的一半），行内按 x 升序。
     * - 两者混排时竖排列在前面 —— 漫画里的台词几乎全是竖排，横排一般是旁注/拟声词说明。
     */
    private fun sortReading(blocks: List<TextBlock>): List<TextBlock> {
        if (blocks.size <= 1) return blocks
        val vertical = blocks.filter { it.vertical }
            .sortedWith(compareByDescending<TextBlock> { (it.left + it.right) / 2f }.thenBy { it.top })
        val horizontal = blocks.filterNot { it.vertical }
        if (horizontal.isEmpty()) return vertical

        val sorted = horizontal.sortedBy { it.top }
        val rows = ArrayList<MutableList<TextBlock>>()
        for (b in sorted) {
            val row = rows.lastOrNull()
            if (row != null) {
                val ref = row[0]
                if (b.top - ref.top < ref.height * 0.6f) { row.add(b); continue }
            }
            rows.add(mutableListOf(b))
        }
        return vertical + rows.flatMap { it.sortedBy { b -> b.left } }
    }

    private fun loadCharset(context: Context): List<String> {
        val lines = context.assets.open(DICT_ASSET).bufferedReader(Charsets.UTF_8).use { r ->
            r.readLines().filter { true }
        }.toMutableList()
        // 文件末尾常有换行，readLines() 已经把空行去掉；保险起见去掉尾部空串
        while (lines.isNotEmpty() && lines.last().isEmpty()) lines.removeAt(lines.size - 1)
        return ArrayList<String>(lines.size + 2).apply {
            add("")          // CTC blank
            addAll(lines)
            add(" ")         // 模型把空格放在词表最后
        }
    }

    override fun close() {
        runCatching { det.close() }
        runCatching { rec.close() }
        runCatching { env.close() }
    }

    companion object {
        private const val TAG = "AST.Ocr"
        private const val DICT_ASSET = "ocr/ppocrv5_dict.txt"

        private const val REC_HEIGHT = 48
        private const val MAX_REC_WIDTH = 1600
        private const val BIN_THRESH = 0.3f
        private const val BOX_THRESH = 0.6f
        private const val UNCLIP = 1.6f

        /** 高 ≥ 宽 × 这个比例就当竖排（日漫单列基本都是 5:1 以上，横排单字也就 1:1）。 */
        private const val VERTICAL_RATIO = 1.8f

        /** 小于这个尺寸的框当图标/噪点丢掉（源图像素）。 */
        private const val MIN_SIDE = 10
        private const val MIN_AREA = 400

        private const val MEAN_R = 0.485f
        private const val MEAN_G = 0.456f
        private const val MEAN_B = 0.406f
        private const val STD_R = 0.229f
        private const val STD_G = 0.224f
        private const val STD_B = 0.225f
    }
}
