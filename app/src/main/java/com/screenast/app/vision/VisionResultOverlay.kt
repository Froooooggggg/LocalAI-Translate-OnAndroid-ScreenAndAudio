package com.screenast.app.vision

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.PixelFormat
import android.graphics.RectF
import android.graphics.Typeface
import android.graphics.drawable.BitmapDrawable
import android.os.Handler
import android.os.Looper
import android.util.Log
import android.view.Gravity
import android.view.View
import android.view.WindowManager
import android.widget.ImageView

/**
 * 把识别结果「贴」回屏幕上原坐标的一层透明浮层（⑬-V3 的 A 档：同色贴片 + 译文）。
 *
 * 现在画的是深色圆角块 + 白字，不是真的「取原文背景色」：`Image` 的 RGBA 字节序与
 * `ARGB_8888` 位图 R/B 相反（见 [VisionCaptureService.toBitmap] 的 TODO），
 * 取色会串成蓝橙颠倒的块。等字节序修好再换 [Settings.VISION_ERASE_MODES] 的 0 档。
 *
 * 关键行为：
 * - **只有用户点一下才消失**：用户 m03182 明确要求「别 8 秒自动消失，看到译文后自己点一下收」。
 *   所以这一层永远可点、点了就 `hide()`，不做定时消失；再次识别会立刻替换旧的一层；
 * - 坐标直接用 OCR 给的**原图像素**，和投屏尺寸 1:1，不需要换算。
 */
object VisionResultOverlay {

    private const val TAG = "AST.Paste"

    private var view: View? = null
    private var params: WindowManager.LayoutParams? = null
    private val main = Handler(Looper.getMainLooper())

    /**
     * 屏幕上贴一层结果。空列表 = 直接撤掉旧层。
     *
     * @param textColor 0 = 跟随设置里的「智能/自定义」；这里传进来的就是最终颜色。
     */
    fun show(
        ctx: Context,
        lines: List<VisionSession.Line>,
        frameW: Int,
        frameH: Int,
        textColor: Int = 0,
    ) {
        main.post {
            val app = ctx.applicationContext
            hide()
            if (lines.isEmpty() || frameW <= 0 || frameH <= 0) return@post
            val wm = app.getSystemService(WindowManager::class.java) ?: return@post

            // 贴片只画有译文的块（没翻译说明后端没配好，画原文没意义）。
            val blocks = lines.filter { it.translation.isNotBlank() }
            if (blocks.isEmpty()) {
                Log.i(TAG, "没有可贴的译文，跳过")
                return@post
            }

            val v = PasteView(app, blocks, frameW, frameH, textColor)
            // 永远可点：点一下屏幕（任何一个位置）就把译文收掉，这是用户唯一要求的下班方式。
            // 不做「本界面在前台就不拦触摸」——那样在自己页面里点屏幕收不掉，用户会以为卡死。
            v.setOnClickListener {
                Log.i(TAG, "贴片被点掉")
                hide()
            }
            v.isClickable = true
            val lp = WindowManager.LayoutParams(
                frameW,
                frameH,
                WindowManager.LayoutParams.TYPE_APPLICATION_OVERLAY,
                // FLAG_LAYOUT_IN_SCREEN 必须加：只有 NO_LIMITS 时窗口原点在状态栏下方，
                // 整层贴片会往下偏一个状态栏高度（实测偏 ~120px），底部一起被裁掉。
                WindowManager.LayoutParams.FLAG_NOT_FOCUSABLE or
                    WindowManager.LayoutParams.FLAG_LAYOUT_NO_LIMITS or
                    WindowManager.LayoutParams.FLAG_LAYOUT_IN_SCREEN,
                PixelFormat.TRANSLUCENT,
            ).apply {
                gravity = Gravity.TOP or Gravity.START
                x = 0
                y = 0
            }
            runCatching { wm.addView(v, lp) }
                .onSuccess {
                    view = v
                    params = lp
                    Log.i(TAG, "贴片已显示：${blocks.size} 块 @ ${frameW}x${frameH}")
                }
                .onFailure { Log.e(TAG, "贴片显示失败（悬浮窗权限？）", it) }
        }
    }

    fun hide() {
        val v = view ?: return
        runCatching { v.context.getSystemService(WindowManager::class.java)?.removeView(v) }
        view = null
        params = null
        // 擦除档贴的是一整张位图（18MB 起），撤层时一起回收，别等 GC 攒爆内存。
        (v as? ImageView)?.let { iv ->
            val bmp = (iv.drawable as? BitmapDrawable)?.bitmap
            iv.setImageDrawable(null)
            if (bmp != null && !bmp.isRecycled) bmp.recycle()
        }
    }

    /**
     * 「擦除档」用：直接把处理好的**整帧**位图铺满屏幕显示。
     *
     * 和 [show] 的区别是这里画的是模型补完背景、写过译文的成品图，不是一圈圈色块；
     * 交互一致 —— 点一下收掉、不自动消失、再次识别会替换旧图。
     */
    fun showBitmap(ctx: Context, bitmap: Bitmap, frameW: Int, frameH: Int) {
        main.post {
            val app = ctx.applicationContext
            hide()
            val wm = app.getSystemService(WindowManager::class.java) ?: return@post
            val iv = ImageView(app).apply {
                setImageBitmap(bitmap)
                scaleType = ImageView.ScaleType.FIT_XY
                setOnClickListener {
                    Log.i(TAG, "贴片被点掉")
                    hide()
                }
                isClickable = true
            }
            val lp = WindowManager.LayoutParams(
                frameW,
                frameH,
                WindowManager.LayoutParams.TYPE_APPLICATION_OVERLAY,
                WindowManager.LayoutParams.FLAG_NOT_FOCUSABLE or
                    WindowManager.LayoutParams.FLAG_LAYOUT_NO_LIMITS or
                    WindowManager.LayoutParams.FLAG_LAYOUT_IN_SCREEN,
                PixelFormat.TRANSLUCENT,
            ).apply {
                gravity = Gravity.TOP or Gravity.START
                x = 0
                y = 0
            }
            runCatching { wm.addView(iv, lp) }
                .onSuccess {
                    view = iv
                    params = lp
                    Log.i(TAG, "擦除档已显示：${frameW}x$frameH")
                }
                .onFailure {
                    Log.e(TAG, "擦除档显示失败（悬浮窗权限？）", it)
                    if (!bitmap.isRecycled) bitmap.recycle()
                }
        }
    }

    fun isShowing(): Boolean = view != null

    private class PasteView(
        ctx: Context,
        private val blocks: List<VisionSession.Line>,
        private val frameW: Int,
        private val frameH: Int,
        textColor: Int = 0,
    ) : View(ctx) {

        private val density = resources.displayMetrics.density
        // 不透明：贴片的目的就是盖住原文，半透明会让日文原文从译文底下透出来（实测很难读）。
        // 底色比页面背景亮一档（#1C1A26 vs #0D0C12）：不然用户看不出哪几行被贴过。
        private val bg = Paint(Paint.ANTI_ALIAS_FLAG).apply { color = Color.rgb(28, 26, 38) }
        private val stroke = Paint(Paint.ANTI_ALIAS_FLAG).apply {
            style = Paint.Style.STROKE
            strokeWidth = 1f * density
            color = Color.argb(110, 168, 140, 255)
        }
        private val text = Paint(Paint.ANTI_ALIAS_FLAG).apply {
            // 贴片底板固定是深色 #1C1A26，所以 0（未指定）就用白字；
            // 用户在设置里挑了颜色就照他挑的来。
            color = if (textColor != 0) textColor else Color.WHITE
            typeface = Typeface.DEFAULT
        }

        /**
         * 竖排贴片：原文是漫画那种竖向对话框，译文按规定也要竖着读（从上往下、从右往左）。
         *
         * 这里**不按框宽死守一列**：中文译文通常比日文长，一列塞不下就向左再开一列
         * （用户 m04122：「无法完全显示的时候让贴片变大一点去容纳文本」）。
         * 只有左侧贴到屏幕边缘、列再开就出屏时才缩字号。
         */
        private fun drawVertical(canvas: Canvas, b: VisionSession.Line, s: String, padX: Float, index: Int) {
            val colW = b.width + padX * 2
            val boxH = b.height + 4f * density
            var size = (b.width * 0.82f).coerceIn(10f, 40f)
            var cols = 1
            var step = size * 1.06f
            var rows = (boxH / step).toInt().coerceAtLeast(1)
            val maxPlateW = frameW - 8f * density
            // 一列能放 rows 个字；字多就加列，加列后左边可能顶到屏幕，那就缩字号再排。
            while (true) {
                rows = (boxH / (size * 1.06f)).toInt().coerceAtLeast(1)
                cols = ((s.length + rows - 1) / rows).coerceAtLeast(1)
                if (cols * colW <= maxPlateW || size <= 9f) break
                size *= 0.9f
            }
            step = size * 1.06f
            text.textSize = size
            val plateW = cols * colW
            // 以原列为基准向左长（竖排阅读方向是右→左），别长到屏幕外。
            val right = (b.left + b.width + padX).coerceAtMost(frameW - 4f * density)
            val left = (right - plateW).coerceAtLeast(4f * density)
            val rect = RectF(left, b.top - 2f * density, right, b.top + boxH - 2f * density)
            val rad = 3f * density
            canvas.drawRoundRect(rect, rad, rad, bg)
            canvas.drawRoundRect(rect, rad, rad, stroke)
            val shown = s.take(rows * cols)
            if (index < 3) {
                Log.i(
                    TAG,
                    "drawV[$index] ${b.text.take(8)} -> ${shown.take(10)} rows=$rows cols=$cols size=$size",
                )
            }
            val save = canvas.save()
            canvas.clipRect(rect)
            shown.chunked(rows).forEachIndexed { col, chunk ->
                // 第 0 段在最右一列，往左依次排（竖排的阅读顺序）。
                val cx = right - padX - col * colW - colW / 2f + padX
                var y = rect.top + size
                chunk.forEach { ch ->
                    val w = text.measureText(ch.toString())
                    canvas.drawText(ch.toString(), cx - w / 2f, y, text)
                    y += step
                }
            }
            canvas.restoreToCount(save)
        }

        /** 按像素宽度断行：CJK 逐字断，拉丁词尽量整词换行；超出行数尾部加省略号。 */
        private fun wrap(s: String, maxW: Float, size: Float, maxRows: Int): List<String> {
            text.textSize = size
            if (maxW <= 4f || s.isEmpty()) return listOf(s)
            val out = ArrayList<String>()
            var i = 0
            while (i < s.length && out.size < maxRows) {
                val n = text.breakText(s, i, s.length, true, maxW, null)
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
            if (i < s.length && out.isNotEmpty()) out[out.size - 1] = out.last().trimEnd() + "…"
            return out
        }

        override fun onDraw(canvas: Canvas) {
            blocks.forEachIndexed { i, b ->
                val padX = 3f * density
                val s = b.translation.trim().ifEmpty { b.text }
                // 竖排块也贴：把字一个个往下摞（漫画竖排对话框宽 40~80px，横写放不下，
                // 但一行一个字完全够）。用户要求「翻译完全贴在屏幕上」，所以不再跳过。
                // 只有「又窄又高」的块才按竖排画（漫画竖排对话框宽 40~80px，横写塞不下）；
                // 宽度够的框一律横排断行 —— 判据看宽高比，不信 OCR 的 vertical 标记，
                // 否则宽框会被画成一列单字，看起来就是「不换行、显示不全」。
                if (b.vertical && b.width < b.height * 0.45f) {
                    drawVertical(canvas, b, s, padX, i)
                    return@forEachIndexed
                }

                // 字号按原行高定，但不小于 14px（小字贴上去根本看不清）。
                // 贴片允许比原框大：用户 m04122 明确「无法完全显示时让贴片变大去容纳文本」，
                // 所以先按一行排满所需宽度撑宽贴片（最宽到屏幕边），太高再缩字号，
                // 实在放不下才截断加省略号。
                val maxPlateW = frameW - 8f * density
                val maxPlateH = frameH * 0.55f
                var size = (b.height * 0.72f).coerceIn(14f, 44f)
                var rows = emptyList<String>()
                var plateW = 0f
                var plateH = 0f
                while (true) {
                    text.textSize = size
                    // 想放一行需要的宽度 → 夹到 [原框宽, 屏幕宽] 之间；原框比屏幕还宽时以屏幕为准。
                    val want = text.measureText(s) + 2f * padX
                    val lo = minOf(b.width + 2f * padX, maxPlateW)
                    val inner = want.coerceIn(lo, maxPlateW) - 6f * density
                    rows = wrap(s, inner, size, 80)
                    plateW = (inner + 6f * density).coerceAtMost(maxPlateW)
                    val lineH = size * 1.18f
                    plateH = maxOf(b.height, rows.size * lineH) + 2f * density
                    if (plateH <= maxPlateH || size <= 13f) break
                    size *= 0.9f
                }
                val lineH = size * 1.18f
                // 居中对齐原框，横坐标夹到屏幕内（贴着边时贴片会往中间挪）。
                var left = b.left + b.width / 2f - plateW / 2f
                left = left.coerceIn(4f * density, (frameW - 4f * density - plateW).coerceAtLeast(4f * density))
                val cy = b.top + b.height / 2f
                var top = cy - plateH / 2f
                top = top.coerceIn(0f, (frameH - plateH).coerceAtLeast(0f))
                val rect = RectF(left, top, left + plateW, top + plateH)
                val rad = 3f * density
                canvas.drawRoundRect(rect, rad, rad, bg)
                canvas.drawRoundRect(rect, rad, rad, stroke)
                if (i < 3) {
                    Log.i(
                        TAG,
                        "draw[$i] ${b.text.take(10)} -> ${rows.joinToString("|").take(28)} " +
                            "box=(${b.left},${b.top},${b.width}x${b.height}) size=$size rows=${rows.size} " +
                            "plate=${plateW.toInt()}x${plateH.toInt()}",
                    )
                }
                val save = canvas.save()
                canvas.clipRect(rect)
                // 多行左对齐比居中好读；贴片被撑宽后行首对齐才不会跳来跳去。
                val tx = rect.left + padX
                var baseline = rect.centerY() - rows.size * lineH / 2f + size
                rows.forEach { row ->
                    canvas.drawText(row, tx, baseline, text)
                    baseline += lineH
                }
                canvas.restoreToCount(save)
            }
        }
    }
}
