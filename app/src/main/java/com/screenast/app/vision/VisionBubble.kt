package com.screenast.app.vision

import android.content.Context
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.Path
import android.graphics.PixelFormat
import android.graphics.RectF
import android.util.Log
import android.view.Gravity
import android.view.MotionEvent
import android.view.View
import android.view.WindowManager

/**
 * 屏幕上的小悬浮按钮（识图手动识别入口）。
 *
 * 外观刻意跟字幕框（`overlay/SubtitleOverlay.kt` 的 `applyStyle()`）一致：
 * 深色圆角矩形 + 细白边 + 浅色图形 —— 用户原话「改成和你那个文本框一样的风格」。
 *
 * 两种状态：
 * - **空闲**：两根竖条（暂停符号）→ 按一下开始识别；
 * - **正在识别**：一个右向三角（播放符号）→ 让人知道「它在忙，别连点」，
 *   翻译贴回屏幕后自动变回暂停。
 *
 * 交互：
 * - 单击 → [onTap]（由 [VisionCaptureService] 去拿最近一帧跑 OCR + 贴回屏幕）；
 * - 拖动 → 移动位置，松手后把坐标存进 `vision_bubble`，下次还在原地；
 * - 拖动阈值 12dp、超过就算拖动不算点击（手指在屏幕上必然有 1~3px 抖动）。
 *
 * 线程：全部在主线程（WindowManager 的 add/remove 必须在有 Looper 的线程）。
 */
class VisionBubble private constructor(private val app: Context) {

    companion object {
        private const val TAG = "AST.Bubble"
        private const val PREFS = "vision_bubble"
        private const val KEY_X = "x"
        private const val KEY_Y = "y"
        private const val WIDTH_DP = 54
        private const val HEIGHT_DP = 38
        private const val DRAG_SLOP_DP = 12

        private var instance: VisionBubble? = null

        fun show(ctx: Context, onTap: () -> Unit) {
            val b = instance ?: VisionBubble(ctx.applicationContext).also { instance = it }
            b.onTap = onTap
            b.attach()
        }

        fun hide() {
            instance?.detach()
            instance = null
        }

        fun isShowing(): Boolean = instance?.view != null

        /** 切换「空闲（暂停符号）／正在识别（三角符号）」。没有按钮时静默忽略。 */
        fun setBusy(busy: Boolean) {
            // 调用方可能在 OCR 线程 —— 只能回主线程碰 View，否则
            // ViewRootImpl$CalledFromWrongThreadException 直接把进程打掉。
            val b = instance ?: return
            b.app.mainExecutor.execute { b.busyView?.setBusy(busy) }
        }
    }

    private var onTap: (() -> Unit)? = null
    private var view: View? = null
    private var busyView: BubbleView? = null
    private var params: WindowManager.LayoutParams? = null

    private fun attach() {
        if (view != null) return
        val wm = app.getSystemService(WindowManager::class.java) ?: return
        val density = app.resources.displayMetrics.density
        val w = (WIDTH_DP * density).toInt()
        val h = (HEIGHT_DP * density).toInt()

        val prefs = app.getSharedPreferences(PREFS, Context.MODE_PRIVATE)
        val v = BubbleView(app, density)
        busyView = v
        val lp = WindowManager.LayoutParams(
            w, h,
            WindowManager.LayoutParams.TYPE_APPLICATION_OVERLAY,
            WindowManager.LayoutParams.FLAG_NOT_FOCUSABLE or
                WindowManager.LayoutParams.FLAG_LAYOUT_NO_LIMITS or
                WindowManager.LayoutParams.FLAG_LAYOUT_IN_SCREEN,
            PixelFormat.TRANSLUCENT,
        ).apply {
            gravity = Gravity.TOP or Gravity.START
            x = prefs.getInt(KEY_X, app.resources.displayMetrics.widthPixels - w - (12 * density).toInt())
            y = prefs.getInt(KEY_Y, app.resources.displayMetrics.heightPixels / 2)
        }

        var downX = 0f
        var downY = 0f
        var startX = 0
        var startY = 0
        var dragging = false
        val slop = DRAG_SLOP_DP * density

        v.setOnTouchListener { _, e ->
            when (e.actionMasked) {
                MotionEvent.ACTION_DOWN -> {
                    downX = e.rawX
                    downY = e.rawY
                    startX = lp.x
                    startY = lp.y
                    dragging = false
                    true
                }
                MotionEvent.ACTION_MOVE -> {
                    val dx = e.rawX - downX
                    val dy = e.rawY - downY
                    if (!dragging && (Math.abs(dx) > slop || Math.abs(dy) > slop)) dragging = true
                    if (dragging) {
                        lp.x = startX + dx.toInt()
                        lp.y = startY + dy.toInt()
                        runCatching { wm.updateViewLayout(v, lp) }
                    }
                    true
                }
                MotionEvent.ACTION_UP -> {
                    if (dragging) {
                        prefs.edit().putInt(KEY_X, lp.x).putInt(KEY_Y, lp.y).apply()
                    } else {
                        Log.i(TAG, "悬浮按钮被点（${lp.x},${lp.y}）")
                        onTap?.invoke()
                    }
                    true
                }
                else -> false
            }
        }

        runCatching { wm.addView(v, lp) }
            .onSuccess {
                view = v
                params = lp
                Log.i(TAG, "悬浮按钮已显示（${w}x$h, ${lp.x},${lp.y}）")
            }
            .onFailure { Log.e(TAG, "加悬浮按钮失败（没有悬浮窗权限？）", it) }
    }

    private fun detach() {
        val v = view ?: return
        runCatching { app.getSystemService(WindowManager::class.java)?.removeView(v) }
        view = null
        busyView = null
        params = null
        Log.i(TAG, "悬浮按钮已移除")
    }

    /**
     * 圆角小方块 + 图标，配色与字幕框一致（深底 0xE6121118、白边 1dp、浅色图形）。
     */
    private class BubbleView(ctx: Context, private val density: Float) : View(ctx) {

        private val bg = Paint(Paint.ANTI_ALIAS_FLAG).apply { color = Color.argb(230, 18, 17, 24) }
        private val border = Paint(Paint.ANTI_ALIAS_FLAG).apply {
            style = Paint.Style.STROKE
            strokeWidth = 1f * density
            color = Color.argb(51, 255, 255, 255) // 0x33FFFFFF，跟字幕框默认描边一致
        }
        private val icon = Paint(Paint.ANTI_ALIAS_FLAG).apply { color = Color.argb(240, 235, 230, 255) }
        private val iconBusy = Paint(Paint.ANTI_ALIAS_FLAG).apply { color = Color.argb(255, 196, 181, 253) }
        private val tri = Path()
        private val rect = RectF()

        private var busy = false

        fun setBusy(b: Boolean) {
            if (busy == b) return
            busy = b
            invalidate()
        }

        override fun onDraw(canvas: Canvas) {
            val inset = 0.5f * density
            rect.set(inset, inset, width - inset, height - inset)
            val rad = 13f * density
            canvas.drawRoundRect(rect, rad, rad, bg)
            canvas.drawRoundRect(rect, rad, rad, border)

            val cx = width / 2f
            val cy = height / 2f
            if (busy) {
                // 播放三角：表示「正在翻这句」
                val hw = width * 0.045f
                val hh = height * 0.145f
                tri.reset()
                tri.moveTo(cx - hw, cy - hh)
                tri.lineTo(cx - hw, cy + hh)
                tri.lineTo(cx + hw * 1.5f, cy)
                tri.close()
                canvas.drawPath(tri, iconBusy)
            } else {
                // 暂停：两根圆角竖条
                val barW = width * 0.075f
                val barH = height * 0.26f
                val gap = width * 0.070f
                val top = cy - barH
                val r = barW / 2f
                canvas.drawRoundRect(cx - gap - barW, top, cx - gap, top + barH * 2, r, r, icon)
                canvas.drawRoundRect(cx + gap, top, cx + gap + barW, top + barH * 2, r, r, icon)
            }
        }
    }
}
