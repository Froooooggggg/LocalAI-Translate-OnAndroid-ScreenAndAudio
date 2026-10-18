package com.screenast.app.vision

import android.content.Context
import android.util.Log
import com.screenast.app.core.Settings
import com.screenast.app.overlay.SubtitleOverlay

/**
 * **识图浮窗** —— 把同传那个「可拖动 / 可缩放 / 可滚动」的悬浮字幕窗搬到识图流程里。
 *
 * ## 为什么复用 [SubtitleOverlay] 而不是再写一个
 *
 * 用户 m04158 的原话是「给那个同声传译的悬浮窗也带过来（可开启）」——
 * 他要的就是**同一个东西**：圆角半透明框、拖顶部条移动、拖右下角改大小、装不下滚。
 * 另写一份只会多出一套外观和一套拖拽 bug；复用同一个类还能让两边设置（字号/字色/底色）保持一致。
 *
 * ## 常驻 + 历史（用户 m04287 的修正）
 *
 * 第一版是「每帧 clear 再重填」，等于只显示当前这一屏，而且**没识别过就什么都不显示**。
 * 用户要的是：开关一开就常驻在屏幕上，识别结果**累积成历史**（新的在上、旧的往下滚）。
 * 所以这里：
 * - [show] 在开关打开 / 开始抓屏时就被调用，不等第一帧；
 * - [render] 只**追加**，不清空；同一段文字在最近几帧里出现过就跳过（否则静止画面会刷出一屏重复）；
 * - 历史条数由 [SubtitleOverlay] 自己的 `MAX_ENTRIES=50` 封顶。
 *
 * ## 位置落盘
 *
 * `SubtitleOverlay` 只通过 [SubtitleOverlay.onGeometryChanged] 把拖动结果（dp）抛出来。
 * 这里写进独立的 `SharedPreferences("vision_float")` —— 不复用同传那份几何，
 * 两边同时开着也不会互相把对方窗口拖走。读回时会校验：太小/越界的记录直接忽略，
 * 免得一次误拖把窗口存成一条 90dp 的细缝（第一版真机上就是这么「看不见」的）。
 */
object VisionFloatList {

    private const val TAG = "AST.FloatList"
    private const val PREFS = "vision_float"

    /** 保存的几何小于这个尺寸就当作没存过：太小的窗口等于不可用。 */
    private const val MIN_SAVED_W_DP = 200f
    private const val MIN_SAVED_H_DP = 110f

    /** 追加历史时，往回看几帧判断「这段文字刚才已经出现过了」。 */
    private const val DEDUPE_FRAMES = 2

    private var overlay: SubtitleOverlay? = null
    private var nextId = 1L
    private val recentFrames = ArrayDeque<Set<String>>()

    /** 有没有「显示在其他应用上层」权限。 */
    fun canShow(ctx: Context): Boolean = SubtitleOverlay.canDraw(ctx)

    fun lastError(): String? = overlay?.lastError

    fun isShowing(): Boolean = overlay?.isShowing == true

    /**
     * 显示浮窗（幂等）。开关打开、开始抓屏时各调一次 —— 不等识别结果就有窗口。
     *
     * 没权限返回 false，由调用方决定提示谁（Activity 去拉授权页，服务只打日志）。
     */
    fun show(ctx: Context, st: Settings): Boolean {
        if (!canShow(ctx)) {
            Log.w(TAG, "没有「显示在其他应用上层」权限，识图浮窗开不了")
            return false
        }
        val ov = overlay ?: SubtitleOverlay(ctx, collapsible = true).also { o ->
            o.onGeometryChanged = { w, h, x, y ->
                ctx.applicationContext.getSharedPreferences(PREFS, Context.MODE_PRIVATE)
                    .edit()
                    .putFloat("w", w)
                    .putFloat("h", h)
                    .putFloat("x", x)
                    .putFloat("y", y)
                    .apply()
                Log.i(TAG, "浮窗几何：${w.toInt()}x${h.toInt()} @ ${x.toInt()},${y.toInt()}")
            }
            overlay = o
        }
        ov.show(withSavedGeometry(ctx, st))
        if (!ov.isShowing) {
            Log.w(TAG, "识图浮窗没加上：${ov.lastError ?: "未知原因"}")
            return false
        }
        return true
    }

    fun hide() {
        val ov = overlay ?: return
        overlay = null
        nextId = 1L
        recentFrames.clear()
        runCatching { ov.hide() }
        Log.i(TAG, "识图浮窗已隐藏")
    }

    /** 开关关了 / 抓屏停了：如果正显示着就收掉。 */
    fun hideIfShowing() {
        if (isShowing()) hide()
    }

    /**
     * 把这一帧的所有块**追加**到浮窗历史里。
     *
     * [lines] 是 OCR 顺序（从上到下、同一行从左到右）。`SubtitleOverlay` 的新条目插在最上面，
     * 所以一帧内倒着塞，让这一屏最上面那块也落在浮窗最上面；跨帧则是新的一帧整体在上。
     */
    fun render(st: Settings, lines: List<VisionSession.Line>) {
        val ov = overlay ?: return
        if (!ov.isShowing) return
        if (lines.isEmpty()) {
            Log.i(TAG, "这一帧没有可显示的块")
            return
        }
        val ordered = lines.sortedWith(compareBy({ it.top }, { it.left }))
        val seenBefore = recentFrames.flatten().toHashSet()
        var added = 0
        // 倒序塞：先塞最下面那块，最后塞最上面那块 → 它在列表最顶部。
        ordered.asReversed().forEach { line ->
            val text = line.text.trim()
            if (text.isEmpty()) return@forEach
            // 静止画面每帧都会认到同一批字，全塞进去会刷出一屏重复行。最近几帧出现过的跳过。
            if (text in seenBefore) return@forEach
            val id = nextId++
            ov.onUtterance(id, text)
            if (line.translation.isNotBlank()) ov.onTranslation(id, line.translation)
            added++
        }
        recentFrames.addFirst(ordered.map { it.text.trim() }.toSet())
        while (recentFrames.size > DEDUPE_FRAMES) recentFrames.removeLast()
        if (added > 0) {
            Log.i(TAG, "浮窗追加 $added 条（本帧 ${ordered.size} 块）")
        } else {
            Log.i(TAG, "浮窗无新增（本帧 ${ordered.size} 块都出现过）")
        }
    }

    /** 把上次拖出来的位置/尺寸盖到设置上；没有记录、或记录小得没法用时就按设置里的默认值。 */
    private fun withSavedGeometry(ctx: Context, st: Settings): Settings {
        val p = ctx.applicationContext.getSharedPreferences(PREFS, Context.MODE_PRIVATE)
        if (!p.contains("w")) return st
        val w = p.getFloat("w", 0f)
        val h = p.getFloat("h", 0f)
        if (w < MIN_SAVED_W_DP || h < MIN_SAVED_H_DP) {
            // 第一版存过 361x90 这种细缝，直接作废，否则窗口小到看不出内容。
            Log.w(TAG, "保存的浮窗尺寸 ${w.toInt()}x${h.toInt()} 太小，改用默认尺寸")
            p.edit().clear().apply()
            return st
        }
        return st.copy(
            subtitleWidthDp = w,
            subtitleHeightDp = h,
            offsetXDp = p.getFloat("x", st.offsetXDp),
            offsetYDp = p.getFloat("y", st.offsetYDp),
        )
    }
}
