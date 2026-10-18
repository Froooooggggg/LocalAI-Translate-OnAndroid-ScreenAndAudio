package com.screenast.app.overlay

import android.accessibilityservice.AccessibilityService
import android.content.Intent
import android.graphics.Color
import android.graphics.PixelFormat
import android.util.Log
import android.view.Gravity
import android.view.WindowManager
import android.view.accessibility.AccessibilityEvent
import android.view.accessibility.AccessibilityNodeInfo
import android.widget.TextView

/**
 * 无障碍服务：承担两件事，都不需要「显示在其他应用上层」权限。
 *
 *  1. 显示层：用 TYPE_ACCESSIBILITY_OVERLAY 画悬浮窗（后续的字幕窗就用这条路）。
 *  2. 兜底采集：读取其他 App 的 View 文本。YouTube / Bilibili 的字幕多为原生 TextView，
 *     若某 App 抓不到音频，可直接读字幕文字去翻译。
 *
 * P0 阶段它只是一个“探针”：把抓到的文字和当前前台包名显示在悬浮窗里，
 * 用来回答“无障碍能不能读到字幕文字”。
 */
class TranslateAccessibilityService : AccessibilityService() {

    companion object {
        private const val TAG = "AST.A11y"

        @Volatile
        var connected: Boolean = false
            private set
    }

    private var overlay: TextView? = null
    private val wm by lazy { getSystemService(WINDOW_SERVICE) as WindowManager }

    /** 上一次读到的文本，避免刷屏。 */
    private var lastText: String = ""

    override fun onServiceConnected() {
        super.onServiceConnected()
        connected = true
        Log.i(TAG, "无障碍服务已连接")
        showOverlay()
        updateOverlay("无障碍已连接 · 正在监听屏幕文字", null)
    }

    override fun onAccessibilityEvent(event: AccessibilityEvent?) {
        val e = event ?: return
        val pkg = e.packageName?.toString() ?: return
        if (pkg == packageName) return   // 忽略自身，避免自读自写循环

        val texts = when (e.eventType) {
            AccessibilityEvent.TYPE_WINDOW_STATE_CHANGED -> listOf("前台应用：$pkg")
            AccessibilityEvent.TYPE_WINDOW_CONTENT_CHANGED -> extractTexts()
            else -> return
        }
        if (texts.isEmpty()) return

        val joined = texts.joinToString(" | ")
        if (joined == lastText) return
        lastText = joined
        updateOverlay("[$pkg]\n$joined", pkg)
    }

    /**
     * 从当前活动窗口读取文本节点。
     * 只保留长度 >= 2 的文本，过滤掉图标按钮、时间戳之类的噪声。
     */
    private fun extractTexts(): List<String> {
        val root: AccessibilityNodeInfo = try {
            rootInActiveWindow ?: return emptyList()
        } catch (t: Throwable) {
            Log.w(TAG, "取根节点失败", t)
            return emptyList()
        } ?: return emptyList()

        val out = ArrayList<String>(16)
        collect(root, out, 0)
        // 取最靠后的节点：字幕通常位于布局末端（屏幕底部）
        return out.takeLast(6)
    }

    private fun collect(node: AccessibilityNodeInfo, out: MutableList<String>, depth: Int) {
        if (depth > 25 || out.size > 40) return
        val t = node.text?.toString()?.trim()
        if (!t.isNullOrEmpty() && t.length >= 2) out.add(t)
        for (i in 0 until node.childCount) {
            val c = try { node.getChild(i) } catch (t: Throwable) { null } ?: continue
            collect(c, out, depth + 1)
        }
    }

    // ---------------------------------------------------------------- 悬浮窗

    private fun showOverlay() {
        if (overlay != null) return
        val tv = TextView(this).apply {
            setTextColor(Color.WHITE)
            textSize = 13f
            setBackgroundColor(Color.argb(190, 0, 0, 0))
            setPadding(28, 20, 28, 20)
            maxLines = 6
        }
        val type = WindowManager.LayoutParams.TYPE_ACCESSIBILITY_OVERLAY
        val params = WindowManager.LayoutParams(
            WindowManager.LayoutParams.MATCH_PARENT,
            WindowManager.LayoutParams.WRAP_CONTENT,
            type,
            WindowManager.LayoutParams.FLAG_NOT_FOCUSABLE
                or WindowManager.LayoutParams.FLAG_NOT_TOUCH_MODAL
                or WindowManager.LayoutParams.FLAG_LAYOUT_NO_LIMITS,
            PixelFormat.TRANSLUCENT
        ).apply {
            gravity = Gravity.TOP or Gravity.START
            y = 120
        }
        try {
            wm.addView(tv, params)
            overlay = tv
            Log.i(TAG, "悬浮窗已添加 (TYPE_ACCESSIBILITY_OVERLAY)")
        } catch (t: Throwable) {
            Log.e(TAG, "添加悬浮窗失败", t)
        }
    }

    private fun updateOverlay(text: String, pkg: String?) {
        val tv = overlay ?: return
        tv.post {
            tv.text = text
            if (pkg != null) Log.d(TAG, "[$pkg] $text")
        }
    }

    private fun removeOverlay() {
        overlay?.let {
            try { wm.removeView(it) } catch (t: Throwable) { Log.w(TAG, "移除悬浮窗失败", t) }
        }
        overlay = null
    }

    override fun onInterrupt() {
        Log.w(TAG, "无障碍服务被中断")
    }

    override fun onUnbind(intent: Intent?): Boolean {
        connected = false
        removeOverlay()
        Log.i(TAG, "无障碍服务已断开")
        return super.onUnbind(intent)
    }

    override fun onDestroy() {
        connected = false
        removeOverlay()
        super.onDestroy()
    }
}
