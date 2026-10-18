package com.screenast.app.overlay

import android.content.ComponentCallbacks
import android.content.Context
import android.content.res.Configuration
import android.graphics.PixelFormat
import android.graphics.drawable.GradientDrawable
import android.os.Handler
import android.os.Looper
import android.text.TextUtils
import android.util.Log
import android.util.TypedValue
import android.view.Gravity
import android.view.MotionEvent
import android.view.View
import android.view.WindowManager
import android.widget.FrameLayout
import android.widget.LinearLayout
import android.widget.ScrollView
import android.widget.TextView
import com.screenast.app.core.Settings

// ⚠️ 必须给系统 Settings 起别名：本文件同时用到自己的 [Settings] 数据类，
// 不别名的话 `Settings.canDrawOverlays(...)` 会被解析到 com.screenast.app.core.Settings，
// 报 Unresolved reference（编译期就会挡住，但值得记一笔）。
import android.provider.Settings as SysSettings

/**
 * **悬浮字幕窗** —— 路线图 ⑤。
 *
 * ## 为什么是 WindowManager 手写，而不是 Compose
 *
 * Compose 的 `ComposeView` 塞进 `WindowManager` 后，View 树里没有
 * `ViewTreeLifecycleOwner` / `ViewTreeSavedStateRegistryOwner`，
 * 一旦内部触发任何 `Lifecycle` 相关的读取就是 `IllegalStateException`。
 * 可以手动造这两个 Owner 喂进去，也可以 `setViewCompositionStrategy(DisposeOnDetachedFromWindow)`，
 * 但两者都是给一个「半透明框 + 滚动列表」付不必要的成本。需求原话是
 * 「只需要一个最简单的半透明框」，于是这里用 `LinearLayout + ScrollView + TextView`。
 * 副作用是白拿原生滚动条，正好满足「装不下可以用滑条下滑」。
 *
 * ## 结构
 *
 * ```
 * FrameLayout (root，背景 = 圆角 + boxColor，alpha 就是「不透明度」)
 *  ├─ LinearLayout (vertical)
 *  │   ├─ TextView  (拖动条，26dp)            ← 只有这里绑拖动，正文区留给滚动
 *  │   └─ ScrollView (weight 1，常显滑条)
 *  │       └─ LinearLayout (vertical)         ← 每条译文一个 row，新的插在 index 0
 *  └─ TextView "◢" (26dp, 右下角)            ← 缩放柄
 * ```
 *
 * **最新一条在最上面**（与 `InterpretActivity` 的历史区一致），所以「下滑」看的是更早的句子。
 * 这么定还有一个实际好处：新内容插到顶部时 `ScrollView` 不需要自动滚到底，
 * 也就不会跟用户主动上滑翻历史的动作打架 —— 自动滚动几乎是所有字幕浮窗最难写顺的地方。
 *
 * ## 线程纪律
 *
 * 所有对外方法都可以从任意线程调（`onUtterance` 从采集线程来、`onTranslation` 从翻译线程来），
 * 内部统一 `runNow` 分派到主线程。已经在主线程时**同步执行**，这样
 * `show()` 之后立刻读 [isShowing] 拿到的就是真实结果，Activity 可以据此提示用户。
 *
 * ## 已知限制（写在这里省得以后重新发现）
 *
 * 窗口挂在 `InterpretActivity` 的生命周期上：切到别的 App（`onStop`）时还在，
 * 但 Activity 真被销毁（从最近任务划掉 / 内存回收）时窗口跟着消失。
 * 要做到「进程级常驻」得把它搬进前台服务 —— 现成的采集中枢是 `AudioCaptureService`，
 * 把 UI 挂进服务是路线图 ⑥ 的编排收尾。
 */
class SubtitleOverlay(
    context: Context,
    /**
     * 允不允许「收成小圆点」。识图浮窗传 true（用户 m04453：点左下角缩成圆点，点圆点呼出）；
     * 同传传默认的 false，行为保持原样。
     */
    private val collapsible: Boolean = false,
) {

    companion object {
        private const val TAG = "AST.Overlay"

        /** 历史条数上限。用户要求「至少展示 5 条」，[VISIBLE_ENTRIES] 是**默认可见**条数。 */
        private const val MAX_ENTRIES = 50

        /** 默认高度按「刚好放得下 5 条」算。 */
        private const val VISIBLE_ENTRIES = 5f

        private const val MIN_WIDTH_DP = 140f
        private const val MIN_HEIGHT_DP = 90f
        private const val DRAG_BAR_DP = 26f
        private const val HANDLE_DP = 26f

        /** 收起后的小圆点直径。要够手指点（48dp 是 Material 的最小触摸目标）但别太占屏幕。 */
        private const val DOT_DP = 48f

        /** 默认高度最多占屏幕的这个比例。字号拉到 40sp 时不能占满整屏。 */
        private const val MAX_DEFAULT_HEIGHT_RATIO = 0.55f

        /** `borderColor` 没指定（alpha=0）时用的内置边框色，见 `applyStyle()`。 */
        private const val DEFAULT_BORDER = 0x33FFFFFF

        /**
         * 有没有「显示在其他应用上层」权限。
         * 没有的话 `addView` 会抛 `BadTokenException: Unable to add window ... permission denied`。
         */
        fun canDraw(ctx: Context): Boolean = SysSettings.canDrawOverlays(ctx)
    }

    private val appCtx: Context = context.applicationContext
    private val main = Handler(Looper.getMainLooper())
    private val wm: WindowManager? =
        appCtx.getSystemService(Context.WINDOW_SERVICE) as? WindowManager

    private var cfg: Settings = Settings()

    private var root: FrameLayout? = null
    private var list: LinearLayout? = null
    private var params: WindowManager.LayoutParams? = null

    // ---- 收起/展开（只有 [collapsible] 时才会创建这几个视图）----
    private var content: LinearLayout? = null
    private var dot: TextView? = null
    private var collapseBtn: TextView? = null
    private var handleBR: TextView? = null
    private var handleTL: TextView? = null

    /** 收起前的几何，展开时原样还原。 */
    private var savedW = 0
    private var savedH = 0
    private var savedX = 0
    private var savedY = 0

    /** 现在是「小圆点」状态。任何线程可读。 */
    var isCollapsed: Boolean = false
        private set

    /** 拖动 / 缩放结束后回调（单位都是 dp），交给 Activity 落盘。 */
    var onGeometryChanged: ((widthDp: Float, heightDp: Float, xDp: Float, yDp: Float) -> Unit)? = null

    /** 窗口是否已经加上去了。任何线程可读。 */
    var isShowing: Boolean = false
        private set

    /** 最近一次失败原因，供界面提示。 */
    var lastError: String? = null
        private set

    // ------------------------------------------------------------ 数据

    private class Entry(val id: Long, var source: String, var target: String)

    private class EntryRow(val row: LinearLayout, val src: TextView, val tgt: TextView)

    /** 最新在前。 */
    private val entries = ArrayDeque<Entry>()
    private val rows = HashMap<Long, EntryRow>()

    /** 拖动 / 缩放过程中的起始量。 */
    private var touchStartX = 0f
    private var touchStartY = 0f
    private var startPosX = 0
    private var startPosY = 0
    private var startW = 0
    private var startH = 0

    /**
     * 转屏监听：横屏时撑宽的窗口转到竖屏后会比屏幕还宽，右下角的缩放柄直接被推到屏幕外，
     * 用户就再也拉不回来了（用户 m04408 报的就是这个）。转屏时把尺寸/位置夹回当前屏幕内。
     */
    private val configCallback = object : ComponentCallbacks {
        override fun onConfigurationChanged(newConfig: Configuration) {
            main.post { clampInto(view = root, p = params) }
        }

        override fun onLowMemory() = Unit
    }

    private var configRegistered = false

    // ------------------------------------------------------------ 对外入口

    /**
     * 显示悬浮窗（幂等）。已经显示时只按 [s] 重刷外观。
     *
     * `addView` 失败（没权限、token 失效）时不抛异常，而是把原因记进 [lastError]
     * 并保持 [isShowing] = false —— 让调用方自己决定怎么提示，别让一个浮窗把整条同传链搞崩。
     */
    fun show(s: Settings) {
        runNow {
            cfg = s
            if (isShowing) {
                // 收成圆点时不能走 applySize()：那会把窗口按「默认尺寸」重新撑开，
                // 等于每来一条译文就把用户的收起动作撤销一次。
                if (isCollapsed) return@runNow
                applyStyle()
                applySize()
                return@runNow
            }
            if (!canDraw(appCtx)) {
                lastError = "没有「显示在其他应用上层」权限"
                Log.w(TAG, lastError!!)
                return@runNow
            }
            val manager = wm
            if (manager == null) {
                lastError = "取不到 WindowManager"
                return@runNow
            }
            try {
                build(manager)
                isShowing = true
                lastError = null
                Log.i(TAG, "悬浮字幕窗已显示")
            } catch (t: Throwable) {
                // BadTokenException / SecurityException 都在这里落地。清干净，别留半个窗口。
                root?.let { v -> runCatching { manager.removeView(v) } }
                root = null
                list = null
                params = null
                isShowing = false
                lastError = "${t.javaClass.simpleName}: ${t.message}"
                Log.e(TAG, "添加悬浮窗失败", t)
            }
        }
    }

    fun hide() {
        runNow {
            val manager = wm
            val v = root
            if (v != null && manager != null) runCatching { manager.removeView(v) }
            root = null
            list = null
            params = null
            entries.clear()
            rows.clear()
            isShowing = false
            isCollapsed = false
            content = null
            dot = null
            collapseBtn = null
            handleBR = null
            handleTL = null
            if (configRegistered) {
                runCatching { appCtx.unregisterComponentCallbacks(configCallback) }
                configRegistered = false
            }
            Log.i(TAG, "悬浮字幕窗已隐藏")
        }
    }

    // ------------------------------------------------------------ 收起 / 展开

    /**
     * 把整块框收成左下角的一个小圆点（用户 m04453）。
     *
     * 只改窗口本身，不动 [isShowing] —— 收起之后窗口仍然挂着、数据仍然在收，
     * 再来译文就安静地进列表，用户点圆点展开时能看到收起期间攒下的内容。
     */
    fun collapse() {
        runNow {
            if (!collapsible) return@runNow
            val v = root ?: return@runNow
            val p = params ?: return@runNow
            if (isCollapsed) return@runNow
            savedW = p.width
            savedH = p.height
            savedX = p.x
            savedY = p.y
            // 圆点落在原窗口的**左下角**：视觉上是「这块框塌到自己的左下角」，
            // 而不是跳到屏幕别处 —— 用户的手指刚点过那里，点完东西还在手边。
            p.width = dp(DOT_DP)
            p.height = dp(DOT_DP)
            p.x = savedX
            p.y = savedY + savedH - dp(DOT_DP)
            clampPos(p)
            content?.visibility = View.GONE
            handleBR?.visibility = View.GONE
            handleTL?.visibility = View.GONE
            collapseBtn?.visibility = View.GONE
            dot?.visibility = View.VISIBLE
            applyDotStyle()
            runCatching { wm?.updateViewLayout(v, p) }
            isCollapsed = true
            Log.i(TAG, "浮窗收成圆点：${p.width}x${p.height} @ ${p.x},${p.y}（原 ${savedW}x${savedH} @ $savedX,$savedY）")
        }
    }

    /** 从圆点还原成收起前的尺寸和位置。 */
    fun expand() {
        runNow {
            if (!isCollapsed) return@runNow
            val v = root ?: return@runNow
            val p = params ?: return@runNow
            if (savedW > 0 && savedH > 0) {
                p.width = savedW
                p.height = savedH
                p.x = savedX
                p.y = savedY
            }
            content?.visibility = View.VISIBLE
            handleBR?.visibility = if (collapsible) View.VISIBLE else View.GONE
            handleTL?.visibility = View.VISIBLE
            collapseBtn?.visibility = View.VISIBLE
            dot?.visibility = View.GONE
            isCollapsed = false
            // clampInto 内部会 updateViewLayout；这里再 applyStyle() 把圆角矩形外观换回来。
            clampInto(v, p)
            applyStyle()
            Log.i(TAG, "浮窗已展开：${p.width}x${p.height} @ ${p.x},${p.y}")
        }
    }

    /** 设置变了就重刷外观（不动窗口本身，也不会丢历史）。 */
    fun applySettings(s: Settings) {
        runNow {
            cfg = s
            if (!isShowing) return@runNow
            // 收起态只换配色，别把圆点撑回大框。
            if (isCollapsed) {
                applyDotStyle()
                return@runNow
            }
            applyStyle()
            applySize()
        }
    }

    /** 清空历史。窗口留着，等下一条。 */
    fun clear() {
        runNow {
            entries.clear()
            rows.clear()
            list?.removeAllViews()
        }
    }

    /**
     * 新的一句识别出来了。[id] 与 [onTranslation] 的 id 是同一个键。
     * 同 id 重复调用（partial 重发）只更新原文，不新建行。
     */
    fun onUtterance(id: Long, source: String) {
        runNow {
            if (!isShowing) return@runNow
            val exist = entries.firstOrNull { it.id == id }
            if (exist != null) {
                exist.source = source
                rows[id]?.let { r ->
                    r.src.text = source
                    // 空白原文在「仅原文」模式下会把整行收成 0 高，交给 styleRow 判显隐。
                    styleRow(r, exist)
                }
                return@runNow
            }
            val e = Entry(id, source, "")
            entries.addFirst(e)
            val r = buildRow(e)
            rows[id] = r
            // 插到 index 0 = 最新在最上面，`ScrollView` 的滚动位置不用动。
            list?.addView(r.row, 0)
            trim()
            refreshEmphasis()
        }
    }

    /**
     * 译文更新。[text] 是**到目前为止的完整译文**（不是增量），直接覆盖即可。
     * 迟到的译文同样能落地 —— 按 id 找行，找不到就补一行只有译文的。
     */
    fun onTranslation(id: Long, text: String) {
        runNow {
            if (!isShowing) return@runNow
            var r = rows[id]
            if (r == null) {
                // 浮窗开得比识别晚，或者这一句的原文回调早于窗口建立 —— 补一行。
                val e = Entry(id, "", text)
                entries.addFirst(e)
                r = buildRow(e)
                rows[id] = r
                list?.addView(r.row, 0)
                trim()
            }
            val ent = entries.firstOrNull { it.id == id }
            if (ent != null) ent.target = text
            r.tgt.text = text.ifEmpty { "…" }
            ent?.let { e -> styleRow(r, e) }
            refreshEmphasis()
        }
    }

    /**
     * 断句合并：把 [fromId] 那一行撤掉，合并后的原文写进 [intoId] 那一行。
     *
     * 场景是强制切段把一句话切成两半（「That was the Healthiest.」+「heard in my life.」），
     * 引擎决定合成一句翻译。界面必须跟着收掉半句 —— 留着两行同样的原文、
     * 只有一行有译文，用户会以为是识别重复了。
     *
     * [intoId] 那一行不存在就补一行（浮窗开得晚时会发生）。
     */
    fun merge(intoId: Long, fromId: Long, source: String) {
        runNow {
            if (!isShowing) return@runNow
            entries.firstOrNull { it.id == fromId }?.let { entries.remove(it) }
            rows.remove(fromId)?.let { r -> list?.removeView(r.row) }
            val e = entries.firstOrNull { it.id == intoId }
            if (e != null) {
                e.source = source
                rows[intoId]?.let { r ->
                    r.src.text = source
                    styleRow(r, e)
                }
            } else {
                val ne = Entry(intoId, source, "")
                entries.addFirst(ne)
                val r = buildRow(ne)
                rows[intoId] = r
                list?.addView(r.row, 0)
                trim()
            }
            refreshEmphasis()
        }
    }

    // ------------------------------------------------------------ 构建窗口

    private fun build(manager: WindowManager) {
        val contentView = LinearLayout(appCtx).apply {
            orientation = LinearLayout.VERTICAL
            layoutParams = FrameLayout.LayoutParams(
                FrameLayout.LayoutParams.MATCH_PARENT,
                FrameLayout.LayoutParams.MATCH_PARENT,
            )
        }

        // ---- 拖动条：只有这里绑拖动 ----
        // 为什么不绑整框：正文区是 `ScrollView`，两者抢 `ACTION_MOVE` 必然打架 ——
        // 想让框跟着手指走，就没法滚历史；想滚历史，框就跟着抖。
        val bar = TextView(appCtx).apply {
            text = "⠿ 拖动     ◢ ◤ 缩放"
            setTextSize(TypedValue.COMPLEX_UNIT_SP, 10f)
            setTextColor(0x66FFFFFF)
            gravity = Gravity.CENTER
            layoutParams = LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.MATCH_PARENT, dp(DRAG_BAR_DP),
            )
            setOnTouchListener(::onDragTouch)
        }
        contentView.addView(bar)

        val scroll = ScrollView(appCtx).apply {
            isFillViewport = true
            isVerticalScrollBarEnabled = true
            scrollBarStyle = View.SCROLLBARS_INSIDE_INSET
            // 常显滑条：用户要「装不下的时候用滑条下滑」，淡出的滑条等于没有。
            isScrollbarFadingEnabled = false
            layoutParams = LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.MATCH_PARENT, 0, 1f,
            )
        }
        val inner = LinearLayout(appCtx).apply {
            orientation = LinearLayout.VERTICAL
            // ⚠️ 不能写 `ScrollView.LayoutParams`：Kotlin **不继承嵌套类**，
            // 而 Java 侧 ScrollView extends FrameLayout 才能那样写。这里只能写父类名。
            layoutParams = FrameLayout.LayoutParams(
                FrameLayout.LayoutParams.MATCH_PARENT,
                FrameLayout.LayoutParams.WRAP_CONTENT,
            )
        }
        scroll.addView(inner)
        contentView.addView(scroll)

        val rootView = FrameLayout(appCtx).apply { addView(contentView) }

        // ---- 缩放柄：右下角 ----
        val handleBRView = TextView(appCtx).apply {
            text = "◢"
            setTextSize(TypedValue.COMPLEX_UNIT_SP, 13f)
            // 0x88FFFFFF 超出 Int 正数范围（> 2147483647），字面量会被推断成 Long，
            // `setTextColor` 只收 Int —— 必须显式 .toInt()。
            setTextColor(0x88FFFFFF.toInt())
            gravity = Gravity.CENTER
            setOnTouchListener(::onResizeTouch)
        }
        rootView.addView(
            handleBRView,
            FrameLayout.LayoutParams(dp(HANDLE_DP), dp(HANDLE_DP)).apply {
                gravity = Gravity.END or Gravity.BOTTOM
            },
        )

        // ---- 缩放柄：左上角 ----
        // 右下角那个一旦被推出屏幕（横屏撑太宽 → 转竖屏）就够不着了，左上角这份是备份入口，
        // 而且从这个角拉也能把窗口缩小、把编辑方向反过来（用户 m04408 要求）。
        val handleTLView = TextView(appCtx).apply {
            text = "◤"
            setTextSize(TypedValue.COMPLEX_UNIT_SP, 13f)
            setTextColor(0x88FFFFFF.toInt())
            gravity = Gravity.CENTER
            setOnTouchListener(::onResizeTopLeftTouch)
        }
        rootView.addView(
            handleTLView,
            FrameLayout.LayoutParams(dp(HANDLE_DP), dp(HANDLE_DP)).apply {
                gravity = Gravity.START or Gravity.TOP
            },
        )

        // ---- 收起柄：左下角 ----
        // 点一下把整块框收成一个小圆点（用户 m04453）。用**点击**而不是拖动：
        // 拖动已经给了三个入口（拖动条/两个缩放柄），再加一个拖动手势只会互相打架。
        // 只有 collapsible 的窗口（识图浮窗）才建这两块视图 —— 同传浮窗的行为保持原样，
        // 少一个角柄也少一处误触。
        if (collapsible) {
            val collapse = TextView(appCtx).apply {
                text = "◱"
                setTextSize(TypedValue.COMPLEX_UNIT_SP, 13f)
                setTextColor(0x88FFFFFF.toInt())
                gravity = Gravity.CENTER
                setOnClickListener {
                    if (isCollapsed) expand() else collapse()
                }
            }
            rootView.addView(
                collapse,
                FrameLayout.LayoutParams(dp(HANDLE_DP), dp(HANDLE_DP)).apply {
                    gravity = Gravity.START or Gravity.BOTTOM
                },
            )
            collapseBtn = collapse

            // ---- 收起态的圆点本体 ----
            // 铺满整个窗口：收起后窗口只有 48dp，所以整块都能点，不存在「点不中」。
            // 单独做一个视图而不是复用 ◱，是因为收起后另外三块（拖动条/缩放柄/列表）全要隐藏，
            // 留一块满铺的视图当唯一触摸目标和视觉主体最省事。
            val dotView = TextView(appCtx).apply {
                text = "◱"
                setTextSize(TypedValue.COMPLEX_UNIT_SP, 18f)
                setTextColor(0xE6FFFFFF.toInt())
                gravity = Gravity.CENTER
                visibility = View.GONE
                setOnClickListener { expand() }
                layoutParams = FrameLayout.LayoutParams(
                    FrameLayout.LayoutParams.MATCH_PARENT,
                    FrameLayout.LayoutParams.MATCH_PARENT,
                )
            }
            rootView.addView(dotView)
            dot = dotView
        }

        val p = WindowManager.LayoutParams(
            defaultWidthPx(),
            defaultHeightPx(),
            WindowManager.LayoutParams.TYPE_APPLICATION_OVERLAY,
            // ⚠️ **绝对不能加 FLAG_NOT_TOUCHABLE** —— 加了就既不能拖也不能滚，
            // 而这正是本次需求的两条主要验收。NOT_FOCUSABLE 只挡键盘焦点，不影响触摸。
            WindowManager.LayoutParams.FLAG_NOT_FOCUSABLE or
                WindowManager.LayoutParams.FLAG_NOT_TOUCH_MODAL,
            PixelFormat.TRANSLUCENT,
        ).apply {
            gravity = Gravity.TOP or Gravity.START
            x = dp(cfg.offsetXDp)
            y = dp(cfg.offsetYDp)
            title = "screenast-subtitle"
        }

        root = rootView
        list = inner
        content = contentView
        handleBR = handleBRView
        handleTL = handleTLView
        params = p
        manager.addView(rootView, p)
        // 挂上之后先夹一次：保存的尺寸可能是横屏时撑出来的，竖屏下比屏幕还宽。
        clampInto(rootView, p)
        if (!configRegistered) {
            runCatching { appCtx.registerComponentCallbacks(configCallback) }
            configRegistered = true
        }
        // addView 之后才刷外观：applyStyle 里有 updateViewLayout，视图没挂上会抛。
        applyStyle()
    }

    private fun buildRow(e: Entry): EntryRow {
        val row = LinearLayout(appCtx).apply {
            orientation = LinearLayout.VERTICAL
            setPadding(0, dp(3f), 0, dp(3f))
        }
        val src = TextView(appCtx).apply {
            ellipsize = TextUtils.TruncateAt.END
            text = e.source
        }
        val tgt = TextView(appCtx).apply {
            ellipsize = TextUtils.TruncateAt.END
            text = e.target.ifEmpty { "…" }
        }
        row.addView(src)
        row.addView(tgt)
        val r = EntryRow(row, src, tgt)
        // 字号/字色/显隐全交给 styleRow —— 显示模式会决定它们是主行还是次行。
        styleRow(r, e)
        return r
    }

    // ------------------------------------------------------------ 显示模式

    private fun isTargetOnly(): Boolean = cfg.subtitleMode == "target"

    private fun isSourceOnly(): Boolean = cfg.subtitleMode == "source"

    /**
     * 按 [Settings.subtitleMode] 摆一行：显示哪几行、各自字号字色行数。
     *
     * 仅原文模式下，原文**升为主行**（正文号、正字色），而不是继续当小灰字 ——
     * 否则用户只剩一行缩小的次要文字，等于把字号砍掉近四成。
     */
    private fun styleRow(r: EntryRow, e: Entry) {
        val maxL = cfg.maxLines.coerceIn(1, 12)
        // 仅原文模式下译文一般藏起来；但「只有译文的行」（浮窗开得比识别晚、
        // 或这句原文始终没回来）必须例外显示 —— 否则整行收成 0 高，内容凭空消失。
        val showSrc = !isTargetOnly() && e.source.isNotBlank()
        val showTgt = !isSourceOnly() || e.source.isBlank()
        r.src.visibility = if (showSrc) View.VISIBLE else View.GONE
        r.tgt.visibility = if (showTgt) View.VISIBLE else View.GONE

        if (isSourceOnly() && showSrc) {
            // 原文升为主行：正文号、正字色，而不是继续当那行小灰字。
            r.src.setTextSize(TypedValue.COMPLEX_UNIT_SP, cfg.textSizeSp)
            r.src.setTextColor(cfg.textColor)
            r.src.maxLines = maxL
        } else {
            r.src.setTextSize(TypedValue.COMPLEX_UNIT_SP, srcTextSizeSp())
            r.src.setTextColor(srcTextColor())
            r.src.maxLines = 2
        }
        if (showTgt) {
            r.tgt.setTextSize(TypedValue.COMPLEX_UNIT_SP, cfg.textSizeSp)
            r.tgt.setTextColor(cfg.textColor)
            r.tgt.maxLines = maxL
        }
    }

    // ------------------------------------------------------------ 外观

    private fun applyStyle() {
        val p = params ?: return
        val v = root ?: return
        v.background = GradientDrawable().apply {
            // setColor 认阿尔法，所以「不透明度」就是 boxColor 的 alpha ——
            // 不用 params.alpha（那个会把字一起调淡，调到 0.3 时字也看不见了）。
            setColor(cfg.boxColor)
            cornerRadius = dp(cfg.cornerRadiusDp).toFloat()
            // 边框：`borderColor` 的 alpha 为 0（数据类默认值就是 0）时退到内置的
            // 0x33FFFFFF 细白边。理由：半透明框如果和背景同色，用户根本分不清
            // 「框在哪、能不能拖」，边框是这个控件唯一的可发现性来源。
            if (cfg.borderWidthDp > 0f) {
                val stroke = if ((cfg.borderColor ushr 24) != 0) cfg.borderColor else DEFAULT_BORDER
                setStroke(dp(cfg.borderWidthDp).coerceAtLeast(1), stroke)
            }
        }
        val pad = dp(cfg.paddingDp)
        list?.setPadding(pad, 0, pad, pad)
        // 已有行按新外观重刷（改字号 / 改显示模式都不需要重建窗口）
        rows.forEach { (id, r) ->
            entries.firstOrNull { it.id == id }?.let { e -> styleRow(r, e) }
        }
        refreshEmphasis()
        runCatching { wm?.updateViewLayout(v, p) }
    }

    /** 收起态的外观：正圆 + 同一套配色。方角会在 48dp 的小方块上露馅，所以单独走 OVAL。 */
    private fun applyDotStyle() {
        val p = params ?: return
        val v = root ?: return
        v.background = GradientDrawable().apply {
            shape = GradientDrawable.OVAL
            setColor(cfg.boxColor)
            val stroke = if ((cfg.borderColor ushr 24) != 0) cfg.borderColor else DEFAULT_BORDER
            setStroke(dp(cfg.borderWidthDp).coerceAtLeast(1), stroke)
        }
        runCatching { wm?.updateViewLayout(v, p) }
    }

    private fun applySize() {
        if (isCollapsed) return
        val p = params ?: return
        val v = root ?: return
        val w = defaultWidthPx()
        val h = defaultHeightPx()   // 内部已处理「有显式高度就用显式高度」
        if (p.width != w) p.width = w
        if (p.height != h) p.height = h
        // 显式尺寸也可能比当前屏幕大（横屏时存下的高度/宽度），统一夹一次再上屏。
        val m = appCtx.resources.displayMetrics
        p.width = p.width.coerceIn(dp(MIN_WIDTH_DP), m.widthPixels)
        p.height = p.height.coerceIn(dp(MIN_HEIGHT_DP), m.heightPixels)
        clampPos(p)
        runCatching { wm?.updateViewLayout(v, p) }
    }

    /** 最新一条全亮，其余压到 0.78 —— 一眼能分辨「现在这句」和「之前的」。 */
    private fun refreshEmphasis() {
        val newest = entries.firstOrNull()?.id
        rows.forEach { (id, r) -> r.row.alpha = if (id == newest) 1f else 0.78f }
    }

    private fun trim() {
        while (entries.size > MAX_ENTRIES) {
            val last = entries.removeLast()
            rows.remove(last.id)?.let { r -> list?.removeView(r.row) }
        }
    }

    private fun srcTextSizeSp(): Float = (cfg.textSizeSp * 0.62f).coerceAtLeast(10f)

    /** 原文用字色的 55% 亮度：同一色系，但明显退到次要位置。 */
    private fun srcTextColor(): Int {
        val a = ((cfg.textColor ushr 24) and 0xFF) * 55 / 100
        return (a shl 24) or (cfg.textColor and 0x00FFFFFF)
    }

    // ------------------------------------------------------------ 尺寸

    private fun defaultWidthPx(): Int {
        if (cfg.subtitleWidthDp > 1f) {
            return dp(cfg.subtitleWidthDp).coerceAtLeast(dp(MIN_WIDTH_DP))
        }
        val screenW = appCtx.resources.displayMetrics.widthPixels
        return (screenW * cfg.widthFraction).toInt().coerceAtLeast(dp(MIN_WIDTH_DP))
    }

    private fun defaultHeightPx(): Int {
        if (cfg.subtitleHeightDp > 1f) {
            return dp(cfg.subtitleHeightDp).coerceAtLeast(dp(MIN_HEIGHT_DP))
        }
        // 没手动调过高度 → 按「刚好放得下 5 条」算。
        // 单语模式下每行少一条小字，高度要跟着缩，否则 5 条只占半屏、剩下全是白框。
        val entryH = if (isTargetOnly() || isSourceOnly()) {
            cfg.textSizeSp * 1.35f + 6f
        } else {
            cfg.textSizeSp * 1.35f + srcTextSizeSp() * 1.30f + 6f
        }
        val want = entryH * VISIBLE_ENTRIES + DRAG_BAR_DP + cfg.paddingDp * 2f
        val cap = (appCtx.resources.displayMetrics.heightPixels * MAX_DEFAULT_HEIGHT_RATIO).toInt()
        val minH = dp(MIN_HEIGHT_DP)
        return dp(want.coerceAtLeast(cfg.minHeightDp)).coerceIn(minH, maxOf(minH, cap))
    }

    // ------------------------------------------------------------ 手势

    private fun onDragTouch(_v: View, e: MotionEvent): Boolean {
        val p = params ?: return false
        val v = root ?: return false
        // ⚠️ 块体函数**没有隐式返回**，`when` 必须显式 return，否则报 Missing return statement。
        return when (e.actionMasked) {
            MotionEvent.ACTION_DOWN -> {
                touchStartX = e.rawX
                touchStartY = e.rawY
                startPosX = p.x
                startPosY = p.y
                true
            }
            MotionEvent.ACTION_MOVE -> {
                p.x = startPosX + (e.rawX - touchStartX).toInt()
                p.y = startPosY + (e.rawY - touchStartY).toInt()
                clampPos(p)
                runCatching { wm?.updateViewLayout(v, p) }
                true
            }
            MotionEvent.ACTION_UP, MotionEvent.ACTION_CANCEL -> {
                reportGeometry(p.width, p.height, p.x, p.y)
                true
            }
            else -> false
        }
    }

    private fun onResizeTouch(_v: View, e: MotionEvent): Boolean {
        val p = params ?: return false
        val v = root ?: return false
        return when (e.actionMasked) {
            MotionEvent.ACTION_DOWN -> {
                touchStartX = e.rawX
                touchStartY = e.rawY
                startW = p.width
                startH = p.height
                true
            }
            MotionEvent.ACTION_MOVE -> {
                // 夹到屏幕内：往右下拖最多到屏幕边缘，再往外拖不会把窗口撑出屏幕。
                val m = appCtx.resources.displayMetrics
                p.width = (startW + (e.rawX - touchStartX)).toInt()
                    .coerceIn(dp(MIN_WIDTH_DP), (m.widthPixels - p.x).coerceAtLeast(dp(MIN_WIDTH_DP)))
                p.height = (startH + (e.rawY - touchStartY)).toInt()
                    .coerceIn(dp(MIN_HEIGHT_DP), (m.heightPixels - p.y).coerceAtLeast(dp(MIN_HEIGHT_DP)))
                runCatching { wm?.updateViewLayout(v, p) }
                true
            }
            MotionEvent.ACTION_UP, MotionEvent.ACTION_CANCEL -> {
                reportGeometry(p.width, p.height, p.x, p.y)
                true
            }
            else -> false
        }
    }

    /**
     * 从**左上角**改大小：右边的底边不动，往左上拖是放大、往右下拖是缩小。
     *
     * 与右下角那份的区别不只是方向：这个柄在窗口最左上角，横屏撑宽后被推出去的是右下角，
     * 左上角始终在屏幕里 —— 所以它是「窗口缩不回来」的兜底入口。
     */
    private fun onResizeTopLeftTouch(_v: View, e: MotionEvent): Boolean {
        val p = params ?: return false
        val v = root ?: return false
        return when (e.actionMasked) {
            MotionEvent.ACTION_DOWN -> {
                touchStartX = e.rawX
                touchStartY = e.rawY
                startW = p.width
                startH = p.height
                startPosX = p.x
                startPosY = p.y
                true
            }
            MotionEvent.ACTION_MOVE -> {
                val right = startPosX + startW
                val bottom = startPosY + startH
                val m = appCtx.resources.displayMetrics
                // 手指往左/上走 = 窗口变大（右边的底边固定），往右/下走 = 变小。
                val w = (startW - (e.rawX - touchStartX)).toInt()
                    .coerceIn(dp(MIN_WIDTH_DP), right)
                val h = (startH - (e.rawY - touchStartY)).toInt()
                    .coerceIn(dp(MIN_HEIGHT_DP), bottom)
                p.width = w
                p.height = h
                p.x = (right - w).coerceIn(0, m.widthPixels - w)
                p.y = (bottom - h).coerceIn(0, m.heightPixels - h)
                runCatching { wm?.updateViewLayout(v, p) }
                true
            }
            MotionEvent.ACTION_UP, MotionEvent.ACTION_CANCEL -> {
                reportGeometry(p.width, p.height, p.x, p.y)
                true
            }
            else -> false
        }
    }

    /** 拖出屏幕就再也点不到了，所以在 `ACTION_MOVE` 里就把位置夹回可视区。 */
    private fun clampPos(p: WindowManager.LayoutParams) {
        val m = appCtx.resources.displayMetrics
        val maxX = (m.widthPixels - p.width).coerceAtLeast(0)
        val maxY = (m.heightPixels - p.height).coerceAtLeast(0)
        p.x = p.x.coerceIn(0, maxX)
        p.y = p.y.coerceIn(0, maxY)
    }

    /**
     * 把窗口整个夹进当前屏幕：尺寸不超过屏幕、位置不越界。
     *
     * 转屏（横→竖）时窗口尺寸是**不跟着变**的 —— 横屏 1400dp 宽转到竖屏 900dp，
     * 右边那 500dp 和右下角的缩放柄就永久待在屏幕外，用户再也没法缩小它。
     * 这里在转屏、重建窗口时各夹一次。
     */
    private fun clampInto(view: View?, p: WindowManager.LayoutParams?) {
        val v = view ?: return
        val params = p ?: return
        val m = appCtx.resources.displayMetrics
        val wWas = params.width
        val hWas = params.height
        params.width = params.width.coerceIn(dp(MIN_WIDTH_DP), m.widthPixels)
        params.height = params.height.coerceIn(dp(MIN_HEIGHT_DP), m.heightPixels)
        clampPos(params)
        if (params.width != wWas || params.height != hWas) {
            Log.i(TAG, "浮窗尺寸夹回屏幕：${wWas}x${hWas} → ${params.width}x${params.height}（屏 ${m.widthPixels}x${m.heightPixels}）")
        }
        runCatching { wm?.updateViewLayout(v, params) }
    }

    private fun reportGeometry(w: Int, h: Int, x: Int, y: Int) {
        val d = density()
        onGeometryChanged?.invoke(w / d, h / d, x / d, y / d)
    }

    // ------------------------------------------------------------ 工具

    private fun runNow(block: () -> Unit) {
        if (Looper.myLooper() == Looper.getMainLooper()) block() else main.post(block)
    }

    private fun density(): Float = appCtx.resources.displayMetrics.density

    private fun dp(v: Float): Int = (v * density() + 0.5f).toInt()
}
