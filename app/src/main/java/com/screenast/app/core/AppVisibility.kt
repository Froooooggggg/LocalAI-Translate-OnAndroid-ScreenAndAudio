package com.screenast.app.core

/**
 * 「我们自己的界面是不是在前台」—— 给悬浮层用的一个朴素开关。
 *
 * 为什么要这个：[VisionResultOverlay] 贴的是 `TYPE_APPLICATION_OVERLAY`，它盖在自己界面之上时
 * 那一层会把点击全吃掉（同传的字幕浮层当年就是因此「设置点不进去」，见 InterpretActivity.onResume 的注释）。
 * 识图这边更明显：贴片铺满 1440×3200，用户在识图页点「停止抓屏」根本没反应，一按只是把贴片收掉，
 * 看起来就像界面卡死。所以本界面在前台时不贴悬浮层，结果直接看页面的「识别结果」卡。
 */
object AppVisibility {

    @Volatile
    var oursInForeground: Boolean = false

    /** 悬浮贴片要不要真的贴到屏幕上。 */
    fun canOverlayNow(): Boolean = !oursInForeground
}
