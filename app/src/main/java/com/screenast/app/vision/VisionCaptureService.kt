package com.screenast.app.vision

import android.app.Notification
import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.PendingIntent
import android.app.Service
import android.content.Context
import android.content.Intent
import android.content.pm.ServiceInfo
import android.graphics.Bitmap
import android.graphics.PixelFormat
import android.provider.Settings
import android.hardware.display.DisplayManager
import android.hardware.display.VirtualDisplay
import android.media.Image
import android.media.ImageReader
import android.media.projection.MediaProjection
import android.media.projection.MediaProjectionManager
import android.os.Build
import android.os.Handler
import android.os.HandlerThread
import android.os.IBinder
import android.os.SystemClock
import android.util.Log
import com.screenast.app.R
import com.screenast.app.core.SettingsStore
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.flow.first
import kotlinx.coroutines.launch

/**
 * 识图抓屏（⑬-V2）。
 *
 * 链路：MediaProjection 授权 → ImageReader(RGBA_8888) → VirtualDisplay(AUTO_MIRROR)
 *      → 按间隔取帧 → 画面指纹去重 → [VisionSession.onFrame]（OCR + 翻译在会话里跑）
 *
 * 三个必须知道的约束（都踩过或写进文档）：
 * 1. **一个 MediaProjection 只能建一个 VirtualDisplay**：它和 [com.screenast.app.capture.AudioCaptureService]
 *    的音频采集互斥，先起的那个占住投屏，另一个必须停（系统限制，绕不过）。
 * 2. **授权 token 不能持久化复用**：进程被杀后必须重新弹授权，所以「抓屏」只能由界面发起。
 * 3. 必须在带 `mediaProjection` 类型的前台服务里跑（targetSdk 34+ 强制），且
 *    `startForeground` 要在 `getMediaProjection` **之前**，否则 Android 14 直接抛 SecurityException。
 */
class VisionCaptureService : Service() {

    companion object {
        private const val TAG = "AST.VisionSvc"
        const val ACTION_START = "com.screenast.app.action.START_VISION"
        const val ACTION_STOP = "com.screenast.app.action.STOP_VISION"
        const val EXTRA_RESULT_CODE = "resultCode"
        const val EXTRA_RESULT_DATA = "resultData"
        const val EXTRA_INTERVAL_MS = "intervalMs"

        private const val CHANNEL_ID = "vision_capture"
        private const val NOTIF_ID = 1002

        /** 界面「自动抓帧=关」时仍按这个节奏抓（否则按钮点下去什么都不会发生）。 */
        const val DEFAULT_INTERVAL_MS = 2500L

        /** 手动模式下等新帧的重试次数 × [RETRY_MS]：屏幕完全静止时 ImageReader 可能不产帧。 */
        private const val GRAB_RETRIES = 6
        private const val RETRY_MS = 120L

        /**
         * 兜底缓存的最短刷新间隔。
         *
         * 只在**真的来了新帧**时才刷新（静止屏幕本来就不产帧），所以看漫画/看文档基本不耗；
         * 视频/滚动这种每帧都变的场景，最多 500ms 拷一次整帧。
         */
        private const val SHOT_CACHE_MS = 500L

        /** 停掉抓屏服务（互斥用：同传开始采集前调一次）。 */
        fun stop(ctx: Context) {
            runCatching {
                ctx.startService(Intent(ctx, VisionCaptureService::class.java).setAction(ACTION_STOP))
            }
        }
    }

    private var projection: MediaProjection? = null
    private var virtualDisplay: VirtualDisplay? = null
    private var reader: ImageReader? = null
    private var thread: HandlerThread? = null
    private var handler: Handler? = null

    /** 只给「读一次设置」这类小活用：不阻塞主线程，也不和采集线程抢。 */
    private val scope = CoroutineScope(SupervisorJob() + Dispatchers.IO)

    @Volatile private var running = false
    @Volatile private var intervalMs = DEFAULT_INTERVAL_MS
    @Volatile private var lastFrameAt = 0L

    /**
     * 手动模式：有人按了悬浮键，等着抓「此刻」这一帧。
     *
     * 平时一帧都不转位图（只 `close()` 掉），按下去才抓 —— 这样贴回屏幕的就是**点的那一刻**，
     * 而不是几百毫秒甚至几秒前的缓存画面。
     */
    @Volatile private var pendingShot = false

    /**
     * 兜底帧：帧来了就按 [SHOT_CACHE_MS] 的节奏存一份。
     *
     * 用户说「没那么在意即时性」——所以点按钮时先试「此刻」的新帧，等不到就用这份兜底；
     * 屏幕静止时 VirtualDisplay 本来就不产帧，这份缓存和当前画面是同一张图。
     */
    @Volatile private var lastShot: Bitmap? = null
    private var lastShotAt = 0L

    private var lastFingerprint = 0L
    private var frames = 0
    private var skipped = 0

    private val projectionCallback = object : MediaProjection.Callback() {
        override fun onStop() {
            Log.w(TAG, "MediaProjection 被系统/用户停止")
            stopCapture("投屏被系统或用户停止")
        }
    }

    override fun onBind(intent: Intent?): IBinder? = null

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        when (intent?.action) {
            ACTION_START -> {
                val code = intent.getIntExtra(EXTRA_RESULT_CODE, 0)
                val data = if (Build.VERSION.SDK_INT >= 33) {
                    intent.getParcelableExtra(EXTRA_RESULT_DATA, Intent::class.java)
                } else {
                    @Suppress("DEPRECATION")
                    intent.getParcelableExtra(EXTRA_RESULT_DATA) as? Intent
                }
                // 0 = 手动模式（只由悬浮按钮触发识别），所以要保留 0，不能在这里兜成默认值。
                intervalMs = intent.getLongExtra(EXTRA_INTERVAL_MS, 0L).coerceAtLeast(0L)
                // 顺序不能反：先 startForeground 再 getMediaProjection（Android 14 强制）。
                startForegroundCompat()
                if (data == null || code == 0) {
                    stopCapture("缺少投屏授权数据")
                } else {
                    startCapture(code, data)
                }
            }
            ACTION_STOP -> stopCapture("用户停止抓屏")
            else -> stopSelf()
        }
        return START_NOT_STICKY
    }

    // ------------------------------------------------------------------ 抓屏

    private fun startCapture(resultCode: Int, data: Intent) {
        if (running) {
            Log.i(TAG, "已在抓屏，忽略重复启动")
            return
        }
        val mpm = getSystemService(MediaProjectionManager::class.java)
        val p = try {
            mpm.getMediaProjection(resultCode, data)
        } catch (t: Throwable) {
            Log.e(TAG, "getMediaProjection 失败", t)
            null
        }
        if (p == null) {
            stopCapture("拿不到 MediaProjection（授权已失效，请重试）")
            return
        }
        projection = p

        val dm = resources.displayMetrics
        val w = dm.widthPixels
        val h = dm.heightPixels
        val dpi = dm.densityDpi

        val t = HandlerThread("vision-capture").also { it.start() }
        thread = t
        val hd = Handler(t.looper)
        handler = hd

        p.registerCallback(projectionCallback, hd)

        val r = ImageReader.newInstance(w, h, PixelFormat.RGBA_8888, 2)
        reader = r
        r.setOnImageAvailableListener({ rd -> onImageAvailable(rd) }, hd)

        virtualDisplay = p.createVirtualDisplay(
            "screenast-vision",
            w, h, dpi,
            DisplayManager.VIRTUAL_DISPLAY_FLAG_AUTO_MIRROR,
            r.surface, null, hd,
        )

        running = true
        frames = 0
        skipped = 0
        lastFrameAt = 0
        lastFingerprint = 0
        VisionSession.clear()
        VisionSession.setCapturing(true)
        Log.i(TAG, "抓屏已启动：${w}x${h} @ ${dpi}dpi，间隔 ${intervalMs}ms")
        notify(
            if (intervalMs <= 0) "识图：悬浮按钮点一次识别一次"
            else "正在识图抓屏…（每 ${intervalMs}ms 取一帧）",
        )
        showBubble()
    }

    /**
     * 悬浮「暂停键」：一直浮在屏幕边缘，点一下 = 对当前画面识别一次并把译文贴回去。
     *
     * 抓屏在跑的时候才显示（它代表「我现在能截到画面」），停止时一并收走。
     * 没给悬浮窗权限就不显示，但自动抓屏照常跑 —— 调试日志里说清楚原因。
     */
    private fun showBubble() {
        if (!Settings.canDrawOverlays(this)) {
            Log.w(TAG, "没有悬浮窗权限，悬浮按钮不显示（自动抓屏不受影响）")
            return
        }
        VisionBubble.show(applicationContext) { onBubbleTap() }
        // 忙碌时把悬浮键从「暂停」换成「三角」，贴回屏幕后再换回来。
        VisionSession.onBusy = { busy -> VisionBubble.setBusy(busy) }
        Log.i(TAG, "悬浮按钮已挂上")
        showFloatList()
    }

    /**
     * 悬浮文字列表：**常驻**挂在屏幕上，不等第一帧（用户 m04287）。
     *
     * 开关在识图页设置里；没打开或没权限就什么都不做 —— 这条路径不该影响抓屏本身。
     * 真正的内容由 [VisionSession] 每帧追加进来，这里只负责「先有窗口」。
     */
    private fun showFloatList() {
        scope.launch {
            val st = SettingsStore(applicationContext).settings.first()
            if (!st.visionFloatList) {
                Log.i(TAG, "悬浮文字列表：设置里没开")
                return@launch
            }
            val ok = VisionFloatList.show(applicationContext, st)
            Log.i(
                TAG,
                if (ok) "悬浮文字列表已常驻显示" else "悬浮文字列表没挂上：${VisionFloatList.lastError() ?: "未知原因"}",
            )
        }
    }

    /**
     * 按钮被按：**当场**抓一帧再识别（不再用缓存画面）。
     *
     * 用户的原话：「用户点的时候软件触发抓取就行」。所以这里先把 [pendingShot] 立起来，
     * 再让采集线程去 `acquireLatestImage()`；屏幕静止时 ImageReader 可能暂时不产帧，
     * 就每 [RETRY_MS] 重试一次，最多 [GRAB_RETRIES] 次。
     */
    private fun onBubbleTap() {
        val h = handler
        if (h == null || !running) {
            Log.w(TAG, "按钮点了，但抓屏没在跑")
            return
        }
        if (VisionSession.isBusy()) {
            Log.i(TAG, "上一次识别还没跑完，忽略这次点击")
            return
        }
        pendingShot = true
        notify("正在抓取当前画面…")
        h.post { grabOnce(0) }
    }

    /** 拿一帧「此刻」的画面；拿不到（静止屏）就隔一会儿再试。 */
    private fun grabOnce(attempt: Int) {
        if (!pendingShot) return
        val r = reader
        if (r == null || !running) {
            pendingShot = false
            return
        }
        val image = runCatching { r.acquireLatestImage() }.getOrNull()
        if (image == null) {
            if (attempt >= GRAB_RETRIES) {
                pendingShot = false
                // 等不到新帧就用兜底缓存 —— 屏幕没动的时候，这份缓存和「此刻」是同一张图。
                val cached = lastShot?.let { if (it.isRecycled) null else it }
                if (cached != null && SystemClock.elapsedRealtime() - lastShotAt < 3000) {
                    Log.i(
                        TAG,
                        "等新帧失败，改用 ${SystemClock.elapsedRealtime() - lastShotAt}ms 前的缓存帧",
                    )
                    notify("正在识别当前画面…")
                    VisionSession.captureOnce(
                        applicationContext,
                        cached.copy(Bitmap.Config.ARGB_8888, false),
                    )
                } else {
                    Log.w(TAG, "等了 ${attempt * RETRY_MS}ms 还是没新画面（屏幕完全静止）")
                    notify("没有拿到新画面，再点一次")
                }
                return
            }
            handler?.postDelayed({ grabOnce(attempt + 1) }, RETRY_MS)
            return
        }
        pendingShot = false
        val bmp = try {
            toBitmap(image)
        } finally {
            image.close()
        }
        if (bmp == null) {
            notify("抓到的画面转不出来")
            return
        }
        Log.i(TAG, "按钮点击 → 当场抓帧 ${bmp.width}x${bmp.height}（重试 $attempt 次）")
        notify("正在识别当前画面…")
        VisionSession.captureOnce(applicationContext, bmp)
    }

    /**
     * 每次有可用帧就进来。
     *
     * - **手动模式（间隔 0）**：没人按按钮就只 `close()`，一帧都不转位图（省内存带宽），
     *   有人按了（[pendingShot]）才交给 [grabOnce] 那条路；
     * - **自动模式**：按间隔跳帧 + 画面指纹去重，静止画面不重复识别。
     */
    private fun onImageAvailable(rd: ImageReader) {
        val image = try {
            rd.acquireLatestImage()
        } catch (t: Throwable) {
            Log.w(TAG, "acquireLatestImage 失败", t)
            null
        } ?: return
        try {
            if (!running) return
            if (intervalMs <= 0) {
                // 手动模式：帧来了先按 SHOT_CACHE_MS 的节奏存一份兜底帧（静止屏压根不会进来），
                // 真识图仍然只由按钮触发。
                skipped++
                if (SystemClock.elapsedRealtime() - lastShotAt >= SHOT_CACHE_MS) {
                    val bmp = toBitmap(image)
                    if (bmp != null) {
                        val old = lastShot
                        lastShot = bmp
                        lastShotAt = SystemClock.elapsedRealtime()
                        if (old !== bmp && old?.isRecycled == false) old.recycle()
                    }
                }
                if (pendingShot) handler?.post { grabOnce(0) }
                return
            }

            val now = SystemClock.elapsedRealtime()
            if (now - lastFrameAt < intervalMs) {
                skipped++
                return
            }
            val fp = fingerprint(image)
            if (fp == lastFingerprint) {
                skipped++
                return
            }
            lastFingerprint = fp
            lastFrameAt = now
            frames++
            val bmp = toBitmap(image) ?: return
            VisionSession.onFrame(applicationContext, bmp)
            if (frames % 10 == 1) {
                Log.i(TAG, "抓帧 #$frames（跳过 $skipped），指纹 ${java.lang.Long.toHexString(fp)}")
            }
        } catch (t: Throwable) {
            Log.e(TAG, "处理图像失败", t)
        } finally {
            image.close()
        }
    }

    /** 采样 4096 个点算一个 64 位指纹，够区分「换页了」和「没动」。 */
    private fun fingerprint(image: Image): Long {
        val buf = image.planes[0].buffer
        val cap = buf.capacity()
        if (cap <= 0) return 0
        val step = (cap / 4096).coerceAtLeast(1)
        var h = 1125899906842597L
        var i = 0
        while (i < cap) {
            h = h * 31 + buf.get(i)
            i += step
        }
        return h
    }

    /**
     * Image → Bitmap。
     *
     * step≠4 时（某些设备 rowStride 有对齐填充）要按 rowStride 建图再裁，不能直接按 width 建，
     * 否则图像会错位成斜的 —— OCR 直接崩胡。
     *
     * TODO: RGBA_8888 的字节序与 ARGB_8888 位图相比 R/B 是反的。黑白文字（漫画绝大多数）
     *       灰度完全一样，暂不处理；将来要还原彩色贴片时再用 ColorMatrix 换回来。
     */
    private fun toBitmap(image: Image): Bitmap? {
        return try {
            val plane = image.planes[0]
            val buf = plane.buffer
            val rowStride = plane.rowStride
            val pixelStride = plane.pixelStride
            val width = image.width
            val height = image.height
            if (pixelStride != 4) {
                Log.e(TAG, "不支持的 pixelStride=$pixelStride")
                return null
            }
            val rowPixels = rowStride / pixelStride
            val raw = Bitmap.createBitmap(rowPixels, height, Bitmap.Config.ARGB_8888)
            buf.rewind()
            raw.copyPixelsFromBuffer(buf)
            if (rowPixels != width) {
                val cropped = Bitmap.createBitmap(raw, 0, 0, width, height)
                raw.recycle()
                cropped
            } else {
                raw
            }
        } catch (t: Throwable) {
            Log.e(TAG, "转 Bitmap 失败", t)
            null
        }
    }

    // ------------------------------------------------------------- 停止/释放

    private fun stopCapture(reason: String) {
        if (!running && projection == null) {
            VisionSession.setCapturing(false, reason)
            VisionBubble.hide()
            VisionResultOverlay.hide()
            VisionFloatList.hide()
            stopSelf()
            return
        }
        Log.i(TAG, "停止抓屏：$reason（共 $frames 帧，跳过 $skipped）")
        running = false
        runCatching { reader?.setOnImageAvailableListener(null, null) }
        runCatching { virtualDisplay?.release() }
        runCatching { reader?.close() }
        runCatching { projection?.unregisterCallback(projectionCallback) }
        runCatching { projection?.stop() }
        virtualDisplay = null
        reader = null
        projection = null
        handler = null
        runCatching { thread?.quitSafely() }
        thread = null

        VisionSession.setCapturing(false, reason)
        VisionSession.onBusy = null
        VisionBubble.hide()
        VisionResultOverlay.hide()
        VisionFloatList.hide()
        // 释放 OCR / 翻译后端（端侧 2B 可能占着 1.4GB）。
        VisionSession.release()
        lastShot?.let { if (!it.isRecycled) it.recycle() }
        lastShot = null
        pendingShot = false

        stopForeground(STOP_FOREGROUND_REMOVE)
        stopSelf()
    }

    override fun onDestroy() {
        Log.i(TAG, "onDestroy")
        if (running) stopCapture("服务销毁")
        super.onDestroy()
    }

    // ------------------------------------------------------------ 前台通知

    private fun startForegroundCompat() {
        val nm = getSystemService(NotificationManager::class.java)
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            if (nm.getNotificationChannel(CHANNEL_ID) == null) {
                nm.createNotificationChannel(
                    NotificationChannel(
                        CHANNEL_ID,
                        getString(R.string.notification_channel_name),
                        NotificationManager.IMPORTANCE_LOW,
                    ).apply {
                        description = "识图抓屏时显示"
                        setShowBadge(false)
                    },
                )
            }
        }
        val n = buildNotification("正在识图抓屏…")
        if (Build.VERSION.SDK_INT >= 34) {
            startForeground(NOTIF_ID, n, ServiceInfo.FOREGROUND_SERVICE_TYPE_MEDIA_PROJECTION)
        } else {
            startForeground(NOTIF_ID, n)
        }
    }

    private fun buildNotification(text: String): Notification {
        val pi = PendingIntent.getActivity(
            this, 0,
            Intent(this, com.screenast.app.InterpretActivity::class.java),
            PendingIntent.FLAG_UPDATE_CURRENT or PendingIntent.FLAG_IMMUTABLE,
        )
        return Notification.Builder(this, CHANNEL_ID)
            .setContentTitle(getString(R.string.app_name))
            .setContentText(text)
            .setSmallIcon(android.R.drawable.ic_menu_camera)
            .setContentIntent(pi)
            .setOngoing(true)
            .build()
    }

    private fun notify(text: String) {
        runCatching {
            getSystemService(NotificationManager::class.java).notify(NOTIF_ID, buildNotification(text))
        }
    }
}
