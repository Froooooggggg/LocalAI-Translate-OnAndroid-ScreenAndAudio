package com.screenast.app.capture

import android.app.Notification
import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.PendingIntent
import android.app.Service
import android.content.Context
import android.content.Intent
import android.content.pm.ServiceInfo
import android.media.AudioAttributes
import android.media.AudioFormat
import android.media.AudioManager
import android.media.AudioPlaybackCaptureConfiguration
import android.media.AudioRecord
import android.media.projection.MediaProjection
import android.media.projection.MediaProjectionManager
import android.os.Build
import android.os.Environment
import android.os.Handler
import android.os.HandlerThread
import android.os.IBinder
import android.os.SystemClock
import android.util.Log
import com.screenast.app.asr.DictationHub
import com.screenast.app.audio.AudioLevelMeter
import com.screenast.app.audio.WavWriter
import java.io.File

/**
 * P0：验证能否采集到目标 App（YouTube / X / Bilibili）播放的音频。
 *
 * 链路：MediaProjection 授权 → AudioPlaybackCaptureConfiguration → AudioRecord
 *      → (a) 实时电平返回给界面  (b) 落成 WAV 供离线分析
 *
 * 必须在带 mediaProjection 类型的前台服务中运行（targetSdk 34+ 强制）。
 */
class AudioCaptureService : Service() {

    companion object {
        private const val TAG = "AudioCapture"
        const val ACTION_START = "com.screenast.app.action.START_CAPTURE"
        const val ACTION_STOP = "com.screenast.app.action.STOP_CAPTURE"
        const val ACTION_STATUS = "com.screenast.app.action.CAPTURE_STATUS"

        const val EXTRA_RESULT_CODE = "resultCode"
        const val EXTRA_RESULT_DATA = "resultData"
        const val EXTRA_LEVEL = "level"
        const val EXTRA_DBFS = "dbfs"
        const val EXTRA_TEXT = "text"
        const val EXTRA_RUNNING = "running"
        const val EXTRA_ELAPSED = "elapsedMs"
        const val EXTRA_HAS_SIGNAL = "hasSignal"

        private const val CHANNEL_ID = "capture"
        private const val NOTIF_ID = 1001
        private const val SAMPLE_RATE = 48_000
        private const val SAMPLE_RATE_FALLBACK = 44_100

        /**
         * 设备是否支持播放捕获。
         *
         * 注意：Android SDK 中 **并不存在** AudioManager.isAudioPlaybackCaptureSupported()
         * 这个公开方法（已在 platforms/android-36 的 android.jar 上用 javap 核实：
         * AudioManager 只有 getAllowedCapturePolicy / setAllowedCapturePolicy）。
         * 因此这里用反射探测，探测不到就不下定论、直接放行，让真正的采集去暴露问题。
         */
        fun isPlaybackCaptureSupported(context: Context): Boolean {
            if (Build.VERSION.SDK_INT < Build.VERSION_CODES.Q) return false
            val am = context.getSystemService(Context.AUDIO_SERVICE) as? AudioManager ?: return false
            return try {
                val m = AudioManager::class.java.getMethod("isAudioPlaybackCaptureSupported")
                m.invoke(am) as? Boolean ?: true
            } catch (_: NoSuchMethodException) {
                true
            } catch (_: Throwable) {
                true
            }
        }
    }

    private var projection: MediaProjection? = null
    private var audioRecord: AudioRecord? = null
    private var loopThread: HandlerThread? = null
    private var loopHandler: Handler? = null
    @Volatile private var running = false
    @Volatile private var startedAt = 0L

    /**
     * 实际建起来的采集格式。给 [DictationHub] 用 —— 重采样必须按**真实**速率换算，
     * 而不是按我们请求的那个速率（见 AudioResampler 的说明）。
     */
    @Volatile private var captureRate = 0
    @Volatile private var captureChannels = 1

    private val meter = AudioLevelMeter()
    private var wav: WavWriter? = null
    private var outFile: File? = null
    private val buf = ByteArray(4096)
    private var overflowBytes = 0

    private val projectionCallback = object : MediaProjection.Callback() {
        override fun onStop() {
            Log.w(TAG, "MediaProjection 被系统/用户停止，结束采集")
            stopCapture("MediaProjection 已停止（用户撤销或投屏结束）")
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
                startForegroundCompat()
                if (data == null || code == 0) {
                    Log.e(TAG, "缺少 MediaProjection 授权数据")
                    stopCapture("缺少投屏授权数据")
                } else {
                    startCapture(code, data)
                }
            }
            ACTION_STOP -> stopCapture("用户手动停止")
            else -> {
                // 系统重启服务时无有效授权，直接结束
                stopSelf()
            }
        }
        return START_NOT_STICKY
    }

    // ---------------------------------------------------------------- 采集

    private fun startCapture(resultCode: Int, data: Intent) {
        if (running) {
            Log.w(TAG, "已在采集中，忽略重复启动")
            return
        }
        try {
            if (!isPlaybackCaptureSupported(this)) {
                reportFail("本机不支持音频播放捕获（isAudioPlaybackCaptureSupported=false）")
                return
            }

            val mpm = getSystemService(Context.MEDIA_PROJECTION_SERVICE) as MediaProjectionManager
            val mp = mpm.getMediaProjection(resultCode, data)
            if (mp == null) {
                reportFail("getMediaProjection 返回 null")
                return
            }
            projection = mp
            mp.registerCallback(projectionCallback, Handler(mainLooper))

            var rate = SAMPLE_RATE
            var format = buildFormat(rate, 1)
            var minBuf = AudioRecord.getMinBufferSize(
                rate, AudioFormat.CHANNEL_IN_MONO, AudioFormat.ENCODING_PCM_16BIT
            )
            if (minBuf <= 0) {
                rate = SAMPLE_RATE_FALLBACK
                format = buildFormat(rate, 1)
                minBuf = AudioRecord.getMinBufferSize(
                    rate, AudioFormat.CHANNEL_IN_MONO, AudioFormat.ENCODING_PCM_16BIT
                )
            }
            if (minBuf <= 0) {
                reportFail("AudioRecord.getMinBufferSize 失败")
                return
            }

            val cfg = AudioPlaybackCaptureConfiguration.Builder(mp)
                .addMatchingUsage(AudioAttributes.USAGE_MEDIA)
                .addMatchingUsage(AudioAttributes.USAGE_GAME)
                .addMatchingUsage(AudioAttributes.USAGE_UNKNOWN)
                .build()

            val bufferSize = maxOf(minBuf * 4, 8192)
            val ar = AudioRecord.Builder()
                .setAudioFormat(format)
                .setBufferSizeInBytes(bufferSize)
                .setAudioPlaybackCaptureConfig(cfg)
                .build()

            if (ar.state != AudioRecord.STATE_INITIALIZED) {
                Log.e(TAG, "AudioRecord 初始化失败, state=${ar.state}")
                ar.release()
                reportFail("AudioRecord 初始化失败（state=${ar.state}）")
                return
            }
            audioRecord = ar

            // 打开输出文件
            val dir = outputDir()
            val f = File(dir, "capture_${System.currentTimeMillis()}.wav")
            outFile = f
            wav = WavWriter(f, rate, 1)
            overflowBytes = 0

            ar.startRecording()
            if (ar.recordingState != AudioRecord.RECORDSTATE_RECORDING) {
                reportFail("AudioRecord.startRecording 未进入录音状态（可能被策略拒绝）")
                return
            }

            running = true
            startedAt = SystemClock.elapsedRealtime()
            captureRate = rate
            captureChannels = 1
            Log.i(TAG, "开始采集: rate=$rate buffer=$bufferSize file=${f.absolutePath}")

            // 通知识别侧「采集起来了」：它会清空残留状态。必须在起 loop 之前，
            // 否则第一段音频可能先于这个通知到达。
            DictationHub.notifyCaptureState(true)

            loopThread = HandlerThread("audio-loop").also { it.start() }
            loopHandler = Handler(loopThread!!.looper)
            loopHandler?.post { captureLoop() }

            notify("采集中：正在写入 ${f.name}")
            broadcast(running = true, text = "采集中\n文件：${f.absolutePath}")
        } catch (t: Throwable) {
            Log.e(TAG, "启动采集异常", t)
            reportFail("启动采集异常：${t.javaClass.simpleName}: ${t.message}")
        }
    }

    private fun buildFormat(rate: Int, channels: Int) = AudioFormat.Builder()
        .setEncoding(AudioFormat.ENCODING_PCM_16BIT)
        .setSampleRate(rate)
        .setChannelMask(if (channels == 1) AudioFormat.CHANNEL_IN_MONO else AudioFormat.CHANNEL_IN_STEREO)
        .build()

    private fun captureLoop() {
        val ar = audioRecord ?: return
        var lastUi = 0L
        var silentFor = 0L

        while (running) {
            val n = try {
                ar.read(buf, 0, buf.size)
            } catch (t: Throwable) {
                Log.e(TAG, "read 异常", t)
                break
            }

            if (n <= 0) {
                Log.w(TAG, "read 返回 $n，可能存在错误或数据溢出")
                overflowBytes += n
                continue
            }

            meter.feed(buf, n)
            wav?.append(buf, n)

            // 分发给识别侧。DictationHub.feed 保证不抛（识别出问题不能带死采集）。
            // 放在写 WAV 之后、广播之前：这条路径上最贵的活是 ASR 解码，
            // 让它紧挨着 read 发生，采样点的新鲜度最高。
            DictationHub.feed(buf, n, captureRate, captureChannels)

            val now = SystemClock.elapsedRealtime()
            if (meter.rms < 1e-5) silentFor += (now - lastUi).coerceAtLeast(0) else silentFor = 0
            if (now - lastUi >= 200) {
                lastUi = now
                val w = wav
                val warn = if (silentFor > 5000) "  ⚠ 已连续 5s 静音" else ""
                broadcast(
                    running = true,
                    text = buildString {
                        append("采集运行中 ")
                        append(((now - startedAt) / 1000)).append("s | ")
                        append("RMS ").append(String.format("%.4f", meter.rms)).append(" | ")
                        append("峰值 ").append(String.format("%.3f", meter.peak)).append(" | ")
                        append("电平表 ").append(String.format("%.1f", meter.lastDbfs)).append(" dBFS")
                        append("\n已写入 ").append(w?.dataBytes() ?: 0L).append(" 字节")
                        append("（").append(w?.durationMs() ?: 0L).append(" ms）")
                        append(warn)
                    },
                    level = meter.normalizedForUi(),
                    dbfs = meter.lastDbfs,
                    elapsed = now - startedAt,
                )
            }
        }

        // 自然结束（read 出错等）
        if (running) stopCapture("采集循环异常结束")
    }

    // ---------------------------------------------------------------- 停止

    private fun stopCapture(reason: String) {
        if (!running && audioRecord == null && projection == null) {
            stopSelf()
            return
        }
        running = false

        // 先告诉识别侧「要停了」：它会 flush VAD 里最后半句并吐出来。
        // 顺序很重要 —— 必须赶在 audioRecord 释放之前，否则最后一段音频就没了。
        // （此后若采集循环还走到一次 feed，那次数据会被 reset 丢掉，无副作用。）
        DictationHub.notifyCaptureState(false)

        try {
            audioRecord?.let {
                if (it.recordingState == AudioRecord.RECORDSTATE_RECORDING) {
                    try { it.stop() } catch (t: Throwable) { Log.w(TAG, "stop 异常", t) }
                }
                it.release()
            }
        } catch (t: Throwable) {
            Log.w(TAG, "释放 AudioRecord 异常", t)
        }
        audioRecord = null

        val w = wav
        val f = outFile
        wav = null
        val msg = try {
            val file = w?.close()
            val rmsAll = w?.rmsNormalized() ?: 0.0
            val peakAll = w?.peakNormalized() ?: 0.0
            val dur = w?.durationMs() ?: 0L
            val silent = w?.isEffectivelySilent() ?: true
            if (silent) {
                "采集结束（$reason）\n⚠ 整段几乎全静音：RMS=${String.format("%.5f", rmsAll)} 峰值=${String.format("%.4f", peakAll)}\n说明该 App 的音频被禁止捕获，或当时没有播放声音。\n文件：${file?.absolutePath}"
            } else {
                "采集结束（$reason）\n✅ 抓到有效音频！RMS=${String.format("%.4f", rmsAll)} 峰值=${String.format("%.3f", peakAll)}，时长 ${dur} ms\n文件：${file?.absolutePath}"
            }
        } catch (t: Throwable) {
            Log.e(TAG, "关闭 WAV 异常", t)
            "采集结束（$reason），但写文件异常：${t.message}"
        }

        try {
            projection?.unregisterCallback(projectionCallback)
            projection?.stop()
        } catch (t: Throwable) {
            Log.w(TAG, "停止 MediaProjection 异常", t)
        }
        projection = null

        loopHandler?.removeCallbacksAndMessages(null)
        loopThread?.quitSafely()
        loopHandler = null
        loopThread = null
        meter.reset()

        Log.i(TAG, msg)
        broadcast(running = false, text = msg, level = 0f)
        notify("采集已停止")
        stopForeground(STOP_FOREGROUND_REMOVE)
        stopSelf()
    }

    private fun reportFail(msg: String) {
        Log.e(TAG, msg)
        broadcast(running = false, text = "❌ $msg")
        stopCapture(msg)
    }

    // ------------------------------------------------------------ 文件/通知

    private fun outputDir(): File {
        val dir = File(
            Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_MUSIC),
            "ScreenAST"
        )
        if (!dir.exists()) dir.mkdirs()
        // 兜底：公共目录不可写时退回应用私有目录
        return if (dir.canWrite()) dir else {
            File(getExternalFilesDir(null) ?: filesDir, "").also { it.mkdirs() }
        }
    }

    private fun startForegroundCompat() {
        val nm = getSystemService(NotificationManager::class.java)
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            if (nm.getNotificationChannel(CHANNEL_ID) == null) {
                nm.createNotificationChannel(
                    NotificationChannel(
                        CHANNEL_ID,
                        getString(com.screenast.app.R.string.notification_channel_name),
                        NotificationManager.IMPORTANCE_LOW
                    ).apply {
                        description = getString(com.screenast.app.R.string.notification_channel_desc)
                        setShowBadge(false)
                    }
                )
            }
        }
        val n = buildNotification("正在采集屏幕音频…")
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
            PendingIntent.FLAG_UPDATE_CURRENT or PendingIntent.FLAG_IMMUTABLE
        )
        return Notification.Builder(this, CHANNEL_ID)
            .setContentTitle(getString(com.screenast.app.R.string.app_name))
            .setContentText(text)
            .setSmallIcon(android.R.drawable.ic_btn_speak_now)
            .setContentIntent(pi)
            .setOngoing(true)
            .build()
    }

    private fun notify(text: String) {
        try {
            getSystemService(NotificationManager::class.java)
                .notify(NOTIF_ID, buildNotification(text))
        } catch (t: Throwable) {
            Log.w(TAG, "更新通知失败", t)
        }
    }

    private fun broadcast(
        running: Boolean,
        text: String,
        level: Float = meter.normalizedForUi(),
        dbfs: Double = meter.lastDbfs,
        elapsed: Long = if (startedAt == 0L) 0 else SystemClock.elapsedRealtime() - startedAt,
    ) {
        val i = Intent(ACTION_STATUS).apply {
            setPackage(packageName)
            putExtra(EXTRA_RUNNING, running)
            putExtra(EXTRA_TEXT, text)
            putExtra(EXTRA_LEVEL, level)
            putExtra(EXTRA_DBFS, dbfs)
            putExtra(EXTRA_ELAPSED, elapsed)
            putExtra(EXTRA_HAS_SIGNAL, level > 0.02f)
        }
        sendBroadcast(i)
    }

    override fun onDestroy() {
        Log.i(TAG, "onDestroy")
        if (running) stopCapture("服务销毁")
        super.onDestroy()
    }
}
