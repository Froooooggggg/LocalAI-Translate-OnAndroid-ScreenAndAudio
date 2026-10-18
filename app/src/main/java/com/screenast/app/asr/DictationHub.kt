package com.screenast.app.asr

import android.util.Log

/**
 * 采集线程与识别引擎之间的**唯一**通道。
 *
 * ## 为什么要有这一层，而不是让 AudioCaptureService 直接持有 DictationEngine
 *
 * 两个理由，都是踩出来的：
 *
 * 1. **模块边界**。`capture` 包的职责是「把系统的音频拿到手」，
 *    它不应该知道 NLLB / SenseVoice 的存在。上次 soname 事故的教训是：
 *    两个 native 库互相踩的时候，越少的地方把它们绑在一起，越容易做隔离实验
 *    （当时靠 `toggle_noasr.sh` 一键摘掉 sherpa 才定位到问题）。
 *    所以这里用「谁都可以挂上来」的观察者，而不是构造期注入。
 *
 * 2. **采集绝不能因为识别挂掉而中断**。ASR 是 native 代码（ONNX Runtime），
 *    出问题的方式是 SIGSEGV 而不是异常 —— 那没得救；但**能救的那些**
 *    （模型文件缺了、VAD 参数非法、识别时内存不够）必须被挡在采集循环之外。
 *    采集中断一次，用户就要重新授权一次 MediaProjection，代价很高。
 *    因此 [onPcm] 的调用点必须包在 try/catch 里（见 AudioCaptureService.captureLoop）。
 *
 * ## 线程约定
 *
 * 所有回调**都在采集线程上**发生，不是主线程。回调实现必须自己切线程
 * 才能碰 UI，且必须**立即返回** —— 这个线程同时负责从 AudioRecord 读数，
 * 阻塞它就会丢音频（表现为溢出计数上涨、识别漏字）。
 */
object DictationHub {

    private const val TAG = "AST.Hub"

    /** 音频消费者。实现见 [DictationEngine]。 */
    interface Listener {
        /**
         * 一段 16 位小端 PCM（采集侧的真实速率与声道数）。
         * @param rate 采集侧真实采样率
         * @param channels 声道数（本项目的播放捕获固定混成单声道）
         */
        fun onPcm(pcm: ByteArray, len: Int, rate: Int, channels: Int)

        /** 采集状态变化：开始 / 停止。用于让识别侧清空残留状态。 */
        fun onCaptureState(running: Boolean)
    }

    @Volatile
    private var listener: Listener? = null

    /** 挂上消费者，返回一个「摘除」动作。调用方负责在停止时执行它。 */
    fun attach(l: Listener): () -> Unit {
        listener = l
        Log.i(TAG, "识别引擎已挂上采集：${l.javaClass.simpleName}")
        return {
            // 只摘自己，避免把后来者误摘
            if (listener === l) {
                listener = null
                Log.i(TAG, "识别引擎已从采集摘除")
            }
        }
    }

    /**
     * 采集循环调用。**绝不抛异常** —— 识别出任何问题都只能降级为「没有字幕」，
     * 不能让采集跟着一起死。
     */
    fun feed(pcm: ByteArray, len: Int, rate: Int, channels: Int): Boolean {
        val l = listener ?: return false
        return try {
            l.onPcm(pcm, len, rate, channels)
            true
        } catch (t: Throwable) {
            Log.e(TAG, "识别侧处理音频失败（已隔离，采集继续）：${t.message}", t)
            false
        }
    }

    /** 采集启停时调用。同样不抛。 */
    fun notifyCaptureState(running: Boolean) {
        val l = listener ?: return
        try {
            l.onCaptureState(running)
        } catch (t: Throwable) {
            Log.e(TAG, "识别侧处理状态变化失败：${t.message}", t)
        }
    }

    /** 当前是否有人在听（界面用来显示「识别已接入」）。 */
    fun hasListener(): Boolean = listener != null
}
