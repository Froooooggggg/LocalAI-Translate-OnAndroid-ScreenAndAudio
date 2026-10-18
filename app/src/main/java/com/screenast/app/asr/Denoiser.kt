package com.screenast.app.asr

import android.util.Log
import com.k2fsa.sherpa.onnx.OfflineSpeechDenoiser
import com.k2fsa.sherpa.onnx.OfflineSpeechDenoiserConfig
import com.k2fsa.sherpa.onnx.OfflineSpeechDenoiserGtcrnModelConfig
import com.k2fsa.sherpa.onnx.OfflineSpeechDenoiserModelConfig
import java.io.File

/**
 * 识别前的降噪（sherpa-onnx 的 GTCRN 语音增强）。
 *
 * ## 为什么只降噪「送进识别器」的这一份，不动 VAD 的那一份
 *
 * 链路是 `PCM → 能量闸门 → VAD → 整段 → 识别`。降噪如果放在最前面，会同时影响
 * 能量闸门和 VAD —— 而这两处现在已经是调好的（闸门阈值 `GATE_RMS=0.004`、
 * silero 阈值 0.5），降噪改变了整体增益，等于把这两个阈值全部推倒重来。
 *
 * 放在「整段已经切出来、准备送识别器」这一步就没有这个问题：
 * 分段策略、切点、闸门全不变，只有识别器看到的音频干净一点。
 *
 * ## 模型
 *
 * `gtcrn_simple.onnx`（535KB，k2-fsa 的 speech-enhancement-models release）。
 * 体积极小、CPU 上每 4 秒段只要几十毫秒，代价可接受。
 * **默认关闭**：只在用户勾选「识别前降噪」时加载，避免给正常音源平白引入失真。
 */
class Denoiser private constructor(
    private val core: OfflineSpeechDenoiser,
) : AutoCloseable {

    /** 上一次降噪耗时（给统计行用）。 */
    var lastRunMs = 0L
        private set

    /** 累计降噪过的段数。 */
    var segments = 0L
        private set

    /** 累计降噪耗时。 */
    var totalRunMs = 0L
        private set

    /**
     * 降噪一整段。失败时**原样返回输入** —— 降噪是增强手段，不能因为它挂掉就丢句。
     */
    fun denoise(samples: FloatArray): FloatArray {
        if (samples.isEmpty()) return samples
        val t0 = System.currentTimeMillis()
        val out = runCatching { core.run(samples, OfflineVadAsr.SAMPLE_RATE).samples }.getOrNull()
        lastRunMs = System.currentTimeMillis() - t0
        segments++
        totalRunMs += lastRunMs
        return if (out == null || out.isEmpty()) samples else out
    }

    override fun close() {
        runCatching { core.release() }.onFailure { Log.w(TAG, "降噪器释放失败", it) }
    }

    companion object {
        private const val TAG = "AST.Denoise"

        const val MODEL_FILE = "gtcrn_simple.onnx"

        /** 在若干候选目录里找降噪模型（`<dir>/gtcrn/gtcrn_simple.onnx` 或 `<dir>/gtcrn_simple.onnx`）。 */
        fun resolve(vararg dirs: File?): File? {
            for (d in dirs) {
                if (d == null) continue
                for (cand in listOf(File(d, "gtcrn/$MODEL_FILE"), File(d, MODEL_FILE))) {
                    if (cand.isFile && cand.length() > 0) return cand
                }
            }
            return null
        }

        /**
         * 打开降噪器。模型不兼容/加载失败时返回 null（调用方退回不降噪）。
         */
        fun open(model: File, numThreads: Int = 1): Denoiser? {
            return try {
                val gt = OfflineSpeechDenoiserGtcrnModelConfig()
                gt.model = model.absolutePath
                val mc = OfflineSpeechDenoiserModelConfig()
                mc.gtcrn = gt
                mc.numThreads = numThreads
                mc.provider = "cpu"
                val cfg = OfflineSpeechDenoiserConfig()
                cfg.model = mc
                // assetManager 必须 null，理由同 OfflineVadAsr 里那段注释。
                val d = Denoiser(OfflineSpeechDenoiser(null, cfg))
                Log.i(TAG, "降噪器就绪：${model.absolutePath}（${model.length() / 1024} KB）")
                d
            } catch (t: Throwable) {
                Log.e(TAG, "降噪器加载失败：${t.message}", t)
                null
            }
        }
    }
}
