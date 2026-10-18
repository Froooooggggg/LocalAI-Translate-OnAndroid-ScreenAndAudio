package com.screenast.app.audio

import kotlin.math.log10
import kotlin.math.sqrt

/**
 * 实时电平表。
 *
 * 用 RMS + 峰值双指标，因为判定“抓到没抓到”需要两者：
 *  - 只有底噪（很小 RMS、偶发尖峰）→ 说明 capture 通了但对方没在放声音；
 *  - 全 0 → 说明根本没抓到（policy 禁止 / 路由不对）。
 */
class AudioLevelMeter(
    /** 电平下跌的平滑系数，0.8 左右可获得较自然的回落。 */
    private val decay: Double = 0.82,
) {
    @Volatile
    var rms: Double = 0.0
        private set

    @Volatile
    var peak: Double = 0.0
        private set

    @Volatile
    var lastDbfs: Double = -120.0
        private set

    fun feed(buf: ByteArray, len: Int) {
        if (len < 2) return
        var sum = 0.0
        var p = 0
        var i = 0
        while (i + 1 < len) {
            val lo = buf[i].toInt() and 0xFF
            val hi = buf[i + 1].toInt()
            val s = (hi shl 8) or lo
            val a = if (s < 0) -s else s
            if (a > p) p = a
            sum += s.toDouble() * s
            i += 2
        }
        val n = len / 2
        val newRms = sqrt(sum / n) / 32768.0
        val newPeak = p / 32768.0

        // 用新值抬高、用 decay 缓慢回落，视觉上更接近真实电平表
        rms = if (newRms > rms) newRms else rms * decay
        peak = if (newPeak > peak) newPeak else peak * decay
        lastDbfs = if (newRms <= 0.0) -120.0 else 20.0 * log10(newRms)
    }

    /** 归一化到 0..1，便于直接画进度条（已做对数压缩）。 */
    fun normalizedForUi(): Float {
        val v = rms
        if (v <= 1e-6) return 0f
        val db = 20.0 * log10(v)          // -120 .. 0
        val mapped = ((db + 60.0) / 60.0) // -60dB 以下算 0
        return mapped.coerceIn(0.0, 1.0).toFloat()
    }

    fun reset() {
        rms = 0.0
        peak = 0.0
        lastDbfs = -120.0
    }
}
