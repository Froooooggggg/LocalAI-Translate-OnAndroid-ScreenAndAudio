package com.screenast.app.audio

/**
 * 采集侧 16 位 PCM → ASR 侧 16kHz 单声道 float 的重采样。
 *
 * ## 为什么不用 AudioRecord 直接要 16kHz
 *
 * 采集走的是 `AudioPlaybackCaptureConfiguration`，它是**按播放端混音**取的。
 * 混音器的工作采样率由系统决定（这台机器是 48kHz），向 `AudioRecord` 请求 16kHz
 * 并不保证真的按 16kHz 交付 —— 而且 `getMinBufferSize` 返回的是「这个请求能不能建起来」，
 * 不是「实际会给我多少」。一旦实际速率与请求不一致，重采样就会整体变调，
 * 识别结果会退化成乱码，且**看不出是速率问题**。所以这里按采集侧的真实速率来换算。
 *
 * ## 为什么自己写而不是引库
 *
 * 任务是定比整数抽取（48k→16k 是 3:1，44.1k→16k 是 2.75625:1），
 * 采样点总数是百万级、每秒要跑 48 次，属于热路径；引一个重采样库既不划算，
 * 也不好在这个项目里做定点验证（PC 侧没有同一份实现可对照）。
 *
 * ## 算法：分组平均（box filter）而不是直接丢弃
 *
 * 48k→16k 最省事的做法是「每 3 个点取第 1 个」，但那是裸抽取，
 * 8kHz 以上的分量会**折叠**回可听频段（混叠）。虽然播放捕获拿到的多半是
 * 已经过一遍编解码、高频本就不多的音乐/人声，但语音识别的前端（fbank）
 * 对混叠相当敏感：混进来的高频会在梅尔滤波器组的高频通道上堆能量，
 * 让模型把清音听成噪声。
 *
 * 所以对每个输出点取它覆盖的输入区间做**均值** —— 这是一个 3 抽头的滑动平均，
 * 在 16kHz 的奈奎斯特频率附近提供约 -13dB 的抑制，代价是高频略钝。
 * 对 SenseVoice 这种在 16kHz 上训练的模型来说，这个取舍是划算的：
 * 它本来就不依赖 8k 以上的信息。
 *
 * ## 相位连续（跨调用不丢样点）
 *
 * [feed] 会被反复调用（采集循环每次给 2048 个采样），而 2048 不是 3 的倍数。
 * 如果每次调用都从 0 相位重新开始，就会在每个调用边界丢掉 1~2 个采样，
 * 长期累积成周期性抖动（听感上是「卡顿」，识别上是丢字）。
 * 这里用 [pos]（以「输入采样数」为单位的浮点相位）跨调用保持连续性，
 * 并用 [pending] 暂存不足以凑齐一个输出窗口的尾部采样。
 *
 * 相位用 `pos += ratio` 且当它 ≥ 1 时减去 1，而不是每次重新乘除 ——
 * 浮点累加在这里是安全的：单次误差 ~1e-16，跑满一小时（1.7 亿个采样）
 * 累积误差也在 1e-8 量级，远小于一个采样。
 */
class AudioResampler(
    /** 输入采样率（采集侧真实速率）。 */
    private val inputRate: Int,
    /** 输出采样率（ASR 要的，SenseVoice 固定 16000）。 */
    private val outputRate: Int = 16_000,
) {
    init {
        require(inputRate >= outputRate) {
            "输入采样率 $inputRate 低于输出 $outputRate，这不是抽取而是插值，本类不支持"
        }
    }

    /** 每产生 1 个输出点，输入相位前进多少（≥1 表示抽取）。 */
    private val ratio: Double = inputRate.toDouble() / outputRate

    /** 环形暂存区：长度取「一次最大输入采样数」的余量，够放 ratio 个尾样即可。 */
    private var pending = ShortArray(0)
    private var pendingLen = 0

    /** 相位，单位是「输入采样数」，落在 [0, ratio)。 */
    private var pos: Double = 0.0

    /** 已丢弃的输出点数（诊断用；正常应恒为 0）。 */
    var dropped: Long = 0L
        private set

    /**
     * 喂入一段 16 位小端 PCM（[len] 字节），返回重采样后的 float 采样。
     *
     * @param len 必须是偶数；奇数尾字节会被忽略（并计入 [dropped]）。
     */
    fun feed(pcm: ByteArray, len: Int): FloatArray {
        if (len < 2) return EMPTY

        // 采样点用 ShortArray 收，不落 DoubleArray —— 百万级调用下这点分配很值钱
        val n = len / 2
        if (len % 2 != 0) dropped++

        // 把上次的尾巴和这次的数据拼起来
        val total = pendingLen + n
        val src = ShortArray(total)
        if (pendingLen > 0) System.arraycopy(pending, 0, src, 0, pendingLen)
        var i = 0
        var p = pendingLen
        while (i < n) {
            val lo = pcm[i * 2].toInt() and 0xFF
            val hi = pcm[i * 2 + 1].toInt()
            src[p] = ((hi shl 8) or lo).toShort()
            i++
            p++
        }

        // 输出上限：total 个输入最多产生 total/ratio + 1 个输出
        val out = FloatArray((total / ratio).toInt() + 2)
        var outN = 0
        var consumed = 0

        while (true) {
            // 本输出点覆盖的输入区间 [pos, pos + ratio)
            val start = consumed + pos
            val end = start + ratio
            if (end > total) break // 输入不够，留给下次
            val i0 = start.toInt()
            val i1 = end.toInt().coerceAtMost(total)

            var sum = 0
            var cnt = 0
            var k = i0
            while (k < i1) {
                sum += src[k].toInt()
                cnt++
                k++
            }
            // cnt 恒 ≥1：因为 end > start 且区间跨度 = ratio ≥ 1
            out[outN++] = (sum.toFloat() / cnt) / 32768f

            pos += ratio
            if (pos >= 1.0) {
                val whole = pos.toInt()
                consumed += whole
                pos -= whole
            }
        }

        // 把还没用掉的输入搬到 pending，供下次接着算
        val left = total - consumed
        if (pending.size < left) pending = ShortArray(left)
        if (left > 0) System.arraycopy(src, consumed, pending, 0, left)
        pendingLen = left

        return if (outN == out.size) out else out.copyOf(outN)
    }

    /** 清空相位与暂存，用于重新开始一段采集。 */
    fun reset() {
        pendingLen = 0
        pos = 0.0
        dropped = 0L
    }

    companion object {
        private val EMPTY = FloatArray(0)
    }
}
