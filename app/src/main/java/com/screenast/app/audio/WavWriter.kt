package com.screenast.app.audio

import java.io.File
import java.io.RandomAccessFile
import java.nio.ByteBuffer
import java.nio.ByteOrder

/**
 * 增量写 WAV 文件。
 *
 * P0 的关键交付物之一：把抓到的音频落成标准 WAV，便于用 Audacity / ffmpeg 确认
 * “到底是抓不到声音，还是抓到了但后面链路有问题”。
 *
 * 同时在写入过程中统计 RMS / 峰值，用于实时电平表与“是否全静音”判定。
 */
class WavWriter(
    private val file: File,
    private val sampleRate: Int,
    private val channels: Int,
    private val bitsPerSample: Int = 16,
) {
    private val raf = RandomAccessFile(file, "rw")
    private var dataBytes = 0L

    /** 累计的平方和，用于计算整体 RMS（判断整段是否静音）。 */
    private var sumSquares = 0.0
    private var totalSamples = 0L
    private var peakAbs = 0

    /** 有声音的采样点数（|sample| > threshold）。 */
    private var loudSamples = 0L

    init {
        raf.setLength(0)
        writeHeader(dataLen = 0)
    }

    /**
     * 追加一小段 PCM（16bit little-endian 交错）。
     * @param buf 数据从 0 开始长度为 len
     */
    fun append(buf: ByteArray, len: Int) {
        if (len <= 0) return
        raf.seek(HEADER_SIZE + dataBytes)
        raf.write(buf, 0, len)
        dataBytes += len
        accumulate(buf, len)
    }

    private fun accumulate(buf: ByteArray, len: Int) {
        val bb = ByteBuffer.wrap(buf, 0, len - (len % 2)).order(ByteOrder.LITTLE_ENDIAN)
        while (bb.remaining() >= 2) {
            val s = bb.short.toInt()
            val a = if (s < 0) -s else s
            if (a > peakAbs) peakAbs = a
            sumSquares += s.toDouble() * s
            totalSamples++
            if (a > SILENCE_THRESHOLD) loudSamples++
        }
    }

    /** 整体 RMS，归一化到 0..1。 */
    fun rmsNormalized(): Double {
        if (totalSamples == 0L) return 0.0
        return Math.sqrt(sumSquares / totalSamples) / 32768.0
    }

    /** 峰值，归一化到 0..1。 */
    fun peakNormalized(): Double = peakAbs / 32768.0

    /** 是否整段静音（用于快速判定“抓不到”）。 */
    fun isEffectivelySilent(): Boolean = loudSamples < totalSamples / 1000

    fun durationMs(): Long =
        if (sampleRate <= 0 || channels <= 0) 0
        else dataBytes * 1000L / (sampleRate.toLong() * channels * (bitsPerSample / 8))

    fun dataBytes(): Long = dataBytes

    /** 补齐 RIFF 头（长度字段此时才已知），必须调用，否则文件不可播放。 */
    fun close(): File {
        writeHeader(dataBytes)
        raf.close()
        return file
    }

    private fun writeHeader(dataLen: Long) {
        val byteRate = sampleRate * channels * bitsPerSample / 8
        val blockAlign = channels * bitsPerSample / 8
        val riffSize = 36 + dataLen

        val header = ByteBuffer.allocate(HEADER_SIZE.toInt()).order(ByteOrder.LITTLE_ENDIAN)
        header.put("RIFF".toByteArray(Charsets.US_ASCII))
        header.putInt(riffSize.toInt())
        header.put("WAVE".toByteArray(Charsets.US_ASCII))
        header.put("fmt ".toByteArray(Charsets.US_ASCII))
        header.putInt(16)                       // PCM 子块大小
        header.putShort(1)                      // audioFormat = PCM
        header.putShort(channels.toShort())
        header.putInt(sampleRate)
        header.putInt(byteRate)
        header.putShort(blockAlign.toShort())
        header.putShort(bitsPerSample.toShort())
        header.put("data".toByteArray(Charsets.US_ASCII))
        header.putInt(dataLen.toInt())

        raf.seek(0)
        raf.write(header.array(), 0, HEADER_SIZE.toInt())
    }

    companion object {
        const val HEADER_SIZE = 44L

        /** 16bit 满量程 32767，取约 -72dBFS 作为“有声音”的门限。 */
        private const val SILENCE_THRESHOLD = 10
    }
}
