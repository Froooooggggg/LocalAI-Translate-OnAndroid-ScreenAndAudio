package com.screenast.app.core

import android.app.ActivityManager
import android.content.Context
import java.io.File

/**
 * 内存探针：给「启动前够不够」这件事一个可解释的数字。
 *
 * ## 为什么不能只看 `MemoryInfo.availMem`
 *
 * 那个值是「系统当前可用」，不含 swap。真机上端侧大模型常靠「虚拟内存 / 内存扩展」撑过去
 * （澎湃 OS 的 3/5/7GB 档就是这么用的），只看 availMem 会把能用的情况误判成不够。
 * 反过来，swap 也可能小到杯水车薪 —— 所以两者都要看，并且把 swap 一起显示给用户。
 *
 * ## 为什么阈值按「当前配置实际会加载什么」加
 *
 * 各后端峰值差别很大（SenseVoice 约 240MB、NLLB 1.3GB、2B gguf 1.27GB + KV）。
 * 数字来自实测常驻峰值，宁可略保守：报错总比进程被 lmkd 杀掉好定位。
 */
object MemProbe {

    /** 一次内存快照（单位 MB）。 */
    data class Snapshot(
        /** 系统当前可用（`availMem`）。 */
        val availMb: Long,
        /** swap 总大小；0 表示没开虚拟内存。 */
        val swapTotalMb: Long,
        /** swap 当前空闲（还能借多少）。 */
        val swapFreeMb: Long,
        /** 单进程堆上限（`memoryClass`），只用于展示。 */
        val heapMb: Long,
    ) {
        val hasSwap: Boolean get() = swapTotalMb > 0
    }

    fun snapshot(context: Context): Snapshot {
        val am = context.getSystemService(Context.ACTIVITY_SERVICE) as? ActivityManager
        val mi = ActivityManager.MemoryInfo()
        am?.getMemoryInfo(mi)
        val (total, free) = swapInfo()
        return Snapshot(
            availMb = mi.availMem / 1048576L,
            swapTotalMb = total,
            swapFreeMb = free,
            heapMb = (am?.memoryClass ?: 256).toLong(),
        )
    }

    /** `/proc/swaps` 第二行起是各个 swap 设备；读不到就是没开虚拟内存。 */
    private fun swapInfo(): Pair<Long, Long> = runCatching {
        val f = File("/proc/swaps")
        if (!f.canRead()) return@runCatching 0L to 0L
        var totalKb = 0L
        var freeKb = 0L
        f.readLines().drop(1).forEach { line ->
            // Filename Type Size Used Priority
            val cols = line.trim().split(Regex("\\s+"))
            if (cols.size >= 4) {
                val size = cols[2].toLongOrNull() ?: 0L
                val used = cols[3].toLongOrNull() ?: 0L
                totalKb += size
                freeKb += (size - used).coerceAtLeast(0L)
            }
        }
        (totalKb / 1024L) to (freeKb / 1024L)
    }.getOrDefault(0L to 0L)

    /**
     * 当前配置大约要多少内存（MB）。
     *
     * 只在「确实会加载」时才加：provider 是云端就不给 gguf 留地方。
     */
    fun needMbFor(settings: Settings): Long {
        var need = 260L // 应用本身 + 界面 + 录音缓冲
        need += when (settings.providerId) {
            "llama" -> 1500L // 1.27GB 权重 + KV/激活
            "nllb" -> 1400L
            else -> 0L // 云端：本地不加载翻译模型
        }
        need += when (settings.asrModel) {
            "sensevoice", "paraformer" -> 420L
            "streaming" -> 380L
            "whisper" -> 620L
            else -> 300L
        }
        if (settings.visionInpaint) need += 120L
        return need
    }

    /**
     * 「用户已经确认过了」的进程内标记。
     *
     * 不做持久化：重启后内存是新的，警告应当重新出现一次。
     */
    var userAcceptedLowMem: Boolean = false
}

/** 「内存不够」警告弹窗要的数据。 */
data class LowMemWarning(
    val availMb: Long,
    val needMb: Long,
    val swapMb: Long,
) {
    /** 一句话讲清楚「差在哪」，避免只给一句「内存不足」。 */
    val detail: String
        get() = "当前可用 ${fmt(availMb)}，这次启动大约需要 ${fmt(needMb)}；" +
            if (swapMb > 0) "虚拟内存 ${fmt(swapMb)}（可用部分已计入）。"
            else "没有检测到虚拟内存（内存扩展）。"

    private fun fmt(mb: Long): String =
        if (mb >= 1024) "%.1fGB".format(mb / 1024.0) else "${mb}MB"
}
