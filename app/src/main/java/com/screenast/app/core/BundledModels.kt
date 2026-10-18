package com.screenast.app.core

import android.content.Context
import android.util.Log
import java.io.File

/**
 * APK 里自带的小模型：Silero VAD（644KB）与 GTCRN 降噪（536KB）。
 *
 * 为什么内置：这两个是「不装就整条语音链跑不起来」的小件，又只有几百 KB ——
 * 让用户去镜像站找、找错了还分辨不出来，不如直接塞进 APK，第一次启动解出来。
 * 大模型（gguf / NLLB / OCR / MI-GAN）仍然要用户自己导入或下载，见 [ModelImporter]。
 *
 * 解压目标是 `files/models/<rel>`，正好和 [ModelImporter.root] 同一套路径，
 * 所以状态行、各引擎的定位器都不用特判。
 */
object BundledModels {

    private const val TAG = "AST.Models"

    /** asset 路径 → 目标相对路径 → 期望字节数。 */
    private val small = listOf(
        Triple("asr/silero_vad.onnx", "asr/silero_vad.onnx", 643_854L),
        Triple("asr/gtcrn/gtcrn_simple.onnx", "asr/gtcrn/gtcrn_simple.onnx", 535_638L),
    )

    /**
     * 把随包小模型解到私有目录。已经有同字节数的文件就直接跳过（幂等，每次启动调一次也不贵）。
     *
     * @return 这次实际写出去的文件数（0 表示本来就是齐的）。
     */
    fun ensure(ctx: Context): Int {
        val root = ModelImporter.root(ctx)
        var wrote = 0
        small.forEach { (asset, rel, want) ->
            val out = File(root, rel)
            if (out.isFile && out.length() == want) return@forEach
            runCatching {
                out.parentFile?.mkdirs()
                val tmp = File(out.parentFile, out.name + ".part")
                ctx.assets.open(asset).use { input ->
                    tmp.outputStream().use { output -> input.copyTo(output, 64 * 1024) }
                }
                if (tmp.length() == want && tmp.renameTo(out)) {
                    wrote++
                    Log.i(TAG, "随包小模型已解出：$rel（${want / 1024}KB）")
                } else {
                    Log.w(TAG, "随包小模型解压失败：$rel（得到 ${tmp.length()}B，期望 ${want}B）")
                    tmp.delete()
                }
            }.onFailure { e ->
                Log.w(TAG, "随包小模型解压异常：$rel", e)
            }
        }
        return wrote
    }

    /** 这些是不是都齐了（给设置页显示用，不触发解压）。 */
    fun ready(ctx: Context): Boolean {
        val root = ModelImporter.root(ctx)
        return small.all { (_, rel, want) ->
            val f = File(root, rel)
            f.isFile && f.length() == want
        } || small.all { (_, rel, want) ->
            val f = File(ModelImporter.externalRoot(ctx), rel)
            f.isFile && f.length() == want
        }
    }
}
