package com.screenast.app.translate.nllb

import android.content.Context
import java.io.File

/**
 * NLLB 模型文件的位置解析。
 *
 * 为什么要有这个类：模型一共 1.27GB（encoder 396MB + decoder 449MB + decoder_past 424MB），
 * **不能打进 APK**。所以运行时必须能找到外部推入的模型目录。开发阶段的现实做法是
 * `adb push` 到 `/data/local/tmp/snat-models/`，那里不受 scoped storage 限制，
 * ONNX Runtime 也能直接按路径打开。
 *
 * 查找顺序（先命中先用）：
 *  1. [preferredDir]（来自 Settings.nllbModelDir，用户显式指定时优先）
 *  2. `filesDir/models/nllb`：App 私有目录，正式版的落位（首次运行从下载/导入解包）
 *  3. `getExternalFilesDir(null)/models/nllb`：`/sdcard/Android/data/<pkg>/files/...`，
 *     免 root 也能用 `adb push` 直接写进去，是**真机上最省事的开发落位**
 *  4. `/data/local/tmp/snat-models/nllb`：模拟器/有 shell 权限时的落位
 *
 * 每个候选目录下都同时接受两种子布局：直接放文件，或放在 `onnx/` 子目录里
 * （后者与 HuggingFace 仓库的 `onnx/` 目录结构一致，搬运时不用改文件名）。
 */
object NllbModels {

    const val ENCODER = "encoder_model_int8.onnx"
    const val DECODER = "decoder_model_quantized.onnx"
    const val DECODER_PAST = "decoder_with_past_model_quantized.onnx"

    /** 一个可用的模型目录。 */
    data class Location(
        val dir: File,
        val encoder: File,
        val decoder: File,
        val decoderPast: File,
        /** 来源标签，用于在界面上显示"模型从哪加载的"。 */
        val source: String,
    ) {
        val complete: Boolean
            get() = encoder.isFile && decoder.isFile && decoderPast.isFile

        fun summary(): String = buildString {
            append(source).append('\n')
            append("  ").append(NllbTranslator.describe(encoder)).append('\n')
            append("  ").append(NllbTranslator.describe(decoder)).append('\n')
            append("  ").append(NllbTranslator.describe(decoderPast))
        }
    }

    /** 所有候选目录，按优先级排好。 */
    fun candidates(context: Context, preferredDir: String): List<Pair<File, String>> {
        val out = ArrayList<Pair<File, String>>(6)
        if (preferredDir.isNotBlank()) {
            out.add(File(preferredDir) to "设置指定")
            out.add(File(preferredDir, "onnx") to "设置指定/onnx")
        }
        val files = File(context.filesDir, "models/nllb")
        out.add(files to "应用私有目录")
        out.add(File(files, "onnx") to "应用私有目录/onnx")

        context.getExternalFilesDir(null)?.let { ext ->
            val d = File(ext, "models/nllb")
            out.add(d to "外部私有目录(adb push 推荐)")
            out.add(File(d, "onnx") to "外部私有目录/onnx")
        }

        val tmp = File("/data/local/tmp/snat-models/nllb")
        out.add(tmp to "shell 临时目录")
        out.add(File(tmp, "onnx") to "shell 临时目录/onnx")

        return out
    }

    /** 找到第一个三件套齐全的目录；都没有则返回 null。 */
    fun resolve(context: Context, preferredDir: String = ""): Location? {
        for ((dir, label) in candidates(context, preferredDir)) {
            val loc = Location(
                dir = dir,
                encoder = File(dir, ENCODER),
                decoder = File(dir, DECODER),
                decoderPast = File(dir, DECODER_PAST),
                source = "$label：${dir.absolutePath}",
            )
            if (loc.complete) return loc
        }
        return null
    }

    /**
     * 诊断用：把每个候选目录里「哪几个文件齐、哪几个缺」列出来。
     * 模型加载失败时最需要的就是这张表 —— 光看一个 null 无法判断是路径不对还是文件没推完。
     */
    fun diagnostics(context: Context, preferredDir: String = ""): List<String> =
        candidates(context, preferredDir).map { (dir, label) ->
            val parts = listOf(ENCODER, DECODER, DECODER_PAST).map { name ->
                val f = File(dir, name)
                when {
                    !f.isFile -> "$name ✗"
                    f.length() == 0L -> "$name ✗(0字节)"
                    else -> "$name ✓${f.length() / 1048576}MB"
                }
            }
            "$label\n  ${dir.absolutePath}\n  " + parts.joinToString("  ")
        }
}
