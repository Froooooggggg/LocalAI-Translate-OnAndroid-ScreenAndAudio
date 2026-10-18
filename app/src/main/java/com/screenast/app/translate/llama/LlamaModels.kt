package com.screenast.app.translate.llama

import android.content.Context
import java.io.File

/**
 * 2B GGUF 模型文件的落位规则。
 *
 * 和 [com.screenast.app.translate.nllb.NllbModels] 是同一套思路，但更简单：
 * llama.cpp 只要**一个** .gguf 文件，不像 NLLB 要凑齐 tokenizer + 三个 ONNX。
 *
 * 查找顺序（先命中先用）：
 *   1. 设置里显式指定的目录
 *   2. `filesDir/models/llama`            应用私有，卸载即清
 *   3. `getExternalFilesDir(null)/models/llama`   ← **adb push 最省事**
 *   4. `/data/local/tmp/snat-models/llama`  不需要任何权限，调试用
 *
 * 每个目录里既可以**直接放** .gguf，也可以放在 `gguf/` 子目录里。
 *
 * ⚠️ 模型 1.27GB，**绝对不能打进 APK**（否则装包体积翻倍且装不上）。
 * 所以「找不到模型」是**正常状态**，要有可读的诊断信息，不能抛异常了事。
 */
object LlamaModels {

    /**
     * 默认期望的文件名。
     *
     * 之所以要有个「期望名」而不是「随便捡个 .gguf」：目录里可能同时躺着
     * 0.8B 和 2B 两个模型（用户就是这种情况），随便捡会捡错，
     * 而两者的译文质量和延迟差一个档，排查时极难发现是模型选错了。
     */
    const val DEFAULT_FILE = "Huihui-Qwen3.5-2B-abliterated.Q4_K_M.gguf"

    /** 子目录名，用于「模型散着放」的情况。 */
    private val SUBDIRS = listOf("", "gguf", "llama")

    /** 一个候选落位：目录 + 该目录里命中的文件（没命中为 null）。 */
    data class Candidate(
        val dir: File,
        val file: File?,
        /** 人话描述这个目录是哪来的，诊断信息里要显示。 */
        val source: String,
    ) {
        val complete: Boolean get() = file != null
    }

    /**
     * 列出所有候选目录（不管有没有文件），供 [diagnostics] 逐条报告。
     *
     * @param preferredDir 设置里指定的目录；空串 / null 表示没指定。
     */
    fun candidates(context: Context, preferredDir: String?): List<Candidate> {
        val specs = mutableListOf<Pair<File, String>>()
        preferredDir?.takeIf { it.isNotBlank() }?.let {
            specs += File(it) to "设置指定"
        }
        specs += File(context.filesDir, "models/llama") to "应用私有目录"
        context.getExternalFilesDir(null)?.let {
            specs += File(it, "models/llama") to "外部私有目录（adb push）"
        }
        specs += File("/data/local/tmp/snat-models/llama") to "/data/local/tmp"

        return specs.map { (base, source) ->
            val hit = SUBDIRS
                .map { sub -> if (sub.isEmpty()) base else File(base, sub) }
                .firstNotNullOfOrNull { dir -> pick(dir) }
            Candidate(base, hit?.second, hit?.let { "$source（${it.first.name}/）" } ?: source)
        }
    }

    /** 在单个目录里挑模型：先认 [DEFAULT_FILE]，没有就捡第一个 .gguf。 */
    private fun pick(dir: File): Pair<File, File>? {
        if (!dir.isDirectory) return null
        val exact = File(dir, DEFAULT_FILE)
        if (exact.isFile) return dir to exact
        val any = dir.listFiles { f: File -> f.isFile && f.name.endsWith(".gguf", ignoreCase = true) }
            ?.minByOrNull { it.name }
        return any?.let { dir to it }
    }

    /** 解析出一份可用的模型；找不到返回 null。 */
    fun resolve(context: Context, preferredDir: String?): File? {
        // 「翻译模型」下拉里可以直接选某个 gguf 文件（用户 2026-10-18：导入什么就选什么，
        // 只负责推荐、不负责校验）。这种路径不是目录，先单独认一下。
        val direct = preferredDir?.takeIf { it.isNotBlank() }?.let { File(it) }
        if (direct != null && direct.isFile && direct.name.endsWith(".gguf", ignoreCase = true)) {
            return direct.takeIf { it.length() > 0 }
        }
        return candidates(context, preferredDir).firstOrNull { it.complete }?.file
    }

    /**
     * 人类可读的诊断串，直接往界面上打。
     * 找到 / 没找到都要能看出「差在哪」，而不是一句「未配置」。
     */
    fun diagnostics(context: Context, preferredDir: String?): String = buildString {
        val list = candidates(context, preferredDir)
        val hit = list.firstOrNull { it.complete }
        if (hit != null) {
            append("✅ 已定位模型：").append(hit.file!!.absolutePath)
            append("\n   大小 ").append(mb(hit.file.length())).append("，来自 ").append(hit.source)
        } else {
            append("❌ 没有找到模型文件（期望 ").append(DEFAULT_FILE).append("）")
        }
        append("\n\n候选目录：")
        list.forEach { c ->
            append("\n  ").append(if (c.complete) "✅" else "·").append(' ')
            append(c.dir.absolutePath)
            if (c.complete) append("  →  ").append(c.file!!.name).append(" (").append(mb(c.file.length())).append(")")
            else if (!c.dir.exists()) append("  （目录不存在）")
            else c.dir.listFiles()?.firstOrNull { it.name.endsWith(".gguf") }?.let {
                append("  有其它 gguf：").append(it.name)
            }
        }
        append("\n\n投递命令（USB）：")
        append("\n  adb push \"").append(DEFAULT_FILE).append("\" ")
        append("/sdcard/Android/data/").append(context.packageName).append("/files/models/llama/")
    }

    /**
     * FastRPC 找 HTP skel 用的**那一个**目录，喂给 `ADSP_LIBRARY_PATH`。
     *
     * ⚠️ **`ADSP_LIBRARY_PATH` 只有一个目录，不是冒号分隔的列表。**
     *    踩过的坑：按 PATH 的习惯传了 `a:b:c`，FastRPC 把整串当一个目录名，
     *    日志里是这么拼的（`apps_std_fopen_fd ... a:b:c/./libggml-htp-v79.so`），
     *    于是永远找不到，HTP0 会话打不开：error 0x80000406。
     *
     * 为什么需要它：`ggml-hexagon.cpp:3206` 把 skel 的 URI 拼成**裸文件名**
     * （`file:///libggml-htp-v79.so?...`），没有目录部分，FastRPC 只能按
     * `ADSP_LIBRARY_PATH` 去找。而 App 里这个 .so 本来在 APK 内部，
     * 所以 `app/build.gradle.kts` 打开了 `jniLibs.useLegacyPackaging`，
     * 让它被解压到 [android.content.pm.ApplicationInfo.nativeLibraryDir] ——
     * 那才是 FastRPC 能用的真实路径。
     *
     * 候选按优先级挑**第一个真实存在该文件的目录**；一个都没有就返回空串
     * （此时不设变量，让 FastRPC 用它自己的默认搜索路径）。
     */
    fun dspLibDirs(context: Context): String {
        val candidates = listOfNotNull(
            context.applicationInfo.nativeLibraryDir,
            File(context.filesDir, "dsp").absolutePath,
            // tools/ 里的 CLI 压测把官方包推在这儿；正式装机不会有，只是调试兜底。
            "/data/local/tmp/llama/lib",
        )
        return candidates.firstOrNull { File(it, HTP_SKEL).isFile } ?: ""
    }

    /** Hexagon 侧的 skel 文件名（v79 = SM8750）。 */
    const val HTP_SKEL = "libggml-htp-v79.so"

    private fun mb(bytes: Long): String =
        if (bytes >= 1024L * 1024 * 1024) String.format("%.2f GB", bytes / 1073741824.0)
        else String.format("%.1f MB", bytes / 1048576.0)
}
