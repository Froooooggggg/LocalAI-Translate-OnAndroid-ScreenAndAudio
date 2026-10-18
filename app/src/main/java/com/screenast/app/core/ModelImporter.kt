package com.screenast.app.core

import android.content.Context
import android.database.Cursor
import android.net.Uri
import java.util.zip.ZipInputStream
import android.provider.DocumentsContract
import android.util.Log
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import java.io.File
import java.io.InputStream

/**
 * **离线模型导入** —— 把用户自己带的 Models 文件夹搬进应用私有目录。
 *
 * ## 为什么需要它
 *
 * 这个 App 的四个能力（翻译 / 识别 / OCR / 擦除补背景）分别要一个到几个模型文件，
 * 加起来 4GB 上下。**打进 APK 不现实**（APK 单文件有上限、装不下），
 * 又不能让最终用户自己去各个站点找模型 —— 他们的原话是「受众对象没那个能力去下载」。
 *
 * 所以：模型文件跟着发布包一起发（同一层目录放一个 Models 文件夹），首次打开时点一次
 * 「导入模型」，或把 Models 整个拷到手机存储根目录（/sdcard/Models）让 App 自己捡。
 *
 * ## 目录约定
 *
 * Models 里的子目录名**与私有目录一一对应**，所以搬过去就是原样递归拷贝：
 *
 * ```
 * Models/asr/model.int8.onnx   ->  files/models/asr/model.int8.onnx
 * Models/nllb/onnx/  下的 onnx 文件  ->  files/models/nllb/onnx/ 同名文件
 * Models/ocr/        下的 onnx 文件  ->  files/models/ocr/ 同名文件
 * Models/llama/      下的 gguf 文件  ->  files/models/llama/ 同名文件
 * ```
 *
 * `Models/inpaint/migan_pipeline_v2.onnx` 是**擦除补背景**（MI-GAN）用的 28MB 模型。
 * 它以前内嵌在 APK 的 assets 里，现在也走导入：没导入只是「擦除档不可用」，
 * 识别、贴片、翻译都不受影响。
 *
 * ## 为什么用 SAF 而不是直接读 /sdcard/Models
 *
 * Android 11+ 的分区存储下，读 /sdcard 里的**非媒体文件**需要 MANAGE_EXTERNAL_STORAGE
 * （应用商店会因此被拒）；SAF 的 ACTION_OPEN_DOCUMENT_TREE 不需要任何权限，
 * 用户点两下选目录即可，兼容所有版本。
 *
 * 直接路径那条腿仍保留（[importFromDir]）：开发期 adb push 到 /sdcard/Models 更省事，
 * 部分 ROM 对自有目录也不拦。
 *
 * ## 为什么不用 androidx.documentfile
 *
 * 项目里没引那个依赖，为一个目录遍历加一条新依赖不划算；这里直接用
 * [DocumentsContract] 查子文档，逻辑不到 60 行。
 */
object ModelImporter {

    private const val TAG = "AST.Models"

    // ---------------------------------------------------------------- 按类别导入
    //
    // 2026-10-18 用户要求：点「导入模型」先问「你导的是哪一类」，再进文件选择器，
    // 只把选中的文件放进对应子目录；其余的 VAD / 词表 / MI-GAN 之类小模型不走这条路。

    /** 翻译大模型（gguf）→ `models/llama/`。 */
    const val CAT_TRANSLATE = "translate"

    /** 识图 OCR 权重（det/rec onnx）→ `models/ocr/`。 */
    const val CAT_VISION = "vision"

    /** 语音识别（SenseVoice / paraformer / whisper / VAD / tokens）→ `models/asr/`。 */
    const val CAT_ASR = "asr"

    /** 类别 → 私有目录里 `models/` 下的子目录名。 */
    fun subdirOf(cat: String): String = when (cat) {
        CAT_VISION -> "ocr"
        CAT_ASR -> "asr"
        else -> "llama"
    }

    /** 一个模型文件：相对 Models 的路径 + 期望字节数（用来判断「已导入」和「换了大模型」）。 */
    data class Item(
        val rel: String,
        val bytes: Long,
        val note: String = "",
        /** 缺了就直接用不了核心功能（同传 / 识图）。界面把这一类单列，用户一眼知道还差几个。 */
        val essential: Boolean = false,
    )

    /**
     * 全量清单。**字节数是从发布包实测抄下来的**，不是估算值 ——
     * 用它判「已就位」比只判文件存在可靠：用户换了个同名但不同大小的模型时能识别出来。
     */
    val items: List<Item> = listOf(
        // ---- 端侧大模型（翻译主力，默认就是它）----
        Item(
            "llama/Huihui-Qwen3.5-2B-abliterated.Q4_K_M.gguf", 1_270_809_024L, "端侧大模型 2B",
            essential = true,
        ),

        // ---- 语音识别：SenseVoice（默认后端）+ VAD + 降噪 ----
        Item("asr/model.int8.onnx", 239_233_841L, "SenseVoice", essential = true),
        Item("asr/tokens.txt", 315_894L, "SenseVoice 词表", essential = true),

        // ---- 语音识别：其余三个可选后端 + 标点 ----
        Item("asr/paraformer-zh/model.int8.onnx", 243_371_218L, "Paraformer 中文"),
        Item("asr/paraformer-zh/tokens.txt", 75_756L, "Paraformer 词表"),
        Item("asr/streaming-paraformer/encoder.int8.onnx", 165_462_184L, "流式 Paraformer 编码器"),
        Item("asr/streaming-paraformer/decoder.int8.onnx", 71_664_561L, "流式 Paraformer 解码器"),
        Item("asr/streaming-paraformer/tokens.txt", 75_756L, "流式 Paraformer 词表"),
        Item("asr/whisper-small/small-encoder.int8.onnx", 112_442_483L, "Whisper-small 编码器"),
        Item("asr/whisper-small/small-decoder.int8.onnx", 262_226_114L, "Whisper-small 解码器"),
        Item("asr/whisper-small/small-tokens.txt", 816_730L, "Whisper 词表"),
        Item("asr/punct-ct-transformer/model.onnx", 294_372_519L, "中英标点模型"),
        Item("asr/punct-ct-transformer/tokens.json", 4_207_480L, "标点词表"),
        Item("asr/ten-vad/ten-vad.onnx", 315_449L, "TEN-VAD"),

        // ---- 备选翻译：NLLB（句级，快但没上下文）----
        Item("nllb/onnx/encoder_model_int8.onnx", 415_346_576L, "NLLB 编码器"),
        Item("nllb/onnx/decoder_model_quantized.onnx", 470_533_055L, "NLLB 解码器"),
        Item("nllb/onnx/decoder_with_past_model_quantized.onnx", 445_006_987L, "NLLB 解码器(带缓存)"),
        // 词表以前内嵌在 APK；它由 CC-BY-NC-4.0 的 NLLB 词表编译而来，不该随包分发，
        // 现在作为可选文件（自己用 tools/build_nllb_tokenizer.py 编，放 models/nllb/）。
        Item("nllb/tokenizer.json", 17_331_176L, "NLLB 词表（下载后自动编译）"),

        // ---- 识图 OCR：mobile 两件是默认，server 两件是可选高精度 ----
        Item("ocr/ppocrv5-mobile-det.onnx", 4_748_769L, "OCR mobile 检测", essential = true),
        Item("ocr/ppocrv5-mobile-rec.onnx", 16_517_247L, "OCR mobile 识别", essential = true),
        Item("ocr/ppocrv5-server-det.onnx", 87_697_340L, "OCR server 检测"),
        Item("ocr/ppocrv5-server-rec.onnx", 84_137_438L, "OCR server 识别"),

        // ---- 擦除补背景（可选；从 APK 里拿出来了，不导入就只是擦除档不可用）----
        Item("inpaint/migan_pipeline_v2.onnx", 28_079_181L, "MI-GAN 补背景"),
    )

    /** 已经内置在 APK assets 里的模型，只用于状态行展示，不参与拷贝。 */
    private val bundled = listOf(
        "ocr/ppocrv5_dict.txt" to 92_395L,
        // 这两个是 2026-10-18 从「要用户下载」改成「随包内置」的：体积只有几百 KB，
        // 缺了整条语音链都跑不起来，没必要让用户去镜像站找（见 BundledModels.ensure）。
        "asr/silero_vad.onnx" to 643_854L,
        "asr/gtcrn/gtcrn_simple.onnx" to 535_638L,
    )

    /** 私有目录里模型的根：files/models。 */
    fun root(ctx: Context): File = File(ctx.filesDir, "models")

    /**
     * 外部私有目录里的模型根：`/sdcard/Android/data/<pkg>/files/models`。
     *
     * 这个位置**不需要任何权限**就能读写（自己的外部目录），adb push 与文件管理器都能往这儿放，
     * 各引擎的定位器也都会查它 —— 所以「不导入也能用」在这里是成立的。
     */
    fun externalRoot(ctx: Context): File =
        File(ctx.getExternalFilesDir(null) ?: ctx.filesDir, "models")

    data class Status(
        val ready: Int,
        val total: Int,
        /** 已就位的字节数（不是总量）—— 显示「3.7GB / 4.0GB」用这个。 */
        val readyBytes: Long,
        val totalBytes: Long,
        /** 核心功能（同传 / 识图）必需的那几项就位了几个。 */
        val essentialReady: Int,
        val essentialTotal: Int,
        /** 还缺哪些必需项（按清单顺序，界面只取前两个显示）。 */
        val missingEssential: List<Item>,
    ) {
        val allReady: Boolean get() = ready == total

        /** 核心功能能不能用。false 时主界面会挂一条「去导入模型」的提示。 */
        val usable: Boolean get() = essentialReady == essentialTotal
    }

    /** 扫一遍私有目录，返回「几个已就位」。不读文件内容，只看存在性与大小。 */
    fun status(ctx: Context): Status {
        val root = root(ctx)
        var ready = 0
        var readyBytes = 0L
        var totalBytes = 0L
        var essReady = 0
        val essMissing = ArrayList<Item>()
        items.forEach { it ->
            totalBytes += it.bytes
            // 允许同名文件大小差一点点（不同量化版本），差 2% 以内算就位。
            //
            // 两个位置都算：私有目录 + 外部私有目录（/sdcard/Android/data/<pkg>/files/models）。
            // 后者是「adb push / 从手机文件管理器拖进去」的落点，定位器（LlamaModels /
            // NllbModels / VisionInpaint）本来就会去那儿找，所以状态行必须同口径，
            // 否则模型明明能用、主界面还挂着「还缺 1 个」。
            val ok = isOk(File(root, it.rel), it.bytes) ||
                isOk(File(externalRoot(ctx), it.rel), it.bytes)
            if (ok) {
                ready++
                readyBytes += it.bytes
                if (it.essential) essReady++
            } else if (it.essential) {
                essMissing += it
            }
        }
        return Status(
            ready, items.size, readyBytes, totalBytes,
            essReady, items.count { it.essential }, essMissing,
        )
    }

    private fun isOk(f: File, expect: Long): Boolean {
        if (!f.isFile) return false
        val n = f.length()
        if (n == expect) return true
        return kotlin.math.abs(n - expect) <= expect / 50  // 2%
    }

    /** 导入结果，界面按这个提示。 */
    data class Report(
        val copied: Int,
        val skipped: Int,
        val copiedBytes: Long,
        val missing: List<String>,
        val failed: List<String>,
        /** 这次真正拷进来的相对路径（给导入后的冒烟测试挑文件用）。 */
        val copiedRels: List<String> = emptyList(),
    )

    /** 进度回调：第 done 个（共 total）/ 当前文件相对路径 / 累计写入字节。 */
    typealias Progress = (done: Int, total: Int, rel: String, bytes: Long) -> Unit

    /**
     * 从 SAF 选中的目录导入。**会递归**：用户选 Models 或 Models/asr 都能跑，
     * 只要树里能找到清单里的相对路径（找不到的记进 [Report.missing]，不是错误）。
     */
    suspend fun importFromTree(ctx: Context, tree: Uri, onProgress: Progress): Report =
        withContext(Dispatchers.IO) {
            // 先把整棵树摊平成 rel -> documentId。4GB 也就二十来个节点，成本可忽略。
            val found = HashMap<String, String>()
            runCatching { flatten(ctx, tree, DocumentsContract.getTreeDocumentId(tree), "", found, 0) }
                .onFailure { Log.e(TAG, "遍历所选目录失败", it) }
            copyAll(
                ctx,
                items.map { it.rel to { rel: String -> found[rel]?.let { id -> openDoc(ctx, tree, id) } } },
                onProgress,
            )
        }

    /** 从真实目录导入（/sdcard/Models 这种，开发期 adb push 用）。 */
    suspend fun importFromDir(ctx: Context, dir: File, onProgress: Progress): Report =
        withContext(Dispatchers.IO) {
            copyAll(
                ctx,
                items.map { it.rel to { _: String -> File(dir, it.rel).takeIf { f -> f.isFile }?.inputStream() } },
                onProgress,
            )
        }

    /** 手机存储根目录下的 Models（[importFromDir] 的来源）。没有就返回 null。 */
    fun autoDetect(ctx: Context? = null): File? {
        // 应用自己的外部目录（/sdcard/Android/data/<pkg>/files/Models）**一定能读**：
        // 分区存储下别人的目录里放 .onnx/.gguf（非媒体文件）是读不到的，只有这里例外。
        // adb 推模型时推到这儿，App 就能一键导入，不必走 SAF 手动选目录。
        val ext = ctx?.let { externalRoot(it) }
        val direct = listOf(
            "/sdcard/Models",
            "/storage/emulated/0/Models",
            "/sdcard/Download/Models",
            // 用户 2026-10-18 让把模型推到「便携存储」；列在这里只是碰运气 ——
            // 能被系统放行就读，读不到就走下面的外部目录。
            "/storage/emulated/0/便携存储",
        ).map { File(it) }
        return (listOfNotNull(ext) + direct)
            .firstOrNull { it.isDirectory && hasModels(it) }
    }

    /** 目录里至少有一个清单文件，才算「模型目录」（否则 /sdcard 根也能通过 isDirectory）。 */
    private fun hasModels(dir: File): Boolean =
        items.any { File(dir, it.rel).isFile }

    /**
     * 从 .zip 导入。
     *
     * 压缩包里的目录层级可以随便套（`Models/asr/model.int8.onnx`、`asr/model.int8.onnx`、
     * `下载/Models/asr/...` 都行）：按「清单里的相对路径是条目名的**后缀**」匹配。
     *
     * 为什么也支持 zip：发布包一般就是压一个 zip 丢网盘，让用户先解压再翻文件夹更容易
     * 出错（手机文件管理器的解压路径各家都不一样）。zip 是流式读的，1.4GB 也不吃内存。
     */
    suspend fun importFromZip(ctx: Context, uri: Uri, onProgress: Progress): Report =
        withContext(Dispatchers.IO) {
            val root = root(ctx)
            val done = HashSet<String>()
            var copied = 0
            var copiedBytes = 0L
            val failed = ArrayList<String>()
            val ins = runCatching { ctx.contentResolver.openInputStream(uri) }.getOrNull()
            if (ins == null) {
                return@withContext Report(0, 0, 0L, items.map { it.rel }, listOf("打不开所选文件"))
            }
            ins.use { raw ->
                ZipInputStream(raw.buffered(1 shl 16)).use { zip ->
                    while (true) {
                        val entry = zip.nextEntry ?: break
                        val name = entry.name
                        if (entry.isDirectory) continue
                        // 去掉可能的前缀目录，再和清单比：谁的 rel 是条目名的后缀就归谁。
                        val item = items.firstOrNull { it.rel == name || name.endsWith("/" + it.rel) }
                        if (item == null || item.rel in done) continue
                        val dest = File(root, item.rel)
                        if (isOk(dest, item.bytes)) {
                            done += item.rel
                            onProgress(done.size, items.size, item.rel, copiedBytes)
                            continue
                        }
                        try {
                            dest.parentFile?.mkdirs()
                            val part = File(dest.parentFile, dest.name + ".part")
                            var n = 0L
                            part.outputStream().use { out ->
                                val buf = ByteArray(1 shl 20)
                                while (true) {
                                    val r = zip.read(buf)
                                    if (r <= 0) break
                                    out.write(buf, 0, r)
                                    n += r
                                }
                            }
                            if (n < item.bytes / 2) {
                                part.delete()
                                failed += item.rel
                            } else {
                                if (dest.exists()) dest.delete()
                                part.renameTo(dest)
                                copied++
                                copiedBytes += n
                                done += item.rel
                            }
                        } catch (t: Throwable) {
                            Log.w(TAG, "zip 里拷 ${item.rel} 失败", t)
                            failed += item.rel
                        }
                        onProgress(done.size, items.size, item.rel, copiedBytes)
                    }
                }
            }
            val missing = items.map { it.rel }.filter { it !in done && it !in failed }
            Report(copied, items.size - copied - missing.size - failed.size, copiedBytes, missing, failed, done.toList())
        }

    // ---------------------------------------------------------------- SAF 遍历

    private fun childUri(tree: Uri, parentId: String): Uri =
        DocumentsContract.buildChildDocumentsUriUsingTree(tree, parentId)

    private fun docUri(tree: Uri, docId: String): Uri =
        DocumentsContract.buildDocumentUriUsingTree(tree, docId)

    private fun openDoc(ctx: Context, tree: Uri, docId: String): InputStream? =
        runCatching { ctx.contentResolver.openInputStream(docUri(tree, docId)) }.getOrNull()

    /**
     * 递归摊平目录树。`mime_type = DIR` 用
     * `vnd.android.document/directory` 判断（各 Provider 都遵守）。
     */
    private fun flatten(
        ctx: Context,
        tree: Uri,
        parentId: String,
        prefix: String,
        out: MutableMap<String, String>,
        depth: Int,
    ) {
        if (depth > 6) return
        val cursor: Cursor = runCatching {
            ctx.contentResolver.query(
                childUri(tree, parentId),
                arrayOf(
                    DocumentsContract.Document.COLUMN_DOCUMENT_ID,
                    DocumentsContract.Document.COLUMN_DISPLAY_NAME,
                    DocumentsContract.Document.COLUMN_MIME_TYPE,
                ),
                null, null, null,
            )
        }.getOrNull() ?: return
        cursor.use { c ->
            while (c.moveToNext()) {
                val id = c.getString(0) ?: continue
                val name = c.getString(1) ?: continue
                val mime = c.getString(2) ?: ""
                val rel = if (prefix.isEmpty()) name else "$prefix/$name"
                if (mime == DocumentsContract.Document.MIME_TYPE_DIR) {
                    flatten(ctx, tree, id, rel, out, depth + 1)
                } else {
                    out[rel] = id
                }
            }
        }
    }

    // ---------------------------------------------------------------- 拷贝主循环

    /**
     * 统一的拷贝主循环。`open(rel)` 返回 null 表示源里没有这个文件。
     *
     * 逐文件上报进度而不是按字节：单文件最大 1.2GB，按字节回调会把主线程刷爆。
     */
    private fun copyAll(
        ctx: Context,
        sources: List<Pair<String, (String) -> InputStream?>>,
        onProgress: Progress,
    ): Report {
        val root = root(ctx)
        var copied = 0
        var skipped = 0
        var copiedBytes = 0L
        val missing = ArrayList<String>()
        val failed = ArrayList<String>()
        val copiedRels = ArrayList<String>()

        sources.forEachIndexed { idx, (rel, open) ->
            val dest = File(root, rel)
            val expect = items.first { it.rel == rel }.bytes
            if (isOk(dest, expect)) {
                skipped++
                onProgress(idx + 1, sources.size, rel, copiedBytes)
                return@forEachIndexed
            }
            val ins = runCatching { open(rel) }.getOrNull()
            if (ins == null) {
                missing += rel
                Log.w(TAG, "源里没有：$rel")
                onProgress(idx + 1, sources.size, rel, copiedBytes)
                return@forEachIndexed
            }
            val t0 = System.currentTimeMillis()
            try {
                dest.parentFile?.mkdirs()
                // 先写 .part 再改名：中途被杀不会留下一个「大小不对但存在」的半个模型，
                // 那种文件最坑 —— 加载器会去读它、然后崩在 native 层。
                val part = File(dest.parentFile, dest.name + ".part")
                var n = 0L
                ins.use { input ->
                    part.outputStream().use { output ->
                        val buf = ByteArray(1 shl 20)
                        while (true) {
                            val r = input.read(buf)
                            if (r <= 0) break
                            output.write(buf, 0, r)
                            n += r
                        }
                    }
                }
                if (n < expect / 2) {
                    // 拷出来的明显不是完整文件（源坏了 / 传了一半）。删掉，算失败。
                    part.delete()
                    failed += rel
                    Log.w(TAG, "拷出来只有 $n 字节（期望 $expect），丢弃：$rel")
                } else {
                    if (dest.exists()) dest.delete()
                    part.renameTo(dest)
                    copied++
                    copiedRels += rel
                    copiedBytes += n
                    Log.i(TAG, "导入 $rel：${n / 1024 / 1024}MB，${System.currentTimeMillis() - t0}ms")
                }
            } catch (t: Throwable) {
                failed += rel
                Log.e(TAG, "导入失败：$rel", t)
            } finally {
                runCatching { ins.close() }
            }
            onProgress(idx + 1, sources.size, rel, copiedBytes)
        }
        Log.i(
            TAG,
            "导入结束：拷 $copied / 跳过 $skipped / 缺 ${missing.size} / 失败 ${failed.size}，" +
                "${copiedBytes / 1024 / 1024}MB",
        )
        return Report(copied, skipped, copiedBytes, missing, failed, copiedRels)
    }

    /** 清单总量，给界面显示「约 4.0GB」。 */
    fun totalBytes(): Long = items.sumOf { it.bytes }

    fun bundledText(): String = bundled.joinToString("、") { "${it.first}（${it.second / 1024 / 1024}MB）" }
}
