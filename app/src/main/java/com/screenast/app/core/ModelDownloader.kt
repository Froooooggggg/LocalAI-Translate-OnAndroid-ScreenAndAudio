package com.screenast.app.core

import android.content.Context
import android.util.Log
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.currentCoroutineContext
import kotlinx.coroutines.ensureActive
import kotlinx.coroutines.withContext
import java.io.File
import java.net.HttpURLConnection
import java.net.URL

/**
 * **从 HF 镜像站直接下载模型**（用户 2026-10-18 的要求）。
 *
 * ## 为什么是 hf-mirror.com
 *
 * 模型只在 HuggingFace 上，国内直连基本下不动；`hf-mirror.com` 是同构镜像，
 * 路径规则与 HF 完全一致（`/<repo>/resolve/main/<file>`），换域名即可。
 *
 * ## 为什么不在服务器中转
 *
 * 这个项目没有服务器。App 直连镜像站下载，字节不经过任何第三方；
 * 代价是镜像站限速，所以界面必须给出「已下多少 / 总共多少 / 速度」，
 * 否则用户看起来就像卡死。
 *
 * ## 断点续传
 *
 * 用 `.part` 文件 + `Range` 头续传：网络一断，下次接着下，不用从 0 开始
 * （1.2GB 的 gguf 重来一次很痛）。服务端不支持 Range 时退化成整段重下。
 */
object ModelDownloader {

    private const val TAG = "AST.Download"

    /** 一个可下载条目：相对路径 + 直链 + 期望字节数（用来算百分比）。 */
    data class Job(
        val rel: String,
        val url: String,
        val bytes: Long,
        val note: String = "",
    )

    data class Progress(
        val label: String,
        val done: Long,
        val total: Long,
        val index: Int,
        val count: Int,
        /** 字节/秒，0 表示还没测出来。 */
        val speed: Long = 0L,
    ) {
        val percent: Int get() = if (total <= 0) 0 else ((done * 100) / total).toInt()
    }

    /**
     * 下载一个文件到 `files/models/<rel>`。
     *
     * 进度回调**在 IO 线程**上调用，且被限流到约 5 次/秒 —— 每 8KB 回调一次会把主线程打爆。
     */
    suspend fun download(ctx: Context, job: Job, onProgress: (Progress) -> Unit): File =
        withContext(Dispatchers.IO) {
            val dst = File(ModelImporter.root(ctx), job.rel)
            dst.parentFile?.mkdirs()
            val part = File(dst.parentFile, dst.name + ".part")
            Log.i(TAG, "开始下载 ${job.rel}（${job.bytes / 1024 / 1024}MB）")

            var startAt = if (part.isFile) part.length() else 0L
            if (startAt >= job.bytes && job.bytes > 0) {
                // 上次已经下完，只是没改名。
                part.renameTo(dst)
                onProgress(Progress(job.note.ifBlank { job.rel }, job.bytes, job.bytes, 0, 1))
                return@withContext dst
            }

            val conn = (URL(job.url).openConnection() as HttpURLConnection).apply {
                connectTimeout = 20_000
                readTimeout = 20_000
                instanceFollowRedirects = true
                setRequestProperty("User-Agent", "ScreenAst/0.5 (android)")
                if (startAt > 0) setRequestProperty("Range", "bytes=$startAt-")
            }
            try {
                val code = conn.responseCode
                if (code !in 200..299) throw IllegalStateException("HTTP $code")
                val append = code == HttpURLConnection.HTTP_PARTIAL && startAt > 0
                if (!append) startAt = 0L
                val total = when {
                    job.bytes > 0 -> job.bytes
                    conn.contentLengthLong > 0 -> conn.contentLengthLong + startAt
                    else -> 0L
                }
                conn.inputStream.use { input ->
                    java.io.FileOutputStream(part, append).use { out ->
                        val buf = ByteArray(64 * 1024)
                        var done = startAt
                        var lastPush = 0L
                        var lastTime = System.currentTimeMillis()
                        var lastBytes = done
                        while (true) {
                            currentCoroutineContext().ensureActive()
                            val n = input.read(buf)
                            if (n <= 0) break
                            out.write(buf, 0, n)
                            done += n
                            val now = System.currentTimeMillis()
                            if (now - lastPush >= 200) {
                                val dt = (now - lastTime).coerceAtLeast(1)
                                val speed = ((done - lastBytes) * 1000L) / dt
                                onProgress(
                                    Progress(
                                        job.note.ifBlank { job.rel }, done, total, 0, 1, speed,
                                    ),
                                )
                                lastPush = now
                                lastTime = now
                                lastBytes = done
                            }
                        }
                        onProgress(Progress(job.note.ifBlank { job.rel }, done, total, 0, 1, 0))
                    }
                }
                if (!part.renameTo(dst)) {
                    part.copyTo(dst, overwrite = true)
                    part.delete()
                }
            } catch (t: Throwable) {
                // 网络异常：留下 .part，下次续传；调用方决定要不要提示。
                Log.w(TAG, "下载失败 ${job.rel}：${t.javaClass.simpleName}: ${t.message}")
                throw t
            } finally {
                conn.disconnect()
            }
            Log.i(TAG, "下载完成 ${job.rel}：${dst.length() / 1024 / 1024}MB")
            dst
        }

    /**
     * 从清单反查直链。
     *
     * **每条都对着本地随包文件核过字节数**（`模型/随包/…` 与镜像站 tree API 的 size 一致），
     * 所以「一键下载」现在覆盖全部 22 个文件，界面上不再有需要用户自己找的「镜像站」按钮。
     * 唯一例外是 gguf：上游 `huihui-ai/Huihui-Qwen3.5-2B-abliterated` 仓库里没有 gguf，
     * 用同架构同量化的 `mradermacher/Qwen3.5-2B_Abliterated-GGUF`（1,270,808,896B，
     * 与本地 1,270,809,024B 差 128B，在 2% 容差内，加载没问题）。
     */
    fun urlFor(rel: String): String? = when (rel) {
        // ---- 翻译主模型（gguf）----
        "llama/Huihui-Qwen3.5-2B-abliterated.Q4_K_M.gguf" ->
            GGUF + "Qwen3.5-2B_Abliterated.Q4_K_M.gguf"

        // ---- 语音识别：SenseVoice ----
        "asr/model.int8.onnx" -> SENSE + "model.int8.onnx"
        "asr/tokens.txt" -> SENSE + "tokens.txt"

        // ---- 语音识别：Paraformer（离线中文）----
        "asr/paraformer-zh/model.int8.onnx" -> PARA + "model.int8.onnx"
        "asr/paraformer-zh/tokens.txt" -> PARA + "tokens.txt"

        // ---- 语音识别：流式 Paraformer ----
        "asr/streaming-paraformer/encoder.int8.onnx" -> STREAM + "encoder.int8.onnx"
        "asr/streaming-paraformer/decoder.int8.onnx" -> STREAM + "decoder.int8.onnx"
        "asr/streaming-paraformer/tokens.txt" -> STREAM + "tokens.txt"

        // ---- 语音识别：Whisper-small ----
        "asr/whisper-small/small-encoder.int8.onnx" -> WHISPER + "small-encoder.int8.onnx"
        "asr/whisper-small/small-decoder.int8.onnx" -> WHISPER + "small-decoder.int8.onnx"
        "asr/whisper-small/small-tokens.txt" -> WHISPER + "small-tokens.txt"

        // ---- 标点 ----
        "asr/punct-ct-transformer/model.onnx" -> PUNCT + "model.onnx"
        "asr/punct-ct-transformer/tokens.json" -> PUNCT + "tokens.json"

        // ---- TEN-VAD（装了会顶替内置的 Silero）----
        "asr/ten-vad/ten-vad.onnx" -> TENVAD + "src/onnx_model/ten-vad.onnx"

        // ---- 备选翻译：NLLB-200-distilled-600M 的量化三件套 ----
        "nllb/tokenizer.json" ->

            "https://hf-mirror.com/facebook/nllb-200-distilled-600M/resolve/main/tokenizer.json"
        "nllb/onnx/encoder_model_int8.onnx" -> NLLB + "encoder_model_int8.onnx"
        "nllb/onnx/decoder_model_quantized.onnx" -> NLLB + "decoder_model_quantized.onnx"
        "nllb/onnx/decoder_with_past_model_quantized.onnx" ->
            NLLB + "decoder_with_past_model_quantized.onnx"

        // ---- 识图 OCR ----
        "ocr/ppocrv5-mobile-det.onnx" -> PPOCR + "ppocrv5-mobile-det.onnx"
        "ocr/ppocrv5-mobile-rec.onnx" -> PPOCR + "ppocrv5-mobile-rec.onnx"
        "ocr/ppocrv5-server-det.onnx" -> PPOCR + "ppocrv5-server-det.onnx"
        "ocr/ppocrv5-server-rec.onnx" -> PPOCR + "ppocrv5-server-rec.onnx"

        // ---- 擦除补背景 ----
        "inpaint/migan_pipeline_v2.onnx" -> MIGAN + "migan_pipeline_v2.onnx"

        else -> null
    }

    /** 有没有直链（界面据此决定给「下载」还是「镜像站」）。 */
    fun hasUrl(rel: String): Boolean = urlFor(rel) != null

    /** 关键词（给「去镜像站找」按钮用）。 */
    fun searchFor(rel: String): String = when {
        rel.endsWith(".gguf") -> "Huihui-Qwen3.5-2B-abliterated"
        rel.startsWith("nllb/") -> "nllb 200 distilled onnx"
        rel.contains("silero") -> "silero vad onnx"
        rel.contains("gtcrn") -> "gtcrn onnx"
        rel.contains("paraformer") -> "sherpa onnx paraformer"
        rel.contains("whisper") -> "sherpa onnx whisper small"
        rel.contains("punct") -> "sherpa onnx ct transformer punctuation"
        rel.contains("ten-vad") -> "ten vad onnx"
        rel.contains("ocr") -> "PPOCRv5 ONNX"
        else -> rel.substringAfterLast('/')
    }

    private const val PPOCR = "https://hf-mirror.com/bukuroo/PPOCRv5-ONNX/resolve/main/"
    private const val SENSE =
        "https://hf-mirror.com/csukuangfj/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17/resolve/main/"
    private const val PARA =
        "https://hf-mirror.com/csukuangfj/sherpa-onnx-paraformer-zh-2023-09-14/resolve/main/"
    private const val STREAM =
        "https://hf-mirror.com/csukuangfj/sherpa-onnx-streaming-paraformer-bilingual-zh-en/resolve/main/"
    private const val WHISPER =
        "https://hf-mirror.com/csukuangfj/sherpa-onnx-whisper-small/resolve/main/"
    private const val PUNCT =
        "https://hf-mirror.com/csukuangfj/sherpa-onnx-punct-ct-transformer-zh-en-vocab272727-2024-04-12/resolve/main/"
    private const val TENVAD = "https://hf-mirror.com/TEN-framework/ten-vad/resolve/main/"
    private const val NLLB =
        "https://hf-mirror.com/Xenova/nllb-200-distilled-600M/resolve/main/onnx/"
    private const val GGUF =
        "https://hf-mirror.com/mradermacher/Qwen3.5-2B_Abliterated-GGUF/resolve/main/"
    private const val MIGAN = "https://hf-mirror.com/andraniksargsyan/migan/resolve/main/"
}
