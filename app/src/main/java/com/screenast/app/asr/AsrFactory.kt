package com.screenast.app.asr

import com.k2fsa.sherpa.onnx.OfflineModelConfig
import com.k2fsa.sherpa.onnx.OfflineParaformerModelConfig
import com.k2fsa.sherpa.onnx.OfflineSenseVoiceModelConfig
import com.k2fsa.sherpa.onnx.OfflineWhisperModelConfig
import com.screenast.app.core.Lang
import java.io.File

/**
 * 识别后端的选择与构造。
 *
 * ## 为什么集中在一个 object
 *
 * 后端的「在哪」和「怎么配」是两件互相牵连的事：换模型不只是换个文件名，
 * 还要同时换 `OfflineModelConfig` 里填哪个字段、语种从哪来、要不要补标点。
 * 这三件事分散在调用方会**各自漂移**（项目里已经吃过「两处实现不一致」的亏），
 * 所以统一收在这里，调用方只说一个字符串。
 *
 * ## 目录约定
 *
 * 所有模型都放在同一个根目录下（`/data/local/tmp/snat-models/asr` 或
 * `filesDir/models/asr`），按子目录区分：
 *
 * ```
 * asr/
 *   model.int8.onnx  tokens.txt   ← SenseVoice（历史布局：直接放在根目录）
 *   silero_vad.onnx
 *   paraformer-zh/         model.int8.onnx  tokens.txt
 *   streaming-paraformer/  encoder.int8.onnx  decoder.int8.onnx  tokens.txt
 *   whisper-small/         small-encoder.int8.onnx  small-decoder.int8.onnx  small-tokens.txt
 *   punct-ct-transformer/  model.onnx  tokens.json
 * ```
 *
 * SenseVoice 之所以特殊（放根目录而不是 `sensevoice/`）：它是项目最早接进来的后端，
 * 真机上已经按老布局落好位了。为一个新加的可选项去搬已有模型，只会制造
 * 「某台机器上跑不起来」的差异。新布局一律走子目录。
 */
object AsrFactory {

    /** 后端标识。**会存进 DataStore**，改名等于清空用户设置，别动。 */
    const val SENSEVOICE = "sensevoice"
    const val PARAFORMER = "paraformer"
    const val STREAMING = "paraformer-stream"
    const val WHISPER = "whisper"

    /** 界面上可选的（标识 → 中文名）。顺序即界面顺序。 */
    val KINDS: List<Pair<String, String>> = listOf(
        SENSEVOICE to "SenseVoice（中英日韩·稳）",
        STREAMING to "Paraformer 流式（中英·快）",
        PARAFORMER to "Paraformer 离线（中英·准）",
        WHISPER to "Whisper Small（多语·慢）",
    )

    fun labelOf(kind: String): String = KINDS.firstOrNull { it.first == kind }?.second ?: kind

    fun normalize(kind: String?): String =
        if (KINDS.any { it.first == kind }) kind!! else SENSEVOICE

    // ------------------------------------------------------------ 目录解析

    /** SenseVoice 模型目录（含 model.int8.onnx + tokens.txt）。 */
    fun senseVoiceDir(root: File): File? =
        listOf(root, File(root, "sensevoice")).firstOrNull {
            File(it, SenseVoiceAsr.MODEL_FILE).isFile && File(it, SenseVoiceAsr.TOKENS_FILE).isFile
        }

    fun paraformerDir(root: File): File? =
        File(root, "paraformer-zh").takeIf {
            File(it, "model.int8.onnx").isFile && File(it, "tokens.txt").isFile
        }

    fun streamingDir(root: File): File? =
        File(root, "streaming-paraformer").takeIf {
            File(it, StreamingParaformerAsr.ENCODER_FILE).isFile &&
                File(it, StreamingParaformerAsr.DECODER_FILE).isFile &&
                File(it, StreamingParaformerAsr.TOKENS_FILE).isFile
        }

    fun whisperDir(root: File): File? =
        File(root, "whisper-small").takeIf {
            File(it, "small-encoder.int8.onnx").isFile &&
                File(it, "small-decoder.int8.onnx").isFile &&
                File(it, "small-tokens.txt").isFile
        }

    fun punctDir(root: File): File? =
        listOf(File(root, "punct-ct-transformer"), File(root, "punct"), root)
            .firstOrNull { Punctuator.resolve(it) != null }

    /** GTCRN 降噪模型（`gtcrn/gtcrn_simple.onnx` 或根目录直接放）。没装就是 null。 */
    fun denoiseFile(root: File): File? = Denoiser.resolve(File(root, "gtcrn"), root)

    /** 该后端在这台机器上装没装（界面据此把没装的选项标灰/回退）。 */
    fun installed(kind: String, root: File): Boolean = when (normalize(kind)) {
        SENSEVOICE -> senseVoiceDir(root) != null
        PARAFORMER -> paraformerDir(root) != null
        STREAMING -> streamingDir(root) != null
        WHISPER -> whisperDir(root) != null
        else -> false
    }

    // ------------------------------------------------------------ 根目录解析

    /**
     * 模型根目录的候选位置（按优先级）。
     *
     * 顺序必须和 `SenseVoiceAsr.resolve` 一致：显式 override → 应用私有目录 →
     * `/data/local/tmp`（只有 adb 调试才有这个目录，正式包读不到）。
     * 两处顺序一旦不同，就会出现「SenseVoice 找得到、Paraformer 找不到」这种怪事。
     */
    fun rootCandidates(filesDir: File?, override: String = ""): List<File> = buildList {
        if (override.isNotBlank()) add(File(override))
        filesDir?.let { add(File(it, "models/asr")) }
        add(File("/data/local/tmp/snat-models/asr"))
    }

    /**
     * 找到「至少装了一个后端」的模型根目录。
     *
     * 判定条件比 `SenseVoiceAsr.resolve` 宽：那里要求必须有 SenseVoice（当时只有它），
     * 这里只要有**任意**一个后端就行。否则用户只装了 Paraformer 的机器会被判成
     * 「模型都没装」，界面上明明有文件却报错，极难自查。
     */
    fun resolveRoot(filesDir: File?, override: String = ""): File? =
        rootCandidates(filesDir, override).firstOrNull { root -> KINDS.any { installed(it.first, root) } }

    /**
     * 找到「装了指定后端」的模型根目录。
     *
     * [resolveRoot] 只保证「这个目录里有**某个**后端」，用它来接引擎会踩一个很隐蔽的坑：
     * 私有目录里有 SenseVoice（老布局），`/data/local/tmp` 里有 Paraformer，
     * 用户把后端切到 Paraformer 后，`resolveRoot` 仍按顺序返回私有目录，
     * 于是报「模型不全」—— 而文件明明就在另一个候选根目录里。
     * 所以选后端之后必须按**这个后端**重新挑根目录。
     */
    fun resolveRootFor(kind: String, filesDir: File?, override: String = ""): File? =
        rootCandidates(filesDir, override).firstOrNull { installed(kind, it) }

    /** 报错时告诉用户该后端期望的子目录名。 */
    fun subdirOf(kind: String): String = when (normalize(kind)) {
        SENSEVOICE -> "根目录（model.int8.onnx + tokens.txt）"
        PARAFORMER -> "paraformer-zh"
        STREAMING -> "streaming-paraformer"
        WHISPER -> "whisper-small"
        else -> "?"
    }

    /** 找 silero VAD（根目录或其上级）。离线后端必需，流式后端用不到。 */
    fun resolveVad(root: File, override: String = ""): File? = buildList {
        if (override.isNotBlank()) add(File(override))
        add(File(root, SenseVoiceAsr.VAD_FILE))
        root.parentFile?.let { add(File(it, SenseVoiceAsr.VAD_FILE)) }
    }.firstOrNull { it.isFile }

    // ------------------------------------------------------------ 构造

    /**
     * 按标识造后端。
     *
     * @param asrRoot 模型根目录（见类注释的目录约定）
     * @param vadFile silero VAD（离线后端必需；流式后端不用）
     * @param threads ONNX 线程数
     * @param punctEnabled 是否给「不带标点的后端」补标点（SenseVoice 自带，不受影响）
     * @param whisperLang Whisper 的目标语言（sherpa 要求显式指定，不能 true-auto）
     * @param denoise 识别前是否用 GTCRN 降噪（只作用于送识别器的那一份，见 [Denoiser]）
     * @throws IllegalStateException 模型文件不全 —— 让调用方显示「没装」而不是悄悄换后端
     */
    fun create(
        kind: String,
        asrRoot: File,
        vadFile: File,
        threads: Int = 4,
        punctEnabled: Boolean = true,
        whisperLang: String = "zh",
        denoise: Boolean = false,
    ): AsrBackend = when (val k = normalize(kind)) {
        SENSEVOICE -> {
            val dir = senseVoiceDir(asrRoot) ?: error("SenseVoice 模型不完整：$asrRoot")
            OfflineVadAsr(
                displayName = "SenseVoice Small",
                modelDir = dir,
                vadFile = vadFile,
                numThreads = threads,
                modelConfig = { senseVoiceConfig(it, threads) },
                resultLang = { Lang.fromSenseVoiceTag(it.lang) },
                // SenseVoice 自带标点与逆文本归一化，再补一遍反而会加出多余标点
                punctuator = null,
                denoiser = makeDenoiser(asrRoot, denoise),
            )
        }

        PARAFORMER -> {
            val dir = paraformerDir(asrRoot) ?: error("Paraformer 离线模型不完整：$asrRoot/paraformer-zh")
            OfflineVadAsr(
                displayName = "Paraformer 离线（中英）",
                modelDir = dir,
                vadFile = vadFile,
                numThreads = threads,
                modelConfig = ::paraformerConfig,
                resultLang = { guessLangFromText(it.text) },
                punctuator = makePunctuator(asrRoot, punctEnabled),
                denoiser = makeDenoiser(asrRoot, denoise),
            )
        }

        STREAMING -> {
            val dir = streamingDir(asrRoot) ?: error("Paraformer 流式模型不完整：$asrRoot/streaming-paraformer")
            // 流式后端拿不到「整段」，降噪没有落点（要降噪得改成逐帧在线降噪），这里忽略该开关
            StreamingParaformerAsr(
                modelDir = dir,
                numThreads = threads.coerceAtMost(3),
                punctuator = makePunctuator(asrRoot, punctEnabled),
            )
        }

        WHISPER -> {
            val dir = whisperDir(asrRoot) ?: error("Whisper 模型不完整：$asrRoot/whisper-small")
            OfflineVadAsr(
                displayName = "Whisper Small",
                modelDir = dir,
                vadFile = vadFile,
                numThreads = threads,
                modelConfig = { whisperConfig(it, whisperLang) },
                resultLang = { Lang.fromCode(whisperLang) },
                punctuator = makePunctuator(asrRoot, punctEnabled),
                denoiser = makeDenoiser(asrRoot, denoise),
            )
        }

        else -> error("未知的 ASR 后端：$k")
    }

    private fun makeDenoiser(root: File, enabled: Boolean): Denoiser? {
        if (!enabled) return null
        val f = denoiseFile(root) ?: return null
        return Denoiser.open(f)
    }

    private fun makePunctuator(root: File, enabled: Boolean): Punctuator? {
        if (!enabled) return null
        val d = punctDir(root) ?: return null
        return runCatching { Punctuator(Punctuator.resolve(d)!!) }.getOrNull()
    }

    // ------------------------------------------------------------ 各模型的 OfflineModelConfig

    private fun base(modelDir: File, threads: Int, tokens: String): OfflineModelConfig {
        val mc = OfflineModelConfig()
        mc.tokens = File(modelDir, tokens).absolutePath
        mc.numThreads = threads
        mc.debug = false
        mc.provider = "cpu"
        return mc
    }

    /**
     * SenseVoice 的模型配置。公开是因为 [SenseVoiceAsr] 自己也用它 ——
     * 同一个模型在两处各写一份 `OfflineSenseVoiceModelConfig` 迟早会漂。
     */
    fun senseVoiceConfig(dir: File, threads: Int = 4): OfflineModelConfig {
        val sv = OfflineSenseVoiceModelConfig()
        sv.model = File(dir, SenseVoiceAsr.MODEL_FILE).absolutePath
        sv.language = "auto"
        sv.useInverseTextNormalization = true
        val mc = base(dir, threads, SenseVoiceAsr.TOKENS_FILE)
        mc.senseVoice = sv
        return mc
    }

    private fun paraformerConfig(dir: File): OfflineModelConfig {
        val pf = OfflineParaformerModelConfig()
        pf.model = File(dir, "model.int8.onnx").absolutePath
        val mc = base(dir, 4, "tokens.txt")
        mc.paraformer = pf
        return mc
    }

    private fun whisperConfig(dir: File, lang: String): OfflineModelConfig {
        val wh = OfflineWhisperModelConfig()
        wh.encoder = File(dir, "small-encoder.int8.onnx").absolutePath
        wh.decoder = File(dir, "small-decoder.int8.onnx").absolutePath
        wh.language = lang
        wh.task = "transcribe"
        // 逐词时间戳用不上，关掉省一点解码开销
        wh.enableTokenTimestamps = false
        wh.enableSegmentTimestamps = false
        val mc = base(dir, 4, "small-tokens.txt")
        mc.whisper = wh
        return mc
    }
}
