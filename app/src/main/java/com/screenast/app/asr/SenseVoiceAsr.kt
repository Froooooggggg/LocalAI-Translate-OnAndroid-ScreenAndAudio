package com.screenast.app.asr

import com.screenast.app.asr.AsrBackend.Segment
import java.io.File

/**
 * SenseVoice Small 语音识别（通过 sherpa-onnx）。
 *
 * ## 为什么是 SenseVoice
 *
 * 需求是中英日韩互译。这四个语种在 sherpa-onnx 的模型库里**没有任何单一流式模型**
 * 能覆盖（只有中英双语流式 Paraformer，日文压根没有流式版本），但 SenseVoice Small
 * 一个 228MB 的 int8 模型就同时覆盖 zh/en/ja/ko/yue，并且自带语种识别与标点。
 * 代价是它**不是流式模型**：必须等一段话说完才能解码。
 *
 * ## 实现已经搬进 [OfflineVadAsr]
 *
 * 「VAD 分段 + 离线解码」这套流程 SenseVoice / Paraformer / Whisper 三家完全一样，
 * 差别只有 `OfflineModelConfig` 怎么填、语种从哪来。为了让「同一段音频只换模型」
 * 的对比成立，共用部分收在 [OfflineVadAsr] 里，本类只负责
 * **保留原有构造签名 + 模型目录解析**，其余全部转调。
 *
 * 保留这个类而不是让调用方直接写 `AsrFactory.create(SENSEVOICE, ...)` 的原因：
 * 它的构造函数参数（VAD 阈值、最长句时长……）是调这整条链路时最常拧的几个旋钮，
 * 藏到字符串工厂后面会让调试变成「改一个字符串 → 重编译 → 猜」。
 *
 * ## 端点检测参数的取舍（这是延迟与准确率的唯一旋钮）
 *
 * - [minSilenceDuration] 调小 → 断句快、延迟低，但一句话会被切成碎片，
 *   翻译出来上下文不全；调大 → 句子完整，但字幕要等更久。
 * - [maxSpeechDuration] 是安全阀：万一一直没检测到静音（背景音乐不停），
 *   到点强制切段，否则字幕永远不出现。**有背景音乐的视频必须依赖这个**。
 *
 * ## 为什么构造函数里**没有** AssetManager 参数
 *
 * 因为模型一律从文件系统加载。sherpa-onnx 的 Java API 允许传 `AssetManager`，
 * 但那表示「模型在 APK 的 assets 里」，此时它会忽略绝对路径。
 * 两者互斥，而传错的代价是进程 `abort()`（不是异常），排查成本极高。
 * 与其留一个几乎必然被误用的参数，不如让它**根本不存在**。
 */
class SenseVoiceAsr(
    modelDir: File,
    numThreads: Int = 4,
    /**
     * silero VAD 模型文件。默认在 [modelDir] 里找 `silero_vad.onnx`；
     * 找不到时**构造期就抛** —— VAD 是这条链路的必需件，缺了以后
     * 症状会退化成「一直没字幕」，那种问题比启动即报错难查得多。
     */
    vadFile: File? = null,
    minSilenceDuration: Float = 0.5f,
    maxSpeechDuration: Float = 8.0f,
    minSpeechDuration: Float = 0.25f,
    vadThreshold: Float = 0.5f,
) : AsrBackend {

    /** 实际用到的 VAD 文件，构造后只读，便于日志/界面显示。 */
    val resolvedVad: File = vadFile
        ?: File(modelDir, VAD_FILE).takeIf { it.isFile }
        ?: modelDir.parentFile?.let { File(it, VAD_FILE) }?.takeIf { it.isFile }
        ?: throw IllegalStateException(
            "找不到 $VAD_FILE：既不在 $modelDir，也不在其上级目录。ASR 需要它做端点检测。"
        )

    private val delegate: AsrBackend = OfflineVadAsr(
        displayName = "SenseVoice Small",
        modelDir = modelDir,
        vadFile = resolvedVad,
        numThreads = numThreads,
        modelConfig = { AsrFactory.senseVoiceConfig(it, numThreads) },
        punctuator = null,
        minSilenceDuration = minSilenceDuration,
        maxSpeechDuration = maxSpeechDuration,
        minSpeechDuration = minSpeechDuration,
        vadThreshold = vadThreshold,
    )

    override val displayName: String get() = delegate.displayName

    override val isStreaming: Boolean get() = delegate.isStreaming

    override fun accept(samples: FloatArray): List<Segment> = delegate.accept(samples)

    override fun flush(): List<Segment> = delegate.flush()

    override fun decodeSegment(samples: FloatArray): Segment? = delegate.decodeSegment(samples)

    override fun isSpeaking(): Boolean = delegate.isSpeaking()

    override fun resetVad() = delegate.resetVad()

    override fun close() = delegate.close()

    companion object {
        const val SAMPLE_RATE = 16000
        const val WINDOW_SIZE = 512

        const val MODEL_FILE = "model.int8.onnx"
        const val TOKENS_FILE = "tokens.txt"
        const val VAD_FILE = "silero_vad.onnx"

        /**
         * 定位 SenseVoice 模型目录：目录里必须有 model.int8.onnx + tokens.txt，
         * 另外还要能找到 silero_vad.onnx（可以放在同一目录或上一级）。
         *
         * @param filesDirOverride 应用私有目录（`context.filesDir`），**必须传**。
         *   `resolve` 只有拿到它才能找到正式落位 `filesDir/models/asr`。
         *   曾经这里写死过 `/sdcard/Android/data/<pkg>/files/models/asr`——
         *   那条路径在 Android 15+ 上**应用自己都读不了**（见 docs/断点续接单.md §4.6），
         *   列在查找顺序里只会浪费时间并掩盖「模型没装上」这个真实原因，故已删除。
         *
         * 顺序：显式 override → 私有目录（正式版） → /data/local/tmp（只有 adb 调试才有）。
         */
        fun resolve(
            modelDirOverride: String,
            vadOverride: String,
            filesDirOverride: File? = null,
        ): Pair<File, File>? {
            val dirs = buildList {
                if (modelDirOverride.isNotBlank()) add(File(modelDirOverride))
                filesDirOverride?.let { add(File(it, "models/asr")) }
                add(File("/data/local/tmp/snat-models/asr"))
            }
            for (d in dirs) {
                // 根目录本身或者 sensevoice/ 子目录（新布局）
                val hit = AsrFactory.senseVoiceDir(d) ?: continue
                val vad = buildList {
                    if (vadOverride.isNotBlank()) add(File(vadOverride))
                    add(File(hit, VAD_FILE))
                    hit.parentFile?.let { add(File(it, VAD_FILE)) }
                }.firstOrNull { it.isFile } ?: continue
                return hit to vad
            }
            return null
        }
    }
}
