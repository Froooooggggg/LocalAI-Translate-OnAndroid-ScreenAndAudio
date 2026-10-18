package com.screenast.app.asr

import android.util.Log
import com.k2fsa.sherpa.onnx.FeatureConfig
import com.k2fsa.sherpa.onnx.OfflineModelConfig
import com.k2fsa.sherpa.onnx.OfflineRecognizer
import com.k2fsa.sherpa.onnx.OfflineRecognizerConfig
import com.k2fsa.sherpa.onnx.OfflineRecognizerResult
import com.k2fsa.sherpa.onnx.SileroVadModelConfig
import com.k2fsa.sherpa.onnx.TenVadModelConfig
import com.k2fsa.sherpa.onnx.Vad
import com.k2fsa.sherpa.onnx.VadModelConfig
import com.screenast.app.asr.AsrBackend.Segment
import com.screenast.app.core.Lang
import java.io.File

/**
 * 「离线识别器 + VAD 分段」这个组合的通用实现。
 *
 * ## 为什么把这些粘在一起
 *
 * sherpa-onnx 的离线模型（SenseVoice / Paraformer / Whisper）用法完全一致：
 *
 * ```
 * 16kHz PCM ──► VAD ──► 端点 ──► 取整段 ──► 离线识别器 ──► 文本
 * ```
 *
 * 差别只有两处：**模型配置怎么填**（[modelConfig]）和**语种从哪来**（[resultLang]）。
 * 全项目的选型结论是「同一段音频、同一条 VAD、同一个切段策略，只换模型」才有可比性
 * （见 [AsrBackend]），所以这里把共用的部分收成一个类，换模型只换那两个 lambda。
 *
 * ## 语种
 *
 * SenseVoice 自带语种标记，直接从 `result.lang` 读；
 * Paraformer 的离线 API 不返回语种，只能用 [guessLangFromText] 猜；
 * Whisper 在 `OfflineWhisperModelConfig.language` 里**指定**了语种，直接复用即可。
 *
 * ## 标点
 *
 * Paraformer / Whisper 的输出通常不带标点，而字幕断句、翻译都对标点敏感
 * （NLLB / Qwen 在长句无标点时容易断错短语）。传 [punctuator] 就会在出句前
 * 走一遍 CT-Transformer 标点模型。**SenseVoice 不需要** —— 它自带逆文本归一化与标点。
 */
class OfflineVadAsr(
    override val displayName: String,
    modelDir: File,
    vadFile: File,
    private val numThreads: Int,
    /** 填空模型配置。参数是 [modelDir]，由调用方决定具体用哪个子模型。 */
    private val modelConfig: (File) -> OfflineModelConfig,
    /** 从识别结果里取语种。默认读 SenseVoice 的标记。 */
    private val resultLang: (OfflineRecognizerResult) -> Lang = { Lang.fromSenseVoiceTag(it.lang) },
    /** 可选的标点模型。null = 不加标点。 */
    private val punctuator: Punctuator? = null,
    minSilenceDuration: Float = 0.5f,
    maxSpeechDuration: Float = 8.0f,
    minSpeechDuration: Float = 0.25f,
    vadThreshold: Float = 0.5f,
    /**
     * 可选的 TEN-VAD 模型。给了就用 TEN-VAD 代替 silero。
     *
     * 为什么要试它：silero 在**带背景音乐**的语音上会把歌声判成非语音，
     * 导致整段音频进不了 VAD（字幕一条都不出）。TEN-VAD 训练集里音乐/噪声样本更多。
     * 但它对本项目是「待实测」状态，所以默认 null（仍用 silero）。
     */
    tenVadFile: File? = null,
    /**
     * 可选降噪器：只在「整段已切出、准备送识别器」时跑，不影响闸门与 VAD（见 [Denoiser]）。
     */
    private val denoiser: Denoiser? = null,
) : AsrBackend {

    private val recognizer: OfflineRecognizer
    private val vad: Vad

    init {
        val rc = OfflineRecognizerConfig()
        val fc = FeatureConfig()
        fc.sampleRate = SAMPLE_RATE
        fc.featureDim = 80
        rc.featConfig = fc
        rc.modelConfig = modelConfig(modelDir)
        rc.decodingMethod = "greedy_search"

        // ⚠️ assetManager 必须传 null：非 null 表示「模型在 APK assets 里」，
        // 此时 sherpa 会忽略绝对路径，读不到就在 native 侧 abort（没有异常、没有 tombstone）。
        // 详见 SenseVoiceAsr 里那段长注释。
        recognizer = OfflineRecognizer(null, rc)

        val vc = VadModelConfig()
        vc.sampleRate = SAMPLE_RATE
        vc.numThreads = 1
        vc.provider = "cpu"
        vc.debug = false
        if (tenVadFile != null) {
            val ten = TenVadModelConfig()
            ten.model = tenVadFile.absolutePath
            ten.threshold = vadThreshold
            ten.minSilenceDuration = minSilenceDuration
            ten.minSpeechDuration = minSpeechDuration
            ten.windowSize = WINDOW_SIZE
            ten.maxSpeechDuration = maxSpeechDuration
            vc.tenVadModelConfig = ten
        } else {
            val silero = SileroVadModelConfig()
            silero.model = vadFile.absolutePath
            silero.threshold = vadThreshold
            silero.minSilenceDuration = minSilenceDuration
            silero.minSpeechDuration = minSpeechDuration
            // 512 是本项目喂给 VAD 的窗口长度（16kHz 下 32ms），必须与 DictationEngine 一致
            silero.windowSize = WINDOW_SIZE
            silero.maxSpeechDuration = maxSpeechDuration
            vc.sileroVadModelConfig = silero
        }
        vad = Vad(null, vc)

        Log.i(TAG, "$displayName 就绪：$modelDir（VAD：${tenVadFile?.name ?: vadFile.name}）")
    }

    override fun accept(samples: FloatArray): List<Segment> {
        vad.acceptWaveform(samples)
        val out = ArrayList<Segment>(2)
        while (!vad.empty()) {
            val seg = vad.front()
            vad.pop()
            if (seg.samples.isEmpty()) continue
            val s = decodeToSegment(seg.samples) ?: continue
            out.add(s)
        }
        return out
    }

    override fun flush(): List<Segment> {
        vad.flush()
        val out = ArrayList<Segment>(2)
        while (!vad.empty()) {
            val seg = vad.front()
            vad.pop()
            if (seg.samples.isEmpty()) continue
            val s = decodeToSegment(seg.samples) ?: continue
            out.add(s)
        }
        return out
    }

    override fun decodeSegment(samples: FloatArray): Segment? {
        if (samples.isEmpty()) return null
        val s = try {
            decodeToSegment(samples)
        } catch (t: Throwable) {
            Log.e(TAG, "强制切段解码失败：${t.message}", t)
            return null
        }
        if (s == null) return null
        Log.i(TAG, "强制切段(${s.lang.code}) ${samples.size}点 ${s.text}")
        return s
    }

    override fun isSpeaking(): Boolean = vad.isSpeechDetected()

    override fun resetVad() {
        runCatching { vad.reset() }.onFailure { Log.w(TAG, "VAD reset 失败", it) }
    }

    override fun close() {
        runCatching { recognizer.release() }
        runCatching { vad.release() }
        runCatching { punctuator?.close() }
        runCatching { denoiser?.close() }
    }

    /** 解码 + 取语种 + 补标点。空文本返回 null。 */
    private fun decodeToSegment(samples: FloatArray): Segment? {
        // 降噪只作用在这一份（识别器输入），VAD 与闸门用的仍是原始音频。
        val input = denoiser?.denoise(samples) ?: samples
        val r = try {
            decode(input)
        } catch (t: Throwable) {
            Log.e(TAG, "解码失败：${t.message}", t)
            return null
        }
        var text = r.text.trim()
        if (text.isEmpty()) return null
        val lang = resultLang(r)
        val p = punctuator
        if (p != null) {
            text = runCatching { p.addPunctuation(text).trim() }.getOrDefault(text)
            if (text.isEmpty()) return null
        }
        Log.i(TAG, "识别(${lang.code}) ${samples.size}点 $text")
        return Segment(text, lang, samples.size)    }

    /**
     * ⚠️ 这里**不能**写成 `recognizer.createStream().use { ... }`。
     * `use` 是 `OfflineStream` 的**成员方法**，签名是 `use(Function1<OfflineStream, Unit>)` ——
     * 它把 lambda 的返回值丢弃，于是 `decode()` 的返回类型变成 Unit，
     * 调用处 `r.text` 全部报 Unresolved reference，报错位置离真正原因很远。
     */
    private fun decode(samples: FloatArray): OfflineRecognizerResult {
        val stream = recognizer.createStream()
        try {
            stream.acceptWaveform(samples, SAMPLE_RATE)
            recognizer.decode(stream)
            return recognizer.getResult(stream)
        } finally {
            stream.release()
        }
    }

    companion object {
        private const val TAG = "AST.ASR"

        /** VAD 与识别器统一用的采样率。 */
        const val SAMPLE_RATE = 16000

        /** 喂给 VAD 的窗口长度（sherpa 的 silero 包装要求 512）。 */
        const val WINDOW_SIZE = 512
    }
}
