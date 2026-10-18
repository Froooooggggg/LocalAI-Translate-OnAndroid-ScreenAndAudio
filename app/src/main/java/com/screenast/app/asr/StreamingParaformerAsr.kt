package com.screenast.app.asr

import android.util.Log
import com.k2fsa.sherpa.onnx.EndpointConfig
import com.k2fsa.sherpa.onnx.EndpointRule
import com.k2fsa.sherpa.onnx.FeatureConfig
import com.k2fsa.sherpa.onnx.OnlineModelConfig
import com.k2fsa.sherpa.onnx.OnlineParaformerModelConfig
import com.k2fsa.sherpa.onnx.OnlineRecognizer
import com.k2fsa.sherpa.onnx.OnlineRecognizerConfig
import com.k2fsa.sherpa.onnx.OnlineStream
import com.screenast.app.asr.AsrBackend.Segment
import com.screenast.app.core.Lang
import java.io.File

/**
 * 流式 Paraformer（中英双语），真流式 ASR 后端。
 *
 * ## 为什么值得单独做一个后端
 *
 * 其他后端（[OfflineVadAsr]）都是「VAD 切段 + 离线解码」：
 * **必须等一段话说完**才能出字。快速连读的对话里，一句话可能十几秒不断，
 * 字幕就只能干等 —— 这正是用户抱怨的「高速人物对话识别跟不上」。
 *
 * 流式 Paraformer 每 32ms 就产出一个增量假设，端点由模型自己的
 * 端点检测（`isEndpoint`：尾静音 + 最长句）决定，所以：
 *
 * - 出字早 —— 不需要等整句；
 * - 断句在**词边界**上 —— 不会像能量切段那样切在字中间；
 * - 端点阈值可调（[minTrailingSilence]），比 silero VAD 的 0.5s 更可控。
 *
 * ## 代价（必须说清楚）
 *
 * 1. **只有中英**。这个模型没有日韩，所以设置里选它等于放弃日韩识别
 *    （[Lang.hasStreamingAsr] 里日文本来就是 false）。
 * 2. 没有语种标记 —— 只能靠 [guessLangFromText] 猜，中英混说时判定会抖。
 * 3. 输出不带标点 —— 用 [punctuator] 补。
 *
 * ## 这个类**有**跨调用状态（和离线后端相反）
 *
 * 流式解码天然是有状态的：`stream` 里存着声学上下文。所以：
 * - 只能由**一个线程**调用（本项目的约定是采集线程）；
 * - [resetVad] 会真的重置解码状态，不能当空实现；
 * - [accept] 里 `while (isReady) decode()` 必须循环到不再 ready，
 *   否则积压的 chunk 会让延迟越滚越大。
 */
class StreamingParaformerAsr(
    modelDir: File,
    private val numThreads: Int = 3,
    /** 尾静音多久算一句说完（秒）。太小会把句子切碎，太大则迟迟不出句。 */
    private val minTrailingSilence: Float = 0.45f,
    /** 最长一句多久强制断（秒）。防止滔滔不绝时永远不结束。 */
    private val maxUtterance: Float = 12f,
    /**
     * 一句话最短多久才允许端点切出（毫秒）。
     *
     * 实测发现光调 `minTrailingSilence` 不够：连续语音里每个词间的小停顿都会
     * 被模型的内部 VAD 当端点，于是每 0.6 秒吐一个 `the qu` / `quick bro` 这样的碎片。
     * 这里再加一道「攒够这么长音频才准切」，端点照样报，但先不认。
     */
    private val minSegmentMs: Int = 1200,
    private val punctuator: Punctuator? = null,
) : AsrBackend {

    /** [minSegmentMs] 对应的采样点数。 */
    private val minSegSamples: Int get() = SAMPLE_RATE * minSegmentMs / 1000

    override val displayName: String get() = "Paraformer 流式（中英）"

    override val isStreaming: Boolean get() = true

    private val recognizer: OnlineRecognizer
    private val stream: OnlineStream

    /** 自上次 reset 以来喂进去多少采样点（用于标出「这段多长」）。 */
    private var fedSinceReset = 0

    /** 上一次解码出的非空文本，仅用于「正在说话」指示。 */
    private var lastText = ""

    init {
        val para = OnlineParaformerModelConfig()
        para.encoder = File(modelDir, ENCODER_FILE).absolutePath
        para.decoder = File(modelDir, DECODER_FILE).absolutePath

        val mc = OnlineModelConfig()
        mc.paraformer = para
        mc.tokens = File(modelDir, TOKENS_FILE).absolutePath
        mc.numThreads = numThreads
        mc.debug = false
        mc.provider = "cpu"

        val fc = FeatureConfig()
        fc.sampleRate = SAMPLE_RATE
        fc.featureDim = 80

        // 端点规则：rule1 = 还没识别出内容时的尾静音，rule2 = 已识别出内容后的尾静音，
        // rule3 = 单句最长时长。sherpa 的默认值（2.4s / 1.2s / 20s）对字幕来说太钝了。
        //
        // EndpointRule 是 Java 类且**没有无参构造**，参数顺序是
        // (mustContainNonSilence, minTrailingSilence, minUtteranceLength)。
        // 写成 `EndpointRule()` 再赋字段编不过：`No value passed for parameter ...`。
        val r1 = EndpointRule(false, minTrailingSilence * 2f, 0f)
        val r2 = EndpointRule(false, minTrailingSilence, 0f)
        // rule3 是「最长一句」的安全阀：流式模型不会因为音乐不停就卡住，但会一直不吐句，
        // 必须有这个上限，否则歌曲场景又回到「半天不出」。mustContainNonSilence=false
        // 是因为强制切段时这一句里可能真的没有人声（纯音乐），照样要切出来。
        val r3 = EndpointRule(false, 0f, maxUtterance)

        val ec = EndpointConfig()
        ec.rule1 = r1
        ec.rule2 = r2
        ec.rule3 = r3

        val rc = OnlineRecognizerConfig()
        rc.featConfig = fc
        rc.modelConfig = mc
        rc.decodingMethod = "greedy_search"
        rc.enableEndpoint = true
        rc.endpointConfig = ec

        // assetManager 必须 null（模型在文件系统上），否则 native 侧直接 abort
        recognizer = OnlineRecognizer(null, rc)
        stream = recognizer.createStream()
        Log.i(TAG, "$displayName 就绪：$modelDir")
    }

    override fun accept(samples: FloatArray): List<Segment> {
        stream.acceptWaveform(samples, SAMPLE_RATE)
        fedSinceReset += samples.size
        // 必须循环到不再 ready：一次喂 32ms，但解码器可能还欠着更早的 chunk
        while (recognizer.isReady(stream)) recognizer.decode(stream)

        val text = recognizer.getResult(stream).text.trim()
        lastText = text
        if (text.isEmpty()) return emptyList()
        // 还没到端点 → 只是中间假设，这一层不吐出去（增量显示是引擎层的事）
        if (!recognizer.isEndpoint(stream)) return emptyList()
        // 端点说「这句完了」，但只有 0.6 秒 —— 那是词间停顿，不是句子边界。
        // 先不切，让文本继续累积；攒够 minSegmentMs 之后的下一个端点才算数。
        if (fedSinceReset < minSegSamples) return emptyList()

        val seg = finishSegment(text)
        recognizer.reset(stream)
        fedSinceReset = 0
        lastText = ""
        return listOfNotNull(seg)
    }

    override fun flush(): List<Segment> {
        // 喂 0.3s 静音把尾巴逼出来，否则最后半句会丢
        val pad = FloatArray(SAMPLE_RATE * 300 / 1000)
        stream.acceptWaveform(pad, SAMPLE_RATE)
        while (recognizer.isReady(stream)) recognizer.decode(stream)
        val text = recognizer.getResult(stream).text.trim()
        val seg = if (text.isEmpty()) null else finishSegment(text)
        recognizer.reset(stream)
        fedSinceReset = 0
        lastText = ""
        return listOfNotNull(seg)
    }

    /**
     * 流式模型本来只能顺着时间轴解，但「强制切段」那条路要的是
     * 「把这一坨音频变成字」。做法是开一条临时 stream、喂进去再喂尾巴静音，
     * 解完立刻释放 —— 不复用主 stream，免得把正在解的那句的上下文搞乱。
     */
    override fun decodeSegment(samples: FloatArray): Segment? {
        if (samples.isEmpty()) return null
        val tmp = recognizer.createStream()
        try {
            tmp.acceptWaveform(samples, SAMPLE_RATE)
            val pad = FloatArray(SAMPLE_RATE * 300 / 1000)
            tmp.acceptWaveform(pad, SAMPLE_RATE)
            while (recognizer.isReady(tmp)) recognizer.decode(tmp)
            val text = recognizer.getResult(tmp).text.trim()
            if (text.isEmpty()) return null
            return finishSegment(text, samples.size)
        } catch (t: Throwable) {
            Log.e(TAG, "流式解码失败：${t.message}", t)
            return null
        } finally {
            runCatching { tmp.release() }
        }
    }

    override fun isSpeaking(): Boolean = lastText.isNotEmpty()

    override fun resetVad() {
        runCatching { recognizer.reset(stream) }
        fedSinceReset = 0
        lastText = ""
    }

    override fun close() {
        runCatching { stream.release() }
        runCatching { recognizer.release() }
        runCatching { punctuator?.close() }
    }

    private fun finishSegment(text: String, samplesOverride: Int = 0): Segment? {
        var out = text
        val p = punctuator
        if (p != null) {
            out = runCatching { p.addPunctuation(out).trim() }.getOrDefault(out)
        }
        if (out.isEmpty()) return null
        val n = if (samplesOverride > 0) samplesOverride else fedSinceReset
        Log.i(TAG, "流式出句 ${n}点 $out")
        return Segment(out, guessLangFromText(out), n)
    }

    companion object {
        private const val TAG = "AST.ASR"

        const val SAMPLE_RATE = 16000
        const val ENCODER_FILE = "encoder.int8.onnx"
        const val DECODER_FILE = "decoder.int8.onnx"
        const val TOKENS_FILE = "tokens.txt"
    }
}
