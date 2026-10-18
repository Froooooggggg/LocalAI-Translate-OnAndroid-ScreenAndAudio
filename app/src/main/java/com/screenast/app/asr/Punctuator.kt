package com.screenast.app.asr

import android.util.Log
import com.k2fsa.sherpa.onnx.OfflinePunctuation
import com.k2fsa.sherpa.onnx.OfflinePunctuationConfig
import com.k2fsa.sherpa.onnx.OfflinePunctuationModelConfig
import java.io.File

/**
 * 标点恢复（CT-Transformer，中英混合）。
 *
 * ## 为什么需要它
 *
 * SenseVoice 自带标点，但 Paraformer / Whisper 的输出是**一串没有标点的字**。
 * 没有标点会同时伤害两件事：
 *
 * 1. **字幕可读性** —— 一整行没有停顿的汉字，观众要自己找断句。
 * 2. **翻译质量** —— NLLB / Qwen 都是按短语对齐的，输入没有标点时
 *    容易把「我说的不是这个意思」这种长句断错地方，译出来意思就歪了。
 *
 * ## 为什么单独一个类
 *
 * 它和识别器是**两个独立的 ONNX 模型、两条独立的生命周期**：
 * 换了 ASR 后端照样能用，ASR 不需要它也能跑。合成一个类会让「关掉标点」
 * 变成一件需要改动识别器构造参数的事，而现在只是传 null。
 *
 * ⚠️ `tokens.json` 必须和 `model.onnx` **放在同一个目录**，sherpa 不会去别处找。
 *    缺了它不会报错，而是所有输入原样返回（症状：加了标点开关却毫无变化）。
 */
class Punctuator(
    model: File,
    numThreads: Int = 2,
) : AutoCloseable {

    private val punctuation: OfflinePunctuation

    init {
        // 同 SenseVoiceAsr 的 init：不用 apply 作用域，避免 `model`/`provider` 这类
        // 与 Kotlin 标准库成员同形的名字被解析到外层。
        val mc = OfflinePunctuationModelConfig()
        mc.ctTransformer = model.absolutePath
        mc.numThreads = numThreads
        mc.debug = false
        mc.provider = "cpu"

        // OfflinePunctuationConfig 是 Java 类且**没有无参构造**（`model` 是必填项），
        // 所以不能用 `OfflinePunctuationConfig()` 再赋值字段。编译错误原文：
        // `No value passed for parameter 'model'`。
        val cfg = OfflinePunctuationConfig(mc)

        // assetManager 必须 null，理由见 SenseVoiceAsr 里那段「进程静默死亡」的注释
        punctuation = OfflinePunctuation(null, cfg)
        Log.i(TAG, "标点模型就绪：${model.absolutePath}")
    }

    /** 给一段无标点文本补标点。失败时返回原文 —— 标点不该成为出句的阻塞点。 */
    fun addPunctuation(text: String): String {
        if (text.isBlank()) return text
        return try {
            punctuation.addPunctuation(text)
        } catch (t: Throwable) {
            Log.w(TAG, "补标点失败，返回原文：${t.message}")
            text
        }
    }

    override fun close() {
        runCatching { punctuation.release() }
    }

    companion object {
        private const val TAG = "AST.Punct"

        /** 模型文件名（tokens.json 必须同目录）。 */
        const val MODEL_FILE = "model.onnx"
        const val TOKENS_FILE = "tokens.json"

        /** 在一组候选目录里找标点模型目录（含 model.onnx + tokens.json）。 */
        fun resolve(vararg dirs: File?): File? = dirs.filterNotNull()
            .firstOrNull { File(it, MODEL_FILE).isFile && File(it, TOKENS_FILE).isFile }
    }
}
