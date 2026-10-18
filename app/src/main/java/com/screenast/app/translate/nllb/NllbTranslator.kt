package com.screenast.app.translate.nllb

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.util.Log
import com.screenast.app.core.Lang
import java.io.File
import java.nio.LongBuffer
import kotlin.math.exp

/**
 * 端侧 NLLB-200-distilled-600M 翻译器（ONNX Runtime Android）。
 *
 * 这个类是 `tools/verify_nllb_twostage.py` 的 Kotlin 移植，解码过程逐行对应，
 * 任何改动都要先改 Python 侧验证脚本、跑通、再同步过来 —— 单靠读代码看不出
 * 这类数值流程的错，必须让两端产出相同 token 序列才算数。
 *
 * ## 解码过程（两个反直觉的坑，都是实测出来的）
 *
 * **坑 1：`decoder_start_token_id` 是 `</s>`（id=2），不是 `<s>`(0)、也不是 `<pad>`(1)。**
 * 只有用 2 时，第 0 步 logits 才会落在「语言 token 族」上（top1 形如 `ron_Latn`、
 * `ell_Grek`），也就是这一步本质是「该输出哪种语言」的决策点。
 * 用 0 或 1 时 top1 恒为 `</s>`，模型当场收尾 —— 症状是**不崩、但一个字都不输出**，
 * 极难反查。穷举脚本：`tools/find_decoder_start.py`。
 *
 * **坑 2：语种是靠 forced BOS 注入的，而且第 0 步的 logits 要丢掉。**
 * 正确做法是「把目标语种 token 直接当成第一个已生成 token」；
 * 若改成把语种 token 当**输入**喂给 decoder，模型进入恒定退化态，
 * 每步都吐同一个 token（实测恒为 `248059 = '▁'`）。
 *
 * **坑 3（工程坑）：不能用 `decoder_model_merged_quantized.onnx`。**
 * merged 变体用 `use_cache_branch` 在「有/无 KV 缓存」两条路径间切换，
 * 首步必然触发 `Reshape node '/model/decoder/layers.0/encoder_attn/Reshape_4'`
 * 的 `The dimension with value zero exceeds the dimension size of the input tensor`；
 * 绕开它（给假长度 past + 走缓存分支）会让跨注意力真的对上零长度张量，输出是垃圾。
 * 独立 decoder 只有 3 个输入，没有这个坑。
 *
 * **坑 4：`decoder_with_past_model` 只输出 decoder 侧的 present（24 个），
 * 但需要 encoder 侧的 past 作为输入（48 个全部）。**
 * 所以首步那 24 个 encoder 侧 present 必须一直留着复用，不能丢。
 *
 * **坑 5（导出坑）：这份导出的 batch 被写死成 1，beam search 根本跑不了。**
 * `num_beams=4` 会在 `/model/decoder/layers.0/encoder_attn/Reshape` 上抛
 * `The input tensor cannot be reshaped to the requested shape`——
 * 报错里 `requested shape:{4,-1,16,64}` 的 4 就是写死的 batch。
 * 所以解码策略只能是贪心，想改行为只能从 logits 层面动手（见坑 6）。
 *
 * **坑 6：NLLB 会在从句边界「提前收尾」，长输入只翻前半截。**
 * 它是句级平行语料训出来的，看到一长串内容就认为一句说完了。实测 39 词的句子
 * 在第 23 步就吐 EOS，只翻了前半截。
 *
 * 关键在于**它停得心不心虚**：那一步 EOS 的概率只有 0.473，第二名「正」0.269——
 * 它其实还想接着写，只是被从句边界带偏了。
 *
 * 判据因此是**双边**的：`[EOS_GATE_FLOOR_DEFAULT, EOS_GATE_DEFAULT)`，
 * 即 p(EOS) 落在 **[0.30, 0.75)** 才压掉 EOS 重选一次，最多 [MAX_EOS_RESCUE_DEFAULT] 次。
 * 两边都实测过（`tools/probe_eos_risk.py`，11 个样例）：
 * ```
 *   0.062  输入 "I"（不是完整的话）    → 不该救，救了续出「我是个子.」
 *   0.473  长句 39 词（带逗号）        → 该救
 *   0.603  长句 39 词（无标点，=ASR）  → 该救
 *   0.83~0.92  其余 8 个正常样例       → 不该救
 * ```
 * 上界曾取 0.60，结果**无标点**那句（0.603）刚好卡在阈值之上不被救——
 * 而 ASR 输出的原文恰恰没有标点，真机上最容易踩的就是这个情况。
 *
 * ⚠️ 这个判据是启发式的，只能压住「模型想继续但不够自信」这一种情况。
 * 真正成熟的做法是训练/微调，或者换一个不按句对齐训的模型——那是另一件事。
 *
 * ## 线程模型
 *
 * 不是线程安全的：[translate] 会把模型状态（`encHidden`、KV 缓存）当成员变量用。
 * 调用方需保证串行，或用 [synchronized] 包住。
 */
class NllbTranslator(
    /**
     * 分词器。`private` 只挡住"外部改字段"，但**读**它是有正当用途的：
     * 调试台要拿同一份 tokenizer 跑自检（编码 → 解码往返），
     * 才能确诊"是分词错了还是模型生成错了"。故用 `internal val`。
     */
    internal val tokenizer: NllbTokenizer,
    encoderFile: File,
    decoderInitFile: File,
    decoderPastFile: File,
    private val threads: Int = 4,
) : AutoCloseable {

    private val env: OrtEnvironment = OrtEnvironment.getEnvironment()

    /**
     * [translate] 的互斥锁。
     *
     * 本类把一次翻译的中间状态（`encHidden` / `encMask` / `cache` 那 48 个 KV 张量）
     * 放在成员变量上，两个线程同时进来就会互相踩内存 —— 真机上表现为 ONNX Runtime
     * JNI 里的 SIGSEGV/SIGBUS，且故障地址是模型文件里的 ASCII 字符串。详见 [translate]。
     */
    private val lock = Any()

    private var encSession: OrtSession
    private val decInitSession: OrtSession
    private val decPastSession: OrtSession

    /** encoder 模型路径。卸载后重建会话要用（[OrtSession] 本身不记得自己从哪来）。 */
    private var encoderPath: String = ""
    private var encClosed: Boolean = false

    /** 首步产出的 48 个 present 名称，按固定顺序缓存，避免每步重新遍历。 */
    private val initPresentNames: List<String>
    private val pastInputNames: List<String>
    private val pastPresentNames: List<String>

    // ---- 一次翻译的中间状态（每次 translate 重写）----
    private var encHidden: TensorHandle? = null
    private var encMask: TensorHandle? = null

    /**
     * 当前 KV 缓存：past 输入名 → 张量句柄。
     *
     * 所有权约定：这个 map **拥有**里面每个句柄的生命周期。写入新一版缓存前，
     * 旧的一版必须整体关闭 —— 关闭时机不能提前（还要被本次 run 当输入），
     * 也不能拖后（泄漏 native 内存，一条长字幕下来就是几十 MB）。
     */
    private var cache: MutableMap<String, TensorHandle> = HashMap(64)

    /**
     * 本次翻译期间所有 [OrtSession.Result] 的归属地。
     *
     * ## 为什么必须攒到最后才释放
     *
     * `Result.close()` 释放的是它**持有的全部输出**，不是某一个张量。首步那次 run
     * 产出 48 个 present，我们按名字取 48 个句柄共享同一个 Result；只要还有任何一个
     * 句柄在用，这个 Result 就不能关 —— 否则会连带释放掉别的句柄正在用的张量。
     *
     * 之前按「一批句柄用完就关一批 Result」的写法，真机上表现为确定性的 SIGSEGV
     * （故障地址是一段 ASCII `values.`，来自模型文件）。详见 [TensorHandle]。
     *
     * 代价：一步不落地攒着，KV 张量（24 × [1,16,len,64]）在长句上会累积到几十 MB。
     * 对字幕场景（一句 10~30 token）完全可接受，且 [releaseRunState] 每句都会清空。
     */
    private val ownedResults = ArrayList<OrtSession.Result>(64)

    /**
     * encoder 会话的创建参数。抽出来是因为 encoder 可能被卸载后重建
     * （见 [unloadEncoder]），重建时必须用同一套参数，否则行为会漂。
     */
    private fun newSessionOptions(): OrtSession.SessionOptions =
        OrtSession.SessionOptions().apply {
            setIntraOpNumThreads(if (threads > 0) threads else 4)
            setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT)
        }

    // ------------------------------------------------------------------ 资源所有权

    /**
     * 一批输出张量，连同产出它们的 [OrtSession.Result]。
     *
     * ## 为什么必须把 Result 一起拿着（这是真机上 SIGBUS 的根因）
     *
     * `OrtSession.Result` 内部有一个 `ownedByResult[]` 标志数组，
     * `Result.close()` 的实现就是遍历它、为 true 的逐个 `OnnxValue.close()`。
     * 从 `run()` 拿到的普通输出张量**全部** ownedByResult=true，
     * 也就是说：
     *
     * ```kotlin
     * val t = r.get("last_hidden_state").get() as OnnxTensor
     * r.close()   // ← 这一行把 t 的 native 内存也释放了
     * ```
     *
     * 之后再用 `t` 就是悬垂指针。真机上的表现是最难查的那种：
     *
     * ```
     * Fatal signal 7 (SIGBUS), code 1 (BUS_ADRALN),
     * fault addr 0x646f6365642e3d in tid (DefaultDispatch)
     * ```
     *
     * 地址 `0x646f6365642e3d` 解出来正是 ASCII `"=decode"` ——
     * 读到的已经是别人的数据，而崩溃点在 ONNX Runtime 的线程池里，
     * 离真正写错的那一行隔了几十步。
     *
     * 所以规则很简单：**Result 和它产出的张量同生共死**。
     * 想留下哪个张量，就得连 Result 一起留到那个张量不再被使用为止。
     */
    /**
     * 一个从 [OrtSession.Result] 里取出来的输出张量句柄。
     *
     * ## 只关张量，**不**关 Result（这是真机 SIGSEGV/SIGBUS 的最终根因）
     *
     * `Result.close()` 不是"释放这一个张量"，而是**遍历它持有的全部输出逐个释放**。
     * 首步那次 run 产出 48 个 present（24 encoder 侧 + 24 decoder 侧），
     * 我按名字取了 48 个句柄、全部共享同一个 Result。于是：
     *
     * ```
     * step#1 产出 24 个新句柄 → closeCache() 关掉首步那 24 个 decoder 侧句柄
     *                        → 每个句柄都去 close 同一个 Result
     *                        → Result 一次性释放掉它持有的 48 个张量
     *                        → 其中 24 个 encoder 侧 KV 我们还要一直用！
     * ```
     *
     * 之后每步的 cross-attention 都在读已经释放的内存。真机症状：
     * ```
     * Fatal signal 11 (SIGSEGV), code 1 (SEGV_MAPERR),
     * fault addr 0x2e7365756c617667        ← 解出来是 ASCII "values."
     * ```
     * 故障地址是模型文件里的字符串 —— 读到的已经是别人的数据。
     * **注意这一步 `Result.close()` 会打一条 "Closing an already closed Result" 警告，
     * 但那是无害的重复调用；真正有害的是它第一次执行时就顺手带走了我们还要用的张量。**
     *
     * 所以规则是：句柄只管自己的张量；Result 统一由 [releaseRunState] 在所有句柄都
     * 不再需要之后释放。
     */
    private class TensorHandle(val t: OnnxTensor) : AutoCloseable {
        override fun close() {
            runCatching { t.close() }
        }
    }

    private class TensorResult(val value: EncOut, private val owner: OrtSession.Result?) : AutoCloseable {
        override fun close() {
            runCatching { owner?.close() }
        }
    }

    private class EncOut(val hidden: TensorHandle, val mask: TensorHandle)

    init {
        encoderPath = encoderFile.absolutePath
        val opts = newSessionOptions()
        encSession = env.createSession(encoderPath, opts)
        decInitSession = env.createSession(decoderInitFile.absolutePath, opts)
        decPastSession = env.createSession(decoderPastFile.absolutePath, opts)

        initPresentNames = decInitSession.outputInfo.keys
            .filter { it.startsWith("present.") }
            .sorted()
        pastInputNames = decPastSession.inputInfo.keys
            .filter { it.startsWith("past_key_values.") }
            .sorted()
        pastPresentNames = decPastSession.outputInfo.keys
            .filter { it.startsWith("present.") }
            .sorted()

        require(initPresentNames.size == 48) {
            "decoder(首步) 的 present 数量应为 48，实际 ${initPresentNames.size}"
        }
        require(pastInputNames.size == 48) {
            "decoder(增量) 的 past 输入数量应为 48，实际 ${pastInputNames.size}"
        }
        require(pastPresentNames.size == 24) {
            "decoder(增量) 的 present 数量应为 24，实际 ${pastPresentNames.size}"
        }

        // ---- 名字对账：这三份模型来自三个不同的 ONNX 导出，命名约定**不一致**，
        // 且差异只在真机上才暴露（PC 侧脚本用的是位置索引，从不依赖名字）。
        // 打印一次全景，比每次靠 "Missing Input: xxx" 反推快得多。
        Log.i(TAG, "首步 decoder 输出 present ${initPresentNames.size} 个")
        Log.i(TAG, "  样例: ${initPresentNames.take(3)} … ${initPresentNames.takeLast(1)}")
        Log.i(TAG, "增量 decoder 输入 past ${pastInputNames.size} 个")
        Log.i(TAG, "  样例: ${pastInputNames.take(3)} … ${pastInputNames.takeLast(1)}")
        Log.i(TAG, "增量 decoder 输出 present ${pastPresentNames.size} 个")
        Log.i(TAG, "  样例: ${pastPresentNames.take(3)} … ${pastPresentNames.takeLast(1)}")
        // 名字能不能对上：首步产出的 48 个 present 换成 past 名之后，
        // 是否**恰好**覆盖增量 decoder 要的 48 个输入名。
        val produced = initPresentNames.map { pastName(it) }.toSet()
        val wanted = pastInputNames.toSet()
        Log.i(TAG, "对账: 首步产出→past 名 ${produced.size} 个，增量要 ${wanted.size} 个，" +
            "交集 ${(produced intersect wanted).size} 个")
        val lack = wanted - produced
        val extra = produced - wanted
        if (lack.isNotEmpty()) Log.w(TAG, "增量要但首步不给的 ${lack.size} 个: ${lack.take(5)}")
        if (extra.isNotEmpty()) Log.w(TAG, "首步给但增量不要的 ${extra.size} 个: ${extra.take(5)}")
        Log.i(TAG, "首步 present 全名单: ${initPresentNames}")
        Log.i(TAG, "增量 past 全名单: ${pastInputNames}")
    }

    /** 一次翻译的结果与耗时。 */
    data class Result(
        val text: String,
        val tokenIds: IntArray,
        /** 感知到的首字延迟（ms）：编码 + 首步解码。 */
        val firstTokenMs: Long,
        /** 整段翻译总耗时（ms）。 */
        val totalMs: Long,
        /** 因连续重复而被折叠掉的 token 数。 */
        val collapsedTokens: Int,
    ) {
        override fun equals(other: Any?): Boolean =
            other is Result && text == other.text && firstTokenMs == other.firstTokenMs

        override fun hashCode(): Int = text.hashCode() * 31 + firstTokenMs.hashCode()
    }

    // ------------------------------------------------------------------ 主流程

    /**
     * 翻译一段文本。
     *
     * ## 为什么整个方法要加锁（这是真机上 SIGSEGV 的真正根因）
     *
     * 本类把一个句子的中间状态放在**成员变量**里：`encHidden` / `encMask` / `cache`
     * （48 个 KV 张量）。两个线程同时进来时，A 正在用缓存里的张量当输入，
     * B 已经把它们 close 掉换成新的一批 —— A 手里就是悬垂指针。
     *
     * 真机症状极具误导性：崩在 ONNX Runtime 的 JNI 里（`DefaultDispatch` 线程池），
     * 故障地址居然是一段 ASCII（`0x2e7365756c617667` 即 `values.`），而**同样的代码、
     * 同样的 onnxruntime 1.31.0、同样的模型文件，在 PC 上用 Python 永远复现不出来**
     * —— 因为 Python 是单线程串行调用。
     *
     * 真凶是靠日志时间戳抓到的：同一个 pid 下有两个 tid 在**交叠**执行
     * （`step#1` / `step#1` / `step#2` / `step#2`）。来源是 `am start` 让 Activity
     * 先 `onCreate` 再收 `onNewIntent`，auto 流程起了两遍；而防重入的标记写在
     * `remember` 里，Activity 重建后被重置了。
     *
     * 所以这里加锁是**必须的**，不是优化：调用方（UI、采集服务、未来的编排层）
     * 谁都无法保证串行。
     *
     * @param text 源文本（一句话，不是整篇）
     * @param from 源语言，不能是 [Lang.AUTO]（NLLB 需要明确的源语码）
     * @param to 目标语言，不能是 [Lang.AUTO]
     * @param maxNewTokens 最多生成多少 token
     * @param repeatPenalty 对已出现 token 的 logits 惩罚，1.0 = 关闭。
     *        NLLB 蒸馏版有复读倾向，实测 1.1 能压住但不影响正常重复用词。
     * @param eosGate 「提前收尾」判据的上界（见类注释坑 6）：argmax 撞上 EOS 时，
     *        p(EOS) 必须**低于**它才考虑救回。太自信的 EOS 就是真的说完了。
     *        取 1.0 等于不启用（任何 EOS 都放行）。
     * @param eosFloor 同一条判据的下界：p(EOS) 必须**高于**它才考虑救回。
     *        太低说明模型压根没主意（输入不是一句完整的话），硬续只会编。
     *        取 0.0 等于关掉这个保护。不满足 `eosFloor <= p(EOS) < eosGate` 一律放行 EOS。
     * @param maxEosRescue 一句最多救回几次。必须有限：这个判据是启发式的，
     *        放任它一路压下去会从「补齐从句」滑向「自由发挥」。
     * @param onProgress 每生成一个正文 token 调用一次，参数是**到目前为止的完整译文**
     *        （不是增量片段）。用途是实时字幕：NLLB 大约 75ms 出一个 token，
     *        整句 750ms，逐 token 显示能把「感觉到出字」的时间从 750ms 压到 80ms。
     *
     *        ⚠️ 三条约定，都踩过：
     *        1. **在持锁期间被调用**（本方法整体在 `synchronized(lock)` 里），
     *           回调里绝不能回头调用本 translator 的任何方法 —— 会死锁。
     *        2. **在调用方线程上被调用**（不是主线程），要碰 UI 必须自己切。
     *        3. 必须**立即返回**。它阻塞的是 native 解码循环。
     */
    fun translate(
        text: String,
        from: Lang,
        to: Lang,
        maxNewTokens: Int = 96,
        repeatPenalty: Float = 1.1f,
        eosGate: Float = EOS_GATE_DEFAULT,
        eosFloor: Float = EOS_GATE_FLOOR_DEFAULT,
        maxEosRescue: Int = MAX_EOS_RESCUE_DEFAULT,
        onProgress: ((String) -> Unit)? = null,
    ): Result = synchronized(lock) {
        require(from != Lang.AUTO) { "NLLB 需要明确的源语言，不能是 AUTO" }
        require(to != Lang.AUTO) { "NLLB 需要明确的目标语言，不能是 AUTO" }
        val src = text.trim()
        if (src.isEmpty()) return@synchronized Result("", IntArray(0), 0, 0, 0)

        val total0 = System.nanoTime()
        var ok = false
        try {
            // ---- 1. 编码源文本 ----
            val srcIds = tokenizer.encode(src, from.floresCode)
            val enc = encode(srcIds).value

            // ---- 2. 首步解码（产出第一段 KV 缓存 + 首个正文 token）----
            val t1 = System.nanoTime()
            val firstToken = firstStep(to.nllbTokenId, enc.hidden.t, enc.mask.t)
            val firstTokenMs = (System.nanoTime() - t1) / 1_000_000

            // ---- 3. 增量解码 ----
            val seq = ArrayList<Int>(maxNewTokens)
            var next = firstToken
            var collapsed = 0
            var rescues = 0
            while (seq.size < maxNewTokens) {
                if (tokenizer.isEos(next)) break
                // 连续重复折叠：同一个 token 紧挨着重复出现是蒸馏模型的已知噪声
                // （「今天天气」会解成「今天天天気」）。折叠发生在 token 层，
                // 不会误伤「very very good」这类跨 token 的正常重复。
                if (seq.isNotEmpty() && seq.last() == next) {
                    collapsed++
                } else {
                    seq.add(next)
                    // 逐 token 推译文。tokenizer.decode 是 O(seq)，而 seq ≤ 96，
                    // 相对一步 native 解码（~75ms）可以忽略，不值得为它做增量解码。
                    onProgress?.invoke(tokenizer.decode(seq.toIntArray()))
                }
                // 只有「提前收尾」才值得救：正常收尾时 EOS 的概率会明显高于阈值，
                // 那种情况下 pickToken 原样把 EOS 还回来，下一轮循环顶部就 break 了。
                val pick = step(next, enc.mask.t, repeatPenalty, seq, rescues < maxEosRescue, eosGate, eosFloor)
                if (pick.rescued) {
                    rescues++
                    // 手写三位小数而不是 "%.3f".format()：后者走默认 Locale，
                    // 在部分欧洲语言环境下小数点会变成逗号，日志不好搜。
                    val shown = (pick.eosProb * 1000f).toInt() / 1000f
                    Log.i(
                        TAG,
                        "提前收尾已救回第 $rescues/$maxEosRescue 次：p(EOS)=$shown" +
                            " ∈ [$eosFloor, $eosGate)，续写「${tokenizer.decode(intArrayOf(pick.token))}」",
                    )
                }
                next = pick.token
            }

            val ids = seq.toIntArray()
            val elapsed = (System.nanoTime() - total0) / 1_000_000
            ok = true
            return@synchronized Result(tokenizer.decode(ids), ids, firstTokenMs, elapsed, collapsed)
        } finally {
            // ⚠️ 必须放在 finally：上面任何一步抛异常，张量都不会泄漏。
            //
            // 这个释放点是**整条链路唯一正确的释放点**，原因是新的所有权模型：
            // encoder 的 hidden/mask 要一直活到增量解码结束（每一步都要当输入），
            // encoder 会话也因此必须陪着活到最后（ORT 里输出张量的底层内存由
            // session 分配并绑定在它上面，提前 close session 等于把它们连根拔掉）。
            releaseRunState()
            if (!ok) Log.w(TAG, "translate 异常退出，已释放全部张量")
        }
    }

    // ------------------------------------------------------------------ 内部

    /**
     * 跑 encoder，拿到 hidden_states。
     *
     * ⚠️ 这里**不能**用嵌套的 `use {}`：`return` 写在 `use` 的 lambda 里是非局部返回，
     * 会跳过内层所有 `finally`，于是「该留的张量被 close、该关的 Result 没关」。
     * 曾经写成 `ids.use { mask.use { session.run().use { ... return } } }`，
     * 结果 mask 被关掉后又在后面几十步里当输入用 —— 崩溃点离病因很远。
     * 现在改成顺序 try/finally，资源谁该活、谁该死一目了然。
     */
    private fun encode(srcIds: IntArray): TensorResult {
        // ⚠️ 输入张量一律用 `LongBuffer.allocate(...)`（**直接缓冲**）而不是
        // `LongBuffer.wrap(array)`（堆缓冲）。JNI 侧要靠 GetDirectBufferAddress 拿到
        // 数据指针，堆缓冲会拿到 NULL —— 曾经在真机上表现为 ORT 原生库里
        // SIGSEGV/SIGBUS，故障地址落在被释放的错误消息字符串上（`0x2e7365756c61767`
        // 解出来是 `values.`），完全看不出与「缓冲区类型」有关。
        val idsArr = LongArray(srcIds.size) { srcIds[it].toLong() }
        val ids = OnnxTensor.createTensor(
            env,
            LongBuffer.allocate(idsArr.size).put(idsArr).rewind() as LongBuffer,
            longArrayOf(1, srcIds.size.toLong()),
        )
        val maskArr = LongArray(srcIds.size) { 1L }
        val mask = OnnxTensor.createTensor(
            env,
            LongBuffer.allocate(maskArr.size).put(maskArr).rewind() as LongBuffer,
            longArrayOf(1, srcIds.size.toLong()),
        )
        var idsKept = false
        var maskKept = false
        try {
            val r = try {
                encSession.run(mapOf("input_ids" to ids, "attention_mask" to mask))
            } catch (e: Throwable) {
                // encoder 会话被卸载过（见 unloadEncoder）：重建后再来一次。
                // 这里捕获的是「already closed」这一类 AIOOBE/IllegalStateException，
                // 用它当信号比自己维护标志位更不容易漏。
                reloadEncoder()
                encSession.run(mapOf("input_ids" to ids, "attention_mask" to mask))
            }
            // ⚠️ 输出名是 `last_hidden_state`，**不是** `hidden_states`（那是 T5 的叫法）。
            // 踩过一次：Python 侧验证脚本一直写 `enc.run(None, ...)[0]` 按**位置**取，
            // 从不依赖名字，于是移植到 Kotlin 时名字写错也没被发现，
            // 直到真机上才炸："encoder 没有输出 hidden_states，实际输出 1 个"。
            // 教训：跨语言移植时，凡是 Python 侧用下标/位置取值的地方，
            // 都必须回头确认真实的名字。
            val h = r.get("last_hidden_state").orElseThrow {
                val names = r.map { it.key }.joinToString()
                r.close()
                IllegalStateException("encoder 没有输出 last_hidden_state，实际输出：$names")
            } as OnnxTensor
            idsKept = true

            // ⚠️ 这里**绝对不能** r.close()：hidden 的 native 内存归 Result 所有，
            // 而 Result.close() 会一次性释放它持有的**全部**输出（见 TensorHandle 的长注释）。
            // Result 交给 ownedResults 统一保管，所有句柄都用完之后才释放。
            val hh = TensorHandle(h)
            val mh = TensorHandle(mask)
            ownedResults.add(r)
            maskKept = true
            encHidden = hh
            encMask = mh
            return TensorResult(EncOut(hh, mh), null)
        } finally {
            if (!idsKept) runCatching { ids.close() }
            if (!maskKept) runCatching { mask.close() }
        }
    }

    /**
     * 首步：把 `</s>` 喂进不带 past 的 decoder，拿回 logits（丢弃）和 48 个 present。
     *
     * @return 第一个正文 token id（= 目标语种 token，由 forced BOS 注入）
     */
    private fun firstStep(tgtTokenId: Int, hidden: OnnxTensor, mask: OnnxTensor): Int {
        val startIds = OnnxTensor.createTensor(
            env,
            LongBuffer.wrap(longArrayOf(tokenizer.eosId.toLong())),
            longArrayOf(1, 1),
        )
        val present = HashMap<String, TensorHandle>(64)
        try {
            val r = decInitSession.run(
                mapOf(
                    "input_ids" to startIds,
                    "encoder_hidden_states" to hidden,
                    "encoder_attention_mask" to mask,
                )
            )
            try {
                // 逐个 get() 而不是遍历迭代器：get() 明确返回一个可管理的 OnnxTensor。
                // ⚠️ 但 get() 拿到的张量仍然归 Result 所有，**不能** r.close()
                // （见 TensorHandle 的长注释）。Result 交 ownedResults 统一保管。
                for (name in initPresentNames) {
                    present[name] = TensorHandle(
                        r.get(name).orElseThrow {
                            IllegalStateException("decoder(首步) 缺少输出 $name")
                        } as OnnxTensor,
                    )
                }
                ownedResults.add(r)
            } catch (e: Throwable) {
                // 出错路径没有句柄来接管这 48 个张量，只能就地整体释放
                runCatching { r.close() }
                throw e
            }
        } finally {
            startIds.close()
        }

        // 旧缓存整体释放（首次为空）。注意此刻 feed 已返回，
        // startIds/hidden/mask 都不受影响。
        closeCache()
        // present.X → past_key_values.X：只换名字，张量本体继续用
        for ((name, t) in present) cache[pastName(name)] = t
        // 第 0 步的 logits 按设计丢弃，语种靠 forced BOS 注入（见类注释坑 2）
        return tgtTokenId
    }

    /** 增量一步：喂 1 个 token + 全部 48 个 past，换回 1 个新 token。 */
    private fun step(
        lastToken: Int,
        mask: OnnxTensor,
        repeatPenalty: Float,
        generated: List<Int>,
        allowEosRescue: Boolean,
        eosGate: Float,
        eosFloor: Float,
    ): StepPick {
        val idT = OnnxTensor.createTensor(
            env,
            LongBuffer.wrap(longArrayOf(lastToken.toLong())),
            longArrayOf(1, 1),
        )
        val nextMap = HashMap<String, TensorHandle>(64)
        val oldHandles = ArrayList<TensorHandle>(64)
        val logits: FloatArray
        try {
            val feed = HashMap<String, OnnxTensor>(64)
            feed["input_ids"] = idT
            feed["encoder_attention_mask"] = mask
            for ((name, h) in cache) feed[name] = h.t
            Log.i(TAG, "step#${generated.size}: cache=${cache.size} feed=${feed.size} " +
                "encPast=${feed.containsKey("past_key_values.0.encoder.key")} " +
                "decPast=${feed.containsKey("past_key_values.0.decoder.key")}")
            val r = decPastSession.run(feed)
            try {
                for (name in pastPresentNames) {
                    nextMap[name] = TensorHandle(
                        r.get(name).orElseThrow {
                            IllegalStateException("decoder(增量) 缺少输出 $name")
                        } as OnnxTensor,
                    )
                }
                ownedResults.add(r)
                // ⚠️ logits 必须在这里就取成 float[]（getValue 是**深拷贝**）。
                // 若把 OnnxTensor 带出这个作用域，它就会一直挂在那个 Result 上，
                // 而 Result 要等到整个句子翻完才释放。
                val raw = (r.get("logits").orElseThrow {
                    IllegalStateException("decoder(增量) 没有产出 logits")
                } as OnnxTensor).value
                // 形状 [1, 1, V]
                @Suppress("UNCHECKED_CAST")
                logits = (((raw as Array<Any?>)[0] as Array<Any?>)[0]) as FloatArray
            } catch (e: Throwable) {
                for (h in nextMap.values) runCatching { h.close() }
                runCatching { r.close() }
                throw e
            }
        } finally {
            idT.close()
        }

        // ⚠️ 关键：**不能**用 nextMap 整个替换缓存。
        //
        // `decoder_with_past` 每一步只回吐 **decoder 侧**的 24 个 present
        // （`present.{i}.decoder.{key,value}`）—— encoder 侧的跨注意力 KV
        // 是首步算好后一直复用的，不随解码步变化，所以它压根不出现在输出里。
        //
        // 曾经写成 `closeCache()` 然后整体覆盖，等于把 48 个缓存砍成 24 个，
        // 从第二步起就报 `Missing Input: past_key_values.0.encoder.key`。
        // 正确的做法是**只替换 decoder 侧**，encoder 侧原样留着。
        for ((name, h) in nextMap) {
            val key = pastName(name)
            oldHandles.add(cache.remove(key) ?: continue)
            cache[key] = h
        }
        // 此刻 run 已返回、旧张量也确实不在 feed 里了，可以安全释放
        for (h in oldHandles) runCatching { h.close() }

        return pickToken(logits, repeatPenalty, generated, allowEosRescue, eosGate, eosFloor)
    }

    /**
     * 一步解码里「选哪个 token」的全部逻辑：重复惩罚 → argmax → 提前收尾救回。
     *
     * ⚠️ [logits] 会被**就地修改**（惩罚、以及压掉 EOS），调用方不要再拿它做别的事。
     */
    private fun pickToken(
        logits: FloatArray,
        penalty: Float,
        generated: List<Int>,
        allowEosRescue: Boolean,
        eosGate: Float,
        eosFloor: Float,
    ): StepPick {
        if (penalty > 1.0f && generated.isNotEmpty()) {
            // 只在序列较短时做惩罚：长序列上惩罚会累积成系统性偏置，
            // 反而让模型避开高频虚词。字幕场景 24 token 以内足够。
            for (id in generated) {
                if (id in logits.indices) {
                    logits[id] = if (logits[id] > 0f) logits[id] / penalty else logits[id] * penalty
                }
            }
        }
        var best = argmaxOf(logits)

        // EOS 没赢，或者这次不打算救 —— 没什么好说的，原样返回。
        if (best != tokenizer.eosId || !allowEosRescue || eosGate >= 1.0f) {
            return StepPick(best, 0f, false)
        }

        // EOS 赢了，但赢得心虚吗？它只拿到 40%~60% 的时候，第二名往往是
        // 「正」「这」这类要接着往下写的词 —— 模型其实没翻完，只是从句边界
        // 让它觉得「差不多可以收尾了」。这是 NLLB 句级平行语料的固有毛病，
        // 长输入尤其明显（实测 39 词的句子在第 23 步就停了，只翻了前半截）。
        //
        // 判据是**双边**的：既要低于 [eosGate]（太自信的 EOS 是真说完了），
        // 又要高于 [eosFloor]（太低概率说明模型压根没主意，硬续只会编）。
        // 中间这条带子才是「想接着说但不够自信」。见两个常量的注释。
        val p = softmaxAt(logits, tokenizer.eosId)
        if (p >= eosGate || p < eosFloor) return StepPick(best, p, false)

        // 压掉 EOS 再选一次。只在这一次前向的 logits 上动手，不改模型状态，
        // 也不动 KV 缓存 —— 张量生命周期那块被崩溃坑过一次，不碰。
        logits[tokenizer.eosId] = Float.NEGATIVE_INFINITY
        return StepPick(argmaxOf(logits), p, true)
    }

    private fun argmaxOf(logits: FloatArray): Int {
        var best = 0
        var bestV = logits[0]
        for (i in 1 until logits.size) {
            if (logits[i] > bestV) {
                bestV = logits[i]
                best = i
            }
        }
        return best
    }

    /**
     * 某个位置上的 softmax 概率。用 `max` 归一化是为了防溢出：
     * logits 之间差 100 以上时 `exp(v)` 会直接变 Inf。
     *
     * 代价是两遍 256k 次浮点运算（约 1ms），而它**只在 EOS 当上 argmax 时才被调用**，
     * 一句里最多几次，相对 75ms/步的解码可以忽略。
     */
    private fun softmaxAt(logits: FloatArray, index: Int): Float {
        var max = logits[0]
        for (i in 1 until logits.size) if (logits[i] > max) max = logits[i]
        val m = max.toDouble()
        var sum = 0.0
        for (v in logits) sum += exp(v - m)
        if (sum <= 0.0) return 0f
        return (exp(logits[index] - m) / sum).toFloat()
    }

    private fun pastName(presentName: String): String =
        "past_key_values." + presentName.removePrefix("present.")

    private fun closeCache() {
        for (t in cache.values) runCatching { t.close() }
        cache = HashMap(64)
    }

    private fun releaseRunState() {
        closeCache()
        encHidden?.let { runCatching { it.close() } }
        encHidden = null
        encMask = null
        // ⚠️ 顺序要紧：先关完所有张量句柄，再关 Result。
        // 反过来会把别的句柄还要用的张量一起带走（见 TensorHandle 的长注释）。
        for (r in ownedResults) runCatching { r.close() }
        ownedResults.clear()
    }

    /**
     * 卸载 encoder 会话，释放它占的 ~400MB。
     *
     * ⚠️ 只有在**确定这一句已经译完**时才能调用。ORT 里输出张量的底层内存由
     * session 分配并绑定在它身上，只要 [encHidden]/[encMask] 还活着，
     * 关掉 session 就等于把它们连根拔掉 —— 后续所有增量步都会读到悬垂内存
     * （真机上表现为 SIGBUS，且崩溃点在 ORT 线程池里，极难定位）。
     *
     * 当前**默认不调用**：省下的是常驻内存，代价是每换个句子就要重新加载
     * 396MB 模型（真机实测加载要若干秒），而且重新加载后旧张量全部作废。
     * 等精度和延迟都验完，再决定要不要为内存让路。
     */
    private fun unloadEncoder() {
        runCatching { encSession.close() }
        encClosed = true
    }

    private fun reloadEncoder() {
        val opts = newSessionOptions()
        encSession = env.createSession(encoderPath, opts)
        encClosed = false
    }

    override fun close() {
        releaseRunState()
        runCatching { encSession.close() }
        runCatching { decInitSession.close() }
        runCatching { decPastSession.close() }
    }

    /**
     * 一步解码选中的 token，附带「这次是不是靠压掉 EOS 才选出来的」。
     *
     * 用数据类而不是直接返回 Int，是因为救回这件事必须能被**观测**：
     * 它是一个启发式判据（[eosGate]），一旦阈值定得不合适，症状是「译文比原文多出
     * 半句自己编的话」——这种错误不报异常、不崩溃，只能靠日志里的救回记录发现。
     */
    private class StepPick(
        /** 最终选中的 token（救回时是把 EOS 压掉之后的第二名）。 */
        val token: Int,
        /** 救回发生时那一刻 EOS 的 softmax 概率；没有救回时为 0。 */
        val eosProb: Float,
        /** 这次是否通过压掉 EOS 换了别的 token。 */
        val rescued: Boolean,
    )

    companion object {
        private const val TAG = "NllbTranslator"

        /**
         * 「提前收尾」判据的**上界**：argmax 撞上 EOS 时，p(EOS) 必须**低于**它才考虑救回。
         * 太自信的 EOS 就是真的说完了。
         *
         * 0.75 是实测值（`tools/probe_eos_risk.py` 扫了 11 个样例）：
         * 正常译完的句子 p(EOS) 全部落在 **0.83~0.92**（含「Yes.」「谢谢」「好」这种短句），
         * 而该救的长句落在 **0.47~0.60**，中间这条 0.60→0.83 的空带就是阈值的安全区。
         *
         * ⚠️ 曾经取 0.60，结果是**无标点**的长句（0.603）刚好卡在阈值之上不被救——
         * 而 ASR 吐出来的原文恰恰没有标点，真机上最容易踩的就是这个情况。
         */
        const val EOS_GATE_DEFAULT = 0.75f

        /**
         * 「提前收尾」判据的**下界**：p(EOS) 必须**高于**它才考虑救回。
         *
         * 低概率的 EOS 是另一种失败模式：模型压根没主意，不是「还想接着说」。
         * 实测把输入截成单个 `I`（不是一句完整的话）时 p(EOS)=0.062，
         * 救回后把「我」续成了「我是个子.」——比不救更糟。
         *
         * 0.30 取在 0.062 与 0.473 之间，两边都留了足够余量。
         */
        const val EOS_GATE_FLOOR_DEFAULT = 0.30f

        /**
         * 一句最多救回几次。**必须有限**：这个判据只看单个位置的 logits，
         * 而「模型还想接着说」和「模型已经说完但不够自信」在单步上无法区分。
         * 放任它一路压下去，长输入会从「补齐从句」滑向「自由发挥」。
         *
         * 2 的来历：实测 39 词长句需要连续救回 2 次（0.473 → 0.502）才能拿全后半句；
         * 58 词超长句救回 1 次即收尾。
         */
        const val MAX_EOS_RESCUE_DEFAULT = 2

        /** 模型三件套是否齐备。缺任何一个都不该尝试加载（会抛底层异常，信息很难读）。 */
        fun missingFiles(dir: File, encoder: String, decoderInit: String, decoderPast: String): List<String> =
            listOf(encoder, decoderInit, decoderPast).filterNot { File(dir, it).isFile }

        fun describe(file: File): String =
            if (file.isFile) "${file.name} ${file.length() / 1048576}MB" else "${file.name} 缺失"
    }
}
