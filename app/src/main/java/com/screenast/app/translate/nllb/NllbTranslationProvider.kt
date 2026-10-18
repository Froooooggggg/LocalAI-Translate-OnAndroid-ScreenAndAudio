package com.screenast.app.translate.nllb

import android.content.Context
import com.screenast.app.core.Lang
import com.screenast.app.translate.ProviderStatus
import com.screenast.app.translate.TranslateChunk
import com.screenast.app.translate.TranslateRequest
import com.screenast.app.translate.TranslationException
import com.screenast.app.translate.TranslationProvider
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.channels.Channel
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.buffer
import kotlinx.coroutines.flow.channelFlow
import kotlinx.coroutines.withContext

/**
 * 把端侧 [NllbTranslator] 包装成 [TranslationProvider]。
 *
 * ## 为什么必须走这个接口，而不是让调用方直接 new 一个 NllbTranslator
 *
 * 两条，都是为后面留的路：
 *
 * 1. **实时同传要的是「流」不是「结果」**。`TranslationProvider.translate` 返回冷流，
 *    本类把 [NllbTranslator.translate] 的逐 token 回调转成 `TranslateChunk`。
 *    调用方（编排层）因此可以完全不知道底下是端侧还是云端 ——
 *    `OpenAiCompatibleProvider` 早就写好了，路线图 ⑤ 的「代理环境下云端兜底」
 *    只需要换一个工厂，编排层一行都不用改。
 *
 * 2. **首字延迟的验收标准是「第一个 chunk 到达的时刻」**，而 NLLB 大约 75ms 出一个
 *    token、整句 750ms。逐 token 推送能把「看到字出来」的时间从 750ms 压到 ~80ms，
 *    这是同传能不能用的关键。反过来如果 provider 攒完整句再返回，
 *    这个延迟就永远降不下来，而且降不下来的原因会被埋在调用方看不见的地方。
 *
 * ## 背压：用 CONFLATED，让「最新」永远赢
 *
 * 逐 token 会产生几十个 chunk，而 UI 可能正在重组。这里刻意用
 * [Channel.CONFLATED]（容量 1、新值覆盖旧值）：
 *  - 消费者慢的时候，中间的中间态直接丢掉，**最后一定会收到 final**；
 *  - `TranslateChunk.text` 本来就是「到目前为止的完整译文」，覆盖式语义天然适合丢弃中间态。
 *
 * 不要改成 `Channel.UNLIMITED`：那只是把「UI 追不上」换成「内存里堆几十个没人看的中间态」。
 */
class NllbTranslationProvider(
    private val translator: NllbTranslator,
    /** 模型来源描述，显示在界面上（排查「译文语种不对」时第一个要看的就是它）。 */
    private val source: String = "",
    /** 模型文件位置。只用于 [checkAvailability]，可以不传。 */
    private val files: NllbModels.Location? = null,
) : TranslationProvider, AutoCloseable {

    override val displayName: String =
        if (source.isBlank()) "端侧 NLLB-600M" else "端侧 NLLB-600M（$source）"

    // ─────────────────────────────────────────────────────────────────────────
    // 【试过并否决】构造时「预热一次」来消掉第一句的首字延迟 —— 不要再加回来
    //
    // 动机看着很有道理：实测里同一台机器上，进程内**第一句**的首字延迟是 465~565ms，
    // 之后每句只要 91~133ms，差的几百毫秒像是「第一次 run 的一次性开销」（ONNX 分配
    // 内存池、惰性初始化内核）。既然如此，用一句 `"Hello."`（maxNewTokens=8）在加载阶段
    // 先跑一遍，把这笔钱挪到已经写着「模型加载中…」的时候付，似乎稳赚。
    //
    // 真机实测结果：**三项全部变差**。
    //   加载耗时      3494ms  →  6752ms   （预热本身花了 2977ms，比预期贵一个数量级）
    //   第 1 句首字    565ms  →  1771ms   （不但没省下，反而慢了三倍）
    //   第 2 句首字    100ms  →   133ms   （热态基线基本不变，说明省下的不是「热态开销」）
    //
    // 为什么反而更慢，符合实测的解释是：ONNX 的 BFC 内存池是**按实际 shape 增长**的。
    // 用极短输入预热，等于先按小 shape 分配了一块池子；真正的句子进来时 shape 更大，
    // 触发的是「在老池子上扩容/搬移」，这条路比「从头分配一块够大的」更贵。而且预热那次
    // 运行本身又把常驻内存推高了一截 —— 本应用本来就有被系统以 critical_mem_pressure
    // 杀掉的病史（见 `docs/断点续接单.md`），在 1.27GB 的模型之上再加一块池子并不划算。
    //
    // 结论：**第一句多付的那几百毫秒就让它付**。要真正解决，方向是换更小的模型或
    // 让导出带上动态 batch，而不是在初始化阶段做手脚。
    // ─────────────────────────────────────────────────────────────────────────

    /**
     * ## 关于 `request.history`（对话上下文）
     *
     * **故意忽略**。NLLB 是句级 seq2seq 模型：它的输入是「一个句子 + 一个目标语种 token」，
     * 没有角色概念、也没有轮次分隔符。把上文拼进同一句只会变成
     * 「前面的话 + 这一句」被整体翻译 —— 实测多出重复和串行内容，比没有上下文更差。
     *
     * 真要上下文只能用 LLM 后端（`--es provider llama`）。界面里对应滑条也已写明这件事，
     * 免得用户以为调到 10 轮却没变化是 bug。
     */
    override fun translate(request: TranslateRequest): Flow<TranslateChunk> = channelFlow {
        val text = request.text.trim()
        if (text.isEmpty()) {
            send(TranslateChunk("", isFinal = true))
            return@channelFlow
        }

        // 端侧没有「自动检测」能力：NLLB 必须知道源语码才能选对编码器行为。
        // 这里抛异常而不是猜一个 —— 猜错的表现是「译文凭空变成另一种语言」，
        // 比明确报错难查得多。
        if (request.source == Lang.AUTO) {
            throw TranslationException("端侧 NLLB 需要明确的源语言（ASR 未判定出语种）")
        }
        if (request.target == Lang.AUTO) {
            throw TranslationException("端侧 NLLB 需要明确的目标语言")
        }

        // 切到 Default：接口约定是「collect 时发起请求」，调用方可能在主线程 collect，
        // 而端侧解码是**阻塞**的 native 调用（整句几百毫秒），留在主线程就是 ANR。
        val result = try {
            withContext(Dispatchers.Default) {
                translator.translate(
                    text = text,
                    from = request.source,
                    to = request.target,
                    onProgress = { partial ->
                        // trySend 在有 CONFLATED 缓冲时永不失败、永不挂起 ——
                        // 解码循环等不起一个会阻塞的发送。
                        trySend(TranslateChunk(partial, isFinal = false))
                    },
                )
            }
        } catch (t: Throwable) {
            // 不要把取消也包成 TranslationException：那会让上层的取消逻辑失灵。
            if (t is CancellationException) throw t
            throw TranslationException("端侧翻译失败：${t.message ?: t.javaClass.simpleName}", t)
        }

        send(TranslateChunk(result.text, isFinal = true))
    }.buffer(Channel.CONFLATED)

    /**
     * 自检**只查文件在不在，不跑一次真实翻译**。
     *
     * 理由：自检要能在界面上随手点，而真实翻译要先加载 1.27GB 模型（3.5 秒）再解码几百毫秒。
     * 「模型文件齐不齐」是自检唯一能提前回答、也最常出问题的那一项；
     * 真跑一次翻译是端到端验收的事，不该混进自检按钮。
     */
    override suspend fun checkAvailability(): ProviderStatus {
        val loc = files ?: return ProviderStatus.Ok("$displayName（未持有文件位置，跳过检查）")
        val missing = NllbTranslator.missingFiles(
            loc.dir, NllbModels.ENCODER, NllbModels.DECODER, NllbModels.DECODER_PAST,
        )
        return if (missing.isEmpty()) {
            ProviderStatus.Ok(loc.summary())
        } else {
            ProviderStatus.Error("模型文件缺失：${missing.joinToString()}")
        }
    }

    override fun close() {
        runCatching { translator.close() }
    }

    companion object {
        /**
         * 从设备上加载一个端侧 provider。
         *
         * ⚠️ **耗时约 3.5 秒、常驻内存 1.27GB 的模型**，绝不能在主线程调用。
         * 编排层是把它放在翻译线程上懒加载的（见 `InterpretEngine.ensureProvider`）。
         *
         * @param preferredDir 用户在设置里显式指定的目录，空串表示按默认顺序找
         */
        /**
         * 缺 `tokenizer.bin` 但有 `tokenizer.json` 时，本机编译一份放进模型目录。
         *
         * HF 上 NLLB 只发布 JSON，`tokenizer.bin` 是本项目自己的紧凑格式（见 [NllbTokenizerCompiler]）。
         * 以前只能回电脑跑 `tools/build_nllb_tokenizer.py`，现在下载完就能直接用。
         */
        private fun compileJsonIfNeeded(dir: java.io.File?): java.io.File? {
            if (dir == null) return null
            val bin = java.io.File(dir, "tokenizer.bin")
            if (bin.isFile && bin.length() > 1_000_000L) return bin
            val json = java.io.File(dir, "tokenizer.json")
            if (!json.isFile || json.length() < 1_000_000L) return null
            return runCatching {
                NllbTokenizerCompiler.compileToFile(json, bin)
                bin
            }.onFailure {
                android.util.Log.w("AST.Nllb", "词表编译失败：${it.message}")
            }.getOrNull()
        }

        fun load(
            context: Context,
            preferredDir: String = "",
            tokenizerAsset: String = "nllb/tokenizer.bin",
        ): NllbTranslationProvider {
            val loc = NllbModels.resolve(context, preferredDir)
                ?: throw TranslationException(
                    "找不到 NLLB 模型。需要 ${NllbModels.ENCODER} / ${NllbModels.DECODER} / " +
                        "${NllbModels.DECODER_PAST}，查找位置：\n" +
                        NllbModels.diagnostics(context, preferredDir).joinToString("\n")
                )
            if (!loc.complete) throw TranslationException("NLLB 模型文件不全：${loc.summary()}")
            // 词表优先找 tokenizer.bin；没有但有 tokenizer.json（软件内下载拿到的就是它）
            // 就在本机编译一份；最后才回退 assets（老安装包里可能还留着）。
            val tokFile = listOf(
                java.io.File(loc.dir, "tokenizer.bin"),
                java.io.File(loc.dir.parentFile ?: loc.dir, "tokenizer.bin"),
            ).firstOrNull { it.isFile && it.length() > 1_000_000L }
                ?: compileJsonIfNeeded(loc.dir)
            val tokenizer = if (tokFile != null) {
                NllbTokenizer.fromFile(tokFile)
            } else {
                runCatching { NllbTokenizer.fromAssets(context, tokenizerAsset) }.getOrElse {
                    throw TranslationException(
                        "找不到 NLLB 词表。要么放 tokenizer.bin，要么放 tokenizer.json" +
                            "（软件会在本机编成 tokenizer.bin），跟三个 onnx 同一个目录" +
                            "（models/nllb/）。",
                    )
                }
            }
            return NllbTranslationProvider(
                translator = NllbTranslator(tokenizer, loc.encoder, loc.decoder, loc.decoderPast),
                source = loc.source,
                files = loc,
            )
        }
    }
}
