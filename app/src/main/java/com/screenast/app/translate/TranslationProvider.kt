package com.screenast.app.translate

import com.screenast.app.core.Lang
import com.screenast.app.core.Settings

/**
 * 一句已经翻完的对话（原文 + 译文）。
 *
 * ## 为什么上下文放在「请求」里，而不是让 provider 自己记
 *
 * provider 是**无状态租户**：它可能在两句之间被重建（换后端、换模型、翻译线程退出），
 * 一旦把历史存在 provider 里，重建就意味着上文丢失，而「什么时候丢的」从界面上完全看不出来。
 * 放在请求里，历史由唯一知道全部句子生命周期的地方（`InterpretEngine`）维护，
 * provider 每次拿到的都是一份完整、自洽的快照 —— 可复现、可打印、可对比。
 */
data class HistoryTurn(
    /** 上一句的原文（已识别、已合并的最终形态）。 */
    val source: String,
    /** 上一句的译文（**已净化**的定稿，绝不塞半成品）。 */
    val target: String,
)

/**
 * 一次翻译请求。
 *
 * [text] 是 ASR 给出的识别结果——注意它**可能是会被修正的中间结果**，
 * 因此 provider 必须假设同一句话会被反复送来重译。
 */
data class TranslateRequest(
    val text: String,
    val source: Lang,
    val target: Lang,
    /**
     * 用户手填的目标语种名（设置里「自定义」那一栏）。
     *
     * 非空时**优先于** [target]：枚举只有中英日韩，用户要翻泰语/越南语时没法靠枚举表达，
     * 直接把名字写进提示词更省事（用户 2026-10-18 的要求）。
     */
    val targetLabel: String = "",
    /** 用户手填的源语种名（设置里「源语言兜底」的自定义项），非空时写进提示词。 */
    val sourceLabel: String = "",
    /** 同一句话的第几次重译（0 = 首次）。用于让 provider 决定要不要省调用。 */
    val revision: Int = 0,
    /**
     * 之前若干轮对话，**按时间正序**（最后一个元素是最接近当前这句的）。
     *
     * 空表示无上下文。是否真的用、怎么用由 provider 决定：LLM 类后端会把它拼成
     * 多轮对话提示词，NLLB 这类句级模型会忽略（它没有对话概念，硬塞只会污染输入）。
     */
    val history: List<HistoryTurn> = emptyList(),
)

/**
 * 流式翻译结果。
 *
 * 首字延迟的验收标准就是「第一个 [TranslateChunk] 到达的时刻」，
 * 所以 provider 必须把服务端的增量文本立刻往下游推，不能攒完整句再返回。
 */
data class TranslateChunk(
    /** 到目前为止的完整译文（不是增量片段）——直接覆盖 UI 即可，避免拼接错乱。 */
    val text: String,
    /** true 表示这句话已经定稿，后续不会再变。 */
    val isFinal: Boolean,
)

/**
 * 翻译后端。实现类必须线程安全地支持被反复调用。
 *
 * 约定：
 *  - 失败时抛 [TranslationException]，不要静默返回空串（会造成"翻译不出来但看不出原因"）。
 *  - 不要吞掉取消异常（CancellationException 必须继续抛出）。
 *  - [close] 用来释放后端持有的资源（端侧 NLLB 是 1.27GB 的 ONNX 会话，云端的连接池也一样）。
 *    默认空实现是**故意的**：把关闭做成接口成员而不是让调用方去 `is AutoCloseable` 试探，
 *    是因为调用方（`InterpretEngine`）必须能无条件地关掉当前后端 ——
 *    「有的后端需要关、有的不需要」这件事不该泄漏到调用点上去。
 *    默认实现是空的，所以无资源的实现可以什么都不写。
 */
interface TranslationProvider : AutoCloseable {
    /** 人类可读的名字，用于界面展示。 */
    val displayName: String

    /**
     * 翻译并流式返回。
     * @return 冷流；collect 时发起请求，取消 collect 即中断请求。
     */
    fun translate(request: TranslateRequest): kotlinx.coroutines.flow.Flow<TranslateChunk>

    /** 用于界面上的连通性自检。 */
    suspend fun checkAvailability(): ProviderStatus

    /**
     * 释放后端资源。
     *
     * ⚠️ 调用方必须保证**此刻没有翻译在跑**：端侧后端关的是正在被读的 ONNX 会话，
     * 并发关闭就是「把脚下的地板拆掉」（本项目为此吃过一次 SIGSEGV）。
     * `InterpretEngine` 把关闭点放在翻译线程的 `finally` 里，正是为了这个保证。
     */
    override fun close() {}
}

sealed interface ProviderStatus {
    data class Ok(val detail: String = "") : ProviderStatus
    data class Error(val message: String) : ProviderStatus
    data object NotConfigured : ProviderStatus
}

class TranslationException(message: String, cause: Throwable? = null) : Exception(message, cause)

/** 由设置构造 provider 的工厂。 */
object ProviderFactory {
    fun create(settings: Settings): TranslationProvider = OpenAiCompatibleProvider(settings)
}
