package com.screenast.app.translate.llama

/**
 * 2B 模型的输出净化器。
 *
 * ## 为什么必须有这一层
 *
 * Qwen3.5-2B 是**思考型模型**。我们用手拼提示词里的空 `<think>\n\n</think>\n\n`
 * 让它直接出译文（见 [LlamaPrompt]），绝大多数时候有效。但在 40 轮连续推理的压测里，
 * 它**随机**（约 40 轮 1~3 次）会掉回思考模式。实测到的两种形态：
 *
 * ```
 * 形态一（重开思考块）
 *   计算中的一个失误让整个实验功亏一篑。
 *   </think>
 *
 *   计算中的一个失误让整个实验功亏一篑。
 *
 *   Wait, let me correct ...
 *   → 第一个 </think> 之前的半句是【正确译文】，切掉后面即可救
 *
 * 形态二（改口给备选）
 *   他每天都去公园喂鸽子。
 *
 *   或者更自然：
 *   他每天都去公园喂鸽子。
 *
 *   或者更简洁：...
 *   → 没有 </think>，要在第一个空行处截断
 * ```
 *
 * 两种形态的共同点：**正确译文永远在最前面**。因此「截断」是安全的救回手段，
 * 不需要重试、不需要检测概率。
 *
 * ## 为什么在流式过程中也能用
 *
 * 流式字幕的语义是「[com.screenast.app.translate.TranslateChunk.text] 是到目前为止的
 * **完整**译文」，UI 直接覆盖。所以净化器可以对**不断增长的缓冲区**反复跑：
 * 一旦触发截断，后续追加的内容全部被丢弃，已经发出去的长文本会被更短的覆盖掉 ——
 * 而覆盖正是这个接口的约定，不会出现「删不掉」的问题。
 *
 * ## 已知取舍
 *
 * 在第一个空行截断，意味着**模型如果真的输出了多段译文，只会保留第一段**。
 * 对字幕场景这是对的：输入永远是 ASR 切出来的**一句话**，一段以上的输出必然是退化。
 * 如果以后要拿这个 provider 翻整篇文档，必须把 [clean] 的 `cutAtBlankLine` 关掉。
 */
object LlamaOutput {

    /** 净化结果。[truncated] 为 true 说明触发了退化截断，值得记一条日志。 */
    data class Cleaned(
        val text: String,
        val truncated: Boolean = false,
        /** 触发截断的原因，用于日志；`null` 表示原文干净。 */
        val reason: String? = null,
    )

    /**
     * 净化模型原始输出。
     *
     * @param cutAtBlankLine 是否在第一个空行处截断（字幕场景应为 true）。
     */
    fun clean(raw: String, cutAtBlankLine: Boolean = true): Cleaned {
        var s = raw.trim()
        if (s.isEmpty()) return Cleaned("")

        var reason: String? = null

        // ① 开头就是 <think>：说明空 think 块那招失效了，模型真的在思考。
        //    答案在第一个 </think> 之后 —— 把它切出来，前面的思考过程全丢。
        if (s.startsWith(TAG_THINK_OPEN)) {
            val close = s.indexOf(TAG_THINK_CLOSE)
            if (close >= 0) {
                s = s.substring(close + TAG_THINK_CLOSE.length).trim()
                reason = "开头有思考块，已丢弃"
            }
        }

        // ② 正文里出现 </think>：形态一。它前面就是正确译文。
        val closeIdx = s.indexOf(TAG_THINK_CLOSE)
        if (closeIdx > 0) {
            s = s.substring(0, closeIdx).trim()
            reason = "遇到 </think>，已截断"
        }

        // ③ 第一个空行之后还有内容：形态二（或思考块尾巴）。
        if (cutAtBlankLine) {
            val blank = indexOfBlankLine(s)
            if (blank >= 0) {
                s = s.substring(0, blank).trim()
                reason = reason ?: "遇到空行，已截断"
            }
        }

        // ④ 残留的控制 token 字面量。正常情况下 eos 会先把它停掉，
        //    但退化输出里出现过模型自己写 <|im_end|> 再接下去的情况。
        s = s.replace(TAG_IM_END, "")
            .replace(TAG_THINK_OPEN, "")
            .replace(TAG_THINK_CLOSE, "")
            .replace(TAG_EOS, "")
            .trim()

        return Cleaned(s, truncated = reason != null, reason = reason)
    }

    /**
     * 找第一个「空行」的起始下标。
     *
     * 判据：连续两个换行，且**中间只有空格/制表符**（模型有时输出 `\n \n`）。
     * 从下标 1 开始找，避免把开头的换行算成空行。
     */
    private fun indexOfBlankLine(s: String): Int {
        var i = 1
        while (i < s.length) {
            if (s[i] == '\n') {
                // 回头看这一行是不是空的（只有空白字符）
                var j = i - 1
                while (j >= 0 && (s[j] == ' ' || s[j] == '\t')) j--
                if (j < 0 || s[j] == '\n') {
                    // 返回「上一行内容结束」的位置
                    return (j + 1).coerceAtLeast(0)
                }
            }
            i++
        }
        return -1
    }

    /** 与 [LlamaPrompt] 里的特殊 token 对应的字面量（tokenize 时 parseSpecial=true 会认）。 */
    private const val TAG_IM_END = "<|im_end|>"
    private const val TAG_THINK_OPEN = "<think>"
    private const val TAG_THINK_CLOSE = "</think>"
    private const val TAG_EOS = "</s>"
}
