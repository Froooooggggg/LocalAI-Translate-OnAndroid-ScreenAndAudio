package com.screenast.app.translate

import com.screenast.app.core.Lang

/**
 * 判断「模型是不是根本没翻译，只是把原文改写了一遍」。
 *
 * ## 为什么需要这一层
 *
 * 2B 的失败形态不是空输出，而是**用原文语言改写**：
 *
 * ```
 * お元気にしていましたか。  →  元気にお過ごしでしょうか。
 * ありがとうございます！    →  ありがとうございます！
 * 안녕하세요, 잘 지내셨어요? →  안녕하세요, 잘 지내셨어요?
 * ```
 *
 * 这类输出如果被当成译文：
 * - 写进 `InterpretEngine.history`（上下文 > 0 时），下一句的 assistant 轮就是日文，
 *   模型会继续用日文答 —— 实测一句坏译文能把下一句本来能翻的 `明日の会議は…` 带成日文；
 * - 写进 `VisionSession.cache`，同一页/同一帧反复贴回同一段日文。
 *
 * 所以「定稿」必须过这一层再决定要不要记入上文/缓存。**只做保守判断**：
 * 宁可多判一次「像没翻」，也不要让原文语言的文本污染后续。
 */
object TranslationQuality {

    /**
     * @param source     源语言（ASR 判的 / 识图按块猜的）
     * @param target     目标语言（不能是 AUTO）
     * @param sourceText 原文
     * @param output     模型输出（已过 [com.screenast.app.translate.llama.LlamaOutput.clean]）
     */
    fun looksUntranslated(
        source: Lang,
        target: Lang,
        sourceText: String,
        output: String,
    ): Boolean {
        val s = output.trim()
        if (s.isEmpty()) return true
        if (source == Lang.AUTO || source == target) return false

        // 日文源、目标不是日文，输出里却还有假名 → 只是把日文改写了一遍。
        if (source == Lang.JA && target != Lang.JA && s.any { isKana(it) }) return true
        // 韩文源、目标不是韩文，输出里却还有谚文 → 同上。
        if (source == Lang.KO && target != Lang.KO && s.any { isHangul(it) }) return true

        // 中日/韩中场景：目标中文却一个汉字都没有（英文、罗马字、符号），
        // 基本可以确定没翻成中文。英文源不做这条：`OK`、`PDF` 这类是合法译文。
        if ((source == Lang.JA || source == Lang.KO) && target == Lang.ZH && s.none { isHan(it) }) {
            return true
        }

        // 原文照抄（忽略空白与标点）。比 `==` 宽松，能抓住只改空格/标点的复读。
        val a = normalize(sourceText)
        val b = normalize(s)
        return a.isNotEmpty() && a == b
    }

    /** 去掉空白与标点、统一小写；只用于「是不是同一句话」的比较。 */
    private fun normalize(s: String): String =
        s.filter { it.isLetterOrDigit() }.lowercase()

    private fun isKana(c: Char): Boolean =
        c.code in 0x3040..0x30FF || c.code in 0x31F0..0x31FF || c.code in 0xFF61..0xFF9F

    private fun isHangul(c: Char): Boolean =
        c.code in 0xAC00..0xD7AF || c.code in 0x1100..0x11FF || c.code in 0x3130..0x318F

    private fun isHan(c: Char): Boolean =
        c.code in 0x4E00..0x9FFF || c.code in 0x3400..0x4DBF || c.code in 0xF900..0xFAFF
}
