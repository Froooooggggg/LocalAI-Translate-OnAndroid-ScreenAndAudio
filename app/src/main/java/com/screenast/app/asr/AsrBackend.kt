package com.screenast.app.asr

import com.screenast.app.core.Lang

/**
 * 语音识别后端的统一契约。
 *
 * ## 为什么要有这一层
 *
 * 项目最初只有 [SenseVoiceAsr] 一个后端，[DictationEngine] 直接 `new` 它，
 * 于是「换一个 ASR 模型」这件事被写死在了引擎的构造函数里。
 * 但现在需要 A/B 对比 SenseVoice / Paraformer / Whisper / 流式 Paraformer
 * 在**快速连读**上的准确率（见 docs/断点续接单.md 的选型记录），
 * 而对比的前提是**同一段音频、同一条 VAD、同一个切段策略**只换模型 ——
 * 如果每个后端各自一套引擎代码，比出来的数字就没有可比性。
 *
 * 所以把「后端」抽象成这个接口，[DictationEngine] 只跟接口打交道。
 *
 * ## 契约里哪些是硬要求
 *
 * - [accept] 与 [flush] 必须是**纯函数**式的「喂进去 → 吐出来」：不得跨调用缓存音频。
 *   端点是 [DictationEngine] 的职责（它要在静音时干脆不喂以省 CPU），
 *   后端只负责「VAD 说这段是语音，你把它变成字」。
 *   例外是流式后端（[isStreaming] = true）：它自带端点逻辑，此时
 *   [DictationEngine] 会把音频连续喂进去，不再自己切片。
 * - [decodeSegment] 必须能独立工作：强制切段那条路绕开 VAD，直接送一段音频过来。
 * - 返回 null 表示「这段没有可用结果」，调用方必须丢弃这段音频继续走 ——
 *   不能塞回缓冲重试，否则没词的音频会无限重试把采集线程拖死。
 *
 * ## 语种判定
 *
 * 不是每个后端都像 SenseVoice 那样自带语种标记（Paraformer 就没有）。
 * 做不到就返回 [Lang.AUTO]，由翻译侧用 [Lang.fromSenseVoiceTag] 之后的兜底逻辑处理。
 */
interface AsrBackend : AutoCloseable {

    /** 显示名，用于日志与界面（例如「SenseVoice Small」）。 */
    val displayName: String

    /**
     * 是否为流式后端。
     *
     * 流式后端（OnlineRecognizer 那套）自带端点检测，靠 [Lang.hasStreamingAsr] 能覆盖的
     * 只有中英双语；非流式后端依赖调用方喂进来的段落边界。
     */
    val isStreaming: Boolean get() = false

    /**
     * 喂入 16kHz 单声道 float 采样，返回**本次调用中**新完成的语音段。
     *
     * 非流式后端：内部 VAD 决定什么时候出段。
     * 流式后端：内部端点检测决定，且可以在 [accept] 里就地返回已定稿的句子。
     */
    fun accept(samples: FloatArray): List<Segment>

    /** 冲刷内部缓冲，把最后半句交出来（停止采集时调用）。 */
    fun flush(): List<Segment>

    /** 直接把一段音频送去解码，**不经过 VAD 分段**（供强制切段使用）。 */
    fun decodeSegment(samples: FloatArray): Segment?

    /** 当前是否正处于语音中（界面「正在说话」指示）。 */
    fun isSpeaking(): Boolean

    /** 丢掉内部的 VAD 缓冲与状态（强制切段后避免同一段音频再出一次）。 */
    fun resetVad()

    override fun close()

    /** 一段识别结果。 */
    data class Segment(
        val text: String,
        /** 后端判定的语种，判不出来为 [Lang.AUTO]。 */
        val lang: Lang,
        /** 这段语音有多少采样点（16kHz），用于在界面上标出长度。 */
        val samples: Int,
    )
}

/**
 * 只认识文字、不认识语种的后端（Paraformer / Whisper 的离线 API 都不给语种标记）
 * 用的启发式判定。
 *
 * 规则极简：含有 CJK 统一表意文字 → 中文；否则 → 英文。
 *
 * 为什么不做更聪明的判断：这条链路上语种只影响**翻译目标是不是等于源语言**
 * （相同就不翻译）以及 NLLB 的源语言提示。判错的代价是多翻一次或漏翻一次，
 * 而做成统计模型/词典的代价是每次出句都多花几毫秒、还引入新的失败点。
 * 日文/韩文在这里会落到「含 CJK → 中文」—— 这也是为什么**日韩场景仍然要用
 * SenseVoice**（它自带语种识别），Paraformer 这条路只承诺中英。
 */
internal fun guessLangFromText(text: String): Lang {
    var kana = 0
    var hangul = 0
    var han = 0
    var latin = 0
    for (ch in text) {
        when (ch.code) {
            // 半角片假名 0xFF61..0xFF9F、片假名语音扩展 0x31F0..0x31FF 都算假名：
            // 用户要求「只要有假名就一律翻译」，半角片假名漏掉会让整句落到「中文」。
            in 0x3040..0x30FF, in 0x31F0..0x31FF, in 0xFF61..0xFF9F -> kana++
            in 0xAC00..0xD7AF, in 0x1100..0x11FF, in 0x3130..0x318F -> hangul++
            in 0x4E00..0x9FFF, in 0xF900..0xFAFF -> han++
            else -> if (ch.isLetter()) latin++
        }
    }
    // ⚠️ 顺序很重要：**先看假名/谚文，再看汉字**。
    // 旧实现是「碰到第一个汉字就返回中文」，于是「今日はいい天気ですね」这种
    // 「以汉字开头、后面才是假名」的日语被判成中文 → 命中「源=目标就不翻译」
    // → 整句日语原样吐出（用户 2026-10-18：「日语有时候还是不翻译」的根因）。
    if (kana > 0) return Lang.JA
    if (hangul > 0) return Lang.KO
    if (han > 0) return Lang.ZH
    return if (latin > 0) Lang.EN else Lang.AUTO
}

/**
 * 收拾识别结果里的空白：中日文里**不该有空格**，但 SenseVoice 会给每个 token 之间塞一个。
 *
 * 实测日文出的是「元気 に して まし たか」「お 時間 です」——送进翻译后模型还能对付，
 * 但悬浮字幕上读起来就是「字被拆开了」；中文同理（「今天 天气 不错」）。
 * 韩语**不能删**：韩文本来就用空格分词（「반갑 습니다」虽然分词错了，但删掉空格会更糟）。
 *
 * 规则：只删「两侧都是汉字/假名」的 ASCII 空格，碰到拉丁字母、标点、韩文一律保留。
 * 不做全角/半角归一 —— 那是另一件事，混在一起改出了问题就分不清是谁的锅。
 */
internal fun tidyText(text: String, lang: Lang): String {
    if (lang != Lang.JA && lang != Lang.ZH) return text
    if (text.indexOf(' ') < 0) return text
    val sb = StringBuilder(text.length)
    for (i in text.indices) {
        val c = text[i]
        if (c == ' ' && i > 0 && i < text.lastIndex) {
            val a = text[i - 1]
            val b = text[i + 1]
            if (isCjkChar(a) && isCjkChar(b)) continue
        }
        sb.append(c)
    }
    return sb.toString()
}

/** 汉字 / 平假名 / 片假名（**不含**韩文谚文：韩文必须保留空格）。 */
internal fun isCjkChar(c: Char): Boolean =
    c.code in 0x4E00..0x9FFF || c.code in 0x3040..0x30FF || c.code in 0x3400..0x4DBF

