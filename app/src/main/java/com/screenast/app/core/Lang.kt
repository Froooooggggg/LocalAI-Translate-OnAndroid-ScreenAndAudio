package com.screenast.app.core

/**
 * 全局语言定义。ASR 与翻译共用同一套代码，避免两边各写一份枚举后不一致。
 *
 * [bcp47] 用于提示 ASR 引擎。
 * [englishName] 用于拼装发给翻译 API 的提示词（模型对英文语种名最敏感）。
 * [floresCode] / [nllbTokenId] 供端侧 NLLB 使用：NLLB 用 FLORES-200 语种码，
 *              且必须把该语种码的 token id 作为解码起始符（forced BOS token），
 *              否则模型不知道该翻译成哪种语言。
 *
 * ⚠️ [nllbTokenId] 曾全部写错（错到完全另一个 id 空间），会导致译文语种随机。
 *    当前值已用两条独立来源交叉校验一致：
 *      - `app/src/main/assets/nllb/tokenizer.bin`（运行时真正读的那份，权威）
 *      - `third_party/nllb/tokenizer.json` 的 added_tokens（207 个 special token 全对上）
 *    校验脚本：`tools/dump_tokenizer_bin_ids.py`（改词表后必须重跑）。
 */
enum class Lang(
    val code: String,
    val bcp47: String,
    val label: String,
    val englishName: String,
    val floresCode: String,
    val nllbTokenId: Int,
    /** SenseVoice 返回的语种标记（<|zh|> <|en|> <|ja|> <|ko|>）。 */
    val senseVoiceTag: String,
    /** 该语种是否有真正的流式 ASR 模型（决定首字延迟）。 */
    val hasStreamingAsr: Boolean,
) {
    AUTO("auto", "auto", "自动检测", "auto-detect", "", -1, "<|unknown|>", false),
    ZH("zh", "zh", "中文", "Chinese", "zho_Hans", 256200, "<|zh|>", true),
    EN("en", "en", "英文", "English", "eng_Latn", 256047, "<|en|>", true),
    JA("ja", "ja", "日文", "Japanese", "jpn_Jpan", 256079, "<|ja|>", false),
    KO("ko", "ko", "韩文", "Korean", "kor_Hang", 256098, "<|ko|>", true);

    companion object {
        /** 翻译的目标语言不能是 AUTO。 */
        val concrete: List<Lang> = listOf(ZH, EN, JA, KO)

        fun fromCode(code: String?): Lang =
            entries.firstOrNull { it.code.equals(code, ignoreCase = true) } ?: AUTO

        /** NLLB 的语种码 → 本枚举。FLORES-200 用下划线，这里统一转换。 */
        fun fromFlores(flores: String?): Lang =
            entries.firstOrNull { it.floresCode.equals(flores, ignoreCase = true) } ?: AUTO

        /** SenseVoice 语种标记 → 本枚举。 */
        fun fromSenseVoiceTag(tag: String?): Lang {
            if (tag.isNullOrBlank()) return AUTO
            val t = if (tag.startsWith("<|")) tag else "<|${tag.trim('<', '>', '|')}|>"
            return entries.firstOrNull { it.senseVoiceTag.equals(t, ignoreCase = true) } ?: AUTO
        }
    }
}
