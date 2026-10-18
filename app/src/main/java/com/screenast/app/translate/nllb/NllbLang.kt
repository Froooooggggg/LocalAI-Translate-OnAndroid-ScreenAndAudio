package com.screenast.app.translate.nllb

import org.json.JSONObject

/**
 * NLLB 的 FLORES-200 语种码工具。
 *
 * **为什么这里没有 `enum NllbLang`**：曾经有过一个，里面对着 [com.screenast.app.core.Lang]
 * 硬编码 `ru -> rus_Cyrl` 的映射，和 `Lang` 枚举本身的 `floresCode` 字段完全重复；
 * 而且 `Lang.KO` 加入后必然要再改一处，属于典型的双份真相。
 * 现在语种码只有一个来源：`Lang.floresCode`。这个文件只保留一个真正有用的能力——
 * 从 tokenizer.json 里抽 special token id（用于校验/工具链，运行时不依赖 json）。
 *
 * 做翻译时的固定范式（由 [NllbTokenizer.encode] 与调用方分别保证）：
 *  - 源语言：把 token **前置到输入序列**（encoder 的第一个 token）
 *  - 目标语言：作为 decoder 的 `forced_bos_token_id`（第 0 步强制生成它）
 *
 * 两处 id 都必须从词表里查，**不能硬编码**——那份 200 语言的顺序一旦换模型就全变了。
 */
object NllbLangCodes {

    /**
     * 从 tokenizer.json 里解出「特殊 token 字符串 → id」。
     *
     * tokenizer.json 的 added_tokens 是快照格式：
     * `{"id": 256047, "content": "zho_Hans", "special": true}`
     * 我们只取 special=true 的那些，避免把普通词也当成语言码。
     */
    fun specialTokenIdsFromTokenizerJson(json: String): Map<String, Int> {
        val root = JSONObject(json)
        val added = root.optJSONArray("added_tokens") ?: return emptyMap()
        val out = HashMap<String, Int>(512)
        for (i in 0 until added.length()) {
            val t = added.optJSONObject(i) ?: continue
            if (!t.optBoolean("special", false)) continue
            val content = t.optString("content")
            if (content.isEmpty()) continue
            out[content] = t.optInt("id", -1)
        }
        return out
    }
}
