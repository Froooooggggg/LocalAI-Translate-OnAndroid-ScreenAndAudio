package com.screenast.app.translate.nllb

import android.content.Context
import java.nio.ByteBuffer
import java.nio.ByteOrder

/**
 * NLLB 的 BPE 分词器。
 *
 * NLLB 用 **BPE + Metaspace**，不是 sentencepiece unigram。词表 256,204 条、
 * 51.4 万条 merge 规则，Android 上没有可用的 HuggingFace tokenizers 实现，
 * 所以这里直接读由 `tools/build_nllb_tokenizer.py` 编译好的紧凑二进制。
 *
 * 分词流程（与 HuggingFace 的 NLLB tokenizer 对齐）：
 *  1. NFC 归一化（用 Java 自带的 Normalizer；原始 charmap 是 4MB 的 sentencepiece
 *     预编译表，对字幕这类规范文本收益极小，不值得搬进 APK）
 *  2. Metaspace：把空格换成 `▁`，并在开头补一个 `▁`
 *  3. 按 UTF-8 字符切分成初始符号，跑 BPE merge
 *
 * ⚠️ **全程以「字节」为单位，不是字符。** `▁` 是 U+2581（3 字节），
 * 而 BPE 的中间符号经常是**半个多字节字符**，非法 UTF-8。
 * 编码侧因此用 ISO-8859-1 字符串当 map key（字节一一对应）；
 * 解码侧必须先把所有 token 的字节拼起来、**最后**才做一次 UTF-8 解码，
 * 否则半个汉字会被 Java 的严格 UTF-8 解码器变成 U+FFFD（真机上显示为 `<unk>`）。
 * 详见 [decode] 与 [bytesToText] 的注释。
 */
class NllbTokenizer private constructor(
    /**
     * 词表的**文本视图**，key 是 ISO-8859-1（字节一一对应）。
     *
     * ⚠️ 这里绝不能用 `String(raw, Charsets.UTF_8)`：词表里大量 token **不是合法 UTF-8**
     * （半个汉字、半个 `▁`），Java 的 UTF-8 解码器会把它们**静默**替换成 U+FFFD，
     * 而 U+FFFD 再 encode 回去只有 1 个字节 `3F`(= `?`)。
     * 真机上表现为「每一个非 ASCII 字符都查不到、全部变成 `<unk>`」——
     * 而磁盘上的 tokenizer.bin 完全正确（SHA256 与源文件一致）。
     */
    private val vocab: Array<String>,
    /** 词表下标（= 该 token 的 UTF-8 字节串）→ token id。 */
    private val tokenIndex: HashMap<String, Int>,
    /** merge 对 (左半 id, 右半 id) → rank。rank 小的先合并。 */
    private val pairRank: HashMap<Long, Int>,
    val unkId: Int,
    val eosId: Int,
    val bosId: Int,
    val padId: Int,
) {

    val size: Int get() = vocab.size

    /**
     * 语种标签 token 的 id 集合（`zho_Hans` / `eng_Latn` / `jpn_Jpan` / `kor_Hang` …）。
     *
     * 由词表内容判定而不是硬编码 —— vocav 里这类 token 有 404 个（含 `▁` 前缀变体），
     * 换模型也不会失效。它们**只在输入侧有意义**（源语靠前置、目标语靠 forced BOS），
     * 一旦出现在 decode 的输入里就该丢掉，否则浮窗上会冒出 `zho_Hans ` 这种东西。
     */
    private val langTokenIds: HashSet<Int> by lazy {
        val set = HashSet<Int>(512)
        for ((i, tok) in vocab.withIndex()) {
            val t = tok.removePrefix("▁".toString())
            // FLORES-200 的形状：3 个小写字母 + `_` + 首字母大写其余小写的 4 字母
            if (t.length == 8 && t[3] == '_' &&
                t[0].isLowerCase() && t[1].isLowerCase() && t[2].isLowerCase() &&
                t[4].isUpperCase() && t[5].isLowerCase() && t[6].isLowerCase() && t[7].isLowerCase()
            ) {
                set.add(i)
            }
        }
        set
    }

    companion object {
        /** 二进制格式版本。改了布局必须同时改 `tools/build_nllb_tokenizer.py`。 */
        private const val MAGIC = "SNATKN02"

        /** SentencePiece 的空格标记 U+2581。 */
        const val SPACE = '\u2581'
        /** 同上，但**以 UTF-8 字节**表示（`E2 96 81`）。字节域里一律用它。 */
        val SM_BYTES: ByteArray = byteArrayOf(0xE2.toByte(), 0x96.toByte(), 0x81.toByte())

        /**
         * 全角/中文标点 → ASCII 等价物。
         *
         * 每一项都是实测**不在 NLLB 词表里**的字符（检查工具：
         * `tools/check_punct_coverage.py`）。不在表里 = 编码成 unk = 信息丢失，
         * 所以宁可在送模型前换成半角。
         *
         * 故意**不含** `。`(U+3002, id=253935)、`、`(U+3001, id=253131)、
         * `·`、`〔〕` —— 这些词表里有，原样保留更贴近模型训练分布。
         */
        private val PUNCT_MAP: Array<Pair<String, String>> = arrayOf(
            "\uFF0C" to ",",      // ， 全角逗号（最要命的一个）
            "\uFF1B" to ";",      // ；
            "\uFF1A" to ":",      // ：
            "\uFF1F" to "?",      // ？
            "\uFF01" to "!",      // ！
            "\u201C" to "\"",     // “ 
            "\u201D" to "\"",     // ”
            "\u2018" to "'",      // ‘
            "\u2019" to "'",      // ’
            "\uFF08" to "(",      // （
            "\uFF09" to ")",      // ）
            "\u300A" to "<",      // 《
            "\u300B" to ">",      // 》
            "\u3008" to "<",      // 〈
            "\u3009" to ">",      // 〉
            "\u3010" to "[",      // 【
            "\u3011" to "]",      // 】
            "\u2014" to "-",      // — 破折号
            "\u2026" to "...",    // … 省略号
            "\u3000" to " ",      // 全角空格
        )

        /** 从 assets 里的编译产物加载。 */
        fun fromAssets(context: Context, assetPath: String): NllbTokenizer =
            context.assets.open(assetPath).use { fromBytes(it.readBytes()) }

        /**
         * 从磁盘文件加载（`models/nllb/tokenizer.bin`）。
         *
         * 2026-10-18 起词表不再打进 APK：它由 NLLB-200（CC-BY-NC-4.0）的词表编译而来，
         * 随包装分发等于分发非商用素材。要用 NLLB 的用户把它和三个 onnx 一起放进
         * 模型目录即可（`tools/build_nllb_tokenizer.py` 里有编译说明）。
         */
        fun fromFile(file: java.io.File): NllbTokenizer = fromBytes(file.readBytes())

        fun fromBytes(bytes: ByteArray): NllbTokenizer {
            val buf = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN)

            val magic = ByteArray(8)
            buf.get(magic)
            val magicStr = String(magic, Charsets.US_ASCII)
            require(magicStr == MAGIC) {
                "分词器文件格式不对：读到 $magicStr，需要 $MAGIC（请重跑 tools/build_nllb_tokenizer.py）"
            }

            val vocabSize = buf.int
            val mergeCount = buf.int
            val unkId = buf.int
            val eosId = buf.int
            val bosId = buf.int
            val padId = buf.int
            val prefixCount = buf.int
            repeat(prefixCount) { buf.int }   // 预留的"文本前缀 token"，目前为空

            val vocab = Array(vocabSize) { "" }
            val tokenIndex = HashMap<String, Int>(vocabSize * 4 / 3)
            for (i in 0 until vocabSize) {
                val len = buf.short.toInt() and 0xFFFF
                val raw = ByteArray(len)
                buf.get(raw)
                // ⚠️ 必须用 ISO-8859-1（字节 ↔ 字符一一对应），**不能**用 UTF-8。
                //
                // 踩过一次，症状极其误导：真机上每一个非 ASCII 字符都编码成 `<unk>`，
                // 而磁盘上的 tokenizer.bin 与源文件 SHA256 完全一致。
                // 根因就在这里 —— BPE 词表里大量 token 是**半个多字节字符**（非法 UTF-8），
                // Java 的 UTF-8 解码器不抛异常，而是把它们**静默**替换成 U+FFFD；
                // U+FFFD 再 encode("utf-8") 得到 3 字节，而这里的旧代码重新提取时
                // 只拿到 1 字节 `3F`(= `?`) —— 词表在装载那一刻就被毁掉了。
                //
                // 之所以 bug 长期潜伏：编码路径用的是另一个 ISO-8859-1 的 map
                // （见下面 tokenIndex），两条视图一份数据、其中一条有损，
                // 于是「编码有时对、解码全错」这种组合极难从现象反推。
                vocab[i] = String(raw, Charsets.ISO_8859_1)
                tokenIndex[String(raw, Charsets.ISO_8859_1)] = i
            }

            // merge 表按「左半 id + 右半 id」的 pair 存 rank。
            // 不能拿"合并后的 token"去查——那是另一种（错误的）表示。
            val pairRank = HashMap<Long, Int>(mergeCount * 4 / 3)
            for (i in 0 until mergeCount) {
                val a = buf.int
                val b = buf.int
                pairRank.putIfAbsent(packPair(a, b), i)   // 保留最早的 rank
            }
            return NllbTokenizer(vocab, tokenIndex, pairRank, unkId, eosId, bosId, padId)
        }

        /** 把 (a, b) 两个 int 压进一个 Long 当 map key，避免每次分配 Pair。 */
        private fun packPair(a: Int, b: Int): Long =
            (a.toLong() shl 32) or (b.toLong() and 0xFFFFFFFFL)
    }

    // ------------------------------------------------------------------ 编码

    /**
     * 把一段文本编码成 token id 序列。
     *
     * @param sourceFlores 源语的 FLORES 码（如 `eng_Latn`）。NLLB 要求把它作为
     *        第一个 token 一起喂给 encoder，否则模型不知道该按哪种语言理解输入。
     */
    fun encode(text: String, sourceFlores: String): IntArray {
        val ids = ArrayList<Int>(text.length / 2 + 8)

        // NLLB 的 TemplateProcessing 是 [源语 token] + 序列 + [</s>]
        vocabIndexOf(sourceFlores)?.let { ids.add(it) }

        val normalized = normalizeForBpe(text)
        if (normalized.isNotEmpty()) {
            for (t in bpe(spmEncode(normalized))) ids.add(t)
        }
        ids.add(eosId)
        return ids.toIntArray()
    }

    /**
     * 送进 BPE 前的完整清洗链。**所有编码入口都必须走这里**，
     * 否则会退化成"有的入口做了标点归一化、有的没做"这种最难查的不一致。
     */
    private fun normalizeForBpe(text: String): String = normalizePunct(normalize(text))

    /** 只编码正文，不加语种前缀与 EOS。 */
    fun encodePlain(text: String): IntArray = bpe(spmEncode(normalizeForBpe(text)))

    private fun vocabIndexOf(token: String): Int? = tokenIndex[token]

    /**
     * Metaspace 预分词：空格→`▁`，开头补一个 `▁`。连续空格会产生多个 `▁`。
     *
     * ⚠️ **返回值是字节，不是字符串。** 这是本文件最重要的一条不变式：
     * 词表里的 token 是任意字节串（很多不是合法 UTF-8），而 Kotlin/Java 的
     * `String` 无法无损承载它们 —— 一旦经过 `String(raw, UTF_8)` 就会变成 U+FFFD。
     * 所以从预分词开始，整条 BPE 链路一律用 `ByteArray` 表示符号，
     * 只在**调试显示**时才转成字符串。
     */
    private fun spmEncode(text: String): ByteArray {
        val out = java.io.ByteArrayOutputStream(text.length * 2 + 8)
        out.write(SM_BYTES)                              // add_prefix_space：开头补一个 ▁
        for (c in text) {
            if (c == ' ') out.write(SM_BYTES) else out.write(c.toString().toByteArray(Charsets.UTF_8))
        }
        return out.toByteArray()
    }

    private fun normalize(text: String): String {
        // 先做 NFC；同时把常见控制字符剔掉，避免它们一路上变成 <unk>
        val filtered = buildString(text.length) {
            for (c in text) {
                if (c == '\n' || c == '\r' || c == '\t') append(' ') else append(c)
            }
        }
        return java.text.Normalizer.normalize(filtered, java.text.Normalizer.Form.NFC)
    }

    /**
     * 把**中文/全角标点归一化成 ASCII 等价物**。必须在 [normalize] 之后、BPE 之前调用。
     *
     * 为什么必须做：实测 NLLB 词表对中文标点的覆盖率只有 5/25 ——
     * `， ； ： ？ ！ “ ” ‘ ’ （ ） 《 》 〈 〉 【 】 — … 　` **全都不在词表里**，
     * 编码时统统落到 unk(3)、解码回空串，于是句子的停顿与语气**在输入端被静默丢弃**。
     * 对比：ASCII 标点 32/32 全有、日文标点 8/8 全有，唯独中文全角这一组大面积缺失。
     *
     * 注意这是**单向**问题：词表里没有这些字符的 *输入* 表示，
     * 但模型照样能把它们作为 *输出* 生成出来（`您好,欢迎回到频道.` 里的 `,`、
     * 以及 `。`=253935 都能正常输出）。所以只需清洗输入。
     *
     * 换成 ASCII 的风险很小：NLLB 在网页语料上训练，`，` 与 `,` 在语义上等价，
     * 而"完全丢失"显然比"换成半角"更糟。改完用真机 `auto=toktest` 复验即可。
     */
    fun normalizePunct(text: String): String {
        var out = text
        for ((from, to) in PUNCT_MAP) {
            if (out.indexOf(from) >= 0) out = out.replace(from, to)
        }
        return out
    }

    /**
     * BPE，全程在**字节域**里跑。
     *
     * 符号用 `ByteArray`，查表时转成 ISO-8859-1 字符串 —— 这个编码把每个字节
     * 映射到一个字符，因此 `String` 与 `ByteArray` 之间**无损可逆**，
     * 正好补上 UTF-8 做不到的那一点。
     *
     * 实现用线性找最小 rank。字幕单句只有几十个符号，朴素实现足够快。
     */
    fun bpe(spmBytes: ByteArray): IntArray {
        if (spmBytes.isEmpty()) return IntArray(0)
        // 初始符号 = **在 Unicode 字符边界**切开后各自的 UTF-8 字节。
        //
        // ⚠️ 不能按字节切（曾经想当然写成按字节切，结果所有中文变成一串 <unk>）：
        // NLLB 的 vocab 绝大部分是**整字符** token，单字节符号（如 E9、A2）大多不在表里，
        // BPE 第一步就无 pair 可合并，直接落成 unk。
        // SentencePiece 的 byte fallback 是"未登录字符的兜底"，不是"切分粒度"。
        val syms = ArrayList<ByteArray>(spmBytes.size + 4)
        var i = 0
        while (i < spmBytes.size) {
            val n = utf8CharLen(spmBytes[i])
            val end = minOf(i + n, spmBytes.size)
            syms.add(spmBytes.copyOfRange(i, end))
            i = end
        }

        while (syms.size > 1) {
            var bestRank = Int.MAX_VALUE
            var bestAt = -1
            for (k in 0 until syms.size - 1) {
                // ⚠️ pairRank 的 key 是**把两个 id 压进一个 Long**（packPair），
                // 不是单个 token id。这里曾经写成 `pairRank[merged]`（Int 当 key），
                // 报的是 "Type inference failed. The value of the type parameter 'K'
                // must be mentioned in input types" —— 报错信息完全看不出真实原因，
                // 因为 HashMap.get 的泛型 K 推不出 Long。
                val left: Int = tokenIndex[syms[k].key()] ?: continue
                val right: Int = tokenIndex[syms[k + 1].key()] ?: continue
                val r = pairRank[packPair(left, right)] ?: continue
                if (r < bestRank) {
                    bestRank = r
                    bestAt = k
                }
            }
            if (bestAt < 0) break
            syms[bestAt] = syms[bestAt] + syms[bestAt + 1]   // ByteArray 拼接，无损
            syms.removeAt(bestAt + 1)
        }

        val out = IntArray(syms.size)
        for (k in syms.indices) out[k] = tokenIndex[syms[k].key()] ?: unkId
        return out
    }

    /** 字节串 → 查表用的 ISO-8859-1 键（1 字节 ↔ 1 字符，无损）。 */
    private fun ByteArray.key(): String = String(this, Charsets.ISO_8859_1)

    /** 一个 UTF-8 前导字节对应的字符长度。非法前导按 1 处理，交给 unk 兜底。 */
    private fun utf8CharLen(b: Byte): Int {
        val v = b.toInt() and 0xFF
        return when {
            v < 0x80 -> 1
            v and 0xE0 == 0xC0 -> 2
            v and 0xF0 == 0xE0 -> 3
            v and 0xF8 == 0xF0 -> 4
            else -> 1
        }
    }

    // ------------------------------------------------------------------ 解码

    /** 单个 token id → 文本片段（**仅供调试显示**，不要用它拼装正文，见 [decode]）。 */
    fun tokenText(id: Int): String =
        if (id in vocab.indices) vocab[id].orEmpty() else ""

    /** 单个 token 的原始 UTF-8 字节。BPE 的中间 token 可能是半个多字节字符。 */
    fun tokenBytes(id: Int): ByteArray =
        if (id in vocab.indices) vocab[id].toByteArray(Charsets.ISO_8859_1) else ByteArray(0)

    /** 该 token 是不是语种标签（`zho_Hans`、`eng_Latn`，或带 `▁` 前缀的变体）。 */
    fun isLangToken(id: Int): Boolean = id in langTokenIds

    /**
     * token id 序列 → 文本。
     *
     * 规则：遇 EOS 停；跳过 `<s>`/`<pad>`/语种标签；`▁` 换回空格；
     * 去掉开头的空格（Metaspace 的 add_prefix_space）；**整串一起做 UTF-8 解码**。
     *
     * ⚠️ 最后一条是关键，也是本项目踩过的一个隐蔽 bug：BPE 词表里存在
     * **半个多字节字符**的 token（例如「频」是 `E9 A2 91`，可能被切成
     * `E9 A2` 与 `91` 两个 token）。旧实现逐个 token 调 `String(bytes, UTF_8)`，
     * Java 的 UTF-8 解码器遇到非法序列会**静默**替换成 U+FFFD，而且无法还原。
     *
     * 正确做法：全程在 ISO-8859-1 域里按字节拼接，最后统一次 UTF-8 解码。
     * 注意装载词表时也必须用 ISO-8859-1（见 [fromBytes]），否则词表本身
     * 在装载那一刻就已经被毁了 —— 那个 bug 更隐蔽，因为它是"全错"而不是"偶尔错"。
     */
    fun decode(ids: IntArray): String = bytesToText(tokenBytesOf(ids))

    /** 语种标签等特殊 token 不参与正文，跳过；其余按字节拼起来。 */
    private fun tokenBytesOf(ids: IntArray): ByteArray {
        val out = java.io.ByteArrayOutputStream(ids.size * 3)
        for (id in ids) {
            if (id == eosId) break
            if (id == bosId || id == padId || id == unkId) continue
            if (isLangToken(id)) continue
            out.write(tokenBytes(id), 0, tokenBytes(id).size)
        }
        return out.toByteArray()
    }

    /**
     * 字节流 → 文本。抽出成独立函数，是为了让上面的「字节域」语义可以被单独验证
     * （调试台的 tokenizer 自检 `tokdump` 会带着十六进制直接调它）。
     */
    fun bytesToText(raw: ByteArray): String {
        val n = raw.size
        val buf = ByteArray(n)
        var len = 0
        var i = 0
        while (i < n) {
            // `▁` = E2 96 81。它代表一个原始空格，且只可能出现在 token 边界上
            // （U+2581 本身是合法 UTF-8，不会被切开），所以按三字节整块替换是安全的。
            if (i + 2 < n && raw[i] == 0xE2.toByte() && raw[i + 1] == 0x96.toByte() && raw[i + 2] == 0x81.toByte()) {
                buf[len++] = ' '.code.toByte()
                i += 3
            } else {
                buf[len++] = raw[i]
                i++
            }
        }
        var s = repairUtf8(String(buf, 0, len, Charsets.UTF_8))
        if (s.startsWith(" ")) s = s.substring(1)          // add_prefix_space 造成的前导空格
        return s.trim()
    }

    /**
     * 尽量救回非法 UTF-8 的残留字节。
     *
     * 正常情况下这里不该有东西可修（[decode] 已经整串解码）；保留它是因为：
     *  - 词表里少数 token 本身就是非法序列（如 em-dash 的 charmap 变体）；
     *  - 早退/截断路径可能让字节流停在半个字符上。
     * 不修的话 U+FFFD 会一路显示到浮窗上，比少一个字符更难排查。
     */
    private fun repairUtf8(s: String): String {
        if (s.indexOf('\uFFFD') < 0) return s
        return s.replace("\uFFFD+", "").replace("\uFFFD", "")
    }

    /** 调试用：把字节序列打成十六进制，配合 [bytesToText] 定位跨 token 的半字符。 */
    fun hexDump(ids: IntArray, limit: Int = 32): String {
        val sb = StringBuilder()
        for (id in ids.take(limit)) {
            if (id == eosId) {
                sb.append(" | ").append(id).append(":EOS")
                break
            }
            val b = tokenBytes(id)
            val hex = b.joinToString("") { "%02X".format(it) }
            sb.append(" | ").append(id).append(':').append(hex)
                .append('(').append(String(b, Charsets.ISO_8859_1)).append(')')
        }
        return sb.toString()
    }

    /**
     * token id 序列 → 文本，但**保留** U+FFFD 不修，用于区分
     * 「半个汉字被拼回来了」和「拼接后仍然是非法序列」。
     *
     * [decode] 会静默清掉 U+FFFD，这在生产路径上是对的，但在自检时会把
     * 「有问题的字符」和「本来就没有的字符」混为一谈。
     */
    fun decodeRaw(ids: IntArray): String {
        val raw = tokenBytesOf(ids)
        val buf = ByteArray(raw.size)
        var len = 0
        var i = 0
        while (i < raw.size) {
            if (i + 2 < raw.size && raw[i] == 0xE2.toByte() && raw[i + 1] == 0x96.toByte() && raw[i + 2] == 0x81.toByte()) {
                buf[len++] = ' '.code.toByte()
                i += 3
            } else {
                buf[len++] = raw[i]
                i++
            }
        }
        return String(buf, 0, len, Charsets.UTF_8).trim()
    }

    /**
     * 自检用：把一串文本编码后立刻解码回来，并给出细节。
     *
     * ⚠️ 比对基准是 **`normalizePunct` 之后**的文本，不是原文。
     * 原因：全角标点不在词表里，本实现会**故意**把它们换成 ASCII 等价物，
     * 所以 `真的吗？` 往返回来必然是 `真的吗?` —— 这是设计行为，不是缺陷。
     * 若不按归一化后的基准比，这个测试会永远红着，从而掩盖真正的回归。
     */
    fun roundTrip(text: String, srcLangCode: String = ""): RoundTrip {
        val ids = encode(text, srcLangCode)
        val back = decode(ids)
        val expected = normalizePunct(text.trim())
        return RoundTrip(ids, back, back == expected, hexDump(ids), decodeRaw(ids), expected)
    }

    /**
     * 词表装载自检。
     *
     * 存在的理由：真机实测出现过「**每一个字符都变成 `<unk>`**」，而同一份
     * tokenizer.bin 在 PC 上用 Python 参考实现算出来完全正常。要区分是
     * 「二进制解析错位」还是「BPE 查表错」，必须把中间量直接打出来：
     *
     *  - `vocab[id]` 来自**按顺序**读出的数组（不经过任何 map），
     *    它能证明「解析有没有错位」；
     *  - `lookup(原文)` 走的是 ISO-8859-1 键的 map，能证明「查表键对不对」。
     *
     * 两者都正常而往返仍然失败，才说明问题在 BPE 合并逻辑里。
     */
    fun diagnostics(): String {
        val sb = StringBuilder()
        sb.append("vocab.size=${vocab.size} tokenIndex.size=${tokenIndex.size} pairRank.size=${pairRank.size}\n")
        sb.append("unk=$unkId eos=$eosId bos=$bosId pad=$padId\n")
        for (id in intArrayOf(0, 1, 2, 3, 65, 208, 541, 598, 8446, 11926, 248195, 248059, 253569, 249624, 256200)) {
            val b = tokenBytes(id)
            sb.append("  vocab[$id] len=${b.size} hex=${b.joinToString("") { "%02X".format(it) }}")
                .append(" text='").append(tokenText(id)).append("'\n")
        }
        val probes = listOf(
            "H", "ello", "\u2581", "\u2581H", "\u2581Hello", "频", "▁频", "，", "道", "N", "LL",
        )
        for (p in probes) {
            sb.append("  lookup('").append(p).append("') = ")
                .append(tokenIndex[p]?.toString() ?: "null").append('\n')
        }
        sb.append("  encodePlain(\"H\") = ").append(encodePlain("H").joinToString(",")).append('\n')
        sb.append("  spmEncode(\"Hello\") = ")
            .append(spmEncode("Hello").joinToString(" ") { "%02X".format(it) }).append('\n')
        return sb.toString()
    }

    data class RoundTrip(
        val ids: IntArray,
        val text: String,
        val ok: Boolean,
        val detail: String,
        val raw: String,
        /** 归一化后的期望文本。与 [text] 不等时说明标点被有意替换过。 */
        val expected: String,
    ) {
        override fun equals(other: Any?): Boolean = this === other
        override fun hashCode(): Int = System.identityHashCode(this)
    }

    /** 位置 0 是否停（供生成循环提前退出）。 */
    fun isEos(id: Int): Boolean = id == eosId

    /** 调试用：把 token id 翻译成可读文本，用于排查分词问题。 */
    fun describe(ids: IntArray, limit: Int = 40): String =
        ids.take(limit).joinToString(" | ") { "$it:${tokenText(it)}" }

    /**
     * 调试用：一行看完「原文 → 归一化 → token 序列 → 解码回来」。
     *
     * 标点归一化和 unk 兜底都会**静默**改动文本，只看最终译文根本判断不出
     * 是哪一步动了什么。这个函数把整条链路的每一跳都摊开。
     */
    fun traceOneLine(text: String): String {
        val norm = normalizeForBpe(text)
        val ids = encodePlain(text)
        return buildString {
            append("原文='").append(text).append("' ")
            if (norm != text) append("归一化='").append(norm).append("' ")
            append("ids=").append(ids.joinToString(","))
            append(" unk数=").append(ids.count { it == unkId })
            append(" 解码='").append(decode(ids)).append('\'')
        }
    }
}
