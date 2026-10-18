package com.screenast.app.translate.llama

/**
 * 跨 token 拼 UTF-8 的增量解码器。
 *
 * **为什么需要它**：llama.cpp 的词表是字节级 BPE，一个汉字占 3 字节，经常被切成两个
 * token —— 前一个 token 的 piece 就是**半截 UTF-8**（例如「酩」的前两字节 `0xE9 0x85`，
 * 后一个 token 是剩下的 `0x9D`）。
 *
 * 所以**不能**逐 token 转字符串再相加：
 *
 * - `String(bytes, Charsets.UTF_8)` 会把半截序列变成 `U+FFFD` 替换符 → 字幕里出现「酩�大」
 *   这种看不见的脏字符（PC 侧探针里看到的 `<unk>` 就是它的近亲）；
 * - 把它交给 JNI 的 `NewStringUTF` **会直接 SIGABRT 掉整个进程**（实测崩在
 *   `nativeTokenToPiece`，abort message 是 `input is not valid Modified UTF-8`）。
 *
 * 正确做法就是攒着：一个字符的字节齐了才吐出来，不齐就留到下一个 token。
 *
 * 这个类不持有任何 native 资源，可以单独写单测。
 */
internal class Utf8Stream {

    /** 上一次没凑齐的尾巴（UTF-8 一个字符最多 4 字节，留 4 字节够用）。 */
    private var pending = ByteArray(4)
    private var pendingLen = 0

    /**
     * 喂入一个 token 的原始字节，返回**这一步能确定完整**的那部分文本。
     *
     * 返回值可能为空串（尾巴还没凑齐），也可能一次吐出多个字符。
     */
    fun feed(bytes: ByteArray): String {
        if (bytes.isEmpty()) return ""

        val buf = ByteArray(pendingLen + bytes.size)
        System.arraycopy(pending, 0, buf, 0, pendingLen)
        System.arraycopy(bytes, 0, buf, pendingLen, bytes.size)

        val cut = completePrefixLength(buf)
        val text = if (cut == 0) "" else String(buf, 0, cut, Charsets.UTF_8)

        val rest = buf.size - cut
        if (rest > pending.size) pending = ByteArray(rest)
        if (rest > 0) System.arraycopy(buf, cut, pending, 0, rest)
        pendingLen = rest

        return text
    }

    /**
     * 生成结束时的收尾。正常情况下没有残留；真有半截字节时（模型吐到一半被 EOS 截断）
     * 用替换符把它显式暴露出来，而不是悄悄丢掉。
     */
    fun flush(): String {
        if (pendingLen == 0) return ""
        val text = String(pending, 0, pendingLen, Charsets.UTF_8)
        pendingLen = 0
        return text
    }

    private companion object {
        /**
         * 返回 buf 里最后一个**完整**字符的结束位置。
         *
         * 从头按首字节推出的长度往前走；走到某处发现剩余字节不够一个字符，就切在那里。
         * 非法首字节（孤立续字节 `0x80~0xBF`、`0xF8~0xFF`）按 1 字节算，直接交给
         * `Charsets.UTF_8` 变成替换符 —— 那是真的脏数据，不该无限等着。
         */
        fun completePrefixLength(b: ByteArray): Int {
            var i = 0
            while (i < b.size) {
                val c = b[i].toInt() and 0xFF
                val need = when {
                    c < 0x80 -> 1
                    c and 0xE0 == 0xC0 -> 2
                    c and 0xF0 == 0xE0 -> 3
                    c and 0xF8 == 0xF0 -> 4
                    else -> 1
                }
                if (i + need > b.size) return i
                i += need
            }
            return b.size
        }
    }
}
