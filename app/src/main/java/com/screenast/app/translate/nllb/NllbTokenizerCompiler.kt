package com.screenast.app.translate.nllb

import android.util.JsonReader
import android.util.Log
import java.io.ByteArrayOutputStream
import java.io.File
import java.io.StringReader

/**
 * 在手机上把 HuggingFace 的 `tokenizer.json` 编译成 [NllbTokenizer] 能直接读的紧凑二进制。
 *
 * 为什么需要它：HF 上 NLLB 只给 JSON（17MB，还含 4MB 预编译 charmap 和一堆用不上的元数据），
 * 我们自己的读取器要的是定长小端结构（魔数 `SNATKN02`）。以前这份二进制只能用
 * `tools/build_nllb_tokenizer.py` 在电脑上生成再随包发，用户点「下载」是无解的 ——
 * 所以补了这个运行时编译器：下载 JSON → 本机编译 → 落成 `tokenizer.bin`。
 *
 * 输出格式和 `tools/build_nllb_tokenizer.py` **逐字节一致**（小端）：
 * ```
 * magic 8B "SNATKN02"
 * u32 vocabSize
 * u32 mergesCount
 * i32 unk, eos, bos, pad
 * u32 prefixTokenCount (=0)
 * vocab:  vocabSize × [u16 byteLen][UTF-8 bytes]
 * merges: mergesCount × [u32 leftId][u32 rightId]
 * ```
 * 注意：脚本的 docstring 里写「每条带 mergeRank」，实际代码没写，这里也不写。
 */
object NllbTokenizerCompiler {

    private const val TAG = "AST.NllbTok"
    private const val MAGIC = "SNATKN02"

    /**
     * 编译 [jsonFile] 并把结果写到 [outFile]（先写 `.part` 再改名，避免半成品被读到）。
     *
     * @return 写出的字节数；解析不了会抛异常，调用方自己决定是回退还是报错
     */
    fun compileToFile(jsonFile: File, outFile: File): Long {
        val bytes = compile(jsonFile.readText())
        outFile.parentFile?.mkdirs()
        val part = File(outFile.absolutePath + ".part")
        part.writeBytes(bytes)
        if (outFile.exists()) outFile.delete()
        if (!part.renameTo(outFile)) {
            part.copyTo(outFile, overwrite = true)
            part.delete()
        }
        Log.i(TAG, "词表编译完成：${jsonFile.name} → ${outFile.name}，${bytes.size} 字节")
        return bytes.size.toLong()
    }

    /** 纯函数版：JSON 文本 → 二进制字节数组（测试时可单独调）。 */
    fun compile(json: String): ByteArray {
        var vocabSize = 0
        val tokToId = HashMap<String, Int>(1 shl 19)
        val idToTok = ArrayList<String>(1 shl 18)
        val merges = ArrayList<String>(1 shl 18)

        JsonReader(StringReader(json)).use { r ->
            r.beginObject()
            while (r.hasNext()) {
                when (r.nextName()) {
                    "model" -> readModel(r, tokToId, idToTok, merges)
                    else -> r.skipValue()
                }
            }
            r.endObject()
        }
        vocabSize = idToTok.size
        Log.i(TAG, "解析完成：词表 $vocabSize 条，merge ${merges.size} 条")

        var unk = tokToId["<unk>"] ?: 3
        var eos = tokToId["</s>"] ?: 2
        var bos = tokToId["<s>"] ?: 0
        var pad = tokToId["<pad>"] ?: 1

        val out = ByteArrayOutputStream(1 shl 23)
        out.write(MAGIC.toByteArray(Charsets.US_ASCII))
        out.writeLe32(vocabSize)
        out.writeLe32(merges.size)
        // 特殊 token 的位置要和 Python 脚本一致：unk, eos, bos, pad
        out.writeLe32(unk); out.writeLe32(eos); out.writeLe32(bos); out.writeLe32(pad)
        out.writeLe32(0) // prefix tokens（NLLB 用不到）

        for (i in 0 until vocabSize) {
            val b = idToTok[i].toByteArray(Charsets.UTF_8)
            if (b.size > 0xFFFF) throw IllegalStateException("token 过长：id=$i")
            out.write(b.size and 0xFF); out.write((b.size ushr 8) and 0xFF) // u16 LE
            out.write(b)
        }

        var unresolved = 0
        val pairs = IntArray(merges.size * 2)
        for ((idx, m) in merges.withIndex()) {
            val sp = m.indexOf(' ')
            if (sp <= 0) { unresolved++; pairs[idx * 2] = 0; pairs[idx * 2 + 1] = 0; continue }
            val a = tokToId[m.substring(0, sp)] ?: -1
            val b = tokToId[m.substring(sp + 1)] ?: -1
            if (a < 0 || b < 0) { unresolved++; pairs[idx * 2] = 0; pairs[idx * 2 + 1] = 0 }
            else { pairs[idx * 2] = a; pairs[idx * 2 + 1] = b }
        }
        if (unresolved > 0) Log.w(TAG, "$unresolved 条 merge 的一半不在词表里（占位为 0，不会被命中）")

        var i = 0
        while (i < pairs.size) {
            out.writeLe32(pairs[i]); out.writeLe32(pairs[i + 1]); i += 2
        }
        return out.toByteArray()
    }

    /**
     * 只读 `model` 这一段：`{"type":"BPE","vocab":{...},"merges":[...]}`。
     *
     * 用流式 [JsonReader] 而不是 `JSONObject` —— 17MB 的 JSON 全建成对象树，
     * 峰值内存会翻好几倍，中低端机上很容易被系统杀掉。
     */
    private fun readModel(
        r: JsonReader,
        tokToId: HashMap<String, Int>,
        idToTok: ArrayList<String>,
        merges: ArrayList<String>,
    ) {
        r.beginObject()
        while (r.hasNext()) {
            when (r.nextName()) {
                "vocab" -> {
                    r.beginObject()
                    while (r.hasNext()) {
                        val tok = r.nextName()
                        val id = r.nextInt()
                        tokToId[tok] = id
                        while (idToTok.size <= id) idToTok.add("")
                        idToTok[id] = tok
                    }
                    r.endObject()
                }
                "merges" -> {
                    r.beginArray()
                    while (r.hasNext()) {
                        // 两种导出都有：["a b", ...] 或 [["a","b"], ...]
                        if (r.peek() == android.util.JsonToken.BEGIN_ARRAY) {
                            r.beginArray()
                            val a = r.nextString()
                            val b = r.nextString()
                            r.endArray()
                            merges.add("$a $b")
                        } else {
                            merges.add(r.nextString())
                        }
                    }
                    r.endArray()
                }
                else -> r.skipValue()
            }
        }
        r.endObject()
    }

    private fun ByteArrayOutputStream.writeLe32(v: Int) {
        write(v and 0xFF); write((v ushr 8) and 0xFF); write((v ushr 16) and 0xFF); write((v ushr 24) and 0xFF)
    }
}
