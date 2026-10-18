package com.screenast.app.core

import android.content.Context
import android.util.Log
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.currentCoroutineContext
import kotlinx.coroutines.ensureActive
import kotlinx.coroutines.withContext
import java.io.File

/**
 * **导入后的兼容性冒烟测试**（用户 2026-10-18 要求，可选）。
 *
 * ## 它测什么、不测什么
 *
 * 测的是「这个文件到底是不是它自称的那个模型」—— 三类最常见的坏文件：
 *
 * 1. **半个文件**（下载/拷贝中断）：大小与清单差得远；
 * 2. **张冠李戴**（用户把 rec 当 det、把 safetensors 当 onnx）：魔数不对；
 * 3. **HTML 错误页**（镜像站挂了返回 404 页面，浏览器另存为就是 .onnx）：魔数不对。
 *
 * **不测**「模型能不能跑」—— 那要真建 session、吃几百 MB 内存、跑一张图，
 * 在导入那一刻做太重。跑不起来是后端的事，日志里有 `AST.Llama` / `AST.Ocr` / `AST.Asr` 的原样异常。
 *
 * ## 为什么魔数比扩展名可靠
 *
 * `.gguf` 开头四个字节就是 `GGUF`；ONNX 是 protobuf，第一个字段几乎总是
 * `ir_version`（tag `0x08`）。扩展名谁都能改，魔数改不了。
 */
object ModelSmoke {

    private const val TAG = "AST.Smoke"

    data class Result(
        val rel: String,
        val ok: Boolean,
        /** 一句话结论，直接显示在弹窗里。 */
        val detail: String,
    )

    private fun expected(rel: String): Long =
        ModelImporter.items.firstOrNull { it.rel == rel }?.bytes ?: 0L

    /** 逐个文件检查；[onStep] 用来刷「第几个 / 共几个」。 */
    suspend fun run(
        ctx: Context,
        rels: List<String>,
        onStep: (index: Int, count: Int, rel: String) -> Unit = { _, _, _ -> },
    ): List<Result> = withContext(Dispatchers.IO) {
        val root = ModelImporter.root(ctx)
        val out = ArrayList<Result>(rels.size)
        rels.forEachIndexed { i, rel ->
            currentCoroutineContext().ensureActive()
            onStep(i + 1, rels.size, rel)
            out += check(File(root, rel), rel)
        }
        Log.i(TAG, "冒烟测试完成：${out.count { it.ok }}/${out.size} 通过")
        out
    }

    /** 单文件检查（也供导入流程在拷完一个文件后顺手调）。 */
    fun check(f: File, rel: String): Result {
        if (!f.isFile) return Result(rel, false, "文件不存在")
        val len = f.length()
        if (len == 0L) return Result(rel, false, "空文件（0 字节）")
        val want = expected(rel)
        if (want > 0) {
            val diff = kotlin.math.abs(len - want)
            // 2% 容差：不同镜像偶尔差几个字节的对齐，但差得多就是没下完 / 下错了。
            if (diff > want / 50) {
                return Result(
                    rel,
                    false,
                    "大小不对：${len / 1024 / 1024}MB，应为 ${want / 1024 / 1024}MB" +
                        if (len < want) "（没下完？）" else "（不是这个文件？）",
                )
            }
        }
        val head = ByteArray(16)
        val n = try {
            f.inputStream().use { it.read(head) }
        } catch (t: Throwable) {
            return Result(rel, false, "读不了：${t.javaClass.simpleName}")
        }
        if (n < 4) return Result(rel, false, "文件太短（$n 字节）")
        return when {
            rel.endsWith(".gguf") ->
                if (head[0] == 'G'.code.toByte() && head[1] == 'G'.code.toByte() &&
                    head[2] == 'U'.code.toByte() && head[3] == 'F'.code.toByte()
                ) {
                    Result(rel, true, "GGUF 头正常，${len / 1024 / 1024}MB")
                } else {
                    Result(rel, false, "不是 gguf（开头不是 GGUF）")
                }

            rel.endsWith(".onnx") ->
                if (head[0] == 0x08.toByte()) {
                    Result(rel, true, "ONNX 头正常，${fmtMb(len)}")
                } else if (head[0] == '<'.code.toByte()) {
                    Result(rel, false, "这是 HTML/XML，不是 onnx（下载到了错误页）")
                } else {
                    Result(rel, false, "不是 onnx（protobuf 头不对）")
                }

            rel.endsWith(".txt") || rel.endsWith(".json") ->
                if (len > 100) Result(rel, true, "文本 ${fmtMb(len)}")
                else Result(rel, false, "文本太小（$len 字节）")

            else -> Result(rel, true, "${fmtMb(len)}")
        }
    }

    private fun fmtMb(bytes: Long): String =
        if (bytes >= 1024L * 1024L * 1024L) String.format("%.2fGB", bytes / 1024.0 / 1024.0 / 1024.0)
        else String.format("%.1fMB", bytes / 1024.0 / 1024.0)
}
