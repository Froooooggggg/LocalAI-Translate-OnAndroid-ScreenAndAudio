package com.screenast.app.vision

import android.content.Context
import java.io.File

/**
 * 识图要用的端侧模型清单与安装状态。
 *
 * ## 为什么只有 det + rec 两个
 *
 * 屏幕文字识别 = 「先找字框（det），再认字（rec）」。缺 det 就没有坐标，
 * 缺 rec 就只有框没有字；两者都不能省，也不能用别的模型顶替。
 *
 * ## 两套权重（用户 m03800：先换 PP-OCRv5 server 权重试试）
 *
 * - [Variant.MOBILE]：`PP-OCRv5_mobile` det 4.7MB + rec 16MB ≈ 21MB，Hmean 79.0，
 *   一帧 1440×3200 屏幕的 OCR 段约 1.2s。
 * - [Variant.SERVER]：`PP-OCRv5_server` det 83.6MB + rec 80.2MB ≈ 164MB，容量大得多。
 *   PC 侧同一张 1440×3200 截图（YouTube 首页）：mobile 检出 28 框、rec 141ms；
 *   server 检出 41 框、rec 355ms，长标题和糊字明显更全（`天地无用` 这类也能连上）。
 *   代价：PC 上 det 555ms vs 73ms，arm64 手机上会再慢 2~3 倍。
 *
 * 两套的词表**完全相同**（18385 维：0 = blank，1..18383 = 字典，18384 = 空格），
 * 所以 `assets/ocr/ppocrv5_dict.txt` 一份够用，不用按变体分开。
 *
 * 🚧 模型文件不进 git，端侧放到 `filesDir/models/ocr/`；PC 侧对比脚本见 `测试数据/probe/ocr/compare.py`。
 */
object VisionModels {

    /** 放 ONNX 的目录（应用私有）。 */
    const val DIR = "models/ocr"

    /**
     * 一套 det+rec 权重。
     *
     * @param maxSide det 的输入边长上限：mobile 按 960 训练，放大只会变慢；
     *   server 容量大，1280 能多检出一批小字。
     */
    enum class Variant(
        val id: String,
        val detName: String,
        val recName: String,
        val maxSide: Int,
        val label: String,
    ) {
        MOBILE(
            "mobile",
            "ppocrv5-mobile-det.onnx",
            "ppocrv5-mobile-rec.onnx",
            960,
            "轻量 mobile · 21MB",
        ),
        SERVER(
            "server",
            "ppocrv5-server-det.onnx",
            "ppocrv5-server-rec.onnx",
            1280,
            "高精度 server · 164MB",
        ),
        ;

        companion object {
            fun of(id: String?): Variant = entries.firstOrNull { it.id == id } ?: MOBILE
        }
    }

    fun dir(ctx: Context): File = File(ctx.filesDir, DIR)

    /**
     * 外部私有目录里的同一份东西（`Android/data/<pkg>/files/models/ocr`）。
     *
     * 排查 2026-10-18 用户报的「下载完了还是未安装」时发现的第二个坑：
     * llama / NLLB / MI-GAN 的定位器早就同时认外部目录，只有 OCR 死认私有目录。
     */
    fun externalDir(ctx: Context): File =
        File(ctx.getExternalFilesDir(null) ?: ctx.filesDir, DIR)

    /**
     * 找到实际存在的那个文件；都没有就返回「规范名」路径（报错里显示它）。
     *
     * 两种命名都认：早期随包用 `ppocrv5_mobile_det.onnx`（下划线），
     * 现在统一成 `ppocrv5-mobile-det.onnx`（连字符）—— 用户从
     * PPOCRv5-ONNX 仓库直接下就是下划线，认不出来的话界面只会说「未安装」，
     * 而文件明明躺在目录里（这正是用户 2026-10-18 撞上的现象）。
     */
    private fun locate(ctx: Context, name: String): File {
        val alt = name.replace('-', '_')
        val candidates = listOf(
            File(dir(ctx), name), File(dir(ctx), alt),
            File(externalDir(ctx), name), File(externalDir(ctx), alt),
        )
        return candidates.firstOrNull { it.isFile } ?: candidates.first()
    }

    fun det(ctx: Context, v: Variant = Variant.MOBILE): File = locate(ctx, v.detName)

    fun rec(ctx: Context, v: Variant = Variant.MOBILE): File = locate(ctx, v.recName)

    /** 两个模型文件的句柄；缺任何一个都返回 null（调用方据此决定是否装载引擎）。 */
    class Files(val det: File, val rec: File, val variant: Variant)

    /** det 与 rec 都到位才算「可跑」。 */
    fun installed(ctx: Context, v: Variant = Variant.MOBILE): Boolean =
        det(ctx, v).isFile && rec(ctx, v).isFile

    /** 已装好的变体（0、1 或 2 个）。 */
    fun installedVariants(ctx: Context): List<Variant> = Variant.entries.filter { installed(ctx, it) }

    /** 严格按变体取；没装齐就是 null。 */
    fun resolve(ctx: Context, v: Variant = Variant.MOBILE): Files? {
        val d = det(ctx, v)
        val r = rec(ctx, v)
        return if (d.isFile && r.isFile) Files(d, r, v) else null
    }

    /**
     * 按想要的变体取，没装就退到另一个已装的。
     *
     * 手机上很常见：切了「高精度」却没放文件。这时不能直接报错罢工 ——
     * 退到轻量继续能用，第二个返回值告诉调用方「退化过」，界面据此提示。
     */
    fun resolveOrFallback(ctx: Context, want: Variant): Pair<Files, Boolean>? {
        resolve(ctx, want)?.let { return it to false }
        Variant.entries.firstOrNull { it != want }?.let { other ->
            resolve(ctx, other)?.let { return it to true }
        }
        return null
    }

    /**
     * 给界面看的一行状态。
     *
     * 故意把「缺哪个文件」写出来：这一步最容易出问题的地方不是模型跑不动，
     * 而是文件放错目录/名字大小写不一致，界面上却只显示一句「模型没装」。
     */
    fun statusText(ctx: Context, want: Variant = Variant.MOBILE): String {
        val others = installedVariants(ctx)
        return when {
            installed(ctx, want) -> "已安装 · ${want.label}"
            others.isNotEmpty() -> "缺 ${want.label}，在用 ${others.first().label}"
            else -> "未安装（缺 ${want.detName} 与 ${want.recName}）"
        }
    }

    /** 供界面显示的选型说明，避免每次都要翻文档。 */
    const val MODEL_LABEL = "PP-OCRv5 · 轻量 mobile 21MB / 高精度 server 164MB"

    /** 推文件到手机时该放的位置（给文档和脚本共用）。 */
    fun pushHint(v: Variant): String = "${v.detName} + ${v.recName} → files/$DIR/"
}
