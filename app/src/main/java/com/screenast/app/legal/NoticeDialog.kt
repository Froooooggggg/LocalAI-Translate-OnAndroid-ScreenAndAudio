package com.screenast.app.legal

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.heightIn
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.AlertDialog
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.Composable
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp

/**
 * 用户公告 / 免责声明。
 *
 * 出现两处（用户 2026-10-18 的要求）：
 * 1. 首次启动：必须点「我已阅读并同意」才能用，点「退出」直接结束进程。
 * 2. 设置页：随时可以再打开看一遍（只读，一个「关闭」）。
 *
 * 文案刻意写得直白、分条，不写「本公司保留最终解释权」这类空话 ——
 * 真出问题时，能说清楚「责任在谁」的只有具体条款。
 */
object Notice {

    /** 公告版本：改了条款就把版本号 +1，让老用户下次启动再看一遍。 */
    // 2：m06633 加了「禁止商业用途」一条，老用户下次启动需要重新确认。
    const val VERSION = 3

    data class Section(val title: String, val body: String)

    val sections: List<Section> = listOf(
        Section(
            "一、软件状态与使用风险",
            "本软件处于测试阶段（0.7.4 Beta），功能与稳定性都不作保证，可能随时修改或停止维护。" +
                "使用过程中可能出现手机卡顿、发热、耗电加快、应用闪退、系统重启、识别或翻译结果错误、" +
                "以及因模型文件损坏或空间不足导致的数据丢失等问题。\n" +
                "以上风险由使用者自行承担；开发者不提供任何形式的明示或默示担保（包括但不限于适销性、" +
                "特定用途适用性、不侵权），也不对由此产生的任何直接或间接损失负责。" +
                "请在了解风险后使用，重要内容请自行备份并人工核对。",
        ),
        Section(
            "二、开发者责任范围",
            "开发者仅提供软件本身，不为使用者使用本软件的任何方式、以及使用者通过本软件处理的任何对象负责。\n" +
                "· 本软件的全部语音识别、屏幕识别、翻译均在本地设备上完成，不上传、不留存使用者的" +
                "音频、画面、文本；只有在你主动开启「云端 API」并填写自己的密钥时，才会把待翻译文本" +
                "发送到你指定的大模型服务商。\n" +
                "· 屏幕抓取、录音、字幕与译文是否合法，取决于你所在地区的法律与你的使用场景；" +
                "使用者应自行确保已取得必要授权，并对处理结果负责。\n" +
                "· 不得将本软件用于违法用途，包括但不限于：未经他人同意录音录像、窃听或跟踪、" +
                "侵犯他人隐私与个人信息权益、规避技术保护措施、制作或传播违法有害信息、" +
                "侵犯他人著作权等。",
        ),
        Section(
            "三、非商业使用（禁止商用）",
            "本软件是非商业开源项目，**禁止任何形式的商业使用**，包括但不限于：\n" +
                "· 直接或修改后出售、出租、以付费方式提供下载或安装；\n" +
                "· 作为收费服务、收费培训、商业产品的一部分使用，或用于承接付费项目；\n" +
                "· 在应用内植入广告、会员、付费解锁、金额打赏等任何变现形式；\n" +
                "· 以本软件或其衍生版本的名义提供商业技术支持、代维、定制开发。\n" +
                "个人学习、研究、自用，以及在非营利场景中的使用不受限制。" +
                "如需商业使用，须事先取得作者书面许可；未取得许可的商业使用视为侵权。" +
                "第三方模型（如 NLLB 的 CC-BY-NC-4.0）另有各自条款，一并适用。",
        ),
        Section(
            "四、法律、许可与第三方声明",
            "· 使用者应遵守所在国家或地区的法律法规；如本软件在你所在地区被限制使用，请停止使用并卸载。\n" +
                "· 识别 / 翻译得到的内容，其著作权归原作者或原权利人，使用这些内容（转载、商用、" +
                "二次传播）需自行取得授权。\n" +
                "· 本软件使用的开源组件与端侧模型各自遵循其原许可证：代码类多为 Apache-2.0 / MIT；" +
                "OCR（PP-OCRv5）、语音（SenseVoice / Paraformer / Whisper / Silero VAD / GTCRN / TEN-VAD）、" +
                "补背景（MI-GAN）为 Apache-2.0 或 MIT；备选翻译模型 NLLB-200 为 CC-BY-NC-4.0，" +
                "仅限非商业用途。完整清单见随软件发布的「第三方许可」说明。\n" +
                "· 本软件为个人开源项目，与 DeepSeek、Meta、阿里云、腾讯、字节跳动、OpenAI、" +
                "Google 等任何公司均无隶属或背书关系；文中出现的产品名称与商标归各自所有者。\n" +
                "· 如你不同意上述任何一条，请立即停止使用并卸载本软件。",
        ),
    )

    fun plainText(): String = sections.joinToString("\n\n") { "${it.title}\n${it.body}" }
}

/**
 * 公告弹窗。
 *
 * @param firstRun true = 首次启动（强制同意，点「退出」结束进程）；false = 设置里回看（只读）。
 */
@Composable
fun NoticeDialog(
    firstRun: Boolean,
    onAccept: () -> Unit,
    onDismiss: () -> Unit,
    onExit: () -> Unit,
) {
    AlertDialog(
        // 首次启动不让点外面关掉：必须明确同意或退出。
        onDismissRequest = { if (!firstRun) onDismiss() },
        title = {
            Text(
                if (firstRun) "使用前请阅读" else "用户公告与免责声明",
                fontSize = 16.sp,
                fontWeight = FontWeight.SemiBold,
            )
        },
        text = {
            Column(
                modifier = Modifier
                    .heightIn(max = 520.dp)
                    .verticalScroll(rememberScrollState()),
                verticalArrangement = Arrangement.spacedBy(10.dp),
            ) {
                Text(
                    "同传互译 0.7.4 Beta · 公告版本 ${Notice.VERSION}",
                    fontSize = 11.sp,
                    color = Color(0xFF9E9EA7),
                )
                Notice.sections.forEach { s ->
                    Text(s.title, fontSize = 13.sp, fontWeight = FontWeight.Medium)
                    Text(s.body, fontSize = 11.sp, color = Color(0xFFC9C7D2))
                }
                Text(
                    "以上条款如与你所在地区的强制性法律规定冲突，以法律规定为准；其余条款仍有效。",
                    fontSize = 11.sp,
                    color = Color(0xFF9E9EA7),
                    modifier = Modifier.padding(top = 2.dp),
                )
            }
        },
        confirmButton = {
            TextButton(onClick = if (firstRun) onAccept else onDismiss) {
                Text(if (firstRun) "我已阅读并同意" else "关闭", color = Color(0xFFB9A6FF))
            }
        },
        dismissButton = if (firstRun) {
            {
                TextButton(onClick = onExit) { Text("不同意，退出") }
            }
        } else {
            null
        },
    )
}
