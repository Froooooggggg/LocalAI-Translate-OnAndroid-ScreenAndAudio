package com.screenast.app.vision

import android.content.Context
import android.content.Intent
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.net.Uri
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.PickVisualMediaRequest
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.Image
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.heightIn
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.statusBarsPadding
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.Button
import androidx.compose.material3.ButtonDefaults
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.Switch
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.Composable
import androidx.compose.runtime.DisposableEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.screenast.app.core.Lang
import com.screenast.app.core.Settings
import com.screenast.app.core.SettingDivider
import com.screenast.app.core.SettingHint
import android.provider.Settings as SysSettings
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.io.File

/**
 * 「识图」页：屏幕上的文字 → 识别 → 翻译 → 贴回原处。
 *
 * ## 这一页和「同传」页的关系
 *
 * 两者共用同一个进程、同一份 [Settings]、同一个翻译后端，但**采集通道完全相反**：
 * 同传抓音频（AudioRecord），识图抓画面（VirtualDisplay + ImageReader）。
 *
 * ⚠️ 一个 MediaProjection 只能一个 VirtualDisplay，两页互斥 —— 这是系统行为。
 *
 * ## 当前进度（V1）
 *
 * - 推理链路已通：[OcrEngine] 跑 PP-OCRv5 mobile 的 det + rec，中文/英文/日文混排都能出框出字；
 * - 取图先走系统图片选择器（可以用截图验证整条链路），真正的「抓当前屏幕」是 V2；
 * - 翻译与贴回是 V2/V3，这一页现在只如实报 OCR 结果与坐标。
 */
@Composable
fun VisionScreen(
    settings: Settings,
    onChange: (Settings) -> Unit,
    onLog: (String) -> Unit,
) {
    val ctx = LocalContext.current
    val scope = rememberCoroutineScope()

    var lines by remember { mutableStateOf(listOf<String>()) }
    fun log(s: String) {
        lines = (lines + s).takeLast(8)
        onLog("识图：$s")
    }

    val wantVariant = if (settings.visionOcrServer) VisionModels.Variant.SERVER else VisionModels.Variant.MOBILE
    var installed by remember(settings.visionOcrServer) { mutableStateOf(VisionModels.installed(ctx, wantVariant)) }
    val status = remember(installed, settings.visionOcrServer) { VisionModels.statusText(ctx, wantVariant) }

    var bitmap by remember { mutableStateOf<Bitmap?>(null) }
    var blocks by remember { mutableStateOf(listOf<OcrEngine.TextBlock>()) }
    var busy by remember { mutableStateOf(false) }
    var lastMs by remember { mutableStateOf(0L) }

    // 识图有自己的设置页（用户 m06696：同传和识图「两个各有一个单独的设置」），
    // 跟同传那个一样：标题栏右边一个「设置 ⚙」，点进去整页换成设置项，左上角返回。
    var showSettings by remember { mutableStateOf(false) }

    // 引擎按「模型文件路径」缓存，避免每次识别都重新装载 21MB 模型
    var engine by remember { mutableStateOf<OcrEngine?>(null) }
    DisposableEffect(Unit) {
        onDispose {
            runCatching { engine?.close() }
            engine = null
        }
    }

    fun runOcr(bmp: Bitmap) {
        val files = VisionModels.resolve(ctx, wantVariant)
        if (files == null) {
            log("模型不全，先放 ${wantVariant.detName} 与 ${wantVariant.recName}（files/models/ocr/）")
            return
        }
        busy = true
        scope.launch {
            val result = withContext(Dispatchers.IO) {
                runCatching {
                    val e = engine ?: OcrEngine(ctx.applicationContext, files.det, files.rec).also { engine = it }
                    val t0 = System.currentTimeMillis()
                    val list = e.recognize(bmp)
                    list to (System.currentTimeMillis() - t0)
                }
            }
            busy = false
            result.onSuccess { (list, ms) ->
                blocks = list
                lastMs = ms
                log("识别完成：${list.size} 块，${ms}ms")
            }.onFailure {
                log("识别失败：${it.javaClass.simpleName}: ${it.message}")
            }
        }
    }

    val picker = rememberLauncherForActivityResult(ActivityResultContracts.PickVisualMedia()) { uri ->
        if (uri == null) {
            log("没有选图")
            return@rememberLauncherForActivityResult
        }
        scope.launch {
            val bmp = withContext(Dispatchers.IO) { decodeBounded(ctx, uri, 4096) }
            if (bmp == null) {
                log("读图失败：$uri")
                return@launch
            }
            bitmap = bmp
            blocks = emptyList()
            log("已选图 ${bmp.width}x${bmp.height}")
            runOcr(bmp)
        }
    }

    // ------------------------------------------------------------ 实时抓屏（V2）
    // MediaProjection 的授权 token 不能持久化、进程重启就作废，所以「开始抓屏」只能
    // 从界面发起：弹系统授权 → 拿到 code/data → 交给前台服务建 VirtualDisplay。
    val visionCtx = ctx
    val captureLauncher = rememberLauncherForActivityResult(
        ActivityResultContracts.StartActivityForResult(),
    ) { res ->
        val data = res.data
        if (res.resultCode != android.app.Activity.RESULT_OK || data == null) {
            log("没授权屏幕捕获，抓屏没起来")
            return@rememberLauncherForActivityResult
        }
        // 互斥：一个 MediaProjection 只能建一个 VirtualDisplay，先把音频采集停掉。
        runCatching {
            visionCtx.startService(
                android.content.Intent(visionCtx, com.screenast.app.capture.AudioCaptureService::class.java)
                    .setAction(com.screenast.app.capture.AudioCaptureService.ACTION_STOP),
            )
        }
        val i = android.content.Intent(visionCtx, VisionCaptureService::class.java)
            .setAction(VisionCaptureService.ACTION_START)
            .putExtra(VisionCaptureService.EXTRA_RESULT_CODE, res.resultCode)
            .putExtra(VisionCaptureService.EXTRA_RESULT_DATA, data)
            .putExtra(
                VisionCaptureService.EXTRA_INTERVAL_MS,
                // 0 也要如实传下去：它表示「手动模式」，由悬浮按钮点一次识别一次。
                settings.visionIntervalMs.toLong(),
            )
        androidx.core.content.ContextCompat.startForegroundService(visionCtx, i)
        log(
            if (settings.visionIntervalMs > 0) "已请求抓屏（每 ${settings.visionIntervalMs}ms 自动识别一次）"
            else "已请求抓屏（手动模式：屏幕上会出现一个悬浮按钮，点一次识别一次）",
        )
    }

    fun startCapture() {
        if (!VisionModels.installed(ctx, wantVariant)) {
            log("先放模型文件：files/models/ocr/")
            return
        }
        val mpm = ctx.getSystemService(android.media.projection.MediaProjectionManager::class.java)
        captureLauncher.launch(mpm.createScreenCaptureIntent())
    }

    Column(
        Modifier.fillMaxSize().statusBarsPadding()
            .verticalScroll(rememberScrollState()).padding(14.dp),
        verticalArrangement = Arrangement.spacedBy(10.dp),
    ) {
        // ---------------------------------------------------------------- 标题
        Row(Modifier.fillMaxWidth(), verticalAlignment = Alignment.CenterVertically) {
            Column(Modifier.weight(1f)) {
                Text(
                    if (showSettings) "识图设置" else "识图",
                    style = MaterialTheme.typography.headlineMedium, fontWeight = FontWeight.Bold,
                )
                Text(
                    if (showSettings) "识别 / 贴回 / 浮窗；只影响识图，与同传无关"
                    else "屏幕文字 → 识别 → 翻译 → 贴回原处",
                    style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                )
            }
            if (!showSettings) {
                Text("V2 · 抓屏", fontSize = 12.sp, color = Color(0xFF9C8CF5))
            }
            TextButton(onClick = { showSettings = !showSettings }) {
                Text(if (showSettings) "← 返回" else "设置 ⚙", fontSize = 12.sp)
            }
        }

        if (!showSettings) {

        // ---------------------------------------------------------------- 引擎状态
        SectionCard("识别引擎") {
            InfoRow("识别模型", VisionModels.MODEL_LABEL)
            InfoRow("模型状态", status, good = installed)
            InfoRow(
                "贴回方式",
                Settings.VISION_ERASE_MODES.firstOrNull { it.first == settings.visionEraseMode }?.second
                    ?: "贴片",
            )
            Text(
                if (installed) {
                    "模型已就位。选一张截图即可跑完整 OCR（det 出框 → rec 出字）。"
                } else {
                    "还没有模型文件：把 ${wantVariant.detName} 与 ${wantVariant.recName} " +
                        "放进 files/models/ocr/（${wantVariant.label}）。"
                },
                fontSize = 12.sp,
                color = MaterialTheme.colorScheme.onSurfaceVariant,
            )
            if (!installed) {
                OutlinedButton(
                    onClick = { installed = VisionModels.installed(ctx, wantVariant) },
                    shape = RoundedCornerShape(50),
                ) { Text("重新检查", fontSize = 12.sp) }
            }
        }

        // ------------------------------------------------------------ 实时抓屏 V2
        SectionCard("实时抓屏（V2）") {
            val capturing = VisionSession.capturing.value
            InfoRow(
                "状态",
                if (capturing) "抓屏中" else "未抓屏",
                good = capturing,
            )
            val vs = VisionSession.status.value
            if (vs.isNotEmpty()) {
                Text(vs, fontSize = 11.sp, fontFamily = FontFamily.Monospace, color = Color(0xFF8A87A0))
            }
            val verr = VisionSession.error.value
            if (verr.isNotEmpty()) {
                Text(verr, fontSize = 12.sp, color = Color(0xFFE06B6B))
            }
            Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                Button(
                    onClick = {
                        if (capturing) {
                            VisionCaptureService.stop(ctx)
                            log("已请求停止抓屏")
                        } else {
                            startCapture()
                        }
                    },
                    shape = RoundedCornerShape(50),
                    colors = ButtonDefaults.buttonColors(
                        containerColor = if (capturing) Color(0xFF6E6B80) else Color(0xFF9C8CF5),
                    ),
                ) { Text(if (capturing) "停止抓屏" else "开始抓屏") }
                OutlinedButton(
                    onClick = {
                        VisionSession.clear()
                        log("已清空识图结果")
                    },
                    shape = RoundedCornerShape(50),
                ) { Text("清空结果") }
            }
            SettingHint(
                if (settings.visionIntervalMs > 0) {
                    "按「自动抓帧」的节奏取帧（每 ${settings.visionIntervalMs}ms），画面没变就不重复识别。"
                } else {
                    "自动抓帧关着 = 手动模式：抓屏期间屏幕上有个「暂停键」小按钮，" +
                        "点一次识别当前画面并把译文贴回屏幕（8 秒后自动消失）。"
                } +
                    "抓屏期间音频采集会自动停 —— 一个投屏只能建一个虚拟屏。",
                fontSize = 11.sp, color = Color(0xFF8A87A0),
            )
        }

        // ---------------------------------------------------------------- 预览
        SectionCard("画面预览") {
            val bmp = bitmap
            Box(
                Modifier.fillMaxWidth().heightIn(min = 120.dp, max = 300.dp)
                    .background(Color(0xFF141320), RoundedCornerShape(10.dp)),
                contentAlignment = Alignment.Center,
            ) {
                if (bmp != null) {
                    Image(
                        bitmap = bmp.asImageBitmap(),
                        contentDescription = "待识别画面",
                        contentScale = ContentScale.Fit,
                        modifier = Modifier.fillMaxWidth().heightIn(max = 300.dp),
                    )
                } else {
                    Column(horizontalAlignment = Alignment.CenterHorizontally) {
                        Text("还没有选图", fontSize = 13.sp, color = Color(0xFF8A87A0))
                        Spacer(Modifier.height(4.dp))
                        Text("V1 用系统图片选择器；V2 换成实时抓屏", fontSize = 11.sp, color = Color(0xFF6E6B80))
                    }
                }
                if (busy) CircularProgressIndicator(Modifier.width(28.dp).height(28.dp), strokeWidth = 2.dp)
            }
            Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                Button(
                    onClick = {
                        if (!VisionModels.installed(ctx, wantVariant)) {
                            log("先放模型文件：files/models/ocr/")
                            return@Button
                        }
                        picker.launch(PickVisualMediaRequest(ActivityResultContracts.PickVisualMedia.ImageOnly))
                    },
                    shape = RoundedCornerShape(50),
                    enabled = !busy,
                    colors = ButtonDefaults.buttonColors(containerColor = Color(0xFF9C8CF5)),
                ) { Text("选一张图") }
                OutlinedButton(
                    onClick = {
                        val b = bitmap
                        if (b != null) runOcr(b) else log("还没有选图")
                    },
                    shape = RoundedCornerShape(50),
                    enabled = !busy && bitmap != null,
                ) { Text("重新识别") }
                OutlinedButton(
                    onClick = {
                        bitmap = null
                        blocks = emptyList()
                        lastMs = 0
                    },
                    shape = RoundedCornerShape(50),
                ) { Text("清空") }
            }
        }

        // ---------------------------------------------------------------- 结果
        val live = VisionSession.lines.value
        val capturing = VisionSession.capturing.value
        val shown = if (capturing || live.isNotEmpty()) live.size else blocks.size
        SectionCard(
            when {
                shown == 0 -> "识别结果"
                capturing || live.isNotEmpty() ->
                    "识别结果（抓屏 · ${live.size} 块 · ${VisionSession.frameSize.value.first}×${VisionSession.frameSize.value.second}）"
                else -> "识别结果（${blocks.size} 块 · ${bitmap?.width ?: 0}×${bitmap?.height ?: 0} · ${lastMs}ms）"
            },
        ) {
            if (shown == 0) {
                Text("这里会出现每块文字的原文、坐标（x, y, w×h）与置信度", fontSize = 12.sp, color = Color(0xFF8A87A0))
            } else if (capturing || live.isNotEmpty()) {
                // 抓屏模式：每块「原文 + 译文」，坐标仍是原图像素，V3 贴回直接复用。
                live.forEach { b ->
                    Column(Modifier.fillMaxWidth()) {
                        Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.spacedBy(6.dp)) {
                            Text(
                                "%.0f,%.0f %.0f×%.0f".format(b.left, b.top, b.width, b.height),
                                fontSize = 10.sp, fontFamily = FontFamily.Monospace,
                                color = Color(0xFF6E6B80), modifier = Modifier.width(104.dp),
                            )
                            if (b.vertical) {
                                Text("竖", fontSize = 10.sp, color = Color(0xFF9C8CF5), modifier = Modifier.width(13.dp))
                            }
                            Text(b.text, fontSize = 13.sp, modifier = Modifier.weight(1f))
                            Text("%.2f".format(b.score), fontSize = 10.sp, color = Color(0xFF7CE0A3))
                        }
                        if (b.translation.isNotEmpty()) {
                            Text(
                                b.translation,
                                fontSize = 13.sp,
                                color = Color(0xFF9C8CF5),
                                modifier = Modifier.padding(start = 104.dp, top = 2.dp),
                            )
                        }
                    }
                }
            } else {
                blocks.forEach { b ->
                    Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.spacedBy(6.dp)) {
                        Text(
                            "%.0f,%.0f %.0f×%.0f".format(b.left, b.top, b.width, b.height),
                            fontSize = 10.sp, fontFamily = FontFamily.Monospace,
                            color = Color(0xFF6E6B80), modifier = Modifier.width(104.dp),
                        )
                        if (b.vertical) {
                            Text("竖", fontSize = 10.sp, color = Color(0xFF9C8CF5), modifier = Modifier.width(13.dp))
                        }
                        Text(b.text, fontSize = 13.sp, modifier = Modifier.weight(1f))
                        Text("%.2f".format(b.score), fontSize = 10.sp, color = Color(0xFF7CE0A3))
                    }
                }
                Text(
                    "译文与贴回是 V2/V3；现在「翻译译文」开关只保存状态，不参与 OCR。",
                    fontSize = 11.sp, color = Color(0xFF8A87A0),
                )
            }
        }

        }   // if (!showSettings)

        if (showSettings) {
            VisionSettingsSection(
                settings = settings,
                onChange = onChange,
                onLog = { log(it) },
            )
        }

        // 用户 m06832：「识图里的进度可以删了，那个开发进度用户不用看」——
        // V0..V3 那份自测清单只对开发者有意义，删掉；留下真正会影响使用的两条：
        // MediaProjection 互斥提醒 + 出错时的本页日志。
        if (showSettings) return@Column
        // 系统限制这类话要一直看得见，别塞进可折叠的卡片里（否则只剩一个空的「ⓘ 说明」）。
        Text(
            "⚠ 一个 MediaProjection 只能一个 VirtualDisplay：同传采集与识图不能同时进行，" +
                "切换时要先停一个。这是系统限制，绕不过去。",
            fontSize = 11.sp, color = Color(0xFFE0A36B),
        )
        if (lines.isNotEmpty()) {
            SectionCard("本页日志") {
                lines.forEach { Text(it, fontSize = 11.sp, color = Color(0xFFB9B6C8)) }
            }
        }

        Spacer(Modifier.height(8.dp))
    }
}

/**
 * 识图设置（从识图页搬进「同传 → 设置」，用户 m06633）。
 *
 * 之所以整块搬过来：识图页原来下半屏全是开关，抓屏结果被挤到最上面；
 * 而这些项在设置页和翻译后端、悬浮窗字号是同一类「先配好再干活」的东西。
 */
@Composable
fun VisionSettingsSection(
    settings: Settings,
    onChange: (Settings) -> Unit,
    onLog: (String) -> Unit,
) {
    val ctx = LocalContext.current
    val log: (String) -> Unit = onLog

    SectionCard("识图（识别 / 贴回 / 浮窗）") {
            Text("译成", fontSize = 13.sp)
            Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                Lang.concrete.forEach { l ->
                    Chip(l.label, settings.visionTargetLang == l) {
                        onChange(settings.copy(visionTargetLang = l))
                        log("译成 → ${l.label}")
                    }
                }
            }

            Text("自动抓帧", fontSize = 13.sp)
            Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                listOf(0 to "关（手动）", 2000 to "2 秒", 3000 to "3 秒", 5000 to "5 秒").forEach { (ms, label) ->
                    Chip(label, settings.visionIntervalMs == ms) {
                        onChange(settings.copy(visionIntervalMs = ms))
                        log(if (ms == 0) "自动抓帧 → 关（屏幕上用悬浮按钮手动识别）" else "自动抓帧 → $label")
                    }
                }
            }

            Row(Modifier.fillMaxWidth(), verticalAlignment = Alignment.CenterVertically) {
                Column(Modifier.weight(1f)) {
                    Text("翻译译文", fontSize = 13.sp)
                    Text(
                        "关掉就只做 OCR，不调用翻译后端（省电省显存）",
                        fontSize = 11.sp, color = Color(0xFF8A87A0),
                    )
                }
                Switch(
                    checked = settings.visionTranslate,
                    onCheckedChange = {
                        onChange(settings.copy(visionTranslate = it))
                        log("翻译译文 → ${if (it) "开" else "关"}（V2 起生效）")
                    },
                )
            }

            // 真机实测（2026-10-18）：识图页 + 端侧 NLLB-600M 一起跑时，NLLB 装载要
            // **73 秒**、进程 RSS 冲到 5.2GB、native swap 787MB —— 机器被换页拖死，
            // 4 行字翻译 45 秒还没出结果。OCR 只占 21MB，问题在 NLLB 这 1.27GB。
            // 这里只提示，不替用户改设置：想用 NLLB 是他的自由，但要知道代价。
            if (settings.visionTranslate && settings.providerId == "nllb" && !settings.netApiEnabled) {
                SettingHint(
                    "⚠ 当前后端是端侧 NLLB：它要和 OCR 同时驻留内存（实测装载 73 秒、RSS 5GB），" +
                        "识图页会明显卡。建议在「同传 → 设置 → 翻译后端」换成「端侧大模型」" +
                        "或打开「网络API调用」（每句一次 HTTP，最轻）。",
                    fontSize = 11.sp, color = Color(0xFFE0A64A),
                )
            }

            Text("贴回方式", fontSize = 13.sp)
            Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                Settings.VISION_ERASE_MODES.forEach { (mode, _) ->
                    val label = when (mode) {
                        0 -> "贴片"
                        1 -> "模糊"
                        else -> "擦除"
                    }
                    Chip(label, settings.visionEraseMode == mode) {
                        onChange(settings.copy(visionEraseMode = mode))
                        log(
                            "贴回 → " + when (mode) {
                                0 -> "贴片（同色块）"
                                1 -> "模糊（原文区域糊掉垫底）"
                                else -> "擦除（MI-GAN 补背景）"
                            },
                        )
                    }
                }
            }
            Text(
                Settings.VISION_ERASE_MODES.firstOrNull { it.first == settings.visionEraseMode }?.second ?: "",
                fontSize = 11.sp, color = Color(0xFF8A87A0),
            )

            // 译文颜色：0 = 智能（按底色 R+G+B 加和切黑白），其余是用户自己挑的。
            Text("译文颜色", fontSize = 13.sp)
            Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                val textColors = listOf(
                    "智能" to 0,
                    "白" to 0xFFFFFFFF.toInt(),
                    "黑" to 0xFF14121A.toInt(),
                    "黄" to 0xFFFFD54A.toInt(),
                )
                textColors.forEach { (label, argb) ->
                    Chip(label, settings.visionTextColor == argb) {
                        onChange(settings.copy(visionTextColor = argb))
                        log(
                            "译文颜色 → " + if (argb == 0) {
                                "智能（底亮黑字 / 底暗白字）"
                            } else {
                                label
                            },
                        )
                    }
                }
            }
            Text(
                if (settings.visionTextColor == 0) {
                    "智能：底色 R+G+B 加起来 > 382.5 用黑字，否则白字"
                } else {
                    "固定颜色（不随底色变化，浅底上可能看不清）"
                },
                fontSize = 11.sp, color = Color(0xFF8A87A0),
            )

            if (settings.visionEraseMode == 2) {
                // 模型不在就别给点：以前这里是常开，点了「开」也只会退回程序贴片，
                // 用户以为补背景生效了，其实是纯色块。
                val inpaintOk = remember { VisionInpaint.available(ctx) }
                Row(Modifier.fillMaxWidth(), verticalAlignment = Alignment.CenterVertically) {
                    Column(Modifier.weight(1f)) {
                        Text(
                            if (inpaintOk) "自动补背景（MI-GAN）" else "自动补背景（MI-GAN·模型未导入）",
                            fontSize = 13.sp,
                            color = if (inpaintOk) Color.Unspecified else Color(0xFF8A87A0),
                        )
                        SettingHint(
                            "开：模型把原字抹掉再按纹理补全（约 27MB 模型，一次 1~2 秒）。" +
                                "关：不加载模型，直接用框四周的平均色盖住原文再写译文，几毫秒，" +
                                "纯色/渐变背景几乎看不出来，照片背景会有点糊。",
                            fontSize = 11.sp, color = Color(0xFF8A87A0),
                        )
                    }
                    Switch(
                        checked = settings.visionInpaint && inpaintOk,
                        enabled = inpaintOk,
                        onCheckedChange = {
                            onChange(settings.copy(visionInpaint = it))
                            log("补背景 → ${if (it) "MI-GAN" else "关（程序贴片）"}")
                        },
                    )
                }
            }

            SettingDivider()
            // ---------------------------------------------------------- 识别权重
            // 手机屏幕常是 1440 宽、正文只有 20~30px 高，整屏缩到 960 长边后小字只剩几个像素，
            // 于是冒出「6175」「ln」「SV」这类噪声块。server 权重容量大、最长边放 1280，
            // 同一张截图能多检出一批小字（PC 实测 28 框 → 41 框），代价是一帧慢 3~4 秒。
            Text("识别权重", fontSize = 13.sp)
            Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                Chip("轻量 mobile", !settings.visionOcrServer) {
                    onChange(settings.copy(visionOcrServer = false))
                    log("识别权重 → 轻量 mobile（21MB，一帧约 1 秒）")
                }
                Chip("高精度 server", settings.visionOcrServer) {
                    onChange(settings.copy(visionOcrServer = true))
                    log("识别权重 → 高精度 server（164MB，一帧 3~5 秒）")
                }
            }
            val wantVariant = if (settings.visionOcrServer) {
                VisionModels.Variant.SERVER
            } else {
                VisionModels.Variant.MOBILE
            }
            SettingHint(
                "server 权重体积大（det 84MB + rec 80MB），手机上没有时会自动退回 mobile，" +
                    "上面「模型状态」会显示退回原因。实际装着：" +
                    VisionModels.installedVariants(ctx).joinToString(" / ") { it.label }
                        .ifEmpty { "无" } +
                    "；当前要的是 " + wantVariant.label + "。",
                fontSize = 11.sp, color = Color(0xFF8A87A0),
            )

            SettingDivider()
            // ---------------------------------------------------------- 短文本过滤
            // 用户 m04363：长度小于 n 个字符（含符号）的块视为无效，识别后直接剔除，
            // 不参与翻译/不贴回。噪声块（`ln`/`SV`/`6175`）大多 1~4 个字符。
            Spacer(Modifier.height(6.dp))
            Text("忽略少于 N 个字符的文本", fontSize = 13.sp)
            Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                (0..5).forEach { n ->
                    Chip(if (n == 0) "0（不过滤）" else "$n", settings.visionMinChars == n) {
                        onChange(settings.copy(visionMinChars = n))
                        log(
                            if (n == 0) "短文本过滤 → 关（全部保留）"
                            else "短文本过滤 → 少于 $n 个字符的块直接剔除"
                        )
                    }
                }
            }
            SettingHint(
                "标点也算一个字符；比较的是去空格后的长度。默认 2：单字符基本只有噪声，" +
                    "`OK`、`3`、`？` 这类两字符的仍会保留。噪声多就往上调，有短词被误杀就调小。",
                fontSize = 11.sp, color = Color(0xFF8A87A0),
            )

            SettingDivider()
            // ---------------------------------------------------------- 悬浮文字列表
            // 用户 m04158：把同传那个浮窗带过来，做成可开关 —— 屏幕上字太小、贴片也放不下时，
            // 同一份原文 + 译文在浮窗里能调字号、能滚，是「看不清」的兜底路径。
            Spacer(Modifier.height(6.dp))
            Row(Modifier.fillMaxWidth(), verticalAlignment = Alignment.CenterVertically) {
                Column(Modifier.weight(1f)) {
                    Text("悬浮文字列表（同传那个浮窗）", fontSize = 13.sp)
                    SettingHint(
                        "开：每次识别完，这一帧所有文本（原文 + 译文）按从上到下列进悬浮窗，" +
                            "拖顶部条移动、拖右下角 ◢ 改大小；字号在「同传 → 设置」里调。" +
                            "关：只贴回屏幕，不开浮窗。",
                        fontSize = 11.sp, color = Color(0xFF8A87A0),
                    )
                }
                Switch(
                    checked = settings.visionFloatList,
                    onCheckedChange = { on ->
                        // 开之前先确认权限：没权限时 View 加不上去（BadTokenException），
                        // 不如直接跳授权页，别让用户开了开关却什么都没有。
                        if (on && !VisionFloatList.canShow(ctx)) {
                            log("浮窗需要「显示在其他应用上层」权限，正在跳转系统设置…")
                            val jumped = runCatching {
                                ctx.startActivity(
                                    Intent(
                                        SysSettings.ACTION_MANAGE_OVERLAY_PERMISSION,
                                        Uri.parse("package:${ctx.packageName}"),
                                    ).addFlags(Intent.FLAG_ACTIVITY_NEW_TASK)
                                )
                            }.isSuccess
                            if (!jumped) {
                                log("❌ 打不开授权页，请手动到「设置 → 应用管理 → 屏幕同传 → 其他权限 → 显示在其他应用上层」开启")
                            }
                            return@Switch
                        }
                        onChange(settings.copy(visionFloatList = on))
                        if (on) {
                            // 立刻把窗口挂上去（不等识别）：用户开关一开就该看见框，否则会以为坏了。
                            val ok = VisionFloatList.show(ctx, settings.copy(visionFloatList = true))
                            log(
                                if (ok) "悬浮文字列表 → 开（框已常驻显示；识别结果会一条条累积进去）"
                                else "❌ 浮窗没挂上：${VisionFloatList.lastError() ?: "未知原因"}"
                            )
                        } else {
                            VisionFloatList.hide()
                            log("悬浮文字列表 → 关")
                        }
                    },
                )
            }
    }
}

/** 按最长边限制解码，避免 4K 截图直接把 Dalvik 堆吃满。 */
private fun decodeBounded(ctx: Context, uri: Uri, maxSide: Int): Bitmap? {
    val bounds = BitmapFactory.Options().apply { inJustDecodeBounds = true }
    ctx.contentResolver.openInputStream(uri)?.use { BitmapFactory.decodeStream(it, null, bounds) }
    if (bounds.outWidth <= 0 || bounds.outHeight <= 0) return null
    var sample = 1
    while (maxOf(bounds.outWidth, bounds.outHeight) / sample > maxSide) sample *= 2
    val opts = BitmapFactory.Options().apply { inSampleSize = sample }
    return ctx.contentResolver.openInputStream(uri)?.use { BitmapFactory.decodeStream(it, null, opts) }
}

@Composable
private fun SectionCard(title: String, content: @Composable () -> Unit) {
    Card(
        Modifier.fillMaxWidth(),
        shape = RoundedCornerShape(12.dp),
        colors = CardDefaults.cardColors(containerColor = Color(0xFF1B1A26)),
    ) {
        Column(Modifier.padding(12.dp), verticalArrangement = Arrangement.spacedBy(8.dp)) {
            Text(title, fontSize = 14.sp, fontWeight = FontWeight.SemiBold)
            content()
        }
    }
}

@Composable
private fun InfoRow(label: String, value: String, good: Boolean? = null) {
    Row(Modifier.fillMaxWidth(), verticalAlignment = Alignment.CenterVertically) {
        // 固定标签宽度：不固定的话「识别模型」会被右侧的长值挤成两行（真机截图确认过）。
        Text(label, fontSize = 12.sp, color = Color(0xFF8A87A0), modifier = Modifier.width(72.dp))
        Text(
            value,
            fontSize = 12.sp,
            textAlign = TextAlign.End,
            modifier = Modifier.weight(1f),
            color = when (good) {
                true -> Color(0xFF7CE0A3)
                false -> Color(0xFFE0A36B)
                null -> MaterialTheme.colorScheme.onSurface
            },
        )
    }
}

@Composable
private fun Chip(text: String, selected: Boolean, onClick: () -> Unit) {
    if (selected) {
        Button(
            onClick = onClick,
            shape = RoundedCornerShape(50),
            colors = ButtonDefaults.buttonColors(containerColor = Color(0xFF9C8CF5)),
        ) { Text(text, fontSize = 12.sp) }
    } else {
        OutlinedButton(onClick = onClick, shape = RoundedCornerShape(50)) {
            Text(text, fontSize = 12.sp)
        }
    }
}

@Composable
private fun PlanRow(tag: String, text: String) {
    Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.spacedBy(8.dp)) {
        Text(tag, fontSize = 12.sp, fontWeight = FontWeight.Bold, color = Color(0xFF9C8CF5))
        Text(text, fontSize = 12.sp, color = Color(0xFFB9B6C8))
    }
}
