package com.screenast.app

import android.Manifest
import android.content.BroadcastReceiver
import android.content.Context
import android.content.Intent
import android.content.IntentFilter
import android.content.pm.PackageManager
import android.media.projection.MediaProjectionManager
import android.net.Uri
import android.os.Build
import android.os.Bundle
import android.os.SystemClock
import android.provider.Settings as SysSettings
import androidx.activity.ComponentActivity
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.compose.setContent
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.horizontalScroll
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.statusBarsPadding
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.heightIn
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.AlertDialog
import androidx.compose.material3.Button
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.DropdownMenu
import androidx.compose.material3.DropdownMenuItem
import androidx.compose.material3.LinearProgressIndicator
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.NavigationBar
import androidx.compose.material3.NavigationBarItem
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Slider
import androidx.compose.material3.Surface
import androidx.compose.material3.Switch
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.material3.darkColorScheme
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.MutableState
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableIntStateOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.core.content.ContextCompat
import com.screenast.app.asr.AsrFactory
import com.screenast.app.asr.DictationHub
import com.screenast.app.asr.SenseVoiceAsr
import com.screenast.app.capture.AudioCaptureService
import com.screenast.app.core.BundledModels
import com.screenast.app.core.Lang
import com.screenast.app.core.LowMemWarning
import com.screenast.app.core.MemProbe
import com.screenast.app.core.ModelDownloader
import com.screenast.app.core.ModelSmoke
import com.screenast.app.core.ModelImporter
import com.screenast.app.core.ModelUsage
import com.screenast.app.core.Settings
import com.screenast.app.core.SettingDivider
import com.screenast.app.core.SettingHint
import com.screenast.app.core.SettingsStore
import com.screenast.app.core.currentSettings
import com.screenast.app.interpret.InterpretEngine
import com.screenast.app.vision.VisionModels
import com.screenast.app.vision.VisionScreen
import com.screenast.app.interpret.InterpretSession
import com.screenast.app.overlay.SubtitleOverlay
import com.screenast.app.translate.OpenAiCompatibleProvider
import com.screenast.app.translate.ProviderFactory
import com.screenast.app.translate.ProviderStatus
import com.screenast.app.translate.TranslateRequest
import com.screenast.app.translate.TranslationProvider
import com.screenast.app.translate.llama.LlamaModels
import com.screenast.app.translate.llama.LlamaRuntime
import com.screenast.app.translate.llama.LlamaTranslationProvider
import com.screenast.app.translate.nllb.NllbModels
import com.screenast.app.translate.nllb.NllbTokenizer
import com.screenast.app.translate.nllb.NllbTranslationProvider
import com.screenast.app.translate.nllb.NllbTranslator
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.collect
import kotlinx.coroutines.flow.first
import kotlinx.coroutines.Job
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import androidx.lifecycle.lifecycleScope
import java.io.File
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale
import java.util.concurrent.atomic.AtomicBoolean
import java.util.concurrent.atomic.AtomicInteger

/**
 * **实时同传台** —— 路线图 ③ 的落点：把识别和翻译两个模型串起来，出中文字幕。
 *
 * ## 界面为什么长这样
 *
 * 与已删除的「听写调试台」（诊断仪表盘）不同，这个界面的第一优先是**读得清**：
 * 字幕占屏幕上半部，译文 26sp、原文 14sp 灰色，扫一眼就知道「现在在说什么」。
 * 统计数字全部压到下面 —— 它们只在出问题时才需要看。
 *
 * 但「当前句」和「历史句」必须**分开**，这不是排版偏好：
 * 翻译落后于识别时（队列里压着两句），如果把迟到的译文写进「当前句」，
 * 屏幕上就会出现「时间轴错位」的字幕 —— 观众看到的译文和正在说的话对不上，
 * 而这是同传最不能犯的错。所以迟到的译文一律落到历史区，当前句只接受
 * **id 与最新一句相等**的译文（见 [onTranslation] 的处理）。
 *
 * ## 状态纪律（和当初的听写调试台同源，这里再重申一次）
 *
 * 所有界面状态都是 Activity 字段上的 `mutableStateOf(...)` **对象**，
 * 不用 `by` 委托、不放组合局部变量、不依赖 `remember`。
 * 理由：采集线程和翻译线程都要写这些值，而组合局部变量的生命周期只覆盖一次组合。
 * 漏 `remember` 的经典症状是「日志一路成功、界面一路空白」，
 * 且 `identityHashCode` 照样恒定 —— 唯一可靠判据是在组合里读 `.length` 看是否推进。
 */
class InterpretActivity : ComponentActivity() {

    companion object {
        private const val TAG = "AST.Interp"

        /** auto 流程防重入。理由同听写调试台：`am start` 会先 onCreate 再 onNewIntent。 */
        private val autoFired = AtomicBoolean(false)
        private val instanceSeq = AtomicInteger(0)

        /**
         * 屏幕录制授权请求码。
         *
         * 用老式 `startActivityForResult` 而不是 Compose 的 `rememberLauncherForActivityResult`：
         * 自动流程（`auto=interp`）在 onCreate 里就要发起授权，那时候还没有组合作用域，
         * 拿不到 launcher。界面上的按钮走 Compose launcher，两条路径殊途同归。
         */
        private const val REQ_PROJECTION = 4001

        /** `Settings.providerId` → 界面标签。列表顺序就是按钮顺序。 */
        private val PROVIDERS = listOf(
            "llama" to "端侧大模型",
            "nllb" to "端侧 NLLB",
            "openai" to "云端",
        )

        /**
         * `Settings.llamaDevice` → 界面标签。
         * NPU 排第一因为实测就是它快（首字 58~107 ms，CPU 是 980~1060 ms）。
         */
        private val DEVICES = listOf(
            "auto" to "自动",
            "HTP0" to "NPU",
            "CPU" to "CPU",
        )

        /**
         * 浮窗字色候选，ARGB。排第一的是默认白。
         *
         * 为什么是固定几个色块而不是取色盘：用户原话「只需要一个最简单的半透明框」，
         * 而字幕要的是**在亮画面和暗画面上都读得清**，白/黄/青/绿正好覆盖这个需求，
         * 颜色再多只是增加选择成本。
         */
        private val TEXT_COLORS = listOf(
            "白" to 0xFFFFFFFF.toInt(),
            "黄" to 0xFFFFEB3B.toInt(),
            "青" to 0xFF4DD0E1.toInt(),
            "绿" to 0xFF81C784.toInt(),
        )

        /**
         * 框底色候选，ARGB（这里的 alpha 会被忽略）。
         *
         * 原始需求（m00009）写的是「框大小**颜色**都可调」，所以底色也要能选；
         * 但**透明度是独立的滑块**，选颜色时只换 RGB、保留当前 alpha —— 否则
         * 选一次颜色就会把用户调好的不透明度重置掉。
         */
        private val BOX_COLORS = listOf(
            "黑" to 0xFF000000.toInt(),
            "深灰" to 0xFF2A2A2E.toInt(),
            "藏青" to 0xFF101A33.toInt(),
            "墨绿" to 0xFF0F2A22.toInt(),
        )

        /** 浮窗显示内容。id 与 [Settings.subtitleMode] 的取值一一对应。 */
        private val SUBTITLE_MODES = listOf(
            "both" to "双语",
            "target" to "仅译文",
            "source" to "仅原文",
        )
    }

    private val instanceId = instanceSeq.incrementAndGet()

    /**
     * 引擎**不再归本 Activity 所有**（线路图 ⑥）：它现在活在 [InterpretSession] 里。
     *
     * 以前 `onDestroy` 里一句 `engine.close()` 就把一场同传结束了 —— 用户切去看视频、
     * 界面被系统回收，字幕就断。现在这里只是「借用」：界面销毁不影响它继续跑，
     * 重建后读同一批状态就能续上画面。要真正停止请走 [InterpretSession.stopAll]。
     */
    private val engine: InterpretEngine? get() = InterpretSession.engine

    private lateinit var prefs: android.content.SharedPreferences

    /**
     * 设置快照。
     *
     * 为什么要提前读、不现用现读：DataStore 的读取是挂起的，而
     * `translatorFactory` 是**同步**的 `() -> TranslationProvider`（它由翻译线程调用，
     * 类型定义在 `InterpretEngine.kt:184`，改起来要动编排层）。所以在 `onCreate` 里
     * 后台读一次存进这个字段，两个工厂方法就能同步取用。
     *
     * 默认值是 `Settings()` —— 里面 `providerId = "nllb"`，所以即使读得慢，
     * 用户点开翻译拿到的也是**已经验证过的 NLLB**，不会因为读不到设置而崩。
     *
     * ## 为什么包了一层 `mutableStateOf`
     *
     * 加「识图」页之后，设置不再只被 `Screen()` 里的局部状态镜像，而是**两个页面共用**：
     * 识图页改「译成/自动抓帧/贴回方式」也要立刻反映到界面。原来 `@Volatile` 字段
     * 对 Compose 是不可见的（改了不会重组），所以换成快照状态；写入都发生在主线程
     * （onCreate 的回调、onResume、按钮手势），读取允许来自翻译线程。
     */
    private val appSettingsState = mutableStateOf(Settings())

    private var appSettings: Settings
        get() = appSettingsState.value
        set(v) {
            appSettingsState.value = v
        }

    // ---- 采集状态广播（协议与主界面/听写台共用）----
    private val statusReceiver = object : BroadcastReceiver() {
        override fun onReceive(context: Context?, intent: Intent?) {
            val i = intent ?: return
            uiLevel.value = i.getFloatExtra(AudioCaptureService.EXTRA_LEVEL, 0f)
            uiDbfs.value = i.getFloatExtra(AudioCaptureService.EXTRA_DBFS, -120f)
            uiCaptureRunning.value = i.getBooleanExtra(AudioCaptureService.EXTRA_RUNNING, false)
        }
    }

    // ------------------------------------------------------------------ UI 状态
    private val uiLevel = mutableStateOf(0f)
    private val uiDbfs = mutableStateOf(-120f)
    private val uiCaptureRunning = mutableStateOf(false)
    // 输出状态全部放在 [InterpretSession]（引擎活在那边，界面只是读它）。这里只做转发，
    // 好处是：界面被回收重建后，这些值还在，画面能直接续上。

    private val uiSpeaking: MutableState<Boolean> get() = InterpretSession.speaking

    /** 当前句原文 / 译文。译文只接受最新 id 的更新（配对逻辑在 Session 的引擎回调里）。 */
    private val uiLiveSource: MutableState<String> get() = InterpretSession.liveSource
    private val uiLiveTarget: MutableState<String> get() = InterpretSession.liveTarget

    /** 历史句（最新的在上）。每行形如「原文 → 译文」。 */
    private val uiHistory: MutableState<String> get() = InterpretSession.history

    private val uiInterpState: MutableState<String> get() = InterpretSession.interpState
    private val uiStats: MutableState<String> get() = InterpretSession.stats
    private val uiError: MutableState<String> get() = InterpretSession.error
    private val uiLog: MutableState<String> get() = InterpretSession.log
    /**
     * 翻译开关。**默认开** —— 这个界面叫「屏幕同传」，用户点「开始采集」的意图就是
     * 「把听到的翻给我看」，让他再点一次开关属于多此一举。
     *
     * 它同时是「翻译到底开没开」的唯一真相：`buildAndAttach` 只在本值为 true 时
     * 才挂翻译后端（见那里的注释），所以按钮文案和实际行为不会再各说各话。
     */
    private val uiTranslateOn: MutableState<Boolean> get() = InterpretSession.translateOn

    /**
     * 正在后台装配引擎（加载 SenseVoice）。界面靠它把按钮变成「准备中…」并挡住重复点击。
     *
     * 为什么需要这个状态：引擎构造要加载 239MB 的 SenseVoice 模型（约 0.9 秒）。
     * 它以前是在按钮的 onClick 里**同步**跑完的，也就是占着主线程 0.9 秒 ——
     * 表现是「按下开始采集，界面定住一下，然后授权框才弹出来」。用户对这种情况的
     * 描述通常是「点了没反应」，然后反复点，于是弹出好几个授权框。
     */
    private val uiStarting = mutableStateOf(false)
    private val uiTargetLang = mutableStateOf(Lang.ZH)
    private val uiFallbackSrc = mutableStateOf<Lang?>(Lang.EN)

    /**
     * 强制切段间隔（秒）。0 = 关闭，只靠 VAD 端点。
     *
     * 改它**不需要**重建引擎：真值在 [appSettings.forceSegmentMs]，引擎每个窗口读一次
     * （见 [InterpretEngine] 的 `forceCutMs`），所以拖动滑条下一句就变。
     * 重建引擎要重新加载 239MB 模型，为了一个时间参数付 0.9 秒不值得。
     */
    private val uiForceSegmentSec = mutableStateOf(4f)

    /**
     * 识别后端标识（见 [AsrFactory.KINDS]）。
     *
     * 和 [uiForceSegmentSec] 不同，**改它必须重建引擎**：模型是在构造时加载进
     * native 内存的，换后端等于换一整个 ONNX 会话。所以这里只是记下「用户想用哪个」，
     * 真正的切换发生在下一次 [buildAndAttach]。界面必须把这件事说清楚，
     * 否则用户会以为点了没反应。
     */
    private val uiAsrModel = mutableStateOf(AsrFactory.SENSEVOICE)

    /** 是否给不自带标点的后端补标点（只对 Paraformer / Whisper 有意义）。 */
    private val uiAsrPunct = mutableStateOf(true)
    private val uiAsrDenoise = mutableStateOf(false)

    /**
     * 带几轮对话上文去翻译（0..10，0 = 无上下文）。
     *
     * 与 [uiForceSegmentSec] 同类：真值在 [appSettings.contextRounds]，
     * 引擎每个任务读一次（`contextRounds` lambda），所以**拖完下一句就生效**，
     * 不需要重建引擎、更不需要重载 1.2GB 的 2B 模型。
     */
    private val uiContextRounds = mutableStateOf(0)

    /**
     * 翻译后端与它的算力位置。
     *
     * 为什么还要这两个 Compose 状态：真值在 [appSettings] 里，但那是个 `@Volatile` 字段，
     * 改了**不会触发重组** —— 界面会停在旧选项上，用户以为没点中，于是反复点。
     */
    private val uiProviderId = mutableStateOf("nllb")
    private val uiLlamaDevice = mutableStateOf("auto")

    // ------------------------------------------------------------ 离线模型导入状态
    //
    // 2026-10-18：这三个状态和两个启动器从 Screen() 的局部 remember 提升成 Activity 字段。
    // 原因：导入界面挪到了底栏「翻译」页，而主界面那条「建议使用」提示卡也要读同一份数字；
    // 各留一份 remember 的话，导入完只有当前页会刷新。

    /** 清单就位情况。初值用清单本身算（不碰文件系统），onCreate 里再扫一次真实值。 */
    private val uiModelStatus = mutableStateOf(
        ModelImporter.Status(
            ready = 0,
            total = ModelImporter.items.size,
            readyBytes = 0L,
            totalBytes = ModelImporter.items.sumOf { it.bytes },
            essentialReady = 0,
            essentialTotal = ModelImporter.items.count { it.essential },
            missingEssential = ModelImporter.items.filter { it.essential },
        ),
    )
    private val uiImportMsg = mutableStateOf("")
    private val uiImporting = mutableStateOf(false)
    private val uiModelHelp = mutableStateOf(false)

    /** 内存快照（主界面那一行要显示「可用 X，需要约 Y」）。 */
    private val uiMem = mutableStateOf(MemProbe.Snapshot(0, 0, 0, 0))
    /** 非空 = 弹出「内存严重不足」警告。 */
    private val uiLowMem = mutableStateOf<LowMemWarning?>(null)
    /** 手填语种（设置里那两栏）；空串 = 用上面选的枚举值。 */
    private val uiCustomTarget = mutableStateOf("")
    private val uiCustomSource = mutableStateOf("")
    /** 正在下载的模型（null = 没在下）。 */
    private val uiDl = mutableStateOf<ModelDownloader.Progress?>(null)
    /** 进度弹窗是否显示（「放到后台」把它关掉，下载继续）。 */
    private val uiDlShow = mutableStateOf(true)

    /**
     * 是否正在下载（用户 2026-10-18：「点掉界面就看不到下载进度了，下载进行时在页面上方
     * 显示一个下载图标，常态隐藏」）。它和 [uiDlShow] 的区别：uiDlShow 只管「进度弹窗开没开」，
     * 这个字段管「有没有下载在跑」——弹窗关掉之后图标还在，点一下就再弹回来。
     */
    private val uiDlActive = mutableStateOf(false)

    // 用户公告：首次启动必弹；之后从「设置 → 用户公告与免责声明」回看。
    // 条款改了就把 Notice.VERSION +1，老用户会被再弹一次（见 Settings.noticeAccepted）。
    private val uiNotice = mutableStateOf(false)

    /**
     * 「API 测试连接」的结果。
     *
     * 用户 m06921：成功/失败只写在日志里太不显眼，要弹窗。这里存一份结构化结果，
     * `AppRoot()` 里统一渲染成对话框（成功与失败都弹，失败的那几行就是原因）。
     */
    private val uiApiTest = mutableStateOf<ApiTestReport?>(null)

    /**
     * 上一次用的「本地」翻译后端（`llama` / `nllb`）。
     *
     * 用户 m06985 要求翻译后端只留两档：「本地模型」「API 调用」，
     * NLLB 与大模型改由「翻译模型」下拉里选。于是外壳只剩这两个 chip，
     * 从「API 调用」切回「本地模型」时得记得上次选的是哪种本地后端。
     */
    private var lastLocalProvider = "llama"

    /**
     * 用户选了「本地没有」的模型时要弹一句「没有模型，是否下载」（用户 m07030）。
     *
     * 以前选完只写一行日志，用户以为切过去了，点开始采集才报 `找不到模型文件`。
     * 现在 picker 里先查一遍可用性，缺了就落到这里，由 AppRoot 统一弹窗。
     */
    private val uiMissingModel = mutableStateOf<MissingModel?>(null)
    /** 下载/导入结束后问「要不要跑冒烟测试」。 */
    private val uiSmokeAsk = mutableStateOf<List<String>>(emptyList())
    /** 冒烟测试结果（非空 = 弹结果框）。 */
    private val uiSmoke = mutableStateOf<List<ModelSmoke.Result>?>(null)
    private var dlJob: Job? = null

    /**
     * 导入的总入口：既支持 SAF 选目录/zip，也支持 `/sdcard/Models` 直接读。
     *
     * 为什么放在 Activity 上而不是 Composable 里：导入可能跑几分钟，切 tab 会把
     * Composable 的 `rememberCoroutineScope` 连带取消 —— 搬到 lifecycleScope 后切页不断。
     */
    private fun beginImport(job: suspend (progress: (Int, Int, String, Long) -> Unit) -> ModelImporter.Report) {
        if (uiImporting.value) return
        uiImporting.value = true
        uiImportMsg.value = "开始导入…"
        lifecycleScope.launch {
            val rep: ModelImporter.Report = try {
                job { done, total, rel, bytes ->
                    uiImportMsg.value =
                        "[$done/$total] ${rel.substringAfterLast('/')} · 已拷 ${bytes / 1024 / 1024}MB"
                }
            } catch (t: Throwable) {
                uiImportMsg.value = "导入失败：${t.message ?: t::class.java.simpleName}"
                uiImporting.value = false
                append("模型导入失败：${t.message ?: t::class.java.simpleName}")
                return@launch
            }
            uiModelStatus.value = ModelImporter.status(this@InterpretActivity)
            uiImporting.value = false
            val text = buildString {
                append("拷 ${rep.copied} 个、跳过 ${rep.skipped} 个")
                if (rep.copiedBytes > 0) append("，写入 ${rep.copiedBytes / 1024 / 1024}MB")
                if (rep.missing.isNotEmpty()) append("，${rep.missing.size} 个源里没有")
                if (rep.failed.isNotEmpty()) append("，${rep.failed.size} 个失败")
            }
            uiImportMsg.value = "导入完成：$text"
            append("模型导入完成：$text")
            // 可选冒烟测试：只测刚碰过的那几个文件（拷进来的 / 已经就位的都算，
            // 用户点「以后再说」就什么也不做）。
            if (rep.copied > 0) {
                uiSmokeAsk.value = ModelImporter.items
                    .filter { it.rel in rep.copiedRels }
                    .map { it.rel }
            }
        }
    }

    /**
     * 从 HF 镜像站下载（用户 2026-10-18：下载仍是最大障碍，要在软件里就能下）。
     *
     * 串行下载、可取消；`uiDl` 只保存最新一条进度（界面 5 次/秒刷一次就够）。
     * 「放到后台」只是把弹窗关掉，协程继续跑 —— 真正的后台受系统调度，
     * 应用被切到后台时 Android 仍会允许一段时间，切回来能看到进度。
     */
    private fun startDownload(items: List<ModelImporter.Item>) {
        if (dlJob?.isActive == true) {
            append("已经在下载了，等它下完")
            return
        }
        val jobs = items.mapNotNull { it2 ->
            ModelDownloader.urlFor(it2.rel)?.let { u ->
                ModelDownloader.Job(it2.rel, u, it2.bytes, it2.note)
            }
        }
        if (jobs.isEmpty()) {
            append("这些模型没有直链，点「镜像站」自己找")
            return
        }
        uiDlShow.value = true
        uiDlActive.value = true
        dlJob = lifecycleScope.launch {
            var okCount = 0
            val doneRels = ArrayList<String>()
            for ((i, j) in jobs.withIndex()) {
                try {
                    ModelDownloader.download(this@InterpretActivity, j) { p ->
                        uiDl.value = p.copy(index = i + 1, count = jobs.size)
                    }
                    okCount++
                    doneRels += j.rel
                } catch (c: kotlinx.coroutines.CancellationException) {
                    append("下载已取消（${j.rel}）")
                    throw c
                } catch (t: Throwable) {
                    append("❌ 下载失败 ${j.rel}：${t.message ?: t::class.java.simpleName}")
                }
            }
            uiDl.value = null
            uiDlActive.value = false
            uiModelStatus.value = ModelImporter.status(this@InterpretActivity)
            append("下载结束：成功 $okCount/${jobs.size} 个，已直接写进私有目录（不用再导入）")
            uiImportMsg.value = "下载即导入：$okCount 个模型已就位"
            kotlinx.coroutines.delay(320)
            // 下完的文件先冒烟测一遍：半个文件在这里就拦住，不必等用户去点识别。
            if (doneRels.isNotEmpty()) uiSmokeAsk.value = doneRels
        }
    }

    /** 没有直链的条目：开浏览器到镜像站搜关键词。 */
    private fun openMirror(rel: String) {
        val q = java.net.URLEncoder.encode(ModelDownloader.searchFor(rel), "UTF-8")
        try {
            startActivity(
                Intent(Intent.ACTION_VIEW, android.net.Uri.parse("https://hf-mirror.com/models?search=$q")),
            )
        } catch (t: Throwable) {
            append("打不开浏览器：${t.message}")
        }
    }

    /** 跑冒烟测试（在 IO 线程读文件头，几百毫秒）。 */
    private fun runSmoke(rels: List<String>) {
        if (rels.isEmpty()) return
        uiSmokeAsk.value = emptyList()
        lifecycleScope.launch {
            val res = try {
                ModelSmoke.run(this@InterpretActivity, rels)
            } catch (t: Throwable) {
                append("冒烟测试失败：${t.message}")
                return@launch
            }
            // 稍等一下再弹：上一个对话框刚关，手指还在屏幕上，立刻弹出的新对话框
            // 会吃到同一次抬手的点击，看起来就是「闪一下就没了」。
            kotlinx.coroutines.delay(320)
            uiSmoke.value = res
            append("冒烟测试：${res.count { it.ok }}/${res.size} 通过")
        }
    }

    private val modelZipLauncher = registerForActivityResult(ActivityResultContracts.OpenDocument()) { uri ->
        if (uri == null) {
            append("没选文件，zip 导入取消")
        } else {
            beginImport { progress -> ModelImporter.importFromZip(this, uri, progress) }
        }
    }

    private val modelTreeLauncher = registerForActivityResult(ActivityResultContracts.OpenDocumentTree()) { uri ->
        if (uri == null) {
            append("没选文件夹，模型导入取消")
        } else {
            beginImport { progress -> ModelImporter.importFromTree(this, uri, progress) }
        }
    }

    // 第 4 项（2026-10-18）：「导入模型」先问类别，再开文件选择器，只把选中的文件
    // 放进对应子目录（翻译 → models/llama、识图 → models/ocr、语音 → models/asr）。
    //
    // 刻意**不做任何校验**（用户原话：用户导入错模型运行报错让他们自己处理，软件提供日志
    // 和简单的提示就行，这个软件那个对应模型只负责推荐）。
    private val uiImportCat = mutableStateOf(false)
    private val uiPendingCat = mutableStateOf(ModelImporter.CAT_TRANSLATE)

    private val importFilesLauncher =
        registerForActivityResult(ActivityResultContracts.OpenMultipleDocuments()) { uris ->
            if (uris.isNullOrEmpty()) {
                uiImportMsg.value = "没选文件，导入取消"
                return@registerForActivityResult
            }
            val cat = uiPendingCat.value
            uiImporting.value = true
            uiImportMsg.value = "拷入 ${uris.size} 个文件…"
            lifecycleScope.launch {
                val dest = File(filesDir, "models/" + ModelImporter.subdirOf(cat))
                dest.mkdirs()
                val ok = withContext(Dispatchers.IO) {
                    var n = 0
                    for (u in uris) {
                        val name = queryDisplayName(u) ?: "picked-${System.currentTimeMillis()}.bin"
                        runCatching {
                            contentResolver.openInputStream(u)?.use { input ->
                                File(dest, name).outputStream().use { out -> input.copyTo(out) }
                            } ?: error("打不开这个 uri")
                            n++
                        }.onFailure { append("拷 ${name} 失败：${it.message}") }
                    }
                    n
                }
                uiModelStatus.value = ModelImporter.status(this@InterpretActivity)
                uiImporting.value = false
                uiImportMsg.value = buildString {
                    append("已拷 $ok/${uris.size} 个到 models/${ModelImporter.subdirOf(cat)}/")
                    append("。请确认导入的模型和你选择的类型相符；选错了直接在源文件里换个名字再导一次。")
                }
                // 翻译模型：拷进来就直接指过去，省得用户再去下拉里找一遍。
                if (cat == ModelImporter.CAT_TRANSLATE && ok > 0) {
                    dest.listFiles { f -> f.isFile && f.name.endsWith(".gguf", ignoreCase = true) }
                        ?.maxByOrNull { it.lastModified() }
                        ?.let { pickLlamaModel(it.absolutePath) }
                }
            }
        }

    /** SAF 里的显示名（拿不到就返回 null，调用方给个兜底名）。 */
    private fun queryDisplayName(uri: android.net.Uri): String? = runCatching {
        contentResolver.query(uri, arrayOf(android.provider.OpenableColumns.DISPLAY_NAME), null, null, null)
            ?.use { c ->
                if (c.moveToFirst()) c.getString(0) else null
            }
    }.getOrNull()?.takeIf { it.isNotBlank() }


    /**
     * 「网络API调用」开关与它那组字段的界面副本。
     *
     * 为什么 key 也放 Compose 状态里：输入框需要**每敲一个字都重组**；真值在 DataStore，
     * 但读写是异步的，直接绑 DataStore 会变成「打字掉字」。所以输入过程中只更新状态，
     * 失去焦点/点按钮时才落盘（见 [saveNetApi]）。
     */
    private val uiNetApi = mutableStateOf(false)
    private val uiNetApiKey = mutableStateOf("")
    private val uiNetBase = mutableStateOf("")
    private val uiNetModel = mutableStateOf("")
    private val uiNetThinking = mutableStateOf("off")


    // ------------------------------------------------------------ 悬浮字幕窗状态

    /** 浮窗本体也归 [InterpretSession]（界面销毁它照样挂在屏幕上）。这里只读不建。 */
    private val subtitleOverlay: SubtitleOverlay? get() = InterpretSession.overlay

    /**
     * 刚跳过「显示在其他应用上层」授权页的标志。
     *
     * 系统授权页是个独立的 Activity，跳出去再回来只会走 `onResume`，**不会**重跑
     * `onCreate`，所以「授权完自动把浮窗加上」必须在这里记一笔，不能指望构造流程。
     */
    private var pendingOverlayPermission = false

    /** `--ez vtnet true` 跑 mock 前记下的真实设置，20 秒后原样写回（见 [runVisionTest]）。 */
    private var vtnetBackup: Settings? = null

    private val uiSubtitleOn = mutableStateOf(false)

    /** 浮窗是不是被「本界面在前台」这个理由临时收起来的（见 [onResume] / [onPause]）。 */
    private var overlayAutoHidden = false
    private val uiSubtitleTextSize = mutableStateOf(22f)
    private val uiSubtitleAlpha = mutableStateOf(0.6f)
    private val uiSubtitleTextColor = mutableStateOf(0xFFFFFFFF.toInt())
    private val uiSubtitleBoxColor = mutableStateOf(0xC0000000.toInt())
    /** 显示内容：both 双语 / target 仅译文 / source 仅原文。 */
    private val uiSubtitleMode = mutableStateOf("both")

    /**
     * 设置读完了没有的闸门。
     *
     * `SettingsStore` 是异步读的（DataStore），但 [handleAuto] 是 `onCreate` 里同步触发的。
     * 不等它就直接建后端，拿到的是**默认** `Settings()` —— 于是 `--es provider` 生效、
     * DataStore 里的选择却不生效，表现成「日志说 llama、实际跑 NLLB」。
     * 读失败也照样放行（否则自测会永远挂着），只是用默认值。
     */
    private val settingsReady = kotlinx.coroutines.CompletableDeferred<Unit>()

    private fun append(line: String) {
        // 日志状态也在 Session 里（它要能在界面销毁后继续记），线程分流同样挪过去了。
        android.util.Log.i(TAG, "#$instanceId UI: $line")
        InterpretSession.append(line)
    }

    /**
     * 装配引擎（后台线程）→ 回主线程执行 [onReady]（拉起屏幕录制授权）。
     *
     * 顺序不能反：[DictationHub] 是单槽位，采集一旦开始而引擎还没挂上，
     * 那段音频就是直接丢掉的（`DictationHub.feed` 在没人听时返回 false）。
     *
     * 做成「回调」而不是自己拉授权框，是因为本 Activity 里有两条并存的授权路径：
     * 界面按钮走 `rememberLauncherForActivityResult`（它是 Composable 里的局部变量，
     * 类方法够不着），而 `auto=interp` 走的是 [startCaptureFlow] 那条老的
     * `startActivityForResult`。让调用方决定授权怎么拉，两条路径才能共用同一段装配逻辑。
     */
    /** 被内存警告打断的那次启动：用户点「仍要继续」后接着跑。 */
    private var pendingReady: (() -> Unit)? = null

    /** MB → 「1.2GB / 640MB」，两处显示共用。 */
    private fun fmtMem(mb: Long): String =
        if (mb >= 1024) "%.1fGB".format(mb / 1024.0) else "${mb}MB"

    /** 字节数 → MiB/X.XGB。下载进度用它（用户看的是「184MiB / 1302MiB」这种）。 */
    private fun fmtMb(bytes: Long): String = when {
        bytes >= 1024L * 1024L * 1024L -> "%.2fGB".format(bytes / 1024.0 / 1024.0 / 1024.0)
        bytes >= 1024L * 1024L -> "${bytes / 1024 / 1024}MiB"
        else -> "${bytes / 1024}KiB"
    }

    private fun prepareThenCapture(onReady: () -> Unit) {
        if (uiStarting.value) return
        // 端侧大模型是 1.2GB 的 gguf，加载后还要给 KV、ASR、OCR 留地方；可用内存不够时
        // 直接装配就是在赌 lmkd 晚一点动手。先问一句，用户愿意赌就继续。
        val mem = MemProbe.snapshot(this)
        val needMb = MemProbe.needMbFor(appSettings)
        if (mem.availMb < needMb && !MemProbe.userAcceptedLowMem) {
            pendingReady = onReady
            uiLowMem.value = LowMemWarning(mem.availMb, needMb, mem.swapFreeMb)
            return
        }
        uiStarting.value = true
        lifecycleScope.launch {
            val ok = withContext(Dispatchers.Default) { buildAndAttach(::append) }
            uiStarting.value = false
            if (ok) onReady()
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        prefs = getSharedPreferences("asr_debug", MODE_PRIVATE)
        // 引擎与浮窗活在进程级的会话里（线路图⑥）：装应用上下文 + 写盘钩子。
        // 这样界面销毁后它们照旧工作，重建时也能读到同一批状态。
        InterpretSession.install(this)
        InterpretSession.persist = { s -> persistSettings { s } }
        android.util.Log.i(TAG, "#$instanceId onCreate extras=${intent?.extras?.keySet()?.joinToString()}")

        // 模型清单是纯文件系统扫描（22 个 isFile），很快；启动时刷一次，底栏「翻译」页和
        // 主界面那条「建议使用」提示卡都要读它。启动器已在字段里注册，这里只补状态。
        uiModelStatus.value = ModelImporter.status(this)

        // 随包小模型（Silero VAD / GTCRN）第一次启动时解到 files/models/asr/ 下。
        // 只有 1.1MB，放在后台线程做，解完刷新一次状态行（不然「建议使用」里会一直挂着它们）。
        lifecycleScope.launch(Dispatchers.IO) {
            val wrote = runCatching { BundledModels.ensure(applicationContext) }.getOrDefault(0)
            if (wrote > 0) {
                withContext(Dispatchers.Main) {
                    uiModelStatus.value = ModelImporter.status(this@InterpretActivity)
                }
            }
        }
        // 内存同样扫一次：主界面要显示可用量，启动前还要和「这次要多少」比一下。
        uiMem.value = MemProbe.snapshot(this)

        // 后台读一次设置（见 [appSettings] 的说明）。读失败不影响可用性，保持默认的 NLLB。
        parseOverrides()
        lifecycleScope.launch {
            runCatching { currentSettings() }.onSuccess {
                appSettings = it
                uiProviderId.value = it.providerId
                uiCustomTarget.value = it.customTargetLang
                uiCustomSource.value = it.customSourceLang
                uiLlamaDevice.value = it.llamaDevice
                uiForceSegmentSec.value = it.forceSegmentMs / 1000f
        uiAsrModel.value = AsrFactory.normalize(it.asrModel)
        uiAsrPunct.value = it.asrPunct
                uiAsrDenoise.value = it.asrDenoise
                uiContextRounds.value = it.contextRounds.coerceIn(0, 10)
                uiNetApi.value = it.netApiEnabled
                uiNetApiKey.value = it.netApiKey
                uiNetBase.value = it.netApiBaseUrl
                uiNetModel.value = it.netApiModel
                uiNetThinking.value = it.netApiThinking
                syncSubtitleUi(it)
                // 上次开过浮窗就记着（真正显示留到本界面退到后台时，见 onPause）。
                // 在本界面里立刻显示没有意义：这里本来就有一份字幕，浮窗只会盖住
                // 自己的按钮（真机上「设置 ⚙」就点不动）。ovOverlay 是 adb 调试口子，照旧。
                if (ovOverlay == true) showSubtitleOverlay()
                // 开机就把状态行写成真正的后端名。不这么做的话，只要没点过任何按钮，
                // 状态行就一直停在「未启用」—— 用户会以为后端没选上（截图里就是这么显示的）。
                if (uiInterpState.value.startsWith("翻译后端：未启用")) {
                    uiInterpState.value = when {
                        !uiTranslateOn.value -> "翻译后端：已关闭（只出原文）"
                        else -> translatorStatusText()
                    }
                }
                android.util.Log.i(
                    TAG,
                    "#$instanceId 设置就绪：provider=${it.providerId} llamaDevice=${it.llamaDevice} " +
                        "nCtx=${it.llamaNCtx} maxNew=${it.llamaMaxNewTokens}",
                )
            }.onFailure {
                android.util.Log.w(TAG, "#$instanceId 读设置失败，用默认值：${it.message}")
            }
            // 无论成败都放行，否则 auto 自测会永远挂在 settingsReady 上。
            settingsReady.complete(Unit)
        }

        ContextCompat.registerReceiver(
            this, statusReceiver, IntentFilter(AudioCaptureService.ACTION_STATUS),
            ContextCompat.RECEIVER_NOT_EXPORTED,
        )

        setContent { InterpTheme { AppRoot() } }

        val auto = intent?.getStringExtra("auto").orEmpty()
        if (auto.isNotEmpty()) handleAuto(auto)
        // `--ez nettest true` 可以单用（不需要 --es auto），排障时最省事
        else if (intent?.getBooleanExtra("nettest", false) == true) handleAuto("nettest")
    }

    override fun onNewIntent(intent: Intent) {
        super.onNewIntent(intent)
        setIntent(intent)
        val auto = intent.getStringExtra("auto").orEmpty()
        if (auto.isNotEmpty()) handleAuto(auto)
    }

    /**
     * 从「显示在其他应用上层」授权页回来时补上浮窗。
     *
     * 只能放 `onResume`：系统授权页是另一个 Activity，跳出去再回来不重跑 `onCreate`。
     */
    override fun onResume() {
        super.onResume()
        com.screenast.app.core.AppVisibility.oursInForeground = true
        // 浮窗是 TYPE_APPLICATION_OVERLAY，盖在**自己界面之上**：它占的那块地方
        // 连本 Activity 的按钮都点不动（真机上「设置」就是这么点不进去的）。
        // 所以本界面在前台时先把浮窗收起来，切到别的 App 再放回去 ——
        // 同一时刻只显示一份字幕，触控也不再打架。
        if (InterpretSession.overlayShowing()) {
            InterpretSession.hideOverlay()
            overlayAutoHidden = true
        }
        if (!pendingOverlayPermission) return
        pendingOverlayPermission = false
        val granted = SubtitleOverlay.canDraw(this)
        append(if (granted) "已获得「显示在其他应用上层」权限" else "⚠ 仍未获得浮窗权限，浮窗无法显示")
        if (granted) {
            uiSubtitleOn.value = true
            appSettings = appSettings.copy(subtitleEnabled = true)
            persistSettings { it.copy(subtitleEnabled = true) }
            showSubtitleOverlay()
        }
    }

    /**
     * 界面退到后台（包括被划掉）时把浮窗放回来 —— 这才是 ⑥ 真正要的形态：
     * 人在别的 App 里看视频，字幕浮在上面。
     */
    override fun onPause() {
        super.onPause()
        com.screenast.app.core.AppVisibility.oursInForeground = false
        if (overlayAutoHidden) overlayAutoHidden = false
        // 「浮窗：开」= 我在别的 App 里也要看到字幕。界面一走就把它放回来。
        if (uiSubtitleOn.value && !InterpretSession.overlayShowing()) InterpretSession.showOverlay()
    }

    override fun onDestroy() {
        super.onDestroy()
        // ★ 线路图 ⑥ 的核心改动：**不再销毁引擎和浮窗**。
        //
        // 以前这里是 `engine?.close()` + `subtitleOverlay?.hide()`，于是用户切到别的 App
        // 看视频、界面被系统回收时同传就断了 —— 而他要的正是「划掉界面，浮窗还在跑」。
        // 现在引擎和浮窗归 [InterpretSession] 管，这个 Activity 只是它的一个视图：
        // 销毁时只解绑广播，重建后又读同一批状态把画面接回来。
        // 真要结束一场同传，走界面上的「停止」→ [InterpretSession.stopAll]。
        runCatching { unregisterReceiver(statusReceiver) }
        android.util.Log.i(TAG, "#$instanceId onDestroy（引擎与浮窗留在会话里继续跑）")
    }

    // ---------------------------------------------------------------- 自动流程

    private fun handleAuto(auto: String) {
        if (!autoFired.compareAndSet(false, true)) {
            android.util.Log.i(TAG, "#$instanceId auto 已跑过，跳过（防重入）")
            return
        }
        android.util.Log.i(TAG, "#$instanceId auto=$auto")
        // ★ 必须等设置读完再动。不等的话 [makeTranslatorFactory] 会拿**默认值**（NLLB）去建后端，
        // 而 DataStore 里存的其实是 llama —— 症状是日志写着「设置就绪：provider=llama」，
        // 实际加载并翻译的却是 NLLB（真机 `auto=wavtest` 踩过这个竞态）。
        lifecycleScope.launch {
            settingsReady.await()
            when (auto) {
                // 建引擎 + 打开翻译 + 让用户授权后开采集（人工授权那一步绕不过去）
                "interp" -> {
                    uiTranslateOn.value = true
                    prepareThenCapture { startCaptureFlow() }
                }
                // 注入式自测：绕开采集，直接把已知文本的语音灌进 Hub，走完整条识别+翻译链路
                "wavtest" -> {
                    uiTranslateOn.value = true
                    val path = intent?.getStringExtra("pcm") ?: "/data/local/tmp/asr_test.pcm"
                    feedPcmFile(path, ::append)
                }
                // 端侧大模型直测：不走 ASR / 采集，只验证「系统提示词 + 模型」本身。
                // 用途：日语提示词回归（真机 HTP0 与 PC CPU 的输出可能不同）。
                "llamatest" -> runLlamaTextTest()
                "nettest" -> testNetApi()
                // `--es auto ocrfile:/sdcard/x.png`：不碰界面，直接对一张图跑 OCR 并把结果打进日志。
                // 真机上浮窗/防误触会吃掉触摸，有这种无头口子才能无人值守回归竖排与横排。
                else -> if (auto.startsWith("ocrfile:")) {
                    runOcrOnFile(auto.removePrefix("ocrfile:"))
                } else if (auto.startsWith("setmodel:")) {
                    // 调试口子：只改「网络API 模型名」并落盘。中文输入法会把 `input text`
                    // 的 `deepseek-flash` 变成「Deepseek－」（首字母大写、连字符变全角），
                    // 用 adb 直接写设置比跟输入法搏斗可靠。
                    val m = auto.removePrefix("setmodel:").trim()
                    persistSettings { it.copy(netApiModel = m) }
                    append("已写入模型名：$m")
                } else if (auto == "setapi") {
                    // 调试口子：把网络API 四件套（开关/地址/模型/Key）一次写进 DataStore。
                    // 用途：设置文件坏掉后重新灌值 —— 比在中文输入法里 `input text` 靠谱
                    // （输入法会把 deepseek-flash 变成「Deepseek－」）。
                    val k = intent?.getStringExtra("apikey").orEmpty()
                    val b = intent?.getStringExtra("netbase").orEmpty()
                    val md = intent?.getStringExtra("netmodel").orEmpty()
                    val on = intent?.getBooleanExtra("netapi", false) == true
                    persistSettings {
                        it.copy(
                            netApiKey = k,
                            netApiBaseUrl = b,
                            netApiModel = md,
                            netApiEnabled = on,
                        )
                    }
                    append("已写入网络API设置：开关=$on 地址=${b.ifEmpty { "(空)" }} " +
                        "模型=${md.ifEmpty { "(空)" }} Key长度=${k.length}")
                } else if (auto == "visiontest") {
                    runVisionTest()
                } else {
                    // `--ez nettest true`：不点界面也能跑一次「测试连接」，把结论打进日志。
                    // 真机上浮窗会盖住设置页，触摸可能被浮窗吃掉 —— 有个 adb 口子才好排障。
                    if (intent?.getBooleanExtra("nettest", false) == true) {
                        testNetApi()
                    } else {
                        append("未知 auto 值：$auto（可用 interp / wavtest）")
                    }
                }
            }
        }
    }

    // ---------------------------------------------------------------- 引擎装配

    /**
     * 建立同传引擎并挂到采集上。**必须在启动采集之前调用**：
     * `DictationHub` 是单槽位，[DictationHub.feed] 在没人听时直接丢掉音频。
     */
    /**
     * 建引擎并挂到采集上。**已建过就直接返回 true**（界面切回来再点「开始」不会重建）。
     *
     * 实现已经搬进 [InterpretSession]（线路图 ⑥：引擎不随界面生死）。这里只保留
     * 「把界面侧算出来的设置灌进去」这一步 —— `effectiveSettings()` 要合并 adb 覆盖，
     * 而覆盖字段是本 Activity 解析的。
     */
    private fun buildAndAttach(onLog: (String) -> Unit): Boolean {
        val ok = InterpretSession.attach(effectiveSettings(), ovWhisperLang)
        // Session 内部的日志会自己进 uiLog；onLog 只用于自测（wavtest）里的额外说明行。
        if (!ok) onLog("❌ 引擎没起来，原因见上方日志")
        return ok
    }

    // ---------------------------------------------------------------- 注入式自测

    /**
     * 日文提示词的真机直测：**不走 ASR、不走采集**，直接对 [TranslationProvider] 发几句日文。
     *
     * 用法：
     * ```
     * adb shell am start -n com.screenast.app.debug/com.screenast.app.InterpretActivity \
     *   --es auto llamatest --es provider llama --es device HTP0
     * ```
     * 日志 `llamatest[n] OK_ZH/FAIL_JA ...` 可直接 grep；HTP0 的输出可能与 PC CPU 不同，
     * 所以这条不能只靠 PC 侧结论。
     */
    private suspend fun runLlamaTextTest() {
        val st = effectiveSettings()
        append("llamatest：provider=${st.providerId} device=${st.llamaDevice} target=${st.targetLang.code}")
        val factory = InterpretSession.translatorFactory(st)
        if (factory == null) {
            append("❌ llamatest：没有可用的翻译后端（检查 provider / 模型文件）")
            return
        }
        val p = withContext(Dispatchers.Default) {
            try {
                factory()
            } catch (t: Throwable) {
                append("❌ llamatest：加载失败 ${t.javaClass.simpleName}: ${t.message}")
                null
            }
        } ?: return
        val cases = listOf(
            "先生、こんにちは。アルナチャンネルの時間です。お元気にしていましたか。",
            "ありがとうございます！とても嬉しいです。",
            "申し訳ありませんが、その件については私では判断できません。",
            "今日はいい天気ですね。",
            "明日の会議は午後三時からです。",
            "おい、そこの君。ちょっと来てくれないか。",
            // 真机日志里中文提示词失败的片假名游戏术语 / ASR 错词样例
            "ダメージを軽減することができます。そのため、回避率の高いタンクを編成することをお勧めします。",
            "味方のコスト回復力が上昇します。はい、というわけでね。",
            "各グループの正面にいる敵は、正面のグループ以外の敵を攻撃する。",
            "伤害を軽減できます。そのため、回避率も。",
        )
        cases.forEachIndexed { i, src ->
            val sb = StringBuilder()
            val t0 = android.os.SystemClock.elapsedRealtime()
            runCatching {
                p.translate(TranslateRequest(src, Lang.JA, Lang.ZH))
                    .collect { c -> sb.setLength(0); sb.append(c.text) }
            }.onSuccess {
                val out = sb.toString().trim()
                val kana = out.any {
                    it.code in 0x3040..0x30FF || it.code in 0x31F0..0x31FF ||
                        it.code in 0xFF61..0xFF9F
                }
                val tag = when {
                    out.isEmpty() -> "EMPTY"
                    kana -> "FAIL_JA"
                    else -> "OK_ZH"
                }
                val human = when {
                    out.isEmpty() -> "⚠ 未翻译（空）"
                    kana -> "❌仍是日文"
                    else -> "✅中文"
                }
                val ms = android.os.SystemClock.elapsedRealtime() - t0
                android.util.Log.i(TAG, "llamatest[$i] $tag ${ms}ms src=$src out=$out")
                append("llamatest[$i] $human ${ms}ms：$out")
            }.onFailure {
                android.util.Log.e(TAG, "llamatest[$i] ERROR", it)
                append("llamatest[$i] ❌ ${it.javaClass.simpleName}: ${it.message}")
            }
        }
        runCatching { p.close() }
        append("llamatest 结束")
    }

    /**
     * 把一个裸 PCM 文件（16 kHz / mono / s16le）灌进 `DictationHub`，走**完整**链路：
     * 闸门 → VAD → SenseVoice → InterpretEngine → NLLB → 字幕。
     *
     * 这是今天唯一能**自动判定对错**的验收手段：采集侧要人工授权、也没有标准答案；
     * 而这段音频是用 Windows TTS 从已知文本合成的，于是「识别对不对 / 翻译对不对」
     * 都有据可依（见 `tools/make_asr_test_wav.ps1`）。
     *
     * ⚠️ 每片必须**单独拷进定长缓冲**再 feed。踩过的坑：直接把整个数组和片长一起传，
     * 日志上一切正常（221 个窗、7.2 秒都对得上），实际读的却是缓冲区里错位的数据，
     * 闸门一次都没开。教训与 `NllbTranslator` 那次 `LongBuffer` 一样 ——
     * 「指针 + 长度」描述同一块内存时，长度写错不报错，只是静默读错地方。
     */
    private suspend fun feedPcmFile(path: String, onLog: (String) -> Unit) {
        if (!buildAndAttach(onLog)) return
        val e = engine ?: return

        // 自测必须独占引擎：真实采集一旦同时在跑，两路 PCM 会从不同线程挤进同一个
        // DictationEngine 的定长窗口缓冲，症状是 `Vad_acceptWaveform: NULL input` /
        // `vector` 的连环报错（真机踩过），而且两段音频互相覆盖，结果毫无意义。
        if (uiCaptureRunning.value) {
            onLog("检测到真实采集在跑，先停掉（自测要独占引擎）")
            startService(
                Intent(this, AudioCaptureService::class.java).apply {
                    action = AudioCaptureService.ACTION_STOP
                },
            )
            delay(400)
        }

        val file = File(path)
        if (!file.isFile) {
            onLog("❌ 找不到 PCM 文件：$path")
            return
        }
        val bytes = withContext(Dispatchers.IO) { file.readBytes() }
        onLog("PCM：${bytes.size} 字节 ≈ ${"%.2f".format(bytes.size / 32000f)} 秒")

        DictationHub.notifyCaptureState(true)
        val chunk = 3200 // 100 ms @ 16 kHz mono s16le
        val buf = ByteArray(chunk)
        val t0 = SystemClock.elapsedRealtime()
        var off = 0
        while (off < bytes.size) {
            val n = minOf(chunk, bytes.size - off)
            System.arraycopy(bytes, off, buf, 0, n)
            val ok = DictationHub.feed(buf, n, SenseVoiceAsr.SAMPLE_RATE, 1)
            require(ok) { "DictationHub.feed 返回 false —— 引擎没挂上采集？" }
            off += n
            delay(n / 32L) // 按真实时间节流，让闸门/VAD 看到的是实时流
        }
        onLog("灌完，用时 ${SystemClock.elapsedRealtime() - t0} ms，正在冲尾句…")

        // 冲尾句：VAD 里可能压着最后半句，`notifyCaptureState(false)` 才会把它兑现。
        // 但它兑现出来的是**识别结果**，翻译还要另排队 —— 所以不能一到这儿就下结论。
        //
        // 判据用「连续 3 次快照不变且队列为空」，而不是固定睡 N 秒：
        // 固定睡眠要么白等（短句），要么截断（长句 + 首次 3.5 秒模型加载）。
        DictationHub.notifyCaptureState(false)
        val deadline = SystemClock.elapsedRealtime() + 25_000
        var lastSnapshot = -1L
        var stable = 0
        while (SystemClock.elapsedRealtime() < deadline) {
            delay(400)
            val snapshot = e.utterances * 1000 + e.translated
            if (snapshot == lastSnapshot && e.queueDepth == 0) {
                if (++stable >= 3) break
            } else {
                stable = 0
            }
            lastSnapshot = snapshot
        }
        onLog(
            "自检结束：喂入窗口 ${e.windowsFed} / 跳过 ${e.windowsSkipped} / 前导救回 ${e.preRollRescued}，" +
                "强制切段 ${e.forcedCuts} 段，" +
                "识别 ${e.utterances} 段，翻译 ${e.translated} 句（丢 ${e.dropped}，跳过 ${e.skippedNoLang}），" +
                "上下文 ${e.lastContextRounds} 轮",
        )
    }

    // ---------------------------------------------------------------- 采集

    private fun startCaptureFlow() {
        val mpm = getSystemService(MediaProjectionManager::class.java)
        startActivityForResult(mpm.createScreenCaptureIntent(), REQ_PROJECTION)
    }

    @Deprecated("Deprecated in Java")
    override fun onActivityResult(requestCode: Int, resultCode: Int, data: Intent?) {
        @Suppress("DEPRECATION")
        super.onActivityResult(requestCode, resultCode, data)
        if (requestCode != REQ_PROJECTION) return
        if (resultCode != RESULT_OK || data == null) {
            append("⚠ 用户没授权屏幕录制，无法采集音频")
            return
        }
        append("已授权，启动采集服务")
        // 互斥：一个 MediaProjection 只能建一个 VirtualDisplay。识图抓屏占着投屏时
        // 音频采集建不起来（拿到的是已被用掉的 token），所以先把抓屏停掉。
        com.screenast.app.vision.VisionCaptureService.stop(this)
        val i = Intent(this, AudioCaptureService::class.java).apply {
            action = AudioCaptureService.ACTION_START
            putExtra(AudioCaptureService.EXTRA_RESULT_CODE, resultCode)
            putExtra(AudioCaptureService.EXTRA_RESULT_DATA, data)
        }
        ContextCompat.startForegroundService(this, i)
    }

    // ---------------------------------------------------------------- UI

    @Composable
    private fun InterpTheme(content: @Composable () -> Unit) {
        MaterialTheme(
            colorScheme = darkColorScheme(
                primary = Color(0xFFD0BCFF),
                background = Color(0xFF101014),
                surface = Color(0xFF1C1B22),
            )
        ) {
            Surface(Modifier.fillMaxSize(), color = MaterialTheme.colorScheme.background) { content() }
        }
    }

    /**
     * 应用外壳：底部菜单栏切换「同传 / 识图」两个页。
     *
     * 为什么是同一个 Activity 里的两个页，而不是两个 Activity：
     * - 两页共用同一个引擎进程、同一份设置、同一个翻译后端，拆成两个 Activity 只会多一份生命周期问题；
     * - 用户要的是「下方菜单栏可切」，切页不该重建引擎（同传正在跑的时候点一下识图就断了，那是 bug）。
     *
     * 两个页都不落盘 tab：重建后回到「同传」是合理的（同传是主用途）。
     */
    @Composable
    private fun AppRoot() {
        var tab by remember { mutableIntStateOf(0) }

        Scaffold(
            containerColor = MaterialTheme.colorScheme.background,
            bottomBar = {
                NavigationBar(containerColor = Color(0xFF17161F)) {
                    NavigationBarItem(
                        selected = tab == 0,
                        onClick = { tab = 0 },
                        icon = { Text("🎧", fontSize = 16.sp) },
                        label = { Text("同传", fontSize = 12.sp) },
                    )
                    NavigationBarItem(
                        selected = tab == 1,
                        onClick = { tab = 1 },
                        icon = { Text("🔍", fontSize = 16.sp) },
                        label = { Text("识图", fontSize = 12.sp) },
                    )
                    NavigationBarItem(
                        selected = tab == 2,
                        onClick = { tab = 2 },
                        icon = { Text("🌐", fontSize = 16.sp) },
                        label = { Text("全局翻译设置", fontSize = 9.sp, maxLines = 1) },
                    )
                }
            },
        ) { innerPad ->
            Box(Modifier.fillMaxSize().padding(innerPad)) {
                if (tab == 0) {
                    Screen()
                } else if (tab == 2) {
                    SharedScreen()
                } else {
                    VisionScreen(
                        settings = appSettings,
                        onChange = { next ->
                            val clean = Settings.sanitize(next)
                            appSettings = clean
                            persistSettings { clean }
                        },
                        onLog = { append(it) },
                    )
                }

                // 下载进行中：右上角常驻一个图标（常态隐藏）。点一下把进度弹窗叫回来 ——
                // 用户原话：「点掉界面就看不到了，下载进行时在页面上方显示一个下载图标」。
                if (uiDlActive.value) {
                    val p = uiDl.value
                    Surface(
                        onClick = { uiDlShow.value = true },
                        shape = RoundedCornerShape(14.dp),
                        color = Color(0xFF2B2740),
                        border = androidx.compose.foundation.BorderStroke(1.dp, Color(0x44FFFFFF)),
                        modifier = Modifier
                            .align(Alignment.TopEnd)
                            .padding(end = 12.dp, top = 8.dp),
                    ) {
                        Row(
                            modifier = Modifier.padding(horizontal = 10.dp, vertical = 7.dp),
                            verticalAlignment = Alignment.CenterVertically,
                            horizontalArrangement = Arrangement.spacedBy(6.dp),
                        ) {
                            Text("⤓", fontSize = 13.sp, color = Color(0xFFB9A6FF))
                            Text(
                                if (p == null) "下载中…"
                                else "${p.percent}% · ${fmtMb(p.done)}/${fmtMb(p.total)}",
                                fontSize = 11.sp, color = Color(0xFFD9D6E8),
                            )
                        }
                    }
                }
            }
        }
        // 三个模型相关的弹窗放在 AppRoot 这一层：不管当前在哪个 tab、设置页开没开，
        // 下载进度和冒烟测试结果都得能弹出来（用户可能从设置页发起下载后切走）。
        ModelDialogs()

        // 首次启动（noticeAccepted < 当前版本）或用户从设置点进来时显示。
        // 首启模式不允许点外面关掉：同意才进主界面，不同意直接退出。
        uiMissingModel.value?.let { miss ->
            val items = miss.rels.mapNotNull { rel -> ModelImporter.items.firstOrNull { it.rel == rel } }
            val downloadable = items.filter { ModelDownloader.urlFor(it.rel) != null }
            AlertDialog(
                onDismissRequest = { uiMissingModel.value = null },
                title = { Text("缺少模型") },
                text = {
                    Column(
                        Modifier.heightIn(max = 420.dp).verticalScroll(rememberScrollState()),
                        verticalArrangement = Arrangement.spacedBy(6.dp),
                    ) {
                        Text("「${miss.label}」需要的文件本地还没有：", fontSize = 12.sp)
                        items.forEach { it2 ->
                            Text(
                                "· ${it2.rel.substringAfterLast('/')}　${fmtMb(it2.bytes)}",
                                fontSize = 11.sp, color = Color(0xFFB9B6C8),
                            )
                        }
                        if (miss.note.isNotBlank()) {
                            Text(miss.note, fontSize = 11.sp, color = Color(0xFF9E9EA7))
                        }
                        if (downloadable.isEmpty()) {
                            Text(
                                "这些文件没有直链，只能从随包 Models 导入。",
                                fontSize = 11.sp, color = Color(0xFFFFC66B),
                            )
                        }
                    }
                },
                confirmButton = {
                    TextButton(
                        onClick = {
                            val m = miss
                            uiMissingModel.value = null
                            if (downloadable.isNotEmpty()) startDownload(downloadable)
                            else openMirror(m.rels.first())
                        },
                    ) { Text(if (downloadable.isNotEmpty()) "下载" else "镜像站") }
                },
                dismissButton = {
                    TextButton(
                        onClick = {
                            val rev = miss.revert
                            uiMissingModel.value = null
                            rev?.invoke()
                        },
                    ) { Text("取消") }
                },
            )
        }
        uiApiTest.value?.let { rep ->
                AlertDialog(
                    onDismissRequest = { uiApiTest.value = null },
                    title = {
                        Text(
                            if (rep.ok) "✅ 连接成功" else "❌ 连接失败",
                            color = if (rep.ok) Color(0xFF8EDDA1) else Color(0xFFFF8A80),
                        )
                    },
                    text = {
                        Column(
                            Modifier.heightIn(max = 420.dp).verticalScroll(rememberScrollState()),
                            verticalArrangement = Arrangement.spacedBy(6.dp),
                        ) {
                            rep.lines.forEach { Text(it, fontSize = 12.sp, color = Color(0xFFD9D6E8)) }
                        }
                    },
                    confirmButton = {
                        TextButton(onClick = { uiApiTest.value = null }) { Text("好") }
                    },
                )
            }
            if (uiNotice.value) {
            val firstRun = appSettings.noticeAccepted < com.screenast.app.legal.Notice.VERSION
            com.screenast.app.legal.NoticeDialog(
                firstRun = firstRun,
                onAccept = {
                    uiNotice.value = false
                    val next = appSettings.copy(noticeAccepted = com.screenast.app.legal.Notice.VERSION)
                    val clean = Settings.sanitize(next)
                    appSettings = clean
                    persistSettings { clean }
                },
                onDismiss = { uiNotice.value = false },
                onExit = { finish() },
            )
        }

        // 首启：只要还没同意过当前版本条款，就弹一次（用户点「以后再说」不存在，
        // 不同意就是退出，避免「点掉就能继续用」）。
        androidx.compose.runtime.LaunchedEffect(appSettings.noticeAccepted) {
            if (appSettings.noticeAccepted < com.screenast.app.legal.Notice.VERSION) {
                uiNotice.value = true
            }
        }
    }

    /**
     * 下载进度 / 「要不要冒烟测试」/ 冒烟测试结果。
     *
     * 进度条的数字直接来自 [ModelDownloader.Progress]：`已下 / 总共 · 百分比 · 速度`。
     * 用户原话：「在下载时弹出一个窗口显示进度条和百分比，以及总进度（如 184MiB/1302MiB）」。
     */
    @Composable
    private fun ModelDialogs() {
        val dl = uiDl.value
        if (dl != null && uiDlShow.value) {
            AlertDialog(
                onDismissRequest = { uiDlShow.value = false },
                title = { Text("正在下载模型", fontSize = 15.sp) },
                text = {
                    Column(verticalArrangement = Arrangement.spacedBy(8.dp)) {
                        Text(
                            "第 ${dl.index.coerceAtLeast(1)}/${dl.count.coerceAtLeast(1)} 个 · ${dl.label}",
                            fontSize = 12.sp,
                        )
                        LinearProgressIndicator(
                            progress = if (dl.total > 0) dl.done.toFloat() / dl.total else 0f,
                            modifier = Modifier.fillMaxWidth(),
                        )
                        Text(
                            "${fmtMb(dl.done)} / ${fmtMb(dl.total)} · ${dl.percent}%" +
                                if (dl.speed > 0) " · ${fmtMb(dl.speed)}/s" else "",
                            fontSize = 11.sp, color = Color(0xFF9E9EA7),
                        )
                        Text(
                            "下载即导入：字节直接写进应用的私有模型目录，下完就能用，" +
                                "不需要再点「导入模型」。\n" +
                                "「放到后台」只关这个窗口，下载继续。",
                            fontSize = 11.sp, color = Color(0xFF6F6C7C),
                        )
                    }
                },
                confirmButton = {
                    TextButton(onClick = { uiDlShow.value = false }) { Text("放到后台") }
                },
                dismissButton = {
                    TextButton(onClick = {
                        dlJob?.cancel()
                        dlJob = null
                        uiDl.value = null
                    }) { Text("取消下载") }
                },
            )
        }

        val ask = uiSmokeAsk.value
        if (ask.isNotEmpty()) {
            AlertDialog(
                onDismissRequest = { uiSmokeAsk.value = emptyList() },
                title = { Text("要不要跑一次兼容性检查？", fontSize = 15.sp) },
                text = {
                    Column(verticalArrangement = Arrangement.spacedBy(6.dp)) {
                        Text("刚处理了 ${ask.size} 个文件。", fontSize = 12.sp)
                        Text(
                            "冒烟测试只读文件头，几百毫秒：查大小对不对、是不是真的 gguf/onnx、" +
                                "有没有下成网页错误页。不加载模型，不占内存。",
                            fontSize = 11.sp, color = Color(0xFF9E9EA7),
                        )
                    }
                },
                confirmButton = {
                    TextButton(onClick = { runSmoke(ask) }) { Text("现在测") }
                },
                dismissButton = {
                    TextButton(onClick = { uiSmokeAsk.value = emptyList() }) { Text("以后再说") }
                },
            )
        }

        uiSmoke.value?.let { res ->
            AlertDialog(
                onDismissRequest = { uiSmoke.value = null },
                title = { Text("兼容性检查：${res.count { it.ok }}/${res.size} 通过", fontSize = 15.sp) },
                text = {
                    Column(
                        Modifier.heightIn(max = 420.dp).verticalScroll(rememberScrollState()),
                        verticalArrangement = Arrangement.spacedBy(6.dp),
                    ) {
                        res.forEach { r ->
                            Text(
                                (if (r.ok) "✔ " else "✘ ") + r.rel.substringAfterLast('/'),
                                fontSize = 12.sp,
                                color = if (r.ok) Color(0xFF8BDC9A) else Color(0xFFFF8A8A),
                            )
                            Text("　${r.detail}", fontSize = 11.sp, color = Color(0xFF9E9EA7))
                        }
                        if (res.any { !it.ok }) {
                            Text(
                                "没通过的先重新下载/重新导入；还不行就看日志（AST.Models / AST.Smoke）。",
                                fontSize = 11.sp, color = Color(0xFFC8B48A),
                            )
                        }
                    }
                },
                confirmButton = {
                    TextButton(onClick = { uiSmoke.value = null }) { Text("知道了") }
                },
            )
        }
    }


    /**
     * 三个「模型速选」下拉框：翻译 / 识图 / 语音。
     *
     * 用户 2026-10-18 的要求原话：「用户选择后可在对应模型选择的下拉快捷栏
     * （现在把那部分改成下拉快捷栏）进行选择，没必要一个字不差，用户导入错模型运行报错
     * 让他们自己处理，软件提供日志和简单的提示就行，这个软件那个对应模型只负责推荐」。
     *
     * 所以这里**只列出现在文件系统里真实存在的候选**，不做校验、不做兼容性判断：
     * 选错了顶多是启动引擎时报错，日志里能看到 `AST.Llama` / `AST.Asr` / `AST.Vision` 的原始异常。
     */
    @Composable
    private fun ModelPickers(modelLocked: Boolean) {
        val ctx = LocalContext.current

        // 翻译：目录里所有 gguf（私有目录 + 外部私有目录），刷新键用模型状态。
        val ggufFiles = remember(uiModelStatus.value, appSettings.llamaModelDir) {
            listOfNotNull(
                File(ctx.filesDir, "models/llama"),
                ctx.getExternalFilesDir(null)?.let { File(it, "models/llama") },
            ).flatMap { d ->
                d.listFiles { f -> f.isFile && f.name.endsWith(".gguf", ignoreCase = true) }
                    ?.toList().orEmpty()
            }.distinctBy { it.name }
        }
        val curLlama = appSettings.llamaModelDir.takeIf { it.isNotBlank() }?.let { File(it).name }
        val providerLabel = PROVIDERS.firstOrNull { it.first == uiProviderId.value }?.second ?: uiProviderId.value
        ModelDropdown(
            title = "翻译模型",
            current = when {
                curLlama != null -> "端侧大模型 · $curLlama"
                uiProviderId.value == "llama" && ggufFiles.isNotEmpty() -> "端侧大模型 · ${ggufFiles.first().name}"
                else -> providerLabel
            },
            enabled = !modelLocked,
            options = buildList {
                add("端侧大模型（自动找 gguf）" to { pickLlamaModel("") })
                ggufFiles.forEach { f -> add("端侧大模型 · ${f.name}" to { pickLlamaModel(f.absolutePath) }) }
                add("端侧 NLLB（句级，约 1.2GB）" to { pickProvider("nllb") })
                add("云端 API（填 Key / 地址 / 模型名）" to { pickProvider("openai") })
            },
        )

        // 识图：OCR 权重只有两档，装没装由 VisionModels 自己在装载时兜底。
        ModelDropdown(
            title = "识图模型（OCR）",
            current = when (appSettings.visionOcrServer) {
                true -> "高精度 server · 164MB"
                false -> "轻量 mobile · 21MB"
            },
            enabled = !modelLocked,
            options = listOf(
                "轻量 mobile · 21MB" to { pickOcrServer(false) },
                "高精度 server · 164MB" to { pickOcrServer(true) },
            ),
        )

        // 语音：四种后端，没装模型的也照旧列出来（选了会在启动时报没找到文件）。
        ModelDropdown(
            title = "语音模型（ASR）",
            current = AsrFactory.labelOf(uiAsrModel.value),
            enabled = !modelLocked,
            options = AsrFactory.KINDS.map { (id, label) -> label to { pickAsrModel(id) } },
        )

        Text(
            "上面三个框只是「推荐 / 选择」，不做文件校验：导入错类型也能选，" +
                "跑起来报错时看日志里 AST.Llama / AST.Asr / AST.Vision 的原始异常即可。",
            fontSize = 11.sp, color = Color(0xFF6F6C7C),
        )
    }

    /**
     * 「导入模型」第一步：问一句导的是哪一类。
     *
     * 用户 2026-10-18：「点击『导入模型』后，弹出提示框，询问用户导入哪个模型
     * （翻译 / 识图 / 语音转文字）【其它的处理小模型直接内置到 APK】后进入文件选择器界面」。
     */
    @Composable
    private fun ImportCategoryDialog(onDismiss: () -> Unit, onPick: (String) -> Unit) {
        AlertDialog(
            onDismissRequest = onDismiss,
            title = { Text("导入哪一类模型？") },
            text = {
                Column(verticalArrangement = Arrangement.spacedBy(10.dp)) {
                    Text(
                        "只需要选「你下载的那个文件属于哪一类」，下一步就是文件选择器。" +
                            "软件不校验内容、只负责推荐对应位置。",
                        fontSize = 12.sp, color = Color(0xFF9E9EA7),
                    )
                    CategoryRow("翻译", "端侧大模型（.gguf）→ models/llama") {
                        onPick(ModelImporter.CAT_TRANSLATE)
                    }
                    CategoryRow("识图", "OCR 识别权重（det / rec .onnx）→ models/ocr") {
                        onPick(ModelImporter.CAT_VISION)
                    }
                    CategoryRow("语音转文字", "SenseVoice / Paraformer / Whisper（.onnx）→ models/asr") {
                        onPick(ModelImporter.CAT_ASR)
                    }
                    Text(
                        "VAD、词表、补背景这类小模型不用你导入：清单导入时一并到位，" +
                            "MI-GAN 也可以单独放进 Models/inpaint/ 再整体导入。",
                        fontSize = 11.sp, color = Color(0xFF6F6C7C),
                    )
                }
            },
            confirmButton = {},
            dismissButton = { TextButton(onClick = onDismiss) { Text("取消") } },
        )
    }

    @Composable
    private fun CategoryRow(title: String, note: String, onClick: () -> Unit) {
        OutlinedButton(onClick = onClick, modifier = Modifier.fillMaxWidth()) {
            Column(Modifier.fillMaxWidth()) {
                Text(title, fontSize = 14.sp, color = MaterialTheme.colorScheme.primary)
                Text(note, fontSize = 11.sp, color = Color(0xFF9E9EA7))
            }
        }
    }

    /** 一个下拉框：标题 + 当前值按钮 + 菜单。 */
    @Composable
    private fun ModelDropdown(
        title: String,
        current: String,
        enabled: Boolean,
        options: List<Pair<String, () -> Unit>>,
    ) {
        var open by remember { mutableStateOf(false) }
        Row(Modifier.fillMaxWidth(), verticalAlignment = Alignment.CenterVertically) {
            // 标题固定 88dp：不固定的话，gguf 名字一长就把「翻译模型」四个字挤成竖排。
            Text(
                title,
                fontSize = 12.sp,
                color = Color(0xFFB9B7C4),
                modifier = Modifier.width(88.dp),
            )
            Box(Modifier.weight(1f)) {
                OutlinedButton(
                    onClick = { open = true },
                    enabled = enabled,
                    modifier = Modifier.fillMaxWidth(),
                ) {
                    Text(
                        current,
                        fontSize = 12.sp,
                        maxLines = 1,
                        overflow = TextOverflow.Ellipsis,
                    )
                }
                DropdownMenu(expanded = open, onDismissRequest = { open = false }) {
                    options.forEach { (label, act) ->
                        DropdownMenuItem(
                            text = { Text(label, fontSize = 12.sp) },
                            onClick = {
                                open = false
                                act()
                            },
                        )
                    }
                }
            }
        }
    }

    /**
     * 「选的模型没装」统一入口：有直链的下、没有的给镜像站，弹窗里也能直接取消。
     */
    private fun askModel(
        label: String,
        rels: List<String>,
        note: String = "",
        revert: (() -> Unit)? = null,
    ) {
        uiMissingModel.value = MissingModel(label, rels, note, revert)
    }

    /** 指定 llama.cpp 的 gguf（空串 = 回到「自动找」）。 */
    private fun pickLlamaModel(path: String) {
        appSettings = appSettings.copy(llamaModelDir = path)
        persistSettings { it.copy(llamaModelDir = path) }
        if (uiProviderId.value != "llama") pickProvider("llama") else rebindTranslator()
        append("翻译模型 → ${if (path.isBlank()) "自动找 gguf" else File(path).name}")
        val prevPath = appSettings.llamaModelDir
        if (LlamaModels.resolve(this, path) == null) {
            askModel(
                label = if (path.isBlank()) "端侧大模型（没找到 gguf）" else File(path).name,
                rels = listOf("llama/Huihui-Qwen3.5-2B-abliterated.Q4_K_M.gguf"),
                note = "本地没有可用的 gguf：要么下载推荐的那个，要么自己放进 models/llama/ 再选一次。",
                revert = { if (path != prevPath) pickLlamaModel(prevPath) },
            )
        }
    }

    /** 识图 OCR 权重档位（mobile / server）。 */
    private fun pickOcrServer(server: Boolean) {
        appSettings = appSettings.copy(visionOcrServer = server)
        persistSettings { it.copy(visionOcrServer = server) }
        append("识图权重 → ${if (server) "高精度 server" else "轻量 mobile"}（下次识别生效）")
        val v = if (server) VisionModels.Variant.SERVER else VisionModels.Variant.MOBILE
        if (!VisionModels.installed(this, v)) {
            askModel(
                label = "识图权重 ${v.label}",
                rels = listOf("ocr/${v.detName}", "ocr/${v.recName}"),
                note = "这一档的两个 onnx 还没装；不装的话识图会继续用另一档（能跑，但精度/速度不同）。",
                revert = { pickOcrServer(!server) },
            )
        }
    }

    /**
     * 底栏第三个 tab：「翻译」——所有**共用**的设置都在这里。
     *
     * 为什么单独开一页（用户 2026-10-18）：
     * 「识图和同传一部分设置是重复的，把这部分设置（尤其是翻译模型的设置）挪到下方快捷栏中，
     * 只有识图和同传对应的设置才在自己的菜单中显示」。
     * 于是翻译后端、目标/源语言、上下文、强制切段、模型导入都只在这里出现一次；
     * 同传页只留识别模型（ASR）与字幕外观，识图页只留 OCR/擦除/浮窗。
     */
    @Composable
    private fun SharedScreen() {
        val ctx = LocalContext.current
        val modelLocked = uiStarting.value || uiCaptureRunning.value
        val llamaProvider = uiProviderId.value == "llama"
        val targetLang = uiTargetLang.value
        val fallbackSrc = uiFallbackSrc.value
        // 流式后端没有「切段」概念（每个 chunk 自己出句），切段滑块要置灰。
        val asrStreaming = uiAsrModel.value == AsrFactory.STREAMING

        Column(
            Modifier
                .fillMaxSize()
                .verticalScroll(rememberScrollState())
                .padding(16.dp),
            verticalArrangement = Arrangement.spacedBy(10.dp),
        ) {
            Row(Modifier.fillMaxWidth(), verticalAlignment = Alignment.CenterVertically) {
                Column(Modifier.weight(1f)) {
                    Text("全局翻译设置", fontSize = 24.sp, color = MaterialTheme.colorScheme.primary)
                    Text(
                        "翻译后端 / 语言 / 模型导入；同传与识图共用这一份",
                        fontSize = 12.sp, color = Color(0xFF9E9EA7),
                    )
                }
            }
            Row(Modifier.fillMaxWidth(), verticalAlignment = Alignment.CenterVertically) {
                Text(
                    "同传互译 · ${getString(com.screenast.app.R.string.app_version)}",
                    fontSize = 11.sp, color = Color(0xFF6D6B78),
                    modifier = Modifier.weight(1f),
                )
                // 用户 m06851：用户公告与免责声明放在全局翻译设置里（首次启动仍会自动弹一次）。
                Text(
                    "用户公告与免责声明 ›",
                    fontSize = 11.sp, color = Color(0xFF8E86C9),
                    modifier = Modifier
                        .clickable { uiNotice.value = true }
                        .padding(vertical = 2.dp, horizontal = 4.dp),
                )
            }
            if (modelLocked) {
                Text(
                    "⏸ 采集中：翻译相关的设置已锁定（先点「停止」再改）。",
                    fontSize = 11.sp, color = Color(0xFFFFC46B),
                )
            }
            SettingDivider()
            Text("模型速选", fontSize = 13.sp, color = MaterialTheme.colorScheme.primary)
            ModelPickers(modelLocked = modelLocked)
            // ---------- 翻译后端 ----------
            Text(
                "翻译后端（改完下一句生效；模型只加载一次）",
                fontSize = 13.sp, color = MaterialTheme.colorScheme.primary,
            )
            // 用户 m06985：「翻译后端只留两个：本地模型 / API 调用，NLLB 和 LLM 让用户自己去选」。
            // 具体是哪一个本地后端由上面「翻译模型」下拉决定（llama gguf / NLLB），这里只分本地与云端。
            val isCloud = uiProviderId.value == "openai"
            Row(
                Modifier.fillMaxWidth().horizontalScroll(rememberScrollState()),
                horizontalArrangement = Arrangement.spacedBy(8.dp),
            ) {
                if (!isCloud) Button(onClick = {}) { Text("本地模型") }
                else OutlinedButton(
                    onClick = { pickProvider(lastLocalProvider) },
                    enabled = !modelLocked,
                ) { Text("本地模型") }

                if (isCloud) Button(onClick = {}) { Text("API 调用") }
                else OutlinedButton(
                    onClick = { pickProvider("openai") },
                    enabled = !modelLocked,
                ) { Text("API 调用") }
            }
            Text(
                if (isCloud) "当前：云端 API（上面「翻译模型」里也能换回本地）"
                else "当前本地后端：" +
                    (PROVIDERS.firstOrNull { it.first == uiProviderId.value }?.second ?: uiProviderId.value) +
                    "（在「翻译模型」里换 NLLB / 大模型）",
                fontSize = 11.sp, color = Color(0xFF6F6C7C),
            )
            if (uiProviderId.value == "llama") {
                Text(
                    "大模型算在哪（NPU=Hexagon，实测比 CPU 快 10 倍以上）",
                    fontSize = 13.sp, color = MaterialTheme.colorScheme.primary,
                )
                Row(
                    Modifier.fillMaxWidth().horizontalScroll(rememberScrollState()),
                    horizontalArrangement = Arrangement.spacedBy(8.dp),
                ) {
                    DEVICES.forEach { (id, label) ->
                        if (uiLlamaDevice.value == id) Button(onClick = {}) { Text(label) }
                        else OutlinedButton(onClick = { pickDevice(id) }, enabled = !modelLocked) {
                            Text(label)
                        }
                    }
                }
            }

            SettingDivider()
            // ---------- 云端 API（选「云端」自动展开；不再有独立开关）----------
            //
            // 用户 2026-10-18：「『云端』选项和『网络API调用』重复，把开关删了，改成选云端后自动
            // 呼出对应界面」。所以这里没有开关了 —— 只要 providerId == openai，就展开这几个框，
            // 翻译走 HTTP、本地 LLM 不加载（判断仍在 InterpretSession.translatorFactory 里）。
            if (uiProviderId.value == "openai") {
                Text(
                    "云端 API（翻译走 HTTP，本地大模型不加载）",
                    fontSize = 13.sp, color = Color(0xFF7CE0A3),
                )
                OutlinedTextField(
                    value = uiNetApiKey.value,
                    onValueChange = { uiNetApiKey.value = it },
                    enabled = !modelLocked,
                    singleLine = true,
                    label = { Text("API Key") },
                    placeholder = { Text("sk-…") },
                    modifier = Modifier.fillMaxWidth(),
                )
                OutlinedTextField(
                    value = uiNetBase.value,
                    onValueChange = { uiNetBase.value = it },
                    enabled = !modelLocked,
                    singleLine = true,
                    label = { Text("接口地址（必填，OpenAI 兼容的 base URL）") },
                    placeholder = { Text("https://你的服务商/v1") },
                    modifier = Modifier.fillMaxWidth(),
                )
                OutlinedTextField(
                    value = uiNetModel.value,
                    onValueChange = {
                        uiNetModel.value = it
                        // 只改状态，落盘走「保存」/拨开关/开始采集 —— 每敲一个字都写盘会把
                        // DataStore 打爆，而且输入法组字阶段的中间态也没必要存。
                    },
                    enabled = !modelLocked,
                    singleLine = true,
                    label = { Text("模型名称（自己填，必须与服务商文档一致）") },
                    placeholder = { Text("填服务商文档里的模型名") },
                    modifier = Modifier.fillMaxWidth(),
                )
                SettingHint(
                    "模型名与接口地址都不预设：各家、各家中转的名字都不一样，猜错只会得到 " +
                        "`Model Not Exist`。照服务商文档原样抄过来即可，随时可以改。",
                    fontSize = 11.sp, color = Color(0xFF9E9EA7),
                )
                Text("思考强度（默认关：不发送任何思考参数）", fontSize = 12.sp, color = Color(0xFFB9B7C4))
                Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                    Settings.NET_API_THINKING.forEach { (id, label) ->
                        if (uiNetThinking.value == id) Button(onClick = {}) { Text(label) }
                        else OutlinedButton(onClick = { pickNetThinking(id) }, enabled = !modelLocked) {
                            Text(label)
                        }
                    }
                }
                SettingHint(
                    if (uiNetThinking.value == "off")
                        "关：请求里不带 thinking / reasoning_effort —— 对不认这两个参数的服务端最安全。"
                    else
                        "会发 thinking.type=enabled + reasoning_effort=${uiNetThinking.value}；" +
                            "翻译这种任务开思考只会让首字变慢。",
                    fontSize = 11.sp, color = Color(0xFF9E9EA7),
                )
                Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                    TextButton(onClick = { saveNetApi() }) { Text("保存 API 设置") }
                    // 「不出译文」时要能自己判断卡在哪一步：key 没存？地址拼错？
                    // 模型名服务端不认？这个按钮把三种情况分别打点到日志里。
                    OutlinedButton(
                        onClick = { testNetApi() },
                        enabled = !modelLocked,
                    ) { Text("测试连接") }
                }
            }

            // ---------- 目标语言 ----------
            Text("目标语言（字幕语种）", fontSize = 13.sp, color = MaterialTheme.colorScheme.primary)
            Row(
                Modifier.fillMaxWidth().horizontalScroll(rememberScrollState()),
                horizontalArrangement = Arrangement.spacedBy(8.dp),
            ) {
                Lang.concrete.forEach { l ->
                    if (l == targetLang) {
                        Button(onClick = {}) { Text(l.label) }
                    } else {
                        OutlinedButton(onClick = {
                            uiTargetLang.value = l
                            append("目标语言 → ${l.label}")
                        }) { Text(l.label) }
                    }
                }
            }
            // 自定义目标语种：留空就走上面选的；填了就把这个名字直接写进提示词，
            // 翻泰语/越南语这类枚举里没有的语言也不用改代码（用户 2026-10-18）。
            OutlinedTextField(
                value = uiCustomTarget.value,
                onValueChange = {
                    uiCustomTarget.value = it
                    appSettings = appSettings.copy(customTargetLang = it)
                    persistSettings { s -> s.copy(customTargetLang = it) }
                },
                label = { Text("自定义目标语种（可留空）", fontSize = 11.sp) },
                placeholder = { Text("例如 泰语 / 越南语 / 繁体中文", fontSize = 11.sp) },
                singleLine = true,
                modifier = Modifier.fillMaxWidth(),
            )

            SettingDivider()
            // ---------- 源语言兜底 ----------
            Text("源语言兜底（ASR 判不出语种时用）", fontSize = 13.sp, color = MaterialTheme.colorScheme.primary)
            Row(
                Modifier.fillMaxWidth().horizontalScroll(rememberScrollState()),
                horizontalArrangement = Arrangement.spacedBy(8.dp),
            ) {
                val autoSelected = fallbackSrc == null
                if (autoSelected) Button(onClick = {}) { Text("不猜") }
                else OutlinedButton(onClick = {
                    uiFallbackSrc.value = null
                    append("源语言兜底 → 不猜（判不出就跳过该句）")
                }) { Text("不猜") }

                Lang.concrete.forEach { l ->
                    if (fallbackSrc == l) {
                        Button(onClick = {}) { Text(l.label) }
                    } else {
                        OutlinedButton(onClick = {
                            uiFallbackSrc.value = l
                            append("源语言兜底 → ${l.label}")
                        }) { Text(l.label) }
                    }
                }
            }
            // 自定义源语种：写进提示词（「原文语言：…」），混语种字幕时模型不必自己猜。
            OutlinedTextField(
                value = uiCustomSource.value,
                onValueChange = {
                    uiCustomSource.value = it
                    appSettings = appSettings.copy(customSourceLang = it)
                    persistSettings { s -> s.copy(customSourceLang = it) }
                },
                label = { Text("自定义源语种（可留空）", fontSize = 11.sp) },
                placeholder = { Text("例如 日语 / 粤语", fontSize = 11.sp) },
                singleLine = true,
                modifier = Modifier.fillMaxWidth(),
            )

            SettingDivider()
            SettingDivider()
            // ---------- 对话上下文（只对端侧大模型有意义）----------
            // 为什么放在识别区而不是翻译区：它和「强制切段」一样，是「喂给下游多少信息」的旋钮，
            // 且两者都是即时生效、不重建引擎。翻译区那边全是「换了就要重新加载」的东西。
            val llamaProvider = uiProviderId.value == "llama"
            Text(
                "对话上下文 ${uiContextRounds.value} 轮" +
                    if (uiContextRounds.value == 0) "（无上下文）" else "（前 ${
                        uiContextRounds.value
                    } 句原文+译文一起送）" +
                    if (!llamaProvider) "（只有端侧大模型会用）" else "",
                fontSize = 12.sp, color = Color(0xFFB9B7C4),
            )
            Slider(
                value = uiContextRounds.value.toFloat(),
                valueRange = 0f..10f,
                // 0..10 共 11 个刻度、10 段区间 → steps = 9（steps 是「区间数 - 1」）。
                steps = 9,
                enabled = !modelLocked,
                onValueChange = {
                    uiContextRounds.value = it.toInt()
                    // 立刻生效：引擎每个任务读一次 appSettings.contextRounds。
                    appSettings = appSettings.copy(contextRounds = it.toInt())
                },
                onValueChangeFinished = {
                    persistSettings { it.copy(contextRounds = uiContextRounds.value) }
                },
            )
            SettingHint(
                if (llamaProvider)
                    "0 = 每句独立翻，最快；4~10 轮能救回省略主语、指代前文的句子，" +
                        "但提示词变长会让首字慢一点（10 轮约 +150~250ms）。"
                else
                    "NLLB 是句级模型，没有对话概念，调到多少都不会变 —— 想要上下文请把翻译后端换成「端侧大模型」。",
                fontSize = 11.sp, color = Color(0xFF9E9EA7),
            )

            // ---------- 强制切段（歌曲 / 连唱 / 无停顿）----------
            Text(
                "强制切段 ${"%.0f".format(uiForceSegmentSec.value)} 秒" +
                    if (asrStreaming) "（流式后端不使用）"
                    else if (uiForceSegmentSec.value < 0.5f) "（关闭：只等静音）"
                    else "（不等静音也按时间出句）",
                fontSize = 12.sp, color = Color(0xFFB9B7C4),
            )
            Slider(
                value = uiForceSegmentSec.value,
                enabled = !asrStreaming && !modelLocked,
                onValueChange = {
                    uiForceSegmentSec.value = it
                    // 立刻生效：引擎每个窗口读一次 appSettings.forceSegmentMs，
                    // 所以不必重建引擎（那要重新加载 239MB 的 SenseVoice）。
                    appSettings = appSettings.copy(forceSegmentMs = (it * 1000).toInt())
                },
                valueRange = 0f..10f,
                steps = 9,
                onValueChangeFinished = {
                    persistSettings { it.copy(forceSegmentMs = (uiForceSegmentSec.value * 1000).toInt()) }
                },
            )
            SettingHint(
                "歌曲/连唱里几乎不存在 0.5 秒静音，VAD 等不到端点就永远不出句。" +
                    "调到 3~5 秒会按时间出句，切点自动落在最安静的换气处；0 = 关闭。",
                fontSize = 11.sp, color = Color(0xFF9E9EA7),
            )

            SettingDivider()
            // ---------- 离线模型 ----------
            Text(
                "离线模型（首次使用先导入一次，之后不再需要）",
                fontSize = 13.sp, color = MaterialTheme.colorScheme.primary,
            )
            run {
                val ms = uiModelStatus.value
                Text(
                    "必备 ${ms.essentialReady}/${ms.essentialTotal} 个" +
                        if (ms.usable) "（够用了）" else "（还差 ${ms.essentialTotal - ms.essentialReady} 个）",
                    fontSize = 12.sp,
                    color = if (ms.usable) Color(0xFF8EDDA1) else Color(0xFFFFC66B),
                )
                SettingHint(
                    "全部 ${ms.ready}/${ms.total} 个 · " +
                        "%.2fGB / %.2fGB".format(ms.readyBytes / 1073741824.0, ms.totalBytes / 1073741824.0),
                    fontSize = 11.sp,
                    color = if (ms.allReady) Color(0xFF8EDDA1) else Color(0xFF9E9EA7),
                )
                if (ms.missingEssential.isNotEmpty()) {
                    val names = ms.missingEssential.take(3).joinToString("、") { it.rel.substringAfterLast('/') }
                    Text(
                        "建议使用：$names" +
                            if (ms.missingEssential.size > 3) " 等 ${ms.missingEssential.size} 个" else "",
                        fontSize = 11.sp, color = Color(0xFFFFC66B),
                    )
                }
            }
            if (uiImportMsg.value.isNotEmpty()) {
                Text(uiImportMsg.value, fontSize = 11.sp, color = Color(0xFF9E9EA7))
            }
            Row(
                Modifier.fillMaxWidth().horizontalScroll(rememberScrollState()),
                horizontalArrangement = Arrangement.spacedBy(8.dp),
                verticalAlignment = Alignment.CenterVertically,
            ) {
                Button(
                    onClick = { uiImportCat.value = true },
                    enabled = !uiImporting.value,
                ) { Text(if (uiImporting.value) "导入中…" else "导入模型…") }

                // 「从 <某个目录> 导入」按钮已删（用户 m06903：那是我自己的文件夹，不该出现在软件里）。
                // 自动探测仍然保留在 ModelImporter 内部，只用来在说明文案里提一句「检测到 Models 文件夹」，
                // 不再往界面上放按钮。

                // 想自己找模型的人从这儿看规格：文件名、大小、格式、哪几个是一套。
                OutlinedButton(onClick = { uiModelHelp.value = true }) {
                    Text("ⓘ  模型要求", color = MaterialTheme.colorScheme.primary)
                }
            }
            if (uiImportCat.value) {
                ImportCategoryDialog(
                    onDismiss = { uiImportCat.value = false },
                    onPick = { cat ->
                        uiImportCat.value = false
                        uiPendingCat.value = cat
                        importFilesLauncher.launch(arrayOf("*/*"))
                    },
                )
            }
            SettingHint(
                "翻译/识别/OCR/擦除的模型都放在随包的 Models/ 文件夹（或一个 zip）里，" +
                    "导入后进应用私有目录，卸载才会清掉。最少只要必备那 5 个文件（约 1.53GB）；" +
                    "其余是可选的第二后端与高精度档。",
                fontSize = 11.sp, color = Color(0xFF9E9EA7),
            )

            SettingDivider()
            // ---------- 额外内容下载 ----------
            // 用户 m06767：把「推荐安装之外」的那些也列出来，点一下就下（跟必备那套一致）。
            // 这些是第二后端与高精度档：不装也能用，装了能换后端/换权重。
            run {
                val ms = uiModelStatus.value
                val extra = ModelImporter.items.filter { !it.essential }
                val installedExtra = extra.count { item ->
                    java.io.File(ctx.filesDir, "models/" + item.rel).isFile ||
                        java.io.File(ModelImporter.externalRoot(ctx), item.rel).isFile
                }
                Text(
                    "额外内容下载（可选：第二后端 / 高精度档）",
                    fontSize = 13.sp, color = MaterialTheme.colorScheme.primary,
                )
                Text(
                    "已装 $installedExtra/${extra.size} 个 · 合计约 " +
                        "%.2fGB".format(extra.sumOf { it.bytes } / 1073741824.0),
                    fontSize = 11.sp, color = Color(0xFF9E9EA7),
                )
                val downloadable = extra.filter { ModelDownloader.urlFor(it.rel) != null }
                if (downloadable.isNotEmpty()) {
                    Button(onClick = { startDownload(downloadable) }) {
                        Text("一键下载全部 ${downloadable.size} 个")
                    }
                }
                extra.forEach { item ->
                    val file = java.io.File(ctx.filesDir, "models/" + item.rel)
                    val ext = java.io.File(ModelImporter.externalRoot(ctx), item.rel)
                    val present = file.isFile || ext.isFile
                    val url = ModelDownloader.urlFor(item.rel)
                    Row(
                        Modifier.fillMaxWidth(),
                        verticalAlignment = Alignment.CenterVertically,
                    ) {
                        Column(Modifier.weight(1f)) {
                            Text(
                                "${if (present) "✔" else "·"} ${item.rel.substringAfterLast('/')}",
                                fontSize = 12.sp,
                                color = if (present) Color(0xFF8EDDA1) else Color(0xFFD9D6E8),
                            )
                            Text(
                                "${fmtMb(item.bytes)} · ${item.note}",
                                fontSize = 10.sp, color = Color(0xFF8A87A0),
                            )
                            // 用户 m06804：光看文件名不知道该不该下，补一句它到底干什么用。
                            Text(
                                ModelUsage.line(item.rel),
                                fontSize = 10.sp, color = Color(0xFF6F6C82),
                            )
                        }
                        if (!present) {
                            TextButton(
                                onClick = {
                                    if (url != null) startDownload(listOf(item))
                                    else openMirror(item.rel)
                                },
                            ) { Text(if (url != null) "下载" else "镜像站", fontSize = 11.sp) }
                        }
                    }
                }
                SettingHint(
                    "这些都不影响「能不能跑」：NLLB 是另一条句级翻译路线（CC-BY-NC，禁止商用）、" +
                        "OCR server 是同一功能的更大权重（一帧慢 3~4 秒但小字更准）、" +
                        "MI-GAN 是识图「擦除」档的补背景模型、Paraformer/Whisper/标点/TEN-VAD 是 ASR 的替代后端。" +
                        "没直链的只能开镜像站自己找（词表类文件格式与上游不完全一致，不能直接下）。",
                    fontSize = 11.sp, color = Color(0xFF9E9EA7),
                )
            }
            if (uiModelHelp.value) {
                ModelRequirementDialog(
                    items = ModelImporter.items,
                    modelsDir = File(ctx.filesDir, "models"),
                    onDismiss = { uiModelHelp.value = false },
                )
            }


        }
    }

    @Composable
    private fun Screen() {
        val ctx = LocalContext.current

        // 探针：长度是否推进才是「界面真读到了」的可靠判据（identityHashCode 会骗人）。
        android.util.Log.i(
            TAG,
            "#$instanceId 组合读取: src.length=${uiLiveSource.value.length} " +
                "tgt.length=${uiLiveTarget.value.length} hist.length=${uiHistory.value.length} " +
                "log.length=${uiLog.value.length} stats.length=${uiStats.value.length}",
        )

        val level = uiLevel.value
        val src = uiLiveSource.value
        val tgt = uiLiveTarget.value
        val hist = uiHistory.value
        val stats = uiStats.value
        val log = uiLog.value
        val err = uiError.value
        val translateOn = uiTranslateOn.value
        val targetLang = uiTargetLang.value
        val fallbackSrc = uiFallbackSrc.value

        /**
         * 「模型已加载 / 正在跑」→ 锁住一切会碰模型的设置。
         *
         * 用户 2026-10-17 的要求：这类选择**只能在模型没加载之前**调，否则不稳定。
         * 技术上确实如此：识别后端的模型活在 native 里，翻译后端是引擎构造时绑定的工厂，
         * 采集中改它们等于把正在读的 ONNX/llama 会话拆掉（本项目为此吃过 SIGSEGV）。
         *
         * 不锁的是「只动浮窗外观」的那批：字号、框大小、颜色、显示内容 —— 它们不碰模型，
         * 任何时候都该能调（用户明确说了「可以调整字体大小和框的大小」）。
         */
        val modelLocked = uiStarting.value || uiCaptureRunning.value

        val needRecord = ContextCompat.checkSelfPermission(ctx, Manifest.permission.RECORD_AUDIO) !=
            PackageManager.PERMISSION_GRANTED
        val needNotify = Build.VERSION.SDK_INT >= 33 &&
            ContextCompat.checkSelfPermission(ctx, Manifest.permission.POST_NOTIFICATIONS) !=
            PackageManager.PERMISSION_GRANTED

        val permLauncher = rememberLauncherForActivityResult(
            ActivityResultContracts.RequestMultiplePermissions()
        ) { r -> append("权限结果：" + r.entries.joinToString { "${it.key.substringAfterLast('.')}=${it.value}" }) }

        val projectionLauncher = rememberLauncherForActivityResult(
            ActivityResultContracts.StartActivityForResult()
        ) { res ->
            val data = res.data
            if (res.resultCode != android.app.Activity.RESULT_OK || data == null) {
                append("⚠ 用户没授权屏幕录制，无法采集音频")
                return@rememberLauncherForActivityResult
            }
            append("已授权，启动采集服务")
            val i = Intent(ctx, AudioCaptureService::class.java).apply {
                action = AudioCaptureService.ACTION_START
                putExtra(AudioCaptureService.EXTRA_RESULT_CODE, res.resultCode)
                putExtra(AudioCaptureService.EXTRA_RESULT_DATA, data)
            }
            ContextCompat.startForegroundService(ctx, i)
        }

        /*
         * 离线模型导入状态见字段区（UI_MODEL_*），启动器也在字段里注册 —— 这里只留注释。
         */

        // 主界面 / 设置页。不落盘：重建后回到主界面是合理的（设置页不是工作现场）。
        var showSettings by remember { mutableStateOf(false) }

        Column(
            // statusBarsPadding 必须加：这个 Activity 是 edge-to-edge，标题行原本画在
            // 状态栏下面，**「设置」按钮有一半落在系统状态栏手势区里，点不动**
            // （真机现象：点它毫无反应）。加了让出状态栏高度，标题也不再和时钟图标叠在一起。
            Modifier.fillMaxSize().statusBarsPadding().verticalScroll(rememberScrollState()).padding(14.dp),
            verticalArrangement = Arrangement.spacedBy(10.dp),
        ) {
            Row(Modifier.fillMaxWidth(), verticalAlignment = Alignment.CenterVertically) {
                Column(Modifier.weight(1f)) {
                    Text(
                        if (showSettings) "设置" else "实时同传",
                        fontSize = 20.sp, color = MaterialTheme.colorScheme.primary,
                    )
                    Text(
                        if (showSettings) "识别 / 翻译 / 浮窗外观；采集中会锁住碰模型的项"
                        else "屏幕声音 → 切句 → 识别 → 翻译",
                        fontSize = 12.sp, color = Color(0xFF9E9EA7),
                    )
                }
                TextButton(onClick = { showSettings = !showSettings }) {
                    Text(if (showSettings) "← 返回" else "设置 ⚙")
                }
            }

            // 版本行：只在设置页露出来。主界面是工作现场，不该常年挂一行机型信息；
            // 设置页是用户主动进来看「这是什么版本」的地方（v0.5.0 改名后尤其需要）。
            if (showSettings) {
                // 版本行不折叠：进设置就是想看「这是哪个版本」，别藏。
                // 2026-10-18 用户 m06851：公告入口挪到「全局翻译设置」，这里只留版本。
                Text(
                    "${getString(com.screenast.app.R.string.app_name)} · " +
                        getString(com.screenast.app.R.string.app_version),
                    fontSize = 11.sp, color = Color(0xFF6D6B78),
                )
            }

            // 缺必备模型 = 现在点「开始采集」也只会出错误提示。与其让用户自己去设置里翻，
            // 不如在主界面顶一条能点的提示（这是新装用户最可能撞上的第一个坎）。
            //
            // 2026-10-18 用户要求：文案从「缺」改成「建议使用」，并且一个模型一行 ——
            // 之前是「缺：a、b 等」，挤成一行看不出到底是哪几个。
            if (!showSettings) {
                val ms = uiModelStatus.value
                if (!ms.usable) {
                    Card(colors = CardDefaults.cardColors(containerColor = Color(0xFF3A2A18))) {
                        Column(
                            Modifier.fillMaxWidth().padding(12.dp),
                            verticalArrangement = Arrangement.spacedBy(6.dp),
                        ) {
                            Text(
                                "建议使用以下模型（还差 ${ms.essentialTotal - ms.essentialReady} 个，" +
                                    "同传和识图才能都跑起来）",
                                fontSize = 13.sp, color = Color(0xFFFFC66B),
                            )
                            ms.missingEssential.forEach { item ->
                                Row(
                                    Modifier.fillMaxWidth(),
                                    verticalAlignment = Alignment.CenterVertically,
                                ) {
                                    Column(Modifier.weight(1f)) {
                                        Text(
                                            "· ${item.rel.substringAfterLast('/')}　${item.note}",
                                            fontSize = 11.sp, color = Color(0xFFC8B48A),
                                        )
                                        Text(
                                            ModelUsage.line(item.rel),
                                            fontSize = 10.sp, color = Color(0xFF9A8B6A),
                                        )
                                    }
                                    val url = ModelDownloader.urlFor(item.rel)
                                    TextButton(
                                        onClick = {
                                            if (url != null) startDownload(listOf(item))
                                            else openMirror(item.rel)
                                        },
                                    ) { Text(if (url != null) "下载" else "镜像站", fontSize = 11.sp) }
                                }
                            }
                            // 缺的里面凡是有直链的，一键下完（没直链的仍走「镜像站」）。
                            val downloadable = ms.missingEssential.filter {
                                ModelDownloader.urlFor(it.rel) != null
                            }
                            if (downloadable.size > 1) {
                                OutlinedButton(onClick = { startDownload(downloadable) }) {
                                    Text("一键下载缺的 ${downloadable.size} 个")
                                }
                            }
                            OutlinedButton(onClick = { showSettings = true }) {
                                Text("去导入模型")
                            }
                        }
                    }
                }

                // 内存一行：加载端侧大模型时最怕的就是「启动成功 → 三秒后闪退」，
                // 与其让用户猜，不如把「现在有多少 / 这次要多少」直接打在主页上。
                val mem = uiMem.value
                val needMb = MemProbe.needMbFor(appSettings)
                val tight = mem.availMb in 1 until needMb
                Text(
                    "可用内存 ${fmtMem(mem.availMb)}" +
                        "　本次启动约需 ${fmtMem(needMb)}" +
                        if (mem.hasSwap) "　虚拟内存 ${fmtMem(mem.swapTotalMb)}" else "　无虚拟内存",
                    fontSize = 11.sp,
                    color = if (tight) Color(0xFFFFC66B) else Color(0xFF7E7C8A),
                )
                if (uiLowMem.value != null) {
                    val w = uiLowMem.value!!
                    AlertDialog(
                        onDismissRequest = { uiLowMem.value = null; pendingReady = null },
                        title = { Text("警告⚠ 当前可用内存严重不足") },
                        text = {
                            Column(verticalArrangement = Arrangement.spacedBy(8.dp)) {
                                Text(w.detail, fontSize = 12.sp)
                                Text(
                                    "继续加载可能会被系统直接杀掉（表现为闪退/黑屏）。" +
                                        "可以先去系统设置里打开「内存扩展」或关掉其它应用。是否继续？",
                                    fontSize = 12.sp, color = Color(0xFF9E9EA7),
                                )
                            }
                        },
                        confirmButton = {
                            TextButton(onClick = {
                                MemProbe.userAcceptedLowMem = true
                                uiLowMem.value = null
                                val next = pendingReady
                                pendingReady = null
                                if (next != null) prepareThenCapture(next)
                            }) { Text("仍要继续") }
                        },
                        dismissButton = {
                            TextButton(onClick = { uiLowMem.value = null; pendingReady = null }) {
                                Text("取消")
                            }
                        },
                    )
                }
            }

            // ---------- 字幕区（第一优先）----------
            Card(colors = CardDefaults.cardColors(containerColor = Color(0xFF191822))) {
                Column(Modifier.fillMaxWidth().padding(14.dp), verticalArrangement = Arrangement.spacedBy(8.dp)) {
                    Text(
                        if (src.isEmpty()) "（等待说话…）" else src,
                        fontSize = 14.sp, color = Color(0xFF9E9EA7),
                    )
                    Text(
                        tgt.ifEmpty { if (translateOn) "…" else "（翻译未开启）" },
                        fontSize = 26.sp, color = Color(0xFFE8E6F0),
                    )
                }
            }

            // ---------- 控制 ----------
            Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                Button(onClick = {
                    if (needRecord || needNotify) {
                        val req = buildList {
                            if (needRecord) add(Manifest.permission.RECORD_AUDIO)
                            if (needNotify) add(Manifest.permission.POST_NOTIFICATIONS)
                        }
                        permLauncher.launch(req.toTypedArray())
                    }
                    if (engine != null) {
                        // 已经装过了，直接拉授权框，不再走一遍 0.9 秒的加载
                        projectionLauncher.launch(
                            getSystemService(MediaProjectionManager::class.java).createScreenCaptureIntent()
                        )
                    } else {
                        prepareThenCapture {
                            projectionLauncher.launch(
                                getSystemService(MediaProjectionManager::class.java)
                                    .createScreenCaptureIntent()
                            )
                        }
                    }
                }) { Text(if (uiStarting.value) "准备中…" else "开始采集") }

                OutlinedButton(onClick = {
                    startService(
                        Intent(ctx, AudioCaptureService::class.java).apply {
                            action = AudioCaptureService.ACTION_STOP
                        }
                    )
                    // 「停止」= 结束整场同传：除了停采集，还要摘掉引擎、收起浮窗。
                    // 只停采集的话引擎还挂在 DictationHub 上，浮窗也还在，看着像没停。
                    InterpretSession.stopAll()
                    if (uiSubtitleOn.value) {
                        uiSubtitleOn.value = false
                        appSettings = appSettings.copy(subtitleEnabled = false)
                        persistSettings { it.copy(subtitleEnabled = false) }
                    }
                    append("已请求停止采集，同传已结束（浮窗已收起）")
                }) { Text("停止") }

                OutlinedButton(onClick = {
                    val e = engine
                    if (e == null) {
                        append("先点「开始采集」建立引擎，再切翻译")
                        return@OutlinedButton
                    }
                    if (!translateOn) {
                        val f = makeTranslatorFactory()
                        if (f == null) {
                            append("❌ 没找到本地模型，开不了翻译（只能出原文）")
                            return@OutlinedButton
                        }
                        e.translatorFactory = f
                        uiTranslateOn.value = true
                        // 立刻改写状态行。不改的话它会一直停在初始值「未启用」，
                        // 而旁边的按钮明明写着「翻译：开」—— 两处自相矛盾，用户没法判断
                        // 到底是哪一处坏了（这正是真机上看到的样子）。真正的加载结果
                        // 稍后由 onTranslatorState 覆盖（加载中… → 就绪）。
                        uiInterpState.value = translatorStatusText()
                        append("翻译已开启：下一句起会出译文")
                    } else {
                        e.translatorFactory = null
                        uiTranslateOn.value = false
                        uiInterpState.value = "翻译后端：已关闭（只出原文）"
                        uiLiveTarget.value = "" // 关掉就把残留译文清掉，避免误读成新译文
                        append("翻译已关闭（只出原文）")
                    }
                }) { Text(if (translateOn) "翻译：开" else "翻译：关") }
            }

            // 浮窗的快捷开关放在主界面：它是「看字幕」的一部分，不该埋在设置第七屏。
            Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                OutlinedButton(onClick = { toggleSubtitle() }) {
                    Text(if (uiSubtitleOn.value) "浮窗：开" else "浮窗：关")
                }
                OutlinedButton(onClick = { InterpretSession.clearOverlay() }) { Text("清空浮窗") }
                if (needRecord || needNotify) {
                    OutlinedButton(onClick = {
                        val req = buildList {
                            if (needRecord) add(Manifest.permission.RECORD_AUDIO)
                            if (needNotify) add(Manifest.permission.POST_NOTIFICATIONS)
                        }
                        permLauncher.launch(req.toTypedArray())
                    }) { Text("授权") }
                }
            }

            // 引擎侧的错误（HTTP 400、没填 Key、模型没装…）以前只写进日志，
            // 主界面一个红字都没有 —— 用户填了 Key 却「一点译文都没有」，屏幕上毫无线索。
            // 这里直接顶到控制按钮下面，一眼能看到是哪一步失败。
            if (err.isNotEmpty()) {
                Card(colors = CardDefaults.cardColors(containerColor = Color(0xFF3A2225))) {
                    Row(
                        Modifier.fillMaxWidth().padding(10.dp),
                        verticalAlignment = Alignment.CenterVertically,
                    ) {
                        Text(
                            "❌ $err",
                            modifier = Modifier.weight(1f),
                            fontSize = 12.sp, color = Color(0xFFFFB4B4),
                        )
                        TextButton(onClick = { uiError.value = "" }) { Text("知道了") }
                    }
                }
            }

            // ================= 设置页（用户点「设置」才显示）=================
            //
            // 为什么要分两屏：这一串设置从识别模型滚到配色有十几屏，主界面真正天天用的
            // 只有「字幕 + 开始/停止 + 状态」。全堆在一屏里，用户滑半天找不到开始按钮，
            // 而开头几行提示还写着「后端可在下方切换」—— 实际要滑两千像素才看得到。
            // 分屏后主界面只留「看的东西」和「按的东西」，其余进设置。
            if (showSettings) {

            SettingDivider()
            // ---------- 识别模型 ----------
            //
            // 放在「强制切段」之前：切段只对离线后端有意义，而「用哪个后端」是更上一层
            // 的决定，先给用户看。顺序反了会让人先去调一个即将失效的滑条。
            // 每个后端的「装没装」必须按它自己的根目录判断：模型可能分散在私有目录
            // 和 /data/local/tmp 两处，拿同一个根目录去判会整排显示「未装」。
            val asrDirOverride = prefs.getString("asrDir", "") ?: ""
            val asrStreaming = uiAsrModel.value == AsrFactory.STREAMING
            if (modelLocked) {
                Text(
                    "⏸ 采集中：识别/翻译相关的设置已锁定（先点「停止」再改）。" +
                        "字号、框大小、颜色随时可调。",
                    fontSize = 11.sp, color = Color(0xFFFFC46B),
                )
            }
            Text(
                "识别模型（改完要重新点「开始采集」才会换，模型是构造时加载的）",
                fontSize = 13.sp, color = MaterialTheme.colorScheme.primary,
            )
            Row(
                Modifier.fillMaxWidth().horizontalScroll(rememberScrollState()),
                horizontalArrangement = Arrangement.spacedBy(8.dp),
            ) {
                AsrFactory.KINDS.forEach { (id, label) ->
                    // 没装模型的选项置灰而不是隐藏：隐藏会让用户以为「这功能不存在」，
                    // 置灰才说明「功能在这，只是这台机器没放模型」。
                    val ok = AsrFactory.resolveRootFor(id, filesDir, asrDirOverride) != null
                    val text = if (ok) label else "$label（未装）"
                    if (uiAsrModel.value == id) {
                        Button(onClick = {}, enabled = ok && !modelLocked) { Text(text) }
                    } else {
                        OutlinedButton(
                            onClick = { pickAsrModel(id) },
                            enabled = ok && !modelLocked,
                        ) { Text(text) }
                    }
                }
            }
            SettingHint(
                when (uiAsrModel.value) {
                    AsrFactory.STREAMING ->
                        "流式后端边说边出字（首字最快），只支持中英；分句由模型自己的端点检测决定，" +
                            "下面那个「强制切段」对它不生效。"
                    AsrFactory.PARAFORMER ->
                        "Paraformer 离线：中文最准，只支持中英；不带标点，靠下面开关补。"
                    AsrFactory.WHISPER ->
                        "Whisper Small：多语种但慢，且必须指定语言（现在按中文跑）。"
                    else ->
                        "SenseVoice：中英日韩粤通吃、自带标点，离线（等静音才出句）。"
                },
                fontSize = 11.sp, color = Color(0xFF9E9EA7),
            )
            if (uiAsrModel.value != AsrFactory.SENSEVOICE) {
                Row(
                    Modifier.fillMaxWidth(),
                    horizontalArrangement = Arrangement.spacedBy(8.dp),
                    verticalAlignment = Alignment.CenterVertically,
                ) {
                    Text("自动补标点", fontSize = 12.sp, color = Color(0xFFB9B7C4))
                    // 选中态用实心按钮、未选中用描边：和上面的后端选择保持同一套视觉语言，
                    // 用户不用再学一种控件。
                    if (uiAsrPunct.value) {
                        Button(onClick = {}) { Text("开") }
                        OutlinedButton(onClick = { pickAsrPunct(false) }, enabled = !modelLocked) {
                            Text("关")
                        }
                    } else {
                        OutlinedButton(onClick = { pickAsrPunct(true) }, enabled = !modelLocked) {
                            Text("开")
                        }
                        Button(onClick = {}) { Text("关") }
                    }
                }
            }
            SettingDivider()
            // ---------- 识别前降噪（GTCRN）----------
            // 只对「先切段、再整段识别」的后端有意义（流式后端拿不到整段）。
            if (uiAsrModel.value != AsrFactory.STREAMING) {
                Row(
                    Modifier.fillMaxWidth(),
                    horizontalArrangement = Arrangement.spacedBy(8.dp),
                    verticalAlignment = Alignment.CenterVertically,
                ) {
                    Text("识别前降噪", fontSize = 12.sp, color = Color(0xFFB9B7C4))
                    if (uiAsrDenoise.value) {
                        Button(onClick = {}) { Text("开") }
                        OutlinedButton(onClick = { pickAsrDenoise(false) }, enabled = !modelLocked) {
                            Text("关")
                        }
                    } else {
                        OutlinedButton(onClick = { pickAsrDenoise(true) }, enabled = !modelLocked) {
                            Text("开")
                        }
                        Button(onClick = {}) { Text("关") }
                    }
                }
                SettingHint(
                    if (AsrFactory.resolveRootFor(uiAsrModel.value, filesDir, asrDirOverride)
                            ?.let { AsrFactory.denoiseFile(it) } == null
                    )
                        "这台机器没放降噪模型（gtcrn_simple.onnx），开了也不生效。"
                    else
                        "对送进识别器的音频先做一次语音增强（GTCRN，535KB）。" +
                            "只为背景音乐/风噪这类环境准备，干净人声别开。",
                    fontSize = 11.sp, color = Color(0xFF9E9EA7),
                )
            }

            SettingDivider()
            // 目标语言 / 上下文 / 切段 / 翻译后端 / 模型导入都挪到了底栏「翻译」页。
            Text(
                "翻译后端、目标语言、上下文与模型导入在底栏「翻译」页。",
                fontSize = 11.sp, color = Color(0xFF6F6C7C),
            )

            SettingDivider()
            // ---------- 悬浮字幕窗（路线图 ⑤）----------
            Text(
                "悬浮字幕窗（切到别的 App 也留在屏幕上）",
                fontSize = 13.sp, color = MaterialTheme.colorScheme.primary,
            )
            Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                if (uiSubtitleOn.value) {
                    Button(onClick = { toggleSubtitle() }) { Text("关闭浮窗") }
                } else {
                    OutlinedButton(onClick = { toggleSubtitle() }) { Text("显示浮窗") }
                }
                OutlinedButton(onClick = {
                    InterpretSession.clearOverlay()
                    append("浮窗历史已清空")
                }) { Text("清空浮窗") }
            }
            SettingHint(
                if (uiSubtitleOn.value) {
                    "拖顶部「⠿ 拖动」挪位置，拖右下角「◢」改大小；内容超出可用滑条下滑"
                } else {
                    "首次开启会跳系统授权页，需打开「显示在其他应用上层」"
                },
                fontSize = 11.sp, color = Color(0xFF9E9EA7),
            )

            Text("显示内容", fontSize = 12.sp, color = Color(0xFFB9B7C4))
            Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                SUBTITLE_MODES.forEach { (id, label) ->
                    if (uiSubtitleMode.value == id) Button(onClick = {}) { Text(label) }
                    else OutlinedButton(onClick = { pickSubtitleMode(id) }) { Text(label) }
                }
            }
            SettingHint(
                when (uiSubtitleMode.value) {
                    "target" -> "只显示译文，浮窗最省地方"
                    "source" -> "只显示原文（原文升为主行），不等翻译也能看"
                    else -> "原文小字在上，译文在下"
                },
                fontSize = 11.sp, color = Color(0xFF9E9EA7),
            )

            Text(
                "字号 ${"%.0f".format(uiSubtitleTextSize.value)} sp（12~40）",
                fontSize = 12.sp, color = Color(0xFFB9B7C4),
            )
            Slider(
                value = uiSubtitleTextSize.value,
                onValueChange = {
                    uiSubtitleTextSize.value = it
                    applySubtitleSettings(persist = false)
                },
                valueRange = 12f..40f,
                onValueChangeFinished = { applySubtitleSettings(persist = true) },
            )

            Text(
                "框底色不透明度 ${"%.0f".format(uiSubtitleAlpha.value * 100f)}%（10~100）",
                fontSize = 12.sp, color = Color(0xFFB9B7C4),
            )
            Slider(
                value = uiSubtitleAlpha.value,
                onValueChange = {
                    uiSubtitleAlpha.value = it
                    applySubtitleSettings(persist = false)
                },
                valueRange = 0.1f..1f,
                onValueChangeFinished = { applySubtitleSettings(persist = true) },
            )

            SettingDivider()
            // ---------- 配色 ----------
            // 这两个状态只决定「调色板弹窗开着给谁调」，不进 Settings。
            var colorTarget by remember { mutableStateOf(ovColorDialog) }

            Text("字体颜色", fontSize = 12.sp, color = Color(0xFFB9B7C4))
            Row(
                Modifier.fillMaxWidth().horizontalScroll(rememberScrollState()),
                horizontalArrangement = Arrangement.spacedBy(8.dp),
                verticalAlignment = Alignment.CenterVertically,
            ) {
                // 当前颜色直接画成一块色板 —— 按钮文字说不出「现在是哪个颜色」。
                Box(
                    Modifier.size(26.dp).background(
                        Color(uiSubtitleTextColor.value),
                        RoundedCornerShape(6.dp),
                    ),
                )
                OutlinedButton(onClick = { colorTarget = "text" }) { Text("调色板…") }
                TEXT_COLORS.forEach { (name, argb) ->
                    val picked = uiSubtitleTextColor.value == argb
                    val onClick = {
                        uiSubtitleTextColor.value = argb
                        applySubtitleSettings(persist = true)
                    }
                    val label: @Composable () -> Unit = {
                        Text("■", color = Color(argb), fontSize = 13.sp)
                        Text(" $name")
                    }
                    if (picked) Button(onClick = {}) { label() } else OutlinedButton(onClick = onClick) { label() }
                }
            }

            Text("框底色（透明度由上面滑块控制）", fontSize = 12.sp, color = Color(0xFFB9B7C4))
            Row(
                Modifier.fillMaxWidth().horizontalScroll(rememberScrollState()),
                horizontalArrangement = Arrangement.spacedBy(8.dp),
                verticalAlignment = Alignment.CenterVertically,
            ) {
                Box(
                    Modifier.size(26.dp).background(
                        Color((uiSubtitleAlpha.value * 255).toInt().coerceIn(0, 255).shl(24)
                            or (uiSubtitleBoxColor.value and 0x00FFFFFF)),
                        RoundedCornerShape(6.dp),
                    ),
                )
                OutlinedButton(onClick = { colorTarget = "box" }) { Text("调色板…") }
                BOX_COLORS.forEach { (name, argb) ->
                    // 比较时只看 RGB：alpha 归滑块管，否则拖过透明度就「选不中」了。
                    val picked = (uiSubtitleBoxColor.value and 0x00FFFFFF) == (argb and 0x00FFFFFF)
                    val onClick = {
                        uiSubtitleBoxColor.value = (uiSubtitleBoxColor.value and 0xFF000000.toInt()) or
                            (argb and 0x00FFFFFF)
                        applySubtitleSettings(persist = true)
                    }
                    val label: @Composable () -> Unit = {
                        Text("■", color = Color(0xFF000000.toInt() or (argb and 0x00FFFFFF)), fontSize = 13.sp)
                        Text(" $name")
                    }
                    if (picked) Button(onClick = {}) { label() } else OutlinedButton(onClick = onClick) { label() }
                }
            }

            // 弹窗只在打开时读一次初值（后面靠 remember(initial) 的 key 变化重建）。
            colorTarget?.let { target ->
                val isText = target == "text"
                val initial = if (isText) {
                    uiSubtitleTextColor.value
                } else {
                    // 框底色只调 RGB，alpha 由上面的不透明度滑块独立控制。
                    uiSubtitleBoxColor.value and 0x00FFFFFF
                }
                ColorPickerDialog(
                    title = if (isText) "字体颜色" else "框底色",
                    initial = initial,
                    previewAlpha = if (isText) {
                        0xFF
                    } else {
                        (uiSubtitleAlpha.value * 255).toInt().coerceIn(0, 255)
                    },
                    onDismiss = { colorTarget = null },
                    onPick = { rgb ->
                        if (isText) {
                            // 字色保留原 alpha（默认 0xFF），只换 RGB。
                            uiSubtitleTextColor.value = (uiSubtitleTextColor.value and 0xFF000000.toInt()) or rgb
                        } else {
                            uiSubtitleBoxColor.value = (uiSubtitleBoxColor.value and 0xFF000000.toInt()) or rgb
                        }
                        applySubtitleSettings(persist = true)
                        colorTarget = null
                    },
                )
            }

            }   // if (showSettings)

            // ---------- 状态与统计 ----------
            Card(colors = CardDefaults.cardColors(containerColor = Color(0xFF23222A))) {
                Column(Modifier.padding(12.dp), verticalArrangement = Arrangement.spacedBy(5.dp)) {
                    Text(uiInterpState.value, fontSize = 12.sp, color = Color(0xFFB9B7C4))
                    Text(
                        (if (uiCaptureRunning.value) "● 采集中" else "○ 未采集") +
                            "   电平 ${"%.3f".format(level)} / ${"%.1f".format(uiDbfs.value)} dBFS" +
                            if (uiSpeaking.value) "   ● 说话中" else "",
                        fontSize = 12.sp, fontFamily = FontFamily.Monospace,
                    )
                    if (stats.isNotEmpty()) {
                        Text(stats, fontSize = 11.sp, fontFamily = FontFamily.Monospace, color = Color(0xFF9E9EA7))
                    }
                    if (err.isNotEmpty()) {
                        Text(err, fontSize = 12.sp, color = Color(0xFFFF8A80))
                    }
                }
            }

            // ---------- 历史 ----------
            if (hist.isNotEmpty()) {
                Card(colors = CardDefaults.cardColors(containerColor = Color(0xFF1A1A22))) {
                    Column(Modifier.padding(12.dp)) {
                        Text("历史", fontSize = 13.sp, color = MaterialTheme.colorScheme.primary)
                        Text(hist, fontSize = 12.sp, fontFamily = FontFamily.Monospace, color = Color(0xFFB9B7C4))
                    }
                }
            }

            // ---------- 日志 ----------
            Card(colors = CardDefaults.cardColors(containerColor = Color(0xFF1A1A22))) {
                Column(Modifier.padding(12.dp)) {
                    Text("日志", fontSize = 13.sp, color = MaterialTheme.colorScheme.primary)
                    Text(
                        log.ifEmpty { "（空）" },
                        fontSize = 11.sp, fontFamily = FontFamily.Monospace, color = Color(0xFF9E9EA7),
                    )
                }
            }

            Box(Modifier.height(24.dp))
        }

        // 统计刷新：单独一个轻量循环。它只读引擎的 @Volatile 统计字段并写一个 State，
        // 与字幕渲染互不干扰。
        LaunchedEffect(Unit) {
            while (true) {
                val e = engine
                if (e != null) {
                    val avgFirst = if (e.translated > 0) e.sumFirstMs / e.translated else 0
                    val avgTotal = if (e.translated > 0) e.sumTotalMs / e.translated else 0
                    uiStats.value = buildString {
                        append("识别 ${e.utterances} 段（解码均 ${if (e.utterances > 0) e.totalDecodeMs / e.utterances else 0} ms）")
                        append("   翻译 ${e.translated} 句  丢 ${e.dropped}  跳过 ${e.skippedNoLang}\n")
                        append("首字 最近 ${e.lastFirstMs} ms / 均 $avgFirst ms")
                        append("   整句 最近 ${e.lastTotalMs} ms / 均 $avgTotal ms")
                        append("   队列 ${e.queueDepth}\n")
                        append("窗口 喂 ${e.windowsFed} / 跳过 ${e.windowsSkipped} / 前导救回 ${e.preRollRescued}\n")
                        append(
                            "强制切段 ${e.forcedCuts} 段（解码均 " +
                                "${if (e.forcedCuts > 0) e.forcedDecodeMs / e.forcedCuts else 0} ms）"
                        )
                        // 实际带了几轮上文：设置里写 10 不代表真的带了 10 轮
                        // （前几句还没攒够历史，或后端根本不用上下文）。
                        append("   上下文 ${e.lastContextRounds} 轮")
                    }
                }
                delay(600)
            }
        }
    }

    /**
     * 按设置造一个翻译后端。**返回 null = 本地模型没找到**，调用方负责告诉用户。
     *
     * 返回的是**延迟构造的 lambda**，不是 provider 本身：真正的构造（读 1.27GB 权重、
     * 建 ONNX/llama 会话）发生在翻译线程第一次用到它的时候。在界面线程上直接造，
     * 症状就是「点一下按钮，界面冻住三四秒」。
     *
     * 三个分支对应 `Settings.providerId` 的三个取值域：
     * - `llama` —— 端侧 Qwen3.5-2B（本次新增，走 `libllama_jni.so`）
     * - `nllb` —— 端侧 NLLB-600M（原来的默认值，保留作低首字延迟的兜底）
     * - 其它（`deepseek`/`qwen`/`custom`…）—— 云端 OpenAI 兼容接口
     */
    /** 后端供应器与状态行都统一出在 [InterpretSession]（它和引擎活在同一个生命周期里）。 */
    private fun makeTranslatorFactory(): (() -> TranslationProvider)? =
        InterpretSession.translatorFactory(effectiveSettings())

    /**
     * `am start` 的临时覆盖，用来在没有设置界面的情况下切后端。只影响本实例、不写盘。
     *
     * 为什么用一组独立字段而不是一个 `Settings` 快照：设置是异步读的，覆盖是在
     * `onCreate` 里同步解析的 —— 如果存成完整快照，就得拿默认 `Settings()` 当基底，
     * 于是没被覆盖的项（模型目录、云端 key…）全都被默认值顶掉。
     * 这里只记「被点名的项」，用的时候再盖到真正读出来的设置上，就没有这个问题。
     */
    @Volatile private var ovProvider: String? = null
    @Volatile private var ovDevice: String? = null
    @Volatile private var ovModelDir: String? = null
    @Volatile private var ovNCtx: Int? = null
    @Volatile private var ovNThreads: Int? = null
    @Volatile private var ovMaxNew: Int? = null
    @Volatile private var ovAsr: String? = null
    @Volatile private var ovPunct: Boolean? = null
    @Volatile private var ovDenoise: Boolean? = null

    /**
     * 上下文轮数覆盖（`--ei ctx 4`），只给注入式自测用。
     *
     * 为什么需要：验证「带上文翻译是否真的更好」必须**同一段音频、只改轮数**各跑一次，
     * 而 UI 上点滑条没法在无人值守的脚本里做。传 0 也有效（表示明确的「无上下文」）。
     */
    @Volatile private var ovCtx: Int? = null

    /**
     * Whisper 的识别语种覆盖（`--es wlang ko`；传空串表示让它自己判断）。
     *
     * Whisper 与 SenseVoice 不一样：它不会自动按音频判断完再自己贴标签，构造时就得告诉
     * 它用哪个语种解码，猜错会把整段翻成另一种语言（实测强制 zh 处理韩语音频会吐中文）。
     * 这里留一个 adb 口子，是为了在真机上把 ko/ja/en/auto 逐个试一遍再定默认值。
     */
    @Volatile private var ovWhisperLang: String? = null

    /**
     * 强制打开悬浮字幕窗（`--es overlay true`），只给注入式自测/截图用。
     *
     * 浮窗开关本身在 DataStore 里，而 adb 改不了 App 私有目录的 DataStore 文件，
     * 没有这个口子就没法在无人值守的脚本里验证「合并后的行会不会就地更新」。
     */
    @Volatile private var ovOverlay: Boolean? = null

    /**
     * 开机就弹出 RGB 调色板（`--es colordialog text|box`），只给截图 / 无人值守验证。
     *
     * 调色板是纯 Compose 弹窗，没有 adb 能触发的入口；而「点了没反应 / 弹窗画歪了」
     * 这类问题只有看见才能发现。留这个口子是为了不必人工点。
     */
    @Volatile private var ovColorDialog: String? = null

    /** 强制打开「网络API调用」（`--es netapi true`）。 */
    @Volatile private var ovNetApi: Boolean? = null

    /** 临时 API Key（`--es apikey sk-xxx`）。真机自测用，避免把 key 写进 DataStore。 */
    @Volatile private var ovApiKey: String? = null

    /** 临时接口地址（`--es netbase http://127.0.0.1:8099/v1`，配合 `adb reverse` 打本地 mock）。 */
    @Volatile private var ovNetBase: String? = null

    /** 临时模型名（`--es netmodel deepseek-v4.1-flash`）。 */
    @Volatile private var ovNetModel: String? = null

    /** 临时思考强度（`--es thinking off|low|medium|high`）。 */
    @Volatile private var ovThinking: String? = null

    /** 用户在界面上点过任何设置后就置 false —— 界面选择优先于 adb 传参，否则点了没反应。 */
    @Volatile private var overrideActive = true

    /**
     * 生效的设置 = 存盘的 [appSettings] + `am start` 的临时覆盖。
     *
     * ```
     * adb shell am start -n com.screenast.app.debug/com.screenast.app.InterpretActivity \
     *   --es provider llama --es device HTP0 --ei nctx 1024 --ei maxnew 160
     * ```
     *
     * ⚠️ 覆盖**不能写回** [appSettings]：那个字段是 DataStore 的快照，回写会让人以为
     * 设置真的改了，下次重启又变回去，排查时会疯。
     */
    private fun effectiveSettings(): Settings {
        var s = appSettings
        if (!overrideActive) return s
        ovProvider?.let { s = s.copy(providerId = it) }
        ovDevice?.let { s = s.copy(llamaDevice = it) }
        ovModelDir?.let { s = s.copy(llamaModelDir = it) }
        ovNCtx?.let { s = s.copy(llamaNCtx = it) }
        ovNThreads?.let { s = s.copy(llamaNThreads = it) }
        ovMaxNew?.let { s = s.copy(llamaMaxNewTokens = it) }
        ovAsr?.let { s = s.copy(asrModel = it) }
        ovPunct?.let { s = s.copy(asrPunct = it) }
        ovDenoise?.let { s = s.copy(asrDenoise = it) }
        ovCtx?.let { s = s.copy(contextRounds = it.coerceIn(0, 10)) }
        ovNetApi?.let { s = s.copy(netApiEnabled = it) }
        ovApiKey?.let { s = s.copy(netApiKey = it) }
        ovNetBase?.let { s = s.copy(netApiBaseUrl = it) }
        ovNetModel?.let { s = s.copy(netApiModel = it) }
        ovThinking?.let { s = s.copy(netApiThinking = it) }
        return s
    }

    /**
     * 读一个「可能是布尔」的 adb extra。
     *
     * 为什么不能直接用 `getBooleanExtra`：`am start --es netapi true` 塞进去的是**字符串**
     * `"true"`，`getBooleanExtra` 对它只返回默认值（内部 ClassCast 被吞掉，只在 logcat 里
     * 留一句 W/Bundle）。本项目为此踩过一次：开关传了 `--es`，结果开关一直是关的，
     * 却要排查半天「为什么还是加载了本地模型」。
     *
     * 所以这里把 Boolean / String / 数字三种形态都认了，`--es` 和 `--ez` 都能用。
     */
    private fun boolExtra(i: Intent, name: String): Boolean? {
        if (!i.hasExtra(name)) return null
        return when (val v = i.extras?.get(name)) {
            is Boolean -> v
            is String -> v.trim().equals("true", true) || v.trim() == "1"
            is Number -> v.toInt() != 0
            else -> null
        }
    }

    private fun parseOverrides() {
        val i = intent ?: return
        ovProvider = i.getStringExtra("provider")?.trim()?.takeIf { it.isNotEmpty() }
        ovDevice = i.getStringExtra("device")?.trim()?.takeIf { it.isNotEmpty() }
        ovModelDir = i.getStringExtra("llamamodel")?.trim()?.takeIf { it.isNotEmpty() }
        ovNCtx = i.getIntExtra("nctx", 0).takeIf { it > 0 }
        ovNThreads = i.getIntExtra("nthreads", 0).takeIf { it > 0 }
        ovMaxNew = i.getIntExtra("maxnew", 0).takeIf { it > 0 }
        // A/B 测试用：--es asr paraformer-stream，一次跑一个后端，不写回 DataStore
        ovAsr = i.getStringExtra("asr")?.trim()?.takeIf { it.isNotEmpty() }
        ovPunct = boolExtra(i, "punct") ?: if (i.hasExtra("punct")) true else null
        // --es denoise true 走 GTCRN 降噪（只影响本实例，不写盘）
        ovDenoise = boolExtra(i, "denoise") ?: if (i.hasExtra("denoise")) true else null
        // `--es wlang auto` 传的是字符串 "auto"，`--es wlang ""` 在 shell 里会被吞掉，
        // 所以统一用 "auto" 表示「不指定」，空串也当 auto。
        ovWhisperLang = if (i.hasExtra("wlang")) {
            (i.getStringExtra("wlang") ?: "auto").trim().ifEmpty { "auto" }
        } else {
            null
        }
        ovOverlay = boolExtra(i, "overlay")
        // --ei ctx 4：带 4 轮上文。用 hasExtra 判，因为 0 是有效值（无上下文）。
        ovCtx = if (i.hasExtra("ctx")) i.getIntExtra("ctx", 0) else null
        ovColorDialog = i.getStringExtra("colordialog")?.trim()?.takeIf { it == "text" || it == "box" }
        // 「网络API调用」的 adb 口子：--es netapi true --es apikey sk-xxx --es netmodel deepseek-v4.1-flash
        ovNetApi = boolExtra(i, "netapi")
        ovApiKey = i.getStringExtra("apikey")?.trim()?.takeIf { it.isNotEmpty() }
        ovNetBase = i.getStringExtra("netbase")?.trim()?.takeIf { it.isNotEmpty() }
        ovNetModel = i.getStringExtra("netmodel")?.trim()?.takeIf { it.isNotEmpty() }
        ovThinking = i.getStringExtra("thinking")?.trim()?.takeIf { it.isNotEmpty() }
        if (listOf(ovProvider, ovDevice, ovModelDir).any { it != null } ||
            listOf(ovNCtx, ovNThreads, ovMaxNew).any { it != null } || ovNetApi != null
        ) {
            android.util.Log.i(
                TAG,
                "#$instanceId 覆盖设置：provider=${ovProvider ?: "-"} device=${ovDevice ?: "-"} " +
                    "nCtx=${ovNCtx ?: "-"} maxNew=${ovMaxNew ?: "-"} asr=${ovAsr ?: "-"} " +
                    "netapi=${ovNetApi ?: "-"} netmodel=${ovNetModel ?: "-"} " +
                    "key=${if (ovApiKey != null) "已给" else "-"}",
            )
        }
    }

    /** 状态行文案。**必须与 [makeTranslatorFactory] 的实际选择一致**，否则又是「按钮和后台对不上」。 */
    private fun translatorStatusText(): String =
        InterpretSession.translatorStatusText(effectiveSettings())

    // ------------------------------------------------------------ 悬浮字幕窗

    /** 把落盘的设置同步进 Compose 状态（否则界面控件停在默认值上，用户以为没保存）。 */
    private fun syncSubtitleUi(s: Settings) {
        uiSubtitleOn.value = s.subtitleEnabled
        uiSubtitleTextSize.value = s.textSizeSp
        uiSubtitleAlpha.value = ((s.boxColor ushr 24) and 0xFF) / 255f
        uiSubtitleTextColor.value = s.textColor
        uiSubtitleBoxColor.value = s.boxColor
        uiSubtitleMode.value = s.subtitleMode
    }

    /** 「显示 / 关闭浮窗」按钮。没权限就跳系统授权页，回来由 [onResume] 补上。 */
    private fun toggleSubtitle() {
        if (uiSubtitleOn.value) {
            uiSubtitleOn.value = false
            appSettings = appSettings.copy(subtitleEnabled = false)
            persistSettings { it.copy(subtitleEnabled = false) }
            InterpretSession.hideOverlay()
            append("悬浮字幕窗已关闭")
            return
        }
        if (!SubtitleOverlay.canDraw(this)) {
            append("浮窗需要「显示在其他应用上层」权限，正在跳转系统设置…")
            pendingOverlayPermission = true
            val ok = runCatching {
                startActivity(
                    Intent(
                        SysSettings.ACTION_MANAGE_OVERLAY_PERMISSION,
                        Uri.parse("package:$packageName"),
                    )
                )
            }.isSuccess
            if (!ok) {
                pendingOverlayPermission = false
                append(
                    "❌ 打不开授权页，请手动到「设置 → 应用管理 → 屏幕同传 → 其他权限 → " +
                        "显示在其他应用上层」开启"
                )
            }
            return
        }
        uiSubtitleOn.value = true
        appSettings = appSettings.copy(subtitleEnabled = true)
        persistSettings { it.copy(subtitleEnabled = true) }
        showSubtitleOverlay()
    }

    private fun showSubtitleOverlay() {
        // 窗口本体、位置落盘回调都在 Session 里建（它比界面活得久）。
        InterpretSession.updateSettings(appSettings)
        if (InterpretSession.showOverlay()) {
            append("✅ 悬浮字幕窗已显示（拖顶部条移动，拖右下角 ◢ 改大小）")
        } else {
            // 典型成因：权限没给、或悬浮层被系统的「禁止后台弹出界面」挡住。
            uiSubtitleOn.value = false
            append("❌ 浮窗没能加上：${InterpretSession.overlay?.lastError ?: "未知原因"}")
        }
    }

    /**
     * 字号 / 不透明度 / 字色 / 框底色：先立刻应用到浮窗，再按需落盘。
     *
     * 滑块拖动过程中**不写盘**（`persist = false`）：每动一像素就 write DataStore，
     * 既卡手又没必要 —— 在 `onValueChangeFinished` 里落一次就够了。
     */
    private fun applySubtitleSettings(persist: Boolean) {
        val alpha = (uiSubtitleAlpha.value * 255f).toInt().coerceIn(0, 255)
        val box = (alpha shl 24) or (uiSubtitleBoxColor.value and 0x00FFFFFF)
        val size = uiSubtitleTextSize.value
        val color = uiSubtitleTextColor.value
        appSettings = appSettings.copy(textSizeSp = size, boxColor = box, textColor = color)
        InterpretSession.updateSettings(appSettings)
        if (persist) {
            persistSettings { it.copy(textSizeSp = size, boxColor = box, textColor = color) }
        }
    }

    // ------------------------------------------------------------ 后端切换

    /**
     * 切换浮窗显示内容（双语 / 仅译文 / 仅原文）。
     *
     * 只影响**显示**：识别和翻译照旧全跑，所以随时切回来都不缺内容 —— 不像
     * 「关掉翻译」那样会改变引擎行为。
     */
    private fun pickSubtitleMode(id: String) {
        if (uiSubtitleMode.value == id) return
        uiSubtitleMode.value = id
        appSettings = appSettings.copy(subtitleMode = id)
        persistSettings { it.copy(subtitleMode = id) }
        InterpretSession.updateSettings(appSettings)
        append("浮窗显示：${SUBTITLE_MODES.firstOrNull { it.first == id }?.second ?: id}")
    }

    /**
     * 切换翻译后端并**立刻落盘**。
     *
     * 三件事一件都不能少：
     * 1. 改 [appSettings]（本实例立刻生效）
     * 2. 写 DataStore（下次启动仍然生效 —— 只改内存的话用户会以为「设置没保存」）
     * 3. 让引擎丢掉缓存的 provider（否则新后端永远不会被构造，见 [rebindTranslator]）
     *
     * 顺带把 [overrideActive] 关掉：用户既然在界面上点了，adb 传进来的临时值就该让位。
     */
    private fun pickProvider(id: String) {
        if (id != "openai") lastLocalProvider = id
        // 选 NLLB 时先看四个文件在不在（用户 m07030：缺模型要提示并给下载）。
        val prevProvider = appSettings.providerId
        if (id == "nllb" && NllbModels.resolve(this) == null) {
            askModel(
                label = "端侧 NLLB",
                rels = listOf(
                    "nllb/onnx/encoder_model_int8.onnx",
                    "nllb/onnx/decoder_model_quantized.onnx",
                    "nllb/onnx/decoder_with_past_model_quantized.onnx",
                    "nllb/tokenizer.bin",
                ),
                note = "三个 onnx 能直接下载；tokenizer.bin 要从随包 Models/nllb/ 里拿（镜像站按钮去上游找）。",
                revert = { if (prevProvider != "nllb") pickProvider(prevProvider) },
            )
        }
        if (uiProviderId.value == id) return
        if (uiStarting.value || uiCaptureRunning.value) {
            append("⚠ 采集中不能切换翻译后端：先点「停止」再换")
            return
        }
        uiProviderId.value = id
        overrideActive = false
        // 选「云端」= 以前的「网络API调用」开关。两个状态一起写，避免出现
        // providerId=openai 但 netApiEnabled=false（那会让工厂退回本地后端）。
        val cloud = id == "openai"
        uiNetApi.value = cloud
        appSettings = appSettings.copy(providerId = id, netApiEnabled = cloud)
        persistSettings { it.copy(providerId = id, netApiEnabled = cloud) }
        rebindTranslator()
        append(
            if (cloud) "翻译后端 → 云端 API（翻译走 HTTP；本地大模型 / NLLB 都还在，随时可切回）"
            else "翻译后端 → ${PROVIDERS.firstOrNull { it.first == id }?.second ?: id}（已保存）",
        )
    }

    /** 切换大模型算力位置。只有 NPU 打不开时才需要手动选 CPU（`auto` 会自动退）。 */
    private fun pickDevice(id: String) {
        if (uiLlamaDevice.value == id) return
        uiLlamaDevice.value = id
        overrideActive = false
        appSettings = appSettings.copy(llamaDevice = id)
        persistSettings { it.copy(llamaDevice = id) }
        rebindTranslator()
        append("大模型后端 → ${DEVICES.firstOrNull { it.first == id }?.second ?: id}（已保存）")
    }

    /**
     * 切换识别后端。
     *
     * 这个操作**必须**把已建好的引擎整个丢掉：模型是在 [AsrFactory.create] 里
     * 加载进 native 内存的，只改一个字符串不会让老模型消失。不丢的话会出现
     * 「设置里写着 Paraformer，实际跑的还是 SenseVoice」，而且内存里两套模型同时驻留。
     *
     * 同理不能悄悄重建引擎：重建要重新加载 230MB，在用户毫无预期的情况下卡两秒，
     * 看起来就像卡死。所以这里只释放，等用户自己点「开始采集」——
     * 界面上那句文案就是为这件事写的。
     */
    private fun pickAsrModel(id: String) {
        val prevKind = uiAsrModel.value
        val kind = AsrFactory.normalize(id)
        if (kind == uiAsrModel.value) return
        uiAsrModel.value = kind
        appSettings = appSettings.copy(asrModel = kind)
        persistSettings { it.copy(asrModel = kind) }

        // 摘监听 → 关引擎（顺序反了会有一次回调打到已释放的 recognizer）。
        // 走 Session：只有引擎重建，浮窗保留。
        InterpretSession.detachEngine()
        append("识别后端 → ${AsrFactory.labelOf(kind)}（旧模型已释放，重新点「开始采集」加载新模型）")
        if (AsrFactory.resolveRootFor(kind, filesDir, getSharedPreferences("asr_debug", MODE_PRIVATE).getString("asrDir", "") ?: "") == null) {
            askModel(
                label = "语音模型 ${AsrFactory.labelOf(kind)}",
                rels = when (kind) {
                    AsrFactory.SENSEVOICE -> listOf("asr/model.int8.onnx", "asr/tokens.txt")
                    AsrFactory.PARAFORMER ->
                        listOf("asr/paraformer-zh/model.int8.onnx", "asr/paraformer-zh/tokens.txt")
                    AsrFactory.STREAMING -> listOf(
                        "asr/streaming-paraformer/encoder.int8.onnx",
                        "asr/streaming-paraformer/decoder.int8.onnx",
                        "asr/streaming-paraformer/tokens.txt",
                    )
                    AsrFactory.WHISPER -> listOf(
                        "asr/whisper-small/small-encoder.int8.onnx",
                        "asr/whisper-small/small-decoder.int8.onnx",
                        "asr/whisper-small/small-tokens.txt",
                    )
                    else -> emptyList()
                },
                note = "这些文件在 models/ 的 ${AsrFactory.subdirOf(kind)} 下；缺了会在点「开始采集」时报错。",
                revert = { if (prevKind != kind) pickAsrModel(prevKind) },
            )
        }
    }

    /** 补标点开关。改完同样要重建引擎才会生效（标点模型在构造后端时一起加载）。 */
    private fun pickAsrPunct(on: Boolean) {
        if (uiAsrPunct.value == on) return
        uiAsrPunct.value = on
        appSettings = appSettings.copy(asrPunct = on)
        persistSettings { it.copy(asrPunct = on) }
        InterpretSession.detachEngine()
        append("自动补标点 → ${if (on) "开" else "关"}（重新点「开始采集」生效）")
    }

    /**
     * 识别前降噪开关。
     *
     * 和补标点一样要重建引擎才生效 —— 降噪器是构造后端时一起加载的。
     * 之所以不让它「即时生效」，是因为降噪器要跟后端的解码生命周期绑定，
     * 换句话说得和后端同生共死，热插拔只会制造出「一半降噪一半不降噪」的怪状态。
     */
    private fun pickAsrDenoise(on: Boolean) {
        if (uiAsrDenoise.value == on) return
        uiAsrDenoise.value = on
        appSettings = appSettings.copy(asrDenoise = on)
        persistSettings { it.copy(asrDenoise = on) }
        InterpretSession.detachEngine()
        append("识别前降噪 → ${if (on) "开" else "关"}（重新点「开始采集」生效）")
    }

    // ------------------------------------------------------------ 云端 API 的动作
    //
    // 2026-10-18 之后这里没有「开关」了：选「云端」就等于以前的开关打开，
    // 状态由 [pickProvider] 一起写（providerId=openai ⇔ netApiEnabled=true）。

    /** 把网络 API 那四个输入框落盘。输入框每敲一个字只改状态，退出/保存/开始采集时才写盘。 */
    private fun saveNetApi(announce: Boolean = true) {
        val key = uiNetApiKey.value.trim()
        val base = uiNetBase.value.trim()
        // 模型名空着就存空：不猜默认值（猜错会变成服务端 400，且掩盖「没填」这个事实）
        val model = uiNetModel.value.trim()
        val thinking = uiNetThinking.value
        appSettings = appSettings.copy(
            netApiKey = key, netApiBaseUrl = base, netApiModel = model, netApiThinking = thinking,
        )
        persistSettings {
            it.copy(netApiKey = key, netApiBaseUrl = base, netApiModel = model, netApiThinking = thinking)
        }
        if (announce) {
            append(
                if (model.isEmpty()) "网络API设置已保存（⚠ 模型名称还是空的，翻译会报错）"
                else "网络API设置已保存：模型 $model，思考强度 $thinking",
            )
        }
    }

    /**
     * 「测试连接」：把「Key 存没存」「地址通不通」「模型名认不认」三件事分开报。
     *
     * 用户填完 Key 却一句译文都没有时，光看状态行分不清是哪一环坏了 ——
     * 这个按钮先发一次 `max_tokens=1` 的 ping 看 HTTP 状态，再真翻一句把
     * 服务端返回的错误原文（例如 `Model Not Exist`）打出来。
     */
    /**
     * 无头 OCR：`--es auto ocrfile:/sdcard/x.png`。
     *
     * 为什么要这个口子：识图页要点两次（选图 → 相册里选张图），而真机上浮窗与
     * MIUI 防误触都可能把触摸吃掉；无人值守回归（尤其竖排/横排这种纯算法差异）
     * 只能靠这条路径。结果直接进日志，`adb logcat -s AST.Ocr AST.Interp` 就能验收。
     */
    private fun runOcrOnFile(path: String) {
        // 无头入口没有 Composable 的设置流，用「想要的没装就退到已装的」规则：
        // 手机上有 server 权重时自动用高精度那套，没有就用 mobile。
        val files = com.screenast.app.vision.VisionModels
            .resolveOrFallback(this, com.screenast.app.vision.VisionModels.Variant.MOBILE)?.first
        if (files == null) {
            append("❌ OCR：模型不在 files/models/ocr/（先跑 tools/push_ocr_models.sh）")
            return
        }
        lifecycleScope.launch {
            val out = withContext(Dispatchers.IO) {
                runCatching {
                    val bmp = android.graphics.BitmapFactory.decodeFile(path)
                        ?: return@runCatching null
                    val engine = com.screenast.app.vision.OcrEngine(applicationContext, files.det, files.rec)
                    try {
                        bmp to engine.recognize(bmp)
                    } finally {
                        engine.close()
                        bmp.recycle()
                    }
                }.getOrElse {
                    append("❌ OCR 失败：${it.javaClass.simpleName}: ${it.message}")
                    null
                }
            }
            if (out == null) {
                append("❌ OCR：读不到图片 $path")
                return@launch
            }
            val (bmp, list) = out
            append("OCR 文件：$path（${bmp.width}x${bmp.height}）→ ${list.size} 块")
            list.forEachIndexed { i, b ->
                append("  [${i + 1}]${if (b.vertical) "竖" else "横"} " +
                    "(${b.left.toInt()},${b.top.toInt()} ${b.width.toInt()}x${b.height.toInt()}) " +
                    "%.2f".format(b.score) + "  ${b.text}")
            }
        }
    }

    /**
     * `--es auto visiontest`：不开投屏、不碰界面，直接往 [com.screenast.app.vision.VisionSession]
     * 里灌一张**代码画出来**的图（左边竖排日文、右边横排中文），把「会话 → OCR → 翻译 → 状态」
     * 这条链路跑一遍。
     *
     * 为什么要有这个口子：真正的抓屏要过系统授权弹窗（防误触模式/锁屏时 adb 点不进去），
     * 而 VirtualDisplay 之外的部分（过滤阈值、译文缓存、busy 丢帧、界面列表）完全可以用
     * 一张构造图验证。抓屏本身另测。
     */
    private fun runVisionTest() {
        // 无头入口没有 Composable 的设置流，用「想要的没装就退到已装的」规则：
        // 手机上有 server 权重时自动用高精度那套，没有就用 mobile。
        val files = com.screenast.app.vision.VisionModels
            .resolveOrFallback(this, com.screenast.app.vision.VisionModels.Variant.MOBILE)?.first
        if (files == null) {
            append("❌ 识图：模型不在 files/models/ocr/（先跑 tools/push_ocr_models.sh）")
            return
        }
        val bmp = android.graphics.Bitmap.createBitmap(900, 1200, android.graphics.Bitmap.Config.ARGB_8888)
        val c = android.graphics.Canvas(bmp)
        c.drawColor(android.graphics.Color.WHITE)
        val p = android.graphics.Paint().apply {
            isAntiAlias = true
            color = android.graphics.Color.BLACK
            textSize = 44f
        }
        // 竖排：逐字往下画（日文）
        val vert = "今日はいい天気ですね"
        vert.forEachIndexed { i, ch ->
            c.drawText(ch.toString(), 760f, 120f + i * 52f, p)
        }
        val vert2 = "明日は雨が降るらしい"
        vert2.forEachIndexed { i, ch ->
            c.drawText(ch.toString(), 660f, 120f + i * 52f, p)
        }
        // 横排：中文一行（对照）
        p.textSize = 40f
        c.drawText("这是横排中文，用来对照", 60f, 1000f, p)
        c.drawText("Hello world", 60f, 1070f, p)
        append("识图自测：灌一张 ${bmp.width}x${bmp.height} 的构造图（竖排日文 + 横排中文）")
        // `--ez vtnet true`：把翻译临时指到本机 mock（`adb reverse tcp:8099`），验证完还原。
        // 端侧 NLLB 在识图页会和 OCR 抢内存（实测 RSS 5GB、装载 73 秒），拿它做回归会把
        // 「功能对不对」和「这台机器内存够不够」两件事混在一起。
        val useMock = intent?.getBooleanExtra("vtnet", false) == true
        lifecycleScope.launch {
            val cur = SettingsStore(this@InterpretActivity).settings.first()
            vtnetBackup = cur
            if (useMock) {
                SettingsStore(this@InterpretActivity).update {
                    it.copy(
                        netApiEnabled = true,
                        netApiBaseUrl = "http://127.0.0.1:8099/v1",
                        netApiKey = "test-key",
                        netApiModel = "mock-model",
                        visionTranslate = true,
                    )
                }
                append("识图自测：翻译临时指到 mock（127.0.0.1:8099/v1，model=mock-model），20 秒后自动还原")
            }
            append(
                "识图设置：translate=${cur.visionTranslate} target=${cur.visionTargetLang.code} " +
                    "source=${cur.visionSourceLang.code} interval=${cur.visionIntervalMs} " +
                    "provider=${cur.providerId} netApi=${cur.netApiEnabled}",
            )
        }
        // 自动还原：以前手工跑一遍 mock，会把用户真实填的 Key/模型名覆盖成测试值（踩过：
        // 覆盖后模型名丢了，界面显示「翻译后端：未启用」）。现在改完 20 秒一定写回去。
        if (useMock) {
            lifecycleScope.launch {
                kotlinx.coroutines.delay(20_000)
                val b = vtnetBackup
                if (b != null) {
                    SettingsStore(this@InterpretActivity).update {
                        it.copy(
                            netApiEnabled = b.netApiEnabled,
                            netApiBaseUrl = b.netApiBaseUrl,
                            netApiKey = b.netApiKey,
                            netApiModel = b.netApiModel,
                            visionTranslate = b.visionTranslate,
                        )
                    }
                    append("识图自测：翻译设置已还原（开关=${b.netApiEnabled} 模型=${b.netApiModel}）")
                }
            }
        }
        com.screenast.app.vision.VisionSession.onFrame(applicationContext, bmp)

        // 等 OCR + 翻译（2B 首次装载要几秒），再把会话状态打出来
        lifecycleScope.launch {
            repeat(60) {
                kotlinx.coroutines.delay(500)
                val n = com.screenast.app.vision.VisionSession.lines.value.size
                val st = com.screenast.app.vision.VisionSession.status.value
                if (n > 0 && st.isNotEmpty() && it > 6) {
                    append("识图状态：$st")
                    com.screenast.app.vision.VisionSession.lines.value.forEachIndexed { i, l ->
                        append("  [${i + 1}]${if (l.vertical) "竖" else "横"} " +
                            "(${l.left.toInt()},${l.top.toInt()} ${l.width.toInt()}x${l.height.toInt()}) " +
                            "%.2f".format(l.score) + "  ${l.text}" +
                            (if (l.translation.isNotEmpty()) "  →  ${l.translation}" else ""))
                    }
                    append("识图自测结束：${n} 块")
                    return@launch
                }
            }
            append("识图自测超时：${com.screenast.app.vision.VisionSession.status.value}")
        }
    }

    private fun testNetApi() {        // 输入框当前值可能还没落盘，先存一次，保证测的就是要用的那份。
        saveNetApi(announce = false)
        val st = effectiveSettings().copy(netApiEnabled = true)
        val model = st.netApiModel.trim()
        val base = st.netApiBaseUrl.trim()
        val out = mutableListOf<String>()
        fun fail(msg: String) {
            append("❌ 测试连接：$msg")
            out += msg
            uiApiTest.value = ApiTestReport(ok = false, lines = out.toList())
        }
        if (st.netApiKey.isBlank()) {
            fail("API Key 是空的 —— 在输入框里填好，再点「保存 API 设置」")
            return
        }
        if (model.isEmpty()) {
            fail("模型名称是空的 —— 照服务商文档把 model 字段的名字填进去")
            return
        }
        if (base.isEmpty()) {
            fail("接口地址是空的 —— 填服务商的 OpenAI 兼容 base URL（一般以 /v1 结尾）")
            return
        }
        append("网络API 测试：$model @ $base（思考：${st.netApiThinking}）")
        lifecycleScope.launch {
            var connOk = false
            val p = OpenAiCompatibleProvider(st)
            when (val s = runCatching { p.checkAvailability() }.getOrElse {
                val m = "连通性测试异常：${it.javaClass.simpleName}: ${it.message}"
                append("❌ $m")
                out += m
                uiApiTest.value = ApiTestReport(ok = false, lines = out.toList())
                return@launch
            }) {
                is ProviderStatus.Ok -> {
                    connOk = true
                    val m = "鉴权通过：${s.detail}"
                    append("✅ 连通性：${s.detail}")
                    out += m
                }
                is ProviderStatus.Error -> {
                    val m = "连通性：${s.message}"
                    append("❌ $m")
                    out += m
                }
                ProviderStatus.NotConfigured -> {
                    val m = "连通性：未配置 Key"
                    append("❌ $m")
                    out += m
                }
            }
            append("正在试翻一句（验证模型名与参数）…")
            val sb = StringBuilder()
            var trOk = false
            runCatching {
                p.translate(TranslateRequest("Hello, how are you?", Lang.EN, st.targetLang))
                    .collect { sb.setLength(0); sb.append(it.text) }
            }.onSuccess {
                if (sb.isEmpty()) {
                    append("❌ 翻译测试：返回空译文")
                    out += "翻译测试：返回空译文"
                } else {
                    trOk = true
                    append("✅ 翻译测试：$sb")
                    out += "试翻一句 →「$sb」"
                }
            }.onFailure {
                append("❌ 翻译测试：${it.message}")
                out += "翻译测试：${it.message}"
            }
            runCatching { p.close() }
            uiApiTest.value = ApiTestReport(ok = connOk && trOk, lines = out.toList())
        }
    }

    private fun pickNetThinking(t: String) {
        if (uiNetThinking.value == t) return
        uiNetThinking.value = t
        saveNetApi(announce = false)
        rebindTranslator()
        append("思考强度 → ${Settings.NET_API_THINKING.firstOrNull { it.first == t }?.second ?: t}")
    }

    private fun persistSettings(transform: (Settings) -> Settings) {
        lifecycleScope.launch {
            runCatching { SettingsStore(this@InterpretActivity).update(transform) }
                .onFailure { append("⚠ 保存设置失败：${it.message}") }
        }
    }

    /**
     * 让引擎按新设置重建翻译后端。
     *
     * 顺序不能反：先换 `translatorFactory`，再 [InterpretEngine.resetProvider]。
     * 反过来的话引擎可能在两者之间就把旧工厂又拿来用了一次。
     */
    private fun rebindTranslator() {
        val e = engine
        if (e == null) {
            // 引擎还没建（用户还没按过「开始采集」）。这时候**不能装作切成功了**：
            // 之前的写法在这里直接 return，结果状态行还停在旧后端，用户以为自己没点中。
            // 状态行照样更新，但要说清「什么时候生效」。
            uiInterpState.value = translatorStatusText() + " · 开始采集后加载"
            append("已切换，下一次「开始采集」时生效")
            return
        }

        // 切走大模型时**必须显式释放**：`LlamaTranslationProvider.close()` 是空实现
        // （模型是进程级单例，被调试页和正式 provider 共享），所以 `resetProvider()`
        // 关不掉那 1.4GB。不释放的话切到 NLLB 后两份权重同时驻留 —— NLLB 自己就要
        // 3.5GB native，16GB 的机器会开始靠 swap 撑，那正是之前把手机压到 load 65 的状态。
        // 打开「网络API调用」同理：它的卖点就是「不加载本地 LLM」，还留着 1.4GB 就白搭。
        if ((appSettings.providerId != "llama" || appSettings.netApiEnabled) && LlamaRuntime.isLoaded) {

            LlamaRuntime.release()
            append("已释放大模型，腾出约 1.4GB")
        }

        if (!uiTranslateOn.value) {
            append("（翻译未开启，打开后即用新后端）")
            return
        }
        val f = makeTranslatorFactory()
        if (f == null) {
            append("⚠ 新后端找不到模型，保持原后端不动")
            return
        }
        e.translatorFactory = f
        e.resetProvider()
        uiInterpState.value = translatorStatusText() + " · 待重建"
    }
}

// ---------------------------------------------------------------------------
// RGB 调色板
// ---------------------------------------------------------------------------
//
// 为什么不用系统 ColorPicker：Android 没有稳定的公共取色 API
// （AOSP 那个 `ColorPickerDialog` 是隐藏类，各 ROM 实现还不一样）。
// 三个 0..255 的滑块 + 实时预览是唯一在所有 ROM 上都成立的做法，
// 而且它把「颜色」这件事说得很清楚：用户看到的 RGB 数字就是最终写进设置的值。

/**
 * RGB 取色弹窗。[initial] 只取低 24 位（RGB），alpha 由 [previewAlpha] 决定 ——
 * 框底色的透明度是独立滑块管的，不应该在这里被改掉。
 */
@Composable
private fun ColorPickerDialog(
    title: String,
    initial: Int,
    previewAlpha: Int,
    onDismiss: () -> Unit,
    onPick: (Int) -> Unit,
) {
    // key 用 initial：同一个弹窗换目标（字体色 ↔ 框底色）时必须回到新初值，
    // 否则 remember 会把上一次调的颜色带过来，看着像「点了没反应」。
    var r by remember(initial) { mutableStateOf((initial ushr 16) and 0xFF) }
    var g by remember(initial) { mutableStateOf((initial ushr 8) and 0xFF) }
    var b by remember(initial) { mutableStateOf(initial and 0xFF) }
    val rgb = (r shl 16) or (g shl 8) or b

    AlertDialog(
        onDismissRequest = onDismiss,
        title = { Text(title) },
        text = {
            Column(verticalArrangement = Arrangement.spacedBy(2.dp)) {
                Box(
                    Modifier.fillMaxWidth().height(44.dp).background(
                        Color((previewAlpha shl 24) or rgb),
                        RoundedCornerShape(8.dp),
                    ),
                )
                ColorChannelSlider("红 R", r) { r = it }
                ColorChannelSlider("绿 G", g) { g = it }
                ColorChannelSlider("蓝 B", b) { b = it }
                Text(
                    "#%02X%02X%02X".format(r, g, b),
                    fontSize = 12.sp,
                    color = Color(0xFF9E9EA7),
                    fontFamily = FontFamily.Monospace,
                )
            }
        },
        confirmButton = { TextButton(onClick = { onPick(rgb) }) { Text("确定") } },
        dismissButton = { TextButton(onClick = onDismiss) { Text("取消") } },
    )
}

@Composable
private fun ColorChannelSlider(label: String, value: Int, onChange: (Int) -> Unit) {
    Row(
        Modifier.fillMaxWidth(),
        verticalAlignment = Alignment.CenterVertically,
        horizontalArrangement = Arrangement.spacedBy(6.dp),
    ) {
        Text(label, fontSize = 12.sp, color = Color(0xFFB9B7C4), modifier = Modifier.width(46.dp))
        Slider(
            value = value.toFloat(),
            valueRange = 0f..255f,
            // 255 个刻度的语义：steps = 区间数 - 1 = 254。不设的话只能拖到整十，
            // 用户会觉得「调不准」。
            steps = 254,
            modifier = Modifier.weight(1f),
            onValueChange = { onChange(it.toInt().coerceIn(0, 255)) },
        )
        Text(
            value.toString(),
            fontSize = 12.sp,
            color = Color(0xFFB9B7C4),
            fontFamily = FontFamily.Monospace,
            modifier = Modifier.width(30.dp),
        )
    }
}

/**
 * 「模型要求」弹窗 —— 给想自己找模型的人看的规格页。
 *
 * 清单直接从 [ModelImporter.items] 来（同一个来源，改了清单这页自动跟着变），
 * 每行标出：路径、要多大、给哪个能力用、必备还是可选、现在有没有装好。
 */
@Composable
private fun ModelRequirementDialog(
    items: List<ModelImporter.Item>,
    modelsDir: File,
    onDismiss: () -> Unit,
) {
    val present = remember(items, modelsDir) {
        items.associate { it.rel to File(modelsDir, it.rel).isFile }
    }
    val essential = items.filter { it.essential }
    val optional = items - essential

    fun mb(bytes: Long): String = "%.1fMB".format(bytes / 1048576.0)
    fun gb(bytes: Long): String = "%.2fGB".format(bytes / 1073741824.0)

    AlertDialog(
        onDismissRequest = onDismiss,
        title = { Text("模型要求") },
        text = {
            Column(
                Modifier.heightIn(max = 560.dp).verticalScroll(rememberScrollState()),
                verticalArrangement = Arrangement.spacedBy(6.dp),
            ) {
                Text(
                    "软件不内置下载、也不校验来源，只认文件名 + 大小 + 能不能加载。" +
                        "所以自己找的模型只要满足下面这些，一样能用。",
                    fontSize = 12.sp, color = Color(0xFFB9B7C4),
                )

                Text("一、硬规矩（不满足就一定会报错）", fontSize = 13.sp,
                    color = MaterialTheme.colorScheme.primary)
                ModelHelpLine("① 文件名必须一字不差：大小写、连字符、扩展名都算。别留 \"xxx (1).onnx\"、别改名。")
                ModelHelpLine("② 文件大小要和清单差 ≤ 2%。换量化档（Q4_K_M ↔ Q8_0）也算不符，会显示「未就位」。")
                ModelHelpLine("③ ONNX 要是未加密、单文件的 float32/int8 导出。外部权重（.onnx.data）、加密包、TensorRT 引擎都不认。")
                ModelHelpLine("④ 子目录名不能改：Models/asr、Models/ocr、Models/nllb/onnx、Models/llama、Models/inpaint。")
                ModelHelpLine("⑤ 一套模型要成套给：OCR 的 det+rec 同代、SenseVoice 三件（模型+词表+VAD）、NLLB 三件同源。")

                Text("二、最少能用（必备 ${essential.size} 个 · ${gb(essential.sumOf { it.bytes })}）",
                    fontSize = 13.sp, color = MaterialTheme.colorScheme.primary)
                essential.forEach { ModelHelpRow(it, present[it.rel] == true, ::mb) }

                Text("三、可选（${optional.size} 个 · 加起来再占 ${gb(optional.sumOf { it.bytes })}，缺了只是少个后端/档位）",
                    fontSize = 13.sp, color = MaterialTheme.colorScheme.primary)
                optional.forEach { ModelHelpRow(it, present[it.rel] == true, ::mb) }

                Text("四、各后端的额外要求", fontSize = 13.sp,
                    color = MaterialTheme.colorScheme.primary)
                ModelHelpLine("翻译（gguf）：llama.cpp 能加载的结构即可，Q4_K_M / Q8_0 / F16 都行；只支持单文件 gguf，分片（-00001-of-00002）不认。")
                ModelHelpLine("翻译（NLLB）：encoder/decoder/decoder_with_past 三件必须同一版本；词表 tokenizer.bin 已内嵌在 APK。")
                ModelHelpLine("识别（ASR）：SenseVoice 要 model.int8.onnx + tokens.txt；paraformer-zh、streaming-paraformer、whisper-small 各自成套（模型 + 对应词表）。")
                ModelHelpLine("断句/降噪：silero_vad.onnx 与 ten-vad.onnx 二选一（有 ten-vad 优先用）；gtcrn/gtcrn_simple.onnx 是降噪，独立生效。")
                ModelHelpLine("OCR：mobile 与 server 的 det/rec 不可混用（混了会乱码）；词典 ppocrv5_dict.txt 已内嵌，不用给。")
                ModelHelpLine("擦除补背景：MI-GAN 的 ONNX pipeline（含裁框/羽化），不导入就退回「程序贴片」（用框周围平均色盖住原文）。")

                Text("五、放哪儿", fontSize = 13.sp, color = MaterialTheme.colorScheme.primary)
                ModelHelpLine("方式 A：整个 Models 文件夹拷到手机存储根目录 → 这个页面点「从 Models 导入」。")
                ModelHelpLine("方式 B：随便放哪 → 点「选 Models 文件夹」手动指过去。")
                ModelHelpLine("方式 C：压成一个 .zip（里面保留 Models/ 这层）→ 点「选 .zip」。")
                ModelHelpLine("导入后都在应用私有目录（卸载才清），源文件删掉不影响。")

                Text(
                    "当前：必备 ${essential.count { present[it.rel] == true }}/${essential.size}，" +
                        "全部 ${items.count { present[it.rel] == true }}/${items.size}",
                    fontSize = 12.sp, color = Color(0xFF8EDDA1),
                )
            }
        },
        confirmButton = { TextButton(onClick = onDismiss) { Text("知道了") } },
    )
}

@Composable
private fun ModelHelpLine(text: String) {
    Text(text, fontSize = 11.sp, color = Color(0xFF9E9EA7))
}

@Composable
private fun ModelHelpRow(item: ModelImporter.Item, installed: Boolean, mb: (Long) -> String) {
    Row(Modifier.fillMaxWidth(), verticalAlignment = Alignment.Top) {
        Text(
            if (installed) "✔" else "✘",
            fontSize = 12.sp,
            color = if (installed) Color(0xFF8EDDA1) else Color(0xFFFFC66B),
            modifier = Modifier.width(18.dp),
        )
        Column(Modifier.weight(1f)) {
            Text(
                item.rel,
                fontSize = 11.sp,
                color = Color(0xFFD6D4DE),
                fontFamily = FontFamily.Monospace,
            )
            Text(
                "${mb(item.bytes)} · ${item.note}" + if (item.essential) " · 必备" else "",
                fontSize = 10.sp, color = Color(0xFF9E9EA7),
            )
        }
    }
}

/**
 * 「测试连接」的一次结果：`ok` 决定标题的勾/叉，`lines` 是逐条结论（鉴权、模型名、试翻）。
 */
data class ApiTestReport(
    val ok: Boolean,
    val lines: List<String>,
)

/**
 * 「你选的模型本地没有」——弹窗用。
 *
 * @param label 用户刚选的东西（「高精度 server · 164MB」「SenseVoice」…）
 * @param rels 这些文件相对 `models/` 的路径，逐个显示在弹窗里
 * @param note 补一句为什么（可选）
 */
data class MissingModel(
    val label: String,
    val rels: List<String>,
    val note: String = "",
    /** 点「取消」时回到上一次能用的选择（否则选了个没装的后端，下次采集直接报错）。 */
    val revert: (() -> Unit)? = null,
)
