package com.screenast.app.interpret

import android.content.res.AssetManager
import android.os.SystemClock
import android.util.Log
import com.screenast.app.asr.AsrBackend
import com.screenast.app.asr.DictationEngine
import com.screenast.app.asr.DictationHub
import com.screenast.app.asr.guessLangFromText
import com.screenast.app.core.Lang
import com.screenast.app.translate.HistoryTurn
import com.screenast.app.translate.TranslateChunk
import com.screenast.app.translate.TranslateRequest
import com.screenast.app.translate.TranslationProvider
import com.screenast.app.translate.TranslationQuality
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.ExecutorCoroutineDispatcher
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.asCoroutineDispatcher
import kotlinx.coroutines.cancel
import kotlinx.coroutines.channels.BufferOverflow
import kotlinx.coroutines.channels.Channel
import kotlinx.coroutines.launch
import java.io.File
import java.util.concurrent.Executors
import java.util.concurrent.atomic.AtomicLong

/**
 * **实时同传**：把「识别」和「翻译」两个模型串成一条流水线。
 *
 * ```
 * 采集线程                         翻译线程（单线程）                  UI
 * ────────                         ─────────────────                  ──
 * DictationHub.feed
 *   └─ DictationEngine.onPcm
 *        ├─ 重采样 48k→16k
 *        ├─ 闸门 + silero VAD
 *        └─ SenseVoice 解码 ──► onUtterance(id, 原文, 语种)
 *                                  │
 *                                  ├──────────────► 显示原文（立刻，不翻译也要看得到）
 *                                  │
 *                                  └─ enqueue(job) ──► 队列 ──► NllbTranslator
 *                                                                  │ 逐 token
 *                                                                  └──► onTranslation(译文) ──► 覆盖显示
 * ```
 *
 * ## 为什么翻译必须在**另一条线程**上：这里有个不能用「就地同步」的硬约束
 *
 * [DictationEngine] 的类注释取舍 3 解释了它为什么**就地同步**解码：
 * 它只有采集线程一个调用者，就地做就不需要锁。那个论证对 ASR 成立，
 * **对翻译不成立**，而且理由不是「锁不好写」：
 *
 * - 采集线程同时负责从 `AudioRecord` 读数。`AudioCaptureService` 按 minBuf×4 申请
 *   （约 1.7 秒）的内核缓冲，ASR 解码那 65~110ms 完全吸收得了。
 *   但 NLLB 整句要 **750ms**，长句更久 —— 一旦在采集线程上等它，
 *   字幕会成段地丢，而且**丢的是音频**（内核缓冲溢出），事后无法补。
 * - 所以翻译必须是生产者-消费者：采集线程只管投递（[Channel.trySend]，永不阻塞），
 *   翻译线程慢慢做。
 *
 * ## 为什么翻译线程是**单线程**（不是线程池）
 *
 * [com.screenast.app.translate.nllb.NllbTranslator] 把一个句子的中间状态
 * （encoder 输出 + 48 个 KV 张量）放在成员变量上，靠一把锁串行化。
 * 用线程池只会让多个任务在锁上排队，**没有吞吐收益，却让「谁先出结果」变成不确定的** ——
 * 字幕必须按说话顺序出现。单线程 + FIFO 队列是最简单且唯一正确的形状。
 *
 * ## 跟不上怎么办：丢最旧的，保住「现在在说什么」
 *
 * 队列容量 [MAX_QUEUE]。满了以后投递**丢最旧**的（[BufferOverflow.DROP_OLDEST]），
 * 因为同传字幕的价值随时间衰减：观众要的是当下这一句，不是十秒前那句。
 *
 * 丢了多少句不是靠猜：每个任务带一个自增 [Job.seq]，worker 拿到 seq 时
 * 与上一句的 seq 一比就知道中间被丢了几句（`gap = seq - lastSeen - 1`），
 * 然后**明确告诉用户**「翻译跟不上，跳过 N 句」。
 * 不这么做的话，用户只会觉得「字幕偶尔少一句」，而永远不知道原因。
 *
 * ## 语种判定：源语种由 ASR 给，判不出来就**明确跳过**，绝不猜
 *
 * NLLB 需要明确的源语码。SenseVoice 正常会给 `<|en|>` 这类标记，但也会给不出
 * （[Lang.AUTO]）。这时用 [fallbackSource]；如果连它也是 AUTO，就跳过这一句并说明原因。
 * 猜一个语种的表现是「译文凭空变成另一种语言」—— 那是本项目花了一整天才查清的一类 bug
 * （`Lang.nllbTokenId` 曾经全部写错），不能重新引入。
 *
 * ## 线程约定
 *
 * - [onPcm] / [onCaptureState] 在**采集线程**上被调用（同 [DictationHub.Listener]）。
 * - [Callback] 的 `onUtterance` 在**采集线程**上；`onTranslation` / `onNotice` /
 *   `onTranslatorState` 在**翻译线程**上。实现方都要自己切线程才能碰 UI。
 */
class InterpretEngine(
    assets: AssetManager,
    asrDir: File,
    vadFile: File,
    /**
     * 强制切段间隔（毫秒），透传给 [DictationEngine]。`0` = 关闭。
     *
     * 为什么是「引擎按时间切」而不是调 VAD 参数：silero 的 minSilenceDuration 是
     * 「静音多久算换句」，而歌曲里根本没有静音 —— 调小它只会把一切切成碎片，
     * 仍然出不了句。需要的是「不等静音也按时间出句」。
     */
    private val forceCutMs: () -> Int = { 0 },
    /**
     * 带几轮对话上文去翻译。`0` = 不带（默认，与加这个参数之前完全等价）。
     *
     * ## 为什么做成 lambda 而不是构造时的 Int
     *
     * 滑条一拖就要生效，而重建引擎意味着**重新加载 1.2GB 的 2B 模型**（约 4 秒）。
     * 每拖一次滑条卡四秒不可接受。做成 lambda 后，每个任务在开始时读一次当前值，
     * 改完下一句就生效，不需要重建任何东西。
     *
     * ## 为什么上限是 [MAX_CONTEXT_ROUNDS]
     *
     * 上下文越长，提示词越长：每轮约 `原文 + 译文 + 8 个模板 token`。
     * 10 轮在实测语料上约 500~900 字符（≈300~500 token），NPU 预填 2000 tok/s
     * 只值 150~250ms —— 这是上限取 10 的依据。再往上，首字延迟会开始吃掉
     * 「<1 秒」这条硬指标（§1），收益却已经很小。
     */
    private val contextRounds: () -> Int = { 0 },

    /**
     * 识别后端的工厂。`null` = 用默认的 SenseVoice。
     *
     * 做成工厂而不是「一个 [AsrBackend] 实例」的理由同 [translatorFactory]：
     * 加载模型要几百毫秒到几秒，必须在采集线程上懒加载，不能在构造引擎时同步加载。
     * 引擎构造本身发生在界面线程上，同步加载会让点击「开始」的瞬间卡住。
     */
    private val asrFactory: (() -> AsrBackend)? = null,
    /** ASR 判不出语种时的兜底源语言。返回 [Lang.AUTO] 表示「不猜，跳过翻译」。 */
    private val fallbackSource: () -> Lang,
    /** 目标语言。做成 lambda 是因为界面上可以随时改，而重建引擎要重新加载两个模型。 */
    private val targetLang: () -> Lang,
    /**
     * 手填的目标语种名（设置里的「自定义」）。非空时覆盖 [targetLang]，
     * 直接写进提示词 —— 这样翻泰语/越南语之类不必扩展枚举。
     */
    private val customTargetLang: () -> String = { "" },
    /** 手填的源语种名，非空时也写进提示词。 */
    private val customSourceLang: () -> String = { "" },
    private val callback: Callback,
) : DictationHub.Listener, AutoCloseable {

    /**
     * 识别结果与翻译结果的出口。
     *
     * 原文和译文**分开回调**（而不是合成一个「句子对象」）是因为它们的时序完全不同：
     * 原文在识别完成时立刻到，译文可能 80ms 后到、也可能 2 秒后到、还可能永远不到
     * （队列丢了、语种判不出）。合成一个对象会逼着 UI 去猜「这个字段是还没填还是就是空」。
     */
    interface Callback {
        /** 一句话识别完成。[id] 是后续 [onTranslation] 的配对键。 */
        fun onUtterance(id: Long, text: String, lang: Lang, samples: Int, decodeMs: Long)

        /**
         * 两段被硬切开的话合并成了一句：[fromId] 那一行应当撤掉，内容并到 [intoId] 那一行。
         *
         * 为什么需要：强制切段（`Settings.forceSegmentMs`）只按能量最低点切，切口经常落在
         * 句子中间。各翻各的会得到「那是最健康的。」+「在我的一生中。」这种读不通的东西 ——
         * 实测英文单口就是这样坏的（`That was the Healthiest.` + `heard in my life.`）。
         * 合并后按一句翻才是「那是我这辈子听过最健康的话。」
         *
         * [source] 是**合并后**的原文，[intoId] 那一行要连原文一起更新。
         */
        fun onUtteranceMerged(intoId: Long, fromId: Long, source: String)

        /**
         * 译文推进。[text] 是**到目前为止的完整译文**（不是增量），直接覆盖显示即可。
         *
         * [firstChunkMs] 是这一句的首字延迟，**从第一个 chunk 起就一直带着真实值**，
         * 包括最后那个 [isFinal] 的 chunk。不要改成「只在第一个 chunk 上带值、其余传 0」——
         * 那样只对「每个 chunk 都处理」的消费者成立，而只关心定稿的消费者
         * （比如日志、历史区、以后的悬浮字幕窗）就**永远拿不到首字延迟**，
         * 表现是「首字延迟恒为 0」这种看起来像没测量、实际是没传值的假数据。
         * 本项目真机上就是这么表现的，改回来之前先想清楚谁在消费这个字段。
         *
         * [totalMs] 只在 [isFinal] 时带真实值。它**不含**模型加载那一次性的几秒
         * （[onTranslatorState] 会单独报加载耗时）—— 混在一起会让所有性能判断失去基准。
         */
        fun onTranslation(id: Long, text: String, isFinal: Boolean, firstChunkMs: Long, totalMs: Long)

        /** 翻译后端状态（加载中 / 就绪 / 失败）。在翻译线程上回调。 */
        fun onTranslatorState(text: String)

        /** 一次性提示：跳过某句、丢了几句。在翻译线程上回调。 */
        fun onNotice(text: String)

        fun onError(message: String)

        fun onSpeaking(speaking: Boolean)
    }

    private val assetsRef = assets
    private val asrDirRef = asrDir
    private val vadRef = vadFile

    /**
     * 已翻完的对话轮（原文 + 净化后的译文），**按时间正序**，最后一个最新。
     *
     * 只在 [worker] 那条单线程上读写，所以不需要锁 —— 这与本项目其它地方
     * 「共享状态要么不可变、要么单线程独占」的约定一致。
     */
    private val history = ArrayDeque<HistoryTurn>()

    /** [history] 最后一轮对应的句子 id，用于「同一句合并后重翻」时替换而不是追加。 */
    private var historyId = -1L

    /** 最近一次任务实际带了几轮上文，供界面/日志核对。 */
    @Volatile
    var lastContextRounds = 0
        private set

    // ------------------------------------------------------------ ASR 级

    private val asr = DictationEngine(
        assets = assets,
        modelDir = asrDir,
        vadFile = vadFile,
        forceCutMs = forceCutMs,
        backendFactory = asrFactory,
        callback = object : DictationEngine.Callback {
            override fun onUtterance(
                text: String,
                lang: Lang,
                samples: Int,
                decodeMs: Long,
                forced: Boolean,
            ) {
                val id = idSeq.incrementAndGet()
                // 先出原文：翻译再快也有 80ms，而且可能永远不来。
                // 原文先到，用户才能把「听错了」和「翻错了」分开看。
                callback.onUtterance(id, text, lang, samples, decodeMs)

                // ---- 断句合并：上一段是被硬切开的半句，就并过来一起翻 ----
                // 判据只有一条：上一段是 forced（按时间切，不是 VAD 等到静音）。
                // 不去猜标点 —— 日语/中文的模型标点本来就是补出来的，不可靠；
                // 也不看去重时间差 —— 语音是连续的，到达时间差里混着解码耗时。
                val prev = lastSeg
                if (prev != null && prev.forced &&
                    prev.lang == lang &&
                    prev.parts < MAX_MERGE_PARTS &&
                    prev.text.length + text.length <= MAX_MERGE_CHARS
                ) {
                    val merged = joinText(prev.text, text)
                    lastSeg = LastSeg(prev.id, merged, lang, prev.parts + 1, forced)
                    callback.onUtteranceMerged(prev.id, id, merged)
                    enqueue(prev.id, merged, lang)
                    return
                }

                lastSeg = LastSeg(id, text, lang, 1, forced)
                enqueue(id, text, lang)
            }

            override fun onError(message: String) = callback.onError(message)

            override fun onSpeaking(speaking: Boolean) = callback.onSpeaking(speaking)
        },
    )

    /** ASR 模型是否真的加载起来了。引擎本身永远可以挂载（见 [DictationEngine]）。 */
    val ready: Boolean get() = asr.ready

    /** 实际在跑的识别后端名（SenseVoice / Paraformer / Whisper…）。 */
    val asrName: String get() = asr.backendName

    /** 是否为流式后端（界面据此决定「强制切段」滑条是否可用）。 */
    val asrStreaming: Boolean get() = asr.streaming

    // ASR 统计透传（界面直接用，不必再持有内部引擎）
    val inputFrames get() = asr.inputFrames
    val resampledSamples get() = asr.resampledSamples
    val windowsFed get() = asr.windowsFed
    val windowsSkipped get() = asr.windowsSkipped
    val preRollRescued get() = asr.preRollRescued
    val forcedCuts get() = asr.forcedCuts
    val forcedDecodeMs get() = asr.forcedDecodeMs
    val totalVadMs get() = asr.totalVadMs
    val utterances get() = asr.utterances
    val lastDecodeMs get() = asr.lastDecodeMs
    val totalDecodeMs get() = asr.totalDecodeMs

    // ------------------------------------------------------------ 翻译级

    /**
     * 翻译后端供应器。`null` = 纯听写模式（只出原文）。
     *
     * 做成「供应器」而不是「一个 provider 实例」：端侧 NLLB 加载要 3.5 秒、
     * 常驻 1.27GB，必须在**翻译线程上懒加载**，不能在构造引擎时同步加载
     * （那会让界面在点击的一瞬间卡死 3.5 秒）。见 [ensureProvider]。
     */
    @Volatile
    var translatorFactory: (() -> TranslationProvider)? = null
        set(value) {
            field = value
            // 重新打开翻译时清掉上一次的失败标记，否则用户「关掉再打开」也救不回来。
            if (value != null) providerFailed = false
        }

    val translationEnabled: Boolean get() = translatorFactory != null

    /**
     * 丢掉已缓存的翻译后端，让下一句按新设置重建。
     *
     * 为什么要专门开个方法：`ensureProvider()` 第一行是 `provider?.let { return it }`，
     * 所以**只改 `translatorFactory` 是没用的** —— 缓存还在，用户选了新后端，
     * 译文还是旧后端出的（这个症状极难自查，因为界面上选项明明变了）。
     *
     * 为什么丢在 [scope] 上而不是就地清空：`close()` 的注释记着「把脚下的地板拆掉」
     * 那次 SIGSEGV —— 关的是正在被读的 ONNX 会话。放到单线程翻译调度器上排队，
     * 就保证了它一定跑在当前这轮翻译之后。
     */
    fun resetProvider() {
        scope.launch {
            provider?.let { p -> runCatching { p.close() }.onFailure {
                android.util.Log.w("AST.Engine", "关闭旧翻译后端失败：${it.message}")
            } }
            provider = null
            providerFailed = false
        }
    }

    private val dispatcher: ExecutorCoroutineDispatcher =
        Executors.newSingleThreadExecutor { r ->
            Thread(r, "siat-interpret").apply { isDaemon = true }
        }.asCoroutineDispatcher()
    private val scope = CoroutineScope(SupervisorJob() + dispatcher)

    private val jobs = Channel<Job>(capacity = MAX_QUEUE, onBufferOverflow = BufferOverflow.DROP_OLDEST)
    private val idSeq = AtomicLong()

    /**
     * 最近一句的合并状态。只在 ASR 回调线程上读写（这个回调本来就是单线程的），
     * 所以不需要锁；[LastSeg.forced] 决定下一句要不要并进来。
     */
    private var lastSeg: LastSeg? = null

    @Volatile private var provider: TranslationProvider? = null
    @Volatile private var providerFailed = false

    // ---- 统计（翻译线程写、UI 读，一律 @Volatile）----

    /** 排进过队列的句子总数。 */
    @Volatile var enqueued = 0L; private set

    /** 真正翻完的句子数。 */
    @Volatile var translated = 0L; private set

    /** 因为队列满而被丢掉的句子数（由 seq 空隙算出来，不是估计值）。 */
    @Volatile var dropped = 0L; private set

    /** 因为「语种判不出 / 源=目标」而没翻译的句子数。 */
    @Volatile var skippedNoLang = 0L; private set

    /** 最近一句的首字延迟与整句耗时（同传的验收指标）。 */
    @Volatile var lastFirstMs = 0L; private set
    @Volatile var lastTotalMs = 0L; private set

    /** 首字延迟 / 整句耗时的累计值，用于算平均。 */
    @Volatile var sumFirstMs = 0L; private set
    @Volatile var sumTotalMs = 0L; private set

    /** 队列里还压着几句（含正在翻的那句）。 */
    @Volatile var queueDepth = 0; private set

    private var lastSeenSeq = 0L

    private val worker = scope.launch {
        try {
            // for 循环在 jobs.close() 后自然结束 —— 这是唯一「确认没有任务在跑」的时刻，
            // 也就是唯一能安全关掉 ONNX 会话的时刻（见 close）。
            for (job in jobs) {
                val gap = job.seq - lastSeenSeq - 1
                if (gap > 0) {
                    dropped += gap
                    callback.onNotice("翻译跟不上，跳过 $gap 句（队列上限 $MAX_QUEUE；同传宁可丢旧句也要跟上当前）")
                }
                lastSeenSeq = job.seq
                // ⚠️ `+1` 不能省：lastSeenSeq 是**刚取出来的这一句**，它正在翻、还没翻完，
                // 所以「还压着几句」必须把它算进去。少了这个 1，队列在整段解码期间恒显示 0，
                // 而 0 正是调用方用来判断「翻译都做完了」的判据 —— 自测脚本因此
                // 在 NLLB 还在加载时就宣布「翻译 0 句」收了尾。
                queueDepth = (enqueued - lastSeenSeq + 1).coerceAtLeast(0L).toInt()
                try {
                    runJob(job)
                } finally {
                    // ⚠️ 翻完了必须把它自己从「还压着几句」里减掉。
                    // 少了这一步，整条流水线跑空之后队列会**永久停在 1**：界面上一直写着
                    // 「队列 1」像是有一句卡住了，而调用方拿 `queueDepth == 0` 当
                    // 「翻译都做完了」的判据，于是那个条件**永远不会成立**。
                    // 真机上就是这么表现的（8 句全翻完、日志里最后一句都出来了，
                    // 统计栏仍显示「队列 1」）。这里不再 `+1`：lastSeenSeq 正是刚翻完的这句，
                    // `enqueued - lastSeenSeq` 剩下的是「已投递但还没取出来」的句数。
                    queueDepth = (enqueued - lastSeenSeq).coerceAtLeast(0L).toInt()
                }
            }
        } finally {
            // 走到这里 = 投递已结束且没有任务在跑。**绝不能把这个释放点挪到 close() 里**：
            // close() 不知道 worker 是不是正卡在 translate() 中间，那时候关 ONNX 会话
            // 就是「把正在被读的张量连根拔掉」—— 本项目为此吃过一次 SIGSEGV。
            runCatching { provider?.close() }
            provider = null
            Log.i(TAG, "翻译线程退出，模型已释放")
        }
    }

    // ------------------------------------------------------------ DictationHub.Listener

    override fun onPcm(pcm: ByteArray, len: Int, rate: Int, channels: Int) =
        asr.onPcm(pcm, len, rate, channels)

    override fun onCaptureState(running: Boolean) = asr.onCaptureState(running)

    // ------------------------------------------------------------ 内部

    private fun enqueue(id: Long, text: String, lang: Lang) {
        if (translatorFactory == null) return

        val target = targetLang()
        if (target == Lang.AUTO) {
            callback.onNotice("没有设置目标语言，这一句只显示原文")
            return
        }
        var source = if (lang != Lang.AUTO) lang else fallbackSource()
        if (source == Lang.AUTO) {
            skippedNoLang++
            callback.onNotice("第 $id 句语种没判定出来，跳过翻译（可在界面上指定源语言）")
            return
        }
        if (source == target) {
            // ASR 偶尔会把日语/韩语判成中文（音乐、噪声、汉字多的短句最容易），
            // 这时「源=目标就跳过」会把整句日语当成品直接显示、根本不翻。
            // 用文字本身再校一次：判到别的语言就照翻，别信 ASR 的这一次。
            val byText = guessLangFromText(text)
            if (byText != Lang.AUTO && byText != target) {
                Log.i(TAG, "第 $id 句：ASR 判 ${source.label}、文本看着像 ${byText.label}，仍送去翻译")
                callback.onNotice("第 $id 句 ASR 判成${source.label}，但看着像${byText.label}，照翻")
                source = byText
            } else {
                // 已经在说目标语言了 —— 原文就是「译文」，直接定稿。
                // 这不是优化：把它送去 NLLB 会得到一句被模型改写过的话，反而更差。
                Log.i(TAG, "第 $id 句：源=目标（${source.label}），原文直接定稿")
                skippedNoLang++
                callback.onTranslation(id, text, isFinal = true, firstChunkMs = 0L, totalMs = 0L)
                return
            }
        }

        enqueued++
        val job = Job(id, enqueued, text, source, target)
        val r = jobs.trySend(job)
        if (r.isFailure) {
            // 只有 channel 被 close 之后才会走到这里
            callback.onNotice("翻译队列已关闭，第 $id 句没有翻译")
        }
    }

    /**
     * 拼接两段原文。中文/日文之间不加空格，拉丁文之间加一个。
     *
     * 直接 `"$a $b"` 会在中文里塞进空格（「那是最健康的。 在我的一生中。」），
     * 中日文都不该有；反过来英文硬切的两段粘成一个词（`theHealthiest`）更糟。
     */
    private fun joinText(a: String, b: String): String {
        if (a.isEmpty()) return b
        if (b.isEmpty()) return a
        val glue = if (isCjk(a.last()) || isCjk(b.first())) "" else " "
        return a + glue + b
    }

    private fun isCjk(c: Char): Boolean =
        c in '\u4e00'..'\u9fff' || c in '\u3040'..'\u30ff' || c in '\uac00'..'\ud7af'

    private fun ensureProvider(): TranslationProvider? {
        provider?.let { return it }
        if (providerFailed) return null
        val f = translatorFactory ?: return null
        val t0 = SystemClock.elapsedRealtime()
        // 文案不能写死成某个后端：这里并不知道 [translatorFactory] 会造出谁
        // （NLLB 约 3.5 秒，2B 大模型约 3 秒）。写死过一次「端侧 NLLB 约 3.5 秒」，
        // 结果日志里加载的明明是大模型、提示却在说 NLLB，排查时被这条骗了很久。
        callback.onTranslatorState("翻译后端：加载中…（首次使用，只加载一次）")
        return try {
            val p = f()
            provider = p
            callback.onTranslatorState(
                "翻译后端：${p.displayName} 就绪（${SystemClock.elapsedRealtime() - t0} ms）"
            )
            p
        } catch (t: Throwable) {
            providerFailed = true
            callback.onTranslatorState("翻译后端：加载失败，后续句子只显示原文")
            callback.onError("翻译后端加载失败：${t.message ?: t.javaClass.simpleName}")
            null
        }
    }

    /**
     * 翻一句。**这个方法整体跑在翻译线程上**（单线程 dispatcher），
     * 所以 [lastFirstMs] 之类的写不会有竞争。
     */
    private suspend fun runJob(job: Job) {
        val p = ensureProvider() ?: return
        // ⚠️ t0 必须在 ensureProvider() **之后**取。
        // 首次翻译会把 3.7 秒的模型加载算进「整句耗时」里，屏幕上就会出现
        // 「整句 3899 ms」这种既不真实、又会让后续所有性能判断失去基准的数字。
        // 加载是**一次性**的，翻译是**每句**的，两者必须分开计量。
        val t0 = SystemClock.elapsedRealtime()
        var firstMs = 0L
        var chunks = 0
        var outText = ""
        // 每个任务开始时读一次当前设置：滑条拖完，下一句就按新轮数走。
        val rounds = contextRounds().coerceIn(0, MAX_CONTEXT_ROUNDS)
        val ctx = if (rounds == 0) emptyList() else history.takeLast(rounds)
        lastContextRounds = ctx.size
        try {
            p.translate(
                TranslateRequest(
                    text = job.text,
                    source = job.source,
                    target = job.target,
                    // 自定义语种：用户在设置里手填的名字直接进提示词，
                    // 枚举翻不了的语言（泰语、越南语…）靠这一条才能用。
                    targetLabel = customTargetLang().trim(),
                    // 用户没自定义语种时，把 ASR 判出来的源语言也带上：
                    // 2B 模型拿到「原文语言：日语」比让它自己猜更不容易照抄原文。
                    sourceLabel = customSourceLang().trim().ifBlank {
                        if (job.source != Lang.AUTO && job.source != job.target) job.source.label else ""
                    },
                    history = ctx,
                ),
            ).collect { chunk ->
                val now = SystemClock.elapsedRealtime()
                chunks++
                if (chunks == 1) {
                    firstMs = now - t0
                    lastFirstMs = firstMs
                }
                outText = chunk.text
                val total = if (chunk.isFinal) now - t0 else 0L
                if (chunk.isFinal) {
                    lastTotalMs = total
                    sumFirstMs += firstMs
                    sumTotalMs += total
                    translated++
                    // 只有定稿才进上文：中途的 chunk 是半句，写进去会让下一句条件在
                    // 一个不完整的助手上，输出质量反而下降（§0.5 的教训）。
                    //
                    // ⚠️ 但「定稿」不等于「真的翻了」：2B 在日语上偶尔会回一段日文改写
                    // （`お元気にしていましたか` → `元気にお過ごしでしょうか`），
                    // 精确相等比较抓不到。让它进 history 后，下一句会条件在日文 assistant
                    // 轮上，本来能翻的句子也会被带成日文。所以这里必须过语言校验。
                    if (TranslationQuality.looksUntranslated(job.source, job.target, job.text, chunk.text)) {
                        Log.w(
                            TAG,
                            "第 ${job.id} 句译文仍像原文（未翻成${job.target.label}），不写入上文：" +
                                "${chunk.text.take(40)}",
                        )
                        callback.onNotice("第 ${job.id} 句模型没翻成${job.target.label}，这句不写入上文")
                    } else {
                        rememberTurn(job, chunk.text)
                    }
                }
                // firstMs 每个 chunk 都传（见 Callback.onTranslation 的说明）。
                callback.onTranslation(job.id, chunk.text, chunk.isFinal, firstMs, total)
            }
        } catch (ce: CancellationException) {
            throw ce
        } catch (t: Throwable) {
            callback.onError("翻译失败（第 ${job.id} 句）：${t.message ?: t.javaClass.simpleName}")
        }
        // 收 chunk 数写进日志：CONFLATED 通道会在消费者慢时丢掉中间态，
        // 所以这个数字明显小于 token 数时，说明「逐 token 推进」没真正体现到界面上 ——
        // 那是首字延迟降不下来的原因，而这一点从界面上看不出来。
        // 译文跟原文一模一样 = 模型照抄了（小模型遇到没见过的语种/提示词里不让翻时会这样）。
        // 用户看到的就是「日语没被翻译」，所以这里必须留一条能 grep 的日志。
        if (outText.isNotBlank() && outText.trim() == job.text.trim()) {
            Log.w(
                TAG,
                "第 ${job.id} 句译文与原文相同（模型照抄）：source=${job.source.label} " +
                    "target=${job.target.label} 原文=${job.text.take(30)}",
            )
        }
        Log.i(
            TAG,
            "第 ${job.id} 句翻译完成：chunks=$chunks 首字=${firstMs}ms " +
                "整句=${lastTotalMs}ms 译文 ${outText.length} 字：$outText",
        )
    }

    /**
     * 把一句定稿记进上文窗口。
     *
     * ## 为什么必须判 [historyId] 而不是直接 add
     *
     * 「合并后重翻」用的是**前一句的 id**（`enqueue(prev.id, merged, lang)`）。
     * 如果无脑追加，同一句会出现两次：先是半句，再是合并后的整句。
     * 下一句拿到的上文里就有一对「原文半句 → 译文」+「原文整句 → 译文」，
     * 模型会以为说话人把话重复了一遍。
     *
     * 所以同 id 的重翻**替换**掉最后一轮，不同 id 才追加。
     *
     * 译文为空（后端拒答 / 整段被净化掉）时不记：一条空的 assistant 轮
     * 比没有这一轮更糟，模型会学着「有时候不说话」。
     */
    private fun rememberTurn(job: Job, target: String) {
        val clean = target.trim()
        if (clean.isEmpty()) return
        val turn = HistoryTurn(job.text.trim(), clean)
        if (job.id == historyId && history.isNotEmpty()) {
            history.removeLast()
            history.addLast(turn)
        } else {
            history.addLast(turn)
            historyId = job.id
        }
        while (history.size > MAX_CONTEXT_ROUNDS) history.removeFirst()
    }

    override fun close() {
        // 顺序有讲究：先停投递（worker 的 for 循环会自然结束并释放模型），
        // 再取消作用域。反过来会把「正常收尾」和「被取消」混在一起，
        // 而模型释放只能发生在正常收尾那条路径上。
        jobs.close()
        scope.cancel()
        runCatching { asr.close() }
    }

    /** 一个待翻译的句子。[seq] 与 [enqueued] 同源，用于算丢句数。 */
    private data class Job(
        val id: Long,
        val seq: Long,
        val text: String,
        val source: Lang,
        val target: Lang,
    )

    /** 最近一句（合并用）：[parts] 是已经并进来的段数，[forced] 表示它是被硬切开的。 */
    private data class LastSeg(
        val id: Long,
        val text: String,
        val lang: Lang,
        val parts: Int,
        val forced: Boolean,
    )

    companion object {
        private const val TAG = "AST.Interpret"

        /**
         * 翻译队列上限（句）。
         *
         * 取 4 的依据：NLLB 一句约 0.75 秒，而正常语速 2~3 秒一句，队列本来就该是空的。
         * 只有「快语速 + 长句」才会压起来；那时留 4 句的余量足够吸收抖动，
         * 再长就说明真的跟不上了 —— 与其排队等，不如丢掉旧的、全力跟当前。
         */
        private const val MAX_QUEUE = 4

        /**
         * 断句合并的上限：最多并 [MAX_MERGE_PARTS] 段、总长 [MAX_MERGE_CHARS] 字。
         *
         * 不设上限就是灾难：连续说话时每一段都是 forced，一路并下去会攒出十几秒的
         * 「一句话」，译文要等更久才出，而且越长越容易翻跑偏。
         *
         * 实测把上限放到 3 会并过头：韩语样例里「…에피소드 에요.」和后面另一件事的
         * 「그래서 지금 제가…」被并成一句，译文变成一长串两件事混在一起。2 段（≈8 秒）
         * 刚好覆盖「一句话被切成两半」这个真实情况，再多就不是切点的问题了。
         */
        private const val MAX_MERGE_PARTS = 2
        private const val MAX_MERGE_CHARS = 140

        /**
         * 上文窗口的硬上限（轮）。界面滑条给的是 0..10，这里是**兜底**：
         * 即使将来界面允许更大、或 DataStore 里存了个脏值，也不会让提示词无限长
         * 把 KV 缓存撑爆（2B 在 nCtx=1024 下，10 轮上下文已经占掉约一半）。
         */
        const val MAX_CONTEXT_ROUNDS = 10
    }
}
