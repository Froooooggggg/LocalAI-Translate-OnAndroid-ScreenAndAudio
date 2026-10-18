package com.screenast.app.asr

import android.content.res.AssetManager
import android.os.SystemClock
import android.util.Log
import com.screenast.app.asr.AsrBackend.Segment
import com.screenast.app.audio.AudioResampler
import com.screenast.app.core.Lang
import java.io.File

/**
 * 「听写」——把采集到的屏幕音频变成一行行原文。
 *
 * 这是路线图 ② 的落点：**先只出原文，不带翻译**。
 * 理由是可诊断性：字幕一旦同时经过识别和翻译两道，错了以后没法判断是哪一道的锅。
 * 用户能看着原文判断「是听错了还是翻错了」，这是把链路调通的必要前提。
 *
 * ## 数据流
 *
 * ```
 * 采集线程 ──16bit PCM──► DictationHub ──► DictationEngine
 *                                            │
 *                                  ① 重采样 48k/44.1k → 16k 单声道 float
 *                                  ② 切齐 512 点窗口（silero VAD 的固定窗口）
 *                                  ③ 功率闸门（静音时不喂 VAD）
 *                                  ④ VAD 端点检测 → 一句话结束
 *                                  ⑤ SenseVoice 离线解码 → 文本 + 语种
 *                                            │
 *                                            ▼
 *                                        Callback（仍在采集线程）
 * ```
 *
 * ## 四个关键取舍
 *
 * ### 1. 为什么在静音时**跳过** VAD，以及为什么这个闸门**绝不能丢句首**
 *
 * silero VAD 本身是小网络，但这里是每秒 31.25 次窗口推理（16kHz / 512），
 * 它和 SenseVoice 抢 CPU。播放捕获经常长时间没有说话（游戏、音乐、暂停），
 * 此时跑 VAD 是纯浪费 —— 手机同时还在解码视频，省下来的核就是字幕的延迟余量。
 *
 * 做法：用**每一窗**自己的 RMS 做闸门，`[HANGOVER_MS]` 内说过话就保持开闸。
 * 真正的「这是不是人声」判据交给 VAD 自己做（它有 minSpeechDuration 之类的
 * 内部门槛），闸门只是省 CPU 的装置 —— 所以它**宁可多跑，不可漏喂**。
 *
 * ## ⚠️ 这里踩过一个把句首吃掉 0.5 秒的坑，修法有三条，缺一不可
 *
 * 第一版是「RMS 按整批算、闸门按整批判」：`AudioCaptureService` 一次喂 100ms，
 * 那 100ms 里可能正好是句首，而整批的 RMS 只让它记了 1 次「有信号」，
 * 达不到 `GATE_WINDOWS = 2`，于是**整批被当作静音丢掉**。
 * 自测里表现为 `"The quick brown fox..."` 被识别成 `"Brown Fox..."`。
 *
 * 1. **RMS 必须按窗口算**（在切片循环内部），不能按批算 —— 批的边界和语音的
 *    边界毫无关系，按批判定等于拿 100ms 的颗粒度去切一个音素级的起点。
 * 2. **开闸条件不能要求「连续 N 窗」**。窗口一过 RMS 门限就立刻开闸；
 *    想滤掉爆音/咔哒声应该靠 VAD 自己的最小时长，而不是靠延迟喂数据 ——
 *    后者是拿「丢字」换「省一点 CPU」，代价完全不对等。
 *    （原写法里 `loudWindows >= GATE_WINDOWS` 这一支其实**恒为死代码**：
 *    只要本窗有信号就会把 `lastLoudAt` 刷成 now，hangover 那支必然同时成立。）
 * 3. **开闸瞬间要把闸门关闭期间攒下的窗口补喂回去**（[preRoll]，[PREROLL_WINDOWS] 个）。
 *    因为 RMS 门限是「已经足够响」而不是「开始有声」，缓慢起音的词
 *    （擦音、气声开头的 "The"、日语促音）会在门限之下待上几十毫秒。
 *    只做 1、2 两条能保住大部分句首，加上 pre-roll 才是真的不漏。
 *
 * ### 2. 为什么不需要额外的「连续缓冲」
 *
 * 第一版实现里我额外维护了一个 10 秒的 `buffer` 用来「攒整段语音」，
 * 后来发现是**多余且有竞态风险**的：silero VAD 自己就在内部攒语音
 * （它按 minSpeechDuration 判定起点、按 minSilenceDuration 判定终点），
 * `sherpa` 的 `Vad.front()` 交给我们的 `seg.samples` 已经是一整句。
 * 再攒一层只会让「VAD 认为的句子边界」和「我们发给 SenseVoice 的边界」不一致。
 *
 * 所以这里只保留闸门（省 CPU），不保留缓冲（交给 VAD）。
 *
 * ### 3. 为什么解码**就地同步**做，不丢给线程池
 *
 * SenseVoice int8 在 8 Elite 上一段几秒的语音约 100~300ms 出结果。
 * 如果解码异步，采集线程会继续灌音频，就出现「解码用的缓冲被下一段盖掉」这类竞态。
 * 就地做的话整个引擎是**单线程**的（只有采集线程会进来），不需要任何锁。
 * 代价是解码那几百毫秒里不读 AudioRecord —— 而 AudioRecord 的内核缓冲
 * （[com.screenast.app.capture.AudioCaptureService] 里按 minBuf×4 申请，约 1.7 秒）
 * 足够吸收这点停顿。
 *
 * ### 4. 为什么 [onPcm] 里的任何异常都不能外抛
 *
 * 采集循环必须活着：它一死，用户就要重新授权一次 MediaProjection。
 * 所以这里所有对 native 的调用都包了 try/catch，失败降级为 [Callback.onError]。
 *
 * ### 5. 强制切段：为什么这里**又**出现了一个缓冲，而不是回到取舍 2 的老路
 *
 * 取舍 2 否掉的是「用缓冲取代 VAD 分段」——那会让两条边界打架。
 * 这里加 [forced]/[forcedLen] 解决的是**另一个问题**：VAD 根本不出段。
 *
 * SenseVoice 是离线模型，「出句」的唯一触发是 VAD 检测到 0.5 秒静音（或夹到
 * maxSpeechDuration 的 8 秒上限）。歌曲、连唱、一路不停的旁白里两者都指望不上：
 * 前者没有静音，后者要看 silero 是否把音乐判成人声（多数情况下判不出来，
 * 于是连 8 秒那个安全阀也不会触发）。表现就是「半天跑不出一句」。
 *
 * 所以这里按 [forceCutMs] 毫秒**匀速攒音频**，攒满就绕过 VAD 直接送离线解码
 * （[SenseVoiceAsr.decodeSegment]），切点还会在最后 [CUT_LOOKBACK_MS] 内
 * 找能量最低处（换气/鼓点间隙），尽量不切在字中间；切剩下的尾巴留给下一段，
 * 保证音频不丢。
 *
 * 两条路不会打架，因为它俩**互斥地消费同一批音频**：
 * - VAD 出了段 → 说明这几秒已经由 VAD 出口了 → 把 [forced] 清空；
 * - 强制切段先动手 → 调 [SenseVoiceAsr.resetVad] 把 VAD 里同一批音频丢掉。
 *
 * `forceCutMs == 0` 时整条路完全不跑（不攒、不切、不多占一块内存）。
 */
class DictationEngine(
    /**
     * 保留参数只为对调用方表明「这个引擎是从 Android Context 构建的」，
     * **不参与模型加载** —— 模型一律走文件路径，见 [SenseVoiceAsr] 的类注释。
     *
     * 之所以不索性删掉：调用方将来可能把模型放进 APK assets，
     * 那时由 [SenseVoiceAsr] 自己决定要不要走 asset 路径，而不是在这个中转层做决定。
     */
    @Suppress("unused") private val assets: AssetManager,
    /** SenseVoice 模型目录（含 model.int8.onnx / tokens.txt）。 */
    private val modelDir: File,
    /** silero VAD 模型文件。 */
    private val vadFile: File,
    private val callback: Callback,
    /**
     * 强制切段间隔（毫秒）。`0` = 关闭，见类注释取舍 5。
     *
     * 做成 lambda 而不是 `Int`：界面上调完要**立刻**生效，而重建引擎等于重新加载
     * 239MB 的 SenseVoice（约 0.9 秒）。这里每个窗口读一次，代价可忽略。
     */
    private val forceCutMs: () -> Int = { 0 },
    /**
     * 识别后端的构造器。为 null 时用默认的 SenseVoice（[modelDir] + [vadFile]）。
     *
     * 做成工厂而不是直接收一个实例：加载 200MB 级的模型要近 1 秒，
     * 放在构造函数里意味着调用方必须在采集线程外面先把它造好，
     * 「模型没装」就会变成调用方的启动异常。让引擎自己 try/catch，
     * 失败时仍有 `ready == false` 的可诊断状态（见下方 [asr] 的初始化）。
     *
     * 也是 A/B 对比的入口：同一段音频、同一条 VAD、同一个切段策略，只换后端，
     * 比出来的数字才有可比性（见 [AsrBackend] 的类注释）。
     */
    private val backendFactory: (() -> AsrBackend)? = null,
) : DictationHub.Listener, AutoCloseable {

    /** 识别结果回调。**全部在采集线程上发生**，实现方需自行切线程再碰 UI。 */
    interface Callback {
        /**
         * 一句话识别完成。[lang] 可能是 [Lang.AUTO]（模型没判定出来）。
         *
         * [forced] = 这句是**按时间硬切**出来的，不是 VAD 等到了静音。
         * 硬切的切口多半落在句子中间（换气、鼓点、随便哪），下游据此决定
         * 「要不要和下一句拼起来再翻译」（见 `InterpretEngine` 的 `mergeable`）。
         * 这个标志只有这里有 —— 事后去猜「是不是半句」只能靠标点和时间差，
         * 而 VAD 切出来的句子也可能没有句末标点，猜错就是整段语义错位。
         */
        fun onUtterance(text: String, lang: Lang, samples: Int, decodeMs: Long, forced: Boolean)

        /** 出错。用于把「模型加载失败」这类问题显示到界面上，而不是只在 logcat 里。 */
        fun onError(message: String)

        /** 是否正处于语音中，用于「正在说话」指示。 */
        fun onSpeaking(speaking: Boolean)
    }

    private val asr: AsrBackend? = try {
        val make = backendFactory ?: { SenseVoiceAsr(modelDir = modelDir, vadFile = vadFile) }
        make().also {
            Log.i(TAG, "ASR 已加载：${it.displayName}（${if (it.isStreaming) "流式" else "离线"}，VAD：$vadFile）")
        }
    } catch (t: Throwable) {
        // 加载失败不抛出去 —— 采集还要继续，界面需要看到原因
        Log.e(TAG, "ASR 加载失败", t)
        callback.onError("ASR 加载失败：${t.message ?: t.javaClass.simpleName}")
        null
    }

    /** 模型没加载起来时引擎仍可挂载，但 [onPcm] 直接返回（避免调用方到处判空）。 */
    val ready: Boolean get() = asr != null

    /** 当前后端的显示名（界面/日志显示「到底在跑哪个模型」，排查时第一眼要看这个）。 */
    val backendName: String get() = asr?.displayName ?: "未加载"

    /** 当前后端是不是流式（流式自带端点，不吃强制切段）。 */
    val streaming: Boolean get() = asr?.isStreaming ?: false

    // ------------------------------------------------------------ 可变状态（只由采集线程访问）

    private var resampler: AudioResampler? = null
    private var lastRate = 0
    private var lastChannels = 0

    /** 攒够 [SenseVoiceAsr.WINDOW_SIZE] 就喂一次 VAD 的窗口暂存区。 */
    private val window = FloatArray(SenseVoiceAsr.WINDOW_SIZE)
    private var windowLen = 0

    /**
     * 交给 native 的那一份。**刻意不复用 [window]**：见 [SenseVoiceAsr] 的说明 ——
     * 复用暂存区在 sherpa 改为持有引用时会静默改掉历史音频。一次分配，长期使用。
     */
    private val feedBuf = FloatArray(SenseVoiceAsr.WINDOW_SIZE)

    /** 闸门状态，见类注释取舍 1。 */
    private var lastLoudAt = 0L
    private var speaking = false

    /**
     * 上一个窗口时能量闸门是否打开。
     *
     * 只有流式后端用得上：流式模型自己做端点检测，没有 `isSpeaking()` 这种
     * 「正在说话」的问询接口，所以「说话中」这个界面指示就退化成能量闸门的状态。
     * 离线后端仍然问 VAD，别改。
     */
    private var gateOpen = false

    /**
     * 闸门关闭期间最近 [PREROLL_WINDOWS] 个窗口的暂存。
     *
     * 开闸时按顺序补喂，见类注释取舍 1 第 3 条。存的是**拷贝**而不是引用
     * [window]：window 是复用缓冲，存引用等于把后面所有窗口都存成同一份数据
     * （这个错本项目在「指针 + 长度」纪律里已经踩过一次）。
     */
    private val preRoll = ArrayDeque<FloatArray>()

    /**
     * 强制切段的累积缓冲，见类注释取舍 5。
     *
     * 只在 [forceCutMs] > 0 时才有内容；一次分配（[FORCED_CAP_SECONDS] 秒），长期使用。
     * **不能**和 [feedBuf] 共用：那个是交给 native 的，这个要跨窗口保留历史音频。
     */
    private val forced = FloatArray(SenseVoiceAsr.SAMPLE_RATE * FORCED_CAP_SECONDS)

    /** [forced] 里已有多少采样点。 */
    private var forcedLen = 0

    // ------------------------------------------------------------ 统计（供界面显示链路是否真的在跑）
    var inputFrames = 0L; private set
    var resampledSamples = 0L; private set
    var windowsFed = 0L; private set
    var windowsSkipped = 0L; private set
    var utterances = 0L; private set
    var totalDecodeMs = 0L; private set
    var lastDecodeMs = 0L; private set

    /**
     * VAD 推理累计耗时。用来**用数据回答**「闸门到底省了多少 CPU」——
     * 如果这个数很小（相对于采集总时长），那闸门的唯一价值就只剩省电，
     * 那时可以把 [PREROLL_WINDOWS] 调得更大甚至取消闸门，进一步保句首。
     */
    var totalVadMs = 0L; private set

    /** 闸门关闭期间被暂存、后来靠 pre-roll 补喂回来的窗口数（验证第 3 条生效）。 */
    var preRollRescued = 0L; private set

    /** 因攒满 [forceCutMs] 而没有等 VAD 端点、直接切出去解码的次数（验证取舍 5 生效）。 */
    var forcedCuts = 0L; private set

    /** 强制切段这条路的解码累计耗时（毫秒），与 [totalDecodeMs] 分开统计：它不经过 VAD。 */
    var forcedDecodeMs = 0L; private set

    // ------------------------------------------------------------ Listener

    override fun onCaptureState(running: Boolean) {
        if (running) {
            reset()
            return
        }
        // 停止采集：把 VAD 里最后半句冲出来，否则用户会觉得「最后一句被吃了」
        val engine = asr
        if (engine != null) {
            // 先冲强制切段的残留：它和 VAD 缓冲覆盖的是**不同**的音频，顺序反了会丢掉尾巴
            // 流式后端没有强制切段缓冲，别去冲一个不存在的段
            if (!engine.isStreaming && forceCutMs() > 0 && forcedLen >= MIN_FLUSH_SAMPLES) {
                val rest = forced.copyOf(forcedLen)
                forcedLen = 0
                val r = runCatching { engine.decodeSegment(rest) }.getOrNull()
                if (r != null) emit(r, 0L, forced = true)
            }
            try {
                val t0 = SystemClock.elapsedRealtime()
                val rest = engine.flush()
                val ms = SystemClock.elapsedRealtime() - t0
                for (s in rest) emit(s, if (rest.size == 1) ms else 0L)
            } catch (t: Throwable) {
                Log.w(TAG, "flush 失败", t)
            }
        }
        reset()
    }

    override fun onPcm(pcm: ByteArray, len: Int, rate: Int, channels: Int) {
        val engine = asr ?: return
        if (rate < SenseVoiceAsr.SAMPLE_RATE || channels < 1) return

        if (rate != lastRate || channels != lastChannels) {
            Log.i(TAG, "输入 $rate Hz / $channels ch → 重采样到 ${SenseVoiceAsr.SAMPLE_RATE} Hz 单声道")
            resampler = AudioResampler(rate, SenseVoiceAsr.SAMPLE_RATE)
            lastRate = rate
            lastChannels = channels
            reset()
        }
        val rs = resampler ?: return

        // 双声道先下混再重采样（顺序不能反，否则会把两声道混成一个之后再平均）
        val mono = if (channels <= 1) {
            rs.feed(pcm, len)
        } else {
            val mixed = downmix(pcm, len, channels)
            rs.feed(mixed, mixed.size)
        }
        if (mono.isEmpty()) return
        inputFrames++
        resampledSamples += mono.size

        // ---- 切窗口 → 闸门 → 喂 VAD ----
        //
        // ⚠️ 闸门判定必须在**这个循环内部**、以**单个窗口**为颗粒度做，
        // 不能挪到循环外面按 `mono` 整批算。见类注释取舍 1：按批算是「句首丢失」的根因。
        var off = 0
        while (off < mono.size) {
            val take = minOf(window.size - windowLen, mono.size - off)
            System.arraycopy(mono, off, window, windowLen, take)
            windowLen += take
            off += take
            if (windowLen < window.size) break
            windowLen = 0

            val now = SystemClock.elapsedRealtime()
            if (rmsOf(window) > GATE_RMS) lastLoudAt = now
            val open = (now - lastLoudAt) <= HANGOVER_MS
            gateOpen = open

            if (!open) {
                windowsSkipped++
                if (preRoll.size >= PREROLL_WINDOWS) preRoll.removeFirst()
                preRoll.addLast(window.copyOf())
                continue
            }

            // 开闸：先把闸门关闭期间攒下的补喂回去 —— 这几窗就是句首（取舍 1 第 3 条）
            while (preRoll.isNotEmpty()) {
                val g = preRoll.removeFirst()
                preRollRescued++
                feedWindow(engine, g)
            }
            feedWindow(engine, window)
        }
    }

    // ------------------------------------------------------------ 内部

    /**
     * 把一个窗口喂给 VAD，并把 VAD 判定的完成句回调出去。
     *
     * 单独抽出来是因为 pre-roll 补喂和正常喂走的是同一条路径 —— 分成两份实现
     * 早晚会漂（本项目在别处已经吃过「两处实现不一致」的亏）。
     */
    private fun feedWindow(engine: AsrBackend, src: FloatArray) {
        windowsFed++
        try {
            // 强制切段的累积必须在 accept 之前：音频要严格按到达顺序进缓冲（取舍 5）
            // 流式后端自己会按端点吐中间结果，「按时间强制切段」对它只会切断上下文
            val forceMs = if (engine.isStreaming) 0 else forceCutMs()
            if (forceMs > 0) appendForced(src)

            System.arraycopy(src, 0, feedBuf, 0, feedBuf.size)
            // accept 内部会立刻解码完成的段，所以耗时统计在这里闭环
            val t0 = SystemClock.elapsedRealtime()
            val segs = engine.accept(feedBuf)
            val ms = SystemClock.elapsedRealtime() - t0
            // 这一窗没出句子 → 这段耗时是**纯 VAD 开销**；出了句子则大部分是 SenseVoice 解码，
            // 记进 totalVadMs 会把两个量搅在一起，就没法回答「闸门省了多少」了。
            if (segs.isEmpty()) totalVadMs += ms
            for (s in segs) {
                // VAD 出了句 ⇒ 这几秒音频已经由它出口了。缓冲里的同一批必须清掉，
                // 否则等攒满强制再切一次，同一段话会出两条字幕。
                if (forceMs > 0) forcedLen = 0
                emit(s, if (segs.size == 1) ms else 0L)
            }
            // 队列清空后再看要不要强制切（可能刚刚被 VAD 清空，那就不该切）
            if (forceMs > 0) maybeForceCut(engine, forceMs)
            // 流式后端没有 VAD，「说话中」用能量闸门的状态代替
            val sp = if (engine.isStreaming) gateOpen else engine.isSpeaking()
            if (sp != speaking) {
                speaking = sp
                callback.onSpeaking(sp)
            }
        } catch (t: Throwable) {
            Log.e(TAG, "VAD/解码失败", t)
            callback.onError("识别失败：${t.message ?: t.javaClass.simpleName}")
        }
    }

    // ------------------------------------------------------------ 强制切段（取舍 5）

    /** 把一个窗口追加进 [forced]，满了就先丢最老的。 */
    private fun appendForced(src: FloatArray) {
        val need = forcedLen + src.size
        if (need > forced.size) {
            // 正常配置（forceCutMs ≤ 10s < FORCED_CAP_SECONDS）走不到这里；
            // 但设置是从 DataStore 读的、也可能被手改，所以宁可贵一点也不能抛 IndexOutOfBounds。
            var drop = minOf(need - forced.size, forcedLen)
            System.arraycopy(forced, drop, forced, 0, forcedLen - drop)
            forcedLen -= drop
        }
        System.arraycopy(src, 0, forced, forcedLen, src.size)
        forcedLen += src.size
    }

    /**
     * 攒够了就绕过 VAD 直接切一段出去解码。
     *
     * 顺序是「先切缓冲、再解码」：解码那几百毫秒里采集线程不读音频（取舍 3），
     * 反过来写的话这段新到的窗口会把缓冲顶乱。
     */
    private fun maybeForceCut(engine: AsrBackend, forceMs: Int) {
        val haveMs = forcedLen * 1000L / SenseVoiceAsr.SAMPLE_RATE
        if (haveMs < forceMs) return
        // 最短一段最少要多长：默认 1.2 秒，但**必须小于**切段间隔本身 ——
        // 间隔被调到 1 秒时若仍要求 1.2 秒，quietestCut 会永远返回 0，
        // 缓冲一路涨到上限再丢最老的音频，表现是「开了强制切段反而一句都不出」。
        val minCut = maxOf(
            SenseVoiceAsr.WINDOW_SIZE * 2,
            minOf(MIN_CUT_SAMPLES, forceMs * SenseVoiceAsr.SAMPLE_RATE / 1000 * 3 / 5),
        )
        val cut = quietestCut(minCut)
        if (cut < minCut) return

        val seg = forced.copyOf(cut)
        val rest = forcedLen - cut
        System.arraycopy(forced, cut, forced, 0, rest)
        forcedLen = rest

        // VAD 缓冲里是**同一段**音频。不丢掉的话，它等静音时再出一次，字幕就重复了。
        // 附带好处：isSpeechDetected 归零，界面「说话中」不会因为强制切段一直亮着。
        engine.resetVad()

        val t0 = SystemClock.elapsedRealtime()
        val r = runCatching { engine.decodeSegment(seg) }.getOrNull()
        val ms = SystemClock.elapsedRealtime() - t0
        forcedCuts++
        forcedDecodeMs += ms
        Log.i(
            TAG,
            "强制切段 #$forcedCuts：${seg.size} 点（${"%.2f".format(seg.size / SenseVoiceAsr.SAMPLE_RATE.toFloat())}s）" +
                "解码 $ms ms，${if (r == null) "无文本（丢弃）" else "出句"}，剩 ${forcedLen} 点",
        )
        // r == null（这段没词/解码失败）也要算切过：否则这段音频会永远赖在缓冲里，
        // 每次窗口进来都重试一遍，把采集线程拖死。
        if (r != null) emit(r, ms, forced = true)
    }

    /**
     * 在缓冲尾部找能量最低处当切点（换气、鼓点间隙）。
     *
     * 只往回看 [CUT_LOOKBACK_MS]：看太远等于为了「切得漂亮」牺牲掉半句话。
     * 步长取 VAD 窗口（32ms）—— 再细的差别人耳分不出来，只是白算。
     * 返回绝对下标；0 表示找不到合适的位置（调用方据此放弃这次切段）。
     */
    private fun quietestCut(minCut: Int): Int {
        val w = SenseVoiceAsr.WINDOW_SIZE
        val tail = minOf(forcedLen - minCut, CUT_LOOKBACK_SAMPLES)
        if (tail < w) return 0
        val base = forcedLen - tail
        var best = -1
        var bestRms = Double.MAX_VALUE
        var off = 0
        while (off + w <= tail) {
            val rms = rmsOf(forced, base + off, w)
            // 严格小于：并列时取最早的，切得更快（延迟优先于「再等 32ms」）
            if (rms < bestRms) {
                bestRms = rms
                best = base + off
            }
            off += w
        }
        return if (best < 0) 0 else best
    }

    private fun emit(s: Segment, decodeMs: Long, forced: Boolean = false) {
        utterances++
        if (decodeMs > 0) {
            lastDecodeMs = decodeMs
            totalDecodeMs += decodeMs
        }
        callback.onUtterance(tidyText(s.text, s.lang), s.lang, s.samples, decodeMs, forced)
    }

    /** 交错多声道 → 单声道（取平均，避免只取一路导致另一路的语音丢失）。 */
    private fun downmix(pcm: ByteArray, len: Int, channels: Int): ByteArray {
        val frameBytes = 2 * channels
        val frames = len / frameBytes
        val out = ByteArray(frames * 2)
        var f = 0
        while (f < frames) {
            var sum = 0
            var c = 0
            while (c < channels) {
                val idx = f * frameBytes + c * 2
                val lo = pcm[idx].toInt() and 0xFF
                val hi = pcm[idx + 1].toInt()
                sum += ((hi shl 8) or lo).toShort().toInt()
                c++
            }
            val v = (sum / channels).coerceIn(-32768, 32767)
            out[f * 2] = (v and 0xFF).toByte()
            out[f * 2 + 1] = ((v shr 8) and 0xFF).toByte()
            f++
        }
        return out
    }

    private fun rmsOf(a: FloatArray): Double {
        var sum = 0.0
        for (v in a) sum += v.toDouble() * v
        return kotlin.math.sqrt(sum / a.size)
    }

    /** 只算 [a] 从 [off] 起 [len] 个点的 RMS（找切点用，避免为 32ms 再复制一次数组）。 */
    private fun rmsOf(a: FloatArray, off: Int, len: Int): Double {
        var sum = 0.0
        var i = off
        val end = off + len
        while (i < end) {
            val v = a[i].toDouble()
            sum += v * v
            i++
        }
        return kotlin.math.sqrt(sum / len)
    }

    private fun reset() {
        resampler?.reset()
        windowLen = 0
        lastLoudAt = 0L
        preRoll.clear()
        forcedLen = 0
        if (speaking) {
            speaking = false
            callback.onSpeaking(false)
        }
    }

    override fun close() {
        runCatching { asr?.close() }
    }

    companion object {
        private const val TAG = "AST.Dictation"

        /** 有信号窗口的 RMS 门限。屏幕音频里人声 RMS 通常 >0.02，底噪 <0.002。 */
        private const val GATE_RMS = 0.004

        /** 说完话之后闸门再多开这么久，避免句尾的轻声被切掉。 */
        private const val HANGOVER_MS = 700L

        /**
         * 闸门关闭时暂存多少个窗口，开闸后补喂。
         *
         * 8 窗 × 32ms ≈ 256ms。取值只需覆盖「起音慢到跨过 RMS 门限」的那一小段：
         * 取值太大只是在长静音里白占内存（每窗 2KB），取太小则句首仍会被削。
         */
        private const val PREROLL_WINDOWS = 8

        // ---- 强制切段（取舍 5）----

        /**
         * 强制切段缓冲最多攒多少秒。
         *
         * 必须**大于**界面上允许的最大切段间隔（[Settings.forceSegmentMs] 上限 10 秒），
         * 否则一满就会先丢最老的音频，等于悄悄吃字。
         */
        private const val FORCED_CAP_SECONDS = 15

        /** 找切点时最多回看多久。见 [quietestCut]。 */
        private const val CUT_LOOKBACK_MS = 800
        private const val CUT_LOOKBACK_SAMPLES = SenseVoiceAsr.SAMPLE_RATE * CUT_LOOKBACK_MS / 1000

        /** 切出来的一段最短多长。再短基本只有气口，解码出来常是噪声幻觉。 */
        private const val MIN_CUT_MS = 1200
        private const val MIN_CUT_SAMPLES = SenseVoiceAsr.SAMPLE_RATE * MIN_CUT_MS / 1000

        /** 停止采集时残留短于这个就不再多花一次解码了。 */
        private const val MIN_FLUSH_SAMPLES = SenseVoiceAsr.SAMPLE_RATE * 400 / 1000
    }
}
