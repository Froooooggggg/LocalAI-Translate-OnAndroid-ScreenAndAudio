package com.screenast.app.core

/**
 * 全部可调项。用户需求：悬浮框透明度、字体颜色、字号、框的大小与颜色都要可调。
 *
 * 用不可变 data class 承载，便于整份存进 DataStore、也便于 Compose 直接做状态比较。
 */
data class Settings(
    // ---- 语言 ----
    /** 源语言；AUTO 表示让 ASR 自行判断。 */
    val sourceLang: Lang = Lang.AUTO,
    /** 目标语言；不能是 AUTO。 */
    val targetLang: Lang = Lang.ZH,
    /**
     * 手填的目标语种名（「自定义」那一栏）。非空时覆盖 [targetLang]。
     *
     * 2026-10-18 用户要求：「万一用户有翻译其它语言需求，让用户自己填写语言而非给出一个
     * 超长的选择框，将目标语言和源语言兜底的对应语言写到对应模型的 Prompt 就行」。
     */
    val customTargetLang: String = "",
    /** 手填的源语种名（「源语言兜底」的自定义项）；非空时写进提示词。 */
    val customSourceLang: String = "",

    /**
     * 用户公告同意状态（公告版本号）。
     *
     * 0 = 没同意过 → 启动时弹强制公告；等于 [com.screenast.app.legal.Notice.VERSION] 就不再弹。
     * 以后改条款只要把 Notice.VERSION +1，老用户下次启动会重新看到。
     */
    val noticeAccepted: Int = 0,

    // ---- 悬浮窗外观 ----
    /** 框的底色，ARGB。默认半透明黑。 */
    val boxColor: Int = 0xC0000000.toInt(),
    /** 字体颜色，ARGB。 */
    val textColor: Int = 0xFFFFFFFF.toInt(),
    /** 字号，sp。 */
    val textSizeSp: Float = 18f,
    /** 圆角半径，dp。 */
    val cornerRadiusDp: Float = 12f,
    /** 内边距，dp。 */
    val paddingDp: Float = 10f,
    /** 框宽占屏幕宽度的比例（0.3–1.0）。1.0 = 满宽。 */
    val widthFraction: Float = 0.92f,
    /** 框的最小高度，dp（内容超出会向下生长）。 */
    val minHeightDp: Float = 0f,
    /** 最多显示几行。 */
    val maxLines: Int = 4,
    /** 是否描边，以及描边颜色。 */
    val borderColor: Int = 0x00000000,
    val borderWidthDp: Float = 1f,

    // ---- 悬浮窗位置 ----
    /** 相对屏幕左上角的偏移，dp。默认贴顶部。 */
    val offsetXDp: Float = 24f,
    val offsetYDp: Float = 120f,

    // ---- 悬浮字幕窗的开关与尺寸（2026-10-17 新增，路线图 ⑤）----
    /**
     * 悬浮窗开关。**落盘**是有意的：用户开过一次之后，下次进同传台就该自动出现，
     * 而不是每次开机都要再点一下。
     */
    val subtitleEnabled: Boolean = false,
    /**
     * 用户拖过右下角之后的**实际**宽高，dp。`0` = 没手动调过，按
     * [widthFraction] / 「刚好放 5 条」自动算。
     *
     * 为什么不用 [widthFraction] 代替：那是比例，用户手动拖出来的是绝对尺寸，
     * 而且宽度和高度要各自独立（拖窄但不改高很常见）。
     */
    val subtitleWidthDp: Float = 0f,
    val subtitleHeightDp: Float = 0f,
    /**
     * 框里显示什么，取值为 [SUBTITLE_MODES] 之一：
     * `both` 双语（原文小字 + 译文）/ `target` 仅译文 / `source` 仅原文。
     *
     * 为什么值得单独一个开关：两种用法差别很大——边看边听要双语对照，
     * 纯看字幕的人只想要一行译文（浮窗占屏越少越好）；而原文模式在读外语
     * 但暂时不想等翻译时最有用。
     */
    val subtitleMode: String = "both",

    // ---- 翻译 provider ----
    /**
     * 取值为 [com.screenast.app.translate.ProviderKind] 的 name。
     * 内置：`nllb`（端侧离线）、`deepseek`/`qwen`/`glm`/`moonshot`/`siliconflow`/`custom`（在线 API）。
     */
    val providerId: String = "nllb",
    val apiKey: String = "",
    val baseUrl: String = "",
    val model: String = "",
    /** 温度，越低越稳定。字幕翻译用低温。 */
    val temperature: Float = 0.2f,
    /** 是否要求服务端流式返回（首字延迟的关键）。 */
    val stream: Boolean = true,

    // ---- 「网络API调用」开关（2026-10-17 新增）----
    /**
     * 打开后翻译**一律**走 HTTP 接口，且**绝不加载本地 LLM**。
     *
     * 这正是这个开关存在的理由：端侧 2B 要占掉 1.2GB 内存、首次加载 4 秒；用户只想看字幕、
     * 不想为一句句子付这个代价时，直接把它关进网络请求里就行。关掉时回到 [providerId]
     * 那套老逻辑（llama / nllb / 云端预设）。
     */
    val netApiEnabled: Boolean = false,
    /**
     * 网络翻译的接口地址，填到 `/chat/completions` 之前。空 = [DEFAULT_NET_API_BASE_URL]。
     *
     * 单独一组字段而不是复用 [baseUrl]：老字段是「云端预设」那一排按钮的，默认值按
     * providerId 变；开关打开时用户明确指定了是 DeepSeek，不该被上面那排按钮带跑。
     */
    val netApiBaseUrl: String = "",
    /** 网络翻译的 API Key。空着也能开开关，但第一句就会明确报错——不静默降级。 */
    val netApiKey: String = "",
    /**
     * 网络翻译的模型名，**由用户自己填**（不给预设）。
     *
     * 为什么不给预设：每个服务商、甚至同一个服务商的每个中转，模型名都不一样；
     * 写死一个"看起来对"的名字（例如 deepseek-v4.1-flash）只会在服务端回
     * `Model Not Exist` 时把用户带进"是不是 Key 坏了"的死胡同。
     * 空 = 没填，翻译会明确报「没填模型名称」而不是拿默认值硬试。
     */
    val netApiModel: String = "",
    /**
     * 思考强度：`off` / `low` / `medium` / `high`，见 [NET_API_THINKING]。
     *
     * `off` 时**一个思考参数都不发**（对不认识它的服务端最安全，普通 flash 模型本来也不思考）；
     * 其它档位发 `thinking.type=enabled` + `reasoning_effort`。
     */
    val netApiThinking: String = "off",

    // ---- 端侧 NLLB ----
    /**
     * 模型目录。空 = 用应用私有目录 `filesDir/models/nllb`。
     * 填绝对路径可指向外部导入的模型（换更大模型时用）。
     */
    val nllbModelDir: String = "",
    val nllbEncoderFile: String = "encoder_model_int8.onnx",
    /**
     * ⚠️ 解码需要**两个** decoder 文件，缺一不可：
     *
     * `nllbDecoderFile` 负责首步（3 个输入：input_ids + encoder_hidden_states +
     * encoder_attention_mask），产出 48 个 present 供后续复用；
     * `nllbDecoderPastFile` 负责之后每一步（2 个普通输入 + 48 个 past），只产出 24 个 present。
     *
     * ⚠️ 必须用**独立** decoder，不要用 `decoder_model_merged_quantized.onnx`。
     * merged 变体把「有无 KV 缓存」两条路径塞进同一个图并用 `use_cache_branch` 选择，
     * 在 ONNX Runtime 1.31 上首步必然触发：
     *   Reshape node '/model/decoder/layers.0/encoder_attn/Reshape_4'
     *   The dimension with value zero exceeds the dimension size of the input tensor.
     * 而绕开它的办法（给个假长度 past + 走缓存分支）会让跨注意力真的对上零长度张量，
     * 输出是垃圾。独立 decoder 只有 3 个输入，没有这个坑。
     *
     * 另一条已验证的接口细节：首步时 `decoder_with_past_model` 需要**全部 48 个**
     * past 输入（encoder 侧 24 + decoder 侧 24），但它只输出 decoder 侧 24 个。
     * 所以首步产出的 encoder 侧 present 必须在整个解码过程里一直留着。
     */
    val nllbDecoderFile: String = "decoder_model_quantized.onnx",
    /** 增量解码用。见 [nllbDecoderFile] 的说明，这个是**必需**的第二个文件。 */
    val nllbDecoderPastFile: String = "decoder_with_past_model_quantized.onnx",
    /**
     * 分词器文件。**Android 侧读的是编译后的二进制**，不是 HF 的 json/model：
     * `app/src/main/assets/nllb/tokenizer.bin`（由 tools/build_nllb_tokenizer.py 生成）。
     * Android 上没有 SentencePiece 的现成绑定，运行时不解析 json/model。
     */
    val nllbTokenizerFile: String = "nllb/tokenizer.bin",
    /** 单次翻译最多生成多少个 token。字幕场景不需要长句。 */
    val nllbMaxNewTokens: Int = 96,
    /** 输入超过这个 token 数就截断，防止极端长句拖垮解码。 */
    val nllbMaxSourceTokens: Int = 256,
    /** 贪心解码时对 logits 的重复惩罚，1.0 = 关闭。NLLB 容易复读，建议略大于 1。 */
    val nllbRepetitionPenalty: Float = 1.1f,
    /** 解码线程数。0 = 交给运行时决定（通常 = 大核数量）。 */
    val nllbThreads: Int = 4,

    // ---- 端侧大模型（llama.cpp + Qwen3.5-2B）----
    /**
     * 模型**目录**（不是文件路径）。空 = 按 [com.screenast.app.translate.llama.LlamaModels] 的默认顺序找。
     *
     * ⚠️ 与 NLLB 不同，这里目录里只要**一个** .gguf（1.27GB）。
     * 查找顺序见 `LlamaModels.candidates`。
     */
    val llamaModelDir: String = "",
    /**
     * 推理后端。取值：`HTP0`（Hexagon NPU，最快）/ `CPU` / `GPUOpenCL` / `auto`。
     *
     * `auto` 先在 HTP0 上试，打不开就退到 CPU —— 见 `LlamaTranslationProvider.load`。
     * 真机实测：NPU 首字 60~92ms、整句 183~501ms；CPU 首字 313~470ms、整句 450~1221ms。
     */
    val llamaDevice: String = "auto",
    /** 上下文长度。无上下文策略下 1024 绰绰有余（单句提示词约 60~120 token）。 */
    val llamaNCtx: Int = 1024,
    val llamaNThreads: Int = 4,
    val llamaNBatch: Int = 256,
    /** 放到加速器上的层数。-1 = 全部（NPU/GPU 上必须全部放，否则每层都要过 PCIe 等价的总线）。 */
    val llamaNGpuLayers: Int = -1,
    /** 单次翻译最多生成多少个 token。中文译文一句话通常 10~30 token，160 是防退化上限。 */
    val llamaMaxNewTokens: Int = 160,

    // ---- 网络 ----
    /** 代理开关。开启后所有翻译请求走下面的代理。 */
    val proxyEnabled: Boolean = false,
    /** 形如 127.0.0.1:5188 或 http://127.0.0.1:5188。 */
    val proxyHost: String = "127.0.0.1",
    val proxyPort: Int = 5188,
    /** 代理协议：HTTP 或 SOCKS。 */
    val proxyType: String = "HTTP",

    // ---- 运行行为 ----
    /** 首字优先：partial 结果即刻送翻译，不等整句。 */
    val eagerTranslate: Boolean = true,
    /** 同一句最多允许几次增量重译（防止把 API 打爆）。 */
    val maxRevisions: Int = 2,
    /**
     * 强制切段长度，毫秒。`0` = 关闭（只靠 VAD 端点）。
     *
     * ## 为什么必须有这个旋钮
     *
     * SenseVoice 是**离线**模型：一句话必须先「说完」才能解码，而「说完」完全由
     * silero VAD 的静音判定给出（minSilenceDuration = 0.5s）。这适用于演讲、对话、
     * 影视剧，但**歌曲 / 连唱 / 一路不停的旁白**里几乎不存在 0.5 秒静音：
     * VAD 永远等不到端点，或者干脆不把音乐当语音，于是字幕半天不出来一句。
     *
     * 设为 4000 后，即使完全没有停顿，攒满约 4 秒也强制切一段直接送解码；
     * 切点还会在最后 800ms 内自动挪到能量最低处（换气/鼓点间隙），
     * 尽量不切在字中间。等价于把「等静音」换成「按时间出句」。
     *
     * 取值 `0`（关闭）或 2000~10000。太低会把一句话切碎（译文的上下文不足），
     * 太高则又回到「半天一句」。
     */
    val forceSegmentMs: Int = 4000,

    /**
     * 识别后端标识，取值见 [com.screenast.app.asr.AsrFactory.KINDS]：
     * `sensevoice` / `paraformer-stream` / `paraformer` / `whisper`。
     *
     * ## 为什么不在这里做合法性校验
     *
     * 合法值定义在 `AsrFactory` 里（那里还绑着「这个后端需要哪些模型文件」）。
     * 在 core 层再抄一份白名单，两处一定会漂。这里只保证**非空**，
     * 真正的归一化由 `AsrFactory.normalize()` 完成 —— 它把任何不认识的串
     * 都退回 SenseVoice，效果等同于「回出厂设置」，不会让 App 起不来。
     */
    val asrModel: String = "sensevoice",

    /**
     * 给**不自带标点**的后端补标点（Paraformer / Whisper）。
     *
     * SenseVoice 自带标点与逆文本归一化，开关对它无效（再补一遍会多出句号）。
     * 补标点要额外加载一个 70MB 的 CT-Transformer 模型（约 20ms/句），
     * 只在真的换了模型时才有意义，所以做成开关而不是无条件开。
     */
    val asrPunct: Boolean = true,

    /**
     * 识别前是否用 GTCRN 降噪（只作用于「送进识别器」的那一份音频）。
     *
     * 默认关：干净音源上降噪只会引入失真。带背景音乐的视频/直播打开它更稳。
     */
    val asrDenoise: Boolean = false,

    /**
     * 带几轮对话上文去翻译。`0` = 无上下文（默认，等价于加这个参数之前的行为）。
     *
     * ## 为什么需要它
     *
     * 逐句独立翻译在**指代、省略、承前启后**上一定会错：日语里主语常省略，
     * 单句看到的「はい、そうです」只能翻成「是的」，而带上文就能翻成
     * 「是的，就是那个意思」。实测同一段日语，2B 带 1 轮上文后
     * 「それもはどれくらい最先端なんだか」才被翻成通顺的中文。
     *
     * ## 为什么上限是 10
     *
     * 每轮约 `原文 + 译文 + 8 个模板 token`。10 轮在实测语料上 ≈300~500 token，
     * NPU 预填 2000 tok/s 只值 150~250ms；再长就会开始顶破「首字 <1 秒」的硬指标。
     * 实现在 [com.screenast.app.interpret.InterpretEngine.MAX_CONTEXT_ROUNDS]。
     */
    val contextRounds: Int = 0,

    // ---- 识图（屏幕文字 → 识别 → 翻译 → 贴回）----

    /**
     * 识图页的源语言。默认 [Lang.AUTO]。
     *
     * PP-OCRv5 的 rec 模型本身覆盖 106 种语言（简中/繁中/英/日同吃），
     * 所以「源语言」对 OCR 不是必需项，只在后续接「按语种选专用 rec」时才有用。
     */
    val visionSourceLang: Lang = Lang.AUTO,
    /** 识图页译成哪种语言（不能是 AUTO）。 */
    val visionTargetLang: Lang = Lang.ZH,
    /** 自动识图间隔，毫秒。0 = 不自动，只手动抓帧。 */
    val visionIntervalMs: Int = 0,
    /** 识别出的原文是否要翻译。关掉 = 只做 OCR（省电、省显存）。 */
    val visionTranslate: Boolean = true,
    /**
     * 贴回方式。见 [VISION_ERASE_MODES]。
     *
     * 0 = 同色贴片（默认）：用框底色盖住原文再写译文，不加载任何模型。
     * 1 = 模糊贴片：把那块区域自己糊掉垫底再写译文，不加载模型、比纯色块自然。
     * 2 = MI-GAN 擦除：先补背景再写字，约 20~50MB，属后续阶段。
     */
    val visionEraseMode: Int = 0,
    /**
     * 擦除档是否用 MI-GAN 补背景。
     *
     * true（默认）= 模型补背景；false = 不加载模型，直接用「框周围平均色」盖住原文再写字
     * （用户 m03633：把自动补背景的 AI 做成可禁用，禁用后走程序贴片）。
     */
    val visionInpaint: Boolean = true,
    /**
     * 识别引擎用哪套 PP-OCRv5 权重。
     *
     * false（默认）= mobile（21MB，一帧约 1s）；true = server（164MB，一帧 3~5s）。
     * 见 `vision/VisionModels.Variant`。切到 server 但文件没放时自动退回 mobile。
     */
    val visionOcrServer: Boolean = false,
    /**
     * 短文本过滤阈值（0~5，含符号）。
     *
     * 用户 m04363：识别出来长度小于这个数的块一律视为噪声（`ln`、`SV`、`6175` 这种），
     * OCR 之后直接剔除，**不进翻译、不贴回、不进浮窗列表**。
     * 判据是 `text.trim().length`（标点算一个字符）；0 = 关掉这条过滤。
     * 默认 2：单字符/空串基本只有噪声，而 `OK`、`3`、`？` 这类两字符的还留着。
     */
    val visionMinChars: Int = 2,
    /**
     * 识图完把这一帧**所有**识别结果（原文 + 译文）列进悬浮窗。
     *
     * 用户 m04158：把同传那个可拖动/缩放的悬浮窗也搬到识图来，做成可开关 ——
     * 屏幕上有些字太小（或贴片本身放不下），需要一份能改字号的文字列表。
     * false（默认）= 不开浮窗，只贴回屏幕。
     */
    val visionFloatList: Boolean = false,
) {
    companion object {
        /** [subtitleMode] 的合法取值。顺序即 UI 展示顺序。 */
        val SUBTITLE_MODES = listOf("both", "target", "source")

        /**
         * 识图「贴回」方式的合法取值 → 说明。顺序即 UI 展示顺序。
         *
         * 只放**端侧真做得到**的两档：0 贴片（已定方案）、2 MI-GAN 擦除（后续）。
         * 生成式重写文字（AnyText/TextDiffuser 那类扩散模型）在手机上跑不动，不进表。
         */
        val VISION_ERASE_MODES = listOf(
            0 to "贴片（同色块盖住原文，不加载模型）",
            1 to "模糊（把原文字那块自己糊掉垫底，不加载模型）",
            2 to "擦除（MI-GAN 补背景，约 20~50MB，后续）",
        )

        /**
         * 网络 API 的接口地址**故意没有默认值**（空 = 没填，直接报错让用户填）。
         *
         * 早先默认填 `https://api.deepseek.com/v1`，等于替用户假定「一定用官方直连」；
         * 实际用中转/自建网关的人一开开关就会打到错误的域名上，报错还像 Key 坏了。
         * 模型名同理，见 [Settings.netApiModel]。
         */
        const val DEFAULT_NET_API_BASE_URL = ""

        /** [netApiThinking] 的合法取值 → 界面文案。第一项是默认值。 */
        val NET_API_THINKING = listOf(
            "off" to "关",
            "low" to "低",
            "medium" to "中",
            "high" to "高",
        )

        /** 目标语言不允许为 AUTO，读盘后用它兜底修正。 */
        fun sanitize(s: Settings): Settings {
            var r = if (s.targetLang == Lang.AUTO) s.copy(targetLang = Lang.ZH) else s
            // 手改 SharedPreferences / 老版本升级都可能留下非法值，别让框变空白。
            if (r.subtitleMode !in SUBTITLE_MODES) r = r.copy(subtitleMode = "both")
            if (r.netApiThinking !in NET_API_THINKING.map { it.first }) r = r.copy(netApiThinking = "off")
            // netApiModel 空着是**合法状态**（表示用户还没填），不要再拿默认值把它顶掉 ——
            // 以前这里会填回一个猜测的模型名，于是「没填」和「填错了」在日志里长得一模一样。
            // 老版本自动填进去的那个猜测值（deepseek-v4.1-flash）在这里清掉：
            // 它是本项目自己编的名字，服务端不认，留着只会继续报 Model Not Exist。
            if (r.netApiModel == "deepseek-v4.1-flash") r = r.copy(netApiModel = "")
            // 老版本默认填进去的官方地址同样清掉：用中转/自建网关的人留着它会一直打到官方域名。
            if (r.netApiBaseUrl.trim() == "https://api.deepseek.com/v1") r = r.copy(netApiBaseUrl = "")

            // 切段间隔越界会让「按时间出句」变成「每秒一句」或「永远不切」，两者都比默认值糟。
            if (r.forceSegmentMs < 0 || r.forceSegmentMs > 10_000) r = r.copy(forceSegmentMs = 4000)
            // 空串会让 AsrFactory.create 走进 error 分支，界面显示「模型没装」；退回默认后端更友好
            if (r.asrModel.isBlank()) r = r.copy(asrModel = "sensevoice")
            // 超上限会让提示词把 KV 缓存撑爆（2B 在 nCtx=1024 下尤其危险），
            // 负数更没意义 —— 统一夹到 0..MAX，等价于「回默认」。
            if (r.contextRounds < 0 || r.contextRounds > 10) {
                r = r.copy(contextRounds = r.contextRounds.coerceIn(0, 10))
            }
            // 识图：0~10 秒；负数/超大间隔会让抓帧线程空转或几乎不跑。
            if (r.visionIntervalMs < 0 || r.visionIntervalMs > 10_000) {
                r = r.copy(visionIntervalMs = r.visionIntervalMs.coerceIn(0, 10_000))
            }
            // 未知的贴回方式会让绘制层没有分支可走，退回「贴片」这个一定能跑的。
            if (r.visionEraseMode !in VISION_ERASE_MODES.map { it.first }) r = r.copy(visionEraseMode = 0)
            if (r.visionTargetLang == Lang.AUTO) r = r.copy(visionTargetLang = Lang.ZH)
            return r
        }
    }
}
