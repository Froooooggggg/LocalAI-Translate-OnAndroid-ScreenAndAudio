package com.screenast.app.core

import android.content.Context
import androidx.datastore.core.DataStore
import androidx.datastore.preferences.core.Preferences
import androidx.datastore.preferences.core.booleanPreferencesKey
import androidx.datastore.preferences.core.edit
import androidx.datastore.preferences.core.floatPreferencesKey
import androidx.datastore.preferences.core.intPreferencesKey
import androidx.datastore.preferences.core.stringPreferencesKey
import androidx.datastore.preferences.preferencesDataStore
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.first
import kotlinx.coroutines.flow.map

/** 应用级 DataStore 单例。 */
val Context.settingsDataStore: DataStore<Preferences> by preferencesDataStore(name = "screenast_settings")

/**
 * 设置读写。整份 [Settings] 拆成扁平 key 存，读的时候再拼回来。
 *
 * 没有用 protobuf/kotlinx-serialization，是为了少一个编译器插件依赖——
 * 这些字段全是简单标量，扁平存反而更稳、更好排查。
 */
class SettingsStore(private val context: Context) {

    private object Keys {
        val sourceLang = stringPreferencesKey("source_lang")
        val targetLang = stringPreferencesKey("target_lang")
        val customTargetLang = stringPreferencesKey("custom_target_lang")
        val customSourceLang = stringPreferencesKey("custom_source_lang")
        val noticeAccepted = intPreferencesKey("notice_accepted")

        val boxColor = intPreferencesKey("box_color")
        val textColor = intPreferencesKey("text_color")
        val textSizeSp = floatPreferencesKey("text_size_sp")
        val cornerRadiusDp = floatPreferencesKey("corner_radius_dp")
        val paddingDp = floatPreferencesKey("padding_dp")
        val widthFraction = floatPreferencesKey("width_fraction")
        val minHeightDp = floatPreferencesKey("min_height_dp")
        val maxLines = intPreferencesKey("max_lines")
        val borderColor = intPreferencesKey("border_color")
        val borderWidthDp = floatPreferencesKey("border_width_dp")

        val offsetXDp = floatPreferencesKey("offset_x_dp")
        val offsetYDp = floatPreferencesKey("offset_y_dp")

        // ---- 悬浮字幕窗（路线图 ⑤）----
        // ⚠️ 新增设置项必须**三处齐加**：这里的 Keys、下面读盘时的 `?: 默认值`、update() 里的写入。
        // 少任何一处，症状都是「改完重启就丢」或「改完永远读不到」，而编译不会报错。
        val subtitleEnabled = booleanPreferencesKey("subtitle_enabled")
        val subtitleWidthDp = floatPreferencesKey("subtitle_width_dp")
        val subtitleHeightDp = floatPreferencesKey("subtitle_height_dp")
        val subtitleMode = stringPreferencesKey("subtitle_mode")

        val providerId = stringPreferencesKey("provider_id")
        val apiKey = stringPreferencesKey("api_key")
        val baseUrl = stringPreferencesKey("base_url")
        val model = stringPreferencesKey("model")
        val temperature = floatPreferencesKey("temperature")
        val stream = booleanPreferencesKey("stream")

        val netApiEnabled = booleanPreferencesKey("net_api_enabled")
        val netApiBaseUrl = stringPreferencesKey("net_api_base_url")
        val netApiKey = stringPreferencesKey("net_api_key")
        val netApiModel = stringPreferencesKey("net_api_model")
        val netApiThinking = stringPreferencesKey("net_api_thinking")

        val proxyEnabled = booleanPreferencesKey("proxy_enabled")
        val proxyHost = stringPreferencesKey("proxy_host")
        val proxyPort = intPreferencesKey("proxy_port")
        val proxyType = stringPreferencesKey("proxy_type")

        // ---- 端侧 NLLB ----
        val nllbModelDir = stringPreferencesKey("nllb_model_dir")
        val nllbEncoderFile = stringPreferencesKey("nllb_encoder_file")
        val nllbDecoderFile = stringPreferencesKey("nllb_decoder_file")
        val nllbDecoderPastFile = stringPreferencesKey("nllb_decoder_past_file")
        val nllbTokenizerFile = stringPreferencesKey("nllb_tokenizer_file")
        val nllbMaxNewTokens = intPreferencesKey("nllb_max_new_tokens")
        val nllbMaxSourceTokens = intPreferencesKey("nllb_max_source_tokens")
        val nllbRepetitionPenalty = floatPreferencesKey("nllb_repetition_penalty")
        val nllbThreads = intPreferencesKey("nllb_threads")

        // ---- 端侧大模型（llama.cpp）----
        val llamaModelDir = stringPreferencesKey("llama_model_file")
        val llamaDevice = stringPreferencesKey("llama_device")
        val llamaNCtx = intPreferencesKey("llama_n_ctx")
        val llamaNThreads = intPreferencesKey("llama_n_threads")
        val llamaNBatch = intPreferencesKey("llama_n_batch")
        val llamaNGpuLayers = intPreferencesKey("llama_n_gpu_layers")
        val llamaMaxNewTokens = intPreferencesKey("llama_max_new_tokens")

        val eagerTranslate = booleanPreferencesKey("eager_translate")
        val maxRevisions = intPreferencesKey("max_revisions")

        /** 强制切段间隔（毫秒），见 [Settings.forceSegmentMs]。 */
        val forceSegmentMs = intPreferencesKey("force_segment_ms")

        /** 识别后端标识，见 [Settings.asrModel]。 */
        val asrModel = stringPreferencesKey("asr_model")

        /** 是否给不带标点的后端补标点，见 [Settings.asrPunct]。 */
        val asrPunct = booleanPreferencesKey("asr_punct")

        /** 识别前是否降噪，见 [Settings.asrDenoise]。 */
        val asrDenoise = booleanPreferencesKey("asr_denoise")

        /** 带几轮对话上文，见 [Settings.contextRounds]。 */
        val contextRounds = intPreferencesKey("context_rounds")

        // ---- 识图（屏幕文字翻译）----
        val visionSourceLang = stringPreferencesKey("vision_source_lang")
        val visionTargetLang = stringPreferencesKey("vision_target_lang")
        val visionIntervalMs = intPreferencesKey("vision_interval_ms")
        val visionTranslate = booleanPreferencesKey("vision_translate")
        val visionEraseMode = intPreferencesKey("vision_erase_mode")
        val visionInpaint = booleanPreferencesKey("vision_inpaint")
        val visionTextColor = intPreferencesKey("vision_text_color")
        val visionOcrServer = booleanPreferencesKey("vision_ocr_server")
        val visionFloatList = booleanPreferencesKey("vision_float_list")
        val visionMinChars = intPreferencesKey("vision_min_chars")
    }

    val settings: Flow<Settings> = context.settingsDataStore.data.map { p ->
        Settings.sanitize(
            Settings(
                sourceLang = Lang.fromCode(p[Keys.sourceLang]),
                targetLang = Lang.fromCode(p[Keys.targetLang]).takeIf { it != Lang.AUTO } ?: Lang.ZH,
                customTargetLang = p[Keys.customTargetLang] ?: Settings().customTargetLang,
                customSourceLang = p[Keys.customSourceLang] ?: Settings().customSourceLang,
                noticeAccepted = p[Keys.noticeAccepted] ?: Settings().noticeAccepted,
                boxColor = p[Keys.boxColor] ?: Settings().boxColor,
                textColor = p[Keys.textColor] ?: Settings().textColor,
                textSizeSp = p[Keys.textSizeSp] ?: Settings().textSizeSp,
                cornerRadiusDp = p[Keys.cornerRadiusDp] ?: Settings().cornerRadiusDp,
                paddingDp = p[Keys.paddingDp] ?: Settings().paddingDp,
                widthFraction = p[Keys.widthFraction] ?: Settings().widthFraction,
                minHeightDp = p[Keys.minHeightDp] ?: Settings().minHeightDp,
                maxLines = p[Keys.maxLines] ?: Settings().maxLines,
                borderColor = p[Keys.borderColor] ?: Settings().borderColor,
                borderWidthDp = p[Keys.borderWidthDp] ?: Settings().borderWidthDp,
                offsetXDp = p[Keys.offsetXDp] ?: Settings().offsetXDp,
                offsetYDp = p[Keys.offsetYDp] ?: Settings().offsetYDp,
                subtitleEnabled = p[Keys.subtitleEnabled] ?: Settings().subtitleEnabled,
                subtitleWidthDp = p[Keys.subtitleWidthDp] ?: Settings().subtitleWidthDp,
                subtitleHeightDp = p[Keys.subtitleHeightDp] ?: Settings().subtitleHeightDp,
                subtitleMode = p[Keys.subtitleMode] ?: Settings().subtitleMode,
                providerId = p[Keys.providerId] ?: Settings().providerId,
                apiKey = p[Keys.apiKey] ?: Settings().apiKey,
                baseUrl = p[Keys.baseUrl] ?: Settings().baseUrl,
                model = p[Keys.model] ?: Settings().model,
                temperature = p[Keys.temperature] ?: Settings().temperature,
                stream = p[Keys.stream] ?: Settings().stream,
                netApiEnabled = p[Keys.netApiEnabled] ?: Settings().netApiEnabled,
                netApiBaseUrl = p[Keys.netApiBaseUrl] ?: Settings().netApiBaseUrl,
                netApiKey = p[Keys.netApiKey] ?: Settings().netApiKey,
                netApiModel = p[Keys.netApiModel] ?: Settings().netApiModel,
                netApiThinking = p[Keys.netApiThinking] ?: Settings().netApiThinking,
                proxyEnabled = p[Keys.proxyEnabled] ?: Settings().proxyEnabled,
                proxyHost = p[Keys.proxyHost] ?: Settings().proxyHost,
                proxyPort = p[Keys.proxyPort] ?: Settings().proxyPort,
                proxyType = p[Keys.proxyType] ?: Settings().proxyType,
                nllbModelDir = p[Keys.nllbModelDir] ?: Settings().nllbModelDir,
                nllbEncoderFile = p[Keys.nllbEncoderFile] ?: Settings().nllbEncoderFile,
                nllbDecoderFile = p[Keys.nllbDecoderFile] ?: Settings().nllbDecoderFile,
                nllbDecoderPastFile = p[Keys.nllbDecoderPastFile] ?: Settings().nllbDecoderPastFile,
                nllbTokenizerFile = p[Keys.nllbTokenizerFile] ?: Settings().nllbTokenizerFile,
                nllbMaxNewTokens = p[Keys.nllbMaxNewTokens] ?: Settings().nllbMaxNewTokens,
                nllbMaxSourceTokens = p[Keys.nllbMaxSourceTokens] ?: Settings().nllbMaxSourceTokens,
                nllbRepetitionPenalty = p[Keys.nllbRepetitionPenalty] ?: Settings().nllbRepetitionPenalty,
                nllbThreads = p[Keys.nllbThreads] ?: Settings().nllbThreads,
                llamaModelDir = p[Keys.llamaModelDir] ?: Settings().llamaModelDir,
                llamaDevice = p[Keys.llamaDevice] ?: Settings().llamaDevice,
                llamaNCtx = p[Keys.llamaNCtx] ?: Settings().llamaNCtx,
                llamaNThreads = p[Keys.llamaNThreads] ?: Settings().llamaNThreads,
                llamaNBatch = p[Keys.llamaNBatch] ?: Settings().llamaNBatch,
                llamaNGpuLayers = p[Keys.llamaNGpuLayers] ?: Settings().llamaNGpuLayers,
                llamaMaxNewTokens = p[Keys.llamaMaxNewTokens] ?: Settings().llamaMaxNewTokens,
                eagerTranslate = p[Keys.eagerTranslate] ?: Settings().eagerTranslate,
                maxRevisions = p[Keys.maxRevisions] ?: Settings().maxRevisions,
                forceSegmentMs = p[Keys.forceSegmentMs] ?: Settings().forceSegmentMs,
                asrModel = p[Keys.asrModel] ?: Settings().asrModel,
                asrPunct = p[Keys.asrPunct] ?: Settings().asrPunct,
                asrDenoise = p[Keys.asrDenoise] ?: Settings().asrDenoise,
                contextRounds = p[Keys.contextRounds] ?: Settings().contextRounds,
                visionSourceLang = Lang.fromCode(
                    p[Keys.visionSourceLang] ?: Settings().visionSourceLang.code
                ),
                visionTargetLang = Lang.fromCode(
                    p[Keys.visionTargetLang] ?: Settings().visionTargetLang.code
                ).takeIf { it != Lang.AUTO } ?: Lang.ZH,
                visionIntervalMs = p[Keys.visionIntervalMs] ?: Settings().visionIntervalMs,
                visionTranslate = p[Keys.visionTranslate] ?: Settings().visionTranslate,
                visionEraseMode = p[Keys.visionEraseMode] ?: Settings().visionEraseMode,
                visionInpaint = p[Keys.visionInpaint] ?: Settings().visionInpaint,
                visionTextColor = p[Keys.visionTextColor] ?: Settings().visionTextColor,
                visionOcrServer = p[Keys.visionOcrServer] ?: Settings().visionOcrServer,
                visionFloatList = p[Keys.visionFloatList] ?: Settings().visionFloatList,
                visionMinChars = p[Keys.visionMinChars] ?: Settings().visionMinChars,
            )
        )
    }

    suspend fun update(transform: (Settings) -> Settings) {
        val current = readOnce()
        val next = Settings.sanitize(transform(current))
        context.settingsDataStore.edit { p ->
            p[Keys.sourceLang] = next.sourceLang.code
            p[Keys.targetLang] = next.targetLang.code
            p[Keys.customTargetLang] = next.customTargetLang
            p[Keys.customSourceLang] = next.customSourceLang
            p[Keys.noticeAccepted] = next.noticeAccepted

            p[Keys.boxColor] = next.boxColor
            p[Keys.textColor] = next.textColor
            p[Keys.textSizeSp] = next.textSizeSp
            p[Keys.cornerRadiusDp] = next.cornerRadiusDp
            p[Keys.paddingDp] = next.paddingDp
            p[Keys.widthFraction] = next.widthFraction
            p[Keys.minHeightDp] = next.minHeightDp
            p[Keys.maxLines] = next.maxLines
            p[Keys.borderColor] = next.borderColor
            p[Keys.borderWidthDp] = next.borderWidthDp

            p[Keys.offsetXDp] = next.offsetXDp
            p[Keys.offsetYDp] = next.offsetYDp

            p[Keys.subtitleEnabled] = next.subtitleEnabled
            p[Keys.subtitleWidthDp] = next.subtitleWidthDp
            p[Keys.subtitleHeightDp] = next.subtitleHeightDp
            p[Keys.subtitleMode] = next.subtitleMode

            p[Keys.providerId] = next.providerId
            p[Keys.apiKey] = next.apiKey
            p[Keys.baseUrl] = next.baseUrl
            p[Keys.model] = next.model
            p[Keys.temperature] = next.temperature
            p[Keys.stream] = next.stream

            p[Keys.netApiEnabled] = next.netApiEnabled
            p[Keys.netApiBaseUrl] = next.netApiBaseUrl
            p[Keys.netApiKey] = next.netApiKey
            p[Keys.netApiModel] = next.netApiModel
            p[Keys.netApiThinking] = next.netApiThinking

            p[Keys.proxyEnabled] = next.proxyEnabled
            p[Keys.proxyHost] = next.proxyHost
            p[Keys.proxyPort] = next.proxyPort
            p[Keys.proxyType] = next.proxyType

            p[Keys.nllbModelDir] = next.nllbModelDir
            p[Keys.nllbEncoderFile] = next.nllbEncoderFile
            p[Keys.nllbDecoderFile] = next.nllbDecoderFile
            p[Keys.nllbDecoderPastFile] = next.nllbDecoderPastFile
            p[Keys.nllbTokenizerFile] = next.nllbTokenizerFile
            p[Keys.nllbMaxNewTokens] = next.nllbMaxNewTokens
            p[Keys.nllbMaxSourceTokens] = next.nllbMaxSourceTokens
            p[Keys.nllbRepetitionPenalty] = next.nllbRepetitionPenalty
            p[Keys.nllbThreads] = next.nllbThreads

            p[Keys.llamaModelDir] = next.llamaModelDir
            p[Keys.llamaDevice] = next.llamaDevice
            p[Keys.llamaNCtx] = next.llamaNCtx
            p[Keys.llamaNThreads] = next.llamaNThreads
            p[Keys.llamaNBatch] = next.llamaNBatch
            p[Keys.llamaNGpuLayers] = next.llamaNGpuLayers
            p[Keys.llamaMaxNewTokens] = next.llamaMaxNewTokens

            p[Keys.eagerTranslate] = next.eagerTranslate
            p[Keys.maxRevisions] = next.maxRevisions
            p[Keys.forceSegmentMs] = next.forceSegmentMs
            p[Keys.contextRounds] = next.contextRounds
            p[Keys.visionSourceLang] = next.visionSourceLang.code
            p[Keys.visionTargetLang] = next.visionTargetLang.code
            p[Keys.visionIntervalMs] = next.visionIntervalMs
            p[Keys.visionTranslate] = next.visionTranslate
            p[Keys.visionEraseMode] = next.visionEraseMode
            p[Keys.visionInpaint] = next.visionInpaint
            p[Keys.visionTextColor] = next.visionTextColor
            p[Keys.visionOcrServer] = next.visionOcrServer
            p[Keys.visionFloatList] = next.visionFloatList
            p[Keys.visionMinChars] = next.visionMinChars
            p[Keys.asrModel] = next.asrModel
            p[Keys.asrPunct] = next.asrPunct
            p[Keys.asrDenoise] = next.asrDenoise
        }
    }

    private suspend fun readOnce(): Settings = settings.first()
}

/** 便于在非 Compose 环境里同步取一次当前设置。 */
suspend fun Context.currentSettings(): Settings = SettingsStore(this).settings.first()
