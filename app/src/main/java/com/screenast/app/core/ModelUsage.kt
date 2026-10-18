package com.screenast.app.core

/**
 * 每个模型文件「能用来干什么」的一句话说明。
 *
 * 2026-10-18 用户 m06804：「在下载的模型下面简单描述下这个模型可以用来干啥」——
 * `ModelImporter.Item.note` 只是「它是什么」（SenseVoice / 标点词表），
 * 装上之后能干嘛还得再说一句，不然一屏文件名很难判断该不该下。
 */
object ModelUsage {

    /** 返回一句人话；清单外的文件返回空串。 */
    fun of(rel: String): String = when (rel) {
        // ---------------- 翻译：端侧大模型 ----------------
        "llama/Huihui-Qwen3.5-2B-abliterated.Q4_K_M.gguf" ->
            "端侧大模型翻译：按句意译、能带上下文，NPU 上一句约 3 秒；不装就只能用 NLLB 或云端。"

        // ---------------- 语音识别 ----------------
        "asr/model.int8.onnx" ->
            "语音识别主模型（SenseVoice）：中英日韩粤都能听，默认就用它。"
        "asr/tokens.txt" ->
            "上一条的词表。少它识别器起不来，跟主模型一起装。"
        "asr/paraformer-zh/model.int8.onnx" ->
            "另一套中文识别：纯中文素材上标点更顺，缺点是只认中文。"
        "asr/paraformer-zh/tokens.txt" ->
            "Paraformer 的词表，和它的 model.int8.onnx 同装。"
        "asr/streaming-paraformer/encoder.int8.onnx" ->
            "流式识别编码器：一边说一边出字，适合直播/长句；延迟低、准度略降。"
        "asr/streaming-paraformer/decoder.int8.onnx" ->
            "流式识别的解码器，必须和编码器一起装。"
        "asr/streaming-paraformer/tokens.txt" ->
            "流式识别的词表，三个文件是一套。"
        "asr/whisper-small/small-encoder.int8.onnx" ->
            "Whisper 多语种识别：小语种兜底（日/韩/俄/西…），比 SenseVoice 慢。"
        "asr/whisper-small/small-decoder.int8.onnx" ->
            "Whisper 的解码器，和编码器成对。"
        "asr/whisper-small/small-tokens.txt" ->
            "Whisper 的词表。"
        "asr/punct-ct-transformer/model.onnx" ->
            "标点恢复：给本来不出标点的后端（流式/Whisper）补逗号句号问号。"
        "asr/punct-ct-transformer/tokens.json" ->
            "标点模型的词表，和 model.onnx 一起装。"
        "asr/ten-vad/ten-vad.onnx" ->
            "更准的断句（静音检测）：装上就顶替内置的 Silero，噪声环境下不容易把一句话切碎。"

        // ---------------- 翻译：NLLB ----------------
        "nllb/onnx/encoder_model_int8.onnx" ->
            "NLLB 句级翻译（编码器）：200 语种互译，几百毫秒一句，但不看上下文。"
        "nllb/onnx/decoder_model_quantized.onnx" ->
            "NLLB 解码器，和编码器一起装。"
        "nllb/onnx/decoder_with_past_model_quantized.onnx" ->
            "NLLB 带缓存解码器：长句更快，缺了它会退化成逐字解码。"
        "nllb/tokenizer.json" ->

            "NLLB 词表（HF 原版 JSON）：下载后应用自己编译成 tokenizer.bin。"
        "nllb/tokenizer.bin" ->
            "NLLB 词表（由 HF 的 tokenizer.json 编译而来）：没有它 NLLB 直接报错。"

        // ---------------- 识图 OCR ----------------
        "ocr/ppocrv5-mobile-det.onnx" ->
            "识图的找字框（mobile）：一帧几十毫秒，日常够用。"
        "ocr/ppocrv5-mobile-rec.onnx" ->
            "识图的认字（mobile）：和 det 配套。"
        "ocr/ppocrv5-server-det.onnx" ->
            "高精度找字框（server）：小字、密集排版、视频字幕更不容易漏；一帧慢 3~4 秒。"
        "ocr/ppocrv5-server-rec.onnx" ->
            "高精度认字（server）：糊字、艺术字更准，和 server det 配套。"
        "inpaint/migan_pipeline_v2.onnx" ->
            "识图的「擦除」档：把原文擦掉再用周围画面补背景（不想要同色块贴片时用）。"

        else -> ""
    }

    /** 一行「干什么用 + 谁需要它」，UI 直接铺在文件名下面。 */
    fun line(rel: String): String = of(rel).ifBlank { "辅助文件：跟着同目录的主模型装。" }
}
