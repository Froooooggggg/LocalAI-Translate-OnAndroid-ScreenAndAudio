# 第三方组件与模型

本仓库自身代码以 Apache-2.0 发布（见 `LICENSE`）。下面列出随包分发的第三方二进制、以及软件内可下载的模型及其许可。
**本仓库不分发模型权重**（`Models/` 与 `模型/` 目录在 `.gitignore` 里），模型由使用者自行下载。

## 随 APK 分发的二进制

| 组件 | 版本/来源 | 许可 | 说明 |
| --- | --- | --- | --- |
| llama.cpp（`libllama.so`、`libggml-*.so`） | 官方预编译 Android 包（`llama-b*`） | MIT | 端侧 GGUF 推理；`libggml-hexagon.so` 用于 Hexagon NPU |
| ONNX Runtime Android | 1.31.0（Maven `com.microsoft.onnxruntime:onnxruntime-android`） | MIT | OCR / NLLB / VAD / 降噪 / 标点推理 |
| sherpa-onnx | 1.13.8（`third_party/sherpa-onnx/sherpa-onnx-1.13.8.aar`，已按 `tools/patch_sherpa_soname.py` 打 soname 补丁） | Apache-2.0 | 流式 ASR 与 VAD 的一层封装；内嵌 `libonnxruntime_sherpa.so` |
| Silero VAD（`assets/asr/silero_vad.onnx`） | sherpa-onnx 定制版（643,854 B） | MIT | 断句；**该定制版与 CSDN/HF 的官方版 I/O 不兼容**，不可互相替换 |
| GTCRN 降噪（`assets/asr/gtcrn/gtcrn_simple.onnx`） | `bitsydarel/gtcrn-onnx`（535,638 B） | MIT | 识别前降噪 |
| PP-OCRv5 字典（`assets/ocr/ppocrv5_dict.txt`） | `bukuroo/PPOCRv5-ONNX` | Apache-2.0 | 识别词表 |

## 软件内可下载的模型（不随包分发）

| 模型 | 直链来源 | 许可 | 备注 |
| --- | --- | --- | --- |
| PP-OCRv5 mobile/server det+rec | `bukuroo/PPOCRv5-ONNX` | Apache-2.0 | 默认 mobile；server 为高精度可选档 |
| SenseVoice（`model.int8.onnx`、`tokens.txt`） | `csukuangfj/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17` | Apache-2.0（模型卡） | 默认 ASR |
| Paraformer / Streaming Paraformer | `csukuangfj/sherpa-onnx-paraformer-zh-2023-09-14`、`...-streaming-paraformer-bilingual-zh-en` | Apache-2.0 | 可选 ASR |
| Whisper small | `csukuangfj/sherpa-onnx-whisper-small` | MIT | 可选 ASR |
| 中英标点 CT-Transformer | `csukuangfj/sherpa-onnx-punct-ct-transformer-zh-en-vocab272727-2024-04-12` | Apache-2.0 | 给不带标点的后端补标点 |
| TEN-VAD | `TEN-framework/ten-vad` | Apache-2.0 | 替代 Silero 的 VAD |
| MI-GAN 补背景 | `andraniksargsyan/migan`（`migan_pipeline_v2.onnx`，26.8MiB） | MIT | 识图「擦除」档用 |
| Qwen3.5-2B GGUF（Q4_K_M） | `mradermacher/Qwen3.5-2B_Abliterated-GGUF` | Apache-2.0（基础模型 Qwen 系列） | 端侧翻译主力；也可换成任意 chat 模板兼容的 gguf |
| **NLLB-200-distilled-600M** | `Xenova/nllb-200-distilled-600M`（ONNX int8）、`facebook/nllb-200-distilled-600M`（词表） | **CC-BY-NC-4.0（禁止商用）** | 可选句级翻译后端；**不随包分发**，词表 `tokenizer.bin` 需用 `tools/build_nllb_tokenizer.py` 自行编译 |

## 其它

- 本项目开发过程中使用了 AI 辅助工具（DeepSeek Harness）。**本项目与 DeepSeek / DeepSeek Harness 官方无隶属关系**，也不构成任何背书。
- 社区/镜像站（如 `hf-mirror.com`）只是下载通道，其可用性与内容由该站点负责。
- 若你是某个组件的权利人且认为此处署名或使用方式有问题，请开 issue，我会调整或移除。
