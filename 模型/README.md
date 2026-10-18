# 模型仓库

这个文件夹放**全部 AI 模型**，不进 git（体积 11GB，`.gitignore` 已忽略）。分三类：

```
模型/
├── 随包/        ← 随 App 发布的部署集（3.9GB）。发版时把这一整个文件夹发给用户
│   ├── asr/        语音识别：SenseVoice / Silero VAD / GTCRN / Paraformer / Whisper / 标点 / TEN-VAD
│   ├── llama/      端侧大模型 gguf（默认 Qwen3.5-2B abliterated Q4_K_M，1.21GB）
│   ├── nllb/       NLLB-200 的三个 onnx（备选翻译后端）
│   ├── ocr/        PP-OCRv5 mobile（默认）+ server（可选高精度）的 det/rec
│   ├── inpaint/    MI-GAN 补背景（28MB，可选；擦除档要用它）
│   └── README.md   面向最终用户的文件夹说明
├── 上游/        ← 从上游下下来的原始包，含 LICENSE/README/测试音频，只为了可追溯
│   ├── asr/        sherpa-onnx 的 SenseVoice 原始目录、streaming-paraformer-zh-en、
│   │               paraformer-zh、whisper-small、punct-ct-transformer、ten-vad、gtcrn
│   └── nllb/       facebook/nllb 的原始导出（含未量化 onnx、tokenizer.json、config）
└── 备选/        ← 备用大模型（更省内存的 0.8B）
    ├── Huihui-Qwen3.5-0.8B-abliterated.Q4_K_M.gguf   528MB
    └── Huihui-Qwen3.5-0.8B-abliterated.Q8_0.gguf     812MB
```

## 随包/ 怎么到手机上

`随包/` 就是给最终用户的那份 `Models/`。装法三种（详见根目录 `模型导入与故障排查.txt`）：

1. 拷到手机 `/sdcard/Models` → App 设置页「离线模型」→「从 Models 导入」（应用内自动探测这个路径）；
2. 放在手机任意位置 → 设置页「选 Models 文件夹」，走系统文件选择器；
3. `adb push 模型/随包 /sdcard/Models` 再按 1 操作。

导入后文件进应用私有目录 `files/models/`，卸载才清。

## 为什么分 随包 / 上游 / 备选

- **随包** = 已经是 App 直接读取的目录结构（`files/models/<子目录>/<文件名>`），文件名是**写死**的，改名会导致「找不到模型」；
- **上游** = 保留原始目录名（例如 `sherpa-onnx-sense-voice-zh-en-ja-ko-yue-int8-2024-07-17/`），方便日后核对版本、重新导出、看 LICENSE；
- **备选** = 换模型/省内存时用，不入部署集。

`随包/asr/model.int8.onnx` 就是 `上游/asr/sherpa-onnx-sense-voice-.../model.int8.onnx` 的副本（一份平铺、一份带原始目录名），二者内容一致。

## 许可（重要）

| 模型 | 许可 | 备注 |
|---|---|---|
| SenseVoice | MIT/Apache-2.0（见上游目录 LICENSE） | 识别默认后端 |
| Silero VAD / GTCRN | MIT | VAD、降噪 |
| Paraformer / Whisper-small | Apache-2.0 / MIT | 可选识别后端 |
| CT-Transformer 标点 | Apache-2.0 | 标点 |
| PP-OCRv5 | Apache-2.0 | OCR |
| MI-GAN | MIT（权重按其发布页） | 擦除补背景 |
| **NLLB-200** | **CC-BY-NC-4.0** | **禁止商用**，商用请换掉 |
| Qwen3.5（含 Huihui 微调） | 按上游 Qwen 许可 | 端侧翻译主力 |

模型权重不随 git 分发；发布二进制包时请一并附上各自的 LICENSE。

## 校验

App 内「离线模型」卡片按**文件大小（2% 容差）**判是否就位；权威清单在
`app/src/main/java/com/screenast/app/core/ModelImporter.kt` 的 `ModelImporter.items`。
