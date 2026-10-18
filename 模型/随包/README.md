# 随包模型（发布时作为 `Models/` 发给用户）

这个 App 的四个能力都靠**端侧模型**跑，不需要联网、不把内容发出去。24 个文件共约 3.93GB（真正必需的只有 5 个、1.53GB；其中 Silero VAD 与 GTCRN 降噪已经内置进 APK，这个文件夹里留着只是备查），
塞不进 APK（Android 对应用包体积有限制，装不上），所以跟着发布包**单独发这一个文件夹**。

> 仓库里它叫 `模型/随包/`（仓库内所有模型统一放 `模型/`）；
> **发给用户时把这一层改名为 `Models/`** —— App 的自动探测找的就是 `/sdcard/Models`。
> 换了别的名字也能用：走「选 Models 文件夹」让用户手动选即可。

```
Models/                     ← 发布时用这个名字；就是你现在看的这个文件夹，整个拷到手机里
├── asr/                    语音识别（SenseVoice / Paraformer / Whisper / VAD / 标点）
├── llama/                  端侧大模型（Qwen3.5-2B，翻译主力）
├── nllb/                   备选翻译（NLLB-200，句级，快但没上下文）
├── ocr/                    识图 OCR（PP-OCRv5 mobile / server 两套）
└── inpaint/                擦除补背景（MI-GAN）
```

## 怎么装到手机上

**方式一（推荐，不用电脑）**：把整个 `Models` 文件夹拷到手机存储根目录
（`/sdcard/Models`），打开 App → 右下「识图」页或「设置」页 → 「离线模型」卡片 →
点「从 Models 导入」。App 会把文件搬进自己的私有目录，之后这个文件夹可以删掉。

**方式二**：拷到手机上任意位置（比如下载目录），点「选 Models 文件夹」，
在系统文件选择器里选中它即可 —— 走系统的目录授权，不需要任何存储权限。

**方式三（开发用）**：

```bash
adb push Models /sdcard/Models
# 或者逐个推（大文件更稳）
adb push Models/llama/Huihui-Qwen3.5-2B-abliterated.Q4_K_M.gguf /sdcard/Models/llama/
```

## 只要一部分可以吗

可以。设置页会显示「必备 x/7」「全部 x/24」，**缺哪个就退到不需要它的能力**：

| 想用的功能 | 最小必需 |
| --- | --- |
| 同传（语音→译文） | `asr/model.int8.onnx`、`asr/tokens.txt`、`asr/silero_vad.onnx`、`llama/*.gguf` |
| 识图（屏幕取字→译文） | `ocr/ppocrv5-mobile-det.onnx`、`ocr/ppocrv5-mobile-rec.onnx`、`llama/*.gguf` |
| 擦除补背景（MI-GAN） | `inpaint/migan_pipeline_v2.onnx`（可选；不装就自动退回「程序贴片」，即用框四周平均色盖住原文） |
| 高精度 OCR | 再加 `ocr/ppocrv5-server-*.onnx`（172MB，慢 3~5 倍） |
| 换翻译后端为 NLLB | `nllb/onnx/` 三个 onnx |

`inpaint/` 与 `nllb/tokenizer.bin` 在 APK 的 assets 里已经有一份，放着也不会被覆盖，
列在这里只是为了保持目录结构一致。

## 文件清单（期望大小）

| 文件 | 大小 |
| --- | --- |
| `llama/Huihui-Qwen3.5-2B-abliterated.Q4_K_M.gguf` | 1270809024 |
| `asr/model.int8.onnx` | 239233841 |
| `asr/silero_vad.onnx` | 643854 |
| `asr/tokens.txt` | 315894 |
| `asr/gtcrn/gtcrn_simple.onnx` | 535638 |
| `asr/paraformer-zh/model.int8.onnx` | 243371218 |
| `asr/paraformer-zh/tokens.txt` | 75756 |
| `asr/streaming-paraformer/encoder.int8.onnx` | 165462184 |
| `asr/streaming-paraformer/decoder.int8.onnx` | 71664561 |
| `asr/streaming-paraformer/tokens.txt` | 75756 |
| `asr/whisper-small/small-encoder.int8.onnx` | 112442483 |
| `asr/whisper-small/small-decoder.int8.onnx` | 262226114 |
| `asr/whisper-small/small-tokens.txt` | 816730 |
| `asr/punct-ct-transformer/model.onnx` | 294372519 |
| `asr/punct-ct-transformer/tokens.json` | 4207480 |
| `asr/ten-vad/ten-vad.onnx` | 315449 |
| `nllb/onnx/encoder_model_int8.onnx` | 415346576 |
| `nllb/onnx/decoder_model_quantized.onnx` | 470533055 |
| `nllb/onnx/decoder_with_past_model_quantized.onnx` | 445006987 |
| `ocr/ppocrv5-mobile-det.onnx` | 4748769 |
| `ocr/ppocrv5-mobile-rec.onnx` | 16517247 |
| `ocr/ppocrv5-server-det.onnx` | 87697340 |
| `ocr/ppocrv5-server-rec.onnx` | 84137438 |

清单的权威版本是代码里的 `app/src/main/java/com/screenast/app/core/ModelImporter.kt`
（`ModelImporter.items`）；两张表不一致时以代码为准，导入器按它判「已就位」。

## 为什么不打包进 APK

- Android 对 APK 内单文件/总体积有实际限制（Play 上 200MB，本地安装也受文件系统与
  解压耗时影响）；4GB 的包用户在安装阶段就会失败。
- 模型与代码分开更新：换模型不用重装 App，删掉 `files/models` 重新导入即可。

## 隐私

模型全部在手机本地推理。**只有**在设置里显式打开「网络API调用」并把密钥填成在线服务时，
译文才会发到那个服务；默认配置全程离线，屏幕内容/录音都不出设备。
