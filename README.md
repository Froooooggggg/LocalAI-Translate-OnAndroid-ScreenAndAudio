# 同传互译（Screen & Audio Translate）

**Android 端侧、离线、不上传**的实时同传与屏幕翻译工具。两条主线：

- **同传**：抓到手机正在播放的声音 → 断句 → 语音识别 → 翻译 → 悬浮字幕实时显示。
- **识图**：点一下屏幕上的悬浮键 → 截屏 → OCR → 翻译 → 把译文**贴回原文所在位置**（可选 MI-GAN 擦除补背景），同时在悬浮窗里列出全部原文/译文。

全程本地推理：录音、截屏、译文都不出设备。唯一的例外是你在设置里**显式**把翻译后端切成「网络 API 调用」并填了在线服务地址。

> 版本 `0.7.2-beta`；个人项目，仍在迭代。`minSdk 29`（Android 10）、`targetSdk 36`、只出 `arm64-v8a`。

> 仓库名：`LocalAI-Translate-Android-Screen-And-Audio`（旧名 `...OnAndriod...` 是拼写笔误）。

## 非商业使用声明

**本项目是非商业项目**：仅供个人学习、研究与自用，源码按 [Apache-2.0](LICENSE) 开源，不收费、无广告、不收集数据，也不提供商业授权或售后承诺。

- 可选的 **NLLB-200 是 CC-BY-NC-4.0（禁止商用）**，只以「可选下载」提供，**不随 APK 分发**（连词表也不打进去了）；用它即表示你遵守其非商业条款。
- 怎么用软件（录音、截屏、翻译、转载译文）由**使用者本人**负责：请遵守当地法律与平台规则，注意他人隐私权、肖像权、著作权。
- 译文著作权归原文权利人，本软件只做技术处理，不对译文主张权利。
- 第三方组件与模型许可见 [THIRD_PARTY.md](THIRD_PARTY.md)。


---

## 功能一览

| 模块 | 能力 |
|---|---|
| 同传 | 屏幕内录（`AudioPlaybackCapture`）、VAD 断句、SenseVoice / Paraformer / Whisper 识别、标点模型、上下文多轮翻译、可拖动/可缩放的悬浮字幕窗 |
| 识图 | 手动单次抓屏（`MediaProjection` + `ImageReader`）、PP-OCRv5 检测+识别（mobile / server 两档）、多块批量翻译、**原位贴片**（覆盖式）与**擦除补背景**（MI-GAN 修补）两种呈现 |
| 悬浮窗 | 悬浮「暂停键」触发出图、结果贴片层（点一下收掉）、常驻文字列表窗（带历史、可折叠成小球） |
| 翻译后端 | 端侧大模型（Qwen 系，llama.cpp，可选 Hexagon NPU）、NLLB-200（句级）、OpenAI 兼容的在线服务（可选） |
| 其它 | 全部设置落 DataStore；模型可在应用内从 `Models/` 一次性导入（不联网下载） |

---

## 推荐配置

想跑顺，先看机器。端侧跑 2B 大模型 + 实时语音识别 + OCR，吃的是**内存**和**内存带宽**，
不是跑分——所以下面的门槛主要看运行内存和 SoC 档次。

| 项目 | 最低 | 推荐 |
|---|---|---|
| 运行内存 | **12 GB** | **16 GB**（开发者测试机就是 16 GB） |
| SoC | 天玑 8300 / 骁龙 7+ Gen3 同级 | **骁龙 8 Elite（8E）**、天玑 9400 同级 |
| 系统 | Android 12（`minSdk 29`），实测 Android 16 / 澎湃 OS 3 | 同上，越新越好 |
| 架构 | `arm64-v8a` | 带 NPU 的骁龙（Hexagon）可把大模型放到 NPU 上跑 |
| 存储 | 8 GB 空闲（必备模型约 1.53 GB，全套 3.93 GB） | 12 GB 以上空闲 |
| 屏幕 | 任意分辨率（识别按实际帧算） | 1440p 屏，小字识别更容易 |

几点说明：

- **12 GB 是硬门槛**：本地大模型（2B Q4_K_M）加载后常驻约 1.2 GB，加上 ASR、OCR、Android 自身，
  12 GB 机器属于「能跑，别同时开太多后台」。软件启动前会自己算一次可用内存，不够会弹警告。
- **推荐配置一定跑得动**：骁龙 8E / 天玑 9400 + 16 GB 是开发者实际使用的配置，
  2B 模型走 NPU 时翻译基本是秒回；天玑平台走 CPU 也能用，只是每句多几百毫秒。
- 低配机想省内存：翻译后端选 **NLLB-200**（约 1.27 GB 常驻，比大模型轻），
  识图权重选 **轻量 mobile**（21 MB，高精度 server 是 164 MB 且更慢）。
- 没有 NPU 也能用：把「大模型算在哪」设成 `CPU` 或留 `自动`，纯 CPU 跑 2B 也是可用的水平。

---

## 构建

环境（实测可用）：

| 项 | 版本 |
|---|---|
| JDK | **17**（AGP 9 不接受 21 及以下以外的组合，务必显式指定） |
| Android SDK | `compileSdk 37`、`build-tools 37` |
| NDK | `28.2.13676358` |
| Gradle | 随 `gradlew`（wrapper） |

```bash
# 1) 本地配置
#    复制一份 local.properties 并填自己的 SDK 路径
echo "sdk.dir=/path/to/Android/Sdk" > local.properties
#    gradle.properties 里有 org.gradle.java.home 一行，改成自己的 JDK 17 路径（或删掉、改用 JAVA_HOME）

# 2) 第三方原生库
#    sherpa-onnx 的 AAR 需要放在 third_party/sherpa-onnx/（体积大，未入库）：
#    https://github.com/k2-fsa/sherpa-onnx/releases  （下载 sherpa-onnx-1.13.8 AAR）
#    本项目对它的 soname 做过改动，复现方法见 tools/patch_sherpa_soname.py

# 3) 编译
./gradlew :app:assembleDebug
#  产物：app/build/outputs/apk/debug/app-debug.apk
```

> 只出了 `arm64-v8a`：llama.cpp 预编译库只有 arm64，且四 ABI 全带会把 APK 撑到 300MB+。
> 要跑模拟器请自行补对应 ABI 的 llama.cpp 库并改 `abiFilters`。

---

## 模型（不进 git，单独发）

模型共 22 个文件、约 **3.93GB**（其中同传+识图真正必需的只有 5 个、1.53GB；Silero VAD 与 GTCRN 降噪 1.1MB 已内置进 APK），不放仓库（`.gitignore` 已忽略）。发布时随包提供一个 `Models/` 文件夹（或压成一个 zip）：

```
Models/
├── asr/      SenseVoice + Silero VAD + GTCRN 降噪 + Paraformer/Whisper/标点/TEN-VAD
├── llama/    端侧大模型 gguf（默认 Qwen3.5-2B Q4_K_M）
├── nllb/     NLLB-200 三个 onnx（备选翻译后端）
├── ocr/      PP-OCRv5 mobile + server（det/rec）
└── inpaint/  MI-GAN（APK assets 里已有一份，放这里只保持一致）
```

- 仓库内布局见 [`模型/README.md`](模型/README.md)（`随包/` 是随 App 发的部署集、`上游/` 是原始下载包、`备选/` 是备用的 0.8B 大模型）。
- 给最终用户的导入步骤与**不兼容排查**见 [`模型导入与故障排查.txt`](模型导入与故障排查.txt)。
- 应用内入口：同传页右上角**设置 → 离线模型**。支持系统文件选择器（SAF，零权限）或 `/sdcard/Models` 自动探测。

---

## 目录结构

```
app/                    Android 应用（Kotlin + Jetpack Compose）
  src/main/java/com/screenast/app/
    asr/                语音识别后端（SenseVoice / Paraformer / Whisper / VAD / 标点 / 降噪）
    audio/              音频电平、重采样、WAV
    capture/            屏幕内录服务（AudioPlaybackCapture + 前台服务）
    core/               设置（DataStore）、语言、可见性、模型导入
    interpret/          同传引擎与会话
    overlay/            悬浮字幕窗、无障碍服务
    translate/          翻译后端：llama / nllb / OpenAI 兼容
    vision/             识图：抓屏、OCR、贴片、擦除补背景、悬浮键与列表窗
  src/main/cpp/         llama.cpp JNI
docs/                   设计文档
tools/                  开发脚本（推模型、探针、AAR 补丁等）
third_party/sherpa-onnx sherpa-onnx AAR（未入库，自行下载）
模型/                    模型仓库（未入库）
测试数据/                测试截图/日志/录音/开发日志（未入库）
模型导入与故障排查.txt    面向最终用户的技术文档
```

---

## 权限与使用

- **同传**：录音权限 + 屏幕录制（`MediaProjection`，用于捕获播放声音）+ 通知权限（前台服务）。
  一个 `MediaProjection` 只能建一个虚拟屏 —— 同传与识图抓屏互斥，同时开会被拒。
- **识图**：屏幕录制 + 「显示在其他应用上层」（悬浮键与贴片层）。
- 首次使用：设置 → 离线模型 → 导入 `Models/`，再回主界面开始。

---

## 隐私

- 默认全部端侧推理：不入网、不上传。
- 只有显式开启「网络API调用」并配置了服务地址/密钥时，译文会发往该服务（屏幕内容与录音仍只在本机）。
- 所有数据在应用私有目录，卸载即清空。

---

## 已知限制

- 只支持 `arm64-v8a`、Android 10+。
- 同传与识图共用一次投屏授权，不能并行抓屏。
- server 版 OCR 精度更高但慢 3~5 倍（整屏一帧约 3~5 秒）。
- 端侧大模型 2B 在 8GB 内存机型上跑得很紧，和 server OCR 同开可能被系统杀。
- 识图目前是「点一次识别一帧」，不是持续跟踪。

---

## 第三方组件与许可

| 组件 | 用途 | 许可 |
|---|---|---|
| [llama.cpp](https://github.com/ggml-org/llama.cpp) | 端侧大模型推理 | MIT |
| [sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx) | 语音识别运行时 | Apache-2.0 |
| ONNX Runtime | OCR / NLLB / MI-GAN 推理 | MIT |
| [PP-OCRv5](https://github.com/PaddlePaddle/PaddleOCR) | 文字检测与识别 | Apache-2.0 |
| [MI-GAN](https://github.com/Picsart-AI-Research/MI-GAN) | 擦除补背景 | MIT（权重见上游说明） |
| [Silero VAD](https://github.com/snakers4/silero-vad) | 语音活动检测 | MIT |
| [NLLB-200](https://huggingface.co/facebook/nllb-200-distilled-600M) | 备选翻译 | **CC-BY-NC-4.0（禁止商用）** |
| Qwen 系列权重 | 端侧大模型 | 按其上游许可 |

模型权重不随仓库分发，各模型的许可请以其发布页为准。**NLLB 为研究/非商用许可，商用场景请换掉它。**

代码本身的许可证待作者补充（建议 MIT）。在补充前，本仓库不构成任何授权。

---

## 致谢

- [AI 语音与 NLP 社区](https://github.com/k2-fsa) 开源的 sherpa-onnx 与一组优秀中文模型；
- llama.cpp 与 Hexagon 后端让 2B 模型在手机上可用；
- PaddleOCR / MI-GAN / NLLB 的开放权重。
