# third_party

只放**第三方原生依赖的二进制包**，源码与模型都不在这里（模型见 `模型/`）。

```
third_party/sherpa-onnx/
├── sherpa-onnx-1.13.8.aar        打了 soname 补丁的版本（构建真正用的那个）
└── sherpa-onnx-1.13.8.orig.aar   补丁前备份
```

两个 AAR 都在 `.gitignore` 里（各 48MB），克隆后需要自己放进来。

## 为什么需要补丁

`sherpa-onnx` 内嵌的 `libonnxruntime.so` 与本项目另外引入的
`com.microsoft.onnxruntime:onnxruntime-android:1.31.0` 同名。两份库同时装进 APK 时，
动态链接器只会加载其中一份：sherpa 的 JNI 要 `OrtGetApiBase@VERS_1.28.2`，
官方包要 `@VERS_1.31.0`，加载错的那份直接让 App 启动即崩。

补丁把 **sherpa 那份的 soname 改成 `libonnxruntime_sherpa.so`**，两套库就能共存
（sherpa 的 `libsherpa-onnx-jni.so` 依赖记录随之指向新名字）。

复现方式（拿到原始 AAR 后跑一次）：

```bash
python tools/patch_sherpa_soname.py third_party/sherpa-onnx/sherpa-onnx-1.13.8.aar
```

脚本会就地替换 AAR 里的 `libonnxruntime.so`（用 NDK 的 `llvm-objcopy` 改 DT_SONAME，
用 `llvm-readelf` 校验），并把原文件备份成 `*.orig.aar`。

## 获取 AAR

上游发布页：<https://github.com/k2-fsa/sherpa-onnx/releases>
文件名形如 `sherpa-onnx-1.13.8.aar`。下载后放到 `third_party/sherpa-onnx/`，
按上面一行命令打补丁即可（不打补丁也能编，但运行时会崩，见 `app/build.gradle.kts` 里的说明）。
