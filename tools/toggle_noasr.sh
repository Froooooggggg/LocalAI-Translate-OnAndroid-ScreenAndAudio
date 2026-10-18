#!/usr/bin/env bash
# 临时摘掉 / 还原 sherpa-onnx，用于隔离两份 onnxruntime。
#
# 背景：真机上 NLLB 的 `OrtSession.run` 稳定 SIGSEGV，崩溃点
#   Java_ai_onnxruntime_OrtSession_run+552  →  ldr x8,[x21,#0x48] ; blr x8
# 即 JNI 接口表指针本身是垃圾。而**同一版本 onnxruntime 1.31.0、同一批模型文件，
# 在 PC 上用 Python 跑得干干净净**（tools/repro_crash_on_pc.py：with-past 增量解码
# 9 步跑通，cache 48 条含 encoder 侧，译文「您好,欢迎回到频道.」完全正确）。
#
# 最大嫌疑：APK 里并存的两份 onnxruntime
#   libonnxruntime.so          官方 1.31.0（libonnxruntime4j_jni.so 依赖它）
#   libonnxruntime_sherpa.so   sherpa 的 1.28.2（soname 已改，见 patch_sherpa_soname.py）
# soname 不同才让它们能共存，但 JNI 导出符号是全局的：`libonnxruntime4j_jni.so`
# 解析 `OrtGetApiBase` 时可能落到 sherpa 那份 → Java 侧按 1.31 的布局持有
# `OrtSession*`，native 侧跑的却是 1.28 的代码。
#
# 为什么不写成 Gradle 的 -PnoAsr 开关：AGP 9 内置 Kotlin 的 source set DSL 三种写法
# 都试过，全部不接受（详见 app/build.gradle.kts 里 kotlin{} 上方的注释）。
# 所以改用最笨但绝对可靠的方式：挪文件 + 注释依赖行。
#
# 用法：
#   bash tools/toggle_noasr.sh on     # 摘掉 sherpa（ASR 相关源码移出源码树）
#   bash tools/toggle_noasr.sh off    # 还原
#   bash tools/toggle_noasr.sh status # 看当前处于哪一态
set -u

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SRC="$ROOT/app/src/main/java/com/screenast/app/asr/SenseVoiceAsr.kt"
STASH="$ROOT/.stash/SenseVoiceAsr.kt"
GRADLE="$ROOT/app/build.gradle.kts"
SHERPA_MARK='//NOASR// '
SHERPA_LINE='implementation(files("../third_party/sherpa-onnx/sherpa-onnx-1.13.8.aar"))'

# ⚠️ 这里踩过两个 sed 的坑，都是"看着能跑、其实没生效"的类型：
#
# 1) `\s` 在 BRE 里不是空白字符类，而是**字面 s**。行首是空格不是 s，所以匹配不上。
#    必须用 POSIX 字符类 `[[:space:]]`。
#
# 2) **BRE 的 pattern 里 `[` 是字符类的开始，不能像 ERE 那样用 `\[` 转义。**
#    原来用 `// [noasr] xxx` 当标记，还原时的 pattern 里那个 `[` 把整段解析成了
#    字符类，于是永远匹配不上 —— 注释时成功、还原时静默失败，
#    结果是构建里悄悄少了 sherpa（表现为一片 `Unresolved reference 'k2fsa'`）。
#    所以标记改成不含任何正则元字符的 `//NOASR// `。
#
# 两个函数都用 `#` 作 sed 分隔符，避免路径里的 `/` 需要转义。
comment_out() {
  sed -i "s#^\([[:space:]]*\)$SHERPA_LINE#\1$SHERPA_MARK$SHERPA_LINE#" "$GRADLE"
}
restore_line() {
  sed -i "s#^\([[:space:]]*\)$SHERPA_MARK$SHERPA_LINE#\1$SHERPA_LINE#" "$GRADLE"
}

mkdir -p "$(dirname "$STASH")"

case "${1:-status}" in
  on)
    if [ -f "$STASH" ]; then
      echo "已经处于 noasr 状态，无需重复操作"
      exit 0
    fi
    if [ ! -f "$SRC" ]; then
      echo "源文件不存在：$SRC" >&2
      exit 1
    fi
    mv "$SRC" "$STASH"
    comment_out
    echo "已摘掉 sherpa-onnx："
    echo "  $SRC  ->  $STASH"
    echo "  已注释 build.gradle.kts 里的 sherpa 依赖行"
    grep -n 'NOASR' "$GRADLE" | head -3
    ;;
  off)
    if [ -f "$STASH" ]; then
      mv "$STASH" "$SRC"
      echo "已还原 SenseVoiceAsr.kt"
    else
      echo "stash 里没有文件，源码树本来就是完整的"
    fi
    restore_line
    echo "已还原 build.gradle.kts 的 sherpa 依赖"
    grep -n 'sherpa-onnx-1.13.8.aar' "$GRADLE" | head -3
    ;;
  status)
    if [ -f "$STASH" ]; then
      echo "当前：noasr（sherpa 已摘掉，SenseVoiceAsr.kt 在 $STASH）"
    else
      echo "当前：完整（sherpa 依赖正常，SenseVoiceAsr.kt 在源码树里）"
    fi
    grep -n 'sherpa-onnx-1.13.8.aar' "$GRADLE" | head -3
    ;;
  *)
    echo "用法: $0 on|off|status" >&2
    exit 2
    ;;
esac
