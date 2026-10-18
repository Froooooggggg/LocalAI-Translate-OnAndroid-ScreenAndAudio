#!/usr/bin/env bash
# 把 SenseVoice + silero VAD 送进应用的**私有目录**。
#
# 为什么不直接 adb push 到 /data/local/tmp 就完了：
#   /data/local/tmp 是 shell 用户的临时目录，正式安装的设备上根本不存在，
#   只能当开发期临时落点。私有目录才是模型该待的地方。
#
# 为什么必须走「两跳」（adb push → run-as cat）：
#   见 docs/断点续接单.md §4.6 —— Android 15+ 收紧了 /sdcard/Android/data 的访问，
#   应用自己都读不了；而 `adb push` 又直接写不进 /data/user/0/（SELinux）。
#   所以：先推到 shell 可写的 /data/local/tmp，再用 run-as 以应用身份 cat 进去。
#
# 注意：`adb exec-out "run-as ..."` 会**挂死**（run-as 需要 PTY），必须用 shell 重定向。
set -euo pipefail

# ⚠️ 必须关掉 MSYS 的路径改写。
# 在 MSYS/git-bash 下运行 adb 时，`/data/local/tmp/x` 会被当成 Unix 路径自动翻译成
# `C:/Users/.../data/local/tmp/x`，于是 adb 在**本机**找目录：
#     adb: error: failed to copy '...model.int8.onnx' to
#          'C:/Users/Admin/.../data/local/tmp/snat-asr-stage/model.int8.onnx':
#          remote secure_mkdirs() failed: No such file or directory
# 报错里那个 `C:/...` 前缀就是证据。设备端路径一律不要改写。
export MSYS_NO_PATHCONV=1
export MSYS2_ARG_CONV_EXCL='*'

# 关掉路径改写之后，adb 也就看不懂 `D:/...` 之外的形式了，所以宿主端路径
# 必须自己转成 Windows 形式（cygpath -w）再交给 adb。
# 注意：转的是**宿主端**路径；设备端路径（/data/local/tmp/...）保持原样。
winpath() {
  if command -v cygpath >/dev/null 2>&1; then cygpath -w "$1"; else printf '%s' "$1"; fi
}

ADB="${ADB:-adb}"
SERIAL="${SERIAL:-$(adb devices | awk 'NR==2{print $1}')}"
PKG="${PKG:-com.screenast.app.debug}"

HOST_DIR="$(cd "$(dirname "$0")/.." && pwd)/模型/上游/asr"
SENSE_DIR="$HOST_DIR/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-int8-2024-07-17"
STAGE="/data/local/tmp/snat-asr-stage"
DEST="files/models/asr"

SRC_MODEL="$(winpath "$SENSE_DIR/model.int8.onnx")"
SRC_TOKENS="$(winpath "$SENSE_DIR/tokens.txt")"
SRC_VAD="$(winpath "$HOST_DIR/silero_vad.onnx")"

echo "== 1/4 推送到 shell 临时区 =="
"$ADB" -s "$SERIAL" shell "rm -rf $STAGE && mkdir -p $STAGE"
time "$ADB" -s "$SERIAL" push "$SRC_MODEL"  "$STAGE/model.int8.onnx"
"$ADB" -s "$SERIAL" push "$SRC_TOKENS" "$STAGE/tokens.txt"
"$ADB" -s "$SERIAL" push "$SRC_VAD"    "$STAGE/silero_vad.onnx"

echo "== 2/4 以应用身份落到私有目录 =="
"$ADB" -s "$SERIAL" shell "run-as $PKG mkdir -p $DEST"
for f in model.int8.onnx tokens.txt silero_vad.onnx; do
  echo "  -> $f"
  time "$ADB" -s "$SERIAL" shell "run-as $PKG sh -c 'cat $STAGE/$f > $DEST/$f'"
done

echo "== 3/4 校验（大小必须一致） =="
"$ADB" -s "$SERIAL" shell "run-as $PKG ls -l $DEST"
echo "-- 宿主端大小 --"
ls -l "$SENSE_DIR/model.int8.onnx" "$SENSE_DIR/tokens.txt" "$HOST_DIR/silero_vad.onnx"
echo "== 4/4 清理临时区 =="
"$ADB" -s "$SERIAL" shell "rm -rf $STAGE"
echo "完成。"
