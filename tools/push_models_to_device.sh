#!/usr/bin/env bash
# 把 1.27GB NLLB + 228MB SenseVoice 搬进真机的**应用私有目录**。
#
# 为什么不直接 adb push 到 /storage/emulated/0/Android/data/<pkg>/files/：
#   真机（Android 16 / 澎湃 OS3）上，那个路径**连应用自己都读不了**。
#   实测 `run-as com.screenast.app.debug ls -l .../files/` → Permission denied。
#   文件推过去字节数完全正确，但应用侧 stat 不到，NllbModels.resolve() 返回 null。
#   这是 Android 15+ 对 /sdcard/Android/data 的收紧，与 MIUI 无关。
#
# 所以改走两跳：
#   adb push  → /data/local/tmp/snat/   （shell 可写，push 速度 45~85 MB/s）
#   run-as cat → files/models/...        （应用私有目录，一定可读）
# 跨命名空间的读取由 shell 完成，应用只需要读自己的家目录。
#
# 用法: bash tools/push_models_to_device.sh [设备序列号]
set -u

ADB="${ADB:-adb}"
DEV="${1:-$(adb devices | awk 'NR==2{print $1}')}"
PKG="com.screenast.app.debug"
export MSYS_NO_PATHCONV=1

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# adb.exe 是原生 Windows 程序，看不懂 MSYS 的 /d/... 路径，必须转成 D:/...
if command -v cygpath >/dev/null 2>&1; then ROOT="$(cygpath -m "$ROOT")"; fi
BASE="$ROOT/third_party"

STAGE="/data/local/tmp/snat"
NLLB_SRC="$BASE/nllb/onnx"
ASR_SRC="$BASE/asr/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-int8-2024-07-17"

sh_dev() { "$ADB" -s "$DEV" shell "$@" 2>&1 | tr -d '\r'; }

echo "### 准备暂存区 $STAGE"
sh_dev "mkdir -p $STAGE"

# 每个文件：push 到暂存区 → run-as cat 进私有目录 → 校验 → 删暂存
move_one() {
  local src="$1" destrel="$2" name
  name="$(basename "$src")"
  if [ ! -f "$src" ]; then echo "  !! 源缺失: $src"; return 1; fi

  local want got
  want="$(stat -c%s "$src")"
  printf "  %-42s %10s B  " "$name" "$want"

  # 已存在且字节数一致就跳过（脚本可重入）
  got="$(sh_dev "run-as $PKG stat -c%s $destrel 2>/dev/null || echo 0")"
  if [ "$got" = "$want" ]; then echo "已就位，跳过"; return 0; fi

  local t0=$SECONDS
  "$ADB" -s "$DEV" push "$src" "$STAGE/$name" >/dev/null 2>&1 || { echo "push 失败"; return 1; }
  sh_dev "run-as $PKG mkdir -p \$(dirname $destrel)"
  sh_dev "run-as $PKG sh -c 'cat $STAGE/$name > $destrel'"
  sh_dev "rm -f $STAGE/$name"

  got="$(sh_dev "run-as $PKG stat -c%s $destrel 2>/dev/null || echo 0")"
  if [ "$got" = "$want" ]; then
    echo "✓ $((SECONDS - t0))s"
  else
    echo "✗ 字节数不符 got=$got"; return 1
  fi
}

echo ""
echo "### NLLB（1.27GB）"
move_one "$NLLB_SRC/encoder_model_int8.onnx"               "files/models/nllb/onnx/encoder_model_int8.onnx"
move_one "$NLLB_SRC/decoder_model_quantized.onnx"          "files/models/nllb/onnx/decoder_model_quantized.onnx"
move_one "$NLLB_SRC/decoder_with_past_model_quantized.onnx" "files/models/nllb/onnx/decoder_with_past_model_quantized.onnx"

echo ""
echo "### ASR"
move_one "$ASR_SRC/model.int8.onnx"  "files/models/asr/model.int8.onnx"
move_one "$ASR_SRC/tokens.txt"       "files/models/asr/tokens.txt"
move_one "$BASE/asr/silero_vad.onnx" "files/models/asr/silero_vad.onnx"

echo ""
echo "### 应用私有目录最终状态"
sh_dev "run-as $PKG ls -l files/models/nllb/onnx files/models/asr"

echo ""
echo "### 清掉之前误推到外部目录的死数据（应用读不到，白占 1.5GB）"
sh_dev "rm -rf /storage/emulated/0/Android/data/$PKG/files/models"
echo "  已清理"
