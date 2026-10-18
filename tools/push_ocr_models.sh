#!/usr/bin/env bash
# 把 PP-OCRv5 mobile 的 det/rec 推进 App 私有目录（重装/清数据后要重跑一次）。
#
# 为什么不用 `adb shell "run-as ... cat > file" < 本地文件`：
#   那样在 git-bash 下会把二进制截断（实测 4.7MB 变 699B）。
# 正确顺序：push 到 /data/local/tmp → run-as cp 到 files/models/ocr/。
set -euo pipefail
export MSYS_NO_PATHCONV=1

ADB=${ADB:-adb}
PKG=${PKG:-com.screenast.app.debug}
SRC=${SRC:-$(dirname "$0")/../模型/随包/ocr}
TMP=/data/local/tmp/ocr

[ -f "$SRC/ppocrv5-mobile-det.onnx" ] || { echo "缺 $SRC/ppocrv5-mobile-det.onnx"; exit 1; }
[ -f "$SRC/ppocrv5-mobile-rec.onnx" ] || { echo "缺 $SRC/ppocrv5-mobile-rec.onnx"; exit 1; }

"$ADB" shell mkdir -p "$TMP"
"$ADB" push "$SRC/ppocrv5-mobile-det.onnx" "$TMP/det.onnx"
"$ADB" push "$SRC/ppocrv5-mobile-rec.onnx" "$TMP/rec.onnx"

"$ADB" shell "run-as $PKG mkdir -p files/models/ocr"
"$ADB" shell "run-as $PKG cp $TMP/det.onnx files/models/ocr/ppocrv5_mobile_det.onnx"
"$ADB" shell "run-as $PKG cp $TMP/rec.onnx files/models/ocr/ppocrv5_mobile_rec.onnx"

echo "--- 结果 ---"
"$ADB" shell "run-as $PKG ls -l files/models/ocr"
echo "期望：det 4748769 字节 / rec 16517247 字节"
