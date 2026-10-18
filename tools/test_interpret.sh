#!/usr/bin/env bash
# 实时同传的端到端自测：注入一段**有标准答案**的英文语音，走完整条
#   闸门 → VAD → SenseVoice → InterpretEngine → NLLB → 字幕
# 链路，然后把「原文」和「译文」都打出来。
#
# 为什么不用真采集做验收：采集要人工点授权，而且**没有标准答案**——
# 听到什么算什么，无法判定对错。这个脚本用的是 Windows TTS 从已知文本合成的音频。
#
# 用法：tools/test_interpret.sh [pcm路径]
#   默认 /data/local/tmp/asr_test.pcm（由 tools/make_asr_test_wav.ps1 生成）

set -euo pipefail

# ⚠️ MSYS 路径改写：Git Bash 会把 `/data/local/tmp/x` 这类**看起来像 Unix 路径**的
# 参数翻译成 Windows 路径（`C:/Users/.../data/local/tmp/x`），于是设备端收到一个
# 根本不存在的路径，而 adb 本身不报错。
# 这个坑本项目在 push_asr_models.sh 里踩过一次，这里是第二次 —— 凡是传给 adb 的
# 设备端绝对路径，都必须在这两个环境变量下执行。
export MSYS_NO_PATHCONV=1
export MSYS2_ARG_CONV_EXCL='*'

ADB="${ADB:-adb}"
PKG="com.screenast.app.debug"
ACT="$PKG/com.screenast.app.InterpretActivity"
PCM="${1:-/data/local/tmp/asr_test.pcm}"

"$ADB" shell am force-stop "$PKG"
"$ADB" logcat -c

echo "=== 启动 auto=wavtest，注入 $PCM，目标语言=中文 ==="
"$ADB" shell am start -n "$ACT" \
    --es auto wavtest \
    --es pcm "$PCM" \
    --ez dummy true >/dev/null

echo "=== 等待链路跑完（最长 75 秒）==="
for i in $(seq 1 25); do
    sleep 3
    if "$ADB" logcat -d -s AST.Interp:I 2>/dev/null | grep -q "自检结束"; then
        echo "--- 第 $((i*3)) 秒：已出现「自检结束」 ---"
        break
    fi
    echo "    ...$((i*3))s"
done

echo
echo "########## 识别与翻译 ##########"
"$ADB" logcat -d -s AST.Interp:I | sed 's/^.*: //'
echo
echo "########## 错误/崩溃（若有）##########"
"$ADB" logcat -d -s AST.Interp:E AndroidRuntime:E DEBUG:E libc:F 2>/dev/null | tail -20
