#!/usr/bin/env bash
# 把同一句话连翻 N 次，看崩溃是"第一次就来"还是"翻几次之后才来"。
#
# 为什么要问这个：真机上 NLLB 崩在 `OrtSession.run` 的 JNI 里，JNIEnv 指针是垃圾，
# 崩溃线程是 ORT 的线程池 worker。最大的假设是 **JNI 局部引用表溢出**：
# ORT 的 JNI 每跑一次 run() 会为每个输入/输出张量建局部引用 —— 我们一次
# `decPastSession.run()` 是 48 个输入 + 25 个输出 ≈ 73 个，再加喂名字时的 50 次
# GetStringUTFChars。Android 局部引用表默认只有 512 槽，溢出时**不抛异常而是静默
# 踩内存**。
#
#   第一次就崩   -> 是"单次 run 的规模"问题（引用数本身超限）
#   第 N 次才崩  -> 是"跨多次 run 累积"问题（引用/句柄泄漏）→ 坐实泄漏假设
#
# 用法: bash tools/repro_repeat.sh [次数]
set -u

ADB="${ADB:-adb}"
DEV="${SERIAL:-$(adb devices | awk 'NR==2{print $1}')}"
PKG="com.screenast.app.debug"
N="${1:-5}"

export MSYS_NO_PATHCONV=1

echo "### 冷启动，连续翻译 $N 次"
"$ADB" -s "$DEV" logcat -c
"$ADB" -s "$DEV" shell am force-stop "$PKG"
sleep 2
"$ADB" -s "$DEV" shell am start -n "$PKG/com.screenast.app.NllbDebugActivity" \
  --es auto "repeat$N" \
  --es src en --es dst zh >/dev/null

# 每次翻译：加载约 4s + 首步约 2s + 每 token 约 0.62s。10 个 token 约 10s，留足余量
for i in $(seq 1 "$N"); do
  sleep 22
  alive="$("$ADB" -s "$DEV" shell pidof "$PKG" | tr -d '\r')"
  echo "  第 $i 轮后: pid=${alive:-<已崩溃>}"
  [ -z "$alive" ] && break
done

echo
echo "### 译文与崩溃信号"
"$ADB" -s "$DEV" logcat -d -s NllbDebug NllbTranslator libc 2>&1 |
  grep -E 'repeat|第 ?[0-9]+ 次|译文|Fatal|signal|step#1:|step#2:' | head -30
