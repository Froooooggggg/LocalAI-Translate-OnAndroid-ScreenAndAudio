#!/usr/bin/env bash
# 真机压测 2B GGUF 翻译（llama.cpp），走 LlamaDebugActivity 的 auto=batch。
#
# 为什么用 batch 而不是每条一次 am start：
#   llama 模型加载要 3.9 秒，`am force-stop` 之后每次都要重新读 1.2GB 的盘。
#   一个 15 条的回归会白等一分钟。batch 模式一次加载、连着跑完。
#
# ⚠️ 句子**不能含单引号**：下面的文本是靠单引号包成一个参数递给 `adb shell` 的。
#    含单引号会被 shell 截断，症状是原文莫名少了一截（不是模型的问题）。
#
# ⚠️ 每条前必须 `am force-stop`：activity 已在前台时新 intent 走 onNewIntent
#    不会重新执行，logcat 里留下的是上一批的旧日志——会得出完全错误的结论。
#
# 用法：
#   tools/device_llama.sh                      # 跑内置样例
#   tools/device_llama.sh "Hello." "Hi there." # 跑指定句子
set -u

ADB="${ADB:-adb}"
PKG=com.screenast.app.debug
ACT="$PKG/com.screenast.app.LlamaDebugActivity"

# 默认样例：故意混入已知的薄弱点（俚语、习语、长句、退化输入）
if [ $# -gt 0 ]; then
    SENTENCES=("$@")
else
    SENTENCES=(
        "The quick brown fox jumps over the lazy dog."
        "What the hell are you doing right now"
        "Damn it, I forgot my keys again"
        "This is a bunch of bullshit and you know it"
        "Break a leg out there tonight"
        "He spent the whole night in a strip club"
        "They got drunk and made out in the car"
        "The report concluded after a six-month investigation that the company had ignored at least four separate safety warnings."
    )
fi

# 用 ||| 连接（adb --es 传含换行的字符串会被吃掉）
PAYLOAD=""
for s in "${SENTENCES[@]}"; do
    case "$s" in
        *"'"*) echo "❌ 句子含单引号，会破坏 adb 传参：$s" >&2; exit 2 ;;
    esac
    if [ -z "$PAYLOAD" ]; then PAYLOAD="$s"; else PAYLOAD="$PAYLOAD|||$s"; fi
done

echo "=== 共 ${#SENTENCES[@]} 条 ==="
"$ADB" shell "am force-stop $PKG; logcat -c"
"$ADB" shell "am start -n $ACT --es auto batch --es text '$PAYLOAD'" >/dev/null 2>&1

# 等待：加载 3.9s + 每条约 1~3s，给足余量
WAIT=$(( 15 + ${#SENTENCES[@]} * 4 ))
echo "=== 等待 ${WAIT}s ==="
sleep "$WAIT"

echo
echo "=== 结果 ==="
"$ADB" logcat -d -s LlamaDebug:* 2>&1 \
    | sed 's/.*LlamaDebug: //' \
    | grep -vE '^$' \
    | tail -n $(( ${#SENTENCES[@]} * 3 + 12 ))
