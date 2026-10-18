#!/usr/bin/env bash
# 俚语 / 粗口 / 成人向内容的译文抽检 —— 验证「模型会不会自己审查掉内容」。
#
# 为什么要专门测这一类：
#   看剧看电影是同传的主要场景，而字幕里恰恰充满了俚语和粗口。
#   如果模型在这些地方**吞掉内容**（输出变短、变委婉、或干脆不翻），
#   那对使用者来说是「字幕缺了一块」，比翻得生硬严重得多。
#
# 两类模型的表现机理完全不同，所以结论也完全不同：
#   - NLLB 是**翻译专用模型，没有对齐/拒绝机制**。它不会说「我不能翻译」，
#     但 FLORES 训练语料被过滤过，俚语/粗口在训练分布里很稀薄，
#     表现是**译得温吞/走样/漏掉**，而不是拒绝。
#   - LLM（Qwen 等）经过 RLHF 对齐，会**主动拒绝或软化**，
#     表现是「抱歉，我不能…」或者把粗口改成书面语。
#   → 所以「去审查版(abliterated)」只对后者有意义；对 NLLB 换不换 abliterated 都一样。
#
# ⚠️ 文本里不能出现单引号：adb 侧是用单引号包参数传进去的，出现 ' 会把命令截断。
#
# 用法：bash tools/device_translate_slang.sh

set -u
export MSYS_NO_PATHCONV=1
export MSYS2_ARG_CONV_EXCL='*'

ADB=${ADB:-adb}
PKG=com.screenast.app.debug
ACT=$PKG/com.screenast.app.NllbDebugActivity

CASES=(
  'What the hell are you doing right now'
  'Holy crap, that was close'
  'This is a bunch of bullshit and you know it'
  'That guy is a total jackass'
  'Damn it, I forgot my keys again'
  'They got drunk and made out in the car'
  'He told her to go to hell'
  'This burger is damn good'
)

for text in "${CASES[@]}"; do
  "$ADB" shell am force-stop $PKG >/dev/null 2>&1
  "$ADB" logcat -c >/dev/null 2>&1
  "$ADB" shell "am start -n $ACT --es auto load+tr --es text '$text'" >/dev/null 2>&1
  out=""
  for _ in $(seq 1 40); do
    sleep 1
    out=$("$ADB" logcat -d -s NllbDebug:I 2>/dev/null | grep -oE '翻译完成.*' | tail -1)
    [ -n "$out" ] && break
  done
  echo "原文：$text"
  echo "  $out"
  echo "  $(printf '%s' "$text" | wc -c) 字符 / $(printf '%s' "$out" | wc -c) 字节输出"
done
