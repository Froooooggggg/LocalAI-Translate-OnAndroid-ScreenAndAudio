#!/usr/bin/env bash
# 真机 NLLB 译文抽检 —— 输出 TSV，供与 LLM 结果做同题对照。
#
# 关键点（都是踩过的坑）：
#   1) 每条前必须 `am force-stop`：activity 已在前台时新 intent 走 onNewIntent，
#      不会重新执行，logcat 里留下的是**上一批**的旧日志，会得出完全错误的结论。
#   2) 单引号包文本，整条 am 命令作为「一个」参数交给 adb shell；
#      分开传参会被 adb 拼回去、设备侧再按空格切碎（曾把长句截成单个 "I"）。
#      → 文本里不能出现单引号。
#   3) 用 `load+tr` 绕过 ASR 直接压翻译。auto 的合法值只有 toktest / load / load+tr / repeatN。
#   4) 判定「这一轮有没有出结果」用「日志里 UI: [ 出现的次数」而不是 tail -1：
#      force-stop 后旧日志可能还没滚出去，tail -1 会读到**上一轮**的译文，
#      或者因为缓冲区被系统压力日志冲掉而漏读。失败要重试并把现场留下来。
#
# 用法：bash tools/device_slang_tsv.sh > 测试数据/probe/nllb_slang.tsv 2> 测试数据/probe/nllb_slang.err

set -u
export MSYS_NO_PATHCONV=1
export MSYS2_ARG_CONV_EXCL='*'

ADB=${ADB:-adb}
PKG=com.screenast.app.debug
ACT=$PKG/com.screenast.app.NllbDebugActivity
DEADLINE=${DEADLINE:-60}   # 单次尝试最长等多久（秒）

CASES=(
  'The quick brown fox jumps over the lazy dog.'
  'The meeting has been postponed until next Tuesday.'
  'What the hell are you doing right now'
  'Holy crap, that was close'
  'That guy is a total jackass'
  'He told her to go to hell'
  'This burger is damn good'
  'This is a bunch of bullshit and you know it'
  'Damn it, I forgot my keys again'
  'They got drunk and made out in the car'
  'He spent the whole night in a strip club'
  'Break a leg out there tonight'
  'That is a piece of cake for him'
)

# 跑一次，成功时把 "毫秒\t译文" 打到 stdout；失败返回非 0。
attempt() {
  local text="$1"
  "$ADB" shell am force-stop $PKG >/dev/null 2>&1
  sleep 2
  "$ADB" logcat -c >/dev/null 2>&1
  sleep 1
  "$ADB" shell "am start -n $ACT --es auto load+tr --es text '$text'" >/dev/null 2>&1

  local line="" i
  for ((i = 0; i < DEADLINE; i++)); do
    sleep 1
    # -T 只取最近 200 行，避免读到上一轮残留
    line=$("$ADB" logcat -d -s NllbDebug:I 2>/dev/null \
             | grep -oE 'UI: \[[a-z]{2}→[a-z]{2}\] [0-9]+ms  .+' | tail -1)
    [ -n "$line" ] && break
  done

  if [ -z "$line" ]; then
    echo "### 失败：[$text] 等了 ${DEADLINE}s 没拿到译文行，现场日志：" >&2
    "$ADB" logcat -d -s NllbDebug:I 2>/dev/null | tail -12 >&2
    echo >&2
    return 1
  fi

  local ms zh
  ms=$(printf '%s' "$line" | grep -oE '[0-9]+ms' | head -1 | tr -d 'ms')
  zh=$(printf '%s' "$line" | sed -E 's/^UI: \[[a-z]{2}→[a-z]{2}\] [0-9]+ms  //')
  printf '%s\t%s\n' "$ms" "$zh"
  return 0
}

for text in "${CASES[@]}"; do
  if out=$(attempt "$text"); then
    printf '%s\t%s\n' "$text" "$out"
  elif out=$(attempt "$text"); then        # 重试一次
    ms=${out%%$'\t'*}
    printf '%s\t%s\n' "$text" "${out}"
    echo "### 重试成功：[$text]" >&2
  else
    printf '%s\tTIMEOUT\t\n' "$text"
  fi
done
