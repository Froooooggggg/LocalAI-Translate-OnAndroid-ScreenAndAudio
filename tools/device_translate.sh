#!/usr/bin/env bash
# 真机译文抽检：绕过 ASR，直接把文本喂给 NllbDebugActivity 的 translate 入口。
#
# 为什么要有这个脚本：PC 侧的 probe_translation_quality.py 用的是**另一份**分词器实现
# （tools/verify_nllb_standalone.py），它在两点上与 App 不一致：
#   1. 不会过滤语种标签（所以译文里会多出 `zho_Hans`）；
#   2. decode 是逐 token 转字符串，跨 token 的半个汉字会显示成 `<unk>`
#      （App 侧是「先拼字节、最后统一 UTF-8 解码」）。
# 所以判「质量好不好」必须看真机输出，PC 只用来做**对照实验**（同一输入下变一个变量）。
#
# 用法：bash tools/device_translate.sh
#
# ⚠️ 传给 adb 的设备端参数必须走单引号 + 整个 am 命令作为**一个**参数，
#    否则 adb 会把后面的词当成新的 shell 参数（见 docs 断点续接单 §4.7）。

set -u
export MSYS_NO_PATHCONV=1
export MSYS2_ARG_CONV_EXCL='*'

ADB=${ADB:-adb}
PKG=com.screenast.app.debug
ACT=$PKG/com.screenast.app.NllbDebugActivity

# 一条一句；用 | 分隔原文与「期望译文」（期望留空表示只看不判）
CASES=(
  'The lazy dog sleeps all afternoon.|（查 <unk>：应为「懒狗整个下午都在睡觉。」）'
  'He asked me whether I was sure, and I said no.|（前一轮 PC 侧到这里就断了，「我说不」整句丢失）'
  'I think the most important thing that we need to understand about this whole situation is that nobody really knows what is going to happen next, and that is exactly why we should all be a little more careful.|（长句，看末尾「因此我们都应该更小心」有没有出来）'
  'The report, which was published last week by a group of researchers who have been studying the region for over a decade, suggests that the situation is getting worse.|（长依赖：主句谓语「表明情况在恶化」有没有出来）'
)

for c in "${CASES[@]}"; do
  text=${c%%|*}
  note=${c##*|}
  "$ADB" shell am force-stop $PKG >/dev/null 2>&1
  "$ADB" logcat -c >/dev/null 2>&1
  "$ADB" shell "am start -n $ACT --es auto load+tr --es text '$text'" >/dev/null 2>&1
  # 等译文出现（最多 60 秒：要先加载 1.27GB 模型约 4 秒，长句本身还要 2~3 秒）
  out=""
  for _ in $(seq 1 60); do
    sleep 1
    out=$("$ADB" logcat -d -s NllbDebug:I 2>/dev/null | grep -oE '翻译完成.*' | tail -1)
    [ -n "$out" ] && break
  done
  echo "──────────────────────────────────────────────"
  echo "原文：$text"
  echo "$out"
  echo "备注：$note"
  # 顺带把 EOS 救回日志打出来（判「是救不动还是没救」）
  "$ADB" logcat -d -s NllbDebug:I 2>/dev/null | grep -E '提前收尾' | sed 's/^.*: //'
done
echo "──────────────────────────────────────────────"
