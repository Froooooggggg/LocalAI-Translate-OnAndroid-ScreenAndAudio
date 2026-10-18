# -*- coding: utf-8 -*-
"""EOS 救回判据的**风险扫描**：找出「救了反而更糟」的输入。

## 为什么需要这个脚本

`NllbTranslator.pickToken` 的救回判据只看一个数：argmax 撞上 EOS 时 EOS 的
softmax 概率是否低于 `eosGate`（默认 0.60）。这个判据在长句上实测有效，
但它**分不清两种「EOS 概率不高」**：

  A. 模型还想接着说，只是从句边界让它勉强收尾   → 该救（p(EOS) 通常 0.4~0.55）
  B. 模型根本没什么可说，输出很短就该停          → 不该救（p(EOS) 可能低到 0.05）

真机上有一次把输入截成单个 "I"，EOS 概率只有 0.053 却照样触发救回，
结果把「我」续写成「我是个子.」。那次的输入是 adb 引号事故造成的，
但它暴露的问题是真的：短输入 / 无意义输入可能被救成废话。

## 这个脚本做什么

对一批**故意挑得很难看**的输入跑贪心解码，把**每一次 EOS 事件**的
概率、当时的第二名、以及救回之后写出什么，全部打出来。
用这些数据决定要不要给判据加一个**下界**（低于它就认为模型只是没话说，放行 EOS）。

## 判读方法

- 如果一个输入本来翻得挺完整，却因为低概率 EOS 被救回、续出废话 → 需要下界。
- 如果所有「该救」的样本 p(EOS) 都明显高于所有「不该救」的样本 → 下界有安全区间。
- 如果两者重叠 → 单看概率这条路走不通，得换判据（比如看第二名的语义）。
"""

import sys
from pathlib import Path

ROOT = Path(str(_R))
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "测试数据" / "probe"))

import numpy as np  # noqa: E402
from verify_nllb_standalone import Tokenizer  # noqa: E402
from probe_long_eos import Eng, ZH, EOS  # noqa: E402
# 仓库根（tools/ 的上一级）：脚本可以放在任何机器上跑
_R = Path(__file__).resolve().parents[1]


tok = Tokenizer(ROOT / "app" / "src" / "main" / "assets" / "nllb" / "tokenizer.bin")
eng = Eng()


def softmax_p(logits, idx):
    m = logits.max()
    e = np.exp(logits - m)
    return float(e[idx] / e.sum())


def scan(text, max_new=96, penalty=1.1):
    """跑一遍贪心解码，**不救回**，只把每次 EOS 事件记录下来。"""
    cache, mask, _ = eng._start(text, tok)
    last, seq, events = ZH, [], []
    n = 0
    while len(seq) < max_new and n < 400:
        n += 1
        if not (seq and seq[-1] == last):
            seq.append(last)
        logits = eng._step(last, cache, mask, penalty, seq)
        pick = int(np.argmax(logits))
        if pick == EOS:
            p = softmax_p(logits, EOS)
            # 第二名是谁（把 EOS 压掉再看一次 argmax）
            saved = logits[EOS]
            logits[EOS] = -1e30
            alt = int(np.argmax(logits))
            alt_p = softmax_p(logits, alt)
            logits[EOS] = saved
            events.append({
                "step": len(seq),
                "p_eos": p,
                "alt": tok.vocab[alt],
                "alt_p": alt_p,
            })
            break
        last = pick
    return tok.decode(seq), seq, events


CASES = [
    # ---- 极短 / 无意义：救回在这里最可能帮倒忙 ----
    ("极短-单代词", "I"),
    ("极短-单词", "Yes."),
    ("极短-道谢", "Thanks."),
    ("极短-问候", "Hello."),
    ("极短-中文", "好"),
    ("极短-语气词", "Okay."),
    # ---- 正常短句：不该触发救回 ----
    ("短句-样例", "Hello everyone, welcome back to the channel."),
    ("短句-中文", "这个功能完全在手机上运行。"),
    # ---- 中等：边界情况 ----
    ("中句-22词", "The association points out some alarming findings, including that men are "
                  "four times more likely than women to die by suicide worldwide."),
    # ---- 长句：救回的正经用武之地 ----
    ("长句-39词带逗号", "I think the most important thing that we need to understand about this whole "
                    "situation is that nobody really knows what is going to happen next, and that "
                    "is exactly why we should all be a little more careful."),
    ("长句-39词无标点", "I think the most important thing that we need to understand about this whole "
                    "situation is that nobody really knows what is going to happen next and that "
                    "is exactly why we should all be a little more careful"),
]

print(f"{'样例':<18} {'步数':>4} {'p(EOS)':>8} {'第二名':<12} {'p(2nd)':>8}  译文")
print("-" * 130)
rows = []
for name, text in CASES:
    out, seq, ev = scan(text)
    if ev:
        e = ev[0]
        rows.append((name, e["p_eos"], e["alt"], e["alt_p"], len(seq)))
        print(f"{name:<18} {e['step']:>4} {e['p_eos']:>8.4f} {e['alt']:<12} {e['alt_p']:>8.4f}  {out[:60]}")
    else:
        print(f"{name:<18} {'-':>4} {'未触发':>8} {'-':<12} {'-':>8}  {out[:60]}")

print()
print("=" * 130)
print("按 p(EOS) 排序（判读：低的一头是不是都「不该救」）")
print("=" * 130)
for name, p, alt, altp, ln in sorted(rows, key=lambda r: r[1]):
    flag = "← 低于 0.30，救回风险高" if p < 0.30 else ""
    print(f"  p(EOS)={p:.4f}  {name:<18} 第二名={alt!r:<12} p2={altp:.4f}  {flag}")
