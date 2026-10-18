#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""译文质量体检：把当前端侧配置（NLLB-600M 蒸馏 + int8 + 贪心 + EOS 救回）
在一组**分门别类**的句子上跑一遍，用来回答「质量差在哪一类」。

## 为什么要分类，而不是笼统说「质量差」

字幕场景里的坏译文有好几种**完全不同的**成因，修法也完全不同：

  A. **词法/搭配**（"lazy dog" → "惰的狗"）—— 模型的词表与训练数据问题，
     换更大的 MT 模型能改善，加提示词没用。
  B. **标点/排版**（中文里混半角逗号）—— 纯后处理问题，一个映射表就能修，
     跟模型大小无关。**这是最容易被误判成「模型不行」的一类。**
  C. **截断/漏译**（只翻前半截）—— 解码策略问题（已用 EOS 救回缓解）。
  D. **语序/长距离依赖**（从句套从句）—— 真正需要更大模型或上下文的部分。
  E. **口语/习语**（"you know"、"kind of"）—— 小模型普遍差。

看这份输出时请**逐类判断**，不要整体打个「好/差」。

## 用法

    python tools/probe_translation_quality.py            # 跑全部
    python tools/probe_translation_quality.py --only D   # 只跑某一类

## 判读要点

- 标点问题（B）如果占比高，那么「换 LLM」是**治错了病**——先修后处理。
- 真正的词法与长距离问题（A/D/E）才是换模型能解决的。
"""

import argparse
import sys
from pathlib import Path

import numpy as np  # noqa: E402

ROOT = Path(str(_R))
sys.path.insert(0, str(ROOT / "tools"))

from verify_nllb_standalone import Tokenizer  # noqa: E402
from probe_long_eos import Eng, ZH, EOS  # noqa: E402
# 仓库根（tools/ 的上一级）：脚本可以放在任何机器上跑
_R = Path(__file__).resolve().parents[1]


# ---- 与 NllbTranslator.kt 保持一致 ----
EOS_GATE = 0.75
EOS_FLOOR = 0.30
MAX_RESCUE = 2

# 分类样例：同传字幕最常遇到的就是这些形态
CASES = [
    ("A", "词法/搭配", "The lazy dog sleeps all afternoon."),
    ("A", "词法/搭配", "She gave me a piece of advice."),
    ("A", "词法/搭配", "This is a game changer for the whole industry."),

    ("B", "标点/排版", "First, we need to check the numbers. Then, we can decide."),
    ("B", "标点/排版", "He asked: \"Are you sure?\" I said no."),

    ("C", "截断/漏译", "I think the most important thing that we need to understand about this whole "
                       "situation is that nobody really knows what is going to happen next, and that "
                       "is exactly why we should all be a little more careful."),

    ("D", "语序/长依赖", "The report, which was published last week by a group of researchers "
                         "who have been studying the region for over a decade, suggests that "
                         "the situation is getting worse."),

    ("E", "口语/习语", "So, you know, it's kind of a big deal, right?"),
    ("E", "口语/习语", "Honestly, I couldn't care less about it."),

    ("F", "正常基准", "Machine translation runs entirely on this phone."),
    ("F", "正常基准", "Welcome back to the channel, everyone."),
]


def softmax_p(logits, idx):
    m = logits.max()
    e = np.exp(logits - m)
    return float(e[idx] / e.sum())


def translate(eng, tok, text, max_new=96, penalty=1.1):
    """与 NllbTranslator.translate() 同构：贪心 + repeatPenalty + 双边 EOS 救回。"""
    cache, mask, _ = eng._start(text, tok)
    last, seq, rescued = ZH, [], []
    n = 0
    while len(seq) < max_new and n < 400:
        n += 1
        if not (seq and seq[-1] == last):
            seq.append(last)
        logits = eng._step(last, cache, mask, penalty, seq)
        pick = int(np.argmax(logits))
        if pick == EOS:
            p = softmax_p(logits, EOS)
            if len(rescued) < MAX_RESCUE and EOS_FLOOR <= p < EOS_GATE:
                rescued.append(round(p, 3))
                logits[EOS] = -1e30
                pick = int(np.argmax(logits))
                seq.append(EOS)          # 与 Kotlin 侧一致：把 EOS 占位补回序列
                last = pick
                continue
            break
        last = pick
    return tok.decode(seq), rescued


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="", help="只跑某一类，如 D")
    args = ap.parse_args()

    tok = Tokenizer(ROOT / "app" / "src" / "main" / "assets" / "nllb" / "tokenizer.bin")
    eng = Eng()

    print("=" * 78)
    print("端侧配置体检：NLLB-200-distilled-600M + int8 + 贪心 + EOS 救回")
    print("=" * 78)

    for tag, kind, src in CASES:
        if args.only and tag != args.only:
            continue
        out, rescued = translate(eng, tok, src)
        print(f"\n[{tag}] {kind}")
        print(f"  原文  {src}")
        print(f"  译文  {out}")
        extra = f"   （EOS 救回 {len(rescued)} 次，p={rescued}）" if rescued else ""
        print(f"  长度  {len(out)} 字{extra}")

    print("\n" + "=" * 78)
    print("逐类判读：B 类只需后处理；A/D/E 类才是换模型能治的；C 类已用 EOS 救回缓解。")


if __name__ == "__main__":
    main()
