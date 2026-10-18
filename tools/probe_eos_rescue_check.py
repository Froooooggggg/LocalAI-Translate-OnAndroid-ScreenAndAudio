# -*- coding: utf-8 -*-
"""EOS 救回判据的**回归验证**：把 `NllbTranslator.pickToken` 的两边判据在 PC 上复现一遍。

## 为什么要单独有这个脚本

`tools/probe_eos_risk.py` 只**观察**（不救回，记录每次 EOS 的概率）；
这个脚本**执行**（按 `[eosFloor, eosGate)` 的规则真的去救），
回答的是「改完判据之后，那 11 个样例分别变成什么样」。

判据必须和 `NllbTranslator.kt` 保持一致：
    EOS_GATE_DEFAULT = 0.75f  /  EOS_GATE_FLOOR_DEFAULT = 0.30f  /  MAX_EOS_RESCUE_DEFAULT = 2
改那边就要改这里，否则这个回归就没有意义。

## 判读

- **正常样例**（"Yes."、"谢谢"、短句、中句）：译文必须**一字不变**——判据不该碰它们。
- **"I"**：必须**不救**（0.062 在下界之下），保持「我」。
- **两个长句**：必须**被救**并补出后半截。
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


# ---- 与 NllbTranslator.kt 的常量保持一致 ----
EOS_GATE = 0.75
EOS_FLOOR = 0.30
MAX_RESCUE = 2

tok = Tokenizer(ROOT / "app" / "src" / "main" / "assets" / "nllb" / "tokenizer.bin")
eng = Eng()


def softmax_p(logits, idx):
    m = logits.max()
    e = np.exp(logits - m)
    return float(e[idx] / e.sum())


def translate(text, max_new=96, penalty=1.1):
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
            # 双边判据：只有落在这条带子里才救
            if len(rescued) < MAX_RESCUE and EOS_FLOOR <= p < EOS_GATE:
                logits[EOS] = -1e30
                pick = int(np.argmax(logits))
                rescued.append((round(p, 4), tok.vocab[pick]))
            else:
                break
        last = pick
    return tok.decode(seq), seq, rescued


CASES = [
    ("极短-单代词", "I"),
    ("极短-单词", "Yes."),
    ("极短-道谢", "Thanks."),
    ("极短-问候", "Hello."),
    ("极短-中文", "好"),
    ("极短-语气词", "Okay."),
    ("短句-样例", "Hello everyone, welcome back to the channel."),
    ("短句-中文", "这个功能完全在手机上运行。"),
    ("中句-22词", "The association points out some alarming findings, including that men are "
                  "four times more likely than women to die by suicide worldwide."),
    ("长句-39词带逗号", "I think the most important thing that we need to understand about this whole "
                    "situation is that nobody really knows what is going to happen next, and that "
                    "is exactly why we should all be a little more careful."),
    ("长句-39词无标点", "I think the most important thing that we need to understand about this whole "
                    "situation is that nobody really knows what is going to happen next and that "
                    "is exactly why we should all be a little more careful"),
    ("超长-58词", "So what I want to talk about today is the fact that when you look at the data "
                  "from the last twenty years, you see something very strange happening, which is "
                  "that despite all of the technological progress that we have made, people report "
                  "feeling more isolated than ever before, and nobody has a good explanation for it."),
]

for name, text in CASES:
    out, seq, rescued = translate(text)
    tag = "救回 %d 次 %s" % (len(rescued), rescued) if rescued else "未触发"
    body = out.split(" ", 1)[1] if " " in out else out
    print(f"\n=== {name} | {len(seq)} tok | {tag} ===")
    print(f"  {body}")
