#!/usr/bin/env python3
"""数一条句子会被切成多少个 token（含 `eng_Latn` 前缀与 EOS）。

存在的理由很具体：真机上出现过「长句子只翻译了前半截」，怀疑撞到了
`NllbTranslator.translate()` 的 `maxNewTokens`（默认 96）。这个脚本用
`tools/reference_tokenizer_bin.py` 里那份**已被真机验证与 Kotlin 实现逐 id 一致**
的参考分词器来数输入侧 token 数，从而判断 96 到底够不够。

用法:
    python tools/count_tokens.py                        # 用下面的长句样例
    python tools/count_tokens.py "任意句子" ...
"""
import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))

from reference_tokenizer_bin import Tok  # noqa: E402

BIN = os.path.join(ROOT, "app", "src", "main", "assets", "nllb", "tokenizer.bin")

DEFAULT_SENTENCES = [
    "Hello everyone, welcome back to the channel.",
    # 用户 2026-10-15 提供的长难句，真机上只翻出了前半截
    "Furthering that point, the association noted a number of somber research findings, "
    "including that men are four times more likely than women to die of suicide worldwide, "
    'are "far more likely" than women to be arrested and charged with intimate partner '
    "violence in the U.S. and commit about 90 percent of all homicides nationwide.",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sentences", nargs="*", default=None)
    ap.add_argument("--max-new-tokens", type=int, default=96,
                    help="对照 NllbTranslator.translate() 的 maxNewTokens 默认值")
    args = ap.parse_args()

    t = Tok(BIN)
    sentences = args.sentences or DEFAULT_SENTENCES

    for s in sentences:
        plain = t.encode_plain(s)
        full = t.encode(s, "eng_Latn")
        print(f"文本      : {s[:70]}{'…' if len(s) > 70 else ''}")
        print(f"  字符数  : {len(s)}")
        print(f"  正文 token: {len(plain)}   （+ 语种前缀 + EOS = {len(full)}）")
        print(f"  maxNewTokens={args.max_new_tokens} → "
              f"{'够' if len(plain) < args.max_new_tokens else '★ 不够，会被截断'}")

        # 中文译文通常比英文长（中文 token 更碎），给个粗略预估
        print(f"  参考：中文译文 token 数常为英文正文的 1.2~1.6 倍 → "
              f"预计需要 {int(len(plain) * 1.2)}~{int(len(plain) * 1.6)} 个")
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
