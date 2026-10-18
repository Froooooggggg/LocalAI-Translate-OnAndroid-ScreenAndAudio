#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
打印一批样例的【参考 token id】与【解码往返结果】，供 Android 侧 NllbTokenizer 自检对照。

为什么需要它：
  `tools/tokenizer_bin_ids.json` 只能证明"词表读对了"，证明不了"编码/解码算对了"。
  而 `<unk>` 这个 bug 正是出在 **decode 的字节还原**上 —— PC 侧 Python 从不复现，
  因为 HF 的 tokenizers 把所有 token 拼成一个 bytes 再一次性 UTF-8 解码。
  Android 侧旧实现逐个 token 解码，半个汉字就变成 U+FFFD。
  所以这里输出的参考值必须包含：
    1. encode(text) 的 id 序列 —— 比对 BPE 实现
    2. decode(那些 id) 的文本    —— 比对字节还原（这是本次 bug 的主战场）

用法：
    python tools/dump_reference_tokenids.py [--json out.json]

依赖：transformers（HF tokenizers）。模型目录用 模型/上游/nllb/tokenizer.json，
      不需要联网、不需要 ONNX。
"""
import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOK_JSON = os.path.join(ROOT, "third_party", "nllb", "tokenizer.json")

# 覆盖点：ASCII、单字词、跨 token 的半字符（就是出 <unk> 的那类）、标点、空格、
# 多语种混排、以及真机上实际翻过的那句。
CASES = [
    "Hello everyone, welcome back to the channel.",
    "您好，欢迎回到频道。",
    "频道",
    "这个功能完全在手机上运行。",
    "今天天气真好。",
    "こんにちは、世界。",
    "안녕하세요 세계",
    "This phone is very durable and can last for two days.",
    "It's a 10% increase — roughly.",
    "NLLB-200 支持 200 种语言。",
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default=os.path.join(ROOT, "tools", "reference_tokenids.json"))
    args = ap.parse_args()

    if not os.path.isfile(TOK_JSON):
        print(f"找不到 {TOK_JSON}", file=sys.stderr)
        return 2

    from tokenizers import Tokenizer  # noqa: E402

    tok = Tokenizer.from_file(TOK_JSON)
    eos_id = tok.token_to_id("</s>")

    out = {"eosId": eos_id, "cases": []}
    for text in CASES:
        ids = tok.encode(text, add_special_tokens=False).ids
        # ⚠️ 用 decode 时要显式跳过 special token，否则 HF 会把 </s> 之类渲染成字符串。
        back = tok.decode(ids, skip_special_tokens=True)
        # 逐 token 的原始字节，用来在出错时肉眼定位是哪个 token 跨了字符边界
        pieces = [tok.id_to_token(i) for i in ids]
        out["cases"].append(
            {"text": text, "ids": ids, "back": back, "pieces": pieces,
             "roundTripOk": back == text}
        )

    with open(args.json, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)

    print(f"eosId={eos_id}  写出 {args.json}")
    for c in out["cases"]:
        flag = "OK " if c["roundTripOk"] else "!! "
        print(f"{flag}{c['text']!r}")
        print(f"     ids   = {c['ids']}")
        print(f"     back  = {c['back']!r}")
        if not c["roundTripOk"]:
            print(f"     pieces= {c['pieces']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
