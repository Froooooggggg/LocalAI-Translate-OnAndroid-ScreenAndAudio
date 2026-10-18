#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
穷举 decoder_start_token 候选 × 是否 forced_bos，找出 NLLB 唯一正确的首步组合。

NLLB 的 decoder_start_token_id 在不同资料里说法不一（0=<s> / 1=<pad> / 2=</s>），
与其猜，不如把三个都跑一遍，看第 1 步能不能吐出**语义正确**的正文。
判据：en->zh 时，正确组合的首个正文 token 应是常见中文 token（如 '▁这' / '的' / '是'），
而不是 </s>、<unk>、或英文词片。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort

sys.path.insert(0, str(Path(__file__).parent))
from verify_nllb_standalone import Tokenizer, TOKENIZER_BIN, FLORES  # noqa: E402
# 仓库根（tools/ 的上一级）：脚本可以放在任何机器上跑
_R = Path(__file__).resolve().parents[1]


ROOT = Path(str(_R))
ONNX = ROOT / "third_party" / "nllb" / "onnx"

SENT = "Hello everyone, welcome back to the channel."
SRC, TGT = "en", "zh"

tok = Tokenizer(TOKENIZER_BIN)
enc = ort.InferenceSession(str(ONNX / "encoder_model_int8.onnx"), providers=["CPUExecutionProvider"])
on = ort.InferenceSession(str(ONNX / "decoder_model_quantized.onnx"), providers=["CPUExecutionProvider"])
past = ort.InferenceSession(str(ONNX / "decoder_with_past_model_quantized.onnx"),
                            providers=["CPUExecutionProvider"])
on_outs = [o.name for o in on.get_outputs()]
past_ins = [i.name for i in past.get_inputs()]
past_outs = [o.name for o in past.get_outputs()]

src_flores, _ = FLORES[SRC]
_, tgt_id = FLORES[TGT]
ids = tok.encode(SENT, src_flores)
hidden = enc.run(None, {
    "input_ids": np.array([ids], dtype=np.int64),
    "attention_mask": np.ones([1, len(ids)], dtype=np.int64),
})[0]
S = hidden.shape[1]
enc_mask = np.ones([1, S], dtype=np.int64)
print(f"输入 {SENT!r}")
print(f"编码 ids={ids}  S={S}  <s>={tok.bos} <pad>={tok.pad} </s>={tok.eos} <unk>={tok.unk}")
print(f"目标 {FLORES[TGT][0]} = {tgt_id}\n")


def topk(logits, k=6):
    idx = np.argsort(-logits[0, -1, :])[:k]
    return "  ".join(f"{int(i)}:{tok.vocab[int(i)]!r}" for i in idx)


def trial(start_id, force_bos, steps=4):
    out = on.run(None, {
        "input_ids": np.array([[start_id]], dtype=np.int64),
        "encoder_hidden_states": hidden,
        "encoder_attention_mask": enc_mask,
    })
    res = dict(zip(on_outs, out))
    enc_cache = {k: v for k, v in res.items() if ".encoder." in k}
    dec_cache = {k: v for k, v in res.items() if ".decoder." in k}

    if force_bos:
        # 首步 logits 作废，直接以目标语种 token 作为第一个「已生成」token
        nxt = tgt_id
        first_choice = f"forced_bos={tgt_id}"
    else:
        nxt = int(np.argmax(res["logits"][0, -1, :]))
        first_choice = f"argmax={nxt}({tok.vocab[nxt]!r})"

    seq = [nxt]
    for _ in range(steps - 1):
        if nxt == tok.eos:
            break
        feed = {"input_ids": np.array([[nxt]], dtype=np.int64), "encoder_attention_mask": enc_mask}
        for k, v in enc_cache.items():
            feed["past_key_values." + k[len("present."):]] = v
        for k, v in dec_cache.items():
            feed["past_key_values." + k[len("present."):]] = v
        o = past.run(None, feed)
        r = dict(zip(past_outs, o))
        nxt = int(np.argmax(r["logits"][0, -1, :]))
        dec_cache = {k: v for k, v in r.items() if k.startswith("present.")}
        seq.append(nxt)

    return {
        "start": start_id,
        "force_bos": force_bos,
        "first_choice": first_choice,
        "unconstrained_top": topk(res["logits"]),
        "seq": seq,
        "text": tok.decode(seq),
    }


print("=" * 96)
print(f"{'起始 token':<22}{'forced_bos':<12}{'第0步 top6（未约束）'}")
print("=" * 96)
rows = []
for start_id, name in ((tok.bos, "<s>=0"), (tok.pad, "<pad>=1"), (tok.eos, "</s>=2")):
    for fb in (False, True):
        r = trial(start_id, fb)
        rows.append((name, fb, r))
        print(f"{name:<22}{str(fb):<12}{r['unconstrained_top'][:70]}")
        print(f"{'':<34}→ 首生成 {r['first_choice']}")
        print(f"{'':<34}→ 序列 {r['seq']}")
        print(f"{'':<34}→ 解码 {r['text']!r}")
        print()

print("=" * 96)
print("判据：en->zh 时首个正文 token 应是常见中文 token，而非 </s>/<unk>/英文词片")
print("=" * 96)
for name, fb, r in rows:
    text = r["text"]
    verdict = "❌ 退化/无效"
    if text and text not in ("zho_Hans",) and "is is" not in text and "</s>" not in text:
        verdict = "✅ 有正文"
    print(f"  {name:<10} forced_bos={str(fb):<5} {verdict:<12} {text!r}")
