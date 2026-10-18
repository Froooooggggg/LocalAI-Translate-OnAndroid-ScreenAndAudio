#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
验证 merged decoder 的正确调用范式（transformers.js 式）。

之前 tools/diag_decoder_step0.py 的结论是错的，原因有两个：
  1. 喂的起始 token 是字面量 2（= <pad>），不是目标语种 token（如 eng_Latn=256047）
  2. 首步 dummy 前向的判断标准没对齐 transformers.js

transformers.js 对 encoder-decoder 的 merged 图是这样用的：
  - 首步：decoder 侧 past 长度给 1（占位），use_cache_branch=True，
          步长 1 输入（就是 target BOS）；这一步的输出**全部丢弃**，
          只用它产出的 present.*.encoder.*（跨注意力 KV，只算一次）。
  - 后续：use_cache_branch=True，输入上一 token，past 用上一步的 present。
"""
from __future__ import annotations

import sys
import time
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


def build_past(names, enc_len, dec_len):
    """按输入名后缀决定形状：.encoder. 侧用 enc_len，.decoder. 侧用 dec_len。"""
    past = {}
    for n in names:
        if ".encoder." in n:
            past[n] = np.zeros([1, 16, enc_len, 64], dtype=np.float32)
        else:
            past[n] = np.zeros([1, 16, dec_len, 64], dtype=np.float32)
    return past


def main():
    tok = Tokenizer(TOKENIZER_BIN)
    src_flores, _ = FLORES[SRC]
    _, tgt_id = FLORES[TGT]

    enc = ort.InferenceSession(str(ONNX / "encoder_model_int8.onnx"), providers=["CPUExecutionProvider"])
    dec = ort.InferenceSession(
        str(ONNX / "decoder_model_merged_quantized.onnx"), providers=["CPUExecutionProvider"]
    )

    ins = [i.name for i in dec.get_inputs()]
    outs = [o.name for o in dec.get_outputs()]
    past_names = [n for n in ins if n.startswith("past_key_values.")]
    print(f"merged decoder: 输入 {len(ins)} 个（past_* {len(past_names)}），输出 {len(outs)} 个")
    print(f"  起始 token = {tgt_id} ({FLORES[TGT][0]})   # 之前错误地用了字面量 2=<pad>")
    print()

    ids = tok.encode(SENT, src_flores)
    enc_out = enc.run(None, {
        "input_ids": np.array([ids], dtype=np.int64),
        "attention_mask": np.ones([1, len(ids)], dtype=np.int64),
    })
    hidden = enc_out[0]
    S = hidden.shape[1]
    enc_mask = np.ones([1, S], dtype=np.int64)
    print(f"编码完成 S={S}，token ids={ids}")

    # ---------- 首步 dummy 前向：decoder past 长度 1，输出丢弃 ----------
    t0 = time.perf_counter()
    past = build_past(past_names, S, 1)
    past["use_cache_branch"] = np.array([True], dtype=np.bool_)
    feed = {
        "input_ids": np.array([[tgt_id]], dtype=np.int64),
        "encoder_hidden_states": hidden,
        "encoder_attention_mask": enc_mask,
    }
    feed.update(past)
    try:
        out = dec.run(None, feed)
    except Exception as e:
        print(f"\n❌ 首步 dummy 失败: {type(e).__name__}")
        print(f"   {str(e)[:600]}")
        return
    print(f"✅ 首步 dummy 成功（{(time.perf_counter()-t0)*1000:.0f} ms），输出 {len(out)} 个")

    res = dict(zip(outs, out))
    logits = res["logits"]
    print(f"   logits 形状 {logits.shape}")
    top = np.argsort(-logits[0, -1, :])[:8]
    print(f"   首步 top8: " + ", ".join(f"{int(i)}:{tok.vocab[int(i)]!r}" for i in top))

    # ---------- 后续步骤：带缓存单步 ----------
    past = {n.replace("past_key_values.", "present."): v
            for n, v in res.items() if n.startswith("present.")}
    past = {k.replace("present.", "past_key_values."): v for k, v in past.items()}
    past["use_cache_branch"] = np.array([True], dtype=np.bool_)

    nxt = int(np.argmax(logits[0, -1, :]))
    generated = []
    t0 = time.perf_counter()
    for step in range(40):
        if nxt == tok.eos:
            print(f"   第 {step} 步命中 EOS，停止")
            break
        generated.append(nxt)
        feed = {
            "input_ids": np.array([[nxt]], dtype=np.int64),
            "encoder_hidden_states": hidden,
            "encoder_attention_mask": enc_mask,
        }
        feed.update(past)
        try:
            out = dec.run(None, feed)
        except Exception as e:
            print(f"   ❌ 第 {step} 步失败: {str(e)[:400]}")
            break
        res = dict(zip(outs, out))
        nxt = int(np.argmax(res["logits"][0, -1, :]))
        past = {k.replace("present.", "past_key_values."): v
                for k, v in res.items() if k.startswith("present.")}
        past["use_cache_branch"] = np.array([True], dtype=np.bool_)

    dt = (time.perf_counter() - t0) * 1000
    print(f"\n生成 {len(generated)} 个 token，用时 {dt:.0f} ms "
          f"（{dt/max(1,len(generated)):.0f} ms/token）")
    print(f"译文: {tok.decode(generated)}")
    print(f"原始 ids: {generated[:40]}")


if __name__ == "__main__":
    main()
