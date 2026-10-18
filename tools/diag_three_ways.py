#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
三路对照：找出 merged decoder 在 ONNX Runtime 1.31 上可用的调用方式。

背景：同一个图、同一个 Reshape 节点
  '/model/decoder/layers.0/encoder_attn/Reshape_4'
走 use_cache_branch=False 正常，走 True 就崩。这更像 ORT 图优化把该节点的目标
形状算错了，而不是模型本身坏。所以对照三路：

  A. 全程 use_cache_branch=False + **每步重喂完整前缀**（语义正确但慢；past 给占位）
  B. 缓存路径 + 关闭图优化（ORT_DISABLE_ALL）
  C. 缓存路径 + 只开基础优化（ORT_ENABLE_BASIC）
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
MERGED = ONNX / "decoder_model_merged_quantized.onnx"

SENT = "Hello everyone, welcome back to the channel."
SRC, TGT = "en", "zh"
MAX_STEPS = 14


def make_past(names, enc_len, dec_len):
    out = {}
    for n in names:
        out[n] = np.zeros([1, 16, enc_len if ".encoder." in n else dec_len, 64], dtype=np.float32)
    return out


def make_past_nocache(names, S):
    """
    无缓存分支（use_cache_branch=False）下 past 的正确填法。

    ⚠️ 曾经写反：给 **encoder 侧** 一个长度 S 的**全零**张量，模型会真的拿它做跨注意力
    （对全零 key/value 做 softmax），输出必然退化成「一个 token 卡死」。
    正确做法是 encoder 侧给**零长度**，让 ONNX 的 Reshape 从实际 concat 结果反推形状；
    decoder 侧同样给零长度。
    """
    return {n: np.zeros([1, 16, 0, 64], dtype=np.float32) for n in names}


def top5(tok, logits):
    idx = np.argsort(-logits[0, -1, :])[:5]
    return "  ".join(f"{int(i)}:{tok.vocab[int(i)]!r}" for i in idx)


def run_variant(tag, opt_level, use_cache, tok, enc, hidden, enc_mask, tgt_id):
    print(f"\n{'='*74}")
    print(f"变体 {tag}  graph_optimization={opt_level}  use_cache_branch={use_cache}")
    print('='*74)

    so = ort.SessionOptions()
    so.graph_optimization_level = opt_level
    so.intra_op_num_threads = 4
    t0 = time.perf_counter()
    dec = ort.InferenceSession(str(MERGED), so, providers=["CPUExecutionProvider"])
    print(f"  加载 {(time.perf_counter()-t0)*1000:.0f} ms")

    ins = [i.name for i in dec.get_inputs()]
    outs = [o.name for o in dec.get_outputs()]
    past_names = [n for n in ins if n.startswith("past_key_values.")]
    S = hidden.shape[1]

    def feed(prefix, past):
        f = {
            "input_ids": np.array([prefix], dtype=np.int64),
            "encoder_hidden_states": hidden,
            "encoder_attention_mask": enc_mask,
            "use_cache_branch": np.array([use_cache], dtype=np.bool_),
        }
        f.update(past)
        return f

    # 首步
    if use_cache:
        past = make_past(past_names, S, 1)
    else:
        past = make_past_nocache(past_names, S)
    prefix = [tgt_id]

    try:
        out = dec.run(None, feed(prefix, past))
    except Exception as e:
        print(f"  ❌ 首步失败: {str(e)[:180]}")
        return None
    res = dict(zip(outs, out))
    print(f"  首步 OK  top5: {top5(tok, res['logits'])}")

    if use_cache:
        past = {k.replace("present.", "past_key_values."): v for k, v in res.items()
                if k.startswith("present.")}
        past["use_cache_branch"] = np.array([True], dtype=np.bool_)

    nxt = int(np.argmax(res["logits"][0, -1, :]))
    generated = []
    t0 = time.perf_counter()
    for step in range(MAX_STEPS):
        if nxt == tok.eos:
            break
        generated.append(nxt)
        if use_cache:
            cur_prefix = [nxt]
        else:
            cur_prefix = [tgt_id] + generated       # 每步重喂完整前缀
        try:
            out = dec.run(None, feed(cur_prefix, past))
        except Exception as e:
            print(f"  ❌ 第 {step+1} 步失败: {str(e)[:180]}")
            break
        res = dict(zip(outs, out))
        if step < 4:
            print(f"    step{step+1} prefix_len={len(cur_prefix)} top5: {top5(tok, res['logits'])}")
        if use_cache:
            past = {k.replace("present.", "past_key_values."): v for k, v in res.items()
                    if k.startswith("present.")}
            past["use_cache_branch"] = np.array([True], dtype=np.bool_)
        nxt = int(np.argmax(res["logits"][0, -1, :]))

    dt = (time.perf_counter() - t0) * 1000
    text = tok.decode(generated)
    per = dt / max(1, len(generated))
    print(f"  → 译文: {text!r}")
    print(f"  → {len(generated)} tok / {dt:.0f} ms / {per:.0f} ms per tok")
    return {"tag": tag, "text": text, "ms_per_tok": per, "ok": True}


def main():
    tok = Tokenizer(TOKENIZER_BIN)
    src_flores, _ = FLORES[SRC]
    _, tgt_id = FLORES[TGT]

    enc = ort.InferenceSession(str(ONNX / "encoder_model_int8.onnx"), providers=["CPUExecutionProvider"])
    ids = tok.encode(SENT, src_flores)
    hidden = enc.run(None, {
        "input_ids": np.array([ids], dtype=np.int64),
        "attention_mask": np.ones([1, len(ids)], dtype=np.int64),
    })[0]
    enc_mask = np.ones([1, hidden.shape[1]], dtype=np.int64)
    print(f"输入: {SENT!r}")
    print(f"编码 S={hidden.shape[1]}  起始 token={tgt_id}({FLORES[TGT][0]})")

    results = []
    for tag, lvl, uc in (
        ("A 重喂完整前缀(无缓存)", ort.GraphOptimizationLevel.ORT_ENABLE_ALL, False),
        ("B 缓存+关优化",        ort.GraphOptimizationLevel.ORT_DISABLE_ALL, True),
        ("C 缓存+基础优化",      ort.GraphOptimizationLevel.ORT_ENABLE_BASIC, True),
    ):
        r = run_variant(tag, lvl, uc, tok, enc, hidden, enc_mask, tgt_id)
        if r:
            results.append(r)

    print(f"\n{'='*74}")
    print("汇总")
    print('='*74)
    for r in results:
        print(f"  {r['tag']:<26} {r['ms_per_tok']:6.0f} ms/tok   {r['text']}")


if __name__ == "__main__":
    main()
