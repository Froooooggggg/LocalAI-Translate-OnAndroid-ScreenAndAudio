#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NLLB 解码正确性对照实验：两段式 KV cache vs 单模型全前缀重放。

为什么做这个实验
----------------
Android 端要选解码方案，两者代价完全不同：
  A. 两段式（decoder_model 首步 + decoder_with_past_model 后续）
     - 每步只算 1 个 token，但需要传 48 个 past 张量（KV cache）
     - 需要同时加载两个 ONNX（470MB + 445MB = 915MB）
  B. 单模型全前缀重放（只用 decoder_model，每次喂完整已生成序列）
     - 只有一个 ONNX（470MB），无 past 张量管理，实现简单得多
     - 代价是第 t 步要算 t 个 token

在 GPU/PC 上 A 一定更快，但**手机是 CPU**，且 ONNX Runtime Android 的
张量拷贝有 JNI 开销 —— 48 个 past 张量每次都要跨 JNI 传递，很可能把
「少算几个 token」的优势吃回去。所以必须实测，不能想当然。

实验同时输出两种方案的**逐 token logits 是否一致** —— 如果不一致，
说明其中一种实现有 bug，这比性能更重要。
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
THREADS = int(sys.argv[sys.argv.index("--threads") + 1]) if "--threads" in sys.argv else 6

CASES = [
    ("en", "zh", "Hello everyone, welcome back to the channel."),
    ("en", "ja", "I think this is the best decision we have ever made."),
    ("zh", "en", "这个手机的续航非常好，能用两天。"),
    ("ja", "zh", "今日は天気がいいので散歩しました。"),
    ("ko", "zh", "오늘 날씨가 정말 좋네요."),
    ("zh", "ko", "请打开字幕。"),
    ("en", "ko", "Please turn on the subtitles if you need them."),
    ("ja", "en", "彼は毎朝コーヒーを飲みます。"),
]

MAX_NEW = 24


class Runner:
    def __init__(self):
        so = ort.SessionOptions()
        so.intra_op_num_threads = THREADS
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.enc = ort.InferenceSession(str(ONNX / "encoder_model_int8.onnx"),
                                        so, providers=["CPUExecutionProvider"])
        self.on = ort.InferenceSession(str(ONNX / "decoder_model_quantized.onnx"),
                                       so, providers=["CPUExecutionProvider"])
        self.past = ort.InferenceSession(str(ONNX / "decoder_with_past_model_quantized.onnx"),
                                         so, providers=["CPUExecutionProvider"])
        self.on_outs = [o.name for o in self.on.get_outputs()]
        self.past_outs = [o.name for o in self.past.get_outputs()]
        self.tok = Tokenizer(TOKENIZER_BIN)

    def encode(self, text, src):
        flores, _ = FLORES[src]
        ids = self.tok.encode(text, flores)
        h = self.enc.run(None, {
            "input_ids": np.array([ids], dtype=np.int64),
            "attention_mask": np.ones([1, len(ids)], dtype=np.int64),
        })[0]
        return h, np.ones([1, h.shape[1]], dtype=np.int64)

    # ---------- 方案 A：两段式 KV cache ----------
    def decode_kv(self, hidden, mask, tgt_id):
        first_logits = None
        t0 = time.perf_counter()
        out = self.on.run(None, {
            "input_ids": np.array([[self.tok.eos]], dtype=np.int64),
            "encoder_hidden_states": hidden,
            "encoder_attention_mask": mask,
        })
        r = dict(zip(self.on_outs, out))
        first_logits = r["logits"][0, -1, :].copy()
        enc_cache = {k: v for k, v in r.items() if ".encoder." in k}
        dec_cache = {k: v for k, v in r.items() if ".decoder." in k}
        t_first = time.perf_counter() - t0

        seq = [tgt_id]
        while len(seq) < MAX_NEW:
            feed = {"input_ids": np.array([[seq[-1]]], dtype=np.int64),
                    "encoder_attention_mask": mask}
            for k, v in enc_cache.items():
                feed["past_key_values." + k[len("present."):]] = v
            for k, v in dec_cache.items():
                feed["past_key_values." + k[len("present."):]] = v
            r = dict(zip(self.past_outs, self.past.run(None, feed)))
            dec_cache = {k: v for k, v in r.items() if k.startswith("present.")}
            nxt = int(np.argmax(r["logits"][0, -1, :]))
            if nxt == self.tok.eos:
                break
            seq.append(nxt)
        return seq, first_logits, t_first, time.perf_counter() - t0

    # ---------- 方案 B：单模型全前缀重放 ----------
    def decode_replay(self, hidden, mask, tgt_id):
        seq = [tgt_id]
        out = self.on.run(None, {
            "input_ids": np.array([seq], dtype=np.int64),
            "encoder_hidden_states": hidden,
            "encoder_attention_mask": mask,
        })
        r = dict(zip(self.on_outs, out))
        first_logits = r["logits"][0, -1, :].copy()
        nxt = int(np.argmax(r["logits"][0, -1, :]))
        if nxt != self.tok.eos:
            seq.append(nxt)
        while len(seq) < MAX_NEW and seq[-1] != self.tok.eos:
            out = self.on.run(None, {
                "input_ids": np.array([seq], dtype=np.int64),
                "encoder_hidden_states": hidden,
                "encoder_attention_mask": mask,
            })
            r = dict(zip(self.on_outs, out))
            nxt = int(np.argmax(r["logits"][0, -1, :]))
            if nxt == self.tok.eos:
                break
            seq.append(nxt)
        return seq, first_logits, 0.0, 0.0


def main():
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", type=int, default=THREADS)
    args = ap.parse_args()

    R = Runner()
    print(f"线程 {args.threads} | decoder_start=</s>={R.tok.eos} | MAX_NEW={MAX_NEW}")
    print("=" * 100)
    print(f"{'用例':<34}{'方案':<10}{'tok':>4}{'首步ms':>9}{'总ms':>8}  {'文本'}")
    print("=" * 100)

    tot = {"kv": [], "replay": []}
    same = 0
    for src, tgt, text in CASES:
        hidden, mask = R.encode(text, src)
        _, tgt_id = FLORES[tgt]

        seq_a, fl_a, t_first, t_a = R.decode_kv(hidden, mask, tgt_id)
        seq_b, fl_b, _, _ = R.decode_replay(hidden, mask, tgt_id)

        # 重新计时 B（上面的调用是第一次，含图预热）
        t0 = time.perf_counter()
        R.decode_replay(hidden, mask, tgt_id)
        t_b = time.perf_counter() - t0

        agree = seq_a == seq_b
        same += agree
        tot["kv"].append(t_a)
        tot["replay"].append(t_b)

        label = f"{src}->{tgt} {text[:16]}"
        print(f"{label:<34}{'A/KV':<10}{len(seq_a):>4}{t_first*1000:>9.0f}{t_a*1000:>8.0f}  {R.tok.decode(seq_a)!r}")
        print(f"{'':<34}{'B/重放':<10}{len(seq_b):>4}{'-':>9}{t_b*1000:>8.0f}  {R.tok.decode(seq_b)!r}")
        print(f"{'':<34}{'一致':<10}{'✅ 是' if agree else '❌ 否'}"
              + ("" if agree else f"   A={seq_a}  B={seq_b}"))
        print()

    n = len(CASES)
    print("=" * 100)
    print(f"输出一致：{same}/{n}")
    print(f"方案 A（两段式 KV cache）平均 {np.mean(tot['kv'])*1000:.0f} ms   "
          f"中位 {np.median(tot['kv'])*1000:.0f} ms")
    print(f"方案 B（全前缀重放）      平均 {np.mean(tot['replay'])*1000:.0f} ms   "
          f"中位 {np.median(tot['replay'])*1000:.0f} ms")
    print(f"→ PC 上 A 比 B 快 {np.mean(tot['replay'])/np.mean(tot['kv']):.2f}×")
    print("=" * 100)


if __name__ == "__main__":
    main()
