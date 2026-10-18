#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
精度消融实验：encoder int8/FP32 × decoder quantized/FP32 的 8 种组合。

要回答的问题
------------
全量测试中观察到「重复 token」（如 日本语 今天天气 → '今天天気'、
中→韩 '자막을 켜면' 与 '자막을 열면' 两种不同输出），
且在 PC 上不同精度模型给出不同 token 序列（方案 A 与方案 B 在同一精度下
前 2 个 token 一致、之后分岔）。

量化误差会累积：每一步 argmax 只要被噪声翻转一次，后续输入就变了，
错误会沿自回归链路放大。所以必须分清：
  - 若只有 int8/quantized 出问题 → 量化损伤，手机端应改用 FP32（但要付大小/速度代价）
  - 若全精度也出现 → 模型固有噪声（NLLB-600M 蒸馏版本身如此），与量化无关

同时统计「连续重复 token」出现率，作为可自动判定的质量指标。
"""
from __future__ import annotations

import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import onnxruntime as ort

sys.path.insert(0, str(Path(__file__).parent))
from verify_nllb_standalone import Tokenizer, TOKENIZER_BIN, FLORES  # noqa: E402
# 仓库根（tools/ 的上一级）：脚本可以放在任何机器上跑
_R = Path(__file__).resolve().parents[1]


ROOT = Path(str(_R))
ONNX = ROOT / "third_party" / "nllb" / "onnx"
THREADS = 6
MAX_NEW = 22

# 覆盖全部 4 种语言，且包含上次出问题的方向
CASES = [
    ("en", "zh", "Hello everyone, welcome back to the channel."),
    ("en", "zh", "The battery lasts about two days on a single charge."),
    ("en", "ja", "I think this is the best decision we have ever made."),
    ("en", "ko", "Please turn on the subtitles if you need them."),
    ("ja", "zh", "今日は天気がいいので散歩しました。"),
    ("ja", "zh", "この動画は彼の人気チャンネルのものです。"),
    ("ja", "en", "彼は毎朝コーヒーを飲みます。"),
    ("zh", "en", "这个手机的续航非常好，能用两天。"),
    ("zh", "ja", "今天的天气非常好，适合出去走走。"),
    ("zh", "ko", "请打开字幕。"),
    ("ko", "zh", "오늘 날씨가 정말 좋네요."),
    ("ko", "zh", "여러분 안녕하세요, 영상을 시작하겠습니다."),
]

ENCODER = "encoder_model_int8.onnx"
ENCODER_FP32 = "encoder_model.onnx"
DEC = "decoder_model_quantized.onnx"
DEC_FP32 = "decoder_model.onnx"
DEC_PAST = "decoder_with_past_model_quantized.onnx"
DEC_PAST_FP32 = "decoder_with_past_model.onnx"


def dup_runs(seq, eos):
    """统计连续重复 token 的「多余次数」——同一个 token 连着出现 n 次算 n-1 次冗余。"""
    extra, i = 0, 1
    while i < len(seq):
        if seq[i] == seq[i - 1] and seq[i] != eos:
            j = i
            while j < len(seq) and seq[j] == seq[i]:
                j += 1
            extra += (j - i)
            i = j
        else:
            i += 1
    return extra


class Pair:
    """一对 (encoder, decoder, decoder_with_past) 组合。"""

    def __init__(self, enc_file, dec_file, past_file, tok, label):
        so = ort.SessionOptions()
        so.intra_op_num_threads = THREADS
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.enc = ort.InferenceSession(str(ONNX / enc_file), so, providers=["CPUExecutionProvider"])
        self.on = ort.InferenceSession(str(ONNX / dec_file), so, providers=["CPUExecutionProvider"])
        self.past = ort.InferenceSession(str(ONNX / past_file), so, providers=["CPUExecutionProvider"])
        self.on_outs = [o.name for o in self.on.get_outputs()]
        self.past_outs = [o.name for o in self.past.get_outputs()]
        self.tok = tok
        self.label = label

    def run(self, text, src, tgt):
        t_all = time.perf_counter()
        flores, _ = FLORES[src]
        ids = self.tok.encode(text, flores)
        hidden = self.enc.run(None, {
            "input_ids": np.array([ids], dtype=np.int64),
            "attention_mask": np.ones([1, len(ids)], dtype=np.int64),
        })[0]
        mask = np.ones([1, hidden.shape[1]], dtype=np.int64)
        _, tgt_id = FLORES[tgt]

        t0 = time.perf_counter()
        r = dict(zip(self.on_outs, self.on.run(None, {
            "input_ids": np.array([[self.tok.eos]], dtype=np.int64),
            "encoder_hidden_states": hidden,
            "encoder_attention_mask": mask,
        })))
        t_first = time.perf_counter() - t0
        enc_cache = {k: v for k, v in r.items() if ".encoder." in k}
        dec_cache = {k: v for k, v in r.items() if ".decoder." in k}
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
        return seq, t_first, time.perf_counter() - t_all


def main():
    tok = Tokenizer(TOKENIZER_BIN)

    # 先自检 tokenizer.decode：把已知会产生重复的序列手工喂进去
    print("=" * 100)
    print("tokenizer.decode 自检（排除「重复是分词器解码造成的」这一可能）")
    print("=" * 100)
    for probe in ([7213, 249910, 249910, 252663], [7213, 249910, 252663],
                  [9680, 38275, 248], [248059, 252854, 249725]):
        print(f"  {probe}\n    → {tok.decode(probe)!r}")
    print()

    # 可用组合
    combos = []
    for enc_f, enc_n in ((ENCODER, "E-int8"), (ENCODER_FP32, "E-fp32")):
        for d_f, d_n, p_f, p_n in ((DEC, "D-q", DEC_PAST, "P-q"),
                                   (DEC_FP32, "D-fp32", DEC_PAST_FP32, "P-fp32")):
            if (ONNX / enc_f).exists() and (ONNX / d_f).exists() and (ONNX / p_f).exists():
                combos.append((enc_f, d_f, p_f, f"{enc_n}+{d_n}"))
    print(f"可用组合 {len(combos)} 个：" + ", ".join(c[3] for c in combos))
    print()

    results = {}
    for enc_f, d_f, p_f, label in combos:
        print("=" * 100)
        print(f"### {label}")
        print("=" * 100)
        p = Pair(enc_f, d_f, p_f, tok, label)
        rows, dups, times = [], 0, []
        for src, tgt, text in CASES:
            seq, t_first, t_all = p.run(text, src, tgt)
            d = dup_runs(seq, tok.eos)
            dups += d
            times.append(t_all)
            rows.append((src, tgt, text, seq, d, t_first, t_all))
            flag = f"  ⚠️ 重复{d}" if d else ""
            print(f"  {src}->{tgt}  {tok.decode(seq)!r}{flag}")
        results[label] = (rows, dups, float(np.mean(times)))
        print(f"\n  → 重复 token 冗余总数 {dups} | 平均 {np.mean(times)*1000:.0f} ms"
              f" | 首步 {np.mean([r[5] for r in rows])*1000:.0f} ms")
        print()

    # 交叉对比：同一用例在不同精度下的 token 序列差异
    print("=" * 100)
    print("精度间一致性（同一用例，各组合的 token 序列是否相同）")
    print("=" * 100)
    base = combos[0][3]
    for label in [c[3] for c in combos]:
        if label == base:
            continue
        diff = 0
        for i in range(len(CASES)):
            if results[label][0][i][3] != results[base][0][i][3]:
                diff += 1
        print(f"  {base:<16} vs {label:<16} 序列不同 {diff}/{len(CASES)} 例")

    print()
    print("=" * 100)
    print(f"{'组合':<20}{'重复冗余':>10}{'平均ms':>10}  结论")
    print("=" * 100)
    for label in [c[3] for c in combos]:
        _, dups, t = results[label]
        verdict = "✅ 无重复" if dups == 0 else f"⚠️ 有重复({dups})"
        print(f"  {label:<18}{dups:>10}{t*1000:>10.0f}  {verdict}")
    print("=" * 100)


if __name__ == "__main__":
    main()
