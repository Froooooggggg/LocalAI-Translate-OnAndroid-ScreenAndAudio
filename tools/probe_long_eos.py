#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""长句提前收尾：先问「EOS 赢得多干净」，再试两条修法。

## 背景

实测（tools/diag_long_sentence.py）：
    Hello everyone, welcome back to the channel.          → 完整
    39 词长句                                              → 第 23 步吐 EOS，只翻了前半截
    22 词长句                                              → 完整

停在一个**逗号之后**——典型的 NLLB 行为：它是句级平行语料训出来的翻译专用模型，
看到一长串内容会认为「一句说完了」就在从句边界收尾。

## 这个脚本要回答两件事

  Q1  那个 EOS 是「干净地赢」还是「凑巧赢」？
      —— 把停止那一步的 EOS 概率与 top-5 打出来。
         EOS 概率很高（>0.5）说明模型是真心要停 → 屏蔽 EOS 只会得到胡话；
         EOS 只是勉强领先（<0.2）说明还有救。

  Q2  两条不改模型的修法哪条能用？
      A. 屏蔽 EOS 继续解（suppress）
      B. 按逗号/分号切开，逐段翻译再拼（split）
"""
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort

ROOT = Path(str(_R))
BASE = ROOT / "third_party" / "nllb" / "onnx"
sys.path.insert(0, str(ROOT / "tools"))
from verify_nllb_standalone import Tokenizer  # noqa: E402
# 仓库根（tools/ 的上一级）：脚本可以放在任何机器上跑
_R = Path(__file__).resolve().parents[1]

ZH, EOS, ENG = 256200, 2, 256047


class Eng:
    """与 NllbTranslator.kt 同构的解码器（见 diag_long_sentence.py 顶部的说明）。"""

    def __init__(self):
        so = ort.SessionOptions()
        so.intra_op_num_threads = 4
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.enc = ort.InferenceSession(str(BASE / "encoder_model_int8.onnx"), so, providers=["CPUExecutionProvider"])
        self.d1 = ort.InferenceSession(str(BASE / "decoder_model_quantized.onnx"), so, providers=["CPUExecutionProvider"])
        self.d2 = ort.InferenceSession(str(BASE / "decoder_with_past_model_quantized.onnx"), so, providers=["CPUExecutionProvider"])
        self.past_names = [i.name for i in self.d2.get_inputs() if i.name.startswith("past_key_values.")]

    def _start(self, text, tok):
        ids = tok.encode(text, "eng_Latn")
        mask = np.ones((1, len(ids)), dtype=np.int64)
        hidden = self.enc.run(None, {"input_ids": np.array([ids], dtype=np.int64),
                                     "attention_mask": mask})[0]
        n1 = [o.name for o in self.d1.get_outputs()]
        r1 = dict(zip(n1, self.d1.run(None, {
            "input_ids": np.array([[EOS]], dtype=np.int64),
            "encoder_hidden_states": hidden, "encoder_attention_mask": mask})))
        cache = {n.replace("present.", "past_key_values."): r1[n] for n in n1 if n.startswith("present.")}
        return cache, mask, ids

    def _step(self, last, cache, mask, penalty, gen):
        feed = {"input_ids": np.array([[last]], dtype=np.int64), "encoder_attention_mask": mask}
        for n in self.past_names:
            feed[n] = cache[n]
        n2 = [o.name for o in self.d2.get_outputs()]
        r2 = dict(zip(n2, self.d2.run(None, feed)))
        for n in n2:
            if n.startswith("present."):
                cache[n.replace("present.", "past_key_values.")] = r2[n]
        logits = np.array(r2["logits"][0, -1, :], dtype=np.float32)
        if penalty > 1.0:
            for i in gen:
                logits[i] = logits[i] / penalty if logits[i] > 0 else logits[i] * penalty
        return logits

    def translate(self, tok, text, max_new=96, penalty=1.1, suppress_eos_steps=0, trace=False):
        cache, mask, _ = self._start(text, tok)
        last, seq, n_eos_seen, n = ZH, [], 0, 0
        first_eos_logits = None
        while len(seq) < max_new and n < max_new:
            n += 1
            if last == EOS:
                n_eos_seen += 1
                if n_eos_seen == 1 and first_eos_logits is None:
                    break          # 记录第一次 EOS 就停（trace 用）
            if not (seq and seq[-1] == last):
                seq.append(last)
            logits = self._step(last, cache, mask, penalty, seq)
            # 屏蔽 EOS：把 eos 的 logit 压到极小
            if suppress_eos_steps > 0 and len(seq) >= 1:
                logits[EOS] = -1e30
            last = int(np.argmax(logits))
        return tok.decode(seq), seq, n


def main():
    tok = Tokenizer(ROOT / "app" / "src" / "main" / "assets" / "nllb" / "tokenizer.bin")
    eng = Eng()
    long1 = ("I think the most important thing that we need to understand about this whole "
             "situation is that nobody really knows what is going to happen next, and that "
             "is exactly why we should all be a little more careful.")

    # ---------- Q1：停止那一步，EOS 赢得多干净 ----------
    print("=" * 70)
    print("Q1  停止那一步的 logits 分布（不屏蔽 EOS）")
    print("=" * 70)
    cache, mask, _ = eng._start(long1, tok)
    last, seq = ZH, []
    while True:
        if last == EOS:
            break
        if not (seq and seq[-1] == last):
            seq.append(last)
        logits_pre = eng._step(last, cache, mask, 1.1, seq)   # 已带 repeatPenalty
        # 再算一次不带惩罚的，用来看惩罚有没有把 EOS 顶上去
        last = int(np.argmax(logits_pre))
    p = np.exp(logits_pre - logits_pre.max())
    p /= p.sum()
    order = np.argsort(-logits_pre)[:8]
    print(f"生成 {len(seq)} 个 token 后，模型选 EOS。该步 top-8：")
    for i in order:
        mark = "  ← EOS" if int(i) == EOS else ""
        print(f"    p={p[int(i)]:.4f}  id={int(i):<7} {tok.vocab[int(i)]!r}{mark}")
    print(f"\nEOS 概率 = {p[EOS]:.4f}"
          + ("   → 真心要停，屏蔽 EOS 没用" if p[EOS] > 0.5 else "   → 只是勉强领先，还有救"))

    # ---------- Q2-A：屏蔽 EOS 继续解 ----------
    print("\n" + "=" * 70)
    print("Q2-A  屏蔽 EOS 继续解 30 步")
    print("=" * 70)
    txt, seq2, _ = eng.translate(tok, long1, max_new=96, suppress_eos_steps=30)
    print(f"生成 {len(seq2)} token: {txt!r}")

    # ---------- Q2-B：按逗号切句逐段翻 ----------
    print("\n" + "=" * 70)
    print("Q2-B  按标点切句，逐段翻译再拼")
    print("=" * 70)
    import re

    parts = [s.strip() for s in re.split(r"(?<=[,;.])\s+", long1) if s.strip()]
    out = []
    for part in parts:
        t, s, _ = eng.translate(tok, part, max_new=64)
        print(f"  [{len(s):>2} tok] {part[:60]}")
        print(f"           → {t}")
        out.append(t)
    print(f"\n拼接结果: {''.join(out)}")


if __name__ == "__main__":
    main()
