#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""诊断「长句只翻译一点」——把 EOS 赢在哪一步、赢在什么上下文里打出来。

## 为什么要这个脚本

用户实测：喂进去的英文一长，译文就只出来一小截。真机上只看到
`译文 首字xxxms 整句xxxms  短短几个字.`，看不出是**模型想停**、**循环写错了**，
还是**重复折叠把内容吃掉了**。在 PC 上跑同一批模型，把每一步的 top-k 连概率一起打出来，
就能直接区分这几种情况。

## ⚠️ 必须与 NllbTranslator.kt 完全同构，否则测的不是真机跑的那条路

真机的解码范式（`app/src/main/java/com/screenast/app/translate/nllb/NllbTranslator.kt`）：

  1. **首步喂的是 `</s>`（eosId=2），不是目标语种码。**
     `firstStep()` 里 `longArrayOf(tokenizer.eosId.toLong())`，
     第 0 步 logits **按设计丢弃**（注释：「语种靠 forced BOS 注入」）。
  2. 第一个正文 token **直接取目标语种码**（`return tgtTokenId`），不是 argmax。
  3. 之后每步：`input_ids=[上一个 token]` + 48 个 past → `decoder_with_past`，
     只替换 **decoder 侧** 24 个 present（encoder 侧那 24 个是首步算好后一直复用的）。
  4. `argmax` 带 `repeatPenalty=1.1`：已在序列里的 token，正 logit 除以 1.1、负 logit 乘 1.1。

第一版脚本漏了第 1、2 条（直接拿语种码当 decoder 输入），结果**连短句都退化成
一直吐 `▁`**——那不是模型的毛病，是脚本自己走错了范式。这个坑值得记下来：
**PC 侧复现脚本一旦与真机实现不同构，得出的结论会完全反过来。**

## 用法

    python tools/diag_long_sentence.py                 # 跑内置的长短对照样例
    python tools/diag_long_sentence.py --text "..."    # 跑指定句子
    python tools/diag_long_sentence.py --collapse      # 额外模拟真机的重复折叠，看界面会显示什么
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort

ROOT = Path(str(_R))
BASE = ROOT / "third_party" / "nllb" / "onnx"
ENC = BASE / "encoder_model_int8.onnx"
DEC_FIRST = BASE / "decoder_model_quantized.onnx"
DEC_PAST = BASE / "decoder_with_past_model_quantized.onnx"

sys.path.insert(0, str(ROOT / "tools"))
from verify_nllb_standalone import Tokenizer  # noqa: E402
# 仓库根（tools/ 的上一级）：脚本可以放在任何机器上跑
_R = Path(__file__).resolve().parents[1]


ENG = 256047
ZH = 256200
EOS = 2


def topk(vocab, logits, k=5):
    idx = np.argsort(-logits)[:k]
    p = np.exp(logits - logits.max())
    p = p / p.sum()
    return [(vocab[int(i)], round(float(p[int(i)]), 3)) for i in idx]


def argmax_penalized(logits, penalty, generated):
    if penalty > 1.0 and generated:
        for i in generated:
            logits[i] = logits[i] / penalty if logits[i] > 0 else logits[i] * penalty
    return int(np.argmax(logits)), logits


def run(tok, text, max_new, label, penalty=1.1, show=True, collapse=False):
    so = ort.SessionOptions()
    so.intra_op_num_threads = 4
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    enc_s = ort.InferenceSession(str(ENC), so, providers=["CPUExecutionProvider"])
    d1 = ort.InferenceSession(str(DEC_FIRST), so, providers=["CPUExecutionProvider"])
    d2 = ort.InferenceSession(str(DEC_PAST), so, providers=["CPUExecutionProvider"])

    ids = tok.encode(text, "eng_Latn")
    print(f"\n===== {label} =====")
    print(f"源文本 {len(text)} 字符 → {len(ids)} token（含语种前缀与 EOS）")

    enc_mask = np.ones((1, len(ids)), dtype=np.int64)
    hidden = enc_s.run(None, {
        "input_ids": np.array([ids], dtype=np.int64),
        "attention_mask": enc_mask,
    })[0]
    S = hidden.shape[1]

    # ---- 第 1 步：喂 </s>，丢弃 logits ----
    n1 = [o.name for o in d1.get_outputs()]
    r1 = dict(zip(n1, d1.run(None, {
        "input_ids": np.array([[EOS]], dtype=np.int64),
        "encoder_hidden_states": hidden,
        "encoder_attention_mask": enc_mask,
    })))
    cache = {n.replace("present.", "past_key_values."): r1[n] for n in n1 if n.startswith("present.")}
    print(f"  首步 present 数 = {len(cache)}（encoder 侧 + decoder 侧）")

    past_names = [i.name for i in d2.get_inputs() if i.name.startswith("past_key_values.")]
    last = ZH                      # forced BOS：目标语种码
    seq = []
    n_steps = 0
    stop = None
    while len(seq) < max_new:
        if last == EOS:
            stop = n_steps
            break
        if seq and seq[-1] == last:
            pass                   # 折叠：不加进 seq
        else:
            seq.append(last)
        feed = {"input_ids": np.array([[last]], dtype=np.int64), "encoder_attention_mask": enc_mask}
        for n in past_names:
            feed[n] = cache[n]
        n2 = [o.name for o in d2.get_outputs()]
        r2 = dict(zip(n2, d2.run(None, feed)))
        for n in n2:
            if n.startswith("present."):
                cache[n.replace("present.", "past_key_values.")] = r2[n]
        logits = np.array(r2["logits"][0, -1, :], dtype=np.float32)
        last, pen = argmax_penalized(logits, penalty, seq)
        n_steps += 1
        if show and n_steps <= 8:
            print(f"  step {n_steps}: {tok.vocab[last]!r}  top5={topk(tok.vocab, pen)}")

    print(f"  解码步数 {n_steps}，去重后正文 token = {len(seq)}"
          + (f"，在第 {stop} 步遇到 EOS" if stop else "（没遇到 EOS，是用完 max_new 停的）"))
    print(f"  译文: {tok.decode(seq)!r}")
    return seq, stop


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--text", default="")
    ap.add_argument("--max-new", type=int, default=96)
    ap.add_argument("--penalty", type=float, default=1.1)
    args = ap.parse_args()

    tok = Tokenizer(ROOT / "app" / "src" / "main" / "assets" / "nllb" / "tokenizer.bin")

    if args.text:
        run(tok, args.text, args.max_new, "指定句子", args.penalty)
        return

    short = "Hello everyone, welcome back to the channel."
    long1 = ("I think the most important thing that we need to understand about this whole "
             "situation is that nobody really knows what is going to happen next, and that "
             "is exactly why we should all be a little more careful.")
    long2 = ("The association points out some alarming findings, including that men are "
             "four times more likely than women to die by suicide worldwide.")

    run(tok, short, args.max_new, f"短句（{len(short.split())} 词）", args.penalty)
    run(tok, long1, args.max_new, f"长句 1（{len(long1.split())} 词）", args.penalty)
    run(tok, long2, args.max_new, f"长句 2（{len(long2.split())} 词）", args.penalty)


if __name__ == "__main__":
    main()
