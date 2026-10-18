#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""在 PC 上复现真机的崩溃，并验证「无 past」解码范式。

## 为什么要这个脚本

真机上 `decoder_with_past_model_quantized.onnx` 跑完第 1、2 步后进程 SIGSEGV：
    Fatal signal 11 (SIGSEGV), code 1 (SEGV_MAPERR),
    fault addr 0x2e7365756c61767 in tid (DefaultDispatch)
崩溃点在 ONNX Runtime 原生库里（`Java_ai_onnxruntime_OrtSession_run+552`），
Java 侧拿不到任何有用信息，只能靠 tombstone 反汇编，迭代一轮要几分钟。

PC 侧是同一个 onnxruntime **1.31.0**、同样的两个模型文件，但 Python 会直接
告诉我们"哪个输入名/形状不对"。所以先用它把问题问清楚。

## 要回答的三个问题

  Q1  with-past 的增量解码，在 PC 上也崩吗？（同一个 bug 还是 Android 独有）
  Q2  只用 `decoder_model_quantized.onnx`（3 输入、无 past、full-seq 重跑）
      能不能产出与两段式参考实现**完全一致**的译文？
  Q3  若 Q2 成立，它的耗时是多少？（决定手机上能不能接受）

Q2 是关键：这条路彻底不碰 past 缓存，而真机上崩的正是反复读写 past 的那条路。
"""
import sys
import time

import numpy as np
import onnxruntime as ort
# 仓库根（tools/ 的上一级）：脚本可以放在任何机器上跑
_R = Path(__file__).resolve().parents[1]

sys.path.insert(0, str(_R / 'tools'))

BASE = str(_R / '模型/上游/nllb/onnx')
TOK = str(_R / 'app/src/main/assets/nllb/tokenizer.bin')

ENCODER = BASE + r"\encoder_model_int8.onnx"
DEC_FIRST = BASE + r"\decoder_model_quantized.onnx"
DEC_PAST = BASE + r"\decoder_with_past_model_quantized.onnx"

# FLORES 语种码 → 语言 token id（与 Lang.kt 对齐）
LANG_TOKEN = {"eng_Latn": 256047, "zho_Hans": 256200, "jpn_Jpan": 256079, "kor_Hang": 256098}


def load_tokenizer():
    """复用 Python 侧的参考 tokenizer 实现（它要 pathlib.Path，不是 str）。"""
    from pathlib import Path
    from verify_nllb_standalone import Tokenizer  # noqa: E402

    return Tokenizer(Path(TOK))


def softmax_argmax(logits):
    return int(np.argmax(logits))


def decode_full_seq(tok, tgt_id, dec, enc_hidden, enc_mask, max_new=64, verbose=False):
    """无 past 范式：每一步都把「已生成的完整 token 序列」重新喂一遍。

    这是唯一不依赖 KV 缓存的解码方式，也是最不容易出所有权/生命周期 bug 的。
    代价是 O(n²) 的计算量。
    """
    seq = []
    first_ms = None

    ids = np.array([[tok.eos]], dtype=np.int64)  # decoder_start_token_id = </s> = 2
    for step in range(max_new):
        out = dec.run(None, {
            "input_ids": ids,
            "encoder_hidden_states": enc_hidden,
            "encoder_attention_mask": enc_mask,
        })
        logits = out[0]  # [1, seq_len, vocab]

        if step == 0:
            # 首步 logits 按设计丢弃，直接强制目标语种 token（forced BOS）
            nxt = tgt_id
            first_ms = 0.0
        else:
            nxt = softmax_argmax(logits[0, -1])

        if nxt == tok.eos:
            break
        seq.append(nxt)
        ids = np.concatenate([ids, np.array([[nxt]], dtype=np.int64)], axis=1)
        if verbose:
            print("    step %2d -> %d %r" % (step, nxt, tok.decode([nxt])))

    return seq, first_ms


def main():
    tok = load_tokenizer()
    print("词表:", tok.vocab_size if hasattr(tok, "vocab_size") else "?", "| eos =", tok.eos)

    so = ort.SessionOptions()
    so.intra_op_num_threads = 4
    enc = ort.InferenceSession(ENCODER, so, providers=["CPUExecutionProvider"])
    dec_first = ort.InferenceSession(DEC_FIRST, so, providers=["CPUExecutionProvider"])
    dec_past = ort.InferenceSession(DEC_PAST, so, providers=["CPUExecutionProvider"])

    print("\n=== Q1: with-past 增量解码在 PC 上会不会崩 ===")
    text = "Hello everyone, welcome back to the channel."
    src_flores, tgt_flores = "eng_Latn", "zho_Hans"
    src_ids = tok.encode(text, src_flores)
    hid = enc.run(None, {
        "input_ids": np.array([src_ids], dtype=np.int64),
        "attention_mask": np.ones([1, len(src_ids)], dtype=np.int64),
    })[0]
    S = hid.shape[1]
    emask = np.ones([1, S], dtype=np.int64)
    print("  encoder 输出:", hid.shape)

    # 首步
    out1 = dec_first.run(None, {
        "input_ids": np.array([[tok.eos]], dtype=np.int64),
        "encoder_hidden_states": hid,
        "encoder_attention_mask": emask,
    })
    print("  首步输出个数:", len(out1), "| logits:", out1[0].shape)
    present_names = [o.name for o in dec_first.get_outputs() if o.name.startswith("present.")]
    name2idx = {o.name: i for i, o in enumerate(dec_first.get_outputs())}
    cache = {}
    for n in present_names:
        cache["past_key_values." + n[len("present."):]] = out1[name2idx[n]]
    print("  首步 present:", len(present_names), "-> cache:", len(cache))

    dec_past_outs = [o.name for o in dec_past.get_outputs()]
    past_outs = [n for n in dec_past_outs if n.startswith("present.")]
    print("  with-past 输出 present:", len(past_outs), past_outs[:2])
    past_name2idx = {n: i for i, n in enumerate(dec_past_outs)}

    nxt = LANG_TOKEN[tgt_flores]
    seq = []
    try:
        for step in range(1, 24):
            feed = dict(cache)
            feed["input_ids"] = np.array([[nxt]], dtype=np.int64)
            feed["encoder_attention_mask"] = emask
            out = dec_past.run(None, feed)
            for n in past_outs:
                cache["past_key_values." + n[len("present."):]] = out[past_name2idx[n]]
            nxt = softmax_argmax(out[0][0, -1])
            if nxt == tok.eos:
                print("  step %d: 命中 eos" % step)
                break
            seq.append(nxt)
        print("  ✓ PC 上 with-past 跑通，共 %d 步" % len(seq))
        print("    cache 条目数:", len(cache),
              "| 含 encoder 侧:", "past_key_values.0.encoder.key" in cache)
    except Exception as e:
        print("  ✗ PC 上 with-past 抛异常:", type(e).__name__, e)
        seq = []

    print("\n=== Q2: 无 past 全序列重跑，能否给出同样的译文 ===")
    t0 = time.perf_counter()
    seq2, _ = decode_full_seq(tok, LANG_TOKEN[tgt_flores], dec_first, hid, emask, max_new=64)
    dt2 = time.perf_counter() - t0

    print("  with-past :", repr(tok.decode(seq)) if seq else "(未跑通)")
    print("  full-seq  :", repr(tok.decode(seq2)))
    print("  两者是否一致:", seq == seq2)
    print("\n=== Q3: full-seq 耗时 ===")
    print("  %.2f s / %d token  (%.0f ms/token)" % (dt2, len(seq2), dt2 * 1000 / max(1, len(seq2))))


if __name__ == "__main__":
    main()
