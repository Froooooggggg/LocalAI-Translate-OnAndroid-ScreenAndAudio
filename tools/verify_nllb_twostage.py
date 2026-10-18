#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
独立 decoder 两段式解码：decoder_model_quantized(on) + decoder_with_past_model_quantized(on).

为什么用两个文件而不是 merged：
  merged 变体把「首步」与「带缓存续步」塞进同一个图，用 use_cache_branch 切换。
  在 ONNX Runtime 1.31 上走缓存分支必然崩：
    RunOptions: ... If node 'optimum::if' ...
    Reshape node '/model/decoder/layers.0/encoder_attn/Reshape_4'
    The dimension with value zero exceeds the dimension size of the input tensor.
  三路对照已确认（tools/diag_three_ways.py）：关优化、基础优化都崩，
  而无缓存分支虽然能跑，但**同一个 prefix 无论多长都吐出同一个 token**，等于没在解码。
  → merged 在 ORT 1.31 上不可用。

两段式的正确流程（各自 3 个输入，没有 use_cache_branch）：
  step0  decoder_model            : input_ids=[tgt_lang]           → logits + present.*
  stepN  decoder_with_past_model  : input_ids=[last_tok] + past.*  → logits + present.*
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort

sys.path.insert(0, str(Path(__file__).parent))
from verify_nllb_standalone import Tokenizer, TOKENIZER_BIN, FLORES, SAMPLES  # noqa: E402
# 仓库根（tools/ 的上一级）：脚本可以放在任何机器上跑
_R = Path(__file__).resolve().parents[1]


ROOT = Path(str(_R))
ONNX = ROOT / "third_party" / "nllb" / "onnx"
ENCODER = ONNX / "encoder_model_int8.onnx"
DEC_ON = ONNX / "decoder_model_quantized.onnx"
DEC_PAST = ONNX / "decoder_with_past_model_quantized.onnx"


class TwoStageDecoder:
    def __init__(self, threads: int = 4):
        so = ort.SessionOptions()
        so.intra_op_num_threads = threads
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        t0 = time.perf_counter()
        self.enc = ort.InferenceSession(str(ENCODER), so, providers=["CPUExecutionProvider"])
        self.on = ort.InferenceSession(str(DEC_ON), so, providers=["CPUExecutionProvider"])
        self.past = ort.InferenceSession(str(DEC_PAST), so, providers=["CPUExecutionProvider"])
        self.load_ms = (time.perf_counter() - t0) * 1000
        self.tok = Tokenizer(TOKENIZER_BIN)

        self.on_ins = [i.name for i in self.on.get_inputs()]
        self.on_outs = [o.name for o in self.on.get_outputs()]
        self.past_ins = [i.name for i in self.past.get_inputs()]
        self.past_outs = [o.name for o in self.past.get_outputs()]
        self.past_names = [n for n in self.past_ins if n.startswith("past_key_values.")]

    def describe(self):
        print(f"decoder_model            输入 {len(self.on_ins)}: {self.on_ins}")
        print(f"                         输出 {len(self.on_outs)}（含 present.* {len(self.on_outs)-1}）")
        print(f"decoder_with_past_model  输入 {len(self.past_ins)}"
              f"（past_key_values.* {len(self.past_names)}）")
        print(f"  非 past 输入: {[n for n in self.past_ins if not n.startswith('past_key_values.')]}")
        print(f"                         输出 {len(self.past_outs)}")

    def translate(self, text: str, src: str, tgt: str, max_new: int = 128, debug: bool = False):
        src_flores, _ = FLORES[src]
        _, tgt_id = FLORES[tgt]

        t0 = time.perf_counter()
        ids = self.tok.encode(text, src_flores)
        hidden = self.enc.run(None, {
            "input_ids": np.array([ids], dtype=np.int64),
            "attention_mask": np.ones([1, len(ids)], dtype=np.int64),
        })[0]
        S = hidden.shape[1]
        enc_mask = np.ones([1, S], dtype=np.int64)
        t_encode = time.perf_counter() - t0

        # ---- 首步：decoder_model ----
        # ⚠️ 两个必须踩对的坑（已用 tools/find_decoder_start.py 穷举验证）：
        #
        #  坑 1：decoder_start_token_id = </s> = 2。既不是 <s>=0 也不是 <pad>=1。
        #        只有 2 能让第 0 步的 logits 落在「语言 token 族」里
        #        （top1 = ron_Latn/ell_Grek/tur_Latn 这类语种标记），
        #        即这一步是「该输出哪种语言」的决策点。
        #        用 0 或 1 时 top1 恒为 </s>，模型当场收尾 —— 表现是
        #        「不崩但一个字都不输出」，极难反查。
        #
        #  坑 2：forced BOS。第 0 步 logits 算出后**丢弃**，直接指定
        #        「第一个生成 token」= 目标语种 token，之后才让它预测正文。
        #        若改为把语种 token 当**输入**喂进去，模型进入恒定退化态，
        #        每步都吐同一个 token（实测恒为 248059='▁'）。
        t0 = time.perf_counter()
        out = self.on.run(None, {
            "input_ids": np.array([[self.tok.eos]], dtype=np.int64),
            "encoder_hidden_states": hidden,
            "encoder_attention_mask": enc_mask,
        })
        res = dict(zip(self.on_outs, out))
        enc_cache = {k: v for k, v in res.items() if ".encoder." in k}
        dec_cache = {k: v for k, v in res.items() if ".decoder." in k}

        # forced BOS：首步 logits 作废，直接指定第一个生成 token 为目标语种
        nxt = tgt_id
        t_first = time.perf_counter() - t0

        if debug:
            top = np.argsort(-res["logits"][0, -1, :])[:5]
            print(f"      step0 (decoder_start=</s>{self.tok.eos}) 未约束 top5: "
                  + "  ".join(f"{int(i)}:{self.tok.vocab[int(i)]!r}" for i in top))
            print(f"      forced_bos -> {nxt} ({FLORES[tgt][0]})")
            print(f"      encoder 侧 present {len(enc_cache)} 个，"
                  f"形状 {next(iter(enc_cache.values())).shape}")

        generated = []
        t0 = time.perf_counter()
        for step in range(max_new):
            if nxt == self.tok.eos:
                break
            generated.append(nxt)

            # ---- 续步：decoder_with_past_model ----
            # 50 个输入 = input_ids + encoder_attention_mask + 48 个 past_key_values.*
            # 其中 encoder 侧 24 个来自首步（保持不变），decoder 侧 24 个来自上一续步。
            feed = {
                "input_ids": np.array([[nxt]], dtype=np.int64),
                "encoder_attention_mask": enc_mask,
            }
            for k, v in enc_cache.items():
                feed["past_key_values." + k[len("present."):]] = v
            for k, v in dec_cache.items():
                feed["past_key_values." + k[len("present."):]] = v
            missing = [n for n in self.past_ins if n not in feed]
            if missing:
                return {"error": f"缺少输入 {missing[:4]}", "generated": generated}

            try:
                out = self.past.run(None, feed)
            except Exception as e:
                return {"error": f"step{step+1}: {type(e).__name__}: {str(e)[:300]}",
                        "generated": generated}
            res = dict(zip(self.past_outs, out))
            nxt = int(np.argmax(res["logits"][0, -1, :]))
            dec_cache = {k: v for k, v in res.items() if k.startswith("present.")}
            if debug and step < 6:
                print(f"      step{step+1} -> {nxt} ({self.tok.vocab[nxt]!r})")

        t_decode = time.perf_counter() - t0
        return {
            "text": self.tok.decode(generated),
            "ids": generated,
            "encode_ms": t_encode * 1000,
            "first_token_ms": t_first * 1000,
            "decode_ms": t_decode * 1000,
            "total_ms": (t_encode + t_first + t_decode) * 1000,
            "n_tokens": len(generated),
            "src_len": S,
        }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--max-new", type=int, default=128)
    ap.add_argument("--debug", action="store_true")
    ap.add_argument("--only", default="")
    ap.add_argument("--out", default=str(ROOT / "tools" / "nllb_reference.json"))
    args = ap.parse_args()

    for p in (ENCODER, DEC_ON, DEC_PAST):
        if not p.exists():
            sys.exit(f"缺文件: {p}")
        print(f"  {p.name:<42} {p.stat().st_size:>14,} 字节")

    dec = TwoStageDecoder(threads=args.threads)
    print(f"\n加载 {dec.load_ms:.0f} ms")
    dec.describe()
    print(f"tokenizer vocab={len(dec.tok.vocab):,}  尾部多余字节={dec.tok.trailing}\n")

    results, ok = [], 0
    for i, (src, tgt, text) in enumerate(SAMPLES, 1):
        if args.only and str(i) not in args.only.split(","):
            continue
        print(f"[{i}] {src}->{tgt}  {text}")
        r = dec.translate(text, src, tgt, max_new=args.max_new, debug=args.debug)
        if "error" in r:
            print(f"    ❌ {r['error']}")
            r = {"i": i, "src": src, "tgt": tgt, "input": text, **r}
        else:
            ok += 1
            print(f"    -> {r['text']}")
            print(f"       首字 {r['first_token_ms']:.0f} ms | 编码 {r['encode_ms']:.0f} ms | "
                  f"解码 {r['decode_ms']:.0f} ms | 共 {r['total_ms']:.0f} ms | "
                  f"{r['n_tokens']} tok | 源长 {r['src_len']}")
            r = {"i": i, "src": src, "tgt": tgt, "input": text, **r}
        results.append(r)
        print()

    print(f"成功 {ok}/{len(results)}")
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"decoder_on": DEC_ON.name, "decoder_past": DEC_PAST.name,
                   "threads": args.threads, "results": results}, f, ensure_ascii=False, indent=2)
    print(f"参考结果已写出 {args.out}")


if __name__ == "__main__":
    main()
