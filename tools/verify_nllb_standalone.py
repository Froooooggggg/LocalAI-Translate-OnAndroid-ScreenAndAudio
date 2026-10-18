#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NLLB 端侧翻译 —— 独立 decoder 参考实现（Kotlin 侧照此移植）。

为什么要单独一个脚本：`decoder_model_merged_quantized.onnx` 在 ONNX Runtime 1.31 上
根本跑不通（详见 tools/diag_decoder_step0.py 的结论），本脚本改用**独立** decoder：

    decoder_model_quantized.onnx
      inputs : input_ids[1,T], encoder_hidden_states[1,S,1024], encoder_attention_mask[1,S]
      outputs: logits[1,T,V] + present.*.decoder.{key,value}[1,16,P,64]
               + present.*.encoder.{key,value}[1,16,S,64]
      present 的 decoder 侧**自带**本步的 token（靠 `cache_position`），
      所以下一步直接把它当 past 喂回去即可，不需要我自己拼 concat。

与 merged 变体的关键差别：没有 `use_cache_branch`，两条路径是两个独立的 ONNX 文件。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
# 仓库根（tools/ 的上一级）：脚本可以放在任何机器上跑
_R = Path(__file__).resolve().parents[1]

ROOT = Path(str(_R))
NLLB = ROOT / "third_party" / "nllb"
ENCODER = NLLB / "onnx" / "encoder_model_int8.onnx"
DECODER = NLLB / "onnx" / "decoder_model_quantized.onnx"
TOKENIZER_BIN = ROOT / "app" / "src" / "main" / "assets" / "nllb" / "tokenizer.bin"

MAGIC = b"SNATKN02"
SPACE = "\u2581"

# 与 app/src/main/java/com/screenast/app/core/Lang.kt 保持一致（已交叉校验）
FLORES = {
    "zh": ("zho_Hans", 256200),
    "en": ("eng_Latn", 256047),
    "ja": ("jpn_Jpan", 256079),
    "ko": ("kor_Hang", 256098),
}

SAMPLES = [
    ("en", "zh", "Hello everyone, welcome back to the channel."),
    ("en", "zh", "The battery lasts about two days on a single charge."),
    ("en", "ja", "I think this is the best decision we have ever made."),
    ("en", "ko", "Please turn on the subtitles if you need them."),
    ("ja", "zh", "\u3053\u306e\u52d5\u753b\u306f\u5f7c\u306e\u4eba\u6c17\u30c1\u30e3\u30f3\u30cd\u30eb\u306e\u3082\u306e\u3067\u3059\u3002"),
    ("ja", "zh", "\u4eca\u65e5\u306f\u5929\u6c17\u304c\u3044\u3044\u306e\u3067\u6563\u6b69\u3057\u307e\u3057\u305f\u3002"),
    ("ja", "en", "\u5f7c\u306f\u6bce\u671d\u30b3\u30fc\u30d2\u30fc\u3092\u98f2\u307f\u307e\u3059\u3002"),
    ("zh", "en", "\u5927\u5bb6\u597d\uff0c\u6b22\u8fce\u56de\u5230\u6211\u7684\u9891\u9053\u3002"),
    ("zh", "en", "\u8fd9\u4e2a\u624b\u673a\u7684\u7eed\u822a\u975e\u5e38\u597d\uff0c\u80fd\u7528\u4e24\u5929\u3002"),
    ("zh", "ja", "\u4eca\u5929\u7684\u5929\u6c14\u975e\u5e38\u597d\uff0c\u9002\u5408\u51fa\u53bb\u8d70\u8d70\u3002"),
    ("ko", "zh", "\uc5ec\ub7ec\ubd84 \uc548\ub155\ud558\uc138\uc694, \uc601\uc0c1\uc744 \uc2dc\uc791\ud558\uaca0\uc2b5\ub2c8\ub2e4."),
    ("ko", "zh", "\uc624\ub298 \ub0a0\uc528\uac00 \uc815\ub9d0 \uc88b\ub124\uc694."),
]


# --------------------------------------------------------------------- tokenizer

class Tokenizer:
    """读 Android 侧同一份 tokenizer.bin，保证 Python 与 Kotlin 行为一致。"""

    def __init__(self, path: Path):
        buf = path.read_bytes()
        if buf[:8] != MAGIC:
            sys.exit(f"magic 不对: {buf[:8]!r}")
        off = 8
        (vocab_size, merge_count, unk, eos, bos, pad, prefix_count) = np.frombuffer(
            buf, dtype="<i4", count=7, offset=off
        )
        off += 28 + 4 * int(prefix_count)

        vocab = []
        for _ in range(int(vocab_size)):
            # 必须显式 int()：np.uint16 标量做 += 会溢出成 numpy 标量而非 Python int
            ln = int(np.frombuffer(buf, dtype="<u2", count=1, offset=off)[0])
            off += 2
            vocab.append(buf[off:off + ln].decode("utf-8", errors="replace"))
            off += ln

        merges = np.frombuffer(buf, dtype="<i4", count=int(merge_count) * 2, offset=off).reshape(-1, 2)
        off += int(merge_count) * 8

        self.vocab = vocab
        self.unk, self.eos, self.bos, self.pad = int(unk), int(eos), int(bos), int(pad)
        self.index = {}
        for i, s in enumerate(vocab):
            self.index.setdefault(s, i)
        # 与 Kotlin 一致：pair -> rank，保留最早出现的 rank
        self.rank = {}
        for i, (a, b) in enumerate(merges):
            self.rank.setdefault((int(a), int(b)), i)
        self.trailing = len(buf) - off

    def _bpe(self, spm: str) -> list[int]:
        if not spm:
            return []
        syms = list(spm)
        while len(syms) > 1:
            best_rank, best_at = None, -1
            for i in range(len(syms) - 1):
                merged = self.index.get(syms[i] + syms[i + 1])
                if merged is None:
                    continue
                r = self.rank.get((self.index[syms[i]], self.index[syms[i + 1]]))
                if r is None:
                    continue
                if best_rank is None or r < best_rank:
                    best_rank, best_at = r, i
            if best_at < 0:
                break
            syms[best_at] = syms[best_at] + syms[best_at + 1]
            del syms[best_at + 1]
        return [self.index.get(s, self.unk) for s in syms]

    def encode(self, text: str, src_flores: str) -> list[int]:
        ids = []
        si = self.index.get(src_flores)
        if si is not None:
            ids.append(si)
        filtered = "".join(" " if c in "\n\r\t" else c for c in text)
        import unicodedata

        filtered = unicodedata.normalize("NFC", filtered)
        spm = SPACE + filtered.replace(" ", SPACE)
        ids.extend(self._bpe(spm))
        ids.append(self.eos)
        return ids

    def decode(self, ids) -> str:
        out = []
        for i in ids:
            if i == self.eos:
                break
            if i in (self.bos, self.pad):
                continue
            if 0 <= i < len(self.vocab):
                out.append(self.vocab[i])
        s = "".join(out)
        # ▁ 是 Metaspace 的空格标记。整串只由 ▁ 组成时它是空串，不能当成内容参与 strip。
        # （曾经因为先 strip 再 replace，导致句子开头的 ▁ 被吃掉，译文首字粘连或丢字。）
        return s.replace(SPACE, " ").strip()


# ----------------------------------------------------------------------- engine

class Translator:
    def __init__(self, threads: int = 4):
        so = ort.SessionOptions()
        so.intra_op_num_threads = threads
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        t0 = time.perf_counter()
        self.enc = ort.InferenceSession(str(ENCODER), so, providers=["CPUExecutionProvider"])
        self.dec = ort.InferenceSession(str(DECODER), so, providers=["CPUExecutionProvider"])
        self.load_ms = (time.perf_counter() - t0) * 1000
        self.tok = Tokenizer(TOKENIZER_BIN)

        self.dec_inputs = [i.name for i in self.dec.get_inputs()]
        self.past_names = [n for n in self.dec_inputs if n.startswith("past_key_values.")]
        self.present_to_past = {n.replace("past_key_values.", "present."): n for n in self.past_names}

    def _decoder_shapes(self):
        return {i.name: i.shape for i in self.dec.get_inputs()}

    def translate(self, text: str, src: str, tgt: str, max_new: int = 128, debug: bool = False):
        src_flores, _ = FLORES[src]
        _, tgt_id = FLORES[tgt]

        t0 = time.perf_counter()
        ids = self.tok.encode(text, src_flores)
        enc_ids = np.array([ids], dtype=np.int64)
        enc_mask = np.ones_like(enc_ids)
        enc_out = self.enc.run(None, {"input_ids": enc_ids, "attention_mask": enc_mask})
        hidden = enc_out[0]                      # [1, S, 1024]
        t_encode = time.perf_counter() - t0

        S = hidden.shape[1]
        # 关键：独立 decoder **没有** past_key_values 输入（实测只有 3 个输入），
        # 所以不能用「只喂最后一个 token」的写法——那样 decoder 的位置编码每步都从 0
        # 重来，输出必然退化成 "is is is..."。
        # 唯一语义正确的做法是每步重喂**完整前缀**（等价于无缓存的完整前向）。
        prefix = [tgt_id]                        # 强制起始 token = 目标语种码
        generated = []
        t_first = None
        t0 = time.perf_counter()

        for step in range(max_new):
            cur = np.array([prefix], dtype=np.int64)
            feed = {
                "input_ids": cur,
                "encoder_hidden_states": hidden,
                "encoder_attention_mask": enc_mask,
            }
            t_step = time.perf_counter()
            out = self.dec.run(None, feed)
            if step == 0:
                t_first = time.perf_counter() - t0
            names = [o.name for o in self.dec.get_outputs()]
            res = dict(zip(names, out))
            logits = res["logits"]
            nxt = int(np.argmax(logits[0, -1, :]))
            if debug and step < 6:
                print(f"      step{step} prefix_len={len(prefix)} -> {nxt} "
                      f"({self.tok.vocab[nxt]!r})  step_ms={(time.perf_counter()-t_step)*1000:.0f}")
            if nxt == self.tok.eos:
                break
            generated.append(nxt)
            prefix.append(nxt)

        t_decode = time.perf_counter() - t0
        return {
            "text": self.tok.decode(generated),
            "ids": generated,
            "encode_ms": t_encode * 1000,
            "first_token_ms": (t_first or 0) * 1000,
            "decode_ms": t_decode * 1000,
            "total_ms": (t_encode + t_decode) * 1000,
            "n_tokens": len(generated),
            "src_len": S,
        }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--max-new", type=int, default=128)
    ap.add_argument("--debug", action="store_true")
    ap.add_argument("--only", default="", help="只跑第几条（从 1 开始）")
    ap.add_argument("--out", default=str(ROOT / "tools" / "nllb_reference.json"))
    args = ap.parse_args()

    for p in (ENCODER, DECODER):
        if not p.exists():
            sys.exit(f"缺少模型文件：{p}")
    if DECODER.name == "decoder_model_merged_quantized.onnx":
        sys.exit("本脚本必须用独立 decoder，不是 merged 变体")

    print(f"encoder : {ENCODER.name}  {ENCODER.stat().st_size:,} 字节")
    print(f"decoder : {DECODER.name}  {DECODER.stat().st_size:,} 字节")
    tr = Translator(threads=args.threads)
    print(f"加载耗时 {tr.load_ms:.0f} ms")
    print(f"decoder 输入 {len(tr.dec_inputs)} 个（其中 past_* {len(tr.past_names)} 个）")
    print(f"tokenizer: vocab={len(tr.tok.vocab):,}  尾部多余字节={tr.tok.trailing}")
    print()

    results = []
    ok = 0
    for i, (src, tgt, text) in enumerate(SAMPLES, 1):
        if args.only and str(i) not in args.only.split(","):
            continue
        print(f"[{i}] {src}->{tgt}  {text}")
        try:
            r = tr.translate(text, src, tgt, max_new=args.max_new, debug=args.debug)
            ok += 1
            print(f"    -> {r['text']}")
            print(f"       首字 {r['first_token_ms']:.0f} ms | 编码 {r['encode_ms']:.0f} ms | "
                  f"解码 {r['decode_ms']:.0f} ms | 共 {r['total_ms']:.0f} ms | "
                  f"{r['n_tokens']} tok | 源长 {r['src_len']}")
            results.append({"i": i, "src": src, "tgt": tgt, "input": text, **r})
        except Exception as e:
            print(f"    ❌ 失败: {type(e).__name__}: {str(e)[:400]}")
            results.append({"i": i, "src": src, "tgt": tgt, "input": text, "error": str(e)[:800]})
        print()

    print(f"成功 {ok}/{len(results)}")
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"decoder": DECODER.name, "threads": args.threads, "results": results},
                  f, ensure_ascii=False, indent=2)
    print(f"参考结果已写出 {args.out}")


if __name__ == "__main__":
    main()
