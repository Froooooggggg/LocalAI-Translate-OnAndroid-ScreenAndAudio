#!/usr/bin/env python3
"""
在开发机上验证 NLLB 端侧翻译链路，**不必装到手机**。

意义：Android 那边一旦跑不通，很难分清是「分词错了」「shape 错了」还是
「ONNX 导出有问题」。这里先把同一批权重按 Android 会用的方式跑一遍，
把数值/分词层面的问题在开发机上解决掉。

同时实现一份与 Kotlin `NllbTokenizer` 完全同构的 BPE，
用来交叉验证两端分词结果一致（这是最容易出错的一环）。

用法:
  python tools/verify_nllb.py                # 全部
  python tools/verify_nllb.py --tokenizer    # 只验分词
  python tools/verify_nllb.py --translate    # 只验翻译
"""
from __future__ import annotations

import argparse
import json
import struct
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
NLLB = ROOT / "third_party" / "nllb"
TOKENIZER_BIN = ROOT / "app" / "src" / "main" / "assets" / "nllb" / "tokenizer.bin"
ENCODER = NLLB / "onnx" / "encoder_model_int8.onnx"
DECODER = NLLB / "onnx" / "decoder_model_merged_quantized.onnx"

SPACE = "\u2581"
EOS_ID, BOS_ID, PAD_ID = 2, 0, 1

LANG = {
    "zh": ("zho_Hans", 256200),
    "en": ("eng_Latn", 256047),
    "ja": ("jpn_Jpan", 256079),
    "ko": ("kor_Hang", 256098),
}


# --------------------------------------------------------------------- 分词器

class Tok:
    """读 build_nllb_tokenizer.py 的产物，语义与 Kotlin 版一致。"""

    def __init__(self, path: Path):
        buf = path.read_bytes()
        assert buf[:8] == b"SNATKN02", f"magic 不对：{buf[:8]!r}（需要重新跑 build_nllb_tokenizer.py）"
        off = 8
        vocab_size, merge_count = struct.unpack_from("<II", buf, off); off += 8
        self.unk, self.eos, self.bos, self.pad = struct.unpack_from("<iiii", buf, off); off += 16
        (prefix_count,) = struct.unpack_from("<I", buf, off); off += 4
        off += 4 * prefix_count

        self.vocab: list[str] = []
        # key 必须是 UTF-8 **字节串**：BPE 的很多中间符号是半个多字节字符
        # （比如 \xe3\x81 这种非法 UTF-8 前缀），用 str 做 key 永远查不到。
        self.index: dict[bytes, int] = {}
        for i in range(vocab_size):
            (n,) = struct.unpack_from("<H", buf, off); off += 2
            raw = buf[off:off + n]; off += n
            self.vocab.append(raw.decode("utf-8", errors="replace"))
            self.index[raw] = i

        # merge 的权威表示是「左半 id + 右半 id -> rank」。
        # rank 小的先合并，所以从后往前填，保证同 pair 出现多次时取最早的 rank。
        self.pair_rank: dict[tuple[int, int], int] = {}
        for i in range(merge_count):
            a, b = struct.unpack_from("<II", buf, off); off += 8
            self.pair_rank.setdefault((a, b), i)

    def bpe(self, spm_text: str) -> list[int]:
        syms = [self.index.get(c.encode("utf-8"), self.unk) for c in spm_text]
        while len(syms) > 1:
            best_rank, best_at = 1 << 30, -1
            for i in range(len(syms) - 1):
                r = self.pair_rank.get((syms[i], syms[i + 1]))
                if r is not None and r < best_rank:
                    best_rank, best_at = r, i
            if best_at < 0:
                break
            merged = self.index.get(
                (self.vocab[syms[best_at]] + self.vocab[syms[best_at + 1]]).encode("utf-8")
            )
            if merged is None:
                break
            syms[best_at] = merged
            del syms[best_at + 1]
        return syms

    def encode(self, text: str, src: str) -> list[int]:
        flores, _ = LANG[src]
        ids = [self.index[flores.encode()]]
        spm = SPACE + text.replace(" ", SPACE)
        ids += self.bpe(spm)
        ids.append(self.eos)
        return ids

    def decode(self, ids) -> str:
        out = []
        for i in ids:
            i = int(i)
            if i == self.eos:
                break
            if i in (self.bos, self.pad):
                continue
            if 0 <= i < len(self.vocab):
                out.append(self.vocab[i])
        s = "".join(out).replace(SPACE, " ")
        return s[1:].strip() if s.startswith(" ") else s.strip()


# --------------------------------------------------------------------- 参考实现
# 直接按官方 tokenizer.json 的「word + word」merge 表重推一遍，
# 用来确认我们的二进制转换没有丢掉任何一条 merge。

def reference_encode(tok_json_path: Path, text: str) -> list[int]:
    raw = json.loads(tok_json_path.read_text(encoding="utf-8"))
    vocab: dict[str, int] = raw["model"]["vocab"]
    merges = raw["model"]["merges"]

    # merge 表的权威键是「左半 token 文本 + 空格 + 右半 token 文本」。
    # 用 setdefault 保留最早的 rank（rank 小的先合并）。
    rank: dict[str, int] = {}
    for i, m in enumerate(merges):
        rank.setdefault(m, i)

    syms = [SPACE] + [c for c in text.replace(" ", SPACE)]
    while len(syms) > 1:
        best_rank, best_at = 1 << 30, -1
        for i in range(len(syms) - 1):
            r = rank.get(f"{syms[i]} {syms[i + 1]}")
            if r is not None and r < best_rank:
                best_rank, best_at = r, i
        if best_at < 0:
            break
        syms[best_at] = syms[best_at] + syms[best_at + 1]
        del syms[best_at + 1]
    unk = vocab.get("<unk>", 3)
    return [vocab.get(s, unk) for s in syms]


# --------------------------------------------------------------------- 翻译

class NllbOnnx:
    """
    decoder_model_merged 把「首步无 cache」与「后续有 cache」两条分支合并进一个
    ONNX 图，用 `use_cache_branch` 切换。踩坑记录（实测）：

      首步若把 use_cache_branch 设为 false 且 encoder 侧 past 长度给 0，会报
          Reshape node '/model/decoder/layers.0/encoder_attn/Reshape_4'
          The dimension with value zero exceeds the dimension size of the input tensor.
      原因是跨注意力的 Reshape 不接受长度 0 的 encoder past。

      正确做法（与 transformers.js 一致）：
        首步 use_cache_branch = **true**，encoder 侧 past 按**真实编码器长度**分配
        全零占位，decoder 侧 past 长度为 0；该步输出（含占位污染）直接丢弃，
        从第二步起用真实 present 喂回。

    另外：所有 past_* 与 use_cache_branch 每步都必须给全，否则报
    "Required inputs ... are missing from input feed"。
    """

    def __init__(self):
        import onnxruntime as ort
        opts = ort.SessionOptions()
        opts.log_severity_level = 3
        self.enc = ort.InferenceSession(str(ENCODER), opts, providers=["CPUExecutionProvider"])
        self.dec = ort.InferenceSession(str(DECODER), opts, providers=["CPUExecutionProvider"])
        self.tok = Tok(TOKENIZER_BIN)

        self.enc_inputs = {i.name: i for i in self.enc.get_inputs()}
        self.dec_inputs = {i.name: i for i in self.dec.get_inputs()}
        self.dec_outputs = [o.name for o in self.dec.get_outputs()]

        # past 输入名 -> present 输出名（唯一区别是前缀）
        self.past_names = [n for n in self.dec_inputs if n.startswith("past_key_values.")]
        self.past_to_present = {
            n: n.replace("past_key_values.", "present.") for n in self.past_names
        }
        print(f"encoder 输入: {list(self.enc_inputs)}")
        print(f"decoder 输入 {len(self.dec_inputs)} 个，其中 past {len(self.past_names)} 个；"
              f"输出 {len(self.dec_outputs)} 个")

    def _placeholder_past(self, enc_len: int) -> dict:
        """首步用的占位 past。

        encoder 侧必须是**真实编码器长度**（否则跨注意力 Reshape 报零长度错误），
        decoder 侧为长度 0。配合 use_cache_branch=True，本步输出会被丢弃。
        """
        out = {}
        for name in self.past_names:
            t = self.dec_inputs[name].type          # 形如 "tensor(float)"
            np_dtype = np.float32 if "float" in t else np.int64
            length = 0 if ".decoder." in name else enc_len
            out[name] = np.zeros((1, 16, length, 64), dtype=np_dtype)
        out["use_cache_branch"] = np.ones((1,), dtype=np.bool_)
        return out

    def translate(self, text: str, src: str, tgt: str, max_new: int = 128, debug: bool = False):
        t0 = time.perf_counter()
        ids = self.tok.encode(text, src)
        src_t = np.array([ids], dtype=np.int64)
        mask = np.ones_like(src_t)

        enc_feed = {k: v for k, v in {"input_ids": src_t, "attention_mask": mask}.items()
                    if k in self.enc_inputs}
        hidden = self.enc.run(None, enc_feed)[0]
        t_enc = time.perf_counter()

        _, start_id = LANG[tgt]
        cur = np.array([[start_id]], dtype=np.int64)
        past = self._placeholder_past(int(hidden.shape[1]))
        generated: list[int] = []
        steps: list[float] = []

        for step in range(max_new):
            t_step = time.perf_counter()
            feed = dict(past)
            feed["input_ids"] = cur
            feed["encoder_hidden_states"] = hidden
            if "encoder_attention_mask" in self.dec_inputs:
                feed["encoder_attention_mask"] = mask
            feed = {k: v for k, v in feed.items() if k in self.dec_inputs}

            outs = self.dec.run(None, feed)
            by_name = dict(zip(self.dec_outputs, outs))
            nxt = int(np.argmax(by_name["logits"][0, -1]))
            steps.append((time.perf_counter() - t_step) * 1000)

            if nxt == EOS_ID:
                break
            generated.append(nxt)
            cur = np.array([[nxt]], dtype=np.int64)

            # 喂回上一步的 present，并切到 cache 分支
            past = {name: by_name[pres] for name, pres in self.past_to_present.items()}
            past["use_cache_branch"] = np.ones((1,), dtype=np.bool_)
            if debug and step < 3:
                print(f"      step{step}: token={nxt} "
                      f"({self.tok.vocab[nxt]!r}) logits{by_name['logits'].shape}")

        t_gen = time.perf_counter()
        return self.tok.decode(generated), {
            "src_tokens": len(ids),
            "out_tokens": len(generated),
            "encode_ms": (t_enc - t0) * 1000,
            "decode_ms": (t_gen - t_enc) * 1000,
            "total_ms": (t_gen - t0) * 1000,
            "first_step_ms": steps[0] if steps else 0.0,
            "steady_step_ms": (sum(steps[1:]) / len(steps[1:])) if len(steps) > 1 else 0.0,
        }


# --------------------------------------------------------------------- 主流程

SAMPLES = [
    ("en", "zh", "The new model runs entirely on your phone, so nothing is uploaded."),
    ("en", "zh", "Latency under one second is the hard requirement here."),
    ("en", "zh", "Some apps block audio capture, and there is no legal way around it."),
    ("ja", "zh", "この機能はスマートフォンだけで動作します。"),
    ("ja", "zh", "字幕の遅延は一秒以内に抑える必要があります。"),
    ("ko", "zh", "이 기능은 스마트폰에서만 동작합니다."),
    ("ko", "zh", "자막 지연은 1초 이내로 줄여야 합니다."),
    ("zh", "en", "这个功能完全在手机上运行。"),
    ("zh", "en", "字幕延迟必须控制在一秒以内。"),
    ("zh", "ja", "这个功能完全在手机上运行。"),
]


def cmd_tokenizer() -> None:
    print("=" * 72)
    print("分词器交叉验证：本地 BPE  vs  HuggingFace tokenizer.json 的参考实现")
    print("=" * 72)
    tok = Tok(TOKENIZER_BIN)
    ref_json = NLLB / "tokenizer.json"

    cases = [
        "hello world",
        "Latency under one second is the hard requirement here.",
        "这个功能完全在手机上运行。",
        "字幕の遅延は一秒以内に抑える必要があります。",
        "이 기능은 스마트폰에서만 동작합니다.",
        "Some apps block audio capture — no legal way around it!",
        "OK", "  多空格   测试  ",
    ]
    all_ok = True
    for text in cases:
        mine = tok.bpe(SPACE + text.replace(" ", SPACE))
        ref = reference_encode(ref_json, text)
        ok = mine == ref
        all_ok &= ok
        mark = "一致" if ok else "**不一致**"
        print(f"  [{mark}] {text[:44]!r}")
        print(f"       我们的 {mine[:14]}{'…' if len(mine) > 14 else ''}")
        if not ok:
            print(f"       参考的 {ref[:14]}{'…' if len(ref) > 14 else ''}")
            print(f"       我们的还原: {tok.decode(mine)!r}")
            print(f"       参考的还原: {tok.decode(ref)!r}")
    print()
    print(f"分词往返（decode(encode)）自检：")
    for text in cases:
        ids = tok.encode(text.strip(), "en")
        back = tok.decode(ids[1:])
        same = back == text.strip()
        print(f"  [{'一致' if same else '有出入'}] {text.strip()[:40]!r} -> {back!r}")
    print()
    print("结论：" + ("分词与官方实现完全一致。" if all_ok else "**存在不一致，必须先修好。**"))


def cmd_translate() -> None:
    print("=" * 72)
    print("NLLB-200-distilled-600M 端侧翻译验证（int8 encoder + quantized decoder）")
    print("=" * 72)
    m = NllbOnnx()
    print()
    ok = 0
    for src, tgt, text in SAMPLES:
        try:
            out, st = m.translate(text, src, tgt)
            got = bool(out.strip())
            ok += got
            print(f"[{src}->{tgt}] {text}")
            print(f"          => {out}")
            print(f"          {st['src_tokens']} 源 token → {st['out_tokens']} 输出 token | "
                  f"编码 {st['encode_ms']:.0f}ms 解码 {st['decode_ms']:.0f}ms 合计 {st['total_ms']:.0f}ms")
            print(f"          解码首步 {st['first_step_ms']:.0f}ms，稳定步 {st['steady_step_ms']:.0f}ms/token")
        except Exception as e:
            print(f"[{src}->{tgt}] {text}\n          !! 失败: {type(e).__name__}: {e}")
        print()
    print(f"成功 {ok}/{len(SAMPLES)}")
    print()
    print("注：以上耗时是开发机 CPU。手机端（骁龙 8 Elite）预计更接近实时，"
          "但必须以真机实测为准。")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer", action="store_true")
    ap.add_argument("--translate", action="store_true")
    a = ap.parse_args()
    if not a.tokenizer and not a.translate:
        cmd_tokenizer()
        print()
        cmd_translate()
    else:
        if a.tokenizer:
            cmd_tokenizer()
        if a.translate:
            cmd_translate()


if __name__ == "__main__":
    main()
