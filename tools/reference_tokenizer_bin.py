#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
纯 Python 参考实现：直接读 `app/src/main/assets/nllb/tokenizer.bin`，
对一批样例给出【参考 token id】与【参考解码结果】，供 Android 侧自检对照。

为什么不用 HuggingFace：
  这台机器上没装 `tokenizers`/`transformers`，而这个环境**不联网装包**。
  好在 BPE + Metaspace 的算法本身很短，且已经用官方 `tokenizer.json` 交叉验证过
  （见 build_nllb_tokenizer.py 的注释与历史验证：'hello world' -> [133863, 15697]）。
  本脚本读的是**同一个** tokenizer.bin，所以它同时也是
  「Android 侧二进制解析是否正确」的对照物。

它专门要证明的一件事：
  **`频道` 这类汉字在 BPE 里会被切成跨 token 的半字符**（如 `E9 A2` + `91`），
  所以解码必须先把所有 token 的字节拼起来、最后再一次性 UTF-8 解码。
  旧 Kotlin 实现逐个 token 解码 → 半个汉字变 U+FFFD → 真机译文里冒出 `<unk>`。
  本脚本的 `decode()` 就是正确实现的参考（= HuggingFace Metaspace decoder 的行为）。

用法：
    python tools/reference_tokenizer_bin.py
    python tools/reference_tokenizer_bin.py --json tools/reference_tokenids.json
"""
import argparse
import json
import os
import struct
import sys
import unicodedata

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BIN = os.path.join(ROOT, "app", "src", "main", "assets", "nllb", "tokenizer.bin")

SP = "\u2581"          # Metaspace 的空格标记
SP_BYTES = SP.encode("utf-8")   # e2 96 81

CASES = [
    "Hello everyone, welcome back to the channel.",
    "您好，欢迎回到频道。",
    "频道",
    "这个功能完全在手机上运行。",
    "今天天气真好。",
    "こんにちは、世界。",
    "안녕하세요 세계",
    "This phone is very durable and can last for two days.",
    "NLLB-200 支持 200 种语言。",
]


class Tok:
    def __init__(self, path):
        with open(path, "rb") as f:
            data = f.read()
        if data[:8] != b"SNATKN02":
            raise SystemExit(f"magic 不对：{data[:8]!r}")
        off = 8
        (self.vocab_size, self.merge_count) = struct.unpack_from("<II", data, off)
        off += 8
        (self.unk, self.eos, self.bos, self.pad) = struct.unpack_from("<iiii", data, off)
        off += 16
        (prefix_count,) = struct.unpack_from("<I", data, off)
        off += 4 + 4 * prefix_count

        # 词表：id -> 原始字节
        self.id2b = []
        for _ in range(self.vocab_size):
            (n,) = struct.unpack_from("<H", data, off)
            off += 2
            self.id2b.append(data[off:off + n])
            off += n
        # merge：(左 id, 右 id) -> rank
        self.rank = {}
        for i in range(self.merge_count):
            a, b = struct.unpack_from("<II", data, off)
            off += 8
            self.rank.setdefault((a, b), i)
        self.trailing = len(data) - off

        self.b2id = {}
        for i, b in enumerate(self.id2b):
            self.b2id.setdefault(b, i)

        # 语种标签 token（与 Kotlin 侧同一套判定）
        self.lang_ids = set()
        for i, b in enumerate(self.id2b):
            t = b.decode("utf-8", "replace")
            if t.startswith(SP):
                t = t[1:]
            if (len(t) == 8 and t[3] == "_"
                    and t[0:3].islower() and t[4].isupper() and t[5:8].islower()):
                self.lang_ids.add(i)

    # ---------------------------------------------------------------- 编码
    def _metaspace(self, text):
        """空格 → ▁，并在开头补 ▁（add_prefix_space=true）。"""
        return SP + text.replace(" ", SP)

    def _initial_symbols(self, s):
        """按 **Unicode 字符**切初始符号。

        ⚠️ 这里踩过一次：想当然写成「按 UTF-8 字节切」，结果所有中文都变成连续
        的 `<unk>`（实测 '频道' -> [3,3,3,3,3,3,3,3,3]）。原因是 NLLB 的 vocab
        里**绝大部分是整字符** token，单字节符号（如 E9、A2）大多不在表里，
        BPE 第一步就无 pair 可合并，直接落成 unk。
        SentencePiece 的 byte fallback 是"未登录字符的兜底"，不是"切分粒度"。
        """
        return list(s)

    def _merge(self, a: str, b: str) -> str:
        return a + b

    def _bpe(self, symbols):
        symbols = list(symbols)
        while len(symbols) > 1:
            best, best_rank = None, None
            for i in range(len(symbols) - 1):
                a = self.b2id.get(self._sym_bytes(symbols[i]))
                b = self.b2id.get(self._sym_bytes(symbols[i + 1]))
                if a is None or b is None:
                    continue
                r = self.rank.get((a, b))
                if r is not None and (best_rank is None or r < best_rank):
                    best, best_rank = i, r
            if best is None:
                break
            symbols[best:best + 2] = [self._merge(symbols[best], symbols[best + 1])]
        return [self.b2id.get(self._sym_bytes(s), self.unk) for s in symbols]

    @staticmethod
    def _sym_bytes(sym):
        return sym.encode("utf-8")

    def encode_plain(self, text):
        text = unicodedata.normalize("NFC", text)
        return self._bpe(self._initial_symbols(self._metaspace(text)))

    def encode(self, text, src_flores=""):
        ids = []
        if src_flores:
            t = self.b2id.get(src_flores.encode("utf-8"))
            if t is None:
                t = self.b2id.get((SP + src_flores).encode("utf-8"))
            if t is not None:
                ids.append(t)
        ids.extend(self.encode_plain(text))
        return ids

    # ---------------------------------------------------------------- 解码
    def decode(self, ids):
        """参考实现：拼字节 → 一次性 UTF-8 解码（HuggingFace Metaspace decoder 行为）。"""
        out = bytearray()
        for i in ids:
            if i == self.eos:
                break
            if i in (self.bos, self.pad, self.unk):
                continue
            if i in self.lang_ids:
                continue
            out += self.id2b[i] if 0 <= i < len(self.id2b) else b""
        s = out.replace(SP_BYTES, b" ").decode("utf-8", "replace")
        s = s.replace("\ufffd", "")
        if s.startswith(" "):
            s = s[1:]
        return s.strip()

    def decode_each_token_wrong(self, ids):
        """**故意写错**的旧实现，用来演示 <unk> 是怎么来的（回归证据）。"""
        parts = []
        for i in ids:
            if i == self.eos:
                break
            if i in (self.bos, self.pad, self.unk) or i in self.lang_ids:
                continue
            parts.append(self.id2b[i].decode("utf-8", "replace"))
        s = "".join(parts).replace(SP, " ")
        if s.startswith(" "):
            s = s[1:]
        return s.strip()

    def pieces(self, ids):
        return [self.id2b[i].decode("utf-8", "replace") for i in ids]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default=os.path.join(ROOT, "tools", "reference_tokenids.json"))
    args = ap.parse_args()

    if not os.path.isfile(BIN):
        print(f"找不到 {BIN}", file=sys.stderr)
        return 2
    t = Tok(BIN)
    print(f"词表 {t.vocab_size} 条  merge {t.merge_count} 条  尾部多余字节 {t.trailing}")
    print(f"特殊 id: unk={t.unk} eos={t.eos} bos={t.bos} pad={t.pad}  语种 token {len(t.lang_ids)} 个")
    print(f"eng_Latn={t.b2id.get(b'eng_Latn')} zho_Hans={t.b2id.get(b'zho_Hans')}")
    print()

    out = {"eosId": t.eos, "unkId": t.unk, "cases": []}
    all_ok = True
    for text in CASES:
        ids = t.encode_plain(text)
        back = t.decode(ids)
        wrong = t.decode_each_token_wrong(ids)
        ok = back == text
        all_ok = all_ok and ok
        out["cases"].append({
            "text": text, "ids": ids, "back": back,
            "ok": ok, "wrongBack": wrong, "pieces": t.pieces(ids),
        })
        mark = "OK " if ok else "!! "
        print(f"{mark}{text!r}")
        print(f"     ids      = {ids}")
        print(f"     pieces   = {t.pieces(ids)}")
        print(f"     decode   = {back!r}")
        if wrong != back:
            print(f"     旧实现会得到 = {wrong!r}   ← 逐个 token 解码的破绽")

    with open(args.json, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print()
    print(f"全部往返一致: {all_ok}    写出 {args.json}")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
