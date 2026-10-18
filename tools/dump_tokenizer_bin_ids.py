#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
从编译好的 tokenizer.bin（Android 运行时真正读取的那一份）里解出特殊 token id。

存在的理由：`Lang.kt` 里硬编码了 NLLB 的语种 token id，而 `NllbTokenizer` 运行时
是从 tokenizer.bin 里按**字符串**查 id 的（encode() 走 vocabIndexOf(sourceFlores)）。
如果两边不一致，就会出现「运行时能编对、但其他地方用错 id」的隐蔽 bug。
所以必须以 tokenizer.bin 为准做一次交叉校验。

格式（与 app/src/main/java/com/screenast/app/translate/nllb/NllbTokenizer.kt 对齐）：
    char[8]  magic "SNATKN02"
    int32    vocabSize
    int32    mergeCount
    int32    unkId / eosId / bosId / padId
    int32    prefixCount, 然后 prefixCount 个 int32
    重复 vocabSize 次: uint16 len + len 字节 UTF-8
    重复 mergeCount 次: int32 a + int32 b
"""
import struct
import sys
from pathlib import Path
import os
import json
# 仓库根（tools/ 的上一级）：脚本可以放在任何机器上跑
_R = Path(__file__).resolve().parents[1]


BIN = str(_R / 'app/src/main/assets/nllb/tokenizer.bin')
JSON_PATH = str(_R / '模型/上游/nllb/tokenizer.json')

WANT = ["zho_Hans", "eng_Latn", "jpn_Jpan", "kor_Hang", "rus_Cyrl", "zho_Hant"]


def read_bin(path):
    with open(path, "rb") as f:
        magic = f.read(8)
        if magic != b"SNATKN02":
            sys.exit(f"magic 不对：{magic!r}，期望 b'SNATKN02'")
        vocab_size, merge_count, unk, eos, bos, pad, prefix_count = struct.unpack("<7i", f.read(28))
        f.read(4 * prefix_count)

        vocab = []
        for _ in range(vocab_size):
            (ln,) = struct.unpack("<H", f.read(2))
            raw = f.read(ln)
            vocab.append(raw.decode("utf-8", errors="replace"))

        merges = []
        for _ in range(merge_count):
            a, b = struct.unpack("<2i", f.read(8))
            merges.append((a, b))

        trailing = f.read()
    return dict(
        vocab=vocab, merges=merges, unk=unk, eos=eos, bos=bos, pad=pad,
        trailing=len(trailing), merge_count_header=merge_count,
    )


def main():
    print(f"读取 {BIN}")
    print(f"  文件大小 {os.path.getsize(BIN):,} 字节")
    d = read_bin(BIN)

    vocab, merges = d["vocab"], d["merges"]
    print(f"  vocabSize={len(vocab):,}  mergeCount={len(merges):,}  "
          f"尾部多余字节={d['trailing']}")
    print(f"  unk={d['unk']}  eos={d['eos']}  bos={d['bos']}  pad={d['pad']}")
    print()

    # 建立字符串 → id 的映射（与 Kotlin 的 tokenIndex 完全一致）
    idx = {}
    for i, s in enumerate(vocab):
        idx.setdefault(s, i)
    dup = len(vocab) - len(idx)

    print("=== tokenizer.bin 里的特殊 token（权威）===")
    ids = {}
    for w in WANT:
        i = idx.get(w)
        if i is None:
            print(f"  {w:<10} ❌ 不在词表里")
        else:
            ids[w] = i
            print(f"  {w:<10} id={i}")
    print()

    print("=== Lang.kt 当前硬编码值 vs tokenizer.bin ===")
    # Lang.kt 现值（zh/en/ja/ko）
    kotlin_current = {"zho_Hans": 56887, "eng_Latn": 256, "jpn_Jpan": 55560, "kor_Hang": 56316}
    allmatch = True
    for w, cur in kotlin_current.items():
        real = ids.get(w)
        ok = (real == cur)
        allmatch &= bool(ok)
        print(f"  {w:<10} Kotlin={cur:<8} bin={real}  {'✅ 一致' if ok else '❌ 不一致 → 必须改'}")
    print()
    print(f"  结论：{'全部一致' if allmatch else '存在不一致，需修正 Lang.kt'}")
    print()

    # 用 HF 的 tokenizer.json 做第三方交叉校验
    if os.path.exists(JSON_PATH):
        print("=== 与 HF tokenizer.json 的 added_tokens 交叉校验 ===")
        with open(JSON_PATH, encoding="utf-8") as f:
            j = json.load(f)
        hf = {t["content"]: t["id"] for t in j["added_tokens"] if t.get("special")}
        for w in WANT:
            print(f"  {w:<10} bin={ids.get(w)}  hf_json={hf.get(w)}  "
                  f"{'✅' if ids.get(w) == hf.get(w) else '❌'}")
        print()
        print(f"  tokenizer.json 的 special token 总数 = {len(hf)}")
        print(f"  bin 词表里的重复字符串条目 = {dup}")
    else:
        print(f"（未找到 {JSON_PATH}，跳过交叉校验）")

    # 供后续脚本 import 使用
    out = {
        "special": ids,
        "vocab_size": len(vocab),
        "merge_count": len(merges),
        "unk": d["unk"], "eos": d["eos"], "bos": d["bos"], "pad": d["pad"],
    }
    side = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tokenizer_bin_ids.json")
    with open(side, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"\n已写出 {side}")


if __name__ == "__main__":
    main()
