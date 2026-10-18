#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
检查 NLLB 词表对**标点符号**的覆盖情况。

动机：真机回归时发现全角逗号 `，` 不在词表里（编码成 unk=3、解码得空串）。
字幕文本里全角标点极其常见，如果它们普遍缺失，就必须在送进模型前做标点归一化，
否则译文会以"整句看不出错、但语气全丢"的方式劣化 —— 这种劣化很难被发现。

用法：
    python tools/check_punct_coverage.py
"""
import os
import struct

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BIN = os.path.join(ROOT, "app", "src", "main", "assets", "nllb", "tokenizer.bin")

GROUPS = {
    "中文全角标点": "，。、；：？！“”‘’（）《》〈〉【】〔〕—…·　",
    "ASCII 标点": "(),.;:?!\"'-/_&%$#@*+=<>[]{}|\\~`^",
    "日文标点": "、。「」『』〜・",
    "其他常见符号": "→←↑↓★☆●○◆◇■□▲△",
}


def main() -> int:
    with open(BIN, "rb") as f:
        d = f.read()
    off = 8
    vocab_size, merge_count = struct.unpack_from("<II", d, off)
    off += 8 + 16                      # 跳过 4 个特殊 id
    (prefix_count,) = struct.unpack_from("<I", d, off)
    off += 4 + 4 * prefix_count

    token_bytes = []
    for _ in range(vocab_size):
        (n,) = struct.unpack_from("<H", d, off)
        off += 2
        token_bytes.append(d[off:off + n])
        off += n

    by_bytes = {}
    for i, b in enumerate(token_bytes):
        by_bytes.setdefault(b, i)

    sm = "\u2581".encode("utf-8")      # ▁

    def lookup(ch: str):
        b = ch.encode("utf-8")
        if b in by_bytes:
            return by_bytes[b]
        return by_bytes.get(sm + b)    # 带 Metaspace 前缀的变体（句首常见）

    print(f"词表 {vocab_size} 条  merge {merge_count} 条\n")
    print(f"{'类别':<14}{'覆盖':>6}{'总数':>6}   缺失的字符")
    print("-" * 74)
    for name, chars in GROUPS.items():
        miss = [c for c in chars if lookup(c) is None]
        print(f"{name:<14}{len(chars) - len(miss):>6}{len(chars):>6}   {''.join(miss)}")

    print("\n中文标点逐字明细：")
    for c in GROUPS["中文全角标点"]:
        v = lookup(c)
        mark = f"id={v}" if v is not None else "❌ 不在词表（编码成 unk，解码为空）"
        print(f"  {c!r:>8}  U+{ord(c):04X}  {mark}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
