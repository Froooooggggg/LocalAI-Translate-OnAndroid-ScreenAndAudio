#!/usr/bin/env python3
"""
把 HuggingFace 的 tokenizer.json 编译成 Android 端能直接读的紧凑二进制。

为什么需要这个：
  NLLB 用的不是 sentencepiece unigram，而是 **BPE + Metaspace**。
  Android 上没有可用的 HuggingFace tokenizers 实现，而原始的 tokenizer.json
  有 17MB（含 4MB 的预编译 charmap、大量我们不需要的元数据）。
  这里只抽出真正需要的东西，并转成可以 mmap 顺序读的定长结构。

输出格式（小端）:
  magic   : 8 字节 ASCII "SNATKN01"
  u32 vocabSize
  u32 mergesCount
  i32 unkId, eosId, bosId, padId
  u32 prefixTokenCount, 然后每个是 u32 tokenId
                          （"额外加到文本前面的语种前缀"，NLLB 用不到，留 0）
  vocab   : vocabSize 次 [u16 byteLen][UTF-8 bytes][i32 mergeRank]   mergeRank=-1 表示无
  merges  : mergesCount 次 [u32 leftId][u32 rightId]

mergeRank 就存在词表条目里，省掉解码端再建一次 map。
"""
import json
import struct
import sys
from pathlib import Path

MAGIC = b"SNATKN02"


def build(src: Path, dst: Path) -> None:
    raw = json.loads(src.read_text(encoding="utf-8"))
    model = raw["model"]
    assert model["type"] == "BPE", f"只支持 BPE，实际是 {model['type']}"

    vocab: dict[str, int] = model["vocab"]
    merges: list[str] = model["merges"]
    if merges and isinstance(merges[0], list):  # 少数导出是 [[a,b],...]
        merges = [f"{a} {b}" for a, b in merges]

    size = len(vocab)
    print(f"词表 {size} 条，merge {len(merges)} 条")

    # id -> token 文本。缺号（理论上不该有）用空串占位，解码时跳过。
    id2tok: list[str | None] = [None] * size
    for tok, i in vocab.items():
        if 0 <= i < size:
            if id2tok[i] is not None:
                raise SystemExit(f"id {i} 重复：{id2tok[i]!r} 与 {tok!r}")
            id2tok[i] = tok
    missing = [i for i, t in enumerate(id2tok) if t is None]
    if missing:
        print(f"警告：{len(missing)} 个 id 没有对应 token，前几个 {missing[:5]}")

    def tid(tok: str) -> int:
        """token 文本 -> id。-1 表示不在词表里。

        vocab 的 key 是 JSON 解码出来的 str：原始字节里非法 UTF-8 的部分已经是
        U+FFFD 了。再次 encode('utf-8') 会把 U+FFFD 写回 efbfbd 而不是原始字节，
        这正是官方 tokenizer.json 里那些 '\ufffd...' merge 的成因，无法还原。
        这类条目会匹配不上，等价于「这条 merge 不存在」——数量极少，可接受。
        """
        return vocab.get(tok, -1)

    specials = {
        "unk": vocab.get("<unk>", 3),
        "eos": vocab.get("</s>", 2),
        "bos": vocab.get("<s>", 0),
        "pad": vocab.get("<pad>", 1),
    }
    print(f"特殊 token: {specials}")

    for code in ("eng_Latn", "zho_Hans", "jpn_Jpan", "kor_Hang"):
        print(f"  {code:10} -> {vocab.get(code)}")

    # 解析 merge 表：每条是「左半 空格 右半」，存成 (a_id, b_id) 的 rank。
    # 注意 rank 必须是**对**的下标——不能拿合并后的 token 去查，那是错的。
    pairs: list[tuple[int, int]] = []
    unresolved = 0
    for m in merges:
        parts = m.split(" ")
        if len(parts) != 2:
            raise SystemExit(f"merge 格式异常: {m!r}")
        a, b = tid(parts[0]), tid(parts[1])
        if a < 0 or b < 0:
            unresolved += 1
            a = b = 0          # 占位；这类条目永远不会被匹配到
        pairs.append((a, b))
    if unresolved:
        print(f"警告：{unresolved} 条 merge 的一半不在词表里（占位为 0，不会被命中）")

    dst.parent.mkdir(parents=True, exist_ok=True)
    with dst.open("wb") as f:
        f.write(MAGIC)
        f.write(struct.pack("<II", size, len(pairs)))
        f.write(struct.pack("<iiii", specials["unk"], specials["eos"], specials["bos"], specials["pad"]))
        f.write(struct.pack("<I", 0))  # prefix tokens count

        for i in range(size):
            tok = id2tok[i] or ""
            b = tok.encode("utf-8")
            if len(b) > 0xFFFF:
                raise SystemExit(f"token 过长: {i}")
            f.write(struct.pack("<H", len(b)))
            f.write(b)

        for a, b in pairs:
            f.write(struct.pack("<II", a, b))

    total = dst.stat().st_size
    print(f"写出 {dst}  {total:,} 字节（{total/1024/1024:.2f} MB）")


def main() -> None:
    if len(sys.argv) != 3:
        print(__doc__)
        raise SystemExit(2)
    build(Path(sys.argv[1]), Path(sys.argv[2]))


if __name__ == "__main__":
    main()
