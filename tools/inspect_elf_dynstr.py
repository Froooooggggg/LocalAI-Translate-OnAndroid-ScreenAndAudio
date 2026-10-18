#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
诊断 sherpa / onnxruntime 的 .so，为「换 soname」补丁探明 ELF 结构。

要回答三个问题：
  1. `libonnxruntime.so` 这 17 字节到底出现在哪个节里（.dynstr？还是也有别处）？
  2. `.gnu.version_r` 里有没有引用库名字符串？（有的话换 soname 必须同步改，
     否则链接器会认为版本需求属于另一个库）
  3. `.dynstr` 后面紧跟什么节？决定能否「原位等长扩展」——
     如果 .dynstr 后面是 .gnu.version_r，而 .dynstr 结尾还有对齐填充，
     就能把名字加长后塞进去，其余节偏移全不变（最安全的改法）。

用法: python inspect_elf_dynstr.py <file.so> [more.so ...]
"""

import struct
import sys

SHT_STRTAB = 3
SHT_DYNAMIC = 6
SHT_GNU_verneed = 0x6FFFFFFE
SHT_GNU_versym = 0x6FFFFFFF

DT_NULL, DT_NEEDED, DT_SONAME, DT_STRTAB, DT_STRSZ = 0, 1, 14, 5, 10


def u16(b, o): return struct.unpack_from("<H", b, o)[0]
def u32(b, o): return struct.unpack_from("<I", b, o)[0]
def u64(b, o): return struct.unpack_from("<Q", b, o)[0]


def cstr(data, off):
    end = data.find(b"\0", off)
    return data[off:end].decode("utf-8", "replace")


def sections(b):
    e_shoff = u64(b, 0x28)
    e_shentsize = u16(b, 0x3A)
    e_shnum = u16(b, 0x3C)
    e_shstrndx = u16(b, 0x3E)
    if e_shoff == 0 or e_shnum == 0:
        return []
    # 先取节名字符串表（可能在扩展索引里，这里够用）
    base = e_shoff + e_shstrndx * e_shentsize
    shstr_off = u64(b, base + 0x18)
    out = []
    for i in range(e_shnum):
        o = e_shoff + i * e_shentsize
        name_off = u32(b, o)
        out.append({
            "idx": i,
            "name": cstr(b, shstr_off + name_off) if name_off else "",
            "type": u32(b, o + 4),
            "flags": u64(b, o + 8),
            "addr": u64(b, o + 0x10),
            "offset": u64(b, o + 0x18),
            "size": u64(b, o + 0x20),
            "link": u32(b, o + 0x28),
            "addralign": u64(b, o + 0x30),
        })
    return out


def dynamic_entries(b, secs):
    dyn = next((s for s in secs if s["type"] == SHT_DYNAMIC), None)
    if not dyn:
        return []
    ents = []
    for o in range(dyn["offset"], dyn["offset"] + dyn["size"], 16):
        tag = u64(b, o)
        val = u64(b, o + 8)
        ents.append((tag, val))
        if tag == DT_NULL:
            break
    return ents


def main():
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)

    for path in sys.argv[1:]:
        b = open(path, "rb").read()
        print("=" * 78)
        print("文件: %s   (%.1f MB)" % (path, len(b) / 1048576))
        print("=" * 78)

        secs = sections(b)
        dyn = dynamic_entries(b, secs)

        # ---- 1. DT_STRTAB / DT_STRSZ，以及两个关键偏移 ----
        strtab_addr = strsz = None
        soname_off = needed_offs = None
        for tag, val in dyn:
            if tag == DT_STRTAB:
                strtab_addr = val
            elif tag == DT_STRSZ:
                strsz = val
            elif tag == DT_SONAME:
                soname_off = val
            elif tag == DT_NEEDED:
                nm = cstr_region = None
                needed_offs = needed_offs or []
                needed_offs.append(val)

        dynstr = next((s for s in secs if s["type"] == SHT_STRTAB
                       and s["addr"] == strtab_addr), None)
        if dynstr:
            print("\n.dynstr  节: offset=0x%X  addr=0x%X  size=%d  addralign=%d"
                  % (dynstr["offset"], dynstr["addr"], dynstr["size"], dynstr["addralign"]))
            ds = b[dynstr["offset"]:dynstr["offset"] + dynstr["size"]]
            print("  其中 DT_SONAME 偏移 = 0x%X -> %r"
                  % (soname_off, cstr(ds, soname_off) if soname_off is not None else None))
            for no in (needed_offs or []):
                print("  DT_NEEDED 偏移 = 0x%X -> %r" % (no, cstr(ds, no)))
            # 末尾填充
            last = ds.rfind(b"\0")
            tail = ds[last + 1:]
            print("  末尾 NUL 之后的填充: %d 字节  %r" % (len(tail), tail[:32]))
        else:
            print("\n!! 找不到 .dynstr")

        # ---- 2. .dynstr 的邻居（决定能否原位扩展）----
        print("\n节布局（按 offset 排序，只列 .dynstr 前后各 3 个）:")
        ordered = sorted([s for s in secs if s["size"] > 0], key=lambda s: s["offset"])
        pos = next((i for i, s in enumerate(ordered)
                    if s["type"] == SHT_STRTAB and s["addr"] == strtab_addr), None)
        if pos is not None:
            for s in ordered[max(0, pos - 3): pos + 4]:
                mark = "  <== .dynstr" if (s["type"] == SHT_STRTAB and s["addr"] == strtab_addr) else ""
                print("  %-22s offset=0x%-9X size=%-8d align=%d%s"
                      % (s["name"], s["offset"], s["size"], s["addralign"], mark))
            nxt = ordered[pos + 1] if pos + 1 < len(ordered) else None
            if nxt:
                gap = nxt["offset"] - (ordered[pos]["offset"] + ordered[pos]["size"])
                print("  .dynstr 与下一个节(%s)之间的空隙: %d 字节" % (nxt["name"], gap))

        # ---- 3. .gnu.version_r 里的库名引用（换 soname 必须同步改）----
        vn = next((s for s in secs if s["type"] == SHT_GNU_verneed), None)
        print("\n.gnu.version_r:")
        if not vn:
            print("  （不存在，无需同步修改）")
        else:
            link = vn["link"]
            vstrtab = secs[link] if link < len(secs) else None
            print("  offset=0x%X size=%d  字符串表=节[%d] %s"
                  % (vn["offset"], vn["size"], link, vstrtab["name"] if vstrtab else "?"))
            if vstrtab:
                vs = b[vstrtab["offset"]:vstrtab["offset"] + vstrtab["size"]]
                o = vn["offset"]
                end = vn["offset"] + vn["size"]
                cnt = 0
                while o < end:
                    vn_version = u16(b, o)
                    vn_cnt = u16(b, o + 2)
                    vn_file = u32(b, o + 4)
                    fname = cstr(vs, vn_file)
                    print("    [%d] ver=%d cnt=%d file=%r  <== 库名在此" % (cnt, vn_version, vn_cnt, fname))
                    # 遍历 vernaux 找版本名
                    ao = o + 16
                    for _ in range(vn_cnt):
                        vna_hash = u32(b, ao)
                        vna_flags = u16(b, ao + 4)
                        vna_other = u16(b, ao + 6)
                        vna_name = u32(b, ao + 8)
                        print("         aux hash=0x%08X flags=%d other=%d name=%r"
                              % (vna_hash, vna_flags, vna_other, cstr(vs, vna_name)))
                        ao += 16
                    cnt += 1
                    o = ao if vn_cnt else o + 16
                    if vn_version == 0:
                        break

        # ---- 4. 直接统计字符串出现位置 ----
        needle = b"libonnxruntime.so"
        print("\n%r 出现位置:" % needle.decode())
        i = 0
        while True:
            i = b.find(needle, i)
            if i < 0:
                break
            which = "?"
            for s in secs:
                if s["offset"] <= i < s["offset"] + s["size"]:
                    which = s["name"]
                    break
            print("  0x%08X  节=%s" % (i, which))
            i += 1
        print()


if __name__ == "__main__":
    main()
