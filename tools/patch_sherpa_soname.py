#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
给 sherpa-onnx 的 AAR 打 soname 补丁，解决它与 onnxruntime-android 的
`libonnxruntime.so` 同名冲突。

════════ 为什么必须这么做（实测证据） ════════
两个 AAR 各自带一份 `jni/<abi>/libonnxruntime.so`：

    官方 com.microsoft.onnxruntime:onnxruntime-android:1.31.0
        x86_64  40,827,944 B
    本地 third_party/sherpa-onnx/sherpa-onnx-1.13.8.aar
        x86_64  25,581,120 B

Android 链接器按 **DT_SONAME** 去重，两份同名绝不可能共存：加载哪份，
另一方的符号解析就断。而两份**谁都不能删**，因为符号版本标签互不兼容：

    sherpa libsherpa-onnx-c-api.so 需要  OrtGetApiBase@VERS_1.28.2
    官方 libonnxruntime4j_jni.so 需要  OrtGetApiBase@VERS_1.31.0
                                       OrtSessionOptionsAppendExecutionProvider_CPU@VERS_1.31.0
                                       OrtSessionOptionsAppendExecutionProvider_Nnapi@VERS_1.31.0

两种"合并成一份"的尝试都验证过，都会 dlopen 失败：
  · 只留官方 1.31.0    -> sherpa 找不到 VERS_1.28.2
  · 只留 sherpa 1.28.2 -> 官方 JNI 找不到 VERS_1.31.0
    （这正是最早那条 `cannot locate symbol "OrtGetApiBase"` 的成因）

唯一出路是给 sherpa 那份换一个 soname，让两份并存。

════════ 关键细节：DT_NEEDED 有两处，不止一处 ════════
实测三个 sherpa 库**各自都** NEEDED 了 libonnxruntime.so：

    libonnxruntime.so          -> 改 DT_SONAME
    libsherpa-onnx-c-api.so    -> 改 DT_NEEDED
    libsherpa-onnx-cxx-api.so  -> 改 DT_NEEDED
    libsherpa-onnx-jni.so      -> 改 DT_NEEDED

而且 AAR 里的 .so 文件名必须跟着改名 —— 否则运行时按新 soname 找不到文件。

════════ 为什么用 lief ════════
`libonnxruntime.so` 这 17 字节在文件里只出现 1 次，看起来"等长二进制替换"就够。
但实测：`.dynstr` 末尾**零填充**，与下一节只有 **6 字节**空隙，
而任何更长的新名字都塞不进去 —— 一旦加长，`.dynstr` 里
DT_SONAME / DT_NEEDED / .gnu.version_r 引用的**全部偏移都要重算**。
手写这个重排极易打乱整个动态链接段。

`lief`（pip install lief，纯 Python，Windows 原生可用）会自己重排 .dynstr
并修正所有偏移：实测改名后 SONAME 偏移从 0x98d 变为 0x4fe，
而**文件字节数完全不变**（25,581,120）。所以名字长度可以随便取。

用法：
    python patch_sherpa_soname.py <输入.aar> <输出.aar> [--name libonnxruntime_sherpa.so] [-v]
"""

import argparse
import os
import sys
import zipfile

import lief

OLD_NAME = "libonnxruntime.so"
DEFAULT_NEW_NAME = "libonnxruntime_sherpa.so"

# AAR 内需要打补丁的成员（按 basename 匹配）
#   True  = 改它的 DT_SONAME
#   False = 改它引用的 DT_NEEDED
TARGETS = {
    "libonnxruntime.so": True,
    "libsherpa-onnx-c-api.so": False,
    "libsherpa-onnx-cxx-api.so": False,
    "libsherpa-onnx-jni.so": False,
}


def patch_one(data: bytes, where: str, new_name: str, is_soname: bool, verbose: bool):
    """就地修改 ELF 的 SONAME 或 NEEDED。返回 (新数据, 改动次数)。"""
    if len(data) < 64 or data[:4] != b"\x7fELF":
        raise SystemExit("!! %s 不是 ELF" % where)

    # lief.parse 只吃路径，用临时文件中转
    import tempfile
    fd, tmp = tempfile.mkstemp(suffix=".so")
    os.close(fd)
    try:
        with open(tmp, "wb") as f:
            f.write(data)
        binary = lief.ELF.parse(tmp)
        if binary is None:
            raise SystemExit("!! lief 无法解析 %s" % where)

        hits = 0

        if is_soname:
            ent = binary.get(lief.ELF.DynamicEntry.TAG.SONAME)
            if ent is None:
                print("     [跳过] %s 没有 DT_SONAME" % where)
            else:
                old = ent.name
                ent.name = new_name
                hits += 1
                if verbose:
                    print("     SONAME    %r -> %r" % (old, new_name))

            # ③ 库自己的版本定义表 base 名（Rev/Flags=BASE 那一条）也要跟着改。
            #    实测它一开始没被 DT_SONAME 的修改带上，仍留着旧名。
            #    lief 把它藏在 auxiliary_symbols 里，没有直接的 .name 属性。
            for d in binary.symbols_version_definition:
                for a in d.auxiliary_symbols:
                    if a.name == OLD_NAME:
                        a.name = new_name
                        hits += 1
                        if verbose:
                            print("     VERDEF    %r -> %r" % (OLD_NAME, new_name))
        else:
            # ① 改 .dynamic 里的 DT_NEEDED
            ent = binary.get(lief.ELF.DynamicEntry.TAG.NEEDED)
            if ent is None:
                print("     [跳过] %s 没有 DT_NEEDED" % where)
            else:
                for e in binary.dynamic_entries:
                    if e.tag == lief.ELF.DynamicEntry.TAG.NEEDED and e.name == OLD_NAME:
                        e.name = new_name
                        hits += 1
                        if verbose:
                            print("     NEEDED    %r -> %r" % (OLD_NAME, new_name))

            # ② 必须同步改 .gnu.version_r 的 vn_file —— 实测 lief 不会自动跟着改。
            #    这一处漏掉的话，.dynamic 说依赖 A、版本需求却说属于 B，
            #    链接器做符号版本匹配时就认不出 VERS_1.28.2 归谁。
            for vr in binary.symbols_version_requirement:
                if vr.name == OLD_NAME:
                    vr.name = new_name
                    hits += 1
                    if verbose:
                        print("     VERSION_R %r -> %r" % (OLD_NAME, new_name))

        if hits == 0:
            return data, 0

        out = binary.write_to_bytes()
        if not out:
            raise SystemExit("!! %s 写回失败" % where)
        return bytes(out), hits
    finally:
        try:
            os.unlink(tmp)
        except OSError:
            pass


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("src", help="输入 AAR")
    ap.add_argument("dst", help="输出 AAR")
    ap.add_argument("--name", default=DEFAULT_NEW_NAME, help="新的 soname")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()

    if not os.path.isfile(args.src):
        raise SystemExit("!! 找不到 %s" % args.src)
    if os.path.abspath(args.src) == os.path.abspath(args.dst):
        raise SystemExit("!! 输入输出不能是同一个文件，请先备份")
    if args.name == OLD_NAME:
        raise SystemExit("!! 新名字与原名字相同，不会产生任何改变")

    print("输入   : %s" % args.src)
    print("输出   : %s" % args.dst)
    print("新 soname: %s" % args.name)
    print()

    renamed = {}   # 原 basename -> 新 basename
    patched = {}   # 显示的成员名 -> 改动次数
    total = 0

    with zipfile.ZipFile(args.src, "r") as zin:
        with zipfile.ZipFile(args.dst, "w", zipfile.ZIP_DEFLATED) as zout:
            for info in zin.infolist():
                name = info.filename
                data = zin.read(name)
                out_name = name

                if name.startswith("jni/") and name.endswith(".so"):
                    base = name.rsplit("/", 1)[-1]
                    is_soname = TARGETS.get(base)
                    if is_soname is not None:
                        print("  [补丁] %s" % name)
                        data, n = patch_one(data, name, args.name, is_soname, args.verbose)
                        if n:
                            total += n
                            patched[name] = n
                            if is_soname:
                                # 文件名也要改，否则运行时按新 soname 找不到文件
                                newbase = args.name
                                out_name = name[: -len(base)] + newbase
                                renamed[base] = newbase
                                print("           文件改名 %s -> %s" % (base, newbase))

                # 保留原压缩方式与时间戳
                zi = zipfile.ZipInfo(out_name, date_time=info.date_time)
                zi.compress_type = info.compress_type
                zi.external_attr = info.external_attr
                zout.writestr(zi, data)

    print()
    print("=== 结果 ===")
    for k in sorted(patched):
        print("  %-46s %d 处" % (k, patched[k]))
    if renamed:
        print("  文件改名: %s" % renamed)
    if total == 0:
        raise SystemExit("!! 一处都没改到，输入 AAR 结构可能与预期不符")

    print()
    print("共 %d 处改动，输出 %s（%.1f MB）"
          % (total, args.dst, os.path.getsize(args.dst) / 1024 / 1024))
    print()
    print("核验命令：")
    print("  unzip -o -q %s 'jni/x86_64/*' -d /tmp/chk" % args.dst)
    print("  RE=$NDK/toolchains/llvm/prebuilt/windows-x86_64/bin/llvm-readelf.exe")
    print("  $RE -d /tmp/chk/jni/x86_64/%s | grep SONAME" % args.name)
    print("  $RE -d /tmp/chk/jni/x86_64/libsherpa-onnx-c-api.so | grep NEEDED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
