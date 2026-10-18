#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""打印独立 decoder 的真实输入/输出契约（供 Kotlin 侧对齐）。"""
import time
from pathlib import Path

import onnxruntime as ort
# 仓库根（tools/ 的上一级）：脚本可以放在任何机器上跑
_R = Path(__file__).resolve().parents[1]


NLLB = Path(str(_R / '模型/上游/nllb/onnx'))

for name in ("decoder_model_quantized.onnx", "encoder_model_int8.onnx"):
    p = NLLB / name
    print(f"=== {name}  ({p.stat().st_size:,} 字节) ===")
    t0 = time.perf_counter()
    s = ort.InferenceSession(str(p), providers=["CPUExecutionProvider"])
    print(f"    加载 {(time.perf_counter() - t0) * 1000:.0f} ms")
    ins, outs = s.get_inputs(), s.get_outputs()
    print(f"    输入 {len(ins)} 个:")
    for i in ins[:6]:
        print(f"      {i.name:<40} {i.type:<10} {i.shape}")
    if len(ins) > 6:
        print(f"      ... 其余 {len(ins) - 6} 个（past_key_values.*）")
        print(f"      末项 {ins[-1].name:<34} {ins[-1].type:<10} {ins[-1].shape}")
    print(f"    输出 {len(outs)} 个:")
    for o in outs[:3]:
        print(f"      {o.name:<40} {o.type:<10} {o.shape}")
    if len(outs) > 3:
        print(f"      ... 其余 {len(outs) - 3} 个（present.*）")
        print(f"      末项 {outs[-1].name:<34} {outs[-1].type:<10} {outs[-1].shape}")
    print()

# 检查独立 decoder 的 decoder 侧 present 是否带 cache_position 维度
p = NLLB / "decoder_model_quantized.onnx"
s = ort.InferenceSession(str(p), providers=["CPUExecutionProvider"])
print("=== 判断 present 的 decoder 侧形状约定 ===")
for o in s.get_outputs():
    if o.name.startswith("present.") and ".decoder." in o.name:
        print(f"  样本 {o.name}: {o.shape}")
        break
print("  说明：若 decoder 侧为 ['batch_size','num_heads','decoder_sequence_length',64]")
print("        则该维度是动态的、且包含本步 token（靠 cache_position 拼接），")
print("        下一步可直接把它当 past 回喂，无需自行 concat。")
