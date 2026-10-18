from pathlib import Path
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""列出两个 decoder 的非 past 输入，重点确认 use_cache_branch 是否存在。

动机：`decoder_with_past_model_quantized.onnx` 其实是 Optimum 导出的**合并式**
decoder（所以它才要 48 个 past 作为输入）。合并式 decoder 有一个必需的开关输入
`use_cache_branch`：
  = true  -> 走「有缓存」分支，复用传入的 past
  = false -> 走「无缓存」分支，重算 cross-attention KV，而那个分支需要
             `encoder_hidden_states`（我们没传）→ ONNX Runtime 直接崩。
之前数 past 输入时按前缀 `past_key_values.` 过滤，正好把它漏掉了。
"""
import onnx
# 仓库根（tools/ 的上一级）：脚本可以放在任何机器上跑
_R = Path(__file__).resolve().parents[1]


BASE = str(_R / '模型/上游/nllb/onnx')

for name in ("decoder_with_past_model_quantized.onnx", "decoder_model_quantized.onnx"):
    m = onnx.load(BASE + "\\" + name, load_external_data=False)
    ins = [i.name for i in m.graph.input]
    non_past = [n for n in ins if not n.startswith("past_key_values.")]
    print("###", name)
    print("  非 past 输入 :", non_past)
    print("  输入总数     :", len(ins), "| past 数:", len(ins) - len(non_past))
    typ = {}
    for i in m.graph.input:
        if i.name in non_past:
            t = i.type.tensor_type.elem_type
            typ[i.name] = onnx.TensorProto.DataType.Name(t)
    print("  非 past 类型 :", typ)
    print()
