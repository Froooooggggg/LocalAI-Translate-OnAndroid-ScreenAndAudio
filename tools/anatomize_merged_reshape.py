#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
静态解剖 merged decoder 里那个必崩的 Reshape 节点，找出它到底在 reshape 什么。

报错节点：'/model/decoder/layers.0/encoder_attn/Reshape_4'
错误：The dimension with value zero exceeds the dimension size of the input tensor.

同时对比首步（无缓存）与续步（有缓存）时 encoder 侧 present 的真实形状——
之前一直假设它是 [1,16,S,64]，也许根本不是。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort

sys.path.insert(0, str(Path(__file__).parent))
from verify_nllb_standalone import Tokenizer, TOKENIZER_BIN, FLORES  # noqa: E402
# 仓库根（tools/ 的上一级）：脚本可以放在任何机器上跑
_R = Path(__file__).resolve().parents[1]


ROOT = Path(str(_R))
ONNX = ROOT / "third_party" / "nllb" / "onnx"
MERGED = ONNX / "decoder_model_merged_quantized.onnx"
DEC_ON = ONNX / "decoder_model_quantized.onnx"
ENC = ONNX / "encoder_model_int8.onnx"

TARGET = "/model/decoder/layers.0/encoder_attn/Reshape_4"


def section(t):
    print(f"\n{'='*76}\n{t}\n{'='*76}")


section("1. merged 图里 Reshape_4 节点的入边")

m = onnx.load(str(MERGED), load_external_data=False)
init_names = {i.name for i in m.graph.initializer}
producer = {}
for n in m.graph.node:
    for o in n.output:
        producer[o] = n

target = None
for n in m.graph.node:
    if TARGET in n.output or n.name == TARGET:
        target = n
        break

if target is None:
    print("  未按名字找到，改用模糊匹配 ...")
    for n in m.graph.node:
        if "encoder_attn" in n.name and n.op_type == "Reshape":
            print(f"  候选: {n.name}")
else:
    print(f"  节点 op_type={target.op_type}  name={target.name}")
    print(f"  输出: {list(target.output)}")
    for idx, inp in enumerate(target.input):
        src = producer.get(inp)
        tag = "initializer" if inp in init_names else (src.op_type if src else "?")
        print(f"  输入[{idx}] = {inp}")
        print(f"          来源: {tag}")
        if inp in init_names:
            t = next(i for i in m.graph.initializer if i.name == inp)
            vals = onnx.numpy_helper.to_array(t)
            print(f"          常量值: {vals.tolist() if vals.size <= 16 else vals.shape}")
        elif src is not None:
            print(f"          上游节点: {src.name}  输入={list(src.input)}")

section("2. 该 Reshape 的上游链条（往上追 6 层）")
cur = target
for depth in range(6):
    if cur is None:
        break
    print(f"  [depth {depth}] {cur.op_type}  name={cur.name}")
    print(f"             input = {list(cur.input)}")
    if cur.input:
        cur = producer.get(cur.input[0])
    else:
        cur = None

section("3. decoder_model（无缓存版）的 present.*.encoder.* 是否真的产出")
s_on = ort.InferenceSession(str(DEC_ON), providers=["CPUExecutionProvider"])
past_out_on = [o.name for o in s_on.get_outputs() if ".encoder." in o.name]
print(f"  decoder_model 输出里的 encoder 侧 present 个数 = {len(past_out_on)}")
print(f"  前 4 个: {past_out_on[:4]}")

s_merged = ort.InferenceSession(str(MERGED), providers=["CPUExecutionProvider"])
past_out_m = [o.name for o in s_merged.get_outputs() if ".encoder." in o.name]
print(f"  merged 输出里的 encoder 侧 present 个数 = {len(past_out_m)}")
print(f"  前 4 个: {past_out_m[:4]}")

section("4. 首步（use_cache_branch=False）后 encoder 侧 present 的真实形状")
tok = Tokenizer(TOKENIZER_BIN)
ids = tok.encode("Hello everyone, welcome back to the channel.", "eng_Latn")
enc = ort.InferenceSession(str(ENC), providers=["CPUExecutionProvider"])
hidden = enc.run(None, {
    "input_ids": np.array([ids], dtype=np.int64),
    "attention_mask": np.ones([1, len(ids)], dtype=np.int64),
})[0]
S = hidden.shape[1]
print(f"  encoder 输出形状 = {hidden.shape}  (S={S})")

ins_m = [i.name for i in s_merged.get_inputs()]
past_names_m = [n for n in ins_m if n.startswith("past_key_values.")]
enc_past_names = [n for n in past_names_m if ".encoder." in n]
dec_past_names = [n for n in past_names_m if ".decoder." in n]
print(f"  merged 输入: past encoder {len(enc_past_names)} 个, past decoder {len(dec_past_names)} 个")

# 用「首步无缓存」跑一次，拿到 present
feed = {
    "input_ids": np.array([[FLORES["zh"][1]]], dtype=np.int64),
    "encoder_hidden_states": hidden,
    "encoder_attention_mask": np.ones([1, S], dtype=np.int64),
    "use_cache_branch": np.array([False], dtype=np.bool_),
}
for n in past_names_m:
    feed[n] = np.zeros([1, 16, 0, 64], dtype=np.float32)

out_names = [o.name for o in s_merged.get_outputs()]
out = s_merged.run(None, feed)
res = dict(zip(out_names, out))
print("\n  首步（无缓存）产出的 encoder 侧 present 形状:")
for n in past_out_m[:4]:
    print(f"    {n:<44} {res[n].shape}")
print("\n  首步产出的 decoder 侧 present 形状:")
for n in [x for x in out_names if ".decoder." in x][:2]:
    print(f"    {n:<44} {res[n].shape}")

section("5. 把首步的 encoder present 回喂给续步，看 Reshape 是否仍崩")
past = {("past_key_values." + k[len("present."):]): v for k, v in res.items()
        if k.startswith("present.")}
past["use_cache_branch"] = np.array([True], dtype=np.bool_)
feed2 = {
    "input_ids": np.array([[248059]], dtype=np.int64),
    "encoder_hidden_states": hidden,
    "encoder_attention_mask": np.ones([1, S], dtype=np.int64),
}
feed2.update(past)
print(f"  回喂的 encoder past 形状: "
      f"{past['past_key_values.0.encoder.key'].shape}")
print(f"  回喂的 decoder past 形状: "
      f"{past['past_key_values.0.decoder.key'].shape}")
try:
    out2 = s_merged.run(None, feed2)
    print(f"  ✅ 续步成功，logits {dict(zip(out_names,out2))['logits'].shape}")
except Exception as e:
    print(f"  ❌ 续步失败: {str(e)[:500]}")
