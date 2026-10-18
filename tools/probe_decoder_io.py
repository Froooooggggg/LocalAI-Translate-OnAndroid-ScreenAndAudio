"""打印 NLLB encoder / decoder_model_merged 的精确输入输出契约。

用途：确定 merged decoder 的正确喂法（use_cache_branch 与 past_key_values 的形状）。
"""
import sys
from pathlib import Path
import onnxruntime as ort
# 仓库根（tools/ 的上一级）：脚本可以放在任何机器上跑
_R = Path(__file__).resolve().parents[1]


ONNX = str(_R / '模型/上游/nllb/onnx')
ENC = ONNX + r"\encoder_model_int8.onnx"
DEC = ONNX + r"\decoder_model_merged_quantized.onnx"


def dump(path, tag):
    print(f"===== {tag} =====")
    s = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    print("-- inputs --")
    for i in s.get_inputs():
        print(f"  {i.name:<58} {str(i.shape):<24} {i.type}")
    print("-- outputs --")
    for o in s.get_outputs():
        print(f"  {o.name:<58} {str(o.shape):<24} {o.type}")
    # 输入名按类别统计
    names = [i.name for i in s.get_inputs()]
    pkv = [n for n in names if n.startswith("past_key_values")]
    print(f"-- 汇总 -- 输入数={len(names)}  past_key_values 输入数={len(pkv)}")
    if pkv:
        print(f"  decoder 侧 past 示例: {pkv[0]}")
        enc_pkv = [n for n in pkv if ".encoder." in n]
        dec_pkv = [n for n in pkv if ".decoder." in n]
        print(f"  encoder 侧 past={len(enc_pkv)}  decoder 侧 past={len(dec_pkv)}")
    others = [n for n in names if not n.startswith("past_key_values")]
    print(f"  非 past 输入: {others}")
    print()


if __name__ == "__main__":
    dump(ENC, "encoder_model_int8.onnx")
    dump(DEC, "decoder_model_merged_quantized.onnx")
