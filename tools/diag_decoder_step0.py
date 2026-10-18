from pathlib import Path
"""验证 decoder_model_merged 首步的正确喂法。

假设：跨注意力 Reshape 需要 encoder 侧 past 具备真实（非零）序列长度，
     因此首步必须用「真实 encoder 长度」的占位 past，配合 use_cache_branch=true，
     让该步输出并丢弃（bogus）——正如 transformers.js 的做法。
"""
import numpy as np
import onnxruntime as ort
# 仓库根（tools/ 的上一级）：脚本可以放在任何机器上跑
_R = Path(__file__).resolve().parents[1]


ONNX = str(_R / '模型/上游/nllb/onnx')
ENC = ONNX + r"\encoder_model_int8.onnx"
DEC = ONNX + r"\decoder_model_merged_quantized.onnx"

enc = ort.InferenceSession(ENC, providers=["CPUExecutionProvider"])
dec = ort.InferenceSession(DEC, providers=["CPUExecutionProvider"])

dec_in = [i.name for i in dec.get_inputs()]
dec_out = [o.name for o in dec.get_outputs()]

# 造一段假的编码器输入（长度 12，模拟一句话）
SRC_LEN = 12
src_ids = np.array([[256047] + [100 + i for i in range(SRC_LEN - 1)]], dtype=np.int64)
enc_mask = np.ones((1, SRC_LEN), dtype=np.int64)
hidden = enc.run(None, {"input_ids": src_ids, "attention_mask": enc_mask})[0]
print(f"encoder 输出: {hidden.shape}")


def past_of(name, enc_len, dec_len, dtype=np.float32):
    """按名字构造 past。encoder 侧用真实 encoder 长度，decoder 侧用 dec_len。"""
    is_enc = ".encoder." in name
    if name.endswith(".key") or name.endswith(".value"):
        length = enc_len if is_enc else dec_len
        return np.zeros((1, 16, length, 64), dtype=dtype)
    raise ValueError(name)


def step0(mode):
    """尝试首步。mode: 'empty' = encoder past 长度 0；'bogus' = encoder past 真实长度。"""
    enc_len = 0 if mode == "empty" else SRC_LEN
    dec_len = 0
    branch = mode == "empty"
    feed = {
        "input_ids": np.array([[2]], dtype=np.int64),          # decoder_start_token_id
        "encoder_hidden_states": hidden,
        "encoder_attention_mask": enc_mask,
        "use_cache_branch": np.array([branch], dtype=np.bool_),
    }
    for n in dec_in:
        if n.startswith("past_key_values."):
            feed[n] = past_of(n, enc_len, dec_len)
    try:
        outs = dec.run(None, feed)
        by = dict(zip(dec_out, outs))
        tok = int(np.argmax(by["logits"][0, -1]))
        return True, tok, by
    except Exception as e:
        msg = str(e).replace("\n", " ")[:150]
        return False, msg, None


for mode in ("empty", "bogus"):
    ok, info, _ = step0(mode)
    print(f"[{mode:>5}] {'成功' if ok else '失败'}  {info}")


def full_greedy(mode, max_new=8):
    """完整解码几步，看是否给出合理 token 序列。"""
    enc_len = 0 if mode == "empty" else SRC_LEN
    branch = mode == "empty"
    feed_base = {
        "encoder_hidden_states": hidden,
        "encoder_attention_mask": enc_mask,
    }
    past = {n: past_of(n, enc_len, 0) for n in dec_in if n.startswith("past_key_values.")}
    past["use_cache_branch"] = np.array([branch], dtype=np.bool_)
    ids = [2]
    generated = []
    for step in range(max_new):
        feed = dict(feed_base)
        feed["input_ids"] = np.array([[ids[-1]]], dtype=np.int64)
        feed.update(past)
        outs = dec.run(None, feed)
        by = dict(zip(dec_out, outs))
        nxt = int(np.argmax(by["logits"][0, -1]))
        generated.append(nxt)
        if nxt == 2:
            break
        past = {n: by[n.replace("past_key_values.", "present.")] for n in dec_in
                if n.startswith("past_key_values.")}
        past["use_cache_branch"] = np.array([True], dtype=np.bool_)
        ids.append(nxt)
    return generated


for mode in ("empty", "bogus"):
    try:
        g = full_greedy(mode)
        print(f"[{mode:>5}] greedy tokens = {g}")
    except Exception as e:
        print(f"[{mode:>5}] greedy 失败: {str(e)[:150]}")
