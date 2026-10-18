#!/usr/bin/env python3
"""分析 P0 采集到的 WAV：判定「真抓到声音」还是「静音/被拒」。

用法：
    python tools/analyze_wav.py <目录或wav文件> [更多路径...]
    python tools/analyze_wav.py Music/ScreenAST

判读标准（脚本会直接给出结论）：
    - 全局 RMS < 0.0005 且峰值 < 0.02  → 基本全静音，采集失败或当时没播声音
    - 峰值 >= 0.02 且 RMS >= 0.001     → 确实抓到了音频
    - 有声音但动态范围极小            → 抓到了但内容可疑（如只有提示音）
"""

import sys
import wave
from pathlib import Path

import numpy as np

# 判据阈值（16bit 归一化到 [-1,1]）
SILENT_RMS = 0.0005
SILENT_PEAK = 0.02
GOOD_RMS = 0.001
CLIP_LEVEL = 0.999


def dbfs(x: float) -> float:
    """线性幅度 → dBFS。0 会返回 -inf。"""
    if x <= 0:
        return float("-inf")
    return 20.0 * np.log10(x)


def analyze(path: Path) -> dict:
    with wave.open(str(path), "rb") as w:
        channels = w.getnchannels()
        width = w.getsampwidth()
        rate = w.getframerate()
        frames = w.getnframes()
        raw = w.readframes(frames)

    if width != 2:
        return {"path": path, "error": f"仅支持 16bit PCM，实际 {width * 8}bit"}

    # 多声道取平均下混成单声道，便于统一分析
    data = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    if channels > 1:
        data = data.reshape(-1, channels).mean(axis=1)

    if data.size == 0:
        return {"path": path, "error": "无采样数据（0 帧）"}

    peak = float(np.max(np.abs(data)))
    rms = float(np.sqrt(np.mean(data ** 2)))
    clipped = int(np.sum(np.abs(data) >= CLIP_LEVEL))

    # 逐秒切片，找「有声音的秒」和「静音的秒」
    seg = max(1, rate)
    n_seg = data.size // seg
    seg_rms = []
    for i in range(n_seg):
        chunk = data[i * seg:(i + 1) * seg]
        seg_rms.append(float(np.sqrt(np.mean(chunk ** 2))))
    seg_rms_arr = np.array(seg_rms) if seg_rms else np.array([0.0])

    active = int(np.sum(seg_rms_arr >= GOOD_RMS))
    silent = int(np.sum(seg_rms_arr < SILENT_RMS))

    return {
        "path": path,
        "rate": rate,
        "channels": channels,
        "duration_s": data.size / rate,
        "peak": peak,
        "rms": rms,
        "peak_dbfs": dbfs(peak),
        "rms_dbfs": dbfs(rms),
        "clipped": clipped,
        "seg_total": n_seg,
        "seg_active": active,
        "seg_silent": silent,
        "seg_max_rms": float(seg_rms_arr.max()) if seg_rms else 0.0,
    }


def verdict(r: dict) -> str:
    if r.get("error"):
        return f"⚠ 无法分析：{r['error']}"
    if r["peak"] < SILENT_PEAK and r["rms"] < SILENT_RMS:
        return "❌ 判定：基本全静音 —— 未被允许捕获，或采集时目标 App 没有出声"
    if r["rms"] >= GOOD_RMS and r["peak"] >= SILENT_PEAK:
        frac = (r["seg_active"] / r["seg_total"] * 100) if r["seg_total"] else 0
        return f"✅ 判定：确实抓到音频（有声秒占比 {frac:.0f}%）"
    return "⚠ 判定：极弱信号 —— 可能是提示音/底噪，建议重测"


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        print(__doc__)
        return 2

    targets: list[Path] = []
    for a in argv[1:]:
        p = Path(a)
        if p.is_dir():
            targets.extend(sorted(p.rglob("*.wav")))
        elif p.is_file():
            targets.append(p)
        else:
            print(f"跳过不存在的路径：{p}")

    if not targets:
        print("没有找到任何 .wav 文件")
        return 1

    print(f"共 {len(targets)} 个 WAV 文件\n")
    print("=" * 96)
    for p in targets:
        r = analyze(p)
        print(f"\n文件：{p}")
        if r.get("error"):
            print(f"  {verdict(r)}")
            continue
        print(
            f"  格式：{r['rate']} Hz / {r['channels']}ch / 16bit    "
            f"时长：{r['duration_s']:.2f} s"
        )
        print(
            f"  峰值：{r['peak']:.4f}  ({r['peak_dbfs']:.1f} dBFS)    "
            f"RMS：{r['rms']:.5f}  ({r['rms_dbfs']:.1f} dBFS)"
        )
        print(
            f"  逐秒：共 {r['seg_total']}s，有声 {r['seg_active']}s，"
            f"近乎全静音 {r['seg_silent']}s，最大秒 RMS {r['seg_max_rms']:.4f}"
        )
        if r["clipped"]:
            print(f"  削波采样数：{r['clipped']}（音量过大，可能失真）")
        print(f"  {verdict(r)}")
    print("\n" + "=" * 96)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
