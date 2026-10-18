"""NLLB 模型下载器：断点续传 + 大小校验 + 真实内容校验。

两个实测踩到的坑，写在这里免得下次再踩：

1. **不能用 curl**：HF 的 resolve 会 302 到 CDN 的签名 URL，curl 跟随时会把签名里的
   %7E/%7C 二次编码，落到 CDN 就返回 "Entry not found"（15 字节）。

2. **urllib 不读 Windows 注册表的系统代理**（它只看 HTTP_PROXY/HTTPS_PROXY 环境变量）。
   直连 HEAD 能被放行，但直连 GET 会被拒 → 404。必须显式传代理。
"""
import os
import sys
from pathlib import Path
import time
import urllib.request
import urllib.error
# 仓库根（tools/ 的上一级）：脚本可以放在任何机器上跑
_R = Path(__file__).resolve().parents[1]


BASE = 'https://huggingface.co/Xenova/nllb-200-distilled-600M/resolve/main'
DEST = str(_R / '模型/上游/nllb/onnx')

# 显式代理（系统设置里的那个）。留空字符串则走直连。
PROXY = 'http://127.0.0.1:5188'

# 文件名 -> 期望字节数（来自 HF API 的 blobs=true）
WANTED = {
    'encoder_model_int8.onnx': 415346576,
    'decoder_model_merged_quantized.onnx': 475505771,
}


def _opener() -> urllib.request.OpenerDirector:
    handlers = []
    if PROXY:
        handlers.append(urllib.request.ProxyHandler({'http': PROXY, 'https': PROXY}))
    else:
        handlers.append(urllib.request.ProxyHandler({}))  # 显式禁用环境代理
    return urllib.request.build_opener(*handlers)


def download(name: str, expect: int) -> bool:
    os.makedirs(DEST, exist_ok=True)
    path = os.path.join(DEST, name)
    part = path + '.part'

    have = os.path.getsize(part) if os.path.exists(part) else 0
    if os.path.exists(path):
        got = os.path.getsize(path)
        if got == expect:
            print(f'  已完整，跳过  {name}  {got / 1e6:.1f} MB', flush=True)
            return True
        print(f'  大小不符（{got} != {expect}），删除重下  {name}', flush=True)
        os.remove(path)
        have = 0

    if have >= expect:
        os.replace(part, path)
        return True

    url = f'{BASE}/{name}'
    headers = {'User-Agent': 'Mozilla/5.0'}
    if have:
        headers['Range'] = f'bytes={have}-'
        print(f'  断点续传  {name}  从 {have / 1e6:.1f} MB 继续', flush=True)
    else:
        print(f'  开始  {name}  期望 {expect / 1e6:.1f} MB', flush=True)

    req = urllib.request.Request(url, headers=headers)
    started = time.time()
    last_report = 0.0

    try:
        resp = _opener().open(req, timeout=120)
    except urllib.error.HTTPError as e:
        body = b''
        try:
            body = e.read()[:400]
        except Exception:
            pass
        print(f'  HTTPError {e.code} {e.reason}  代理={PROXY}', flush=True)
        print(f'  响应体: {body!r}', flush=True)
        hdrs = dict(e.headers.items()) if e.headers else {}
        for k in ('Location', 'X-Error-Code', 'X-Error-Message'):
            if k in hdrs:
                print(f'  {k}: {hdrs[k]}', flush=True)
        return False

    with resp, open(part, 'ab') as f:
        # 服务端不支持 Range 时会回 200 并从头给，此时必须清掉已有的部分
        if have and resp.status == 200:
            print('  服务端不支持 Range，从头下载', flush=True)
            f.seek(0)
            f.truncate()
            have = 0

        total = expect
        done = have
        while True:
            block = resp.read(1024 * 512)
            if not block:
                break
            f.write(block)
            done += len(block)

            now = time.time()
            if now - last_report >= 3:
                last_report = now
                pct = done * 100.0 / total
                speed = (done - have) / max(now - started, 0.001) / 1e6
                print(f'    {pct:5.1f}%  {done / 1e6:7.1f}/{total / 1e6:.1f} MB  {speed:.2f} MB/s',
                      flush=True)

    got = os.path.getsize(part)
    print(f'  下载结束  {name}  {got / 1e6:.1f} MB  用时 {time.time() - started:.0f}s', flush=True)
    if got != expect:
        print(f'  !! 大小不符：{got} != {expect}，保留 .part 供续传', flush=True)
        return False

    os.replace(part, path)
    print(f'  OK  {name}', flush=True)
    return True


def verify_onnx(name: str) -> None:
    """ONNX 是 protobuf：首字节 0x08，且前若干字节内含 'onnx' 标识串。"""
    path = os.path.join(DEST, name)
    if not os.path.exists(path):
        print(f'  {name}: 文件不存在')
        return
    size = os.path.getsize(path)
    with open(path, 'rb') as f:
        head = f.read(4096)
        f.seek(max(0, size - 64))
        tail = f.read()
    has_magic = b'onnx' in head
    print(f'  {name:<40} {size / 1e6:8.1f} MB  首字节={head[0]:02x}  含onnx标识={has_magic}  '
          f'尾部{tail[-4:].hex()}')


def main() -> int:
    print(f'下载目录: {DEST}')
    ok = True
    for name, expect in WANTED.items():
        if not download(name, expect):
            ok = False
    print()
    print('=== 校验 ===')
    for name in WANTED:
        verify_onnx(name)
    total = sum(
        os.path.getsize(os.path.join(DEST, n))
        for n in WANTED if os.path.exists(os.path.join(DEST, n))
    )
    print(f'\n模型总计 {total / 1e6:.1f} MB')
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
