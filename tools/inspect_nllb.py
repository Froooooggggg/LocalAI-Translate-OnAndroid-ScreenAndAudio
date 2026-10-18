from pathlib import Path
import json, sys, urllib.request, os
# 仓库根（tools/ 的上一级）：脚本可以放在任何机器上跑
_R = Path(__file__).resolve().parents[1]


TOK = str(_R / '模型/上游/nllb/tokenizer.json')

print('=== 1. tokenizer.json 关键结构 ===')
with open(TOK, 'r', encoding='utf-8') as f:
    j = json.load(f)

print('顶层键:', list(j.keys()))
m = j.get('model', {})
print('model.type =', m.get('type'))
print('model 键:', list(m.keys())[:12])
print()
pt = j.get('pre_tokenizer')
print('pre_tokenizer =', json.dumps(pt, ensure_ascii=False)[:400])
print()
print('decoder =', json.dumps(j.get('decoder'), ensure_ascii=False)[:300])
print('normalizer =', json.dumps(j.get('normalizer'), ensure_ascii=False)[:300])
print()
print('=== 2. 语言特殊 token 的 id（关键：不能猜） ===')
added = j.get('added_tokens', [])
print('added_tokens 总数 =', len(added))
by_content = {t['content']: t for t in added if isinstance(t, dict)}
for c in ['eng_Latn', 'zho_Hans', 'jpn_Jpan', 'rus_Cyrl', '<s>', '</s>', '<pad>', '<unk>']:
    t = by_content.get(c)
    if t:
        print(f'  {c:<10} id={t.get("id"):<8} special={t.get("special")}')
    else:
        print(f'  {c:<10} 不在 added_tokens 中')

vocab = m.get('vocab', {})
print('vocab 条目数 =', len(vocab))
for c in ['eng_Latn', 'zho_Hans', 'jpn_Jpan', 'rus_Cyrl']:
    if c in vocab:
        print(f'  vocab[{c}] = {vocab[c]}')

print()
print('=== 3. 验证 ONNX 下载 URL 是否真的可达 ===')
urls = [
    'https://huggingface.co/Xenova/nllb-200-distilled-600M/resolve/main/onnx/encoder_model_int8.onnx',
    'https://huggingface.co/Xenova/nllb-200-distilled-600M/resolve/main/onnx/decoder_model_merged_quantized.onnx',
]
for u in urls:
    try:
        req = urllib.request.Request(u, method='HEAD', headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=45) as r:
            print(f'  HTTP {r.status}  Content-Length={r.headers.get("Content-Length")}  {os.path.basename(u)}')
    except Exception as e:
        print(f'  FAIL {type(e).__name__}: {e}  <- {os.path.basename(u)}')
