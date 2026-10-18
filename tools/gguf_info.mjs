// 解析 GGUF 元数据并打印。接入分析的第一步：
// 「这个模型到底是不是混合 SSM 架构、有多少层、KV 头多宽」直接决定
// 哪个推理运行时能跑它、KV 缓存要留多大、以及层数能不能做增量 prefill。
//
// GGUF 格式（v3）：
//   magic "GGUF" | version u32 | tensor_count u64 | kv_count u64
//   然后 kv_count 组：key(string) type(u32) value
//   string = len(u64) + utf8 字节
//   array  = elem_type(u32) count(u64) + count 个元素
//
// 用法：node tools/gguf_info.mjs <file.gguf>

import { open } from 'node:fs/promises';

const T = { UINT8: 0, INT8: 1, UINT16: 2, INT16: 3, UINT32: 4, INT32: 5, FLOAT32: 6, BOOL: 7, STRING: 8, ARRAY: 9, UINT64: 10, INT64: 11, FLOAT64: 12 };
const SIZE = { 0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8 };

const path = process.argv[2];
if (!path) { console.error('用法: node tools/gguf_info.mjs <file.gguf>'); process.exit(2); }

const fh = await open(path, 'r');

// 滑动窗口顺序读。tokenizer 的词表数组有好几 MB，一次性开大缓冲会浪费，
// 固定在 1MB 窗口里按需补数据即可（用完就丢，buf 始终很小）。
const CHUNK = 1 << 20;
let buf = Buffer.alloc(0);
let fileOff = 0;
let pos = 0; // 已消费的逻辑字节数，只用于报错

async function need(n) {
  while (buf.length < n) {
    const tmp = Buffer.alloc(Math.max(CHUNK, n - buf.length));
    const { bytesRead } = await fh.read(tmp, 0, tmp.length, fileOff);
    if (!bytesRead) throw new Error(`EOF：想读 ${n} 字节，文件在 ${fileOff} 处结束（已消费 ${pos}）`);
    fileOff += bytesRead;
    buf = Buffer.concat([buf, tmp.subarray(0, bytesRead)]);
  }
}
async function take(n) {
  await need(n);
  const out = buf.subarray(0, n);
  buf = buf.subarray(n);
  pos += n;
  return out;
}
const u32 = async () => (await take(4)).readUInt32LE(0);
const u64 = async () => Number((await take(8)).readBigUInt64LE(0));
const i64 = async () => Number((await take(8)).readBigInt64LE(0));
async function str() { const n = await u64(); return (await take(n)).toString('utf8'); }

async function value(type) {
  switch (type) {
    case T.STRING: return await str();
    case T.ARRAY: {
      const et = await u32(); const n = await u64();
      const items = [];
      for (let i = 0; i < n; i++) items.push(await value(et));
      return items;
    }
    case T.BOOL: return (await take(1))[0] !== 0;
    case T.FLOAT32: return (await take(4)).readFloatLE(0);
    case T.FLOAT64: return (await take(8)).readDoubleLE(0);
    case T.UINT64: return await u64();
    case T.INT64: return await i64();
    case T.UINT32: return await u32();
    case T.INT32: return (await take(4)).readInt32LE(0);
    case T.UINT16: return (await take(2)).readUInt16LE(0);
    case T.INT16: return (await take(2)).readInt16LE(0);
    case T.UINT8: return (await take(1))[0];
    case T.INT8: return (await take(1)).readInt8(0);
    default: throw new Error(`未知 value type ${type} @${pos}`);
  }
}

const magic = (await take(4)).toString('ascii');
if (magic !== 'GGUF') throw new Error(`不是 GGUF：magic=${JSON.stringify(magic)}`);
const version = await u32();
const tensorCount = await u64();
const kvCount = await u64();

console.log(`文件    : ${path}`);
console.log(`magic   : ${magic}  version=${version}`);
console.log(`张量数  : ${tensorCount}`);
console.log(`元数据  : ${kvCount} 条\n`);

const md = {};
for (let i = 0; i < kvCount; i++) {
  const k = await str();
  const t = await u32();
  md[k] = await value(t);
}

// 全量打印（tokenizer 的巨型数组合并成一行摘要，否则会刷屏）
const ftype = { 0: 'ALL_F32', 1: 'MOSTLY_F16', 2: 'Q4_0', 3: 'Q4_1', 7: 'Q8_0', 8: 'Q5_0', 9: 'Q5_1', 10: 'Q2_K', 11: 'Q3_K_S', 12: 'Q3_K_M', 13: 'Q3_K_L', 14: 'Q4_K_S', 15: 'Q4_K_M', 16: 'Q5_K_S', 17: 'Q5_K_M', 18: 'Q6_K' };
for (const [k, v] of Object.entries(md)) {
  let shown;
  if (Array.isArray(v)) {
    const nums = v.every((x) => typeof x === 'number');
    shown = nums
      ? `[${v.length} 个数] ${JSON.stringify(v.slice(0, 12))}${v.length > 12 ? ' …' : ''}`
      : `[${v.length} 项] 首项=${JSON.stringify(String(v[0])).slice(0, 80)} …`;
  } else if (typeof v === 'string' && v.length > 200) {
    shown = `len=${v.length}  ${JSON.stringify(v.slice(0, 160))} …`;
  } else {
    shown = JSON.stringify(v);
  }
  if (k === 'general.file_type' && ftype[v]) shown += `  (${ftype[v]})`;
  console.log(`  ${k.padEnd(42)} = ${shown}`);
}

// 架构关键参数摘要
console.log('\n=== 架构要点 ===');
const a = md['general.architecture'];
const g = (suffix) => md[`${a}.${suffix}`];
const keys = Object.keys(md).filter((k) => k.startsWith(a + '.'));
const ssmKeys = keys.filter((k) => k.includes('ssm'));
const attnKeys = keys.filter((k) => k.includes('attention'));

console.log(`architecture          = ${a}`);
console.log(`block_count(层数)     = ${g('block_count')}`);
console.log(`context_length        = ${g('context_length')}`);
console.log(`embedding_length      = ${g('embedding_length')}`);
console.log(`feed_forward_length   = ${g('feed_forward_length')}`);
console.log(`attention.head_count  = ${g('attention.head_count')}`);
console.log(`attention.head_count_kv = ${g('attention.head_count_kv')}`);
console.log(`attention.key_length  = ${g('attention.key_length')}`);
console.log(`attention.value_length= ${g('attention.value_length')}`);
console.log(`attention 相关键(${attnKeys.length}) : ${attnKeys.map((k) => k.replace(a + '.', '')).join(', ')}`);
console.log(`ssm 相关键(${ssmKeys.length})       : ${ssmKeys.map((k) => k.replace(a + '.', '')).join(', ')}`);
for (const k of ssmKeys) console.log(`   ${k.padEnd(46)} = ${JSON.stringify(md[k])}`);
const all = keys.map((k) => k.replace(a + '.', ''));
console.log(`\n全部 ${a}.* 键(${all.length}): ${all.join(', ')}`);

// 算一下 KV 缓存到底多大（同传只要 1K 上下文就能用）
const nl = g('block_count'), nkv = g('attention.head_count_kv') ?? g('attention.head_count');
const kl = g('attention.key_length') ?? (g('embedding_length') / g('attention.head_count'));
const vl = g('attention.value_length') ?? kl;
if (nl && nkv && kl) {
  let lay = nl;
  const fai = g('full_attention_interval') ?? md['full_attention_interval'];
  if (fai) { lay = Math.ceil(nl / fai); console.log(`\nfull_attention_interval=${fai} → ${nl} 层里只有 ${lay} 层是全注意力`); }
  const per = 2 * lay * nkv * (kl + vl) * 2; // K+V，fp16
  console.log(`KV 缓存/token = ${(per / 1024).toFixed(1)} KB  （按 ${lay} 层 × ${nkv} KV头 × ${kl}+${vl} dim × fp16）`);
  for (const ctx of [512, 1024, 2048, 4096]) console.log(`   上下文 ${String(ctx).padStart(4)} → ${(per * ctx / 1048576).toFixed(1)} MB`);
}

await fh.close();
