// 用「Android 端将要拼的那种提示词」跑完整俚语/难句集，产出可直接引用的质量表。
//
// 与 tools/llm_probe.mjs 的区别：那个走 ollama 的 /api/chat（由 ollama 套模板），
// 这个走 /api/generate + raw:true（提示词完全由本脚本拼），所以这里的输出
// **就是** Android 端拼同样的字符串会得到的输出（同一模型、同一 temp=0）。
import { appendFileSync } from 'node:fs';

const MODEL = process.env.MODEL || 'qwen2b-abl';
const BASE = 'http://127.0.0.1:11434';

// ⚠️ 这条提示词是**实验选出来的**，不要凭直觉加长。
// 测试数据/probe/sysprompt_ab.mjs 在 5 条已知薄弱样例上比过 4 套措辞：这条（P2）在 4 条上
// 优于我那套写得很详细的版本 —— 加长指令列表会让 2B 模型把 fox 译成「熊」、
// 把 strip club 原样留成英文。显式写「动物名称必须准确」这种约束**完全无效**。
// 改动这条字符串 = 改动译文质量，务必用 tools/llm_production_prompt_test.mjs 复测。
const SYS = process.env.SYS || '把下面的内容翻译成简体中文，只输出译文。';

function buildPrompt(text) {
  return `<|im_start|>system\n${SYS}<|im_end|>\n`
       + `<|im_start|>user\n${text}<|im_end|>\n`
       + `<|im_start|>assistant\n<think>\n\n</think>\n\n`;
}

async function translate(text) {
  const t0 = Date.now();
  const r = await fetch(`${BASE}/api/generate`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({
      model: MODEL, prompt: buildPrompt(text), raw: true, stream: false,
      options: { temperature: 0, top_k: 1, num_predict: 128, num_ctx: 2048 },
    }),
  });
  if (!r.ok) throw new Error(`${r.status} ${await r.text()}`);
  const j = await r.json();
  return { out: (j.response || '').trim(), ms: Date.now() - t0, tok: j.eval_count, done: j.done_reason };
}

// 与 tools/llm_probe.mjs 同一套样例，便于和已有结果对照
const CASES = [
  ['基准', 'The quick brown fox jumps over the lazy dog.'],
  ['基准', 'The meeting has been postponed until next Tuesday.'],
  ['口语', 'What the hell are you doing right now'],
  ['习语', 'Holy crap, that was close'],
  ['粗口', 'That guy is a total jackass'],
  ['粗口', 'He told her to go to hell'],
  ['粗口', 'This burger is damn good'],
  ['粗口', 'This is a bunch of bullshit and you know it'],
  ['粗口', 'Damn it, I forgot my keys again'],
  ['成人', 'They got drunk and made out in the car'],
  ['成人', 'He spent the whole night in a strip club'],
  ['习语', 'Break a leg out there tonight'],
  ['习语', 'That is a piece of cake for him'],
  // 长句 —— 对照 NLLB 提前收尾那个坑
  ['长句', 'I think the most important thing we need to understand is that nobody really knows what is going to happen next, and that is exactly why we should all be a lot more careful than we usually are.'],
  ['长句', 'The report, which was published on Tuesday after a six-month investigation into the causes of the accident, concluded that the company had ignored at least four separate safety warnings.'],
];

console.log(`模型 ${MODEL}   ${CASES.length} 条   提示词 = Android 生产形态\n`);
console.log('='.repeat(90));
const rows = [];
for (const [kind, src] of CASES) {
  try {
    const r = await translate(src);
    const clean = !r.out || r.out.includes('<think>') || r.out.includes('</think>');
    console.log(`[${kind}] ${src}`);
    console.log(`   → ${JSON.stringify(r.out)}`);
    console.log(`   ${r.tok} tok / ${r.ms}ms / done=${r.done}${clean ? '   ⚠️ 输出含思考标记或为空' : ''}\n`);
    rows.push({ kind, src, ...r, clean });
  } catch (e) {
    console.log(`[${kind}] ${src}\n   ✗ ${e.message}\n`);
  }
}

// 落一份 TSV，方便和 NLLB 真机结果并排看
const tsv = ['kind\tsrc\tllm_out\ttok\tms\tdone'];
for (const r of rows) tsv.push([r.kind, r.src, r.out, r.tok, r.ms, r.done].join('\t'));
appendFileSync('测试数据/probe/llm_production_prompt.tsv', tsv.join('\n') + '\n');

const bad = rows.filter(r => r.clean).length;
const tot = rows.reduce((a, r) => a + r.tok, 0);
const ms = rows.reduce((a, r) => a + r.ms, 0);
console.log('='.repeat(90));
console.log(`汇总：${rows.length} 条，异常输出 ${bad} 条，总 ${tot} tok / ${ms}ms（均 ${(ms / rows.length).toFixed(0)}ms，${(tot / (ms / 1000)).toFixed(0)} tok/s，PC 仅供参考）`);
console.log('已写入 测试数据/probe/llm_production_prompt.tsv');
