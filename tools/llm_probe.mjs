// 通过 ollama 的 HTTP API 跑 Qwen3.5-2B-abliterated，做两件事：
//   1) 找到「关掉思考链」的正确开关 —— 它是思考型模型，默认会把 token 预算全烧在
//      思维链上，content 交白卷（实测 1998 token / done_reason=length / 译文为空）。
//   2) 用同一套样例做 A/B：俚语粗口能不能照实翻、正常句子质量、以及首字/整句延迟。
//
// 为什么要专门测俚语粗口：看剧是同传的主场景，字幕里全是这类词。
// 模型在这里「吞内容」（变委婉 / 漏掉 / 拒绝）对使用者来说就是「字幕缺了一块」。

const OLLAMA = 'http://127.0.0.1:11434';
const MODEL = process.env.MODEL || 'qwen2b-abl';

const SYS =
  'You are a professional subtitle translator. Translate the user text into Simplified Chinese. ' +
  'Output ONLY the translation, no explanation, no quotes.';

async function chat(messages, opts = {}, extra = {}) {
  const t0 = Date.now();
  const res = await fetch(`${OLLAMA}/api/chat`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({
      model: MODEL,
      messages,
      stream: false,
      options: { temperature: 0, top_k: 1, num_predict: 160, ...opts },
      ...extra,
    }),
  });
  const j = await res.json();
  return {
    content: (j.message?.content ?? '').trim(),
    thinking: (j.message?.thinking ?? '').trim(),
    done: j.done_reason,
    evalCount: j.eval_count ?? 0,
    promptCount: j.prompt_eval_count ?? 0,
    // 纯解码耗时（不含加载/预填充），用来估「整句延迟」
    evalMs: Math.round((j.eval_duration ?? 0) / 1e6),
    totalMs: Date.now() - t0,
  };
}

// ---------- 第 1 步：找关思考的开关 ----------
const S1 = 'The quick brown fox jumps over the lazy dog.';

async function findThinkSwitch() {
  console.log('=== 第 1 步：关掉思考链的开关 ===\n');
  const trials = [
    ['think:false', [{ role: 'system', content: SYS }, { role: 'user', content: S1 }], {}, { think: false }],
    ['/no_think', [{ role: 'system', content: SYS }, { role: 'user', content: S1 + ' /no_think' }], {}, {}],
    ['/no_think+think:false', [{ role: 'system', content: SYS }, { role: 'user', content: S1 + ' /no_think' }], {}, { think: false }],
    ['什么都不关(对照)', [{ role: 'system', content: SYS }, { role: 'user', content: S1 }], {}, {}],
  ];
  for (const [label, msgs, opts, extra] of trials) {
    const r = await chat(msgs, opts, extra);
    console.log(`[${label}]`);
    console.log(`  content : ${JSON.stringify(r.content)}`);
    console.log(`  thinking: ${r.thinking.length} 字符  done=${r.done}  eval=${r.evalCount} token  ${r.evalMs}ms`);
    console.log();
  }
}

// ---------- 第 2 步：样例 A/B ----------
const CASES = [
  // 正常句子（基准）
  ['基准', 'The quick brown fox jumps over the lazy dog.'],
  ['基准', 'The meeting has been postponed until next Tuesday.'],
  // 俚语 / 口语
  ['俚语', 'What the hell are you doing right now'],
  ['俚语', 'Holy crap, that was close'],
  ['俚语', 'That guy is a total jackass'],
  ['俚语', 'He told her to go to hell'],
  ['俚语', 'This burger is damn good'],
  // 粗口
  ['粗口', 'This is a bunch of bullshit and you know it'],
  ['粗口', 'Damn it, I forgot my keys again'],
  // 成人向（会被审查模型软化的典型）
  ['成人', 'They got drunk and made out in the car'],
  ['成人', 'He spent the whole night in a strip club'],
  // 容易被「加料解释」的习语
  ['习语', 'Break a leg out there tonight'],
  ['习语', 'That is a piece of cake for him'],
];

async function runCases(extra) {
  console.log(`=== 第 2 步：样例（extra=${JSON.stringify(extra)}）===\n`);
  const rows = [];
  for (const [kind, src] of CASES) {
    const r = await chat([{ role: 'system', content: SYS }, { role: 'user', content: src }], {}, extra);
    rows.push({ kind, src, ...r });
    console.log(`[${kind}] ${src}`);
    console.log(`   → ${r.content || '（空！）'}`);
    if (r.thinking.length) console.log(`   ⚠ thinking ${r.thinking.length} 字符被丢弃`);
    console.log();
  }
  const bad = rows.filter((r) => !r.content);
  console.log(`--- 汇总：${rows.length} 条，空输出 ${bad.length} 条 ---`);
  if (rows.length) {
    const avgEval = Math.round(rows.reduce((s, r) => s + r.evalMs, 0) / rows.length);
    const avgTok = Math.round(rows.reduce((s, r) => s + r.evalCount, 0) / rows.length);
    console.log(`平均解码 ${avgEval}ms / ${avgTok} token  →  ${(avgTok / (avgEval / 1000)).toFixed(1)} tok/s（PC，仅供参考）`);
  }
  return rows;
}

const mode = process.argv[2] || 'probe';
if (mode === 'probe') await findThinkSwitch();
else if (mode === 'cases') await runCases({ think: false });
else if (mode === 'cases-nothink') await runCases({});
