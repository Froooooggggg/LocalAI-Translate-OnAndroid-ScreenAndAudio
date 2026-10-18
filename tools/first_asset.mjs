// Binary-search the first llama.cpp build whose release assets include a given regex.
// usage: node first_asset.mjs <lo> <hi> <regex>
const lo0 = Number(process.argv[2]), hi0 = Number(process.argv[3]);
const re = new RegExp(process.argv[4]);
const H = { Accept: 'application/vnd.github+json', 'User-Agent': 'research-agent' };

const cache = new Map();
async function probe(b) {
  if (cache.has(b)) return cache.get(b);
  const r = await fetch(`https://api.github.com/repos/ggml-org/llama.cpp/releases/tags/b${b}`, { headers: H });
  let v;
  if (r.status === 404) v = { b, exists: false, hit: false };
  else { const j = await r.json(); v = { b, exists: true, published: j.published_at, hit: (j.assets || []).some(a => re.test(a.name)), n: (j.assets || []).length }; }
  cache.set(b, v);
  return v;
}
let lo = lo0, hi = hi0;
while (lo < hi) {
  let mid = Math.floor((lo + hi) / 2);
  let p = await probe(mid);
  let t = 0;
  while (!p.exists && t < 8) { mid++; t++; p = await probe(mid); }
  console.log(`probe b${p.b}${p.b !== mid ? ` (from ${mid})` : ''}: exists=${p.exists} assets=${p.n} hit=${p.hit}`);
  if (p.hit) hi = p.b; else lo = p.b + 1;
}
const f = await probe(lo);
console.log('ANSWER first build with', re, '=>', JSON.stringify(f));
