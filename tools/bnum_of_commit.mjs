// Binary-search which llama.cpp build number (bXXXX) first contains a given commit.
// usage: node bnum_of_commit.mjs <sha> <lo> <hi>
const [sha, loArg, hiArg] = process.argv.slice(2);
let lo = Number(loArg), hi = Number(hiArg);
const H = { Accept: 'application/vnd.github+json', 'User-Agent': 'research-agent' };

async function contains(b) {
  const r = await fetch(`https://api.github.com/repos/ggml-org/llama.cpp/compare/${sha}...b${b}`, { headers: H });
  const j = await r.json();
  if (!j.status) return { b, status: 'MISSING' };
  return { b, status: j.status, behind: j.behind_by, ahead: j.ahead_by };
}

while (lo < hi) {
  let mid = Math.floor((lo + hi) / 2);
  let c = await contains(mid);
  let tries = 0;
  while (c.status === 'MISSING' && tries < 6) { mid += 1; tries++; c = await contains(mid); }
  const isAfter = c.status === 'ahead' && c.behind === 0;
  console.log(`probe b${c.b}${c.b !== mid ? ' (from b' + mid + ')' : ''}: ${c.status} behind=${c.behind} ahead=${c.ahead} -> ${isAfter ? 'CONTAINS' : 'does not contain'}`);
  if (isAfter) hi = c.b; else lo = c.b + 1;
}
const final = await contains(lo);
console.log('ANSWER first build containing', sha, '=>', JSON.stringify(final));
