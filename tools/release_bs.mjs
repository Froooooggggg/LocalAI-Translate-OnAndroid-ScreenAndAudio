// Binary-search the llama.cpp releases list for the first release published at/after a target date.
// usage: node release_bs.mjs 2026-02-10T16:00:26Z
const target = new Date(process.argv[2]);
const H = { Accept: 'application/vnd.github+json', 'User-Agent': 'research-agent' };

async function page(n) {
  const r = await fetch(`https://api.github.com/repos/ggml-org/llama.cpp/releases?per_page=100&page=${n}`, { headers: H });
  const j = await r.json();
  if (!Array.isArray(j)) throw new Error('bad page ' + n + ' ' + JSON.stringify(j).slice(0, 200));
  return j; // newest -> oldest
}

// find upper bound of pages
let hi = 1;
let first = await page(1);
console.log('newest:', first[0].tag_name, first[0].published_at);
while (true) {
  const p = await page(hi);
  if (!p.length) break;
  const oldest = new Date(p[p.length - 1].published_at);
  if (oldest < target) break;
  hi *= 2;
  if (hi > 512) break;
}
let lo = Math.max(1, hi / 2);
// binary search: smallest page containing a release < target
while (lo < hi) {
  const mid = Math.floor((lo + hi) / 2);
  const p = await page(mid);
  const oldest = new Date(p[p.length - 1].published_at);
  if (oldest < target) hi = mid; else lo = mid + 1;
}
const boundary = await page(lo);
const asc = boundary.slice().reverse(); // oldest -> newest
const idx = asc.findIndex(r => new Date(r.published_at) >= target);
const firstAfter = asc[idx];
const lastBefore = asc[idx - 1];
console.log(JSON.stringify({
  page: lo,
  last_release_before_target: lastBefore ? { tag: lastBefore.tag_name, published: lastBefore.published_at } : null,
  first_release_after_target: firstAfter ? { tag: firstAfter.tag_name, published: firstAfter.published_at, name: firstAfter.name } : null,
}, null, 1));
// also: does the first-after tag actually contain the commit? (checked separately)
