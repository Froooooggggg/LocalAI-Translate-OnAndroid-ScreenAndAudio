// Find the first llama.cpp release published after a given ISO date.
// usage: node release_after.mjs 2026-02-10T16:00:26Z
const target = new Date(process.argv[2]);
const H = { Accept: 'application/vnd.github+json', 'User-Agent': 'research-agent' };

let page = 1;
let candidates = [];
while (page <= 6) {
  const r = await fetch(`https://api.github.com/repos/ggml-org/llama.cpp/releases?per_page=100&page=${page}`, { headers: H });
  const j = await r.json();
  if (!Array.isArray(j) || !j.length) break;
  for (const rel of j) {
    const p = new Date(rel.published_at);
    if (p >= target) candidates.push({ tag: rel.tag_name, published: rel.published_at, name: rel.name });
  }
  const oldest = new Date(j[j.length - 1].published_at);
  if (oldest < target) break;
  page++;
}
candidates.sort((a, b) => new Date(a.published) - new Date(b.published));
console.log('target:', target.toISOString());
console.log('total candidates after target:', candidates.length);
console.log('earliest 3:', JSON.stringify(candidates.slice(0, 3), null, 1));
if (candidates.length) {
  // show the last release BEFORE it too
  const r = await fetch('https://api.github.com/repos/ggml-org/llama.cpp/releases?per_page=100&page=' + page, { headers: H });
  const j = await r.json();
  const before = (Array.isArray(j) ? j : []).filter(x => new Date(x.published_at) < target)
    .sort((a, b) => new Date(b.published_at) - new Date(a.published_at)).slice(0, 2);
  console.log('last 2 before:', JSON.stringify(before.map(x => ({ tag: x.tag_name, published: x.published_at })), null, 1));
}
