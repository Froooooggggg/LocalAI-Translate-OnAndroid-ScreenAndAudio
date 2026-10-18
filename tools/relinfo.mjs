// Show llama.cpp release info for given tags, with android-related assets highlighted.
const H = { Accept: 'application/vnd.github+json', 'User-Agent': 'research-agent' };
for (const tag of process.argv.slice(2)) {
  const r = await fetch(`https://api.github.com/repos/ggml-org/llama.cpp/releases/tags/${tag}`, { headers: H });
  if (!r.ok) { console.log(tag, 'HTTP', r.status); continue; }
  const j = await r.json();
  const names = (j.assets || []).map(a => a.name);
  console.log(JSON.stringify({
    tag: j.tag_name,
    published_at: j.published_at,
    name: j.name,
    asset_count: names.length,
    android_assets: names.filter(n => /android|arm64/i.test(n)),
    sample_of_assets: names.slice(0, 6),
  }, null, 1));
}
