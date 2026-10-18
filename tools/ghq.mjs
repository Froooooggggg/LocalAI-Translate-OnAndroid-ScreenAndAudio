// Compact GitHub API query helper.
// usage: node ghq.mjs <url> [path1 path2 ...]
// A path is a dotted selector, e.g. "title", "merged_at", "items[0].sha", "body:600" (trailing :N truncates strings).
const [, , url, ...paths] = process.argv;

function get(o, p) {
  if (p.startsWith('[]')) {
    const rest = p.slice(2).replace(/^\./, '');
    const arr = Array.isArray(o) ? o : [];
    return arr.map((el) => {
      let v = rest ? get(el, rest) : el;
      return typeof v === 'object' && v ? JSON.stringify(v).slice(0, 160) : v;
    });
  }
  return p.split('.').reduce((a, k) => {
    if (a == null) return undefined;
    const m = k.match(/^(.*)\[(\d+)\]$/);
    if (m) {
      const arr = a[m[1]];
      return Array.isArray(arr) ? arr[Number(m[2])] : undefined;
    }
    return a[k];
  }, o);
}

const res = await fetch(url, {
  headers: { Accept: 'application/vnd.github+json', 'User-Agent': 'research-agent' },
});
const text = await res.text();
let j;
try { j = JSON.parse(text); } catch { console.log('HTTP', res.status, text.slice(0, 300)); process.exit(0); }
if (res.status >= 400) { console.log('HTTP', res.status, JSON.stringify(j).slice(0, 400)); process.exit(0); }

const out = {};
if (!paths.length) {
  console.log(JSON.stringify(j).slice(0, 3000));
} else {
  for (const raw of paths) {
    const tm = raw.match(/^(.*):(\d+)$/);
    const p = tm ? tm[1] : raw;
    const lim = tm ? Number(tm[2]) : 240;
    let v = get(j, p);
    if (typeof v === 'string' && v.length > lim) v = v.slice(0, lim).replace(/\s+/g, ' ') + '…';
    if (Array.isArray(v)) v = v.map(x => (typeof x === 'object' && x ? (x.tag_name || x.name || x.sha || JSON.stringify(x).slice(0, 120)) : x));
    out[p] = v;
  }
  console.log(JSON.stringify(out, null, 1));
}
