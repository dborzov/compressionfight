// Choose Your Codec — the data sheet: what the sample data is made of, printed as the bytes it is and measured.
// Contract (docs/ARCHITECTURE.md):
//   CYC.DataSheet.render(container, kind, opts)        full sheet: literal, picture, measured facts, what fits / fails
//   CYC.DataSheet.renderMini(container, kind)          two-line preview for task cards
//   CYC.DataSheet.renderCross(container, opts)         every codec's pipeline on every dataset (live engine runs)
//   CYC.DataSheet.profile(kind)                        the measured facts (cached per kind)
//   CYC.DataSheet.calc.*                               pure functions, testable in node (no DOM)
// Numbers shown to the player are computed from the sample by the stated formula; the prose comes from CYC.DATAKINDS.
// Plain ES2020, no dependencies. Needs CYC.Engine.payload(kind) and CYC.Engine.run; CYC.Viz for the canvases.

window.CYC = window.CYC || {};

(function () {
  'use strict';
  const CYC = window.CYC;
  const calc = {};

  // ============================================================ pure math
  const log2 = (x) => Math.log(x) / Math.LN2;
  const fmt = (x, d) => Number(x).toLocaleString('en-US', { maximumFractionDigits: d == null ? 2 : d, minimumFractionDigits: d == null ? 0 : d });
  const fmtInt = (x) => Number(Math.round(x)).toLocaleString('en-US');
  const pct = (x, d) => `${fmt(100 * x, d == null ? 0 : d)}%`;

  calc.histogram = function (values) { const m = new Map(); for (let i = 0; i < values.length; i++) { const v = values[i]; m.set(v, (m.get(v) || 0) + 1); } return m; };
  calc.entropy = function (values) { // order-0 Shannon entropy in bits/symbol
    const m = calc.histogram(values); const n = values.length || 1; let H = 0;
    for (const c of m.values()) { const p = c / n; H -= p * log2(p); }
    return { H, alphabet: m.size, n: values.length };
  };
  calc.diffs = function (values, order) { // order 1: x[n] − x[n−1]; order 2: x[n] − (2x[n−1] − x[n−2]). First samples are kept as-is.
    const out = new Float64Array(values.length);
    for (let i = 0; i < values.length; i++) {
      if (order === 2) out[i] = i < 2 ? (i === 1 ? values[1] - values[0] : values[0]) : values[i] - (2 * values[i - 1] - values[i - 2]);
      else out[i] = i === 0 ? values[0] : values[i] - values[i - 1];
    }
    return out;
  };
  calc.rowDiffs = function (plane, mode) { // per-row residuals of a 2-D plane: 'sub' (left neighbour) or 'up' (pixel above). Row 0 of 'up' is kept raw.
    const { w, h, data } = plane; const out = new Float64Array(w * h);
    for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
      const i = y * w + x;
      if (mode === 'up') out[i] = y === 0 ? data[i] : data[i] - data[i - w];
      else out[i] = x === 0 ? data[i] : data[i] - data[i - 1];
    }
    return out;
  };
  calc.shareWithin = function (values, t) { let k = 0; for (let i = 0; i < values.length; i++) if (Math.abs(values[i]) <= t) k++; return values.length ? k / values.length : 0; };
  calc.distinct = function (values) { return calc.histogram(values).size; };

  // Greedy LZ77-style pass: at each position take the longest earlier match within `window`; returns the matches and coverage.
  calc.repeats = function (bytes, opts) {
    opts = opts || {}; const minLen = opts.minLen || 3, window = opts.window || 4096, maxLen = opts.maxLen || 258;
    const n = bytes.length; const matches = []; let covered = 0; let i = 0;
    while (i < n) {
      let best = 0, bestDist = 0; const start = Math.max(0, i - window);
      for (let j = i - 1; j >= start; j--) {
        if (bytes[j] !== bytes[i]) continue;
        let l = 1; while (l < maxLen && i + l < n && bytes[j + l] === bytes[i + l]) l++;
        if (l > best) { best = l; bestDist = i - j; if (l >= maxLen) break; }
      }
      if (best >= minLen) { matches.push({ pos: i, dist: bestDist, len: best }); covered += best; i += best; } else i++;
    }
    const top = matches.slice().sort((a, b) => b.len - a.len || a.pos - b.pos).slice(0, opts.top || 5);
    return { matches, coverage: n ? covered / n : 0, top, n };
  };

  // ---- literals (how the bytes print) ----
  const ESC = { 9: '\\t', 10: '\\n', 13: '\\r', 92: '\\\\', 39: "\\'" };
  calc.pyChar = function (b) { if (ESC[b]) return ESC[b]; if (b >= 32 && b < 127) return String.fromCharCode(b); return '\\x' + b.toString(16).padStart(2, '0'); };
  calc.pyBytes = function (bytes, start, count) { // python bytes literal b'...' of bytes[start, start+count)
    start = start || 0; const end = Math.min(bytes.length, start + (count == null ? bytes.length : count));
    let s = "b'"; for (let i = start; i < end; i++) s += calc.pyChar(bytes[i]);
    return s + "'" + (end < bytes.length ? ' …' : '');
  };
  calc.pyList = function (values, count, digits) {
    const n = Math.min(values.length, count == null ? values.length : count); const out = [];
    for (let i = 0; i < n; i++) out.push(digits == null ? String(values[i]) : Number(values[i]).toFixed(digits));
    return '[' + out.join(', ') + (n < values.length ? ', …' : '') + ']';
  };
  calc.hexRow = function (bytes, start, count) { let s = ''; for (let i = start; i < Math.min(bytes.length, start + count); i++) s += bytes[i].toString(16).padStart(2, '0') + ' '; return s.trim(); };

  // ---- the measured profile of one payload ----
  calc.profile = function (payload) {
    const P = { kind: payload.kind, domain: payload.domain, label: payload.label, n: payload.n, rawBits: payload.rawBits, bitDepth: payload.bitDepth, bytes: payload.bytes.length, bytesWidth: payload.bytesWidth, facts: [] };
    const bytes = payload.bytes;
    const lz = calc.repeats(bytes, { minLen: 3, window: 4096, top: 5 });
    P.lz = { coverage: lz.coverage, matches: lz.matches.length, top: lz.top, long: lz.matches.filter((m) => m.len >= 8) };
    const byteH = calc.entropy(bytes); P.byteEntropy = byteH.H; P.byteAlphabet = byteH.alphabet;
    if (payload.domain === 'text') {
      const d1 = calc.diffs(bytes, 1); const dH = calc.entropy(d1);
      P.diffEntropy = dH.H; P.diffAlphabet = dH.alphabet;
      const t = lz.top[0];
      P.facts = [
        { n: fmtInt(bytes.length), t: 'bytes, 8 bits each' },
        { n: `${byteH.alphabet} of 256`, t: 'byte values in use; no value dominates', lead: true },
        { n: `${fmt(byteH.H, 2)} bits/byte`, t: 'order-0 entropy of the bytes (fixed-length code: 8)' },
        t ? { n: `${t.len} bytes`, t: `longest exact repeat, ${fmtInt(t.dist)} bytes back: ${calc.pyBytes(bytes, t.pos, Math.min(t.len, 28)).replace(/ …$/, '')}${t.len > 28 ? '…' : ''}` } : null,
        { n: pct(lz.coverage), t: `of the bytes are inside a copy of something earlier (greedy LZ77, window 4096, min 3; ${fmtInt(lz.matches.length)} matches)` },
        { n: `${fmt(dH.H, 2)} bits`, t: `entropy of x[n] − x[n−1]: ${dH.H > byteH.H ? 'wider than the bytes themselves; subtracting neighbours does not predict labels' : 'narrower than the bytes'}`, bad: dH.H > byteH.H, lead: true },
      ].filter(Boolean);
    } else if (payload.domain === 'image' || payload.domain === 'video') {
      const planes = payload.planes.filter((p) => p.h > 1);
      let allVals = 0, zeroUp = 0, nearUp = 0, zeroSub = 0, nearSub = 0; const upAll = [], rawAll = [];
      for (const p of planes) {
        const up = calc.rowDiffs(p, 'up'), sub = calc.rowDiffs(p, 'sub');
        for (let i = 0; i < up.length; i++) { const y = Math.floor(i / p.w), x = i % p.w; if (y > 0) { allVals++; if (up[i] === 0) zeroUp++; if (Math.abs(up[i]) <= 2) nearUp++; upAll.push(up[i]); } if (x > 0) { if (sub[i] === 0) zeroSub++; if (Math.abs(sub[i]) <= 2) nearSub++; } }
        for (let i = 0; i < p.data.length; i++) rawAll.push(p.data[i]);
      }
      const rawH = calc.entropy(rawAll), upH = calc.entropy(upAll);
      P.pixelEntropy = rawH.H; P.upEntropy = upH.H; P.zeroUp = zeroUp / (allVals || 1); P.nearUp = nearUp / (allVals || 1); P.zeroSub = zeroSub / (allVals || 1); P.nearSub = nearSub / (allVals || 1);
      const t = lz.top[0];
      P.facts = [
        { n: `${planes[0].w}×${planes[0].h}×${planes.length}`, t: `samples, ${payload.bitDepth} bits each (${fmtInt(payload.n)} values)` },
        { n: `${fmt(rawH.H, 2)} bits/sample`, t: `order-0 entropy of the pixel values (${rawH.alphabet} distinct values)` },
        { n: pct(P.zeroUp), t: `of pixels equal the pixel above exactly; ${pct(P.nearUp)} are within ±2 (the "Up" filter's residual)`, lead: true },
        { n: pct(P.zeroSub), t: `of pixels equal their left neighbour exactly; ${pct(P.nearSub)} are within ±2 (the "Sub" filter's residual)` },
        { n: `${fmt(upH.H, 2)} bits`, t: `entropy of the Up residual x[y][x] − x[y−1][x]${upH.H < rawH.H ? ', narrower than the pixels' : ''}` },
        { n: pct(lz.coverage), t: `of the raw bytes are inside an exact copy of something earlier (greedy LZ77, window 4096, min 3)${t ? `; longest repeat ${t.len} bytes, ${t.dist === 1 ? 'a run of one value' : `${fmtInt(t.dist)} back`}` : ''}`, bad: lz.coverage < 0.25, lead: true },
      ];
    } else if (payload.domain === 'audio') {
      const vals = []; for (const p of payload.planes) for (let i = 0; i < p.data.length; i++) vals.push(p.data[i]);
      const rawH = calc.entropy(vals);
      const d1 = calc.diffs(payload.planes[0].data, 1), d2 = calc.diffs(payload.planes[0].data, 2);
      const d1H = calc.entropy(d1), d2H = calc.entropy(d2);
      P.sampleEntropy = rawH.H; P.distinct = rawH.alphabet; P.d1Entropy = d1H.H; P.d2Entropy = d2H.H; P.d2Within256 = calc.shareWithin(d2, 256); P.d1Within256 = calc.shareWithin(d1, 256);
      P.facts = [
        { n: fmtInt(payload.n), t: `samples of ${payload.bitDepth} bits (${payload.planes.length} channels × ${fmtInt(payload.planes[0].data.length)})` },
        { n: fmtInt(rawH.alphabet), t: `distinct sample values out of ${fmtInt(Math.pow(2, payload.bitDepth))}; most appear once`, lead: true },
        { n: `${fmt(rawH.H, 2)} bits/sample`, t: 'order-0 entropy of the raw samples: Huffman on them would need a table nearly as big as the data' },
        { n: `${fmt(d1H.H, 2)} bits`, t: `entropy of x[n] − x[n−1]; ${pct(P.d1Within256)} of those residuals are within ±256` },
        { n: `${fmt(d2H.H, 2)} bits`, t: `entropy of x[n] − (2x[n−1] − x[n−2]), the order-2 fixed predictor; ${pct(P.d2Within256)} within ±256 (a spike around zero)` },
        { n: pct(lz.coverage, 1), t: 'of the bytes are inside an exact copy of something earlier (greedy LZ77, window 4096, min 3): nothing to match', bad: true, lead: true },
      ];
    } else { // vectors, weights, activations: floats
      const vals = []; for (const p of payload.planes) for (let i = 0; i < p.data.length; i++) vals.push(p.data[i]);
      let s = 0, s2 = 0; for (const v of vals) { s += v; s2 += v * v; } const mean = s / (vals.length || 1); const sd = Math.sqrt(Math.max(0, s2 / (vals.length || 1) - mean * mean));
      const d1 = calc.diffs(vals, 1); let sd1 = 0; for (let i = 1; i < d1.length; i++) sd1 += d1[i] * d1[i]; sd1 = Math.sqrt(sd1 / Math.max(1, d1.length - 1));
      P.mean = mean; P.sd = sd; P.diffSd = sd1;
      P.facts = [
        { n: fmtInt(payload.n), t: `values of ${payload.bitDepth} bits` },
        { n: `σ = ${fmt(sd, 4)}`, t: `standard deviation around a mean of ${fmt(mean, 4)}` },
        { n: `σ = ${fmt(sd1, 4)}`, t: `standard deviation of x[n] − x[n−1]: ${sd1 > sd ? 'wider than the data; neighbours do not predict each other' : 'narrower than the data'}`, bad: sd1 > sd, lead: true },
        { n: pct(lz.coverage, 1), t: 'of the bytes are inside an exact copy of something earlier (greedy LZ77): the shared sign and exponent bytes of nearby values recur, the mantissa bytes do not' },
        { n: `${fmt(byteH.H, 2)} bits/byte`, t: `order-0 entropy of the bytes (fixed: 8)${deflateNote(payload.kind)}` },
      ];
    }
    return P;
  };

  function deflateNote(kind) { // what the level-1 byte coder actually gets on this data, measured live
    try { const r = CYC.Engine.run(kind, { predict: ['lz77'], entropy: ['huffman'] }); const ratio = r.rawBits / r.finalBits; return `; LZ77 + Huffman on these bytes: ${fmt(ratio, 2)}×${ratio < 1 ? ' (larger than the input: the Huffman tables cost more than they save)' : ''}`; } catch (e) { return ''; }
  }

  // ---- every codec's pipeline on every dataset ----
  calc.crossMatrix = function (kinds, pipelines, runFn) {
    return kinds.map((kind) => {
      const cells = pipelines.map((p) => {
        const r = runFn(kind, p.squad);
        return { kind, pipeline: p.id, ratio: r.finalBits ? r.rawBits / r.finalBits : 0, rawBits: r.rawBits, finalBits: r.finalBits, lossless: r.lossless, home: p.home === kind };
      });
      const best = Math.max(...cells.map((c) => c.ratio));
      cells.forEach((c) => { c.best = c.ratio === best; });
      return { kind, cells };
    });
  };

  // ============================================================ DOM
  function h(tag, attrs, ...kids) {
    const n = document.createElement(tag);
    if (attrs) for (const [k, v] of Object.entries(attrs)) {
      if (v == null || v === false) continue;
      if (k === 'class') n.className = v;
      else if (k === 'text') n.textContent = v;
      else if (k === 'html') n.innerHTML = v;
      else if (k.startsWith('on') && typeof v === 'function') n.addEventListener(k.slice(2), v);
      else if (k === 'style' && typeof v === 'object') { for (const [p, val] of Object.entries(v)) { if (p.startsWith('--')) n.style.setProperty(p, val); else n.style[p] = val; } }
      else n.setAttribute(k, v === true ? '' : v);
    }
    for (const kid of kids.flat(Infinity)) { if (kid == null || kid === false) continue; n.append(kid.nodeType ? kid : document.createTextNode(String(kid))); }
    return n;
  }
  const engineOK = () => !!(CYC.Engine && typeof CYC.Engine.payload === 'function' && typeof CYC.Engine.run === 'function');
  const vizOK = () => !!(CYC.Viz && typeof CYC.Viz.draw === 'function');
  const info = (kind) => (CYC.DATAKINDS && CYC.DATAKINDS[kind]) || { name: kind, shape: '', structure: '', rightFit: '', wrongFits: [], realWorld: '' };
  const payloadCache = {}, profileCache = {};
  function payload(kind) { if (!payloadCache[kind]) payloadCache[kind] = CYC.Engine.payload(kind); return payloadCache[kind]; }
  function profile(kind) { if (!profileCache[kind]) profileCache[kind] = calc.profile(payload(kind)); return profileCache[kind]; }

  // ---- literal blocks: the data printed as what it is ----
  function textLiteral(p, prof, count) {
    // the first `count` bytes as a python bytes literal; copies found by the greedy pass are marked, with their source
    const bytes = p.bytes; const n = Math.min(bytes.length, count);
    const marks = new Array(n).fill(0); const titles = new Array(n).fill(null);
    for (const m of prof.lz.top.concat(prof.lz.long)) {
      if (m.pos >= n) continue;
      for (let i = m.pos; i < Math.min(n, m.pos + m.len); i++) { if (!marks[i]) { marks[i] = 2; titles[i] = `copy of ${m.len} bytes from ${fmtInt(m.dist)} bytes back`; } }
      for (let i = m.pos - m.dist; i < Math.min(n, m.pos - m.dist + m.len); i++) if (!marks[i]) { marks[i] = 1; titles[i] = `source of a later copy (${m.len} bytes)`; }
    }
    const code = h('code', {}); code.append("data = b'");
    let i = 0;
    while (i < n) {
      let j = i; while (j < n && marks[j] === marks[i] && titles[j] === titles[i]) j++;
      let s = ''; for (let k = i; k < j; k++) s += calc.pyChar(bytes[k]);
      if (marks[i] === 2) code.append(h('mark', { class: 'ds-rep', title: titles[i] }, s));
      else if (marks[i] === 1) code.append(h('span', { class: 'ds-src', title: titles[i] }, s));
      else code.append(s);
      i = j;
    }
    code.append("'" + (n < bytes.length ? `  # … ${fmtInt(bytes.length - n)} more bytes` : ''));
    return h('pre', { class: 'ds-literal' }, code);
  }
  function numRow(label, values, hi, sep) { // one "name = [..]" line with small/zero values marked
    const code = h('code', {}); code.append(`${label} = [`);
    values.forEach((v, i) => {
      const s = (typeof v === 'string') ? v : String(v);
      const cell = hi && typeof v === 'number' && hi(v) ? h('mark', { class: 'ds-zero' }, s) : document.createTextNode(s);
      code.append(cell); if (i < values.length - 1) code.append(sep || ', ');
    });
    code.append(']'); return code;
  }
  function imageLiteral(p, y) {
    const R = p.planes[0]; y = Math.min(R.h - 1, Math.max(1, y == null ? 12 : y)); const w = R.w;
    const row = (yy) => Array.from(R.data.subarray(yy * w, yy * w + w));
    const r0 = row(y - 1), r1 = row(y);
    const up = r1.map((v, x) => v - r0[x]); const sub = r1.map((v, x) => (x === 0 ? v : v - r1[x - 1]));
    const pre = h('pre', { class: 'ds-literal' });
    pre.append(numRow(`${R.name}[${y - 1}]`, r0), '\n', numRow(`${R.name}[${y}]`, r1), '\n',
      h('span', { class: 'ds-comment' }, `# PNG "Up" filter: ${R.name}[${y}] − ${R.name}[${y - 1}]`), '\n',
      numRow('up', up, (v) => v === 0), '\n',
      h('span', { class: 'ds-comment' }, `# PNG "Sub" filter: ${R.name}[${y}][x] − ${R.name}[${y}][x−1]`), '\n',
      numRow('sub', sub, (v) => v === 0), '\n',
      h('span', { class: 'ds-comment' }, `# rows ${y - 1} and ${y} of the ${R.name} channel, ${w} pixels each; ${p.planes.length} channels, ${R.h} rows`));
    return pre;
  }
  function audioLiteral(p, count) {
    const L = p.planes[0]; count = count || 20;
    const vals = Array.from(L.data.subarray(0, count));
    const d2 = calc.diffs(L.data, 2); const res = Array.from(d2.subarray(0, count)).map((v, i) => (i < 2 ? '·' : v));
    const pre = h('pre', { class: 'ds-literal' });
    pre.append(numRow(`${L.name}`, vals), '\n',
      h('span', { class: 'ds-comment' }, `# FLAC's order-2 fixed predictor: ${L.name}[n] − (2·${L.name}[n−1] − ${L.name}[n−2])`), '\n',
      numRow('residual', res, (v) => Math.abs(v) <= 256), '\n',
      h('span', { class: 'ds-comment' }, `# first ${count} of ${fmtInt(L.data.length)} samples per channel, signed ${p.bitDepth}-bit; ${p.planes.length} channels`));
    return pre;
  }
  function floatLiteral(p, count) {
    const P0 = p.planes[0]; count = count || 12;
    const vals = Array.from(P0.data.subarray(0, count)).map((v) => Number(v).toPrecision(4));
    const pre = h('pre', { class: 'ds-literal' });
    pre.append(numRow(`${P0.name}`, vals), '\n',
      h('span', { class: 'ds-comment' }, `# first ${count} of ${fmtInt(p.n)} values, ${p.bitDepth}-bit floats; ${p.planes.length} plane${p.planes.length === 1 ? '' : 's'} of ${P0.w}×${P0.h}`));
    return pre;
  }
  function literalFor(kind, p, prof, opts) {
    const d = info(kind);
    if (d.literal === 'rows' || p.domain === 'image' || p.domain === 'video') return imageLiteral(p, opts.row != null ? opts.row : d.row);
    if (d.literal === 'samples' || p.domain === 'audio') return audioLiteral(p, opts.count);
    if (d.literal === 'bytes' || p.domain === 'text') return textLiteral(p, prof, opts.count || 520);
    return floatLiteral(p, opts.count);
  }

  // ---- the picture ----
  function vizFor(p) {
    if (p.domain === 'image' || p.domain === 'video') {
      const P = p.planes.filter((q) => q.frame === undefined || q.frame === 0).slice(0, 3); const w = P[0].w, hh = P[0].h;
      const rgb = new Uint8ClampedArray(w * hh * 4);
      for (let i = 0; i < w * hh; i++) { rgb[i * 4] = P[0].data[i]; rgb[i * 4 + 1] = (P[1] || P[0]).data[i]; rgb[i * 4 + 2] = (P[2] || P[0]).data[i]; rgb[i * 4 + 3] = 255; }
      return { type: 'grid', planes: [], rgb: { w, h: hh, data: rgb, name: 'RGB' } };
    }
    if (p.domain === 'audio') {
      let peak = 1; for (const q of p.planes) for (let i = 0; i < q.data.length; i++) peak = Math.max(peak, Math.abs(q.data[i]));
      return { type: 'waveform', series: p.planes.map((q) => Float32Array.from(q.data, (v) => v / peak)), labels: p.planes.map((q) => q.name) };
    }
    if (p.domain === 'text') {
      const hist = calc.histogram(p.bytes); const bins = [], labels = [];
      for (let b = 32; b < 127; b++) { bins.push(hist.get(b) || 0); labels.push(String.fromCharCode(b)); }
      return { type: 'histogram', bins, labels, entropyBitsPerSymbol: calc.entropy(p.bytes).H, fixedBits: 8 };
    }
    const vals = []; for (const q of p.planes) for (let i = 0; i < q.data.length; i++) vals.push(q.data[i]);
    const lo = Math.min(...vals), hi = Math.max(...vals); const B = 48; const bins = new Array(B).fill(0);
    for (const v of vals) bins[Math.min(B - 1, Math.floor((v - lo) / ((hi - lo) || 1) * B))]++;
    return { type: 'histogram', bins, labels: [fmt(lo, 3), fmt(hi, 3)], entropyBitsPerSymbol: calc.entropy(vals).H, fixedBits: p.bitDepth };
  }
  function drawLater(canvas, viz) {
    const draw = () => { try { if (vizOK() && canvas.isConnected && canvas.clientWidth) CYC.Viz.draw(canvas, viz); } catch (e) { if (window.console) console.error('DataSheet viz', e); } };
    requestAnimationFrame(draw);
    if (typeof ResizeObserver === 'function') { const ro = new ResizeObserver(draw); ro.observe(canvas); }
    return canvas;
  }

  // ---- renderers ----
  function fitBlock(kind, d, opts) {
    const f = CYC.byId || {}; const pred = d.predictor && f[d.predictor], coder = d.coder && f[d.coder];
    const slug = d.article && CYC.MAP ? `${CYC.MAP.base || 'map/'}zoo/${d.article}/` : null;
    return h('div', { class: 'ds-fit' },
      d.structure ? h('p', { class: 'ds-structure' }, h('b', {}, 'What the structure is. '), d.structure) : null,
      d.rightFit ? h('p', { class: 'ds-right' }, h('b', {}, 'What fits. '), d.rightFit,
        pred || coder ? h('span', { class: 'ds-picks' }, ' ', pred ? h('span', { class: 'chip', style: { '--rc': (CYC.ROLES[pred.role] || {}).color || '' }, text: pred.name }) : null, coder ? h('span', { class: 'chip', style: { '--rc': (CYC.ROLES[coder.role] || {}).color || '' }, text: coder.name }) : null) : null) : null,
      d.wrongFits && d.wrongFits.length ? h('details', { class: 'ds-wrong', open: opts.openWrong ? true : null }, h('summary', {}, 'What does not fit, and why'), h('ul', {}, d.wrongFits.map((t) => h('li', { text: t })))) : null,
      d.realWorld ? h('p', { class: 'ds-real' }, h('b', {}, 'In the real world. '), d.realWorld) : null,
      slug ? h('a', { class: 'map-link map-link-sm', href: slug, text: 'Read the introduction in the Map →' }) : null);
  }
  function render(container, kind, opts) {
    opts = opts || {};
    const d = info(kind);
    const root = h('section', { class: `datasheet ${opts.compact ? 'datasheet-compact' : ''}`, 'data-kind': kind });
    if (!engineOK()) { root.append(h('div', { class: 'placeholder', text: 'Data sheet: engine not loaded (js/engine.js).' })); container.append(root); return root; }
    let p, prof; try { p = payload(kind); prof = profile(kind); } catch (e) { root.append(h('div', { class: 'callout callout-red', text: `Data sheet failed: ${e.message || e}` })); container.append(root); return root; }
    root.append(h('div', { class: 'ds-head' },
      h('span', { class: 'eyebrow', text: opts.eyebrow || 'The data' }),
      h('h4', { class: 'ds-name', text: d.name || kind }),
      h('div', { class: 'ds-shape dim', text: `${d.shape ? d.shape + ' · ' : ''}${p.label}` })));
    const canvas = h('canvas', { class: 'ds-canvas', width: 320, height: 150, 'aria-label': `Picture of the ${kind} sample` });
    root.append(h('div', { class: 'ds-grid' },
      h('div', { class: 'ds-col-literal' }, h('div', { class: 'ds-sub eyebrow', text: 'Printed as what it is' }), literalFor(kind, p, prof, opts)),
      h('div', { class: 'ds-col-facts' },
        drawLater(canvas, vizFor(p)),
        h('div', { class: 'ds-sub eyebrow', text: 'Measured on this sample' }),
        h('ul', { class: 'ds-facts' }, prof.facts.map((f) => h('li', { class: f.bad ? 'is-bad' : '' }, h('b', { class: 'ds-num', text: f.n }), ' ', f.t))))));
    if (!opts.noFit) root.append(fitBlock(kind, d, opts));
    container.append(root);
    return root;
  }
  function renderMini(container, kind) {
    const d = info(kind);
    const root = h('div', { class: 'ds-mini', 'data-kind': kind });
    if (!engineOK()) { root.append(h('div', { class: 'dim', text: `Sample: ${kind}` })); container.append(root); return root; }
    let p, prof; try { p = payload(kind); prof = profile(kind); } catch (e) { root.append(h('div', { class: 'dim', text: `Sample: ${kind}` })); container.append(root); return root; }
    let pre;
    if (p.domain === 'text') pre = h('pre', { class: 'ds-literal ds-literal-mini' }, h('code', { text: calc.pyBytes(p.bytes, 0, 96) }));
    else if (p.domain === 'image' || p.domain === 'video') { const R = p.planes[0]; const y = d.row != null ? d.row : 12; pre = h('pre', { class: 'ds-literal ds-literal-mini' }, h('code', { text: `${R.name}[${y}] = ${calc.pyList(R.data.subarray(y * R.w, y * R.w + R.w), 14)}` })); }
    else if (p.domain === 'audio') { const L = p.planes[0]; pre = h('pre', { class: 'ds-literal ds-literal-mini' }, h('code', { text: `${L.name} = ${calc.pyList(L.data, 12)}` })); }
    else { const P0 = p.planes[0]; pre = h('pre', { class: 'ds-literal ds-literal-mini' }, h('code', { text: `${P0.name} = ${calc.pyList(Array.from(P0.data.subarray(0, 8)).map((v) => Number(v).toPrecision(3)), 8)}` })); }
    const leads = prof.facts.filter((f) => f.lead).slice(0, 2); if (!leads.length) leads.push(prof.facts[1] || prof.facts[0]);
    root.append(h('div', { class: 'ds-mini-name' }, h('span', { class: 'eyebrow', text: 'The data' }), ' ', h('span', { text: d.name || kind })), h('div', { class: 'ds-mini-label dim', text: p.label }), pre,
      h('div', { class: 'ds-mini-fact dim' }, leads.map((f, i) => [i ? ' · ' : null, h('b', { class: f.bad ? 'is-bad' : '', text: f.n }), ' ', f.t])));
    container.append(root);
    return root;
  }
  // opts: { codecs: ['deflate','png','flac'] } (uses CYC.CODECS squads and their tasks' payloads) or { kinds, pipelines:[{id,name,sub,squad,home}] }
  //       onPick(kind, pipeline) makes cells clickable; caption overrides the footnote.
  function renderCross(container, opts) {
    opts = opts || {};
    const root = h('div', { class: 'cross-matrix' });
    if (!engineOK()) { root.append(h('div', { class: 'placeholder', text: 'Engine not loaded.' })); container.append(root); return root; }
    let pipelines = opts.pipelines, kinds = opts.kinds;
    if (!pipelines && opts.codecs) {
      pipelines = opts.codecs.map((id) => { const c = (CYC.CODECS || []).find((x) => x.id === id); if (!c) return null; const hz = (CYC.HEISTS || []).find((x) => x.id === c.heist); return { id: c.id, name: c.name, home: hz ? hz.payload : null, squad: c.squad, sub: squadWords(c.squad) }; }).filter(Boolean);
      kinds = kinds || pipelines.map((p) => p.home).filter((k, i, a) => k && a.indexOf(k) === i);
    }
    if (!pipelines || !pipelines.length || !kinds || !kinds.length) { root.append(h('div', { class: 'placeholder', text: 'Nothing to compare.' })); container.append(root); return root; }
    let rows; try { rows = calc.crossMatrix(kinds, pipelines, CYC.Engine.run); } catch (e) { root.append(h('div', { class: 'callout callout-red', text: `Comparison failed: ${e.message || e}` })); container.append(root); return root; }
    const table = h('table', { class: 'cm-table' },
      h('thead', {}, h('tr', {}, h('th', { class: 'cm-corner' }, h('span', { class: 'eyebrow', text: 'dataset ↓ · pipeline →' })),
        pipelines.map((p) => h('th', { class: 'cm-pipe' }, h('div', { class: 'cm-pipe-name', text: p.name }), p.sub ? h('div', { class: 'cm-pipe-sub dim', text: p.sub }) : null)))),
      h('tbody', {}, rows.map((r) => h('tr', {},
        h('th', { class: 'cm-kind' }, h('div', { class: 'cm-kind-name', text: shortName(r.kind) }), h('div', { class: 'cm-kind-sub dim', text: payload(r.kind).label })),
        r.cells.map((c) => {
          const txt = `${fmt(c.ratio, 2)}×`;
          const body = opts.onPick ? h('button', { type: 'button', class: 'cm-btn', title: `Open ${c.pipeline} on this data in the pipeline builder`, onclick: () => opts.onPick(c.kind, c.pipeline) }, txt) : h('span', { class: 'cm-val', text: txt });
          return h('td', { class: `cm-cell ${c.best ? 'is-best' : ''} ${c.home ? 'is-home' : ''}` }, body,
            h('div', { class: 'cm-sub dim', text: `${fmtBytesLocal(c.rawBits)} → ${fmtBytesLocal(c.finalBits)}` }),
            c.home ? h('div', { class: 'cm-tag', text: 'its own data' }) : null);
        })))));
    root.append(h('div', { class: 'cm-scroll' }, table),
      h('p', { class: 'cm-caption dim', text: opts.caption || 'Ratio = raw bits ÷ final bits, computed live by the same engine that runs the tasks, all lossless. Bold: the best pipeline for that row. Frame: the codec on the data it was designed for. Every pipeline wins at home and loses away, because each one is a bet on a structure the other data does not have.' }));
    container.append(root);
    return root;
  }
  function squadWords(squad) { const names = []; for (const role of Object.keys(CYC.ROLES || {})) for (const id of (squad[role] || [])) { const f = CYC.byId && CYC.byId[id]; if (f) names.push(f.name); } return names.join(' + '); }
  function shortName(kind) { const n = info(kind).name || kind; return n; }
  function fmtBytesLocal(bits) { return vizOK() && CYC.Viz.fmtBits ? CYC.Viz.fmtBits(bits) : `${Math.round(bits / 8)} B`; }

  CYC.DataSheet = { calc, render, renderMini, renderCross, profile, payload, info };
})();
