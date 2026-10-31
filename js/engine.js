// Choose Your Codec — engine. Toy but REAL pipeline math on small deterministic payloads.
// Contract: docs/ARCHITECTURE.md ("Engine contract"). Exports CYC.Engine.run(payloadKind, squad) and
// CYC.Engine.payloadLabel(kind). Plain ES2020, no dependencies. Everything here is honest bit counting:
// where a fighter is a stand-in for the real algorithm the caption says so.
(function () {
  'use strict';
  window.CYC = window.CYC || {};

  // ============================================================ utilities
  function mulberry32(seed) {
    let a = seed >>> 0;
    return function () {
      a = (a + 0x6D2B79F5) >>> 0;
      let t = a;
      t = Math.imul(t ^ (t >>> 15), t | 1);
      t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }
  function gaussian(rnd) {
    let u = 0;
    while (u === 0) u = rnd();
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * rnd());
  }
  const log2 = Math.log2;
  const clamp = (v, lo, hi) => (v < lo ? lo : v > hi ? hi : v);
  const fmt = (x, d = 1) => (Number.isFinite(x) ? Number(x).toFixed(d) : String(x));
  const fmtInt = (x) => Math.round(x).toLocaleString('en-US');
  const pct = (x, d = 0) => fmt(100 * x, d) + '%';
  function fmtBytes(b) {
    if (b >= 1e6) return fmt(b / 1e6, b >= 10e6 ? 0 : 1) + ' MB';
    if (b >= 1024) return fmt(b / 1024, 1) + ' KB';
    return fmtInt(b) + ' B';
  }
  const ceilLog2 = (m) => (m <= 1 ? 1 : Math.ceil(log2(m)));
  const smoothstep = (e0, e1, x) => { const t = clamp((x - e0) / (e1 - e0), 0, 1); return t * t * (3 - 2 * t); };
  const isInt = (v) => Math.abs(v - Math.round(v)) < 1e-9;

  // float32 / float16 bit patterns (raw symbol alphabets for unquantized floats)
  const F32 = new Float32Array(1), U32 = new Uint32Array(F32.buffer);
  function f32bits(x) { F32[0] = x; return U32[0]; }
  function toHalfBits(x) {
    F32[0] = x; const b = U32[0];
    const sign = (b >>> 16) & 0x8000, exp = (b >>> 23) & 0xff; let mant = b & 0x7fffff;
    if (exp === 0xff) return sign | 0x7c00 | (mant ? 0x200 : 0);
    let e = exp - 127 + 15;
    if (e >= 31) return sign | 0x7c00;
    if (e <= 0) {
      if (e < -10) return sign;
      mant |= 0x800000;
      const shift = 14 - e, half = 1 << (shift - 1), rem = mant & ((1 << shift) - 1);
      let hm = mant >>> shift;
      if (rem > half || (rem === half && (hm & 1))) hm++;
      return sign | hm;
    }
    let hm = mant >>> 13; const rem = mant & 0x1fff;
    if (rem > 0x1000 || (rem === 0x1000 && (hm & 1))) { hm++; if (hm === 0x400) { hm = 0; e++; if (e >= 31) return sign | 0x7c00; } }
    return sign | (e << 10) | hm;
  }
  function halfBitsToFloat(h) {
    const s = (h & 0x8000) ? -1 : 1, e = (h >>> 10) & 0x1f, m = h & 0x3ff;
    if (e === 0) return s * m * Math.pow(2, -24);
    if (e === 31) return m ? NaN : s * Infinity;
    return s * (1 + m / 1024) * Math.pow(2, e - 15);
  }
  const roundHalf = (x) => halfBitsToFloat(toHalfBits(x));

  function stats(arr) {
    let min = Infinity, max = -Infinity, sum = 0, sq = 0, zeros = 0, allInt = true;
    for (let i = 0; i < arr.length; i++) {
      const v = arr[i];
      if (v < min) min = v; if (v > max) max = v;
      sum += v; sq += v * v; if (v === 0) zeros++;
      if (allInt && !isInt(v)) allInt = false;
    }
    const n = arr.length || 1;
    return { min, max, mean: sum / n, rms: Math.sqrt(sq / n), energy: sq, zeros, zeroFrac: zeros / n, allInt, n: arr.length };
  }
  function quantile(arr, q) {
    const a = Array.from(arr).sort((x, y) => x - y);
    if (!a.length) return 0;
    return a[Math.min(a.length - 1, Math.max(0, Math.floor(q * (a.length - 1))))];
  }
  function histogram(values) { // Map value -> count
    const m = new Map();
    for (let i = 0; i < values.length; i++) m.set(values[i], (m.get(values[i]) || 0) + 1);
    return m;
  }
  function entropyOf(values) { // order-0 Shannon entropy in bits/symbol
    const m = histogram(values); const n = values.length || 1; let h = 0;
    for (const c of m.values()) { const p = c / n; h -= p * log2(p); }
    return { H: h, alphabet: m.size, hist: m, n: values.length };
  }

  // ============================================================ payloads (deterministic)
  const JSON_EVENTS = ['page_view', 'page_view', 'click', 'purchase', 'signup', 'logout'];
  const DEVICES = ['mobile', 'mobile', 'desktop', 'tablet'];
  const BROWSERS = ['chrome', 'chrome', 'safari', 'firefox', 'edge'];
  const COUNTRIES = ['CA', 'US', 'US', 'DE', 'FR', 'JP', 'BR'];
  const STATUS = ['ok', 'ok', 'ok', 'ok', 'error', 'retry'];
  const PLANS = ['free', 'free', 'pro', 'team'];
  const REFERRERS = ['https://www.google.com/', 'https://news.ycombinator.com/', 'direct', 'https://t.co/'];
  const pick = (rnd, arr) => arr[Math.floor(rnd() * arr.length)];
  const pad2 = (n) => String(n).padStart(2, '0');
  function hexStr(rnd, n) { let s = ''; for (let i = 0; i < n; i++) s += Math.floor(rnd() * 16).toString(16); return s; }
  function jsonRecords(seed, count) {
    const rnd = mulberry32(seed); const lines = [];
    for (let i = 0; i < count; i++) {
      const rec = {
        event: pick(rnd, JSON_EVENTS),
        user_id: 'u_' + (10000 + Math.floor(rnd() * 89999)),
        session_id: 's_' + hexStr(rnd, 12),
        timestamp: `2024-03-${pad2(10 + Math.floor(rnd() * 9))}T${pad2(Math.floor(rnd() * 24))}:${pad2(Math.floor(rnd() * 60))}:${pad2(Math.floor(rnd() * 60))}Z`,
        device: pick(rnd, DEVICES), browser: pick(rnd, BROWSERS), country: pick(rnd, COUNTRIES), plan: pick(rnd, PLANS),
        duration_ms: Math.floor(rnd() * 5000), page: '/products/' + Math.floor(rnd() * 400),
        referrer: pick(rnd, REFERRERS), status: pick(rnd, STATUS),
        ab_test: { name: 'checkout_v2', variant: pick(rnd, ['control', 'treatment']) },
        tags: [pick(rnd, ['new', 'returning']), pick(rnd, ['organic', 'paid'])],
      };
      lines.push(JSON.stringify(rec));
    }
    return lines.join('\n') + '\n';
  }

  const PROSE =
    'The quick brown fox jumps over the lazy dog. The lazy dog does not care about the fox; the dog cares about dinner. ' +
    'Every evening the fox returns to the same field, and every evening the dog pretends not to notice the fox. ' +
    'In the morning the farmer counts the chickens, and the farmer finds the same number of chickens as the morning before, ' +
    'because the fox is a careful fox and the dog is a lazy dog. The farmer, the fox and the dog have an understanding: ' +
    'the fox takes nothing, the dog sees nothing, and the farmer counts nothing twice. This arrangement has lasted longer ' +
    'than the fence, longer than the barn, and longer than the farmer\'s memory of how it began. Some evenings the fox brings ' +
    'a friend, and the friend is also a fox, and the dog pretends not to notice two foxes instead of one. The chickens, for ' +
    'their part, notice everything and say nothing, because chickens have learned that saying something changes nothing. ' +
    'So the field stays quiet, the fence stays broken, the barn stays red, and the story stays the same from one evening to ' +
    'the next, which is exactly how the farmer, the fox, the dog and the chickens prefer it. When winter comes the field is ' +
    'white instead of green, the fox is hungrier than before, and the dog sleeps in the barn instead of the field; but the ' +
    'understanding holds, the counting continues, and the number of chickens in the morning is the number of chickens the ' +
    'evening before. Nothing is compressed in this story except the truth, and the truth was small to begin with.';
  const PROSE_DICT =
    'The farmer owns a field, a fence, a barn and a dog. The dog is lazy and the fox is careful. Every morning the farmer ' +
    'counts the chickens; every evening the fox watches the chickens and the dog watches the fox. The fence is broken, the ' +
    'barn is red, and the field is green in summer and white in winter. The chickens notice everything. Nothing changes from ' +
    'one evening to the next, and the farmer, the fox, the dog and the chickens prefer it that way. The understanding holds. ' +
    'The fox takes nothing; the dog sees nothing; the farmer counts nothing twice. This is the story of the field and the fox.';

  function textPlanes(str) {
    const bytes = new Float64Array(str.length);
    for (let i = 0; i < str.length; i++) bytes[i] = str.charCodeAt(i) & 0xff;
    return [{ name: 'bytes', w: str.length, h: 1, data: bytes, vkind: 'byte', M: 256 }];
  }

  function synthImage(seed, W, H) {
    const rnd = mulberry32(seed);
    const R = new Float64Array(W * H), G = new Float64Array(W * H), B = new Float64Array(W * H);
    for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
      const u = x / (W - 1), v = y / (H - 1);
      let r = 70 + 130 * u + 20 * v, g = 100 + 90 * v - 20 * u, b = 150 - 70 * u + 60 * v;
      const dx = x - 20, dy = y - 12, d = Math.sqrt(dx * dx + dy * dy);
      const disc = smoothstep(7.5, 5.5, d);
      r += 90 * disc; g += 30 * disc; b -= 50 * disc;
      const edgeY = 21 + 2.5 * Math.sin(x * 0.35);
      const dark = smoothstep(edgeY - 1.5, edgeY + 1.5, y);
      r -= 45 * dark; g -= 40 * dark; b -= 25 * dark;
      const i = y * W + x;
      R[i] = clamp(Math.round(r + (rnd() - 0.5) * 6), 0, 255);
      G[i] = clamp(Math.round(g + (rnd() - 0.5) * 6), 0, 255);
      B[i] = clamp(Math.round(b + (rnd() - 0.5) * 6), 0, 255);
    }
    return [
      { name: 'R', w: W, h: H, data: R, vkind: 'sample', frame: 0 },
      { name: 'G', w: W, h: H, data: G, vkind: 'sample', frame: 0 },
      { name: 'B', w: W, h: H, data: B, vkind: 'sample', frame: 0 },
    ];
  }
  function shiftedFrame(planes, seed, dx, dy) {
    const rnd = mulberry32(seed);
    return planes.map((p) => {
      const out = new Float64Array(p.w * p.h);
      for (let y = 0; y < p.h; y++) for (let x = 0; x < p.w; x++) {
        const sx = clamp(x - dx, 0, p.w - 1), sy = clamp(y - dy, 0, p.h - 1);
        let v = p.data[sy * p.w + sx];
        if (x >= 4 && x < 12 && y >= 22 && y < 30) v += 8; // a small change: something brightened in one corner
        out[y * p.w + x] = clamp(Math.round(v + (rnd() - 0.5) * 2), 0, 255);
      }
      return { name: p.name, w: p.w, h: p.h, data: out, vkind: 'sample', frame: 1 };
    });
  }
  function synthAudio(seed) {
    const rnd = mulberry32(seed); const n = 1024;
    const L = new Float64Array(n), R = new Float64Array(n);
    for (let t = 0; t < n; t++) {
      const s = 9000 * Math.exp(-t / 900) * Math.sin(2 * Math.PI * 0.0123 * t)
        + 5000 * Math.exp(-t / 600) * Math.sin(2 * Math.PI * 0.0311 * t + 1)
        + 2500 * Math.exp(-t / 400) * Math.sin(2 * Math.PI * 0.0714 * t + 2)
        + (rnd() - 0.5) * 80;
      L[t] = clamp(Math.round(s), -32768, 32767);
      R[t] = clamp(Math.round(0.8 * s + (rnd() - 0.5) * 120), -32768, 32767);
    }
    return [
      { name: 'L', w: n, h: 1, data: L, vkind: 'sample' },
      { name: 'R', w: n, h: 1, data: R, vkind: 'sample' },
    ];
  }
  function randomOrthogonal(rnd, d) { // Gram-Schmidt on a random Gaussian matrix, rows are basis vectors
    const Q = [];
    for (let i = 0; i < d; i++) {
      const v = new Float64Array(d); for (let j = 0; j < d; j++) v[j] = gaussian(rnd);
      for (const q of Q) { let dot = 0; for (let j = 0; j < d; j++) dot += v[j] * q[j]; for (let j = 0; j < d; j++) v[j] -= dot * q[j]; }
      let nrm = 0; for (let j = 0; j < d; j++) nrm += v[j] * v[j]; nrm = Math.sqrt(nrm) || 1;
      for (let j = 0; j < d; j++) v[j] /= nrm;
      Q.push(v);
    }
    return Q;
  }
  function synthVectors(seed) {
    const rnd = mulberry32(seed); const n = 256, d = 8;
    const stds = [4, 2, 1, 0.5, 0.25, 0.12, 0.06, 0.03];
    const Q = randomOrthogonal(rnd, d);
    const X = new Float64Array(n * d);
    for (let i = 0; i < n; i++) {
      const z = stds.map((s) => s * gaussian(rnd));
      for (let j = 0; j < d; j++) { let v = 0; for (let k = 0; k < d; k++) v += Q[k][j] * z[k]; X[i * d + j] = Math.fround(v); }
    }
    return [{ name: 'X', w: d, h: n, data: X, vkind: 'sample' }];
  }
  function synthWeights(seed) {
    const rnd = mulberry32(seed); const n = 1024; const W = new Float64Array(n);
    for (let i = 0; i < n; i++) { let v = 0.02 * gaussian(rnd); if (rnd() < 0.01) v *= 15; W[i] = roundHalf(v); }
    return [{ name: 'W', w: 32, h: 32, data: W, vkind: 'sample' }];
  }
  function synthActivations(seed) {
    const rnd = mulberry32(seed); const T = 32, C = 32; const A = new Float64Array(T * C);
    for (let t = 0; t < T; t++) for (let c = 0; c < C; c++) {
      let v = gaussian(rnd); if (c === 5 || c === 21) v *= 20; A[t * C + c] = roundHalf(v);
    }
    return [{ name: 'K', w: C, h: T, data: A, vkind: 'sample' }];
  }

  const PAYLOAD_DEFS = {
    json:        { domain: 'text', bitDepth: 8, seed: 1001, label: (p) => `${fmt(p.n / 1024, 1)} KB of JSON events (${p.extra.records} records, same schema)` },
    prose:       { domain: 'text', bitDepth: 8, seed: 1002, label: (p) => `${fmt(p.n / 1024, 1)} KB of English prose` },
    image:       { domain: 'image', bitDepth: 8, seed: 1003, label: () => '32×32 RGB image, 8 bits/channel (3,072 samples)' },
    video:       { domain: 'video', bitDepth: 8, seed: 1004, label: () => 'two 32×32 RGB frames, the second shifted by (2,1) px' },
    audio:       { domain: 'audio', bitDepth: 16, seed: 1005, label: () => '1,024 stereo samples of 16-bit audio (3 decaying tones + noise)' },
    vectors:     { domain: 'vectors', bitDepth: 32, seed: 1006, label: () => '256 embedding vectors × 8 dims, float32, strongly correlated' },
    weights:     { domain: 'weights', bitDepth: 16, seed: 1007, label: () => '1,024 fp16 weights, bell-shaped (σ=0.02) with 1% outliers ×15' },
    activations: { domain: 'activations', bitDepth: 16, seed: 1008, label: () => '32 tokens × 32 channels of fp16 keys, 2 outlier channels ×20' },
  };
  const payloadCache = {};
  function generatePayload(kind, seedOffset) {
    const def = PAYLOAD_DEFS[kind] || PAYLOAD_DEFS.json;
    const k = PAYLOAD_DEFS[kind] ? kind : 'json';
    const seed = def.seed + (seedOffset || 0);
    let planes, extra = {};
    switch (k) {
      case 'json': { const txt = jsonRecords(seed, 5); planes = textPlanes(txt); extra = { records: 5, text: txt }; break; }
      case 'prose': planes = textPlanes(PROSE); extra = { text: PROSE }; break;
      case 'image': planes = synthImage(seed, 32, 32); break;
      case 'video': { const f0 = synthImage(seed, 32, 32); planes = f0.concat(shiftedFrame(f0, seed + 7, 2, 1)); break; }
      case 'audio': planes = synthAudio(seed); break;
      case 'vectors': planes = synthVectors(seed); break;
      case 'weights': planes = synthWeights(seed); break;
      case 'activations': planes = synthActivations(seed); break;
    }
    let n = 0, energy = 0;
    for (const p of planes) { n += p.data.length; for (let i = 0; i < p.data.length; i++) energy += p.data[i] * p.data[i]; }
    return { kind: k, domain: def.domain, bitDepth: def.bitDepth, planes, n, rawBits: n * def.bitDepth, energy, extra, def };
  }
  function getPayload(kind) {
    if (!payloadCache[kind]) payloadCache[kind] = generatePayload(kind, 0);
    return payloadCache[kind];
  }
  function clonePlanes(planes) { return planes.map((p) => Object.assign({}, p, { data: Float64Array.from(p.data) })); }

  // The "trained dictionary" the Inside Man preloads: similar data that is NOT part of the payload.
  const dictCache = {};
  function dictionaryBytes(kind) {
    if (dictCache[kind]) return dictCache[kind];
    let bytes;
    if (kind === 'json') bytes = Uint8Array.from(jsonRecords(4242, 2), (c) => c.charCodeAt(0) & 0xff);
    else if (kind === 'prose') bytes = Uint8Array.from(PROSE_DICT, (c) => c.charCodeAt(0) & 0xff);
    else { const other = generatePayload(kind, 99); bytes = serializePlanes(other.planes).bytes.slice(0, 600); }
    dictCache[kind] = bytes;
    return bytes;
  }

  // ============================================================ serialization (planes -> bytes for LZ / BWT)
  // Integer planes become 1, 2 or 4 bytes per value (offset by the minimum); floats become float32 little-endian.
  function serializePlanes(planes) {
    let all = 0, min = Infinity, max = -Infinity, allInt = true;
    for (const p of planes) { all += p.data.length; const s = stats(p.data); if (s.min < min) min = s.min; if (s.max > max) max = s.max; if (!s.allInt) allInt = false; }
    let width;
    if (allInt) { const range = max - min + 1; width = range <= 256 ? 1 : range <= 65536 ? 2 : 4; } else width = 4;
    const bytes = new Uint8Array(all * width); let o = 0;
    for (const p of planes) for (let i = 0; i < p.data.length; i++) {
      if (allInt) { let v = p.data[i] - min; for (let b = 0; b < width; b++) { bytes[o++] = v & 255; v = Math.floor(v / 256); } }
      else { const u = f32bits(p.data[i]); bytes[o++] = u & 255; bytes[o++] = (u >>> 8) & 255; bytes[o++] = (u >>> 16) & 255; bytes[o++] = u >>> 24; }
    }
    return { bytes, width, min, allInt, count: all };
  }
  function bytesToPlane(bytes, name) {
    return [{ name: name || 'bytes', w: bytes.length, h: 1, data: Float64Array.from(bytes), vkind: 'byte', M: 256 }];
  }

  // ============================================================ entropy coding primitives
  function huffmanLengths(hist) { // hist: Map symbol -> count. Returns Map symbol -> code length (real tree build).
    const nodes = []; const len = new Map();
    for (const [s, c] of hist) { nodes.push({ c, syms: [s] }); len.set(s, 0); }
    if (nodes.length === 1) { len.set(nodes[0].syms[0], 1); return len; }
    nodes.sort((a, b) => a.c - b.c);
    while (nodes.length > 1) {
      const a = nodes.shift(), b = nodes.shift();
      for (const s of a.syms) len.set(s, len.get(s) + 1);
      for (const s of b.syms) len.set(s, len.get(s) + 1);
      const m = { c: a.c + b.c, syms: a.syms.concat(b.syms) };
      let lo = 0, hi = nodes.length;
      while (lo < hi) { const mid = (lo + hi) >> 1; if (nodes[mid].c <= m.c) lo = mid + 1; else hi = mid; }
      nodes.splice(lo, 0, m);
    }
    return len;
  }
  function huffmanBits(symbols) {
    const { hist, H, alphabet } = entropyOf(symbols);
    const len = huffmanLengths(hist);
    let bits = 0, maxP = 0, maxSym = null;
    for (const [s, c] of hist) { bits += c * len.get(s); if (c > maxP) { maxP = c; maxSym = s; } }
    return { bits, H, alphabet, maxP: maxP / (symbols.length || 1), maxSym, maxLen: len.get(maxSym) };
  }
  // Adaptive order-0 model, Krichevsky–Trofimov estimator (alpha = 1/2): cost = sum of -log2 p(symbol) in bits.
  function adaptiveOrder0Bits(symbols, M) {
    const counts = new Map(); let total = 0, bits = 0; const alpha = 0.5;
    for (let i = 0; i < symbols.length; i++) {
      const s = symbols[i]; const c = counts.get(s) || 0;
      bits -= log2((c + alpha) / (total + alpha * M));
      counts.set(s, c + 1); total++;
    }
    return bits;
  }
  // Order-2 PPM-style adaptive context model (escape to order 1, 0, then uniform). Stand-in for context mixing.
  function ppmOrder2Bits(symbols, M) {
    const ctx2 = new Map(), ctx1 = new Map(); const c0 = new Map(); let n0 = 0, bits = 0;
    const lookup = (map, key) => { let t = map.get(key); if (!t) { t = { n: 0, counts: new Map() }; map.set(key, t); } return t; };
    for (let i = 0; i < symbols.length; i++) {
      const s = symbols[i]; let p = 1, coded = false;
      const t2 = i >= 2 ? lookup(ctx2, symbols[i - 2] + ',' + symbols[i - 1]) : null;
      const t1 = i >= 1 ? lookup(ctx1, symbols[i - 1]) : null;
      for (const t of [t2, t1]) {
        if (!t || coded) continue;
        const u = t.counts.size; if (u === 0) continue;
        const c = t.counts.get(s) || 0;
        if (c > 0) { p *= c / (t.n + u); coded = true; } else p *= u / (t.n + u);
      }
      if (!coded) {
        const u = c0.size, c = c0.get(s) || 0;
        if (c > 0) { p *= c / (n0 + u); coded = true; } else { if (u > 0) p *= u / (n0 + u); p *= 1 / M; }
      }
      bits -= log2(p);
      if (t2) { t2.n++; t2.counts.set(s, (t2.counts.get(s) || 0) + 1); }
      if (t1) { t1.n++; t1.counts.set(s, (t1.counts.get(s) || 0) + 1); }
      c0.set(s, (c0.get(s) || 0) + 1); n0++;
    }
    return bits;
  }
  function conditionalEntropyOrder2(symbols) { // empirical H(X_n | X_{n-1}, X_{n-2}) in bits/symbol
    const ctx = new Map(); let n = 0;
    for (let i = 2; i < symbols.length; i++) {
      const key = symbols[i - 2] + ',' + symbols[i - 1];
      let t = ctx.get(key); if (!t) { t = new Map(); ctx.set(key, t); }
      t.set(symbols[i], (t.get(symbols[i]) || 0) + 1); n++;
    }
    let h = 0;
    for (const t of ctx.values()) { let tn = 0; for (const c of t.values()) tn += c; for (const c of t.values()) h -= c * log2(c / tn); }
    return n ? h / n : 0;
  }
  const zigzagMap = (v) => (v >= 0 ? 2 * v : -2 * v - 1);
  // Golomb-Rice: actual bits with the best k per 64-symbol partition (5 bits to send k), zigzag for signed inputs.
  function riceBits(values) {
    let anyNeg = false; for (let i = 0; i < values.length; i++) if (values[i] < 0) { anyNeg = true; break; }
    let bits = 0, maxQ = 0; const ks = [];
    for (let p = 0; p < values.length; p += 64) {
      const end = Math.min(values.length, p + 64);
      let best = Infinity, bestK = 0, bestQ = 0;
      for (let k = 0; k <= 30; k++) {
        let b = 0, q = 0; const div = Math.pow(2, k);
        for (let i = p; i < end; i++) { const u = anyNeg ? zigzagMap(values[i]) : values[i]; const qq = Math.floor(u / div); b += qq + 1 + k; if (qq > q) q = qq; }
        if (b < best) { best = b; bestK = k; bestQ = q; }
      }
      bits += best + 5; ks.push(bestK); if (bestQ > maxQ) maxQ = bestQ;
    }
    return { bits, ks, maxQ, zigzag: anyNeg, partitions: ks.length };
  }
  // JPEG-style zero-run / size symbols: symbol = run*33 + size (run 0..15, size 0..32), ZRL = 16*33, EOB = 0.
  // Returns the pair symbols plus the raw amplitude bits that ride alongside them.
  const RLE_M = 16 * 33 + 1;
  function rleZeroRuns(values, blockLen) {
    const syms = []; let extraBits = 0, pairs = 0, zrl = 0, eob = 0, zeros = 0;
    const step = blockLen || values.length;
    for (let b = 0; b < values.length; b += step) {
      const end = Math.min(values.length, b + step); let run = 0;
      for (let i = b; i < end; i++) {
        const v = values[i];
        if (v === 0) { run++; zeros++; continue; }
        while (run > 15) { syms.push(16 * 33); run -= 16; zrl++; }
        const size = Math.floor(log2(Math.abs(v))) + 1;
        syms.push(run * 33 + size); extraBits += size; pairs++; run = 0;
      }
      if (run > 0) { syms.push(0); eob++; }
    }
    return { symbols: Int32Array.from(syms), extraBits, pairs, zrl, eob, zeros };
  }
  const ZIGZAG8 = (() => { // index order for an 8×8 block
    const out = []; for (let s = 0; s < 15; s++) { const cells = []; for (let y = 0; y <= s; y++) { const x = s - y; if (x < 8 && y < 8) cells.push(y * 8 + x); } if (s % 2 === 0) cells.reverse(); out.push(...cells); } return out;
  })();

  // ============================================================ transforms
  const DCT8 = (() => { const m = new Float64Array(64); for (let k = 0; k < 8; k++) for (let n = 0; n < 8; n++) m[k * 8 + n] = (k === 0 ? Math.sqrt(1 / 8) : 0.5) * Math.cos(Math.PI * (2 * n + 1) * k / 16); return m; })();
  function dct2d8(plane) { // orthonormal 8×8 DCT-II per block, in place layout
    const { w, h, data } = plane; const out = new Float64Array(w * h); const tmp = new Float64Array(64);
    for (let by = 0; by < h; by += 8) for (let bx = 0; bx < w; bx += 8) {
      for (let y = 0; y < 8; y++) for (let k = 0; k < 8; k++) { let s = 0; for (let n = 0; n < 8; n++) s += DCT8[k * 8 + n] * data[(by + y) * w + bx + n]; tmp[y * 8 + k] = s; }
      for (let x = 0; x < 8; x++) for (let k = 0; k < 8; k++) { let s = 0; for (let n = 0; n < 8; n++) s += DCT8[k * 8 + n] * tmp[n * 8 + x]; out[(by + k) * w + bx + x] = s; }
    }
    return out;
  }
  function dct1d8(data) { // 8-point DCT-II along a 1-D signal, blocks of 8
    const out = new Float64Array(data.length);
    for (let b = 0; b + 8 <= data.length; b += 8) for (let k = 0; k < 8; k++) { let s = 0; for (let n = 0; n < 8; n++) s += DCT8[k * 8 + n] * data[b + n]; out[b + k] = s; }
    for (let i = data.length - (data.length % 8); i < data.length; i++) out[i] = data[i];
    return out;
  }
  function dct4Frames(data, N) { // orthonormal DCT-IV on N-sample frames (MDCT stand-in)
    const out = new Float64Array(data.length); const sc = Math.sqrt(2 / N);
    for (let b = 0; b + N <= data.length; b += N) for (let k = 0; k < N; k++) { let s = 0; for (let n = 0; n < N; n++) s += data[b + n] * Math.cos(Math.PI / N * (n + 0.5) * (k + 0.5)); out[b + k] = sc * s; }
    for (let i = data.length - (data.length % N); i < data.length; i++) out[i] = data[i];
    return out;
  }
  function haar1d(src, n, stride, base, dst) { // one level, orthonormal, lows first then highs
    const half = n >> 1, r = Math.SQRT1_2;
    for (let i = 0; i < half; i++) { const a = src[base + (2 * i) * stride], b = src[base + (2 * i + 1) * stride]; dst[base + i * stride] = (a + b) * r; dst[base + (half + i) * stride] = (a - b) * r; }
    if (n & 1) dst[base + (n - 1) * stride] = src[base + (n - 1) * stride];
  }
  function haar2d(plane) {
    const { w, h, data } = plane; const a = new Float64Array(w * h), b = new Float64Array(w * h);
    for (let y = 0; y < h; y++) haar1d(data, w, 1, y * w, a);
    if (h > 1) { for (let x = 0; x < w; x++) haar1d(a, h, w, x, b); return b; }
    return a;
  }
  function hadamardBlocks(data, N) { // normalized Walsh–Hadamard on consecutive blocks of N (power of two)
    const out = Float64Array.from(data); const sc = 1 / Math.sqrt(N);
    for (let b = 0; b + N <= data.length; b += N) {
      for (let len = 1; len < N; len <<= 1) for (let i = b; i < b + N; i += len << 1) for (let j = i; j < i + len; j++) { const u = out[j], v = out[j + len]; out[j] = u + v; out[j + len] = u - v; }
      for (let i = b; i < b + N; i++) out[i] *= sc;
    }
    return out;
  }
  function covariance(X, n, d) {
    const mean = new Float64Array(d); for (let i = 0; i < n; i++) for (let j = 0; j < d; j++) mean[j] += X[i * d + j] / n;
    const C = new Float64Array(d * d);
    for (let i = 0; i < n; i++) for (let a = 0; a < d; a++) { const xa = X[i * d + a] - mean[a]; for (let b = a; b < d; b++) C[a * d + b] += xa * (X[i * d + b] - mean[b]); }
    for (let a = 0; a < d; a++) for (let b = 0; b < a; b++) C[a * d + b] = C[b * d + a];
    for (let i = 0; i < d * d; i++) C[i] /= Math.max(1, n - 1);
    return { C, mean };
  }
  function pcaPowerIteration(X, n, d, kComp, iters) { // returns { basis: [k][d] unit vectors, eig: [k], mean }
    const { C, mean } = covariance(X, n, d); const A = Float64Array.from(C); const basis = [], eig = [];
    const rnd = mulberry32(77);
    for (let c = 0; c < kComp; c++) {
      let v = new Float64Array(d); for (let j = 0; j < d; j++) v[j] = gaussian(rnd) + (j === c ? 1 : 0);
      let lambda = 0;
      for (let it = 0; it < (iters || 60); it++) {
        const w = new Float64Array(d); for (let a = 0; a < d; a++) { let s = 0; for (let b = 0; b < d; b++) s += A[a * d + b] * v[b]; w[a] = s; }
        let nrm = 0; for (let a = 0; a < d; a++) nrm += w[a] * w[a]; nrm = Math.sqrt(nrm);
        if (nrm < 1e-12) { lambda = 0; break; }
        for (let a = 0; a < d; a++) v[a] = w[a] / nrm; lambda = nrm;
      }
      basis.push(v); eig.push(lambda);
      for (let a = 0; a < d; a++) for (let b = 0; b < d; b++) A[a * d + b] -= lambda * v[a] * v[b]; // deflate
    }
    return { basis, eig, mean };
  }
  function kmeans(X, n, d, K, iters) {
    const rnd = mulberry32(31337); const cent = new Float64Array(K * d); const assign = new Uint16Array(n);
    const used = new Set();
    for (let k = 0; k < K; k++) { let i = Math.floor(rnd() * n), guard = 0; while (used.has(i) && guard++ < 100) i = Math.floor(rnd() * n); used.add(i); for (let j = 0; j < d; j++) cent[k * d + j] = X[i * d + j]; }
    let se = 0;
    for (let it = 0; it < (iters || 12); it++) {
      se = 0; const sums = new Float64Array(K * d), cnt = new Int32Array(K);
      for (let i = 0; i < n; i++) {
        let best = Infinity, bk = 0;
        for (let k = 0; k < K; k++) { let dd = 0; for (let j = 0; j < d; j++) { const t = X[i * d + j] - cent[k * d + j]; dd += t * t; } if (dd < best) { best = dd; bk = k; } }
        assign[i] = bk; se += best; cnt[bk]++; for (let j = 0; j < d; j++) sums[bk * d + j] += X[i * d + j];
      }
      for (let k = 0; k < K; k++) {
        if (cnt[k] === 0) { const i = Math.floor(rnd() * n); for (let j = 0; j < d; j++) cent[k * d + j] = X[i * d + j]; }
        else for (let j = 0; j < d; j++) cent[k * d + j] = sums[k * d + j] / cnt[k];
      }
    }
    se = 0;
    for (let i = 0; i < n; i++) { let best = Infinity, bk = 0; for (let k = 0; k < K; k++) { let dd = 0; for (let j = 0; j < d; j++) { const t = X[i * d + j] - cent[k * d + j]; dd += t * t; } if (dd < best) { best = dd; bk = k; } } assign[i] = bk; se += best; }
    return { centroids: cent, assign, se };
  }

  // ============================================================ dictionary coders / BWT
  function lz77(bytes, dict, window, minMatch, maxMatch) {
    window = window || 4096; minMatch = minMatch || 3; maxMatch = maxMatch || 258;
    const pre = dict ? dict.length : 0; const N = pre + bytes.length;
    const buf = new Uint8Array(N); if (dict) buf.set(dict, 0); buf.set(bytes, pre);
    const head = new Int32Array(1 << 16).fill(-1), prev = new Int32Array(N).fill(-1);
    const h3 = (i) => (Math.imul((buf[i] << 16) | (buf[i + 1] << 8) | buf[i + 2], 2654435761) >>> 16) & 0xffff;
    const insert = (i) => { if (i + 2 < N) { const h = h3(i); prev[i] = head[h]; head[h] = i; } };
    for (let i = 0; i < pre; i++) insert(i);
    const tokens = []; let i = pre;
    while (i < N) {
      let bestLen = 0, bestDist = 0;
      if (i + minMatch <= N) {
        let cand = head[h3(i)], steps = 0; const maxL = Math.min(maxMatch, N - i);
        while (cand >= 0 && i - cand <= window && steps < 256) {
          let l = 0; while (l < maxL && buf[cand + l] === buf[i + l]) l++;
          if (l > bestLen) { bestLen = l; bestDist = i - cand; if (l === maxL) break; }
          cand = prev[cand]; steps++;
        }
      }
      if (bestLen >= minMatch) { tokens.push({ type: 'match', dist: bestDist, len: bestLen, dict: i - bestDist < pre }); for (let j = 0; j < bestLen; j++) insert(i + j); i += bestLen; }
      else { tokens.push({ type: 'lit', c: buf[i] }); insert(i); i++; }
    }
    return tokens;
  }
  function lz77Decode(tokens, dict) {
    const pre = dict ? dict.length : 0; let n = 0; for (const t of tokens) n += t.type === 'lit' ? 1 : t.len;
    const buf = new Uint8Array(pre + n); if (dict) buf.set(dict, 0); let o = pre;
    for (const t of tokens) { if (t.type === 'lit') buf[o++] = t.c; else for (let j = 0; j < t.len; j++) { buf[o] = buf[o - t.dist]; o++; } }
    return buf.slice(pre);
  }
  // LZ token stream as entropy-coder symbols: literal 0..255, match length 256 + (len-3), distance bucket 512 + floor(log2 dist)
  // with floor(log2 dist) raw extra bits (DEFLATE's scheme, simplified). Alphabet: 512 + 14 buckets.
  const LZ_M = 512 + 14;
  function lzSymbols(tokens) {
    const syms = []; let extra = 0; const lits = [], matches = [];
    for (const t of tokens) {
      if (t.type === 'lit') { syms.push(t.c); lits.push(t.c); }
      else { const ls = 256 + Math.min(255, t.len - 3), b = Math.floor(log2(t.dist)), ds = 512 + Math.min(13, b); syms.push(ls, ds); matches.push(ls, ds); extra += Math.min(13, b); }
    }
    return { symbols: Int32Array.from(syms), extraBits: extra, literals: Int32Array.from(lits), matches: Int32Array.from(matches) };
  }
  function lzw(bytes, maxBits) {
    maxBits = maxBits || 12; const maxCode = 1 << maxBits;
    const dict = new Map(); let next = 256; const codes = []; let bits = 0;
    let w = -1;
    for (let i = 0; i < bytes.length; i++) {
      const c = bytes[i];
      if (w < 0) { w = c; continue; }
      const key = w * 256 + c; const code = dict.get(key);
      if (code !== undefined) { w = code; continue; }
      codes.push(w); bits += Math.max(9, ceilLog2(next + 1));
      if (next < maxCode) dict.set(key, next++);
      w = c;
    }
    if (w >= 0) { codes.push(w); bits += Math.max(9, ceilLog2(next + 1)); }
    return { codes: Int32Array.from(codes), dictSize: next, varBits: bits };
  }
  function lzwDecode(codes, maxBits) {
    maxBits = maxBits || 12; const maxCode = 1 << maxBits;
    const table = []; for (let i = 0; i < 256; i++) table.push([i]);
    const out = []; let prevEntry = null;
    for (const code of codes) {
      let entry;
      if (code < table.length) entry = table[code]; else if (prevEntry) entry = prevEntry.concat([prevEntry[0]]); else throw new Error('bad LZW code');
      out.push(...entry);
      if (prevEntry && table.length < maxCode) table.push(prevEntry.concat([entry[0]]));
      prevEntry = entry;
    }
    return Uint8Array.from(out);
  }
  function bwt(bytes) {
    const n = bytes.length; const idx = new Array(n); for (let i = 0; i < n; i++) idx[i] = i;
    idx.sort((a, b) => { for (let k = 0; k < n; k++) { const ca = bytes[(a + k) % n], cb = bytes[(b + k) % n]; if (ca !== cb) return ca - cb; } return a - b; });
    const last = new Uint8Array(n); let primary = 0;
    for (let i = 0; i < n; i++) { last[i] = bytes[(idx[i] + n - 1) % n]; if (idx[i] === 0) primary = i; }
    return { last, primary };
  }
  function bwtInverse(last, primary) {
    const n = last.length; const count = new Int32Array(257);
    for (let i = 0; i < n; i++) count[last[i] + 1]++;
    for (let i = 1; i <= 256; i++) count[i] += count[i - 1];
    const next = new Int32Array(n); const seen = new Int32Array(256);
    for (let i = 0; i < n; i++) { const c = last[i]; next[count[c] + seen[c]] = i; seen[c]++; }
    const out = new Uint8Array(n); let p = primary;
    for (let i = 0; i < n; i++) { p = next[p]; out[i] = last[p]; }
    return out;
  }
  function mtfEncode(bytes) {
    const list = []; for (let i = 0; i < 256; i++) list.push(i);
    const out = new Uint8Array(bytes.length);
    for (let i = 0; i < bytes.length; i++) { const c = bytes[i]; const k = list.indexOf(c); out[i] = k; list.splice(k, 1); list.unshift(c); }
    return out;
  }
  function mtfDecode(idx) {
    const list = []; for (let i = 0; i < 256; i++) list.push(i);
    const out = new Uint8Array(idx.length);
    for (let i = 0; i < idx.length; i++) { const c = list[idx[i]]; out[i] = c; list.splice(idx[i], 1); list.unshift(c); }
    return out;
  }

  /*__NEXT__*/
})();
