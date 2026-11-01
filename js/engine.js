// Choose Your Codec — engine. Toy but REAL pipeline math on small deterministic payloads.
// Contract: docs/ARCHITECTURE.md ("Engine contract"). Exports CYC.Engine.run(payloadKind, squad) and
// CYC.Engine.payloadLabel(kind). Plain ES2020, no dependencies. Everything here is honest bit counting:
// where an algorithm is a stand-in for the real algorithm the caption says so.
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
  const pct = (x, d = 0) => { const v = 100 * x; return (v !== 0 && Math.abs(v) < 0.5 * Math.pow(10, -d) ? v.toPrecision(2) : fmt(v, d)) + '%'; };
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
      const base = 60 + 110 * u + 50 * v; // shared brightness gradient (natural images are mostly luma) with a mild tint
      let r = base + 22 * u + 8, g = base + 12 * v, b = base - 18 * u + 24 * v + 10;
      const dx = x - 20, dy = y - 12, d = Math.sqrt(dx * dx + dy * dy);
      const disc = smoothstep(7.5, 5.5, d);
      r += 75 * disc; g += 45 * disc; b += 5 * disc;
      const edgeY = 21 + 2.5 * Math.sin(x * 0.35);
      const dark = smoothstep(edgeY - 1.5, edgeY + 1.5, y);
      r -= 50 * dark; g -= 45 * dark; b -= 35 * dark;
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
  // Keys of one attention head: 32 tokens × 32 channels. Pre-RoPE: per channel an AR(1) drift (ρ = 0.97) plus noise,
  // so neighbouring tokens are similar (CacheGen's observation); two outlier channels with a stable value
  // (channel 9 ≈ +16, channel 21 ≈ −14); the first token (attention sink) with a 4× norm. Then standard RoPE
  // (pairs (2i, 2i+1), θ_i = 10000^(−2i/d), angle = position × θ_i): channel 9's pair turns 0.1 rad/token
  // (it wanders), channel 21's pair 0.003 rad/token (it sits still). Stored as fp16.
  const ROPE_SINK_GAIN = 4;
  function ropeRotate(data, T, C, sign, tokenIds) {
    const out = new Float64Array(data.length);
    for (let t = 0; t < T; t++) { const pos = tokenIds ? tokenIds[t] : t;
      for (let i = 0; i < C / 2; i++) { const a = sign * pos * Math.pow(10000, -2 * i / C), cs = Math.cos(a), sn = Math.sin(a); const x = data[t * C + 2 * i], y = data[t * C + 2 * i + 1]; out[t * C + 2 * i] = x * cs - y * sn; out[t * C + 2 * i + 1] = x * sn + y * cs; }
    }
    return out;
  }
  function synthActivations(seed) {
    const rnd = mulberry32(seed); const T = 32, C = 32; const pre = new Float64Array(T * C); const rho = 0.97, drift = new Float64Array(C);
    for (let c = 0; c < C; c++) drift[c] = gaussian(rnd);
    for (let t = 0; t < T; t++) for (let c = 0; c < C; c++) {
      if (t > 0) drift[c] = rho * drift[c] + Math.sqrt(1 - rho * rho) * gaussian(rnd);
      let v = drift[c] + 0.25 * gaussian(rnd); if (c === 9) v = 16 + 3 * v; else if (c === 21) v = -14 + 3 * v; if (t === 0) v *= ROPE_SINK_GAIN; pre[t * C + c] = v;
    }
    const post = ropeRotate(pre, T, C, 1); for (let i = 0; i < post.length; i++) post[i] = roundHalf(post[i]);
    const ids = new Int32Array(T); for (let t = 0; t < T; t++) ids[t] = t;
    return [{ name: 'K', w: C, h: T, data: post, vkind: 'sample', tokenIds: ids, rope: true }];
  }

  const PAYLOAD_DEFS = {
    json:        { domain: 'text', bitDepth: 8, seed: 1001, label: (p) => `${fmt(p.n / 1024, 1)} KB of JSON events (${p.extra.records} records, same schema)` },
    prose:       { domain: 'text', bitDepth: 8, seed: 1002, label: (p) => `${fmt(p.n / 1024, 1)} KB of English prose` },
    image:       { domain: 'image', bitDepth: 8, seed: 1003, label: () => '32×32 RGB image, 8 bits/channel (3,072 samples)' },
    video:       { domain: 'video', bitDepth: 8, seed: 1004, label: () => 'two 32×32 RGB frames, the second shifted by (2,1) px' },
    audio:       { domain: 'audio', bitDepth: 16, seed: 1005, label: () => '1,024 stereo samples of 16-bit audio (3 decaying tones + noise)' },
    vectors:     { domain: 'vectors', bitDepth: 32, seed: 1006, label: () => '256 embedding vectors × 8 dims, float32, strongly correlated' },
    weights:     { domain: 'weights', bitDepth: 16, seed: 1007, label: () => '1,024 fp16 weights, bell-shaped (σ=0.02) with 1% outliers ×15' },
    activations: { domain: 'activations', bitDepth: 16, seed: 1008, label: () => '32 tokens × 32 channels of fp16 RoPE\'d keys: 2 outlier channels, a 4× attention-sink token' },
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
  function clonePlanes(planes) { return planes.map((p) => Object.assign({}, p, { data: Float64Array.from(p.data), tokenIds: p.tokenIds ? Int32Array.from(p.tokenIds) : undefined })); }

  // The "trained dictionary" the shared-prior stage preloads: similar data that is NOT part of the payload.
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
    let width, off = 0;
    if (allInt) { // keep natural byte values where they fit (so a dictionary of raw bytes lines up); offset only when needed
      const range = max - min + 1;
      if (min >= 0 && max <= 255) width = 1; else if (range <= 256) { width = 1; off = min; } else if (min >= 0 && max <= 65535) width = 2; else if (range <= 65536) { width = 2; off = min; } else { width = 4; off = min; }
    } else width = 4;
    const bytes = new Uint8Array(all * width); let o = 0;
    for (const p of planes) for (let i = 0; i < p.data.length; i++) {
      if (allInt) { let v = p.data[i] - off; for (let b = 0; b < width; b++) { bytes[o++] = v & 255; v = Math.floor(v / 256); } }
      else { const u = f32bits(p.data[i]); bytes[o++] = u & 255; bytes[o++] = (u >>> 8) & 255; bytes[o++] = (u >>> 16) & 255; bytes[o++] = u >>> 24; }
    }
    return { bytes, width, min: off, allInt, count: all };
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
  function gramMatrix(X, n, d) { // uncentered XᵀX (eigenvalues = squared singular values)
    const C = new Float64Array(d * d);
    for (let i = 0; i < n; i++) for (let a = 0; a < d; a++) { const xa = X[i * d + a]; for (let b = a; b < d; b++) C[a * d + b] += xa * X[i * d + b]; }
    for (let a = 0; a < d; a++) for (let b = 0; b < a; b++) C[a * d + b] = C[b * d + a];
    return { C, mean: new Float64Array(d) };
  }
  function pcaPowerIteration(X, n, d, kComp, iters, uncentered) { // returns { basis: [k][d] unit vectors, eig: [k], mean }
    const { C, mean } = uncentered ? gramMatrix(X, n, d) : covariance(X, n, d); const A = Float64Array.from(C); const basis = [], eig = [];
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

  // ============================================================ pipeline state, viz helpers, symbolization
  const NONE = () => CYC.byId && CYC.byId.none ? CYC.byId.none : { id: 'none', name: 'None (pass-through)', real: 'Nobody (identity / raw fixed-length)' };
  function makeState(payload) {
    const tokenPlane = payload.planes.find((p) => p.tokenIds); const nTok = tokenPlane ? tokenPlane.h : 0;
    const energyTok = new Float64Array(nTok); if (tokenPlane) for (let t = 0; t < nTok; t++) for (let j = 0; j < tokenPlane.w; j++) energyTok[t] += Math.pow(tokenPlane.data[t * tokenPlane.w + j], 2);
    return {
      kind: payload.kind, domain: payload.domain, bitDepth: payload.bitDepth,
      nOrig: payload.n, rawBits: payload.rawBits, sigEnergy: payload.energy,
      rep: 'planes', planes: clonePlanes(payload.planes), tokens: null, lzw: null, lzMeta: null,
      lossless: true, seSum: 0, hasQuantizer: false,
      nTokens: nTok, seTok: new Float64Array(nTok), energyTok, tokenWeights: null, protectedTokens: null, evicted: null,
      sideInfoBytes: 0, headerBits: 0, extraBits: 0, fixedBitsExact: null,
      flags: { dict: null, tables: false, codebook: false, decoderNet: false, neuralEnc: false, contextModel: null, maskThreshold: 0, codebookBytes: 0, basis: false, tiered: false },
      pending: {}, warnings: [], stages: [], roundedWarned: false,
    };
  }
  // Squared-error bookkeeping. Errors on token-row planes are also attributed to their token (for attention-weighted distortion).
  function addErr(state, plane, i, e2) {
    state.seSum += e2;
    if (!state.nTokens) return;
    if (plane && plane.tokenIds) state.seTok[plane.tokenIds[Math.floor(i / plane.w)]] += e2;
    else { const share = e2 / state.nTokens; for (let t = 0; t < state.nTokens; t++) state.seTok[t] += share; }
  }
  const tokenPlaneOf = (state) => (state.rep === 'planes' ? state.planes.find((p) => p.tokenIds && p.h === p.tokenIds.length) : null);
  function emit(state, role, f, caption, metrics, viz) {
    const st = { role, fighterId: f.id, title: `${f.name}: ${f.real}`, caption, metrics: metrics || {}, viz: viz || vizText([{ text: caption, cls: 'lit' }]) };
    state.stages.push(st); return st;
  }
  function passThrough(state, role, f, why) {
    const n = countSamples(state);
    return emit(state, role, f, `${f.name} passes the ${fmtInt(n)} ${state.rep === 'planes' ? 'values' : 'tokens'} through untouched (0 bits changed): ${why}`, { symbols: n }, currentViz(state));
  }
  function countSamples(state) {
    if (state.rep === 'tokens') return state.tokens.length;
    if (state.rep === 'lzw') return state.lzw.codes.length;
    let n = 0; for (const p of state.planes) n += p.data.length; return n;
  }
  function warn(state, msg) { if (!state.warnings.includes(msg)) state.warnings.push(msg); }

  // ---- viz descriptors (shapes from docs/ARCHITECTURE.md)
  function vizText(spans) { return { type: 'text', spans }; }
  function normPlane(p) {
    const s = stats(p.data); const out = new Float32Array(p.data.length);
    if (p.vkind === 'sample' && s.min >= 0 && s.max <= 255 && s.allInt) { for (let i = 0; i < out.length; i++) out[i] = p.data[i] / 255; }
    else if (s.min < 0) { const m = Math.max(Math.abs(s.min), Math.abs(s.max)) || 1; for (let i = 0; i < out.length; i++) out[i] = 0.5 + 0.5 * p.data[i] / m; }
    else { const r = (s.max - s.min) || 1; for (let i = 0; i < out.length; i++) out[i] = (p.data[i] - s.min) / r; }
    return { name: p.name, w: p.w, h: p.h, values: out };
  }
  function vizGrid(planes, withRgb) {
    const out = { type: 'grid', planes: planes.slice(0, 6).map(normPlane) };
    const rgbP = planes.filter((p) => p.frame === 0 || p.frame === undefined).slice(0, 3);
    if (withRgb && rgbP.length === 3 && rgbP[0].name === 'R' && rgbP[1].name === 'G' && rgbP[2].name === 'B') {
      const w = rgbP[0].w, h = rgbP[0].h, data = new Uint8ClampedArray(w * h * 4);
      for (let i = 0; i < w * h; i++) { data[i * 4] = rgbP[0].data[i]; data[i * 4 + 1] = rgbP[1].data[i]; data[i * 4 + 2] = rgbP[2].data[i]; data[i * 4 + 3] = 255; }
      out.rgb = { w, h, data };
    }
    return out;
  }
  function vizHeatmapOf(planes, blockSize) { // log-scaled magnitudes; 1-D planes are folded into rows of blockSize (or 32)
    let zeros = 0, total = 0, maxAbs = 0;
    for (const p of planes) for (let i = 0; i < p.data.length; i++) { const a = Math.abs(p.data[i]); if (a > maxAbs) maxAbs = a; }
    const sc = log2(1 + maxAbs) || 1;
    const rows = []; let w = 0;
    for (const p of planes) {
      if (p.h > 1) { for (let y = 0; y < p.h; y++) rows.push(p.data.subarray(y * p.w, (y + 1) * p.w)); w = Math.max(w, p.w); }
      else { const fold = p.blockSize || 32; for (let y = 0; y * fold < p.data.length; y++) rows.push(p.data.subarray(y * fold, Math.min(p.data.length, (y + 1) * fold))); w = Math.max(w, fold); }
      if (rows.length * w > 4096) break;
    }
    const h = Math.min(rows.length, Math.floor(4096 / Math.max(1, w)));
    const values = new Float32Array(w * h);
    for (let y = 0; y < h; y++) for (let x = 0; x < rows[y].length; x++) { const a = Math.abs(rows[y][x]); values[y * w + x] = log2(1 + a) / sc; total++; if (a < 1e-9) zeros++; }
    return { type: 'heatmap', w, h, values, zeroFrac: total ? zeros / total : 0, blockSize: blockSize || (planes[0] && planes[0].blockSize) || undefined };
  }
  function vizWaveform(seriesArr, labels) {
    return { type: 'waveform', labels, series: seriesArr.map((s) => { const n = Math.min(s.length, 4096); let m = 0; for (let i = 0; i < n; i++) m = Math.max(m, Math.abs(s[i])); const out = new Float32Array(n); for (let i = 0; i < n; i++) out[i] = m ? s[i] / m : 0; return out; }) };
  }
  function vizHistogram(values, M, offset) { // values: symbols (non-negative) ; offset: subtract for labels (so residuals show signed)
    const { H, hist, alphabet } = entropyOf(values); const fixedBits = ceilLog2(M);
    const keys = Array.from(hist.keys()).sort((a, b) => a - b);
    let bins, labels;
    if (alphabet <= 64 && keys.length) {
      const lo = keys[0], hi = keys[keys.length - 1];
      if (hi - lo < 64) { bins = []; labels = []; for (let v = lo; v <= hi; v++) { bins.push(hist.get(v) || 0); labels.push(String(v - (offset || 0))); } }
    }
    if (!bins) {
      const lo = keys[0], hi = keys[keys.length - 1], nb = 64, step = (hi - lo + 1) / nb; bins = new Array(nb).fill(0); labels = [];
      for (const [v, c] of hist) bins[Math.min(nb - 1, Math.floor((v - lo) / step))] += c;
      for (let b = 0; b < nb; b++) labels.push(String(Math.round(lo + b * step - (offset || 0))));
    }
    return { type: 'histogram', bins, labels, entropyBitsPerSymbol: H, fixedBits };
  }
  function vizScatter(X, n, d, centroids, assign, K) { // project to the first two PCA directions of X
    const { basis, mean } = pcaPowerIteration(X, n, d, 2, 40);
    const proj = (src, i) => [0, 1].map((c) => { let s = 0; for (let j = 0; j < d; j++) s += (src[i * d + j] - mean[j]) * basis[c][j]; return s; });
    const pts = new Float32Array(2 * n); let m = 0;
    for (let i = 0; i < n; i++) { const [a, b] = proj(X, i); pts[2 * i] = a; pts[2 * i + 1] = b; m = Math.max(m, Math.abs(a), Math.abs(b)); }
    let cen; if (centroids) { cen = new Float32Array(2 * K); for (let k = 0; k < K; k++) { const [a, b] = proj(centroids, k); cen[2 * k] = a; cen[2 * k + 1] = b; m = Math.max(m, Math.abs(a), Math.abs(b)); } }
    m = m || 1; for (let i = 0; i < pts.length; i++) pts[i] /= m; if (cen) for (let i = 0; i < cen.length; i++) cen[i] /= m;
    const out = { type: 'scatter', points: pts }; if (cen) { out.centroids = cen; out.assignments = assign; } return out;
  }
  const printable = (c) => (c >= 32 && c < 127 ? String.fromCharCode(c) : c === 10 ? '\n' : '·');
  function bytesToSpans(bytes, cls) { let s = ''; for (let i = 0; i < Math.min(bytes.length, 2000); i++) s += printable(bytes[i]); return [{ text: s, cls: cls || 'lit' }]; }
  function currentViz(state) {
    if (state.rep === 'tokens') return lzSpans(state.tokens, state.flags.dict);
    if (state.rep === 'lzw') return vizText(bytesToSpans(lzwDecode(state.lzw.codes)));
    const p = state.planes;
    if (p[0].h === 1 && p[0].vkind !== 'byte') return vizWaveform(p.slice(0, 2).map((q) => q.data), p.slice(0, 2).map((q) => q.name));
    if (p[0].vkind === 'byte') return vizText(bytesToSpans(p[0].data));
    if (p[0].vkind === 'coef') return vizHeatmapOf(p);
    return vizGrid(p, true);
  }
  function lzSpans(tokens, dict) {
    const spans = []; let lit = '';
    const pre = dict ? dict.length : 0; let n = 0; for (const t of tokens) n += t.type === 'lit' ? 1 : t.len;
    const buf = new Uint8Array(pre + n); if (dict) buf.set(dict, 0); let o = pre;
    for (const t of tokens) {
      if (t.type === 'lit') { buf[o] = t.c; lit += printable(t.c); o++; continue; }
      if (lit) { spans.push({ text: lit, cls: 'lit' }); lit = ''; }
      let s = ''; for (let j = 0; j < t.len; j++) { buf[o] = buf[o - t.dist]; s += printable(buf[o]); o++; }
      spans.push({ text: s, cls: t.dict ? 'dict' : 'match', title: `${t.dict ? 'dictionary' : 'back'} ${t.dist} bytes, copy ${t.len}` });
      if (spans.length > 1500) break;
    }
    if (lit) spans.push({ text: lit, cls: 'lit' });
    return vizText(spans);
  }

  // ---- the symbol stream that the entropy stage sees
  function scanValues(planes) {
    let n = 0; for (const p of planes) n += p.data.length;
    const out = new Float64Array(n); let o = 0;
    for (const p of planes) {
      if (p.zigzag && p.blockSize === 8 && p.h % 8 === 0 && p.w % 8 === 0) {
        for (let by = 0; by < p.h; by += 8) for (let bx = 0; bx < p.w; bx += 8) for (const z of ZIGZAG8) out[o++] = p.data[(by + (z >> 3)) * p.w + bx + (z & 7)];
      } else { out.set(p.data, o); o += p.data.length; }
    }
    return out;
  }
  function symbolize(state, forCoding) {
    if (state.rep === 'tokens') { const s = lzSymbols(state.tokens); return { symbols: s.symbols, M: LZ_M, extraBits: s.extraBits, signed: s.symbols, lz: s, offset: 0, desc: 'LZ77 token symbols (literal bytes, match lengths, distance buckets)' }; }
    if (state.rep === 'lzw') return { symbols: state.lzw.codes, M: Math.max(257, state.lzw.dictSize), extraBits: 0, signed: state.lzw.codes, offset: 0, desc: 'LZW phrase codes' };
    const vals = scanValues(state.planes); const s = stats(vals);
    if (!vals.length) return { symbols: new Int32Array(0), M: 1, extraBits: 0, signed: new Int32Array(0), offset: 0, desc: 'symbols (none left: everything went out of band)' };
    if (s.allInt) {
      const allM = state.planes.every((p) => p.M !== undefined) && s.min >= 0;
      if (allM) { const M = Math.max(...state.planes.map((p) => p.M)); return { symbols: Int32Array.from(vals), M, extraBits: 0, signed: Int32Array.from(vals), offset: 0, desc: state.planes[0].vkind === 'byte' ? 'bytes' : 'indices' }; }
      const min = s.min; const sym = new Int32Array(vals.length); for (let i = 0; i < vals.length; i++) sym[i] = vals[i] - min;
      const kind = state.planes[0].vkind;
      return { symbols: sym, M: s.max - min + 1, extraBits: 0, signed: Int32Array.from(vals), offset: -min, desc: kind === 'residual' ? 'residuals' : kind === 'index' ? 'quantizer indices' : min < 0 ? 'signed integers' : 'integer samples' };
    }
    if (state.bitDepth === 32) { const sym = new Float64Array(vals.length); for (let i = 0; i < vals.length; i++) sym[i] = f32bits(vals[i]); return { symbols: sym, M: 4294967296, extraBits: 0, signed: sym, offset: 0, desc: 'float32 bit patterns' }; }
    if (state.domain === 'weights' || state.domain === 'activations') { const sym = new Int32Array(vals.length); for (let i = 0; i < vals.length; i++) sym[i] = toHalfBits(vals[i]); return { symbols: sym, M: 65536, extraBits: 0, signed: sym, offset: 0, desc: 'fp16 bit patterns' }; }
    // transform coefficients with no quantizer: round to integers to get symbols (a real lossless codec uses an integer transform)
    if (forCoding && !state.roundedWarned) { state.roundedWarned = true; warn(state, 'Transform coefficients were rounded to integers to form symbols: a lossless codec would use an integer (reversible) transform here. The rounding error is not counted as distortion.'); }
    const r = new Int32Array(vals.length); let min = Infinity, max = -Infinity;
    for (let i = 0; i < vals.length; i++) { r[i] = Math.round(vals[i]); if (r[i] < min) min = r[i]; if (r[i] > max) max = r[i]; }
    const sym = new Int32Array(vals.length); for (let i = 0; i < vals.length; i++) sym[i] = r[i] - min;
    return { symbols: sym, M: max - min + 1, extraBits: 0, signed: r, offset: -min, desc: 'rounded coefficients', rounded: true };
  }
  function streamEntropy(state) { const s = symbolize(state); const e = entropyOf(s.symbols); return { H: e.H, n: s.symbols.length, alphabet: e.alphabet, M: s.M, sym: s }; }
  function compaction(values, frac) { // share of energy held by the largest `frac` of |values|
    const mags = Array.from(values, Math.abs).sort((a, b) => b - a); let tot = 0, top = 0; const k = Math.max(1, Math.round(mags.length * frac));
    for (let i = 0; i < mags.length; i++) { tot += mags[i] * mags[i]; if (i < k) top += mags[i] * mags[i]; }
    return tot ? top / tot : 1;
  }
  function allData(planes) { return scanValues(planes); }
  function planeIsImage(p) { return p.h >= 8 && p.w % 8 === 0 && p.h % 8 === 0; }

  // ============================================================ PRIOR stage
  function stagePrior(state, f) {
    const k = state.kind;
    if (f.id === 'tables') {
      state.flags.tables = true; state.sideInfoBytes += 200;
      emit(state, 'prior', f, 'Standard tables (Huffman code lengths, a quantization matrix) are baked into the format: ~200 bytes the decoder already holds, so the stream carries no table header. Tuned for typical data, never optimal for this file.', { sideInfoBytes: 200 });
    } else if (f.id === 'dictionary') {
      const dict = dictionaryBytes(k); state.flags.dict = dict; state.sideInfoBytes += dict.length;
      const isText = state.domain === 'text';
      emit(state, 'prior', f, `${dict.length} bytes of ${isText ? 'similar records, trained ahead of time,' : 'a similar sample'} are preloaded into the LZ window before the first payload byte. The decoder holds the same bytes; matches can point into them from the very start.${isText ? '' : ' For this payload kind a dictionary rarely finds literal repeats.'}`,
        { sideInfoBytes: dict.length }, vizText(bytesToSpans(dict, 'dict')));
    } else if (f.id === 'codebook') {
      state.flags.codebook = true;
      state.pending.codebookStage = emit(state, 'prior', f, 'The VQ / PQ / RVQ codebook is shipped to the decoder out of band. (No vector quantizer in this pipeline: 0 B to hold.)', { sideInfoBytes: 0 });
    } else if (f.id === 'decoder_net') {
      state.flags.decoderNet = true; state.sideInfoBytes += 20e6;
      emit(state, 'prior', f, 'A trained synthesis network (~20 MB of weights, stated constant) sits at the decoder. It is the prior: the bitstream only has to say which plausible output. Matched pair with Learned encoder if present; otherwise 20 MB with nothing to decode.', { sideInfoBytes: 20e6 });
      state.pending.decoderStage = state.stages[state.stages.length - 1];
    } else if (f.id === 'basis') {
      state.flags.basis = true;
      state.pending.basisStage = emit(state, 'prior', f, 'The decoder holds the transform basis (a fitted PCA matrix, a learned rotation) as side information, so it never travels in the stream. (No fitted transform in this pipeline yet: PCA / Hadamard rotation would use a 0-byte Hadamard without it.)', { sideInfoBytes: 0 });
    } else if (f.id === 'tiered') {
      state.flags.tiered = true; const bytes = state.rawBits / 8; state.sideInfoBytes += bytes;
      emit(state, 'prior', f, `A lossless fp16 copy of the whole cache (${fmtBytes(bytes)} here; 40 GB for a 128k Llama-3-70B context) waits in host DRAM, one PCIe hop away. It is side information measured in HBM: whatever the GPU-resident form loses can be fetched back exactly, where a selector asks for it. Nothing is ever forgotten.`, { sideInfoBytes: bytes });
    } else passThrough(state, 'prior', f, 'no setup needed.');
  }

  // ============================================================ TRANSFORM stage
  function energyShares(planes) { const e = planes.map((p) => stats(p.data).energy); const t = e.reduce((a, b) => a + b, 0) || 1; return e.map((x) => x / t); }
  function stagePrism(state, f) {
    if (state.rep !== 'planes') return passThrough(state, 'transform', f, 'channels only exist before tokenization.');
    const P = state.planes;
    if (state.domain === 'image' || state.domain === 'video') {
      const frames = [...new Set(P.map((p) => p.frame))]; const out = []; const before = energyShares(P.filter((p) => p.frame === frames[0]));
      for (const fr of frames) {
        const R = P.find((p) => p.frame === fr && p.name === 'R'), G = P.find((p) => p.frame === fr && p.name === 'G'), B = P.find((p) => p.frame === fr && p.name === 'B');
        if (!R || !G || !B) { out.push(...P.filter((p) => p.frame === fr)); continue; }
        const n = R.data.length, Y = new Float64Array(n), Co = new Float64Array(n), Cg = new Float64Array(n);
        for (let i = 0; i < n; i++) { const co = R.data[i] - B.data[i]; const t = B.data[i] + (co >> 1); const cg = G.data[i] - t; Y[i] = t + (cg >> 1); Co[i] = co; Cg[i] = cg; }
        out.push({ name: 'Y', w: R.w, h: R.h, data: Y, vkind: 'sample', frame: fr }, { name: 'Co', w: R.w, h: R.h, data: Co, vkind: 'sample', frame: fr, isChroma: true }, { name: 'Cg', w: R.w, h: R.h, data: Cg, vkind: 'sample', frame: fr, isChroma: true });
      }
      state.planes = out; const after = energyShares(out.filter((p) => p.frame === frames[0]));
      const sd = (p) => Math.sqrt(stats(p.data).energy / p.data.length - Math.pow(stats(p.data).mean, 2));
      const vY = sd(out[0]), vCo = sd(out[1]), vCg = sd(out[2]);
      return emit(state, 'transform', f, `RGB → YCoCg-R (reversible integer): R/G/B carried ${before.map((x) => pct(x)).join(' / ')} of the energy; now Y holds ${pct(after[0])} and the two chroma planes ${pct(after[1])} + ${pct(after[2])}. Spread (σ): Y ${fmt(vY, 1)}, Co ${fmt(vCo, 1)}, Cg ${fmt(vCg, 1)}. Chroma is now a small, dull signal: the Chroma subsampling can halve it, the quantization stage can be rough with it.`,
        { symbols: countSamples(state) }, vizGrid(out.filter((p) => p.frame === frames[0])));
    }
    if (state.domain === 'audio' && P.length >= 2 && P[0].vkind === 'sample') {
      const L = P[0], R = P[1], n = L.data.length, mid = new Float64Array(n), side = new Float64Array(n);
      for (let i = 0; i < n; i++) { mid[i] = (L.data[i] + R.data[i]) >> 1; side[i] = L.data[i] - R.data[i]; }
      const rL = stats(L.data).rms, rS = stats(side).rms;
      state.planes = [{ name: 'mid', w: n, h: 1, data: mid, vkind: 'sample' }, { name: 'side', w: n, h: 1, data: side, vkind: 'sample' }].concat(P.slice(2));
      return emit(state, 'transform', f, `L/R → mid/side (FLAC-style, reversible with one LSB trick). The side channel's RMS is ${fmt(rS, 0)} against ${fmt(rL, 0)} for L: ${fmt(rL / Math.max(1, rS), 1)}× smaller, so the second channel is now cheap to code.`,
        { symbols: countSamples(state) }, vizWaveform([mid, side], ['mid', 'side']));
    }
    return passThrough(state, 'transform', f, 'there are no correlated channels to decorrelate in this payload.');
  }
  function stageFrequency(state, f) {
    if (state.rep !== 'planes') return passThrough(state, 'transform', f, 'tokens have no frequency content.');
    let shifted = 0;
    for (const p of state.planes) { const s = stats(p.data); if (p.vkind === 'sample' && s.allInt && s.min >= 0 && s.max <= 255) { for (let i = 0; i < p.data.length; i++) p.data[i] -= 128; shifted++; } } // JPEG-style level shift
    const before = compaction(allData(state.planes), 0.1); let n2d = 0, n1d = 0, blocks = 0; let coefLow = 0, coefAll = 0;
    for (const p of state.planes) {
      if (f.id === 'dct') {
        if (planeIsImage(p)) { p.data = dct2d8(p); p.blockSize = 8; p.zigzag = true; n2d++; blocks += (p.w / 8) * (p.h / 8); for (let by = 0; by < p.h; by += 8) for (let bx = 0; bx < p.w; bx += 8) for (let y = 0; y < 2; y++) for (let x = 0; x < 2; x++) coefLow += Math.pow(p.data[(by + y) * p.w + bx + x], 2); }
        else if (p.w % 8 === 0) { for (let y = 0; y < p.h; y++) p.data.set(dct1d8(p.data.subarray(y * p.w, (y + 1) * p.w)), y * p.w); p.blockSize = 8; n1d++; blocks += p.data.length / 8; for (let b = 0; b < p.data.length; b += 8) coefLow += p.data[b] * p.data[b] + p.data[b + 1] * p.data[b + 1]; }
        else continue;
      } else if (f.id === 'mdct') {
        if (p.w % 32 !== 0) continue;
        for (let y = 0; y < p.h; y++) p.data.set(dct4Frames(p.data.subarray(y * p.w, (y + 1) * p.w), 32), y * p.w); p.blockSize = 32; n1d++; blocks += p.data.length / 32;
      } else if (f.id === 'dwt') {
        if (p.w < 2) continue; p.data = haar2d(p); if (p.h > 1) n2d++; else n1d++; blocks++;
        const hw = p.w >> 1, hh = Math.max(1, p.h >> 1); for (let y = 0; y < hh; y++) for (let x = 0; x < hw; x++) coefLow += Math.pow(p.data[y * p.w + x], 2);
      }
      p.vkind = 'coef'; delete p.M; coefAll += stats(p.data).energy;
    }
    if (!n2d && !n1d) return passThrough(state, 'transform', f, 'the data does not come in blocks this transform can take.');
    const after = compaction(allData(state.planes), 0.1); const small = allData(state.planes).filter((v) => Math.abs(v) < 1).length / countSamples(state);
    let caption;
    if (f.id === 'dct') caption = n2d ? `8×8 DCT-II on ${n2d} plane(s), ${blocks} blocks: the DC + 3 lowest coefficients (4 of 64) hold ${pct(coefLow / coefAll, 1)} of the energy; the largest 10% of coefficients hold ${pct(after)} (raw samples: ${pct(before)}). ${pct(small)} of coefficients are already smaller than 1.` : `8-sample DCT-II along the signal (${blocks} blocks): the 2 lowest of 8 coefficients hold ${pct(coefLow / coefAll, 1)} of the energy; top-10% compaction ${pct(before)} → ${pct(after)}. Blocks this short are not how audio is done (see MDCT).`;
    else if (f.id === 'mdct') caption = `DCT-IV on ${blocks} frames of 32 samples (stand-in for the MDCT: no 50% overlap, so block edges would click). The largest 10% of coefficients hold ${pct(after)} of the energy, up from ${pct(before)} in the time domain; ${pct(small)} of coefficients are below 1.`;
    else caption = `One Haar level (orthonormal${n2d ? ', 2-D' : ''}): the low-pass quadrant, 1/${n2d ? 4 : 2} of the coefficients, holds ${pct(coefLow / coefAll, 1)} of the energy. Top-10% compaction ${pct(before)} → ${pct(after)}. Stand-in for the 5/3 and 9/7 wavelets of JPEG 2000 with several levels.`;
    if (shifted) caption += ` (8-bit planes level-shifted by −128 first, as JPEG does.)`;
    return emit(state, 'transform', f, caption, { symbols: countSamples(state) }, vizHeatmapOf(state.planes));
  }
  function stagePCA(state, f) {
    if (state.rep !== 'planes') return passThrough(state, 'transform', f, 'nothing to rotate in a token stream.');
    const P = state.planes;
    if (state.domain === 'vectors' && P.length === 1 && P[0].h >= P[0].w) {
      const p = P[0], n = p.h, d = p.w; const { basis, eig, mean } = pcaPowerIteration(p.data, n, d, d, 80);
      const sub = 4, per = d / sub; const order = eig.map((e, i) => i).sort((a, b) => eig[b] - eig[a]);
      const groups = Array.from({ length: sub }, () => []), logsum = new Array(sub).fill(0);
      for (const c of order) { let best = 0; for (let g = 1; g < sub; g++) if (groups[g].length < per && (groups[best].length >= per || logsum[g] < logsum[best])) best = g; groups[best].push(c); logsum[best] += log2(Math.max(eig[c], 1e-12)); }
      const perm = groups.flat(); const out = new Float64Array(n * d);
      for (let i = 0; i < n; i++) for (let j = 0; j < d; j++) { let s = 0; const b = basis[perm[j]]; for (let k = 0; k < d; k++) s += (p.data[i * d + k] - mean[k]) * b[k]; out[i * d + j] = s; }
      p.data = out; p.vkind = 'coef';
      const matBytes = (d * d + d) * 4; if (state.flags.codebook) state.sideInfoBytes += matBytes; else state.headerBits += matBytes * 8;
      const origVar = []; for (let j = 0; j < d; j++) { let s = 0; for (let i = 0; i < n; i++) s += Math.pow(getPayload(state.kind).planes[0].data[i * d + j], 2); origVar.push(s / n); }
      return emit(state, 'transform', f, `Real PCA (power iteration) on ${n}×${d}: eigenvalues ${eig.map((e) => fmt(e, 2)).join(', ')} against per-axis variances of ${fmt(Math.min(...origVar), 2)}–${fmt(Math.max(...origVar), 2)} before. The axes are then dealt to 4 sub-vectors so each gets an equal variance product (parametric OPQ): a product quantizer now spends its 4 bits where they count. Rotation matrix ${matBytes} B ${state.flags.codebook ? 'held by the decoder' : 'travels in the header'}.`,
        { symbols: countSamples(state) }, vizScatter(out, n, d));
    }
    const tp = tokenPlaneOf(state);
    if (tp && state.flags.basis) { // Trained basis holds a fitted basis: real PCA across channels (KVTC style), coordinates sorted by variance
      const n = tp.h, d = tp.w; const { basis, eig, mean } = pcaPowerIteration(tp.data, n, d, d, 80);
      const out = new Float64Array(n * d); for (let i = 0; i < n; i++) for (let k = 0; k < d; k++) { let s = 0; for (let j = 0; j < d; j++) s += (tp.data[i * d + j] - mean[j]) * basis[k][j]; out[i * d + k] = s; }
      tp.data = out; tp.vkind = 'coef'; delete tp.M; const bytes = (d * d + d) * 2; state.sideInfoBytes += bytes;
      if (state.pending.basisStage) { state.pending.basisStage.caption = `The decoder holds the ${d}×${d} orthonormal PCA basis plus the ${d}-dim mean (${bytes} B, fitted once; KVTC fits one on 160k calibration tokens and ships it with the model). Without it the coefficients are numbers in an unknown frame.`; state.pending.basisStage.metrics.sideInfoBytes = bytes; }
      const top = eig.slice(0, 4).reduce((a, b) => a + b, 0) / (eig.reduce((a, b) => a + b, 0) || 1);
      return emit(state, 'transform', f, `Real PCA across the ${d} channels (power iteration), mean removed: eigenvalues ${eig.slice(0, 4).map((e) => fmt(e, 1)).join(', ')} … ${eig.slice(-2).map((e) => fmt(e, 2)).join(', ')}; the top 4 coordinates hold ${pct(top, 1)} of the variance, so a bit allocator has something to allocate. The basis (${bytes} B) is Trained basis's side information; a Hadamard would cost 0 bytes but spreads variance evenly instead of sorting it.`,
        { symbols: countSamples(state) }, vizHeatmapOf([tp]));
    }
    const total = countSamples(state); const N = total % 32 === 0 ? 32 : total % 8 === 0 ? 8 : 0;
    if (!N) return passThrough(state, 'transform', f, 'sample count is not a multiple of 8.');
    let maxBefore = 0, maxAfter = 0, nb = 0;
    for (const p of P) { for (let b = 0; b + N <= p.data.length; b += N) { let m = 0; for (let i = b; i < b + N; i++) m = Math.max(m, Math.abs(p.data[i])); maxBefore += m; nb++; } p.data = hadamardBlocks(p.data, N); for (let b = 0; b + N <= p.data.length; b += N) { let m = 0; for (let i = b; i < b + N; i++) m = Math.max(m, Math.abs(p.data[i])); maxAfter += m; } p.vkind = 'coef'; delete p.M; delete p.blockSize; }
    const q99 = quantile(Array.from(allData(P), Math.abs), 0.99), rms = stats(allData(P)).rms;
    return emit(state, 'transform', f, `Normalized ${N}-point Hadamard rotation on each block of ${N} values (QuaRot / SpinQuant style, orthonormal so the quantization error maps 1:1). Mean block peak |x| ${fmt(maxBefore / nb, 3)} → ${fmt(maxAfter / nb, 3)}: outliers are smeared across the block, so a per-block absmax scale is no longer hijacked by one channel. Peak/RMS now ${fmt(q99 / (rms || 1), 1)} at the 99th percentile.`,
      { symbols: total }, P[0].h > 1 ? vizHeatmapOf(P) : vizWaveform(P.slice(0, 2).map((p) => p.data), P.slice(0, 2).map((p) => p.name)));
  }
  // ---- level 5 (KV cache) transforms
  function channelRanges(p) { const r = new Float64Array(p.w); for (let j = 0; j < p.w; j++) { let lo = Infinity, hi = -Infinity; for (let i = 0; i < p.h; i++) { const v = p.data[i * p.w + j]; if (v < lo) lo = v; if (v > hi) hi = v; } r[j] = hi - lo; } return r; }
  function stageUnrope(state, f) {
    const P = state.rep === 'planes' ? state.planes.filter((p) => p.rope && p.tokenIds) : [];
    if (!P.length) return passThrough(state, 'transform', f, 'nothing here was rotated by RoPE (only cached keys are).');
    const p = P[0]; const before = channelRanges(p);
    for (const q of P) { q.data = ropeRotate(q.data, q.h, q.w, -1, q.tokenIds); q.rope = false; q.unroped = true; q.vkind = 'coef'; }
    const after = channelRanges(p); const mean = (a) => a.reduce((s, v) => s + v, 0) / a.length;
    const moved = Array.from(before, (b, j) => ({ j, b, a: after[j] })).sort((u, v) => (v.b - v.a) - (u.b - u.a)).slice(0, 2);
    const still = Array.from(before, (b, j) => ({ j, b, a: after[j] })).filter((o) => o.b > 6 * mean(before) / 2 && Math.abs(o.b - o.a) < 0.15 * o.b).slice(0, 2);
    return emit(state, 'transform', f, `Undo RoPE per token (angle = position × θ_pair, θ_pair = 10000^(−2i/${p.w})): 0 bits, the decoder knows every position. Per-channel range (max−min) over ${p.h} tokens: ${moved.map((o) => `channel ${o.j} ${fmt(o.b, 1)} → ${fmt(o.a, 1)}`).join(', ')}; mean over all channels ${fmt(mean(before), 1)} → ${fmt(mean(after), 1)}.${still.length ? ` Channel${still.length > 1 ? 's' : ''} ${still.map((o) => o.j).join('/')} barely moved: their pair rotates ${fmt(Math.pow(10000, -2 * Math.floor(still[0].j / 2) / p.w), 4)} rad per token, too slowly to wander.` : ''} A stable outlier channel now sits still from token to token, so per-channel scales, codebooks and low-rank bases can fit it.`,
      { symbols: countSamples(state) }, vizHeatmapOf(P));
  }
  function stageSVD(state, f) {
    const tp = tokenPlaneOf(state); const vec = !tp && state.domain === 'vectors' && state.rep === 'planes' && state.planes.length === 1 ? state.planes[0] : null; const p = tp || vec;
    if (!p) return passThrough(state, 'transform', f, 'the SVD stand-in only knows a token×channel (or vector) matrix.');
    const n = p.h, d = p.w, r = Math.max(1, Math.round(d / 4)); const { basis, eig } = pcaPowerIteration(p.data, n, d, d, 80, true);
    const sv = eig.map((e) => Math.sqrt(Math.max(0, e))); let total = 0; for (let i = 0; i < p.data.length; i++) total += p.data[i] * p.data[i];
    const A = new Float64Array(n * r), B = new Float64Array(r * d); let kept = 0;
    for (let i = 0; i < n; i++) { let rowE = 0, rowK = 0; for (let k = 0; k < r; k++) { let s = 0; for (let j = 0; j < d; j++) s += p.data[i * d + j] * basis[k][j]; A[i * r + k] = s; rowK += s * s; } for (let j = 0; j < d; j++) rowE += p.data[i * d + j] * p.data[i * d + j]; kept += rowK; addErr(state, p, i * d, Math.max(0, rowE - rowK)); }
    for (let k = 0; k < r; k++) for (let j = 0; j < d; j++) B[k * d + j] = basis[k][j];
    state.lossless = false; state.hasQuantizer = true;
    const Aplane = { name: 'A (L×r)', w: r, h: n, data: A, vkind: 'coef', tokenIds: p.tokenIds }, Bplane = { name: 'B (r×d)', w: d, h: r, data: B, vkind: 'coef' };
    state.planes = state.planes.map((q) => (q === p ? Aplane : q)); state.planes.push(Bplane);
    const dropped = Math.max(0, total - kept);
    return emit(state, 'transform', f, `SVD of the ${n}×${d} matrix (power iteration): singular values ${sv.slice(0, r).map((s) => fmt(s, 1)).join(', ')} | ${sv.slice(r, r + 3).map((s) => fmt(s, 1)).join(', ')}… Rank ${r} of ${d} kept: factors A (${n}×${r}) and B (${r}×${d}) = ${n * r + r * d} numbers instead of ${n * d} (${fmt(n * d / (n * r + r * d), 1)}:1); the ${d - r} dropped directions held ${pct(dropped / (total || 1), 2)} of ‖X‖². Any row is rebuilt on demand as A[i]·B. ShadowKV keeps rank 160 of 1024 on pre-RoPE keys${p.rope ? '; these keys still carry RoPE, which inflates the rank' : ''}.`,
      { symbols: countSamples(state), distortion: currentDistortion(state) }, vizHeatmapOf([Aplane]));
  }
  function stageMLA(state, f) {
    const p = tokenPlaneOf(state); if (!p) return passThrough(state, 'transform', f, 'a latent bottleneck is decided at pretraining for KV caches; nothing to apply here.');
    const n = p.h, d = p.w, r = Math.max(1, Math.round(d / 4)); const { basis, eig, mean } = pcaPowerIteration(p.data, n, d, r, 60);
    const lat = new Float64Array(n * r); let total = 0, kept = 0;
    for (let i = 0; i < n; i++) { let rowE = 0, rowK = 0; for (let j = 0; j < d; j++) rowE += Math.pow(p.data[i * d + j] - mean[j], 2); for (let k = 0; k < r; k++) { let s = 0; for (let j = 0; j < d; j++) s += (p.data[i * d + j] - mean[j]) * basis[k][j]; lat[i * r + k] = s; rowK += s * s; } total += rowE; kept += rowK; addErr(state, p, i * d, Math.max(0, rowE - rowK)); }
    state.lossless = false; state.hasQuantizer = true; const wBytes = (d * r + d) * 2; state.sideInfoBytes += wBytes;
    const Lp = { name: 'latent', w: r, h: n, data: lat, vkind: 'coef', tokenIds: p.tokenIds }; state.planes = state.planes.map((q) => (q === p ? Lp : q));
    return emit(state, 'transform', f, `Cache a ${r}-dim latent per token instead of ${d} channels: ${fmt(d / r, 1)}:1 before any quantizer, zero encode cost and zero decode cost because the up-projection is absorbed into W_Q and W_O (the latent is attended to directly). The dropped directions held ${pct((total - kept) / (total || 1), 2)} of the centred variance (eigenvalues kept: ${eig.map((e) => fmt(e, 1)).join(', ')}). Stand-in: a post-hoc PCA; the real bottleneck is trained with the model and is stronger, not weaker, than full MHA. Its ${wBytes} B projection is part of the model weights.`,
      { symbols: countSamples(state), sideInfoBytes: wBytes, distortion: currentDistortion(state) }, vizHeatmapOf([Lp]));
  }
  function stageBWT(state, f) {
    if (state.rep !== 'planes') return passThrough(state, 'transform', f, 'BWT sorts bytes, not tokens.');
    const ser = serializePlanes(state.planes); const bytes = ser.bytes; if (bytes.length < 4) return passThrough(state, 'transform', f, 'too short.');
    const H0 = entropyOf(bytes).H; const { last, primary } = bwt(bytes); const mtf = mtfEncode(last);
    let runs = 0, maxRun = 0, cur = 0; for (let i = 0; i < last.length; i++) { if (i > 0 && last[i] === last[i - 1]) cur++; else { if (cur + 1 >= 3) runs++; maxRun = Math.max(maxRun, cur + 1); cur = 0; } } maxRun = Math.max(maxRun, cur + 1);
    const H1 = entropyOf(mtf).H; const zeros = mtf.filter((v) => v === 0).length / mtf.length;
    state.planes = [{ name: 'MTF', w: mtf.length, h: 1, data: Float64Array.from(mtf), vkind: 'byte', M: 256 }]; state.headerBits += ceilLog2(bytes.length); state.bwtPrimary = primary;
    const spans = []; let i = 0; while (i < last.length && spans.length < 1500) { let j = i; while (j + 1 < last.length && last[j + 1] === last[i]) j++; const txt = Array.from(last.subarray(i, j + 1), printable).join(''); spans.push({ text: txt, cls: j - i + 1 >= 3 ? 'run' : 'lit' }); i = j + 1; }
    return emit(state, 'transform', f, `Burrows–Wheeler on ${bytes.length} bytes${ser.width > 1 ? ` (${ser.width} bytes per sample)` : ''}: sorting all rotations clustered equal bytes into ${runs} runs of 3+ (longest ${maxRun}). Move-to-front turns runs into zeros: ${pct(zeros)} of symbols are now 0 and order-0 entropy drops ${fmt(H0, 2)} → ${fmt(H1, 2)} bits/byte. Fully reversible with one ${ceilLog2(bytes.length)}-bit index.`,
      { symbols: mtf.length, alphabet: 256, entropyBitsPerSymbol: H1 }, vizText(spans));
  }
  function stageNeuralEnc(state, f) {
    if (state.rep !== 'planes') return passThrough(state, 'transform', f, 'nothing to encode in a token stream.');
    const P = state.planes; let X, n, d, what;
    if ((state.domain === 'image' || state.domain === 'video') && P.every(planeIsImage)) {
      d = 64; const rows = []; for (const p of P) for (let by = 0; by < p.h; by += 8) for (let bx = 0; bx < p.w; bx += 8) { const v = new Float64Array(64); for (let y = 0; y < 8; y++) for (let x = 0; x < 8; x++) v[y * 8 + x] = p.data[(by + y) * p.w + bx + x]; rows.push(v); }
      n = rows.length; X = new Float64Array(n * d); rows.forEach((v, i) => X.set(v, i * d)); what = `${n} patches of 8×8`;
    } else if (state.domain === 'audio' && P.every((p) => p.w % 32 === 0)) {
      d = 32; const rows = []; for (const p of P) for (let b = 0; b + 32 <= p.data.length; b += 32) rows.push(p.data.subarray(b, b + 32));
      n = rows.length; X = new Float64Array(n * d); rows.forEach((v, i) => X.set(v, i * d)); what = `${n} frames of 32 samples`;
    } else return passThrough(state, 'transform', f, 'the stand-in encoder only knows image patches and audio frames.');
    const keep = 16; const { basis, mean } = pcaPowerIteration(X, n, d, keep, 50);
    const lat = new Float64Array(n * keep); let totE = 0, keptE = 0;
    for (let i = 0; i < n; i++) { for (let j = 0; j < d; j++) totE += Math.pow(X[i * d + j] - mean[j], 2); for (let k = 0; k < keep; k++) { let s = 0; for (let j = 0; j < d; j++) s += (X[i * d + j] - mean[j]) * basis[k][j]; lat[i * keep + k] = s; keptE += s * s; } }
    const dropped = Math.max(0, totE - keptE); state.seSum += dropped; state.lossless = false;
    state.planes = [{ name: 'latent', w: keep, h: n, data: lat, vkind: 'coef' }]; state.flags.neuralEnc = true;
    const netBytes = state.flags.decoderNet ? 0 : 20e6; state.sideInfoBytes += netBytes; if (!state.flags.decoderNet) warn(state, 'Learned encoder produced latents that only a matched decoder network can read; without Learned decoder in the shared-prior stage the ~20 MB decoder still has to exist somewhere.');
    if (state.pending.decoderStage) state.pending.decoderStage.caption = 'A trained synthesis network (~20 MB of weights, stated constant) sits at the decoder, matched to the learned encoder: the pair was trained together, so the latents mean something only to this decoder.';
    return emit(state, 'transform', f, `${what} → ${keep} latents each (${n * d} numbers in, ${n * keep} out, ${fmt(d / keep, 0)}:1 fewer). The ${d - keep} dropped components held ${pct(dropped / (totE || 1), 1)} of the variance: the first lossy step, before any quantizer. Stand-in: linear PCA on patches; the real thing is a nonlinear conv/transformer analysis net trained with its decoder (~20 MB of weights${state.flags.decoderNet ? ', counted at Learned decoder' : ' counted here as side information'}).`,
      { symbols: n * keep, sideInfoBytes: netBytes }, vizHeatmapOf(state.planes));
  }
  function stageTransform(state, f) {
    if (f.id === 'prism') return stagePrism(state, f);
    if (f.id === 'dct' || f.id === 'mdct' || f.id === 'dwt') return stageFrequency(state, f);
    if (f.id === 'pca') return stagePCA(state, f);
    if (f.id === 'bwt') return stageBWT(state, f);
    if (f.id === 'neural_enc') return stageNeuralEnc(state, f);
    if (f.id === 'unrope') return stageUnrope(state, f);
    if (f.id === 'svd_lowrank') return stageSVD(state, f);
    if (f.id === 'mla') return stageMLA(state, f);
    return passThrough(state, 'transform', f, 'unknown transform.');
  }

  // ============================================================ PERCEPTUAL stage
  function stagePerceptual(state, f) {
    if (f.id === 'chroma') {
      if (state.rep !== 'planes') return passThrough(state, 'perceptual', f, 'no planes left to subsample.');
      const chroma = state.planes.filter((p) => p.isChroma && p.w % 2 === 0 && p.h % 2 === 0 && p.h > 1);
      if (!chroma.length) { warn(state, 'Chroma subsampling found no chroma planes to subsample: RGB has no separate colour channels. Put YCbCr / mid-side in the transform stage first.'); return passThrough(state, 'perceptual', f, 'there are no chroma planes (needs YCbCr / mid-side before it).'); }
      let before = 0, after = 0, se = 0;
      for (const p of chroma) {
        const w2 = p.w / 2, h2 = p.h / 2, out = new Float64Array(w2 * h2);
        for (let y = 0; y < h2; y++) for (let x = 0; x < w2; x++) { const i = 2 * y * p.w + 2 * x; out[y * w2 + x] = Math.round((p.data[i] + p.data[i + 1] + p.data[i + p.w] + p.data[i + p.w + 1]) / 4); }
        for (let y = 0; y < p.h; y++) for (let x = 0; x < p.w; x++) se += Math.pow(p.data[y * p.w + x] - out[(y >> 1) * w2 + (x >> 1)], 2);
        before += p.data.length; after += out.length; p.data = out; p.w = w2; p.h = h2;
      }
      state.seSum += se; state.lossless = false;
      return emit(state, 'perceptual', f, `4:2:0 chroma subsampling: ${chroma.map((p) => p.name).filter((v, i, a) => a.indexOf(v) === i).join('/')} stored at ${chroma[0].w}×${chroma[0].h} instead of ${chroma[0].w * 2}×${chroma[0].h * 2}. ${before} chroma samples → ${after}; total samples now ${countSamples(state)} of ${state.nOrig}. Nearest-neighbour upsampling error RMS ${fmt(Math.sqrt(se / before), 1)} on chroma. Colour acuity is roughly half of luma acuity, so this is the cheapest 2:1 in imaging. Lossless is gone.`,
        { symbols: countSamples(state) }, vizGrid(state.planes.filter((p) => p.frame === state.planes[0].frame)));
    }
    if (f.id === 'psychoacoustic') {
      state.flags.maskThreshold = 0.02;
      state.pending.maskStage = emit(state, 'perceptual', f, 'Masking threshold armed: anything below 2% of its frame\'s peak will be zeroed before quantization. (Waiting for the transform stage.)', {});
      return state.pending.maskStage;
    }
    if (f.id === 'sinkguard') {
      const p = tokenPlaneOf(state); if (!p) return passThrough(state, 'perceptual', f, 'there are no tokens to exempt in this payload.');
      const ids = Array.from(p.tokenIds); const n = ids.length; const set = new Set(ids.slice(0, 2).concat(ids.slice(Math.max(2, n - 4)))); state.protectedTokens = set; const bits = set.size * p.w * 16;
      return emit(state, 'perceptual', f, `Tokens ${[...set].join(', ')} (the first 2 attention sinks and the newest 4) bypass every quantizer and stay fp16: ${set.size} of ${n} tokens = ${pct(set.size / n)} of this cache, ${fmtInt(bits)} bits of memory. At 128k tokens the same 6 tokens would be 0.005%; KIVI's 128-token residual window is 0.1%. Attention amplifies error on exactly these tokens, so this is where precision buys the most.`, { symbols: set.size * p.w });
    }
    if (f.id === 'spotlight') {
      const p = tokenPlaneOf(state); if (!p) return passThrough(state, 'perceptual', f, 'there is no query and no softmax here: only cached keys have an audience.');
      const n = p.h, d = p.w; const K = p.unroped ? ropeRotate(p.data, n, d, 1, p.tokenIds) : p.data; // attention always sees RoPE'd keys, whatever the cache stores
      const q = new Float64Array(d); for (let j = 0; j < d; j++) q[j] = (K[(n - 1) * d + j] + K[(n - 2) * d + j]) / 2;
      const logits = new Float64Array(n); for (let i = 0; i < n; i++) { let s = 0; for (let j = 0; j < d; j++) s += q[j] * K[i * d + j]; logits[i] = s / Math.sqrt(d); }
      // synthetic keys have no trained W_Q to keep q·k in range: standardize the logits to std 1.5 (a stand-in for the learned query scale)
      const ls = stats(logits); const sd = Math.sqrt(Math.max(1e-12, ls.rms * ls.rms - ls.mean * ls.mean)); let mx = -Infinity; for (let i = 0; i < n; i++) { logits[i] = 1.5 * (logits[i] - ls.mean) / sd; if (logits[i] > mx) mx = logits[i]; }
      const w = new Float64Array(state.nTokens); let Z = 0; for (let i = 0; i < n; i++) { const e = Math.exp(logits[i] - mx); w[p.tokenIds[i]] = e; Z += e; } for (let t = 0; t < w.length; t++) w[t] /= Z;
      state.tokenWeights = w; const order = Array.from(p.tokenIds).sort((a, b) => w[b] - w[a]); const topK = Math.ceil(n * 0.25); state.spotlightTop = new Set(order.slice(0, topK)); let mass = 0; for (const t of order.slice(0, topK)) mass += w[t];
      return emit(state, 'perceptual', f, `Query = mean of the newest 2 keys; softmax over ${n} tokens of q·k/√${d}, logits standardized to std 1.5 (stand-in for the trained query scale): the top ${topK} tokens (25%: ${order.slice(0, topK).join(', ')}) hold ${pct(mass)} of the attention mass, token ${order[0]} alone ${pct(w[order[0]])}${order[0] === 0 ? ' (the attention sink)' : ''}; the bottom half of the tokens share ${pct(order.slice(Math.floor(n / 2)).reduce((a, t) => a + w[t], 0), 1)}. From here on distortion is weighted by these scores: error on a token the softmax ignores costs nothing. No bits change. The proxy is the address book, read in full every step: ${n} × ${d} × 16 bits = ${fmtBytes(n * d * 2)} here (DSA's indexer: 64 heads × 128 dims in FP8 per token).`,
        { symbols: n }, vizWaveform([w], ['attention weight per token']));
    }
    if (f.id === 'gan') return emit(state, 'perceptual', f, `0 bits changed, 0 samples touched: an adversarial / feature loss changes what the decoder is trained to produce (plausible detail instead of the blurry mean), not what is transmitted. The ${countSamples(state)} samples pass through. It only means something next to a learned decoder${state.flags.decoderNet || state.flags.neuralEnc ? ' (present in this pipeline)' : '; with sample-exact algorithms it is a spectator'}.`, { symbols: countSamples(state) });
    return passThrough(state, 'perceptual', f, 'unknown perceptual model.');
  }
  function applyProtection(state) { // Sink & window exemption: protected tokens leave the stream as raw fp16 before any quantizer sees them
    const set = state.protectedTokens; if (!set || !set.size || state.rep !== 'planes') return;
    for (const p of state.planes) {
      if (!p.tokenIds || p.h !== p.tokenIds.length) continue;
      const keep = []; for (let i = 0; i < p.h; i++) if (!set.has(p.tokenIds[i])) keep.push(i);
      if (keep.length === p.h || keep.length === 0) continue;
      const out = new Float64Array(keep.length * p.w), ids = new Int32Array(keep.length);
      keep.forEach((i, k) => { out.set(p.data.subarray(i * p.w, (i + 1) * p.w), k * p.w); ids[k] = p.tokenIds[i]; });
      state.extraBits += (p.h - keep.length) * p.w * 16; p.data = out; p.h = keep.length; p.tokenIds = ids;
    }
  }
  function applyMasking(state) { // called once transforms are done
    const st = state.pending.maskStage; if (!st || state.rep !== 'planes') return;
    let zeroed = 0, total = 0, removed = 0, energy = 0;
    for (const p of state.planes) {
      const N = p.blockSize === 8 && p.zigzag ? 0 : (p.blockSize || 32);
      const blocks = [];
      if (N === 0) { for (let by = 0; by < p.h; by += 8) for (let bx = 0; bx < p.w; bx += 8) { const idx = []; for (let y = 0; y < 8; y++) for (let x = 0; x < 8; x++) idx.push((by + y) * p.w + bx + x); blocks.push(idx); } }
      else for (let b = 0; b < p.data.length; b += N) { const idx = []; for (let i = b; i < Math.min(p.data.length, b + N); i++) idx.push(i); blocks.push(idx); }
      for (const idx of blocks) {
        let m = 0; for (const i of idx) m = Math.max(m, Math.abs(p.data[i]));
        const thr = state.flags.maskThreshold * m;
        for (const i of idx) { const v = p.data[i]; total++; energy += v * v; if (v !== 0 && Math.abs(v) < thr) { removed += v * v; p.data[i] = 0; zeroed++; } }
      }
    }
    if (zeroed) state.lossless = false; state.seSum += removed;
    st.caption = `Masking threshold stand-in: in each frame/block, coefficients below 2% of that block's peak are declared inaudible and zeroed before quantization. ${zeroed} of ${total} values (${pct(zeroed / (total || 1))}) went to zero, removing ${pct(removed / (energy || 1), 2)} of the energy (${fmt(10 * Math.log10(energy / Math.max(removed, 1e-9)), 0)} dB below the signal). A real model computes per-band thresholds from loud components with a spreading function and needs lookahead.`;
    st.metrics = { symbols: total }; st.viz = vizHeatmapOf(state.planes);
    if (!state.planes.some((p) => p.vkind === 'coef')) warn(state, 'Psychoacoustic model is zeroing raw samples: a masking model only makes sense on frequency-domain coefficients (put MDCT or DCT in the transform stage).');
  }

  // ============================================================ PREDICT stage
  const wrapByte = (d) => ((((d % 256) + 256 + 128) % 256) - 128); // residual of 8-bit data folded into -128..127 (PNG's mod-256, bijective)
  function residualPlanes(state, fn, label) { // fn(rowIn, rowOut, plane) fills residuals for one row
    const before = streamEntropy(state); const rms0 = stats(allData(state.planes)).rms;
    for (const p of state.planes) {
      const s = stats(p.data); const wrap = s.allInt && s.min >= 0 && s.max <= 255; const out = new Float64Array(p.data.length);
      for (let y = 0; y < p.h; y++) fn(p.data.subarray(y * p.w, (y + 1) * p.w), out.subarray(y * p.w, (y + 1) * p.w), p, y);
      if (wrap) for (let i = 0; i < out.length; i++) out[i] = wrapByte(out[i]);
      p.data = out; p.vkind = 'residual'; delete p.M; delete p.zigzag;
    }
    const after = streamEntropy(state); const rms1 = stats(allData(state.planes)).rms;
    return { before, after, rms0, rms1, text: `order-0 entropy ${fmt(before.H, 2)} → ${fmt(after.H, 2)} bits/sample (alphabet ${before.alphabet} → ${after.alphabet}), RMS ${fmt(rms0, 1)} → ${fmt(rms1, 1)}` };
  }
  function residualViz(state) {
    const P = state.planes; if (!P.length) return vizText([{ text: '(empty)', cls: 'lit' }]);
    if (P[0].h === 1) return vizWaveform(P.slice(0, 2).map((p) => p.data), P.slice(0, 2).map((p) => p.name + ' residual'));
    return vizGrid(P.filter((p) => p.frame === P[0].frame));
  }
  function stagePredict(state, f) {
    if (f.id === 'cm' || f.id === 'neural_prior') {
      state.flags.contextModel = f.id;
      if (f.id === 'neural_prior') state.sideInfoBytes += 20e6;
      const s = symbolize(state); const H0 = entropyOf(s.symbols).H, H2 = conditionalEntropyOrder2(s.symbols);
      return emit(state, 'predict', f, `Order-0 entropy of the stream is ${fmt(H0, 2)} bits/symbol; given the two previous symbols it is ${fmt(H2, 2)} (empirical, with the model's own learning cost not yet paid). ${f.name} hands P(next | context) to the entropy-coding stage: this only pays off with an arithmetic-style coder (Arithmetic coding / ANS). Stand-in: an order-2 adaptive PPM model${f.id === 'neural_prior' ? '; the real thing is a network whose ~20 MB of weights the decoder must hold' : '; PAQ mixes dozens of contexts with a tiny neural net'}.`,
        { symbols: s.symbols.length, entropyBitsPerSymbol: H0, sideInfoBytes: f.id === 'neural_prior' ? 20e6 : 0 }, vizHistogram(s.symbols, s.M, s.offset));
    }
    if (state.rep !== 'planes') return passThrough(state, 'predict', f, 'the data is already a token stream.');
    if (f.id === 'lz77') {
      const ser = serializePlanes(state.planes); const dict = state.flags.dict; const tokens = lz77(ser.bytes, dict, 4096, 3, 258);
      let lits = 0, matches = 0, covered = 0, dictM = 0; for (const t of tokens) { if (t.type === 'lit') lits++; else { matches++; covered += t.len; if (t.dict) dictM++; } }
      state.rep = 'tokens'; state.tokens = tokens; state.lzMeta = { bytes: ser.bytes.length, width: ser.width, dictLen: dict ? dict.length : 0 };
      const sym = lzSymbols(tokens); const e = entropyOf(sym.symbols);
      return emit(state, 'predict', f, `${state.lzDeferred ? 'Runs after the quantizer, on its indices (DEFLATE after quantization, as KVTC and PNG order it). ' : ''}${ser.bytes.length} bytes in${ser.width > 1 ? ` (${ser.width} bytes per sample)` : ''} → ${lits} literals + ${matches} matches (average length ${fmt(matches ? covered / matches : 0, 1)}) covering ${pct(covered / ser.bytes.length)} of the input${dict ? `; ${dictM} match${dictM === 1 ? ' points' : 'es point'} into the ${dict.length}-byte preloaded dictionary (${pct(tokens.filter((t) => t.dict).reduce((a, t) => a + t.len, 0) / ser.bytes.length)} of the input)` : ''}. Window 4096, min match 3, greedy. Tokens become ${sym.symbols.length} symbols for the entropy-coding stage plus ${sym.extraBits} raw distance bits.`,
        { symbols: sym.symbols.length, alphabet: e.alphabet, entropyBitsPerSymbol: e.H }, lzSpans(tokens, dict));
    }
    if (f.id === 'lzw') {
      const ser = serializePlanes(state.planes); const res = lzw(ser.bytes, 12); state.rep = 'lzw'; state.lzw = res; state.lzMeta = { bytes: ser.bytes.length, width: ser.width };
      const e = entropyOf(res.codes);
      const spans = []; const phrases = lzwPhrases(res.codes); let alt = 0; for (const ph of phrases) { spans.push({ text: Array.from(ph, printable).join(''), cls: ph.length > 1 ? (alt++ % 2 ? 'match' : 'run') : 'lit', title: `phrase of ${ph.length}` }); if (spans.length > 1500) break; }
      return emit(state, 'predict', f, `${ser.bytes.length} bytes → ${res.codes.length} phrase codes; the dictionary grew to ${res.dictSize} entries, average phrase ${fmt(ser.bytes.length / res.codes.length, 2)} bytes. Both sides build the same table, so only numbers travel: ${res.varBits} bits with GIF-style 9–12-bit growing code widths if sent raw.`,
        { symbols: res.codes.length, alphabet: e.alphabet, entropyBitsPerSymbol: e.H }, vizText(spans));
    }
    if (f.id === 'delta') {
      const r = residualPlanes(state, (x, out) => { out[0] = x[0]; for (let i = 1; i < x.length; i++) out[i] = x[i] - x[i - 1]; });
      return emit(state, 'predict', f, `x̂[n] = x[n−1], send x[n] − x̂[n] along each row: ${r.text}. ${r.after.H < r.before.H ? 'The residual peaks at zero: exactly what Rice and Huffman love.' : 'No gain: neighbouring values are not similar in this data.'}`,
        { symbols: r.after.n, alphabet: r.after.alphabet, entropyBitsPerSymbol: r.after.H }, residualViz(state));
    }
    if (f.id === 'lpc') {
      const r = residualPlanes(state, (x, out) => { out[0] = x[0]; if (x.length > 1) out[1] = x[1] - x[0]; for (let i = 2; i < x.length; i++) out[i] = x[i] - (2 * x[i - 1] - x[i - 2]); });
      return emit(state, 'predict', f, `FLAC fixed predictor, order 2: x̂[n] = 2x[n−1] − x[n−2] (extrapolate the slope), code the residual. ${r.text}. Stand-in for adaptive LPC, which would fit the filter per block and send its coefficients.`,
        { symbols: r.after.n, alphabet: r.after.alphabet, entropyBitsPerSymbol: r.after.H }, residualViz(state));
    }
    if (f.id === 'spatial') {
      const counts = [0, 0, 0, 0, 0]; const names = ['None', 'Sub', 'Up', 'Average', 'Paeth']; let rows = 0;
      const r = residualPlanes(state, (x, out, p, y) => {
        const up = y > 0 ? p.data.subarray((y - 1) * p.w, y * p.w) : null; const cands = [];
        for (let t = 0; t < 5; t++) {
          if ((t === 2 || t === 3 || t === 4) && !up) continue;
          const res = new Float64Array(x.length); let cost = 0;
          for (let i = 0; i < x.length; i++) {
            const a = i > 0 ? x[i - 1] : 0, b = up ? up[i] : 0, c = up && i > 0 ? up[i - 1] : 0; let pr = 0;
            if (t === 1) pr = a; else if (t === 2) pr = b; else if (t === 3) pr = Math.floor((a + b) / 2);
            else if (t === 4) { const pp = a + b - c, pa = Math.abs(pp - a), pb = Math.abs(pp - b), pc = Math.abs(pp - c); pr = pa <= pb && pa <= pc ? a : pb <= pc ? b : c; }
            res[i] = x[i] - pr; cost += Math.abs(res[i]);
          }
          cands.push({ t, res, cost });
        }
        cands.sort((u, v) => u.cost - v.cost); out.set(cands[0].res); counts[cands[0].t]++; rows++;
      });
      state.extraBits += rows * 3;
      const chosen = names.map((nm, i) => counts[i] ? `${nm} ×${counts[i]}` : null).filter(Boolean).join(', ');
      return emit(state, 'predict', f, `PNG-style filter per row, best of None/Sub/Up/Average/Paeth by least |residual|: ${chosen}. ${r.text}. ${rows} filter choices ride along as ${rows * 3} raw bits.${state.planes[0].h === 1 ? ' (1-D data: only Sub applies.)' : ''}`,
        { symbols: r.after.n, alphabet: r.after.alphabet, entropyBitsPerSymbol: r.after.H }, residualViz(state));
    }
    if (f.id === 'anchor_delta') {
      const before = streamEntropy(state); const G = 8; let anchors = 0, anchorBits = 0, varRaw = 0, varDelta = 0, nRaw = 0, nDelta = 0; const out = [];
      const f0 = state.planes.filter((p) => p.frame === 0), f1 = state.planes.filter((p) => p.frame === 1);
      if (state.domain === 'video' && f0.length && f0.length === f1.length) { // frame 0 is the anchor (I-frame), frame 1 a difference from it
        f1.forEach((p, k) => { const r = f0[k]; const d = new Float64Array(p.data.length); for (let i = 0; i < d.length; i++) { d[i] = p.data[i] - r.data[i]; varRaw += p.data[i] * p.data[i]; varDelta += d[i] * d[i]; } nRaw += d.length; nDelta += d.length; p.data = d; p.vkind = 'residual'; delete p.M; });
        for (const r of f0) { anchors += r.data.length; anchorBits += r.data.length * state.bitDepth; }
        state.planes = f1;
      } else {
        for (const p of state.planes) {
          const s = stats(p.data); const wrap = s.allInt && s.min >= 0 && s.max <= 255;
          if (p.h > 1) { // rows are tokens: one anchor per group of G rows (the row nearest the group mean), the rest are differences from it
            const anchorOf = new Int32Array(p.h);
            for (let g0 = 0; g0 < p.h; g0 += G) { const g1 = Math.min(p.h, g0 + G); const mean = new Float64Array(p.w); for (let i = g0; i < g1; i++) for (let j = 0; j < p.w; j++) mean[j] += p.data[i * p.w + j] / (g1 - g0); let best = g0, bd = Infinity; for (let i = g0; i < g1; i++) { let dd = 0; for (let j = 0; j < p.w; j++) dd += Math.pow(p.data[i * p.w + j] - mean[j], 2); if (dd < bd) { bd = dd; best = i; } } for (let i = g0; i < g1; i++) anchorOf[i] = best; }
            const keep = []; for (let i = 0; i < p.h; i++) if (anchorOf[i] !== i) keep.push(i);
            const d = new Float64Array(keep.length * p.w); const ids = p.tokenIds ? new Int32Array(keep.length) : undefined;
            keep.forEach((i, k) => { const a = anchorOf[i]; for (let j = 0; j < p.w; j++) { let v = p.data[i * p.w + j] - p.data[a * p.w + j]; if (wrap) v = wrapByte(v); d[k * p.w + j] = v; } if (ids) ids[k] = p.tokenIds[i]; });
            // per-channel centred variance, raw rows vs delta rows
            for (let j = 0; j < p.w; j++) { let s = 0, s2 = 0; for (let i = 0; i < p.h; i++) { const x = p.data[i * p.w + j]; s += x; s2 += x * x; } varRaw += s2 / p.h - Math.pow(s / p.h, 2); s = 0; s2 = 0; for (let k = 0; k < keep.length; k++) { const x = d[k * p.w + j]; s += x; s2 += x * x; } varDelta += s2 / keep.length - Math.pow(s / keep.length, 2); }
            nRaw += p.w; nDelta += p.w; anchors += (p.h - keep.length) * p.w; anchorBits += (p.h - keep.length) * p.w * state.bitDepth;
            out.push(Object.assign({}, p, { data: d, h: keep.length, tokenIds: ids, vkind: 'residual', M: undefined }));
          } else { // 1-D: every G-th sample is an anchor
            const d = []; for (let i = 0; i < p.data.length; i++) { if (i % G === 0) { anchors++; anchorBits += state.bitDepth; continue; } let v = p.data[i] - p.data[i - (i % G)]; if (wrap) v = wrapByte(v); d.push(v); varRaw += p.data[i] * p.data[i]; varDelta += v * v; }
            nRaw += d.length; nDelta += d.length; out.push(Object.assign({}, p, { data: Float64Array.from(d), w: d.length, vkind: 'residual', M: undefined }));
          }
        }
        state.planes = out;
      }
      state.extraBits += anchorBits; const after = streamEntropy(state);
      const ratio = (varRaw / (nRaw || 1)) / ((varDelta / (nDelta || 1)) || 1e-9);
      return emit(state, 'predict', f, `${state.domain === 'video' ? 'Frame 1 is the anchor kept raw' : state.nTokens ? `One anchor token per group of ${G} (the token nearest its group's mean; CacheGen takes the first) kept raw` : `Every ${G}th sample is an anchor kept raw`} (${fmtInt(anchors)} values, ${fmtInt(anchorBits)} bits at ${state.bitDepth} bits each); the other ${fmtInt(countSamples(state))} values are differences from their anchor, never from each other. ${state.nTokens ? 'Per-channel variance' : 'Variance'} of the differences is ${ratio >= 1 ? `${fmt(ratio, 1)}× lower` : `${fmt(1 / ratio, 1)}× higher`} than of the raw values (CacheGen measured 2.4–2.9× lower); order-0 entropy ${fmt(before.H, 2)} → ${fmt(after.H, 2)} bits/value. Every ${state.nTokens ? 'token' : 'value'} decodes alone on its own thread, and quantization error cannot accumulate down a chain.`,
        { symbols: after.n, alphabet: after.alphabet, entropyBitsPerSymbol: after.H }, residualViz(state));
    }
    if (f.id === 'motion') {
      const f0 = state.planes.filter((p) => p.frame === 0), f1 = state.planes.filter((p) => p.frame === 1);
      if (!f0.length || !f1.length || f1.length !== f0.length || !planeIsImage(f1[0])) return passThrough(state, 'predict', f, 'motion compensation needs two frames of the same size.');
      const ref = f0[0], cur = f1[0], W = cur.w, Hh = cur.h; const vecs = []; const at = (p, x, y) => p.data[clamp(y, 0, p.h - 1) * p.w + clamp(x, 0, p.w - 1)];
      for (let by = 0; by < Hh; by += 8) for (let bx = 0; bx < W; bx += 8) {
        let best = Infinity, bdx = 0, bdy = 0;
        for (let dy = -4; dy <= 4; dy++) for (let dx = -4; dx <= 4; dx++) {
          let sad = 0; for (let y = 0; y < 8; y++) for (let x = 0; x < 8; x++) sad += Math.abs(cur.data[(by + y) * W + bx + x] - at(ref, bx + x + dx, by + y + dy));
          if (sad < best || (sad === best && dx === 0 && dy === 0)) { best = sad; bdx = dx; bdy = dy; }
        }
        vecs.push({ bx, by, dx: bdx, dy: bdy });
      }
      const before = streamEntropy(state); let eCur = 0, eRes = 0;
      f1.forEach((p, k) => { // subsampled chroma planes reuse the luma vectors at their own scale, as real codecs do
        const r = f0[k]; if (!r || r.w !== p.w || r.h !== p.h) return; const sx = p.w / W, sy = p.h / Hh, bw = Math.max(1, Math.round(8 * sx)), bh = Math.max(1, Math.round(8 * sy));
        const out = new Float64Array(p.data.length);
        for (const v of vecs) { const bx = Math.round(v.bx * sx), by = Math.round(v.by * sy), dx = Math.round(v.dx * sx), dy = Math.round(v.dy * sy); for (let y = 0; y < bh; y++) for (let x = 0; x < bw; x++) { const i = (by + y) * p.w + bx + x; const pred = at(r, bx + x + dx, by + y + dy); eCur += p.data[i] * p.data[i]; out[i] = p.data[i] - pred; eRes += out[i] * out[i]; } }
        p.data = out; p.vkind = 'residual'; delete p.M;
      });
      state.extraBits += vecs.length * 8; const after = streamEntropy(state);
      const tally = new Map(); for (const v of vecs) { const k = `(${v.dx},${v.dy})`; tally.set(k, (tally.get(k) || 0) + 1); } const dom = [...tally.entries()].sort((a, b) => b[1] - a[1])[0];
      return emit(state, 'predict', f, `${vecs.length} blocks of 8×8 searched ±4 px in frame 1 (SAD, ${81 * vecs.length} block comparisons): vector ${dom[0]} won ${dom[1]} blocks. Frame 2 is now a residual with ${pct(eRes / (eCur || 1), 1)} of its energy; stream entropy ${fmt(before.H, 2)} → ${fmt(after.H, 2)} bits/sample. ${vecs.length * 8} bits of motion vectors ride along. Frame 1 stays intra (a keyframe).`,
        { symbols: after.n, alphabet: after.alphabet, entropyBitsPerSymbol: after.H }, vizGrid(f1));
    }
    return passThrough(state, 'predict', f, 'unknown predictor.');
  }
  function lzwPhrases(codes, maxBits) {
    maxBits = maxBits || 12; const maxCode = 1 << maxBits; const table = []; for (let i = 0; i < 256; i++) table.push([i]);
    const out = []; let prev = null;
    for (const code of codes) { const entry = code < table.length ? table[code] : prev.concat([prev[0]]); out.push(entry); if (prev && table.length < maxCode) table.push(prev.concat([entry[0]])); prev = entry; }
    return out;
  }

  // ============================================================ QUANTIZE stage (the only lossy stage)
  function currentDistortion(state) {
    if (state.lossless) return null;
    const se = state.seSum;
    if (state.domain === 'image' || state.domain === 'video') { const mse = se / state.nOrig; const v = mse > 0 ? 10 * Math.log10(255 * 255 / mse) : Infinity; return { metric: 'PSNR', value: v, label: Number.isFinite(v) ? `${fmt(v, 1)} dB PSNR` : 'PSNR ∞ (no error)' }; }
    if (state.domain === 'audio') { const v = se > 0 ? 10 * Math.log10(state.sigEnergy / se) : Infinity; return { metric: 'SNR', value: v, label: Number.isFinite(v) ? `${fmt(v, 1)} dB SNR` : 'SNR ∞ (no error)' }; }
    const v = Math.sqrt(se / (state.sigEnergy || 1));
    if (state.tokenWeights && state.nTokens) { // Top-k selection is in the pipeline: the softmax is the judge
      let num = 0, den = 0; for (let t = 0; t < state.nTokens; t++) { num += state.tokenWeights[t] * state.seTok[t]; den += state.tokenWeights[t] * state.energyTok[t]; }
      const wv = Math.sqrt(num / (den || 1)); return { metric: 'rel-err', value: wv, label: `${pct(wv, 2)} attention-weighted relative error (plain: ${pct(v, 2)})` };
    }
    return { metric: 'rel-err', value: v, label: `${pct(v, 2)} relative error (‖x−x̂‖/‖x‖)` };
  }
  function finishQuant(state, f, se, caption, viz) { // se: squared error not yet recorded through addErr
    state.hasQuantizer = true; state.lossless = false; if (se) addErr(state, null, 0, se);
    const d = currentDistortion(state); const e = streamEntropy(state);
    return emit(state, 'quantize', f, `${caption} Result so far: ${d.label}.`, { symbols: e.n, alphabet: e.alphabet, entropyBitsPerSymbol: e.H, distortion: d }, viz || vizHistogram(e.sym.symbols, e.M, e.sym.offset));
  }
  function groupVectors(state, d) {
    const all = allData(state.planes); const n = Math.ceil(all.length / d); const X = new Float64Array(n * d); X.set(all);
    return { X, n, d, padded: n * d - all.length, plane: state.planes.length === 1 ? state.planes[0] : null };
  }
  const JPEG_Q50 = [16, 11, 10, 16, 24, 40, 51, 61, 12, 12, 14, 19, 26, 58, 60, 55, 14, 13, 16, 24, 40, 57, 69, 56, 14, 17, 22, 29, 51, 87, 80, 62, 18, 22, 37, 56, 68, 109, 103, 77, 24, 35, 55, 64, 81, 104, 113, 92, 49, 64, 78, 87, 103, 121, 120, 101, 72, 92, 95, 98, 112, 100, 103, 99];
  function pickDelta(state) {
    if (state.bitDepth === 8) return 16;
    const mags = allData(state.planes); const lo = quantile(mags, 0.01), hi = quantile(mags, 0.99); const d = (hi - lo) / 12; return d > 0 ? d : 1;
  }
  const fmtDelta = (d) => (isInt(d) ? String(d) : d.toPrecision(3));
  const E2M1 = [0, 0.5, 1, 1.5, 2, 3, 4, 6];
  function toE4M3(s) { if (!(s > 0)) return 0; let e = clamp(Math.floor(log2(s)), -6, 8); let m = Math.round((s / Math.pow(2, e) - 1) * 8); if (m >= 8) { m = 0; e = Math.min(8, e + 1); } return Math.pow(2, e) * (1 + m / 8); }
  // GGUF Q4_K-style blocks: int4 over [min, max] per block, 6-bit scale + 6-bit min per block against an fp16 (d, dmin) pair per 8 blocks.
  function q4kBlocks(p, blocks) {
    const n = p.data.length; const q = new Float64Array(n), recon = new Float64Array(n);
    const sc = blocks.map((b) => { let lo = Infinity, hi = -Infinity; for (const i of b) { const v = p.data[i]; if (v < lo) lo = v; if (v > hi) hi = v; } return { lo, s: (hi - lo) / 15 }; });
    let se = 0, sMin = Infinity, sMax = 0, metaBits = 0;
    for (let g = 0; g < blocks.length; g += 8) {
      const grp = blocks.slice(g, g + 8); let dMax = 0, mMax = 0; grp.forEach((_, k) => { dMax = Math.max(dMax, sc[g + k].s); mMax = Math.max(mMax, Math.abs(sc[g + k].lo)); });
      const D = roundHalf(dMax / 63) || 1e-9, Dm = roundHalf(mMax / 31) || 1e-9; metaBits += 32;
      grp.forEach((b, k) => {
        const c = sc[g + k]; const sq = Math.max(1, Math.round(c.s / D)) * D, mq = Math.round(c.lo / Dm) * Dm; metaBits += 12; sMin = Math.min(sMin, sq); sMax = Math.max(sMax, sq);
        for (const i of b) { const qq = clamp(Math.round((p.data[i] - mq) / sq), 0, 15); q[i] = qq; recon[i] = mq + qq * sq; se += Math.pow(p.data[i] - recon[i], 2); }
      });
    }
    return { q, recon, se, sMin, sMax, metaBits };
  }
  function stageQuantize(state, f, anotherFollows) {
    if (state.rep !== 'planes') return passThrough(state, 'quantize', f, 'LZ tokens / phrase codes are already small integers; nothing to round.');
    if (f.id !== 'evict' && state.planes.length && state.planes.every((p) => p.vkind === 'index')) { const prev = state.stages.filter((s) => s.role === 'quantize').map((s) => CYC.byId[s.fighterId].name).pop(); return passThrough(state, 'quantize', f, `the values are already ${prev || 'quantizer'} indices; a second grid on top of indices only adds error (a real stack picks one quantizer per tensor).`); }
    const before = streamEntropy(state); const distinct0 = before.alphabet;
    if (f.id === 'uniform' || f.id === 'qmatrix') {
      const delta = pickDelta(state); let se = 0, zeros = 0, n = 0, usedMatrix = 0;
      for (const p of state.planes) {
        const useM = f.id === 'qmatrix' && p.blockSize === 8 && p.zigzag && planeIsImage(p); if (useM) usedMatrix++;
        for (let i = 0; i < p.data.length; i++) {
          const step = useM ? JPEG_Q50[((Math.floor(i / p.w) % 8) * 8) + (i % 8)] : delta;
          const q = Math.round(p.data[i] / step); const e2 = Math.pow(p.data[i] - q * step, 2); se += e2; addErr(state, p, i, e2); p.data[i] = q; if (q === 0) zeros++; n++;
        }
        p.vkind = 'index'; delete p.M;
      }
      if (f.id === 'qmatrix') { if (usedMatrix && !state.flags.tables) state.headerBits += 64 * 8; else if (!usedMatrix) state.headerBits += 32; } else state.headerBits += 32;
      const after = streamEntropy(state);
      const cap = f.id === 'qmatrix'
        ? (usedMatrix ? `JPEG luma quantization table (quality 50) on ${usedMatrix} plane(s) of 8×8 DCT blocks: Δ=16 for DC up to Δ=121 for the highest frequencies, coarse where the eye is blind. ${pct(zeros / n)} of coefficients are now exactly 0, ${after.alphabet} distinct values remain (was ${distinct0}), entropy ${fmt(before.H, 2)} → ${fmt(after.H, 2)} bits/coefficient. The table (64 B) ${state.flags.tables ? 'is a standard one the decoder holds' : 'travels in the header'}.`
          : `No 8×8 DCT blocks to weight, so the matrix degenerates to a uniform Δ=${fmtDelta(delta)}: ${after.alphabet} levels remain (was ${distinct0}), ${pct(zeros / n)} zeros, entropy ${fmt(before.H, 2)} → ${fmt(after.H, 2)} bits/sample. Quantization matrix needs DCT in front of it.`)
        : `q = round(x/Δ) with Δ=${fmtDelta(delta)}: ${after.alphabet} distinct levels remain (was ${distinct0}), ${pct(zeros / n)} of values are 0, entropy ${fmt(before.H, 2)} → ${fmt(after.H, 2)} bits/sample. Error RMS ${fmt(Math.sqrt(se / n), 3)} (theory Δ/√12 = ${fmt(delta / Math.sqrt(12), 3)}).`;
      return finishQuant(state, f, 0, cap);
    }
    if (f.id === 'companding') {
      let se = 0, n = 0; const mu = 255;
      for (const p of state.planes) {
        const scale = Math.max(...Array.from(p.data, Math.abs)) || 1; state.headerBits += 16;
        for (let i = 0; i < p.data.length; i++) {
          const x = p.data[i], y = Math.sign(x) * Math.log(1 + mu * Math.abs(x) / scale) / Math.log(1 + mu); const q = Math.round(y * 127);
          const rec = Math.sign(q) * scale * (Math.pow(1 + mu, Math.abs(q) / 127) - 1) / mu; const e2 = Math.pow(x - rec, 2); se += e2; addErr(state, p, i, e2); p.data[i] = q; n++;
        }
        p.vkind = 'index'; delete p.M;
      }
      const after = streamEntropy(state);
      return finishQuant(state, f, 0, `8-bit μ-law (μ=255) through a log curve: steps near zero are ~${fmt((Math.pow(1 + mu, 1 / 127) - 1) / mu * 100, 2)}% of full scale, steps near full scale ~${fmt((1 - Math.pow(1 + mu, -1 / 127)) * 100, 1)}%: constant relative error instead of constant absolute error. ${after.alphabet} of 255 levels used, entropy ${fmt(after.H, 2)} bits/sample. Error RMS ${fmt(Math.sqrt(se / n), 3)} on this ${state.bitDepth}-bit data.`);
    }
    if (f.id === 'vq' || f.id === 'rvq' || f.id === 'pq') {
      const K = 16; const d = (f.id === 'pq' || state.domain === 'vectors') ? 8 : 4; const g = groupVectors(state, d); const { X, n } = g;
      let se = 0, planes, codebookBytes, cap, viz; const recon = new Float64Array(X.length);
      if (f.id === 'vq') {
        const km = kmeans(X, n, d, K, 12); se = km.se; codebookBytes = K * d * 2;
        for (let i = 0; i < n; i++) for (let j = 0; j < d; j++) recon[i * d + j] = km.centroids[km.assign[i] * d + j];
        planes = [{ name: 'vq-index', w: n, h: 1, data: Float64Array.from(km.assign), vkind: 'index', M: K }];
        cap = `k-means, K=${K} on ${n} vectors of ${d} dims (${state.domain === 'vectors' ? 'whole vectors' : `${d}-sample groups`}, 12 iterations): each vector becomes a 4-bit index, ${fmt(4 / d, 2)} bits/sample before entropy coding. Codebook ${K}×${d}×2 B = ${codebookBytes} B`;
        viz = vizScatter(X, n, d, km.centroids, km.assign, K);
      } else if (f.id === 'rvq') {
        let R = Float64Array.from(X); planes = []; const cents = []; codebookBytes = 3 * K * d * 2; const stageErr = [];
        for (let s = 0; s < 3; s++) { const km = kmeans(R, n, d, K, 12); cents.push(km); stageErr.push(km.se); planes.push({ name: `rvq-${s + 1}`, w: n, h: 1, data: Float64Array.from(km.assign), vkind: 'index', M: K }); const nextR = new Float64Array(R.length); for (let i = 0; i < n; i++) for (let j = 0; j < d; j++) { const c = km.centroids[km.assign[i] * d + j]; nextR[i * d + j] = R[i * d + j] - c; recon[i * d + j] += c; } R = nextR; }
        se = stageErr[2]; let e0 = 0; for (let i = 0; i < X.length; i++) e0 += X[i] * X[i];
        cap = `Residual VQ, 3 stages × K=${K} on ${n} vectors of ${d} dims: stage 1 leaves ${pct(stageErr[0] / (e0 || 1), 1)} of the energy, stage 2 ${pct(stageErr[1] / (e0 || 1), 2)}, stage 3 ${pct(stageErr[2] / (e0 || 1), 2)}. 3 × 4 bits = 12 bits per vector (${fmt(12 / d, 2)} bits/sample), 48 centroids instead of the 4096 a single 12-bit VQ would need. Drop a stage and it still decodes, coarser. Codebooks ${codebookBytes} B`;
        viz = vizScatter(X, n, d, cents[0].centroids, cents[0].assign, K);
      } else {
        const nSub = 4, ds = d / nSub; planes = []; codebookBytes = K * d * 2; const idx = new Float64Array(n * nSub); let km0 = null; const sub0 = new Float64Array(n * ds);
        for (let s = 0; s < nSub; s++) {
          const S = new Float64Array(n * ds); for (let i = 0; i < n; i++) for (let j = 0; j < ds; j++) S[i * ds + j] = X[i * d + s * ds + j];
          const km = kmeans(S, n, ds, K, 12); se += km.se; for (let i = 0; i < n; i++) { idx[i * nSub + s] = km.assign[i]; for (let j = 0; j < ds; j++) recon[i * d + s * ds + j] = km.centroids[km.assign[i] * ds + j]; } if (s === 0) { km0 = km; sub0.set(S); }
        }
        planes.push({ name: 'pq-codes', w: nSub, h: n, data: idx, vkind: 'index', M: K });
        cap = `Product quantization: ${d} dims split into ${nSub} sub-vectors of ${ds}, each with its own K=${K} k-means codebook: ${nSub} × 4 bits = ${nSub * 4} bits per vector (${fmt(nSub * 4 / d, 1)} bits/sample), fixed length, ${Math.pow(K, nSub).toLocaleString('en-US')} effective centroids from ${K * nSub} real ones. Distances can be looked up per sub-vector without decoding. Codebooks ${codebookBytes} B`;
        viz = vizScatter(sub0, n, ds, km0.centroids, km0.assign, K);
      }
      if (state.flags.codebook) { state.sideInfoBytes += codebookBytes; state.flags.codebookBytes += codebookBytes; cap += ' held by the decoder (VQ codebook).'; if (state.pending.codebookStage) { state.pending.codebookStage.caption = `The decoder already holds the ${state.flags.codebookBytes} B of trained centroids that ${f.name}'s indices point into (K×d×2 bytes per codebook). Without it the indices are meaningless integers; with it they are the whole message.`; state.pending.codebookStage.metrics.sideInfoBytes = state.flags.codebookBytes; } }
      else { state.headerBits += codebookBytes * 8; cap += ' travel in the header (no VQ codebook in the pipeline).'; }
      if (g.padded) cap += ` (Last group padded with ${g.padded} zeros.)`;
      for (let i = 0; i < n; i++) { let e = 0; for (let j = 0; j < d; j++) e += Math.pow(X[i * d + j] - recon[i * d + j], 2); addErr(state, g.plane, i * d, e); } // per vector, so per token where rows are tokens
      state.planes = planes;
      return finishQuant(state, f, 0, cap, viz);
    }
    if (f.id === 'blockint') {
      let seTot = 0, nBlocks = 0, sMin = Infinity, sMax = 0, n = 0, metaBits = 0; const notes = [];
      for (const p of state.planes) {
        const raster = []; for (let b = 0; b < p.data.length; b += 32) { const idx = []; for (let i = b; i < Math.min(p.data.length, b + 32); i++) idx.push(i); raster.push(idx); }
        let best = q4kBlocks(p, raster), mode = p.tokenIds ? 'per-token' : 'raster';
        if (p.tokenIds && p.h > 1 && p.w > 1) { // keys: try per-channel groups too (KIVI) and keep the better layout at the same bit cost
          const cols = []; for (let j = 0; j < p.w; j++) for (let r0 = 0; r0 < p.h; r0 += 32) { const idx = []; for (let r = r0; r < Math.min(p.h, r0 + 32); r++) idx.push(r * p.w + j); cols.push(idx); }
          const alt = q4kBlocks(p, cols);
          notes.push(`per-token blocks give error RMS ${Math.sqrt(best.se / p.data.length).toPrecision(3)}, per-channel blocks ${Math.sqrt(alt.se / p.data.length).toPrecision(3)}`);
          if (alt.se < best.se) { best = alt; mode = 'per-channel'; nBlocks += cols.length; } else nBlocks += raster.length;
          notes.push(`${mode} wins and is used (KIVI: per-channel keys, per-token values; QuaRot rotates first and goes per token)`);
        } else nBlocks += raster.length;
        for (let i = 0; i < p.data.length; i++) addErr(state, p, i, Math.pow(p.data[i] - best.recon[i], 2));
        seTot += best.se; n += p.data.length; metaBits += best.metaBits; sMin = Math.min(sMin, best.sMin); sMax = Math.max(sMax, best.sMax);
        p.data = best.q; p.vkind = 'index'; p.M = 16;
      }
      state.extraBits += metaBits; const after = streamEntropy(state);
      return finishQuant(state, f, 0, `${nBlocks} blocks of ≤32 → int4 (0..15) over each block's [min, max], with a 6-bit scale and a 6-bit min per block against one fp16 pair per 256 values (GGUF Q4_K layout): 4 + ${fmt(metaBits / n, 2)} = ${fmt(4 + metaBits / n, 2)} bits/value, dequantize = one multiply-add, any block readable alone. Steps span ${sMax.toPrecision(3)} / ${sMin.toPrecision(3)} = ${fmt(sMax / sMin, 1)}× (a block with an outlier gets a coarse grid for all its neighbours).${notes.length ? ' ' + notes.join('; ') + '.' : ''} Error RMS ${Math.sqrt(seTot / n).toPrecision(3)}; the ${after.alphabet} int4 values carry ${fmt(after.H, 2)} bits of entropy but are stored at 4 flat.`);
    }
    if (f.id === 'fp4') {
      let se = 0, n = 0, nBlocks = 0, sMin = Infinity, sMax = 0;
      for (const p of state.planes) {
        for (let b = 0; b < p.data.length; b += 16) {
          const end = Math.min(p.data.length, b + 16); let m = 0; for (let i = b; i < end; i++) m = Math.max(m, Math.abs(p.data[i]));
          const s = toE4M3(m / 6) || Math.pow(2, -6); nBlocks++; sMin = Math.min(sMin, s); sMax = Math.max(sMax, s);
          for (let i = b; i < end; i++) { const x = p.data[i], a = Math.abs(x) / s; let k = 0, bd = Infinity; for (let t = 0; t < 8; t++) { const dd = Math.abs(a - E2M1[t]); if (dd < bd) { bd = dd; k = t; } } const rec = Math.sign(x) * E2M1[k] * s; const e2 = Math.pow(x - rec, 2); se += e2; addErr(state, p, i, e2); p.data[i] = x < 0 && k > 0 ? 8 + k : k; n++; }
        }
        p.vkind = 'index'; p.M = 16;
      }
      state.extraBits += nBlocks * 8; const after = streamEntropy(state);
      return finishQuant(state, f, 0, `${nBlocks} blocks of 16 → E2M1 floats (15 values: ±{0, 0.5, 1, 1.5, 2, 3, 4, 6}) with one FP8 (E4M3) scale per block: 4 + 8/16 = 4.5 bits/value, the same budget as int4-per-32, but a floating grid that is fine near zero and coarse at the top, and a block half the size to localize outliers. Scales span ${sMax.toPrecision(3)} / ${sMin.toPrecision(3)}. Error RMS ${Math.sqrt(se / n).toPrecision(3)}; ${after.alphabet} codes used, ${fmt(after.H, 2)} bits of entropy, stored at 4 flat. Blackwell tensor cores read this format natively.`);
    }
    if (f.id === 'escape') {
      const all = []; state.planes.forEach((p, pi) => { for (let i = 0; i < p.data.length; i++) all.push({ pi, i, a: Math.abs(p.data[i]) }); });
      const n = all.length, k = Math.max(Math.ceil(0.01 * n), Math.floor(n / 64)); all.sort((u, v) => v.a - u.a); const out = all.slice(0, k); const thr = out.length ? out[out.length - 1].a : Infinity;
      const outl = state.planes.map(() => new Set()); for (const o of out) outl[o.pi].add(o.i);
      const idxBits = ceilLog2(n); state.extraBits += k * (16 + idxBits);
      if (anotherFollows) { // dense-and-sparse: the hatch only pulls the outliers out; the next quantizer gets the dense remainder
        let before0 = 0, after0 = 0; state.planes.forEach((p, pi) => { for (let b = 0; b < p.data.length; b += 32) { let m0 = 0, m = 0; for (let i = b; i < Math.min(p.data.length, b + 32); i++) { const a = Math.abs(p.data[i]); m0 = Math.max(m0, a); if (!outl[pi].has(i)) m = Math.max(m, a); } before0 = Math.max(before0, m0); after0 = Math.max(after0, m); } for (const i of outl[pi]) p.data[i] = 0; });
        state.hasQuantizer = true; state.lossless = false;
        return emit(state, 'quantize', f, `The ${k} largest |x| (≥ ${thr.toPrecision(3)}, ${pct(k / n, 1)} of ${n}) leave the grid and ride in a sparse fp16 list: 16 + ${idxBits} index bits each = ${k * (16 + idxBits)} bits, exact. They are zeroed in the dense tensor, whose largest value drops from ${before0.toPrecision(3)} to ${after0.toPrecision(3)} (${fmt(before0 / (after0 || 1e-9), 1)}×), so the quantizer that follows can use a grid that much finer. KVQuant's dense-and-sparse split.`, { symbols: n - k }, vizHeatmapOf(state.planes));
      }
      let se = 0, sMaxB = 0, sMaxBefore = 0, nBlocks = 0;
      state.planes.forEach((p, pi) => {
        for (let b = 0; b < p.data.length; b += 32) {
          const end = Math.min(p.data.length, b + 32); let m = 0, m0 = 0; for (let i = b; i < end; i++) { const a = Math.abs(p.data[i]); m0 = Math.max(m0, a); if (!outl[pi].has(i)) m = Math.max(m, a); }
          const s = roundHalf(m / 3) || 1e-9; nBlocks++; sMaxB = Math.max(sMaxB, s); sMaxBefore = Math.max(sMaxBefore, roundHalf(m0 / 3));
          for (let i = b; i < end; i++) { if (outl[pi].has(i)) { p.data[i] = 4; continue; } const q = clamp(Math.round(p.data[i] / s), -4, 3); const e2 = Math.pow(p.data[i] - q * s, 2); se += e2; addErr(state, p, i, e2); p.data[i] = q + 4; }
        }
        p.vkind = 'index'; p.M = 8;
      });
      state.extraBits += nBlocks * 16; const after = streamEntropy(state);
      return finishQuant(state, f, 0, `The ${k} largest |x| (≥ ${thr.toPrecision(3)}, ${pct(k / n, 1)} of ${n}) leave the grid and ride in a sparse fp16 list: 16 + ${idxBits} index bits each = ${k * (16 + idxBits)} bits, exact. The other ${n - k} values fit an int3 grid (−4..3) with an absmax scale per 32: the coarsest step dropped from ${sMaxBefore.toPrecision(3)} to ${sMaxB.toPrecision(3)} (${fmt(sMaxBefore / sMaxB, 1)}× finer). Total ≈ 3 + ${fmt(k * (16 + idxBits) / n, 2)} (list) + 0.5 (scales) = ${fmt(3.5 + k * (16 + idxBits) / n, 2)} bits/value. Error RMS ${Math.sqrt(se / n).toPrecision(3)}; ${after.alphabet} int3 codes, ${fmt(after.H, 2)} bits of entropy. A sparse gather inside attention is the price.`);
    }
    if (f.id === 'coupled') {
      const K = 16, d = 4; const tp = tokenPlaneOf(state) || (state.planes.length === 1 && state.planes[0].h >= 16 && state.planes[0].w % d === 0 ? state.planes[0] : null);
      let codebookBytes, cap, viz, planes;
      if (tp && tp.w % d === 0) { // one codebook per group of 4 channels, fitted over the tokens
        const n = tp.h, w = tp.w, G = w / d; const idx = new Float64Array(n * G); codebookBytes = G * K * d * 2; let se = 0, km0 = null, S0 = null;
        for (let g = 0; g < G; g++) {
          const S = new Float64Array(n * d); for (let i = 0; i < n; i++) for (let j = 0; j < d; j++) S[i * d + j] = tp.data[i * w + g * d + j];
          const km = kmeans(S, n, d, K, 12); if (g === 0) { km0 = km; S0 = S; }
          for (let i = 0; i < n; i++) { idx[i * G + g] = km.assign[i]; let e = 0; for (let j = 0; j < d; j++) e += Math.pow(S[i * d + j] - km.centroids[km.assign[i] * d + j], 2); se += e; addErr(state, tp, i * w + g * d, e); }
        }
        planes = state.planes.map((p) => (p === tp ? { name: 'cq-codes', w: G, h: n, data: idx, vkind: 'index', M: K, tokenIds: tp.tokenIds } : p));
        cap = `${G} groups of ${d} channels, each with its own ${K}-entry codebook (k-means over the ${n} tokens; CQ calibrates offline with Fisher-weighted centroids): one 4-bit code per ${d} channels = ${fmt(4 / d, 2)} bit/channel, fixed length, decoded by one table lookup inside attention. Channels in a group are quantized jointly, so what they share is not paid for twice${tp.rope ? ' (these keys still carry RoPE: the outlier pair walks a circle the codebook has to cover)' : ''}. Error RMS ${Math.sqrt(se / (n * w)).toPrecision(3)}. Codebooks ${G}×${K}×${d}×2 B = ${codebookBytes} B`;
        viz = vizScatter(S0, n, d, km0.centroids, km0.assign, K);
      } else { // generic: one shared codebook over groups of 4 consecutive values
        const g = groupVectors(state, d); const km = kmeans(g.X, g.n, d, K, 12); codebookBytes = K * d * 2;
        for (let i = 0; i < g.n; i++) { let e = 0; for (let j = 0; j < d; j++) e += Math.pow(g.X[i * d + j] - km.centroids[km.assign[i] * d + j], 2); addErr(state, g.plane, i * d, e); }
        planes = [{ name: 'cq-codes', w: g.n, h: 1, data: Float64Array.from(km.assign), vkind: 'index', M: K }];
        cap = `No channel groups in this payload, so one shared ${K}-entry codebook over groups of ${d} consecutive values: ${fmt(4 / d, 2)} bit/value, error RMS ${Math.sqrt(km.se / (g.n * d)).toPrecision(3)}. Codebook ${codebookBytes} B`;
        viz = vizScatter(g.X, g.n, d, km.centroids, km.assign, K);
      }
      if (state.flags.codebook) { state.sideInfoBytes += codebookBytes; state.flags.codebookBytes += codebookBytes; cap += ' held by the decoder (VQ codebook).'; if (state.pending.codebookStage) { state.pending.codebookStage.caption = `The decoder already holds the ${state.flags.codebookBytes} B of trained centroids that ${f.name}'s codes point into (K×d×2 bytes per codebook; about 1% of the weights for a real model). Without it the codes are meaningless integers.`; state.pending.codebookStage.metrics.sideInfoBytes = state.flags.codebookBytes; } }
      else { state.headerBits += codebookBytes * 8; cap += ' travel in the header (no VQ codebook in the pipeline).'; }
      state.planes = planes;
      return finishQuant(state, f, 0, cap, viz);
    }
    if (f.id === 'waterline') {
      const views = [];
      for (const p of state.planes) {
        if (p.vkind !== 'coef') continue;
        if (p.zigzag && p.blockSize === 8 && planeIsImage(p)) { const bw = p.w / 8, bh = p.h / 8; views.push({ p, rows: bw * bh, cols: 64, at: (i, j) => { const bx = (i % bw) * 8, by = Math.floor(i / bw) * 8, z = ZIGZAG8[j]; return (by + (z >> 3)) * p.w + bx + (z & 7); } }); }
        else if (p.h === 1 && p.blockSize) { const N = p.blockSize; views.push({ p, rows: Math.floor(p.w / N), cols: N, at: (i, j) => i * N + j }); }
        else views.push({ p, rows: p.h, cols: p.w, at: (i, j) => i * p.w + j });
      }
      if (!views.length) {
        warn(state, 'Water-filling allocation on raw channels: water on flat ground. Every coordinate has about the same variance, so there is nothing to allocate; degraded to a uniform Δ. Put PCA / Hadamard rotation, Low-rank SVD or DCT in front of it.');
        const delta = pickDelta(state); let se = 0, n = 0; for (const p of state.planes) { for (let i = 0; i < p.data.length; i++) { const q = Math.round(p.data[i] / delta); const e2 = Math.pow(p.data[i] - q * delta, 2); se += e2; addErr(state, p, i, e2); p.data[i] = q; n++; } p.vkind = 'index'; delete p.M; } state.headerBits += 32;
        const after = streamEntropy(state);
        return finishQuant(state, f, 0, `Water on flat ground: no transform coefficients to allocate across, so this collapses to a uniform quantizer with Δ=${fmtDelta(delta)} (${after.alphabet} levels, ${fmt(after.H, 2)} bits/sample of entropy, error RMS ${Math.sqrt(se / (n || 1)).toPrecision(3)}). Reverse water-filling needs coordinates with different variances.`);
      }
      const budget = 4; let fixedBits = 0, totalBits = 0, totalEl = 0, dropped = 0, kept = 0, scaleBits = 0, theta = 0, varMax = 0, varMin = Infinity; const allocs = [], outPlanes = [];
      for (const v of views) {
        const { p, rows, cols, at } = v; const mean = new Float64Array(cols), varr = new Float64Array(cols), lo = new Float64Array(cols).fill(Infinity), hi = new Float64Array(cols).fill(-Infinity);
        for (let j = 0; j < cols; j++) { for (let i = 0; i < rows; i++) { const x = p.data[at(i, j)]; mean[j] += x / rows; if (x < lo[j]) lo[j] = x; if (x > hi[j]) hi[j] = x; } for (let i = 0; i < rows; i++) varr[j] += Math.pow(p.data[at(i, j)] - mean[j], 2) / rows; }
        const bitsFor = (th) => Array.from(varr, (s2) => clamp(Math.round(0.5 * log2(Math.max(s2, 1e-18) / th)), 0, 8));
        let lo2 = Math.log(Math.max(1e-18, Math.min(...varr))) - 20, hi2 = Math.log(Math.max(1e-18, ...varr)) + 1; let R = bitsFor(Math.exp(hi2));
        for (let it = 0; it < 60; it++) { const mid = (lo2 + hi2) / 2; const r = bitsFor(Math.exp(mid)); if (r.reduce((a, b) => a + b, 0) <= budget * cols) { hi2 = mid; R = r; } else lo2 = mid; }
        theta = Math.exp(hi2); varMax = Math.max(varMax, ...varr); varMin = Math.min(varMin, ...varr);
        const keepCols = []; R.forEach((r, j) => { if (r > 0) keepCols.push(j); else { for (let i = 0; i < rows; i++) addErr(state, p, at(i, j), Math.pow(p.data[at(i, j)], 2)); dropped++; } });
        const q = new Float64Array(rows * keepCols.length); let maxLevels = 2;
        keepCols.forEach((j, k) => { const levels = 1 << R[j]; maxLevels = Math.max(maxLevels, levels); const step = (hi[j] - lo[j]) / (levels - 1) || 1; scaleBits += 32; for (let i = 0; i < rows; i++) { const x = p.data[at(i, j)]; const qq = clamp(Math.round((x - lo[j]) / step), 0, levels - 1); q[i * keepCols.length + k] = qq; addErr(state, p, at(i, j), Math.pow(x - (lo[j] + qq * step), 2)); } fixedBits += rows * R[j]; });
        kept += keepCols.length; totalBits += R.reduce((a, b) => a + b, 0) * rows; totalEl += rows * cols; allocs.push(R);
        outPlanes.push({ name: p.name + ' (allocated)', w: keepCols.length, h: rows, data: q, vkind: 'index', M: maxLevels, tokenIds: p.tokenIds && rows === p.h ? p.tokenIds : undefined });
      }
      state.planes = state.planes.map((p) => { const k = views.findIndex((v) => v.p === p); return k >= 0 ? outPlanes[k] : p; });
      state.extraBits += scaleBits;
      state.fixedBitsExact = fixedBits + state.planes.filter((p) => !outPlanes.includes(p)).reduce((a, p) => { const s = stats(p.data); return a + p.data.length * (s.allInt ? ceilLog2(s.max - s.min + 1) : state.bitDepth); }, 0);
      const R0 = allocs[0]; const shown = R0.length <= 40 ? R0.join(',') : R0.slice(0, 40).join(',') + ',…';
      return finishQuant(state, f, 0, `Reverse water-filling to a budget of ${budget} bits/element: water level θ=${theta.toPrecision(3)}, each coordinate gets R = round(½·log₂(σ²/θ)) bits, clipped to 0..8. Variances span ${varMax.toPrecision(3)} / ${varMin.toPrecision(3)} = ${fmt(varMax / Math.max(varMin, 1e-18), 0)}×; allocation [${shown}]: ${kept} coordinates coded at ${fmt(totalBits / totalEl, 2)} bits/element on average, ${dropped} under water and dropped (reconstructed as 0). Each coded coordinate gets its own uniform grid over its range (32 bits of scale each, ${scaleBits} bits). KVTC does this with dynamic programming over PCA coordinates.`,
        { type: 'histogram', bins: Array.from(R0.slice(0, 64)), labels: R0.slice(0, 64).map((_, j) => String(j)), entropyBitsPerSymbol: totalBits / (totalEl || 1), fixedBits: 8 });
    }
    if (f.id === 'evict') {
      const tp = tokenPlaneOf(state); if (!tp) return passThrough(state, 'quantize', f, 'eviction drops tokens; this payload has none.');
      const n = tp.h; const score = new Float64Array(n); const useSpot = !!state.tokenWeights;
      for (let i = 0; i < n; i++) { if (useSpot) score[i] = state.tokenWeights[tp.tokenIds[i]]; else { let s = 0; for (let j = 0; j < tp.w; j++) s += tp.data[i * tp.w + j] * tp.data[i * tp.w + j]; score[i] = Math.sqrt(s); } }
      const order = Array.from({ length: n }, (_, i) => i).sort((a, b) => score[b] - score[a]); const keepN = Math.ceil(n / 2); const keepRows = new Set(order.slice(0, keepN));
      const evicted = new Set(); let massKept = 0, massAll = 0;
      for (let i = 0; i < n; i++) { const wgt = useSpot ? score[i] : state.energyTok[tp.tokenIds[i]]; massAll += wgt; if (keepRows.has(i)) massKept += wgt; else { evicted.add(tp.tokenIds[i]); for (let j = 0; j < tp.w; j++) addErr(state, tp, i * tp.w + j, Math.pow(tp.data[i * tp.w + j], 2)); } }
      for (const p of state.planes) { if (!p.tokenIds || p.h !== p.tokenIds.length) continue; const rows = []; for (let i = 0; i < p.h; i++) if (!evicted.has(p.tokenIds[i])) rows.push(i); const out = new Float64Array(rows.length * p.w), ids = new Int32Array(rows.length); rows.forEach((i, k) => { out.set(p.data.subarray(i * p.w, (i + 1) * p.w), k * p.w); ids[k] = p.tokenIds[i]; }); p.data = out; p.h = rows.length; p.tokenIds = ids; }
      state.evicted = evicted;
      const PAGE = 16, nPages = Math.ceil(state.nTokens / PAGE); let freed = 0, occupied = 0;
      for (let pg = 0; pg < nPages; pg++) { let alive = 0; for (let t = pg * PAGE; t < Math.min(state.nTokens, (pg + 1) * PAGE); t++) if (!evicted.has(t)) alive++; if (alive === 0) freed++; else occupied++; }
      const survivors = state.nTokens - evicted.size;
      return finishQuant(state, f, 0, `Kept ${keepN} of ${n} tokens by ${useSpot ? "Top-k selection's attention score" : 'L2 norm (no Top-k selection: the norm is a blind proxy)'}; evicted tokens ${[...evicted].sort((a, b) => a - b).join(', ')}, which reconstruct as zero. The survivors carry ${pct(massKept / (massAll || 1))} of the ${useSpot ? 'attention mass' : 'energy'}. Memory: with ${PAGE}-token pages, ${freed} of ${nPages} pages are completely empty and freed; survivors are scattered over ${occupied} page${occupied === 1 ? '' : 's'}, so VRAM is ${freed ? 'only partly' : 'not'} returned until compaction copies the ${survivors} survivors into ${Math.ceil(survivors / PAGE)}. Forgotten forever: there is no lossless copy to call${state.flags.tiered ? ' (Tiered cache has one, but eviction never asks)' : ''}.`,
        vizWaveform([score], [useSpot ? 'attention score per token' : 'L2 norm per token']));
    }
    return passThrough(state, 'quantize', f, 'unknown quantizer.');
  }

  // ============================================================ ENTROPY stage (the only stage that produces bits)
  function codeStream(state, f, sym, what) {
    const n = sym.symbols.length; const e = entropyOf(sym.symbols); const H = e.H, bound = H * n;
    const ctx = state.flags.contextModel; const ctxName = ctx ? CYC.byId[ctx].name : null;
    let bits, caption;
    if (f.id === 'huffman') {
      const hb = huffmanBits(sym.symbols); const tableBits = state.flags.tables ? 0 : e.alphabet * 4; bits = hb.bits + tableBits;
      caption = `Real Huffman tree from the histogram of ${fmtInt(n)} ${what} (alphabet ${e.alphabet}): average code ${fmt(hb.bits / (n || 1), 2)} bits/symbol against an entropy of ${fmt(H, 2)}; the most common symbol (p=${fmt(hb.maxP, 2)}) gets ${hb.maxLen} bit(s). Tree header ≈ ${e.alphabet}×4 = ${tableBits} bits${state.flags.tables ? ' waived: standard tables (Standard tables)' : ''}. Total ${fmtInt(bits)} bits.`;
      if (n && hb.bits > 1.3 * bound) warn(state, `Huffman on a symbol with p=${fmt(hb.maxP, 2)}: paying ${hb.maxLen} bit${hb.maxLen > 1 ? 's' : ''} for ${fmt(-log2(hb.maxP), 2)} bits of information. Integer code lengths cost ${pct(hb.bits / Math.max(bound, 1e-9) - 1)} over the bound here; run-length coding first or an arithmetic coder fixes it.`);
      if (ctx) { caption += ` ${ctxName}'s context probabilities are ignored: a static prefix code cannot use a model that changes every symbol.`; warn(state, `${ctxName} computed P(next | context) but Huffman coding cannot spend fractional or adaptive bits: put Arithmetic coding or ANS in the entropy-coding stage.`); }
    } else if (f.id === 'arith' || f.id === 'ans') {
      const flush = 32;
      if (ctx) { bits = ppmOrder2Bits(sym.symbols, sym.M) + flush; caption = `Driven by ${ctxName}'s order-2 adaptive context model (stand-in for the real thing): Σ −log₂ p over ${fmtInt(n)} ${what} = ${fmtInt(bits - flush)} bits, ${fmt((bits - flush) / (n || 1), 2)} bits/symbol, ${bits - flush < bound ? `${pct(1 - (bits - flush) / (bound || 1))} below` : `${pct((bits - flush) / (bound || 1) - 1)} above`} the order-0 bound of ${fmt(H, 2)}: context sees structure a histogram cannot. Plus ${flush} bits to flush.`; }
      else { bits = adaptiveOrder0Bits(sym.symbols, sym.M) + flush; caption = `Adaptive order-0 model (Krichevsky–Trofimov counts over an alphabet of ${fmtInt(sym.M)}): each symbol costs −log₂ of its running probability, fractional bits allowed. ${fmtInt(n)} ${what} → ${fmtInt(bits - flush)} bits = ${fmt((bits - flush) / (n || 1), 2)} bits/symbol, ${pct((bits - flush) / Math.max(bound, 1e-9) - 1, 1)} above the order-0 bound (the cost of learning the histogram on the fly) + ${flush} bits flush.`; }
      caption += f.id === 'ans' ? ' rANS/tANS reach the same count with a table-driven state machine, decoding last-in-first-out.' : ' Each step is a multiply and a renormalization, strictly serial.';
    } else if (f.id === 'rice') {
      const r = riceBits(sym.signed); bits = r.bits;
      const ks = [...new Set(r.ks)].sort((a, b) => a - b);
      caption = `Golomb-Rice, best k per 64-symbol partition (${r.partitions} partitions, k ∈ {${ks.slice(0, 6).join(', ')}${ks.length > 6 ? ', …' : ''}}): ${r.zigzag ? 'zigzag-map the signed residual, then ' : ''}write v>>k in unary and k low bits raw. ${fmtInt(n)} ${what} → ${fmtInt(bits)} bits = ${fmt(bits / (n || 1), 2)} bits/symbol against an entropy of ${fmt(H, 2)}. No table to send, no tree to build.`;
      if (r.maxQ > 31) caption += ` One value needed a ${r.maxQ}-bit unary prefix: FLAC uses escape codes for those.`;
      if (bits > 1.5 * bound && n) warn(state, `Rice spends ${pct(bits / Math.max(bound, 1e-9) - 1)} more than the entropy bound: this stream is not geometric-shaped (Rice wants small residuals peaking at zero).`);
    } else { // none: fixed length
      const b = ceilLog2(sym.M); bits = b * n;
      if (state.rep === 'lzw') { bits = state.lzw.varBits; caption = `No entropy coder: LZW codes are written at their natural 9–12-bit growing width, ${fmtInt(bits)} bits for ${fmtInt(n)} codes (entropy ${fmt(H, 2)} bits/code).`; }
      else if (state.fixedBitsExact != null && state.rep === 'planes') { bits = state.fixedBitsExact; caption = `Pass-through = fixed-length codes, but Water-filling allocation already set a width per coordinate: ${fmtInt(n)} ${what} at ${fmt(bits / (n || 1), 2)} bits each on average = ${fmtInt(bits)} bits (entropy ${fmt(H, 2)} bits/symbol). Each coordinate is still a flat field a kernel can index.`; }
      else caption = `Pass-through = fixed-length codes: every one of ${fmtInt(n)} ${what} costs ⌈log₂ ${fmtInt(sym.M)}⌉ = ${b} bits, ${fmtInt(bits)} bits total. Shannon says ${fmt(H, 2)} bits/symbol would do${H < 0.9 * b ? `: ${pct(1 - H / b)} is left on the table, in exchange for random access and a decoder that is a single array index` : ': the symbols are near-uniform, so there was almost nothing to gain'}.`;
    }
    return { bits, bound, H, n, alphabet: e.alphabet, caption };
  }
  function stageEntropy(state, ids) {
    const sym0 = symbolize(state, true); let sym = sym0; const streamExtra = sym0.extraBits;
    const coders = ids.filter((id) => id !== 'rle'); const hasRle = ids.includes('rle');
    let coderBits = 0, bound = 0, rleExtra = 0, last = null; const what0 = sym0.desc;
    if (hasRle) {
      const f = CYC.byId.rle; const blockLen = state.rep === 'planes' && state.planes.some((p) => p.zigzag && p.blockSize === 8) ? 64 : 0;
      const r = rleZeroRuns(sym.signed, blockLen); const e0 = entropyOf(sym.symbols); const e1 = entropyOf(r.symbols);
      sym = { symbols: r.symbols, M: RLE_M, extraBits: 0, signed: r.symbols, offset: 0, desc: 'run/size pairs' }; rleExtra = r.extraBits;
      const zf = r.zeros / (sym0.symbols.length || 1);
      let cap = `${blockLen ? 'Zigzag scan each 8×8 block (low to high frequency), then ' : ''}JPEG-style run/size pairs: "${r.zeros ? 'r zeros then a value of s bits' : 'no zeros to count'}" as one symbol (${r.pairs} pairs, ${r.zrl} ZRL, ${r.eob} end-of-block) plus the ${fmtInt(r.extraBits)} raw amplitude bits. ${pct(zf)} of the ${fmtInt(sym0.symbols.length)} input symbols were zeros${zf > 0.5 ? ` (p=${fmt(zf, 2)} > 0.5: exactly the case Huffman handles badly on its own)` : ''}; ${fmtInt(sym0.symbols.length)} symbols with ${fmt(e0.H, 2)} bits/symbol became ${fmtInt(r.symbols.length)} pair symbols with ${fmt(e1.H, 2)} bits/symbol.`;
      if (!coders.length) { const b = ceilLog2(RLE_M); coderBits = b * r.symbols.length; bound = e1.H * r.symbols.length; cap += ` Alone, the pairs are written at a flat ${b} bits each: ${fmtInt(coderBits)} bits. RLE is a pre-pass; it wants a Huffman behind it.`; }
      last = emit(state, 'entropy', f, cap, { symbols: r.symbols.length, alphabet: e1.alphabet, entropyBitsPerSymbol: e1.H }, vizHistogram(r.symbols, RLE_M, 0));
      if (zf < 0.15 && sym0.symbols.length) warn(state, `Run-length coding found only ${pct(zf)} zeros: run-length coding expands a stream that does not run.`);
    }
    if (coders.length === 1 || (coders.length === 2 && !(state.rep === 'tokens' && !hasRle))) {
      const f = CYC.byId[coders[0]]; const r = codeStream(state, f, sym, hasRle ? 'run/size pairs' : what0); coderBits = r.bits; bound = r.bound;
      last = emit(state, 'entropy', f, r.caption, { symbols: r.n, alphabet: r.alphabet, entropyBitsPerSymbol: r.H, bits: Math.round(r.bits) }, vizHistogram(sym.symbols, sym.M, sym.offset));
      if (coders.length === 2) { const g = CYC.byId[coders[1]]; last = emit(state, 'entropy', g, `${g.name} has nothing left to code (0 bits): ${f.name} already turned all ${fmtInt(r.n)} symbols into bits. One stream, one coder (zstd splits literals from match tokens; this stream has no such split).`, { bits: 0 }, vizHistogram(sym.symbols, sym.M, sym.offset)); }
    } else if (coders.length === 2) { // zstd style: first coder takes match lengths/distances, second takes literals
      const lz = sym0.lz; const fa = CYC.byId[coders[0]], fb = CYC.byId[coders[1]];
      const ra = codeStream(state, fa, { symbols: lz.matches, M: LZ_M, signed: lz.matches }, 'match length/distance symbols');
      const rb = codeStream(state, fb, { symbols: lz.literals, M: 256, signed: lz.literals }, 'literal bytes');
      coderBits = ra.bits + rb.bits; bound = ra.bound + rb.bound;
      emit(state, 'entropy', fa, `Match stream. ${ra.caption}`, { symbols: ra.n, alphabet: ra.alphabet, entropyBitsPerSymbol: ra.H, bits: Math.round(ra.bits) }, vizHistogram(lz.matches, LZ_M, 0));
      last = emit(state, 'entropy', fb, `Literal stream (the split zstd makes: FSE for lengths/offsets, Huffman for literals). ${rb.caption}`, { symbols: rb.n, alphabet: rb.alphabet, entropyBitsPerSymbol: rb.H, bits: Math.round(rb.bits) }, vizHistogram(lz.literals, 256, 0));
    } else if (!hasRle) {
      const r = codeStream(state, NONE(), sym, what0); coderBits = r.bits; bound = r.bound;
      last = emit(state, 'entropy', NONE(), r.caption, { symbols: r.n, alphabet: r.alphabet, entropyBitsPerSymbol: r.H, bits: Math.round(r.bits) }, vizHistogram(sym.symbols, sym.M, sym.offset));
    }
    const sideBits = state.extraBits + streamExtra + rleExtra; const finalBits = coderBits + sideBits + state.headerBits; const boundBits = bound + sideBits;
    if (sideBits || state.headerBits) last.caption += ` Plus ${fmtInt(sideBits)} raw side bits${state.headerBits ? ` and ${fmtInt(state.headerBits)} header bits` : ''} → ${fmtInt(finalBits)} bits in the stream.`;
    last.metrics.bits = Math.round(finalBits); last.metrics.sideInfoBytes = Math.round(state.sideInfoBytes);
    last.viz = { type: 'bits', raw: state.rawBits, out: Math.round(finalBits), bound: Math.round(boundBits), sideInfoBytes: Math.round(state.sideInfoBytes) };
    if (state.flags.contextModel && !coders.some((c) => c === 'arith' || c === 'ans')) warn(state, `${CYC.byId[state.flags.contextModel].name} is modelling but nobody is spending fractional bits: context models need an arithmetic-style coder.`);
    return { finalBits, boundBits };
  }

  // ============================================================ run
  const ORDER = ['prior', 'transform', 'perceptual', 'predict', 'transform2', 'quantize', 'entropy'];
  function run(payloadKind, squad) {
    const payload = getPayload(payloadKind); const state = makeState(payload); squad = squad || {};
    const chair = (role) => (Array.isArray(squad[role]) ? squad[role] : []).filter((id) => id && id !== 'none').map((id) => CYC.byId[id]).filter(Boolean);
    const dispatch = (role, f, fn) => { if (f.role !== role) return passThrough(state, role, f, `${f.name} is a ${CYC.ROLES[f.role] ? CYC.ROLES[f.role].title.toLowerCase() : f.role} algorithm and does nothing in the ${CYC.ROLES[role] ? CYC.ROLES[role].title.toLowerCase() : role} stage.`); return fn(state, f); };
    for (const f of chair('prior')) dispatch('prior', f, stagePrior);
    const tr = chair('transform');
    const early = (f) => f.family === 'channel' || f.id === 'unrope'; // Pre-RoPE storage is a fixed, position-known rotation: it runs with YCbCr / mid-side, before anything looks at the data
    for (const f of tr.filter(early)) dispatch('transform', f, stageTransform);
    for (const f of chair('perceptual')) dispatch('perceptual', f, stagePerceptual);
    const quant = chair('quantize'); const lossyChain = quant.some((f) => f.role === 'quantize') || tr.some((f) => !early(f) && f.role === 'transform');
    if (quant.some((f) => f.role === 'quantize')) applyProtection(state); // Sink & window exemption's tokens leave the stream before anyone predicts, transforms or quantizes them
    // A dictionary matcher next to a quantizer or a transform codes the quantized indices (DEFLATE after quantization: KVTC, PNG-style ordering),
    // so LZ77/LZW are deferred until just before the entropy-coding stage in that case. On their own (DEFLATE, zstd) they run in the predict slot.
    const isLZ = (f) => f.id === 'lz77' || f.id === 'lzw'; const deferred = [];
    for (const f of chair('predict')) { if (isLZ(f) && lossyChain) deferred.push(f); else dispatch('predict', f, stagePredict); }
    for (const f of tr.filter((f) => !early(f))) dispatch('transform', f, stageTransform);
    applyMasking(state);
    quant.forEach((f, i) => dispatch('quantize', f, (s, g) => stageQuantize(s, g, quant.slice(i + 1).some((h) => h.role === 'quantize' && h.id !== 'evict'))));
    for (const f of deferred) { state.lzDeferred = true; dispatch('predict', f, stagePredict); }
    if (state.flags.tiered && !state.tokenWeights) warn(state, 'Tiered cache keeps a lossless copy in host DRAM, but nothing on the GPU decides which entries to fetch: without Top-k selection the only option is to fetch everything, and PCIe cannot carry that.');
    if (state.protectedTokens && !quant.some((f) => f.role === 'quantize')) warn(state, 'Sink & window exemption exempted tokens from a quantizer that is not in the pipeline.');
    const ent = chair('entropy'); const entIds = [];
    for (const f of ent) {
      if (f.role === 'entropy') entIds.push(f.id);
      else if (isLZ(f)) { state.lzDeferred = true; const st = stagePredict(state, f); st.role = 'entropy'; } // DEFLATE as the entropy-coding stage's pre-pass (KVTC: transform → quantize → LZ77 + Huffman)
      else dispatch('entropy', f, () => null);
    }
    const { finalBits, boundBits } = stageEntropy(state, entIds);
    if (state.flags.decoderNet && !state.flags.neuralEnc) warn(state, 'Learned decoder holds a 20 MB decoder network but no learned encoder produced latents for it: dead weight.');
    return {
      payloadKind: payload.kind, rawBits: state.rawBits, finalBits: Math.round(finalBits), entropyBoundBits: Math.round(boundBits),
      sideInfoBytes: Math.round(state.sideInfoBytes), lossless: state.lossless, distortion: currentDistortion(state), stages: state.stages, warnings: state.warnings,
      backingCopy: !!state.flags.tiered,
    };
  }
  function payloadLabel(kind) { const p = getPayload(kind); return p.def.label(p); }

  CYC.Engine = {
    run, payloadLabel, PAYLOAD_KINDS: Object.keys(PAYLOAD_DEFS), ORDER,
    _internals: { getPayload, generatePayload, dictionaryBytes, serializePlanes, lz77, lz77Decode, lzSymbols, lzw, lzwDecode, bwt, bwtInverse, mtfEncode, mtfDecode, huffmanLengths, huffmanBits, adaptiveOrder0Bits, ppmOrder2Bits, riceBits, rleZeroRuns, kmeans, pcaPowerIteration, dct2d8, haar2d, hadamardBlocks, entropyOf, toHalfBits, halfBitsToFloat },
  };
})();
