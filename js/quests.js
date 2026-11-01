// Choose Your Codec — exercises (mechanics). Spec: docs/SIDEQUESTS.md. Content: js/quests_data.js (CYC.SIDEQUESTS).
//
//   CYC.Quests.render(container, questId, { onComplete(result) })   result = { score, summary, stats }
//   CYC.Quests.has(questId)
//   CYC.Quests.calc.*                                              pure, node-testable
//
// Plain ES2020, no dependencies. Loads in node without `document`: the DOM is touched only inside render().
// All randomness is seeded (mulberry32); a quest replays identically until NEW ROUND reseeds it.

window.CYC = window.CYC || {};

(function () {
  'use strict';

  // ======================================================================= shared math

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

  function rng(seed) {
    const next = mulberry32(seed);
    return {
      next,
      int: (n) => Math.floor(next() * n),
      range: (a, b) => a + (b - a) * next(),
      gauss() {
        let u = 0;
        while (u === 0) u = next();
        const v = next();
        return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
      },
      pick: (arr) => arr[Math.floor(next() * arr.length)],
    };
  }

  const clamp = (v, lo, hi) => (v < lo ? lo : v > hi ? hi : v);
  const log2 = Math.log2;

  // Order-0 entropy (bits/symbol) of a list of counts.
  function entropyOfCounts(counts) {
    counts = Array.from(counts); // may be a one-shot iterator (Map.values())
    let n = 0;
    for (const c of counts) n += c;
    if (!n) return 0;
    let h = 0;
    for (const c of counts) if (c > 0) { const p = c / n; h -= p * log2(p); }
    return h;
  }

  function countValues(arr) {
    const m = new Map();
    for (const v of arr) m.set(v, (m.get(v) || 0) + 1);
    return m;
  }

  // Real Huffman: counts[i] is the weight of symbol i. Returns code lengths (0 for absent symbols).
  function huffmanLengths(counts) {
    const lengths = new Array(counts.length).fill(0);
    const pool = [];
    for (let i = 0; i < counts.length; i++) if (counts[i] > 0) pool.push({ w: counts[i], sym: i, kids: null });
    if (pool.length === 0) return lengths;
    if (pool.length === 1) { lengths[pool[0].sym] = 1; return lengths; }
    while (pool.length > 1) {
      pool.sort((a, b) => a.w - b.w);
      const a = pool.shift(), b = pool.shift();
      pool.push({ w: a.w + b.w, sym: -1, kids: [a, b] });
    }
    (function walk(n, d) {
      if (n.kids) { walk(n.kids[0], d + 1); walk(n.kids[1], d + 1); } else lengths[n.sym] = d;
    })(pool[0], 0);
    return lengths;
  }

  // Orthonormal 8×8 DCT-II (JPEG's transform). Blocks are 64-element row-major arrays.
  const DCT_C = (() => {
    const C = [];
    for (let u = 0; u < 8; u++) {
      C[u] = [];
      const a = u === 0 ? Math.sqrt(1 / 8) : Math.sqrt(2 / 8);
      for (let x = 0; x < 8; x++) C[u][x] = a * Math.cos((2 * x + 1) * u * Math.PI / 16);
    }
    return C;
  })();

  function dct8(blk, out) {
    out = out || new Float64Array(64);
    const tmp = new Float64Array(64);
    for (let y = 0; y < 8; y++) for (let u = 0; u < 8; u++) {
      let s = 0; for (let x = 0; x < 8; x++) s += blk[y * 8 + x] * DCT_C[u][x];
      tmp[y * 8 + u] = s;
    }
    for (let u = 0; u < 8; u++) for (let v = 0; v < 8; v++) {
      let s = 0; for (let y = 0; y < 8; y++) s += tmp[y * 8 + u] * DCT_C[v][y];
      out[v * 8 + u] = s;
    }
    return out;
  }

  function idct8(coef, out) {
    out = out || new Float64Array(64);
    const tmp = new Float64Array(64);
    for (let v = 0; v < 8; v++) for (let x = 0; x < 8; x++) {
      let s = 0; for (let u = 0; u < 8; u++) s += coef[v * 8 + u] * DCT_C[u][x];
      tmp[v * 8 + x] = s;
    }
    for (let x = 0; x < 8; x++) for (let y = 0; y < 8; y++) {
      let s = 0; for (let v = 0; v < 8; v++) s += tmp[v * 8 + x] * DCT_C[v][y];
      out[y * 8 + x] = s;
    }
    return out;
  }

  // JPEG zigzag: ZIGZAG[z] = raster index of the z-th coefficient.
  const ZIGZAG = (() => {
    const z = [];
    for (let s = 0; s <= 14; s++) {
      const d = [];
      for (let r = Math.max(0, s - 7); r <= Math.min(7, s); r++) d.push([r, s - r]);
      if (s % 2 === 0) d.reverse();
      for (const [r, c] of d) z.push(r * 8 + c);
    }
    return z;
  })();

  // Image = { w, h, data: Float32Array (0..255) }, w and h multiples of 8.
  function dctImage(img) {
    const { w, h, data } = img;
    const blocks = [];
    const blk = new Float64Array(64);
    for (let by = 0; by < h / 8; by++) for (let bx = 0; bx < w / 8; bx++) {
      for (let y = 0; y < 8; y++) for (let x = 0; x < 8; x++) blk[y * 8 + x] = data[(by * 8 + y) * w + bx * 8 + x];
      blocks.push(dct8(blk));
    }
    return blocks;
  }

  function idctImage(blocks, w, h) {
    const data = new Float32Array(w * h);
    const px = new Float64Array(64);
    let i = 0;
    for (let by = 0; by < h / 8; by++) for (let bx = 0; bx < w / 8; bx++) {
      idct8(blocks[i++], px);
      for (let y = 0; y < 8; y++) for (let x = 0; x < 8; x++) data[(by * 8 + y) * w + bx * 8 + x] = clamp(px[y * 8 + x], 0, 255);
    }
    return { w, h, data };
  }

  function mse(a, b) {
    let s = 0;
    for (let i = 0; i < a.length; i++) { const d = a[i] - b[i]; s += d * d; }
    return s / a.length;
  }
  const psnr = (m) => (m <= 0 ? Infinity : 10 * Math.log10(255 * 255 / m));

  // The engine-style synthetic test image: smooth gradient + soft bright disc + hard vertical edge + noise.
  // Parameterized by size so the same scene renders at 32×32 or 64×64.
  function synthImage(seed, N) {
    const r = rng(seed);
    const data = new Float32Array(N * N);
    for (let y = 0; y < N; y++) for (let x = 0; x < N; x++) {
      const u = x / (N - 1), v = y / (N - 1);
      let g = 50 + 100 * u + 50 * v;
      const dx = u - 0.36, dy = v - 0.42;
      const d = Math.sqrt(dx * dx + dy * dy);
      g += 80 * clamp((0.23 - d) / 0.03, 0, 1);
      if (u > 0.72) g -= 90;
      g += 4 * r.gauss();
      data[y * N + x] = clamp(g, 0, 255);
    }
    return { w: N, h: N, data };
  }

  // Greedy LZ77 with the dictionary preloaded in the window. Cost model: literal 9 bits, match 20 bits.
  function lz77(text, dict, opts) {
    opts = opts || {};
    const W = opts.window || 4096, MIN = opts.minMatch || 3, MAX = opts.maxMatch || 258;
    const LIT = opts.litBits == null ? 9 : opts.litBits, MATCH = opts.matchBits == null ? 20 : opts.matchBits;
    dict = dict || '';
    const s = dict + text, D = dict.length, n = s.length;
    const tokens = [];
    let bits = 0, lits = 0, matches = 0, dictMatches = 0;
    let i = D;
    while (i < n) {
      let bestLen = 0, bestPos = -1;
      const c0 = s.charCodeAt(i);
      for (let j = Math.max(0, i - W); j < i; j++) {
        if (s.charCodeAt(j) !== c0) continue;
        let l = 1;
        while (l < MAX && i + l < n && s.charCodeAt(j + l) === s.charCodeAt(i + l)) l++;
        if (l > bestLen) { bestLen = l; bestPos = j; }
      }
      if (bestLen >= MIN) {
        const fromDict = bestPos < D;
        tokens.push({ type: 'match', text: s.slice(i, i + bestLen), dist: i - bestPos, len: bestLen, dict: fromDict });
        bits += MATCH; matches++; if (fromDict) dictMatches++;
        i += bestLen;
      } else {
        tokens.push({ type: 'lit', text: s[i] });
        bits += LIT; lits++; i++;
      }
    }
    return { tokens, bits, bytes: Math.ceil(bits / 8), lits, matches, dictMatches, litBits: LIT, matchBits: MATCH };
  }

  // ======================================================================= calc: per quest

  const calc = { mulberry32, rng, entropyOfCounts, countValues, huffmanLengths, dct8, idct8, ZIGZAG, dctImage, idctImage, synthImage, lz77, mse, psnr };

  // ---- shannon
  calc.shannon = {
    SENTENCES: [
      'the quick brown fox jumps over the lazy dog',
      'compression is prediction plus arithmetic',
      'a back reference is a prediction that came true',
      'entropy is the floor and the model lowers it',
    ],
    ALPHABET: 'abcdefghijklmnopqrstuvwxyz ',
    REFS: { fixed: 4.70, freq: 4.14, human: 1.0 },
    cost: (g) => log2(Math.max(1, g)),
    score: (bpl) => clamp(100 * (4.7 - bpl) / (4.7 - 1.5), 0, 100),
  };

  // ---- hufftree
  calc.hufftree = (() => {
    const SYMBOLS = ['A', 'B', 'C', 'D', 'E', 'F'];
    const COUNTS = [45, 13, 12, 16, 9, 5];
    const total = COUNTS.reduce((a, b) => a + b, 0);
    function meanLength(lengths) { let s = 0; for (let i = 0; i < COUNTS.length; i++) s += COUNTS[i] * lengths[i]; return s / total; }
    function optimal() { const lengths = huffmanLengths(COUNTS); return { lengths, mean: meanLength(lengths) }; }
    return {
      SYMBOLS, COUNTS, total,
      meanLength, optimal,
      entropy: () => entropyOfCounts(COUNTS),
      kraft: (lengths) => lengths.reduce((s, l) => s + Math.pow(2, -l), 0),
      score: (mean, opt) => clamp(100 - 250 * Math.max(0, mean - opt - 1e-9), 0, 100),
    };
  })();

  // ---- waterfill
  calc.waterfill = (() => {
    const N = 32, BUDGET = 192, BANDS = 8, COEFS_PER_BAND = 8, MAXBITS = 8;
    const image = (seed) => synthImage(seed, N);
    // Blocks, per-band energy (mean square per coefficient, i.e. the variance of a zero-mean band) and max |coef|.
    function analyze(img) {
      const blocks = dctImage(img);
      const variances = new Array(BANDS).fill(0), maxAbs = new Array(BANDS).fill(0);
      for (const blk of blocks) for (let z = 0; z < 64; z++) {
        const k = Math.floor(z / COEFS_PER_BAND), c = blk[ZIGZAG[z]];
        variances[k] += c * c;
        if (Math.abs(c) > maxAbs[k]) maxAbs[k] = Math.abs(c);
      }
      for (let k = 0; k < BANDS; k++) variances[k] /= blocks.length * COEFS_PER_BAND;
      return { img, blocks, variances, maxAbs, w: img.w, h: img.h };
    }
    // Uniform quantizer per band: Δ_k = 2·max|coef|/2^b, b = 0 zeroes the band.
    function reconstruct(an, bits) {
      const out = [];
      const deltas = bits.map((b, k) => (b > 0 ? (an.maxAbs[k] * 2) / Math.pow(2, b) : 0));
      for (const blk of an.blocks) {
        const q = new Float64Array(64);
        for (let z = 0; z < 64; z++) {
          const k = Math.floor(z / COEFS_PER_BAND), i = ZIGZAG[z];
          q[i] = bits[k] > 0 ? Math.round(blk[i] / deltas[k]) * deltas[k] : 0;
        }
        out.push(q);
      }
      const rec = idctImage(out, an.w, an.h);
      const m = mse(an.img.data, rec.data);
      return { img: rec, mse: m, psnr: psnr(m), deltas, bits: bits.slice(), blockBits: blockBits(bits) };
    }
    const blockBits = (bits) => COEFS_PER_BAND * bits.reduce((a, b) => a + b, 0);
    // Reverse water-filling: θ such that Σ_k 8·max(0, ½log2(σ²_k/θ)) = budget (bisection on log θ).
    function theta(variances, budget) {
      const rate = (th) => variances.reduce((s, v) => s + COEFS_PER_BAND * Math.max(0, 0.5 * log2(v / th)), 0);
      let lo = Math.log(Math.min(...variances.filter((v) => v > 0)) * 1e-6), hi = Math.log(Math.max(...variances));
      for (let it = 0; it < 200; it++) {
        const mid = (lo + hi) / 2;
        if (rate(Math.exp(mid)) > budget) lo = mid; else hi = mid;
      }
      const th = Math.exp((lo + hi) / 2);
      return { theta: th, rates: variances.map((v) => Math.max(0, 0.5 * log2(v / th))) };
    }
    // Integer allocation: one bit at a time to the band with the largest σ²_k / 4^{b_k} (marginal-return greedy).
    function greedy(variances, bandBits, maxBits) {
      maxBits = maxBits || MAXBITS;
      const bits = new Array(variances.length).fill(0);
      for (let n = 0; n < bandBits; n++) {
        let best = -1, bestGain = -1;
        for (let k = 0; k < bits.length; k++) {
          if (bits[k] >= maxBits) continue;
          const gain = variances[k] / Math.pow(4, bits[k]);
          if (gain > bestGain) { bestGain = gain; best = k; }
        }
        if (best < 0) break;
        bits[best]++;
      }
      return bits;
    }
    const uniform = () => new Array(BANDS).fill(BUDGET / 64);
    return {
      N, BUDGET, BANDS, COEFS_PER_BAND, MAXBITS,
      image, analyze, reconstruct, blockBits, theta, greedy, uniform,
      optimal: (an) => greedy(an.variances, BUDGET / COEFS_PER_BAND, MAXBITS),
      score: (optMse, yourMse) => (yourMse <= 0 ? 100 : clamp(100 * optMse / yourMse, 0, 100)),
    };
  })();

  // ---- jnd
  calc.jnd = (() => {
    const N = 64, TRIALS = 12, START = 24, STEP = 1.25;
    const image = (seed) => synthImage(seed, N);
    // Uniform quantizer step Δ on all 64 coefficients of every block.
    function quantizeBlocks(blocks, delta) {
      return blocks.map((blk) => { const q = new Float64Array(64); for (let i = 0; i < 64; i++) q[i] = Math.round(blk[i] / delta) * delta; return q; });
    }
    function degrade(img, delta) {
      const blocks = dctImage(img);
      const rec = idctImage(quantizeBlocks(blocks, delta), img.w, img.h);
      return { img: rec, mse: mse(img.data, rec.data) };
    }
    // Chroma variant: Y = the gray image, Cb/Cr smooth colored gradients (a blue→yellow wash and a warm disc).
    function makeChroma(img) {
      const { w, h, data } = img;
      const Cb = new Float32Array(w * h), Cr = new Float32Array(w * h);
      for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
        const u = x / (w - 1), v = y / (h - 1), i = y * w + x;
        const dx = u - 0.36, dy = v - 0.42, d = Math.sqrt(dx * dx + dy * dy);
        Cb[i] = clamp(128 + 70 * (u - 0.5) - 30 * clamp((0.23 - d) / 0.03, 0, 1), 0, 255);
        Cr[i] = clamp(128 + 50 * (0.5 - v) + 40 * clamp((0.23 - d) / 0.03, 0, 1), 0, 255);
      }
      return { Y: img, Cb: { w, h, data: Cb }, Cr: { w, h, data: Cr } };
    }
    function degradeChroma(ycc, delta) {
      const cb = degrade(ycc.Cb, delta), cr = degrade(ycc.Cr, delta);
      return { ycc: { Y: ycc.Y, Cb: cb.img, Cr: cr.img }, mse: (cb.mse + cr.mse) / 2 };
    }
    function ycbcrToRgb(ycc) {
      const { w, h } = ycc.Y, out = new Uint8ClampedArray(w * h * 4);
      for (let i = 0; i < w * h; i++) {
        const Y = ycc.Y.data[i], cb = ycc.Cb.data[i] - 128, cr = ycc.Cr.data[i] - 128;
        out[i * 4] = Y + 1.402 * cr; out[i * 4 + 1] = Y - 0.344136 * cb - 0.714136 * cr; out[i * 4 + 2] = Y + 1.772 * cb; out[i * 4 + 3] = 255;
      }
      return { w, h, data: out };
    }
    // Staircase: seeing the difference makes the next trial gentler (smaller Δ); missing it makes it harsher.
    const nextDelta = (delta, correct) => (correct ? delta / STEP : delta * STEP);
    function threshold(deltas) {
      const last = deltas.slice(-6);
      return Math.exp(last.reduce((s, d) => s + Math.log(d), 0) / last.length);
    }
    // Order-0 entropy (bits/coefficient) of all quantized coefficient indices of a set of planes at step Δ.
    function coefEntropy(planes, delta) {
      const vals = [];
      for (const p of planes) for (const blk of dctImage(p)) for (let i = 0; i < 64; i++) vals.push(Math.round(blk[i] / delta));
      return entropyOfCounts(countValues(vals).values());
    }
    return {
      N, TRIALS, START, STEP,
      image, degrade, makeChroma, degradeChroma, ycbcrToRgb, nextDelta, threshold, coefEntropy,
      bitsSaved: (delta) => log2(delta),                           // per coefficient, vs Δ = 1
      blockCost: (planes, delta) => 64 * coefEntropy(planes, delta), // bits for one 8×8 block at order-0 entropy
    };
  })();

  // ---- phrasebook
  calc.phrasebook = (() => {
    const SAMPLES = [
      '{"id":4821,"type":"click","ok":true,"ts":1717,"user":"u_204","region":"eu-west"}',
      '{"id":4822,"type":"view","ok":true,"ts":1718,"user":"u_091","region":"us-east"}',
      '{"id":4823,"type":"purchase","ok":false,"ts":1719,"user":"u_204","region":"eu-west"}',
      '{"id":4824,"type":"click","ok":true,"ts":1720,"user":"u_377","region":"ap-south"}',
      '{"id":4825,"type":"view","ok":true,"ts":1721,"user":"u_091","region":"us-east"}',
      '{"id":4826,"type":"click","ok":true,"ts":1722,"user":"u_150","region":"eu-west"}',
    ];
    const TEST = '{"id":4830,"type":"view","ok":true,"ts":1725,"user":"u_311","region":"us-east"}';
    const FOREIGN = '<event id="4830" kind="VIEW" ok="1" ts="1725" who="U311" zone="US_EAST"/>';
    const BUDGET = 96;
    // Fragments: keys with quotes and colon, quoted string values, numbers, bare words, punctuation runs.
    const TOKEN_RE = /"[^"]*":|"[^"]*"|-?\d+(?:\.\d+)?|[A-Za-z_]+|[^\w"\s]+|\s+/g;
    function tokenize(msg) {
      const out = [];
      let m;
      TOKEN_RE.lastIndex = 0;
      while ((m = TOKEN_RE.exec(msg)) !== null) {
        const t = m[0];
        const kind = /^"[^"]*":$/.test(t) ? 'key' : /^"/.test(t) ? 'str' : /^-?\d/.test(t) ? 'num' : /^[A-Za-z_]/.test(t) ? 'word' : 'punct';
        out.push({ text: t, kind, start: m.index });
      }
      return out;
    }
    const compress = (msg, dict) => lz77(msg, dict || '');
    function savings(msg, dict) {
      const without = compress(msg, ''), withDict = compress(msg, dict);
      const saved = without.bits - withDict.bits;
      return { without, withDict, saved, pct: without.bits ? 100 * saved / without.bits : 0 };
    }
    // Greedy trainer: every substring of length 3..24 that occurs in ≥ 2 samples, scored (occurrences−1)×length,
    // taken best-first under the budget, skipping anything already covered by the dictionary so far.
    function train(samples, budget) {
      budget = budget || BUDGET;
      const df = new Map();
      for (const s of samples) {
        const seen = new Set();
        for (let len = 3; len <= 24; len++) for (let i = 0; i + len <= s.length; i++) seen.add(s.slice(i, i + len));
        for (const sub of seen) df.set(sub, (df.get(sub) || 0) + 1);
      }
      const list = [];
      for (const [sub, d] of df) {
        if (d < 2) continue;
        let occ = 0;
        for (const s of samples) { let p = 0; while ((p = s.indexOf(sub, p)) !== -1) { occ++; p += sub.length; } }
        list.push({ s: sub, occ, score: (occ - 1) * sub.length });
      }
      list.sort((a, b) => b.score - a.score || b.s.length - a.s.length || (a.s < b.s ? -1 : 1));
      let dict = '';
      const picks = [];
      for (const c of list) {
        if (dict.includes(c.s)) continue;
        if (dict.length + c.s.length > budget) continue;
        dict += c.s; picks.push(c);
        if (dict.length >= budget) break;
      }
      return { dict, picks, candidates: list.length };
    }
    return {
      SAMPLES, TEST, FOREIGN, BUDGET,
      tokenize, compress, savings, train,
      score: (yourSaved, trainerSaved) => (trainerSaved <= 0 ? 100 : clamp(100 * yourSaved / trainerSaved, 0, 100)),
    };
  })();

  // ---- safehouses
  calc.safehouses = (() => {
    const K = 4, N = 60;
    const CLUSTERS = [
      { cx: -0.55, cy: -0.35, sa: 0.24, sb: 0.07, th: 0.55 },
      { cx: 0.45, cy: 0.55, sa: 0.18, sb: 0.06, th: -0.9 },
      { cx: 0.5, cy: -0.5, sa: 0.26, sb: 0.08, th: 0.15 },
    ];
    function points(seed) {
      const r = rng(seed);
      const pts = new Float64Array(2 * N);
      for (let i = 0; i < N; i++) {
        const c = CLUSTERS[i % CLUSTERS.length];
        const a = r.gauss() * c.sa, b = r.gauss() * c.sb;
        pts[2 * i] = clamp(c.cx + a * Math.cos(c.th) - b * Math.sin(c.th), -0.97, 0.97);
        pts[2 * i + 1] = clamp(c.cy + a * Math.sin(c.th) + b * Math.cos(c.th), -0.97, 0.97);
      }
      return pts;
    }
    function assign(pts, cents) {
      const n = pts.length / 2, k = cents.length / 2, a = new Uint16Array(n);
      for (let i = 0; i < n; i++) {
        let best = 0, bd = Infinity;
        for (let j = 0; j < k; j++) {
          const dx = pts[2 * i] - cents[2 * j], dy = pts[2 * i + 1] - cents[2 * j + 1], d = dx * dx + dy * dy;
          if (d < bd) { bd = d; best = j; }
        }
        a[i] = best;
      }
      return a;
    }
    function distortion(pts, cents, a) {
      a = a || assign(pts, cents);
      let s = 0;
      for (let i = 0; i < a.length; i++) { const j = a[i]; const dx = pts[2 * i] - cents[2 * j], dy = pts[2 * i + 1] - cents[2 * j + 1]; s += dx * dx + dy * dy; }
      return s;
    }
    // One Lloyd step: move each centroid to the mean of its points (an empty cell keeps its centroid).
    function step(pts, cents) {
      const a = assign(pts, cents), k = cents.length / 2;
      const sum = new Float64Array(2 * k), cnt = new Uint32Array(k);
      for (let i = 0; i < a.length; i++) { sum[2 * a[i]] += pts[2 * i]; sum[2 * a[i] + 1] += pts[2 * i + 1]; cnt[a[i]]++; }
      const next = Float64Array.from(cents);
      for (let j = 0; j < k; j++) if (cnt[j]) { next[2 * j] = sum[2 * j] / cnt[j]; next[2 * j + 1] = sum[2 * j + 1] / cnt[j]; }
      return { cents: next, assign: a, counts: cnt };
    }
    function lloyd(pts, cents, maxIter) {
      maxIter = maxIter || 20;
      const history = [Float64Array.from(cents)];
      let cur = Float64Array.from(cents), a = assign(pts, cur), converged = false, it = 0;
      for (; it < maxIter; it++) {
        const s = step(pts, cur);
        cur = s.cents;
        const a2 = assign(pts, cur);
        history.push(cur);
        let same = true;
        for (let i = 0; i < a.length; i++) if (a[i] !== a2[i]) { same = false; break; }
        a = a2;
        if (same) { converged = true; it++; break; }
      }
      return { centroids: cur, assign: a, history, iterations: it, converged, distortion: distortion(pts, cur, a) };
    }
    // Random restarts: Forgy init (K distinct data points), Lloyd to convergence, keep the lowest distortion.
    function bestOf(pts, k, runs, seed) {
      const r = rng(seed), n = pts.length / 2;
      let best = null;
      const all = [];
      for (let t = 0; t < runs; t++) {
        const idx = new Set();
        while (idx.size < k) idx.add(r.int(n));
        const init = new Float64Array(2 * k);
        let j = 0;
        for (const i of idx) { init[2 * j] = pts[2 * i]; init[2 * j + 1] = pts[2 * i + 1]; j++; }
        const res = lloyd(pts, init, 50);
        all.push(res.distortion);
        if (!best || res.distortion < best.distortion) best = res;
      }
      return { centroids: best.centroids, distortion: best.distortion, assign: best.assign, runs: all };
    }
    return {
      K, N, CLUSTERS,
      points, assign, distortion, step, lloyd, bestOf,
      bitsPerPoint: log2(K), codebookBytes: K * 2 * 4, rawBitsPerPoint: 2 * 32,
      score: (best, yours) => (yours <= 0 ? 100 : clamp(100 * best / yours, 0, 100)),
    };
  })();

  // ---- blockpack
  calc.blockpack = (() => {
    const BITS = [2, 3, 4, 5, 6, 8], BLOCKS = [8, 16, 32, 64, 128, 256], M = 256, BUDGET = 4.5;
    function weights(seed) {
      const r = rng(seed);
      const w = new Float64Array(M);
      for (let i = 0; i < M; i++) w[i] = 0.02 * r.gauss();
      const idx = new Set();
      while (idx.size < 3) idx.add(r.int(M));
      for (const i of idx) w[i] = (r.next() < 0.5 ? -1 : 1) * r.range(0.3, 0.5);
      return w;
    }
    const bpw = (b, n, min) => (16 + (min ? 16 : 0) + n * b) / n;
    // Normalized Sylvester–Hadamard transform (fast Walsh–Hadamard). Symmetric and orthogonal: its own inverse.
    function hadamard(x) {
      const n = x.length, y = Float64Array.from(x);
      for (let h = 1; h < n; h <<= 1) for (let i = 0; i < n; i += 2 * h) for (let j = i; j < i + h; j++) {
        const a = y[j], b = y[j + h]; y[j] = a + b; y[j + h] = a - b;
      }
      const s = 1 / Math.sqrt(n);
      for (let i = 0; i < n; i++) y[i] *= s;
      return y;
    }
    function quantize(w, o) {
      const b = o.b, n = o.n, min = !!o.min, rot = !!o.rot;
      const x = rot ? hadamard(w) : Float64Array.from(w);
      const y = new Float64Array(x.length), q = new Int32Array(x.length);
      let blocks = 0;
      for (let s = 0; s < x.length; s += n) {
        const e = Math.min(s + n, x.length);
        blocks++;
        if (min) {
          let lo = Infinity, hi = -Infinity;
          for (let i = s; i < e; i++) { if (x[i] < lo) lo = x[i]; if (x[i] > hi) hi = x[i]; }
          const levels = (1 << b) - 1, scale = (hi - lo) / levels || 1;
          for (let i = s; i < e; i++) { const qi = clamp(Math.round((x[i] - lo) / scale), 0, levels); q[i] = qi; y[i] = lo + qi * scale; }
        } else {
          let am = 0;
          for (let i = s; i < e; i++) if (Math.abs(x[i]) > am) am = Math.abs(x[i]);
          const L = (1 << (b - 1)) - 1, scale = am / L || 1;
          for (let i = s; i < e; i++) { const qi = clamp(Math.round(x[i] / scale), -L, L); q[i] = qi; y[i] = qi * scale; }
        }
      }
      const recon = rot ? hadamard(y) : y;
      let maxErr = 0, se = 0;
      for (let i = 0; i < w.length; i++) { const d = Math.abs(w[i] - recon[i]); if (d > maxErr) maxErr = d; se += d * d; }
      return { recon, q, transformed: x, transformedRecon: y, maxErr, rms: Math.sqrt(se / w.length), bpw: bpw(b, n, min), blocks };
    }
    function formatName(b, n, min, rot) {
      const v = bpw(b, n, min).toFixed(2);
      let name;
      if (b === 4 && n === 32 && !min) name = 'Q4_0: 4.5 bpw';
      else if (b === 4 && n === 32 && min) name = 'Q4_1: 5.0 bpw';
      else if (b === 8 && n === 32 && !min) name = 'Q8_0: 8.5 bpw';
      else if (b === 8 && n === 32 && min) name = 'Q8_1: 9.0 bpw';
      else if (b === 4 && n === 64 && min) name = '≈ Q4_K sub-block (real one uses 6-bit scales)';
      else if (b === 4 && n === 64 && !min) name = '≈ NF4 (bitsandbytes; the real one uses a non-uniform 4-bit grid)';
      else if (b === 5 && n === 32 && !min) name = 'Q5_0: 5.5 bpw';
      else if (b === 5 && n === 32 && min) name = 'Q5_1: 6.0 bpw';
      else if ((b === 2 || b === 3 || b === 5 || b === 6) && n <= 32) name = `≈ Q${b}_K-style sub-block (${v} bpw; real K-quants nest 16/32-weight sub-blocks in 256-weight super-blocks)`;
      else name = `no shipping format: ${b}-bit, ${n}-weight blocks${min ? ' with min' : ''} (${v} bpw)`;
      if (rot) name += ' + QuaRot/SpinQuant-style rotation';
      return name;
    }
    return {
      BITS, BLOCKS, M, BUDGET,
      weights, bpw, hadamard, quantize, formatName,
      modelBytes: (v, params) => (params || 7e9) * v / 8,
      score: (refRms, yourRms, v) => (v > BUDGET + 1e-9 ? 0 : yourRms <= 0 ? 100 : clamp(100 * refRms / yourRms, 0, 100)),
    };
  })();

  // ---- raceaccess
  calc.raceaccess = (() => {
    const N = 4096, SPACINGS = [1, 16, 64, 256, 1024, null], Q = 0.925, CP_BITS = 16;
    const label = (s) => (s == null ? 'none' : String(s));
    // Skewed source: geometric with ratio 0.925 truncated to 256 symbols (H ≈ 5.1 bits).
    function symbols(seed) {
      const r = rng(seed), s = new Uint8Array(N), lq = Math.log(Q);
      for (let i = 0; i < N; i++) s[i] = Math.min(255, Math.floor(Math.log(1 - r.next()) / lq));
      return s;
    }
    function analyze(sym) {
      const counts = new Array(256).fill(0);
      for (const v of sym) counts[v]++;
      const lengths = huffmanLengths(counts);
      let totalBits = 0;
      for (let v = 0; v < 256; v++) totalBits += counts[v] * lengths[v];
      const H = entropyOfCounts(counts);
      return { counts, lengths, totalBits, meanBits: totalBits / sym.length, H, entropyBits: H * sym.length, alphabet: counts.filter((c) => c > 0).length };
    }
    const checkpoints = (spacing) => (spacing == null ? 0 : Math.ceil(N / spacing));
    function bytes(totalBits, spacing) {
      return { fixed: N, huffman: Math.ceil(totalBits / 8) + (CP_BITS / 8) * checkpoints(spacing), payload: Math.ceil(totalBits / 8), checkpointBytes: (CP_BITS / 8) * checkpoints(spacing) };
    }
    // Decode steps: symbols decoded from the last checkpoint up to and including the target (fixed length = 1).
    function steps(index, spacing) {
      const cp = spacing == null ? 0 : Math.floor(index / spacing) * spacing;
      return { checkpoint: cp, steps: index - cp + 1 };
    }
    function meanSteps(spacing) {
      let t = 0;
      for (let i = 0; i < N; i++) t += steps(i, spacing).steps;
      return t / N;
    }
    function row(totalBits, spacing) {
      const b = bytes(totalBits, spacing), ms = meanSteps(spacing);
      const smaller = b.huffman < b.fixed;
      return { spacing, label: label(spacing), checkpoints: checkpoints(spacing), fixedBytes: b.fixed, huffmanBytes: b.huffman, payloadBytes: b.payload, checkpointBytes: b.checkpointBytes, meanSteps: ms, smaller, fast: ms <= 64, score: smaller && ms <= 64 ? 100 : smaller ? 60 : 30 };
    }
    return {
      N, SPACINGS, CP_BITS, label,
      symbols, analyze, checkpoints, bytes, steps, meanSteps, row,
      table: (totalBits) => SPACINGS.map((s) => row(totalBits, s)),
    };
  })();

  // ---- siamese (level 5): joint vs marginal entropy
  calc.siamese = (() => {
    const N = 4000, SLOPES = [0.9, 0.6, 0.3, 0.0], BINS = 16, LO = -4, HI = 4;
    function pairs(seed, slope) {
      const r = rng(seed), a = new Float64Array(N), b = new Float64Array(N);
      for (let i = 0; i < N; i++) { a[i] = r.gauss(); b[i] = slope * a[i] + 0.1 * r.gauss(); }
      return { a, b, slope, n: N };
    }
    const weights = (a) => Float64Array.from(a, (v) => Math.abs(v) + 0.1); // Fisher-information stand-in
    // 1-D Lloyd with K centers (init at evenly spaced quantiles); optional sample weights in the mean step.
    function kmeans1d(x, K, iters, w) {
      const sorted = Float64Array.from(x).sort(), c = new Float64Array(K);
      for (let k = 0; k < K; k++) c[k] = sorted[Math.floor(((k + 0.5) / K) * x.length)];
      const assign = new Uint8Array(x.length);
      for (let it = 0; it < iters; it++) {
        const sum = new Float64Array(K), cnt = new Float64Array(K);
        for (let i = 0; i < x.length; i++) {
          let bk = 0, bd = Infinity;
          for (let k = 0; k < K; k++) { const d = Math.abs(x[i] - c[k]); if (d < bd) { bd = d; bk = k; } }
          assign[i] = bk; const wi = w ? w[i] : 1; sum[bk] += wi * x[i]; cnt[bk] += wi;
        }
        for (let k = 0; k < K; k++) if (cnt[k] > 0) c[k] = sum[k] / cnt[k];
      }
      return { centers: c, assign };
    }
    // 2-D Lloyd, K centroids, seeded Forgy init; optional sample weights.
    function kmeans2d(a, b, K, iters, seed, w) {
      const r = rng(seed), n = a.length, c = new Float64Array(2 * K), idx = new Set();
      while (idx.size < K) idx.add(r.int(n));
      let k = 0;
      for (const i of idx) { c[2 * k] = a[i]; c[2 * k + 1] = b[i]; k++; }
      const assign = new Uint8Array(n);
      for (let it = 0; it < iters; it++) {
        const sum = new Float64Array(2 * K), cnt = new Float64Array(K);
        for (let i = 0; i < n; i++) {
          let bk = 0, bd = Infinity;
          for (let j = 0; j < K; j++) { const dx = a[i] - c[2 * j], dy = b[i] - c[2 * j + 1], d = dx * dx + dy * dy; if (d < bd) { bd = d; bk = j; } }
          assign[i] = bk; const wi = w ? w[i] : 1; sum[2 * bk] += wi * a[i]; sum[2 * bk + 1] += wi * b[i]; cnt[bk] += wi;
        }
        for (let j = 0; j < K; j++) if (cnt[j] > 0) { c[2 * j] = sum[2 * j] / cnt[j]; c[2 * j + 1] = sum[2 * j + 1] / cnt[j]; }
      }
      return { centroids: c, assign };
    }
    // MSE per value (both coordinates) and |a|-weighted MSE, for a reconstruction (ra, rb).
    function errors(a, b, ra, rb) {
      const w = weights(a);
      let se = 0, wse = 0, ws = 0;
      for (let i = 0; i < a.length; i++) { const d = (a[i] - ra[i]) ** 2 + (b[i] - rb[i]) ** 2; se += d; wse += w[i] * d; ws += w[i]; }
      return { mse: se / (2 * a.length), wmse: wse / (2 * ws) };
    }
    // SCALAR: one 1-bit quantizer per axis → a 2×2 grid of reconstruction points (2 bits per pair).
    function scalar(P, w) {
      const qa = kmeans1d(P.a, 2, 10, w), qb = kmeans1d(P.b, 2, 10, w);
      const ra = Float64Array.from(qa.assign, (k) => qa.centers[k]), rb = Float64Array.from(qb.assign, (k) => qb.centers[k]);
      const points = [];
      for (const x of qa.centers) for (const y of qb.centers) points.push([x, y]);
      return Object.assign({ mode: 'scalar', ca: Array.from(qa.centers), cb: Array.from(qb.centers), points, ra, rb }, errors(P.a, P.b, ra, rb));
    }
    // JOINT: one 2-D k-means with 4 centroids (2 bits per pair).
    function joint(P, seed, w) {
      const km = kmeans2d(P.a, P.b, 4, 15, seed, w);
      const ra = Float64Array.from(km.assign, (k) => km.centroids[2 * k]), rb = Float64Array.from(km.assign, (k) => km.centroids[2 * k + 1]);
      const points = [];
      for (let k = 0; k < 4; k++) points.push([km.centroids[2 * k], km.centroids[2 * k + 1]]);
      return Object.assign({ mode: 'joint', centroids: km.centroids, points, ra, rb }, errors(P.a, P.b, ra, rb));
    }
    // Marginal and joint order-0 entropies from 16 equal-width bins over [−4, 4].
    function entropy(P) {
      const bin = (v) => clamp(Math.floor(((v - LO) / (HI - LO)) * BINS), 0, BINS - 1);
      const ca = new Array(BINS).fill(0), cb = new Array(BINS).fill(0), cab = new Array(BINS * BINS).fill(0);
      for (let i = 0; i < P.n; i++) { const x = bin(P.a[i]), y = bin(P.b[i]); ca[x]++; cb[y]++; cab[x * BINS + y]++; }
      const Ha = entropyOfCounts(ca), Hb = entropyOfCounts(cb), Hab = entropyOfCounts(cab);
      return { Ha, Hb, Hab, I: Ha + Hb - Hab, bins: BINS };
    }
    return { N, SLOPES, BINS, pairs, weights, kmeans1d, kmeans2d, scalar, joint, entropy, score: (s, j) => clamp((100 * (s / j - 1)) / 1.5, 20, 100) };
  })();

  // ---- anchorchain (level 5): delta coding with parallel decode
  calc.anchorchain = (() => {
    const T = 64, C = 16, G = [4, 8, 10, 16, 32], DELTA = 1, RAW_BITS = 8;
    // x[t][c] = offset_c + random walk (step N(0, 0.6)) + noise N(0, 0.3), offsets U(−20, 20).
    function matrix(seed) {
      const r = rng(seed), x = new Float64Array(T * C);
      for (let c = 0; c < C; c++) {
        const off = r.range(-20, 20);
        let walk = 0;
        for (let t = 0; t < T; t++) { walk += 0.6 * r.gauss(); x[t * C + c] = off + walk + 0.3 * r.gauss(); }
      }
      return x;
    }
    const q = (v) => Math.round(v / DELTA) * DELTA;
    // NONE: q(x). CHAIN: open-loop residual q(x[t] − x[t−1]), reconstructed by accumulating quantized deltas (drift is real).
    // ANCHOR: every g-th token raw (quantized, costed at 8 bits), others as q(x[t] − x̂[anchor]) against the anchor's reconstruction.
    function code(x, mode, g) {
      const recon = new Float64Array(T * C), symbols = [];
      let raw = 0;
      for (let c = 0; c < C; c++) for (let t = 0; t < T; t++) {
        const i = t * C + c;
        if (mode === 'none') { recon[i] = q(x[i]); symbols.push(recon[i]); }
        else if (mode === 'chain') {
          if (t === 0) { recon[i] = q(x[i]); raw++; }
          else { const d = q(x[i] - x[i - C]); symbols.push(d); recon[i] = recon[i - C] + d; }
        } else {
          if (t % g === 0) { recon[i] = q(x[i]); raw++; }
          else { const ai = (t - (t % g)) * C + c, d = q(x[i] - recon[ai]); symbols.push(d); recon[i] = recon[ai] + d; }
        }
      }
      const H = entropyOfCounts(countValues(symbols).values());
      const bitsPerElement = (raw * RAW_BITS + H * symbols.length) / (T * C);
      const errByToken = new Float64Array(T);
      let maxErr = 0;
      for (let t = 0; t < T; t++) for (let c = 0; c < C; c++) { const e = Math.abs(x[t * C + c] - recon[t * C + c]); if (e > maxErr) maxErr = e; if (e > errByToken[t]) errByToken[t] = e; }
      const depth = mode === 'none' ? 1 : mode === 'anchor' ? 2 : T;
      return { mode, g, H, bitsPerElement, raw, nSymbols: symbols.length, depth, maxErr, errByToken, recon };
    }
    const score = (mode, g) => (mode === 'anchor' ? ([8, 10, 16].includes(g) ? 100 : 70) : mode === 'chain' ? 40 : 20);
    return { T, C, G, DELTA, RAW_BITS, matrix, code, score };
  })();

  // ---- rankkeys (level 5): low-rank truncation of keys vs values
  calc.rankkeys = (() => {
    const L = 256, D = 64, NQ = 8, TOPK = 16, NOISE = 0.05;
    // Columns of a random matrix with orthonormal columns (Gram–Schmidt on Gaussians, two passes).
    function orthonormal(r, rows, cols) {
      const out = [];
      for (let j = 0; j < cols; j++) {
        const v = new Float64Array(rows);
        for (let i = 0; i < rows; i++) v[i] = r.gauss();
        for (let pass = 0; pass < 2; pass++) for (const u of out) { let d = 0; for (let i = 0; i < rows; i++) d += u[i] * v[i]; for (let i = 0; i < rows; i++) v[i] -= d * u[i]; }
        let n = 0;
        for (let i = 0; i < rows; i++) n += v[i] * v[i];
        n = Math.sqrt(n);
        for (let i = 0; i < rows; i++) v[i] /= n;
        out.push(v);
      }
      return out;
    }
    // M = U·diag(s)·Vᵀ + noise; the noise is NOISE × the RMS entry of the signal (U, V orthonormal make entries O(1/√L)).
    function lowrankPlusNoise(r, s) {
      const U = orthonormal(r, L, D), V = orthonormal(r, D, D), M = new Float64Array(L * D);
      let ss = 0;
      for (let i = 0; i < D; i++) ss += s[i] * s[i];
      const sigmaNoise = NOISE * Math.sqrt(ss / (L * D));
      for (let t = 0; t < L; t++) for (let j = 0; j < D; j++) {
        let v = 0;
        for (let i = 0; i < D; i++) v += U[i][t] * s[i] * V[i][j];
        M[t * D + j] = v + sigmaNoise * r.gauss();
      }
      return M;
    }
    function matrices(seed) {
      const r = rng(seed);
      const sK = Array.from({ length: D }, (_, i) => 8 * Math.pow(0.85, i)), sV = new Array(D).fill(2);
      const K = lowrankPlusNoise(r, sK), V = lowrankPlusNoise(r, sV);
      const Q = new Float64Array(NQ * D);
      for (let i = 0; i < NQ * D; i++) Q[i] = r.gauss() / Math.sqrt(D);
      return { K, V, Q, sK, sV };
    }
    // Thin SVD of an L×D matrix via cyclic Jacobi on the D×D Gram matrix: sigma (desc), right singular vectors V, P = M·V.
    function svd(M) {
      const A = new Float64Array(D * D), Vm = new Float64Array(D * D);
      for (let i = 0; i < D; i++) for (let j = i; j < D; j++) { let s = 0; for (let t = 0; t < L; t++) s += M[t * D + i] * M[t * D + j]; A[i * D + j] = s; A[j * D + i] = s; }
      for (let i = 0; i < D; i++) Vm[i * D + i] = 1;
      for (let sweep = 0; sweep < 60; sweep++) {
        let off = 0;
        for (let p = 0; p < D; p++) for (let qq = p + 1; qq < D; qq++) off += A[p * D + qq] * A[p * D + qq];
        if (off < 1e-20) break;
        for (let p = 0; p < D; p++) for (let qq = p + 1; qq < D; qq++) {
          const apq = A[p * D + qq];
          if (Math.abs(apq) < 1e-300) continue;
          const theta = (A[qq * D + qq] - A[p * D + p]) / (2 * apq);
          const t = (theta >= 0 ? 1 : -1) / (Math.abs(theta) + Math.sqrt(theta * theta + 1));
          const c = 1 / Math.sqrt(t * t + 1), s = t * c;
          for (let k = 0; k < D; k++) { const akp = A[k * D + p], akq = A[k * D + qq]; A[k * D + p] = c * akp - s * akq; A[k * D + qq] = s * akp + c * akq; }
          for (let k = 0; k < D; k++) { const apk = A[p * D + k], aqk = A[qq * D + k]; A[p * D + k] = c * apk - s * aqk; A[qq * D + k] = s * apk + c * aqk; }
          for (let k = 0; k < D; k++) { const vkp = Vm[k * D + p], vkq = Vm[k * D + qq]; Vm[k * D + p] = c * vkp - s * vkq; Vm[k * D + qq] = s * vkp + c * vkq; }
        }
      }
      const order = Array.from({ length: D }, (_, i) => i).sort((i, j) => A[j * D + j] - A[i * D + i]);
      const sigma = order.map((i) => Math.sqrt(Math.max(0, A[i * D + i])));
      const V = order.map((i) => { const v = new Float64Array(D); for (let k = 0; k < D; k++) v[k] = Vm[k * D + i]; return v; });
      const P = new Float64Array(L * D);
      for (let t = 0; t < L; t++) for (let i = 0; i < D; i++) { let s = 0; for (let k = 0; k < D; k++) s += M[t * D + k] * V[i][k]; P[t * D + i] = s; }
      return { sigma, V, P };
    }
    function softmax(z) {
      let m = -Infinity;
      for (const v of z) if (v > m) m = v;
      const out = new Float64Array(z.length);
      let s = 0;
      for (let i = 0; i < z.length; i++) { out[i] = Math.exp(z[i] - m); s += out[i]; }
      for (let i = 0; i < z.length; i++) out[i] /= s;
      return out;
    }
    const topk = (z, k) => new Set(Array.from(z.keys()).sort((i, j) => z[j] - z[i]).slice(0, k));
    // For K: per rank r, mean over queries of Σ_t |softmax(QKᵀ/√d) − softmax(QK_rᵀ/√d)| and the top-16 overlap.
    // QK_rᵀ = (QV_r)(KV_r)ᵀ, accumulated one singular direction at a time.
    function analyzeK(K, Q) {
      const { sigma, V, P } = svd(K), scale = 1 / Math.sqrt(D);
      const Qv = new Float64Array(NQ * D);
      for (let qi = 0; qi < NQ; qi++) for (let i = 0; i < D; i++) { let s = 0; for (let k = 0; k < D; k++) s += Q[qi * D + k] * V[i][k]; Qv[qi * D + i] = s; }
      const exact = [], exactTop = [];
      for (let qi = 0; qi < NQ; qi++) {
        const sc = new Float64Array(L);
        for (let t = 0; t < L; t++) { let s = 0; for (let k = 0; k < D; k++) s += Q[qi * D + k] * K[t * D + k]; sc[t] = s * scale; }
        exact.push(softmax(sc)); exactTop.push(topk(sc, TOPK));
      }
      const acc = Array.from({ length: NQ }, () => new Float64Array(L));
      const scoreErr = new Array(D + 1).fill(0), overlap = new Array(D + 1).fill(0);
      for (let r = 1; r <= D; r++) {
        let errSum = 0, ovSum = 0;
        for (let qi = 0; qi < NQ; qi++) {
          const a = acc[qi], qv = Qv[qi * D + (r - 1)];
          for (let t = 0; t < L; t++) a[t] += qv * P[t * D + (r - 1)];
          const sc = Float64Array.from(a, (v) => v * scale), sm = softmax(sc);
          let e = 0;
          for (let t = 0; t < L; t++) e += Math.abs(sm[t] - exact[qi][t]);
          errSum += e;
          let ov = 0;
          for (const t of topk(sc, TOPK)) if (exactTop[qi].has(t)) ov++;
          ovSum += ov;
        }
        scoreErr[r] = errSum / NQ; overlap[r] = ovSum / NQ;
      }
      return { sigma, scoreErr, overlap };
    }
    // For V: relative Frobenius error of the rank-r truncation = sqrt(Σ_{i≥r} σ²) / sqrt(Σ σ²).
    function analyzeV(M) {
      const { sigma } = svd(M), total = sigma.reduce((s, v) => s + v * v, 0), relErr = new Array(D + 1).fill(0);
      for (let r = 0; r <= D; r++) { let tail = 0; for (let i = r; i < D; i++) tail += sigma[i] * sigma[i]; relErr[r] = Math.sqrt(tail / total); }
      return { sigma, relErr };
    }
    const storage = (r) => ({ lowrank: L * r + r * D, full: L * D, crossover: (L * D) / (L + D) });
    // 100 at the smallest rank whose overlap ≥ 15/16 (±2), −10 per rank above that window, 0 if the overlap is below 15/16.
    function score(r, overlap) {
      let rStar = -1;
      for (let i = 1; i <= D; i++) if (overlap[i] >= 15) { rStar = i; break; }
      if (rStar < 0 || overlap[r] < 15) return { score: 0, rStar };
      return { score: clamp(100 - 10 * Math.max(0, r - rStar - 2), 0, 100), rStar };
    }
    return { L, D, NQ, TOPK, NOISE, matrices, svd, softmax, topk, analyzeK, analyzeV, storage, score };
  })();

  // ---- pagepurge (level 5): paged eviction and compaction
  calc.pagepurge = (() => {
    const N = 256, PAGE = 16, PAGES = 16, BUDGET = 64;
    // Importance: Exp(1) per token, 3 sink tokens (in the first two pages) × 10, the last 16 tokens × 3, normalized to Σ = 1.
    function weights(seed) {
      const r = rng(seed), w = new Float64Array(N);
      for (let i = 0; i < N; i++) w[i] = -Math.log(1 - r.next());
      const sinks = new Set();
      while (sinks.size < 3) sinks.add(r.int(2 * PAGE));
      for (const i of sinks) w[i] *= 10;
      for (let i = N - PAGE; i < N; i++) w[i] *= 3;
      let s = 0;
      for (const v of w) s += v;
      for (let i = 0; i < N; i++) w[i] /= s;
      return { w, sinks: [...sinks].sort((a, b) => a - b) };
    }
    function auto(w, keep) {
      keep = keep || BUDGET;
      const order = Array.from(w.keys()).sort((i, j) => w[j] - w[i]), ev = new Uint8Array(N).fill(1);
      for (let k = 0; k < keep; k++) ev[order[k]] = 0;
      return ev;
    }
    function stats(w, evicted) {
      const perPage = new Array(PAGES).fill(0);
      let kept = 0, mass = 0, total = 0;
      for (let i = 0; i < N; i++) { total += w[i]; if (!evicted[i]) { kept++; mass += w[i]; perPage[Math.floor(i / PAGE)]++; } }
      const pagesFreed = perPage.filter((c) => c === 0).length;
      return { kept, mass: total ? mass / total : 0, perPage, pagesFreed, pagesUsed: PAGES - pagesFreed, pagesNeeded: Math.ceil(kept / PAGE) };
    }
    // Fewest copies that reach ceil(kept/16) pages: keep the fullest pages in place, move everything else.
    function minCopies(evicted) {
      const st = stats(new Float64Array(N).fill(1), evicted), sorted = st.perPage.slice().sort((a, b) => b - a);
      let stay = 0;
      for (let i = 0; i < st.pagesNeeded; i++) stay += sorted[i];
      return st.kept - stay;
    }
    function finish(layout, copies, evicted, moves) {
      const perPage = new Array(PAGES).fill(0);
      for (let s = 0; s < N; s++) if (layout[s] >= 0) perPage[Math.floor(s / PAGE)]++;
      const pagesFreed = perPage.filter((c) => c === 0).length;
      return { layout, copies, moves: moves || [], perPage, pagesFreed, pagesUsed: PAGES - pagesFreed, minCopies: minCopies(evicted) };
    }
    // HOLE-FILL: repeatedly move one survivor from the sparsest non-empty page into the fullest non-full page.
    function holeFill(evicted) {
      const lay = new Int16Array(N), moves = [];
      for (let i = 0; i < N; i++) lay[i] = evicted[i] ? -1 : i;
      const counts = new Array(PAGES).fill(0);
      for (let i = 0; i < N; i++) if (lay[i] >= 0) counts[Math.floor(i / PAGE)]++;
      const kept = counts.reduce((a, b) => a + b, 0), target = Math.ceil(kept / PAGE);
      while (counts.filter((c) => c > 0).length > target) {
        let src = -1, dst = -1;
        for (let p = 0; p < PAGES; p++) if (counts[p] > 0 && (src < 0 || counts[p] < counts[src])) src = p;
        for (let p = 0; p < PAGES; p++) if (p !== src && counts[p] > 0 && counts[p] < PAGE && (dst < 0 || counts[p] > counts[dst])) dst = p;
        if (dst < 0) break;
        let from = -1, to = -1;
        for (let s = src * PAGE; s < (src + 1) * PAGE; s++) if (lay[s] >= 0) { from = s; break; }
        for (let s = dst * PAGE; s < (dst + 1) * PAGE; s++) if (lay[s] < 0) { to = s; break; }
        lay[to] = lay[from]; lay[from] = -1; counts[src]--; counts[dst]++; moves.push([from, to]);
      }
      return finish(lay, moves.length, evicted, moves);
    }
    // IN-ORDER: survivors written sequentially in token order; a copy for every survivor whose slot changes.
    function inOrder(evicted) {
      const lay = new Int16Array(N).fill(-1), moves = [];
      let slot = 0;
      for (let i = 0; i < N; i++) if (!evicted[i]) { lay[slot] = i; if (slot !== i) moves.push([i, slot]); slot++; }
      return finish(lay, moves.length, evicted, moves);
    }
    function score(kept, pagesUsed, copies, minC) {
      if (kept > BUDGET || kept === 0) return 0;
      if (pagesUsed !== Math.ceil(kept / PAGE)) return 40;
      return copies <= 1.2 * minC + 1e-9 ? 100 : 70;
    }
    return { N, PAGE, PAGES, BUDGET, weights, auto, stats, minCopies, holeFill, inOrder, score };
  })();

  // ======================================================================= DOM helpers (browser only)

  const BASE_SEED = 20240601;
  let cssInjected = false;

  const CSS = `
.cyc-quest { --qc: var(--role-color, var(--quest-color, var(--accent, #b8892e))); --qfg: var(--fg, var(--ink, #2a211b)); --qpanel: var(--panel, var(--paper-2, #ebdfc8)); --qdim: var(--ink-2, var(--ink-dim, #5a4a3d)); --qline: var(--line, #b89f78); --qpaper: var(--paper, #f3eadb); --qsunk: var(--paper-3, #e0d2b6); --qgood: var(--good, #4f7f3f); --qbad: var(--bad, #a7403a); --qwarn: var(--rose, #b05a4c);
  color: var(--qfg); display: flex; flex-direction: column; gap: 12px; max-width: 100%; }
.cyc-quest * { box-sizing: border-box; }
.cyc-quest canvas { display: block; max-width: 100%; border: 1px solid var(--qline); background: var(--qpanel); }
.cyc-quest .cyc-q-row { display: flex; flex-wrap: wrap; gap: 12px; align-items: flex-start; }
.cyc-quest .cyc-q-col { display: flex; flex-direction: column; gap: 8px; min-width: 0; }
.cyc-quest .cyc-q-grow { flex: 1 1 260px; }
.cyc-quest .cyc-q-panel { background: var(--qpanel); border: 1px solid var(--qline); padding: 10px 12px; }
.cyc-quest .cyc-q-title { font-size: .72em; letter-spacing: .08em; font-variant: small-caps; color: var(--qc); margin: 0 0 6px; }
.cyc-quest .cyc-q-actions { display: flex; flex-wrap: wrap; gap: 8px; align-items: center; }
.cyc-quest .cyc-q-btn { font: inherit; font-size: .85em; letter-spacing: .04em; padding: 8px 12px; border: 1px solid var(--qc); border-radius: 2px; background: var(--qpanel); color: var(--qfg); cursor: pointer; }
.cyc-quest .cyc-q-btn:hover { filter: brightness(1.06); }
.cyc-quest .cyc-q-btn:active { transform: translateY(1px); }
.cyc-quest .cyc-q-btn:disabled { opacity: .45; cursor: not-allowed; filter: grayscale(.6); }
.cyc-quest .cyc-q-btn.primary { background: var(--qc); color: var(--qpaper); font-weight: bold; }
.cyc-quest .cyc-q-btn.on { background: var(--qc); color: var(--qpaper); }
.cyc-quest .cyc-q-readout { display: grid; grid-template-columns: auto 1fr; gap: 2px 14px; font-variant-numeric: tabular-nums; }
.cyc-quest .cyc-q-readout dt { color: var(--qdim); margin: 0; }
.cyc-quest .cyc-q-readout dd { margin: 0; color: var(--qfg); }
.cyc-quest .cyc-q-readout dd.hi { color: var(--qc); }
.cyc-quest .cyc-q-readout dd.good { color: var(--qgood); }
.cyc-quest .cyc-q-readout dd.bad { color: var(--qbad); }
.cyc-quest .cyc-q-status { min-height: 1.4em; color: var(--qdim); }
.cyc-quest .cyc-q-status.ok { color: var(--qgood); }
.cyc-quest .cyc-q-status.err { color: var(--qbad); }
.cyc-quest .cyc-q-result { border-left: 4px solid var(--qc); padding: 6px 10px; background: var(--qsunk); }
.cyc-quest input[type=range] { width: 100%; accent-color: var(--qc); }
.cyc-quest input[type=text] { font: inherit; padding: 6px 8px; width: 9.5em; background: var(--qpaper); color: var(--qfg); border: 1px solid var(--qline); }
.cyc-quest label { cursor: pointer; }
.cyc-quest table { border-collapse: collapse; font-variant-numeric: tabular-nums; width: 100%; }
.cyc-quest th, .cyc-quest td { text-align: right; padding: 3px 8px; border-bottom: 1px solid var(--qline); }
.cyc-quest th:first-child, .cyc-quest td:first-child { text-align: left; }
.cyc-quest tr.chosen td { color: var(--qc); font-weight: bold; }
.cyc-quest code, .cyc-quest .mono { font-family: var(--font-mono, "JetBrains Mono", ui-monospace, Menlo, monospace); word-break: break-all; }
/* shannon */
.cyc-quest .cyc-q-reveal { display: flex; flex-wrap: wrap; gap: 3px; font-size: 1.35em; line-height: 1; }
.cyc-quest .cyc-q-cell { display: inline-flex; flex-direction: column; align-items: center; width: 1.1em; }
.cyc-quest .cyc-q-cell b { font-weight: normal; border-bottom: 2px solid var(--qline); min-height: 1.1em; }
.cyc-quest .cyc-q-cell i { font-style: normal; font-size: .5em; color: var(--qdim); min-height: 1.2em; }
.cyc-quest .cyc-q-cell.cur b { border-bottom-color: var(--qc); color: var(--qc); animation: cycq-blink 1.6s ease-in-out infinite; }
.cyc-quest .cyc-q-cell.hard i { color: var(--qwarn); }
@keyframes cycq-blink { 0%, 100% { opacity: 1; } 50% { opacity: .45; } }
@media (prefers-reduced-motion: reduce) { .cyc-quest * { animation: none !important; } }
.cyc-quest .cyc-q-keys { display: flex; flex-wrap: wrap; gap: 4px; }
.cyc-quest .cyc-q-key { font: inherit; width: 2em; height: 2em; border: 1px solid var(--qline); background: var(--qpanel); color: var(--qfg); cursor: pointer; }
.cyc-quest .cyc-q-key.space { width: 6em; }
.cyc-quest .cyc-q-key.tried { text-decoration: line-through; opacity: .35; cursor: not-allowed; }
/* hufftree */
.cyc-quest .cyc-q-tiles { display: flex; flex-wrap: wrap; gap: 8px; }
.cyc-quest .cyc-q-tile { font: inherit; min-width: 3.4em; padding: 6px 8px; border: 1px solid var(--qline); background: var(--qpanel); color: var(--qfg); cursor: pointer; }
.cyc-quest .cyc-q-tile b { display: block; font-size: 1.2em; }
.cyc-quest .cyc-q-tile.sel { background: var(--qc); color: var(--qpaper); }
.cyc-quest .cyc-q-bar { position: relative; height: 16px; border: 1px solid var(--qline); background: var(--qsunk); }
.cyc-quest .cyc-q-bar > i { position: absolute; left: 0; top: 0; bottom: 0; background: var(--qc); }
.cyc-quest .cyc-q-bar > i.over { background: var(--qbad); }
/* waterfill */
.cyc-quest .cyc-q-band { display: grid; grid-template-columns: 4.5em 1fr 5.5em; gap: 8px; align-items: center; }
.cyc-quest .cyc-q-band .var { display: flex; flex-direction: column; gap: 2px; }
.cyc-quest .cyc-q-band .var i { display: block; height: 4px; background: var(--qc); opacity: .7; }
.cyc-quest .cyc-q-band .var small { color: var(--qdim); font-size: .75em; }
/* phrasebook */
.cyc-quest .cyc-q-frags { display: flex; flex-wrap: wrap; gap: 2px; line-height: 1.6; }
.cyc-quest .cyc-q-frag { font: inherit; padding: 0 3px; border: 1px solid var(--qline); background: transparent; color: var(--qfg); cursor: pointer; white-space: pre; }
.cyc-quest .cyc-q-frag.on { background: var(--qc); color: var(--qpaper); border-color: var(--qc); }
.cyc-quest .cyc-q-frag:disabled { opacity: .4; cursor: not-allowed; }
.cyc-quest .cyc-q-toks { line-height: 1.7; }
.cyc-quest .cyc-q-tok { white-space: pre; padding: 1px 0; }
.cyc-quest .cyc-q-tok.lit { color: var(--qdim); }
.cyc-quest .cyc-q-tok.match { background: var(--qsunk); color: var(--qgood); outline: 1px solid var(--qgood); }
.cyc-quest .cyc-q-tok.dict { background: var(--qsunk); color: var(--qc); outline: 1px solid var(--qc); }
/* blockpack */
.cyc-quest .cyc-q-seg { display: flex; flex-wrap: wrap; gap: 4px; align-items: center; }
.cyc-quest .cyc-q-seg span { color: var(--qdim); margin-right: 4px; }
.cyc-quest .cyc-q-budget { font-weight: bold; }
.cyc-quest .cyc-q-budget.good { color: var(--qgood); }
.cyc-quest .cyc-q-budget.bad { color: var(--qbad); }
/* pagepurge */
.cyc-quest .cyc-q-grid16 { display: grid; grid-template-columns: 3.4em repeat(16, minmax(0, 1fr)); gap: 2px; align-items: center; max-width: 560px; }
.cyc-quest .cyc-q-pagelabel { font-size: .75em; color: var(--qdim); white-space: nowrap; }
.cyc-quest .cyc-q-tok { aspect-ratio: 1 / 1; width: 100%; padding: 0; border: 1px solid var(--qline); background: var(--qpanel); cursor: pointer; }
.cyc-quest .cyc-q-tok.ev { opacity: .18; }
.cyc-quest .cyc-q-tok.moved { outline: 2px solid var(--qgood); outline-offset: -2px; }
.cyc-quest .cyc-q-tok.empty { background: var(--qsunk); cursor: default; }
`;

  function injectCSS() {
    if (cssInjected || typeof document === 'undefined') return;
    if (document.getElementById('cyc-quest-style')) { cssInjected = true; return; }
    const st = document.createElement('style');
    st.id = 'cyc-quest-style';
    st.textContent = CSS;
    document.head.appendChild(st);
    cssInjected = true;
  }

  function el(tag, attrs, ...kids) {
    const n = document.createElement(tag);
    if (attrs) for (const [k, v] of Object.entries(attrs)) {
      if (v == null || v === false) continue;
      if (k === 'class') n.className = v;
      else if (k === 'text') n.textContent = v;
      else if (k === 'html') n.innerHTML = v;
      else if (k === 'style') n.style.cssText = v;
      else if (k.startsWith('on') && typeof v === 'function') n.addEventListener(k.slice(2), v);
      else n.setAttribute(k, v === true ? '' : v);
    }
    for (const k of kids.flat(Infinity)) if (k != null && k !== false) n.appendChild(typeof k === 'string' || typeof k === 'number' ? document.createTextNode(String(k)) : k);
    return n;
  }

  function btn(id, label, onclick, cls) {
    return el('button', { type: 'button', id, class: 'cyc-q-btn' + (cls ? ' ' + cls : ''), onclick, text: label });
  }

  function panel(title, ...kids) {
    return el('div', { class: 'cyc-q-panel' }, title ? el('h4', { class: 'cyc-q-title', text: title }) : null, ...kids);
  }

  // Readout: a <dl> of label/value pairs with a `set(key, value, cls)` updater.
  function readout(rows) {
    const dl = el('dl', { class: 'cyc-q-readout' });
    const dds = {};
    for (const [key, label] of rows) {
      dl.appendChild(el('dt', { text: label }));
      dds[key] = dl.appendChild(el('dd', { text: '' }));
    }
    dl.set = (key, value, cls) => { const d = dds[key]; if (!d) return; d.textContent = value; d.className = cls || ''; };
    return dl;
  }

  // devicePixelRatio-aware canvas with a CSS-pixel coordinate system.
  function mkCanvas(cssW, cssH, id) {
    const c = document.createElement('canvas');
    const dpr = Math.max(1, Math.min(3, (typeof window !== 'undefined' && window.devicePixelRatio) || 1));
    c.width = Math.round(cssW * dpr); c.height = Math.round(cssH * dpr);
    c.style.width = cssW + 'px'; c.style.maxWidth = '100%'; c.style.height = 'auto'; // keeps the aspect ratio when squeezed
    if (id) c.id = id;
    const g = c.getContext('2d');
    g.setTransform(dpr, 0, 0, dpr, 0, 0);
    c.cssW = cssW; c.cssH = cssH; c.dpr = dpr; c.g = g;
    return c;
  }

  function clearCanvas(c, bg) {
    c.g.setTransform(c.dpr, 0, 0, c.dpr, 0, 0);
    c.g.fillStyle = bg || TK.bg;
    c.g.fillRect(0, 0, c.cssW, c.cssH);
  }

  // Draw a gray image {w,h,data 0..255} or an RGBA {w,h,data} pixelated into a canvas region.
  function drawImageData(c, img, x, y, w, h) {
    const off = document.createElement('canvas');
    off.width = img.w; off.height = img.h;
    const og = off.getContext('2d');
    const id = og.createImageData(img.w, img.h);
    if (img.data.length === img.w * img.h * 4) id.data.set(img.data);
    else for (let i = 0; i < img.w * img.h; i++) { const v = img.data[i]; id.data[i * 4] = v; id.data[i * 4 + 1] = v; id.data[i * 4 + 2] = v; id.data[i * 4 + 3] = 255; }
    og.putImageData(id, 0, 0);
    c.g.imageSmoothingEnabled = false;
    c.g.drawImage(off, x, y, w, h);
  }

  function cssVar(elm, name, fallback) {
    try { const v = getComputedStyle(elm).getPropertyValue(name).trim(); return v || fallback; } catch (e) { return fallback; }
  }

  // Design tokens (docs/DESIGN.md) for canvas drawing; refreshed from the container's CSS variables in render().
  const TK = { bg: '#ebdfc8', paper: '#f3eadb', sunk: '#e0d2b6', ink: '#2a211b', ink2: '#5a4a3d', line: '#b89f78', gold: '#b8892e', teal: '#246b66', sage: '#6f8f4e', rose: '#b05a4c', plum: '#6b4a7a', good: '#4f7f3f', bad: '#a7403a', mono: '"JetBrains Mono", ui-monospace, Menlo, monospace' };
  function hexA(hex, a) {
    const m = /^#([0-9a-f]{6})$/i.exec(String(hex).trim()); const n = m ? parseInt(m[1], 16) : 0x2a211b;
    return 'rgba(' + (n >> 16) + ',' + ((n >> 8) & 255) + ',' + (n & 255) + ',' + a + ')';
  }
  const hexRGB = (hex) => { const m = /^#([0-9a-f]{6})$/i.exec(String(hex).trim()); const n = m ? parseInt(m[1], 16) : 0; return [n >> 16, (n >> 8) & 255, n & 255]; };
  function refreshTokens(elm) {
    const read = (name, key) => { TK[key] = cssVar(elm, name, TK[key]); };
    read('--paper-2', 'bg'); read('--paper', 'paper'); read('--paper-3', 'sunk'); read('--ink', 'ink'); read('--ink-2', 'ink2'); read('--line', 'line');
    read('--gold', 'gold'); read('--teal', 'teal'); read('--sage', 'sage'); read('--rose', 'rose'); read('--plum', 'plum'); read('--good', 'good'); read('--bad', 'bad');
    read('--font-mono', 'mono');
  }

  const fmt = (v, d) => (Number.isFinite(v) ? v.toFixed(d == null ? 2 : d) : '—');
  const fmtBig = (v) => (v >= 1000 ? Math.round(v).toLocaleString('en-US') : v >= 10 ? v.toFixed(1) : v.toFixed(2));
  const roleColorFor = (q) => {
    try {
      const f = q && q.fighters && CYC.byId && CYC.byId[q.fighters[0]];
      return (f && CYC.ROLES && CYC.ROLES[f.role] && CYC.ROLES[f.role].color) || null;
    } catch (e) { return null; }
  };

  // ======================================================================= renderers

  // ---- shannon: Guess the next letter
  function renderShannon(ctx) {
    const S = calc.shannon, root = ctx.root;
    const r = rng(ctx.seed);
    const sentence = S.SENTENCES[r.int(S.SENTENCES.length)];
    const st = { pos: 0, tried: new Set(), log: [], total: 0 };

    const reveal = el('div', { class: 'cyc-q-reveal', id: 'quest-shannon-reveal' });
    const keys = el('div', { class: 'cyc-q-keys', id: 'quest-shannon-keys' });
    const keyBtns = {};
    for (const ch of S.ALPHABET) {
      keyBtns[ch] = keys.appendChild(el('button', { type: 'button', class: 'cyc-q-key' + (ch === ' ' ? ' space' : ''), id: 'quest-shannon-key-' + (ch === ' ' ? 'space' : ch), text: ch === ' ' ? '␣ space' : ch, onclick: () => guess(ch) }));
    }
    const input = el('input', { type: 'text', maxlength: 1, id: 'quest-shannon-input', autocomplete: 'off', spellcheck: 'false', placeholder: 'a–z / space', 'aria-label': 'Your guess for the next letter',
      onkeydown: (e) => { if (e.key === 'Enter') { e.preventDefault(); guess(input.value); } } });
    const submit = btn('quest-shannon-submit', 'GUESS', () => guess(input.value));
    const status = el('div', { class: 'cyc-q-status', id: 'quest-shannon-status', text: 'Guess the first letter.' });
    const ro = readout([['done', 'letters revealed'], ['total', 'total bits'], ['bpl', 'bits / letter'], ['score', 'score if you stopped now']]);
    const bars = mkCanvas(440, 118, 'quest-shannon-bars');
    const lock = btn('quest-shannon-lock', 'LOCK IN', finish, 'primary'); lock.disabled = true;
    const fresh = btn('quest-shannon-newround', 'NEW ROUND', () => ctx.newRound());
    const result = el('div', { class: 'cyc-q-result', id: 'quest-shannon-result', style: 'display:none' });

    root.append(
      panel('hidden sentence', reveal, status),
      el('div', { class: 'cyc-q-row' },
        el('div', { class: 'cyc-q-col cyc-q-grow' }, el('div', { class: 'cyc-q-actions' }, input, submit), keys),
        el('div', { class: 'cyc-q-col cyc-q-grow' }, panel('readout', ro), bars)),
      result,
      el('div', { class: 'cyc-q-actions' }, lock, fresh));

    function guess(ch) {
      if (st.pos >= sentence.length) return;
      ch = (ch || '').toLowerCase();
      input.value = '';
      if (ch.length !== 1 || !S.ALPHABET.includes(ch)) { setStatus('Letters a–z or a space.', 'err'); return; }
      if (st.tried.has(ch)) { setStatus(`"${ch}" is already crossed out.`, 'err'); return; }
      if (ch === sentence[st.pos]) {
        const g = st.tried.size + 1, cost = S.cost(g);
        st.log.push({ ch, g, cost, tried: [...st.tried] });
        st.total += cost; st.pos++; st.tried = new Set();
        setStatus(`"${ch === ' ' ? 'space' : ch}" in ${g} guess${g > 1 ? 'es' : ''} → ${fmt(cost)} bits`, 'ok');
      } else {
        st.tried.add(ch);
        setStatus(`not "${ch === ' ' ? 'space' : ch}" (${st.tried.size} wrong so far → ${fmt(S.cost(st.tried.size + 1))} bits if the next one hits)`, 'err');
      }
      update();
      if (st.pos >= sentence.length) complete(); else try { input.focus(); } catch (e) { /* ignore */ }
    }
    function setStatus(t, cls) { status.textContent = t; status.className = 'cyc-q-status' + (cls ? ' ' + cls : ''); }

    function update() {
      reveal.innerHTML = '';
      for (let i = 0; i < sentence.length; i++) {
        const done = i < st.pos, e = st.log[i];
        const cell = el('span', { class: 'cyc-q-cell' + (i === st.pos ? ' cur' : '') + (e && e.g >= 4 ? ' hard' : '') },
          el('b', { text: done ? (sentence[i] === ' ' ? '␣' : sentence[i]) : (i === st.pos ? '?' : '_') }),
          el('i', { text: done ? String(e.g) : '' }));
        if (e) cell.title = `${e.g} guess${e.g > 1 ? 'es' : ''}${e.tried.length ? ': tried ' + e.tried.map((c) => (c === ' ' ? '␣' : c)).join(' ') : ''} → ${fmt(e.cost)} bits`;
        reveal.appendChild(cell);
      }
      for (const ch of S.ALPHABET) { const b = keyBtns[ch]; const t = st.tried.has(ch); b.classList.toggle('tried', t); b.disabled = t || st.pos >= sentence.length; }
      const bpl = st.pos ? st.total / st.pos : NaN;
      ro.set('done', `${st.pos} / ${sentence.length}`);
      ro.set('total', `${fmt(st.total)} bits`);
      ro.set('bpl', st.pos ? `${fmt(bpl)} bits/letter` : '—', 'hi');
      ro.set('score', st.pos ? `${Math.round(S.score(bpl))} / 100` : '—');
      drawBars(bpl);
    }

    function drawBars(yours) {
      clearCanvas(bars);
      const g = bars.g, rows = [
        ['fixed 5-bit code', S.REFS.fixed, TK.line],
        ['letter frequencies', S.REFS.freq, TK.ink2],
        ['you', yours, cssVar(root, '--qc', TK.gold)],
        ['Shannon’s humans ≈', S.REFS.human, TK.good],
      ];
      const x0 = 150, w = bars.cssW - x0 - 54, maxV = 5;
      g.font = `13px ${TK.mono}`; g.textBaseline = 'middle';
      rows.forEach((row, i) => {
        const y = 14 + i * 26;
        g.fillStyle = TK.ink2; g.textAlign = 'right'; g.fillText(row[0], x0 - 8, y);
        g.fillStyle = TK.sunk; g.fillRect(x0, y - 8, w, 16);
        if (Number.isFinite(row[1])) {
          g.fillStyle = row[2]; g.fillRect(x0, y - 8, w * clamp(row[1] / maxV, 0, 1), 16);
          g.fillStyle = TK.ink; g.textAlign = 'left'; g.fillText(fmt(row[1]) + ' b', x0 + w + 6, y);
        } else { g.fillStyle = TK.line; g.textAlign = 'left'; g.fillText('—', x0 + w + 6, y); }
      });
    }

    let final = null;
    function complete() {
      const bpl = st.total / sentence.length, score = S.score(bpl);
      final = { score, bpl };
      result.style.display = '';
      result.innerHTML = '';
      result.append(
        el('div', { html: `<b>${fmt(st.total)} bits for ${sentence.length} letters = ${fmt(bpl)} bits/letter.</b> Fixed code 4.70, letter frequencies 4.14, Shannon’s human guessers ≈ 1.0.` }),
        el('div', { text: `Hardest letters: ${st.log.map((e, i) => ({ e, i })).filter((o) => o.e.g >= 3).slice(0, 8).map((o) => `"${o.e.ch === ' ' ? '␣' : o.e.ch}"@${o.i + 1} (${o.e.g})`).join(', ') || 'none above 2 guesses'}.` }),
        el('div', { text: `Score: ${Math.round(score)} / 100 (100 at ≤ 1.5 bits/letter, 0 at 4.7).` }));
      setStatus('Sentence complete. LOCK IN to record it.', 'ok');
      lock.disabled = false;
    }
    function finish() {
      if (!final) return;
      lock.disabled = true;
      ctx.complete({
        score: Math.round(final.score),
        summary: `${fmt(final.bpl)} bits/letter on "${sentence}" (fixed code 4.70, frequencies 4.14, humans ≈ 1.0).`,
        stats: { bitsPerLetter: final.bpl, totalBits: st.total, letters: sentence.length, sentence, guesses: st.log.map((e) => e.g) },
      });
    }
    update();
  }

  // ---- hufftree: Build the tree
  function renderHufftree(ctx) {
    const H = calc.hufftree, root = ctx.root;
    const opt = H.optimal(), ent = H.entropy();
    let nextId = 0;
    const mkLeaf = (i) => ({ id: nextId++, sym: H.SYMBOLS[i], idx: i, count: H.COUNTS[i], kids: null });
    const st = { forest: H.COUNTS.map((_, i) => mkLeaf(i)), history: [], sel: null };

    const tiles = el('div', { class: 'cyc-q-tiles', id: 'quest-hufftree-tiles' });
    const canvas = mkCanvas(520, 300, 'quest-hufftree-tree');
    const kraftBar = el('div', { class: 'cyc-q-bar', id: 'quest-hufftree-kraft' }, el('i'));
    const kraftLabel = el('div', { class: 'mono', id: 'quest-hufftree-kraft-label' });
    const ro = readout([['mean', 'mean code length'], ['opt', 'Huffman optimum'], ['H', 'entropy H'], ['codes', 'codes']]);
    const status = el('div', { class: 'cyc-q-status', id: 'quest-hufftree-status', text: 'Click two tiles to merge them.' });
    const undo = btn('quest-hufftree-undo', 'UNDO', doUndo);
    const lock = btn('quest-hufftree-lock', 'LOCK IN', finish, 'primary'); lock.disabled = true;
    const fresh = btn('quest-hufftree-newround', 'RESET', () => ctx.newRound());
    const result = el('div', { class: 'cyc-q-result', id: 'quest-hufftree-result', style: 'display:none' });

    root.append(
      panel('nodes on the table (click two to merge)', tiles, status),
      el('div', { class: 'cyc-q-row' },
        el('div', { class: 'cyc-q-col cyc-q-grow' }, canvas),
        el('div', { class: 'cyc-q-col cyc-q-grow' }, panel('readout', ro), panel('Kraft sum Σ 2^−length', kraftBar, kraftLabel))),
      result,
      el('div', { class: 'cyc-q-actions' }, undo, lock, fresh));

    function leaves(node, depth, out) { if (node.kids) { leaves(node.kids[0], depth + 1, out); leaves(node.kids[1], depth + 1, out); } else out.push({ node, depth }); return out; }
    function codes(node, prefix, out) { if (node.kids) { codes(node.kids[0], prefix + '0', out); codes(node.kids[1], prefix + '1', out); } else out[node.idx] = prefix; return out; }

    function onTile(node) {
      if (st.sel === null) { st.sel = node.id; }
      else if (st.sel === node.id) { st.sel = null; }
      else {
        const a = st.forest.find((n) => n.id === st.sel), b = node;
        st.history.push(st.forest.slice());
        st.forest = st.forest.filter((n) => n !== a && n !== b);
        st.forest.push({ id: nextId++, sym: null, count: a.count + b.count, kids: [a, b] });
        st.sel = null;
        status.textContent = `Merged ${a.count} + ${b.count} → ${a.count + b.count}.`;
        status.className = 'cyc-q-status';
      }
      update();
    }
    function doUndo() {
      if (!st.history.length) return;
      st.forest = st.history.pop(); st.sel = null;
      result.style.display = 'none'; lock.disabled = true;
      status.textContent = 'Undone.'; status.className = 'cyc-q-status';
      update();
    }

    function update() {
      tiles.innerHTML = '';
      for (const n of st.forest) {
        tiles.appendChild(el('button', { type: 'button', class: 'cyc-q-tile' + (st.sel === n.id ? ' sel' : ''), id: 'quest-hufftree-tile-' + n.id, onclick: () => onTile(n) },
          el('b', { text: String(n.count) }), el('span', { text: n.sym || `(${leaves(n, 0, []).map((l) => l.node.sym).join('')})` })));
      }
      const complete = st.forest.length === 1;
      const lengths = new Array(H.COUNTS.length).fill(0);
      for (const rootNode of st.forest) for (const { node, depth } of leaves(rootNode, 0, [])) lengths[node.idx] = depth;
      const mean = H.meanLength(lengths);
      ro.set('mean', complete ? `${fmt(mean)} bits/symbol` : `${fmt(mean)}… (partial: unmerged leaves count as 0)`, complete ? 'hi' : '');
      ro.set('opt', `${fmt(opt.mean)} bits/symbol`);
      ro.set('H', `${fmt(ent, 3)} bits/symbol`);
      if (complete) {
        const cs = codes(st.forest[0], '', []);
        ro.set('codes', H.SYMBOLS.map((s, i) => `${s}=${cs[i]}`).join('  '));
        const k = H.kraft(lengths);
        kraftBar.firstChild.style.width = `${Math.min(100, k * 100)}%`;
        kraftBar.firstChild.className = k > 1 + 1e-9 ? 'over' : '';
        kraftLabel.textContent = `Σ 2^−l = ${fmt(k, 3)} (${H.SYMBOLS.map((s, i) => `2^−${lengths[i]}`).join(' + ')})`;
        showResult(mean, lengths, cs);
      } else {
        ro.set('codes', '…');
        kraftBar.firstChild.style.width = '0%';
        kraftLabel.textContent = 'Σ 2^−l = … (defined once everything is in one tree)';
        result.style.display = 'none'; lock.disabled = true;
      }
      undo.disabled = !st.history.length;
      draw(complete ? codes(st.forest[0], '', []) : null);
    }

    let final = null;
    function showResult(mean, lengths, cs) {
      const score = H.score(mean, opt.mean);
      final = { score, mean, lengths, cs };
      result.style.display = '';
      result.innerHTML = '';
      const gap = mean - opt.mean;
      result.append(
        el('div', { html: `<b>Mean length ${fmt(mean)} bits/symbol</b> vs Huffman ${fmt(opt.mean)} vs H = ${fmt(ent, 3)}.` }),
        el('div', { text: gap < 1e-9 ? 'Optimal: every merge took the two smallest counts.' : `${fmt(gap)} bits above optimal: a merge took a bigger node than necessary (−25 points per 0.1 bit).` }),
        el('div', { text: `Integer-length tax: ${fmt(opt.mean - ent, 3)} bits/symbol between Huffman and H. Score ${Math.round(score)} / 100.` }));
      lock.disabled = false;
    }
    function finish() {
      if (!final) return;
      lock.disabled = true;
      ctx.complete({ score: Math.round(final.score), summary: `Mean code length ${fmt(final.mean)} bits/symbol (Huffman ${fmt(opt.mean)}, H ${fmt(ent, 3)}).`, stats: { mean: final.mean, optimal: opt.mean, entropy: ent, lengths: final.lengths, codes: final.cs } });
    }

    function draw(cs) {
      clearCanvas(canvas);
      const g = canvas.g, qc = cssVar(root, '--qc', TK.gold);
      const allLeaves = [];
      for (const rt of st.forest) leaves(rt, 0, allLeaves);
      const total = allLeaves.length, colW = canvas.cssW / total;
      let maxDepth = 0;
      for (const l of allLeaves) maxDepth = Math.max(maxDepth, l.depth);
      const rowH = Math.min(58, (canvas.cssH - 50) / Math.max(1, maxDepth));
      let leafI = 0;
      function layout(node, depth) {
        const y = 24 + depth * rowH;
        if (!node.kids) { const x = (leafI + 0.5) * colW; leafI++; return { node, x, y, depth }; }
        const a = layout(node.kids[0], depth + 1), b = layout(node.kids[1], depth + 1);
        return { node, x: (a.x + b.x) / 2, y, depth, kids: [a, b] };
      }
      const laid = st.forest.map((rt) => layout(rt, 0));
      g.font = `12px ${TK.mono}`; g.textAlign = 'center'; g.textBaseline = 'middle';
      function drawNode(L) {
        if (L.kids) {
          L.kids.forEach((K, i) => {
            g.strokeStyle = TK.line; g.lineWidth = 2; g.beginPath(); g.moveTo(L.x, L.y); g.lineTo(K.x, K.y); g.stroke();
            g.fillStyle = TK.ink2; g.fillText(String(i), (L.x + K.x) / 2 + (i ? 8 : -8), (L.y + K.y) / 2);
            drawNode(K);
          });
        }
        const w = L.node.kids ? 34 : 40, h = L.node.kids ? 20 : 30;
        const selected = st.sel === L.node.id, isRoot = st.forest.includes(L.node);
        g.fillStyle = selected ? qc : L.node.kids ? TK.sunk : TK.sunk;
        g.strokeStyle = isRoot ? qc : TK.ink; g.lineWidth = 2;
        g.fillRect(L.x - w / 2, L.y - h / 2, w, h); g.strokeRect(L.x - w / 2, L.y - h / 2, w, h);
        g.fillStyle = selected ? TK.paper : TK.ink;
        if (L.node.kids) g.fillText(String(L.node.count), L.x, L.y);
        else { g.fillText(`${L.node.sym} ${L.node.count}`, L.x, L.y - 6); g.fillStyle = TK.good; g.fillText(cs ? cs[L.node.idx] : '', L.x, L.y + 8); }
      }
      laid.forEach(drawNode);
    }
    update();
  }

  // ---- waterfill: The bit budget
  function renderWaterfill(ctx) {
    const W = calc.waterfill, root = ctx.root;
    const img = W.image(ctx.seed), an = W.analyze(img);
    const maxVar = Math.max(...an.variances);
    let bits = W.uniform();
    const sliders = [], valueEls = [];
    const bandRows = an.variances.map((v, k) => {
      const s = el('input', { type: 'range', min: 0, max: W.MAXBITS, step: 1, value: bits[k], id: `quest-waterfill-band-${k}`, 'aria-label': `bits per coefficient, band ${k}` });
      s.addEventListener('input', () => onSlider(k, parseInt(s.value, 10)));
      sliders.push(s);
      const val = el('span', { class: 'mono', text: '' }); valueEls.push(val);
      const bar = el('i', { style: `width:${Math.max(2, 100 * Math.log10(v + 1) / Math.log10(maxVar + 1)).toFixed(1)}%` });
      return el('div', { class: 'cyc-q-band' },
        el('div', { class: 'var' }, el('b', { text: k === 0 ? 'band 0 (DC+)' : `band ${k}` }), el('small', { text: `σ² ${fmtBig(v)}` }), bar),
        s, val);
    });
    const budget = el('div', { class: 'cyc-q-budget', id: 'quest-waterfill-budget' });
    const orig = mkCanvas(192, 192, 'quest-waterfill-original'), recon = mkCanvas(192, 192, 'quest-waterfill-recon'), optC = mkCanvas(192, 192, 'quest-waterfill-optimal');
    const ro = readout([['mse', 'MSE'], ['psnr', 'PSNR'], ['alloc', 'your allocation (bits/coef, band 0→7)']]);
    const status = el('div', { class: 'cyc-q-status', id: 'quest-waterfill-status', text: 'Each band has 8 coefficients, so one slider step costs 8 bits per block.' });
    const lock = btn('quest-waterfill-lock', 'LOCK IN', finish, 'primary');
    const fresh = btn('quest-waterfill-newround', 'NEW ROUND', () => ctx.newRound());
    const result = el('div', { class: 'cyc-q-result', id: 'quest-waterfill-result', style: 'display:none' });
    const chart = mkCanvas(440, 170, 'quest-waterfill-chart');
    const optWrap = el('div', { class: 'cyc-q-col', style: 'display:none' }, el('div', { class: 'cyc-q-title', text: 'optimal' }), optC);

    root.append(
      el('div', { class: 'cyc-q-row' },
        el('div', { class: 'cyc-q-col cyc-q-grow' }, panel('bits per coefficient, per band', ...bandRows, budget, status)),
        el('div', { class: 'cyc-q-col' },
          el('div', { class: 'cyc-q-row' },
            el('div', { class: 'cyc-q-col' }, el('div', { class: 'cyc-q-title', text: 'original 32×32' }), orig),
            el('div', { class: 'cyc-q-col' }, el('div', { class: 'cyc-q-title', text: 'yours' }), recon),
            optWrap),
          panel('readout', ro))),
      result,
      el('div', { class: 'cyc-q-actions' }, lock, fresh));
    drawImageData(orig, img, 0, 0, 192, 192);

    function onSlider(k, v) {
      const trial = bits.slice(); trial[k] = v;
      if (W.blockBits(trial) > W.BUDGET) {
        sliders[k].value = bits[k];
        status.textContent = `Budget full: take a bit from another band first (${W.BUDGET} bits per block).`;
        status.className = 'cyc-q-status err';
        return;
      }
      bits = trial;
      status.textContent = v === 0 ? `Band ${k} dropped entirely (b = 0 zeroes its coefficients).` : `Band ${k}: Δ = 2·max|coef| / 2^${v} = ${fmt((an.maxAbs[k] * 2) / Math.pow(2, v), 1)}.`;
      status.className = 'cyc-q-status';
      update();
    }
    let cur = null;
    function update() {
      cur = W.reconstruct(an, bits);
      const used = W.blockBits(bits);
      budget.textContent = `${used} / ${W.BUDGET} bits per block (raw block = 64 × 8 = 512)`;
      budget.className = 'cyc-q-budget ' + (used >= W.BUDGET ? 'good' : '');
      bits.forEach((b, k) => { valueEls[k].textContent = `${b} bit${b === 1 ? '' : 's'}`; });
      drawImageData(recon, cur.img, 0, 0, 192, 192);
      ro.set('mse', fmt(cur.mse), 'hi');
      ro.set('psnr', `${fmt(cur.psnr, 1)} dB`);
      ro.set('alloc', bits.join(' '));
    }
    function finish() {
      const optBits = W.optimal(an), optRes = W.reconstruct(an, optBits), th = W.theta(an.variances, W.BUDGET);
      const score = W.score(optRes.mse, cur.mse);
      drawImageData(optC, optRes.img, 0, 0, 192, 192);
      optWrap.style.display = '';
      result.style.display = '';
      result.innerHTML = '';
      result.append(
        el('div', { html: `<b>Your MSE ${fmt(cur.mse)}</b> with [${bits.join(' ')}]; reverse water-filling gives MSE ${fmt(optRes.mse)} with [${optBits.join(' ')}].` }),
        el('div', { text: `Water level θ = ${fmtBig(th.theta)}: real-valued rates ½log₂(σ²/θ) = [${th.rates.map((x) => fmt(x, 1)).join(' ')}], rounded greedily to integers (one bit at a time to the band with the largest σ²/4^b).` }),
        el('div', { text: `Bands under water (σ² < θ) get nothing: ${an.variances.map((v, k) => (v < th.theta ? k : null)).filter((k) => k != null).join(', ') || 'none at this budget'}. Score ${Math.round(score)} / 100 = 100 × optimal MSE / your MSE.` }),
        chart);
      drawChart(th, optBits);
      lock.disabled = true;
      sliders.forEach((s) => { s.disabled = true; });
      ctx.complete({ score: Math.round(score), summary: `MSE ${fmt(cur.mse)} with [${bits.join(' ')}] vs water-filling MSE ${fmt(optRes.mse)} with [${optBits.join(' ')}].`, stats: { bits: bits.slice(), mse: cur.mse, optimalBits: optBits, optimalMse: optRes.mse, theta: th.theta, variances: an.variances.slice() } });
    }
    function drawChart(th, optBits) {
      clearCanvas(chart);
      const g = chart.g, qc = cssVar(root, '--qc', TK.gold);
      const x0 = 36, y0 = chart.cssH - 26, w = chart.cssW - x0 - 10, hgt = y0 - 14;
      const vals = an.variances.map((v) => 0.5 * log2(Math.max(v, 1e-6))), level = 0.5 * log2(th.theta);
      const vmax = Math.max(...vals) + 0.5, vmin = Math.min(level, ...vals) - 0.5;
      const Y = (v) => y0 - hgt * (v - vmin) / (vmax - vmin);
      const bw = w / an.variances.length;
      g.font = `11px ${TK.mono}`; g.textAlign = 'center'; g.textBaseline = 'top';
      vals.forEach((v, k) => {
        const x = x0 + k * bw + 4, top = Y(v);
        g.fillStyle = TK.sunk; g.fillRect(x, top, bw - 8, y0 - top);
        if (v > level) { g.fillStyle = qc; g.fillRect(x, top, bw - 8, Y(level) - top); }
        g.fillStyle = TK.ink; g.fillText(`b${k}`, x + (bw - 8) / 2, y0 + 4);
        g.fillStyle = TK.ink; g.fillText(String(optBits[k]), x + (bw - 8) / 2, Math.min(top + 3, Y(level) - 12));
      });
      g.strokeStyle = TK.teal; g.lineWidth = 2; g.setLineDash([6, 4]); g.beginPath(); g.moveTo(x0, Y(level)); g.lineTo(x0 + w, Y(level)); g.stroke(); g.setLineDash([]);
      g.textAlign = 'right'; g.textBaseline = 'bottom';
      g.fillStyle = hexA(TK.bg, 0.85); g.fillRect(x0 + w - 62, Y(level) - 16, 62, 14);
      g.fillStyle = TK.teal; g.fillText('water θ', x0 + w - 2, Y(level) - 2);
      g.textBaseline = 'top';
      g.save(); g.translate(10, y0 - hgt / 2); g.rotate(-Math.PI / 2); g.fillStyle = TK.ink2; g.textAlign = 'center'; g.fillText('½ log₂ σ²', 0, 0); g.restore();
    }
    update();
  }

  // ---- jnd: Spot the difference
  function renderJnd(ctx) {
    const J = calc.jnd, root = ctx.root;
    const base = J.image(ctx.seed), ycc = J.makeChroma(base);
    let variant = 'luma';
    let st = null;

    const left = mkCanvas(256, 256, 'quest-jnd-left'), right = mkCanvas(256, 256, 'quest-jnd-right');
    left.style.cursor = right.style.cursor = 'pointer';
    left.addEventListener('click', () => answer('L')); right.addEventListener('click', () => answer('R'));
    const same = btn('quest-jnd-same', 'SAME TO ME', () => answer(null));
    const vLuma = btn('quest-jnd-variant-luma', 'LUMA', () => setVariant('luma'));
    const vChroma = btn('quest-jnd-variant-chroma', 'CHROMA', () => setVariant('chroma'));
    const status = el('div', { class: 'cyc-q-status', id: 'quest-jnd-status' });
    const ro = readout([['trial', 'trial'], ['delta', 'current Δ'], ['last', 'last answer'], ['thr', 'your threshold Δ']]);
    const stair = mkCanvas(440, 130, 'quest-jnd-staircase');
    const lock = btn('quest-jnd-lock', 'LOCK IN', finish, 'primary'); lock.disabled = true;
    const fresh = btn('quest-jnd-newround', 'NEW ROUND', () => ctx.newRound());
    const result = el('div', { class: 'cyc-q-result', id: 'quest-jnd-result', style: 'display:none' });

    root.append(
      el('div', { class: 'cyc-q-actions' }, el('span', { text: 'variant:' }), vLuma, vChroma),
      el('div', { class: 'cyc-q-row' },
        el('div', { class: 'cyc-q-col' }, el('div', { class: 'cyc-q-title', text: 'A (click if degraded)' }), left),
        el('div', { class: 'cyc-q-col' }, el('div', { class: 'cyc-q-title', text: 'B (click if degraded)' }), right),
        el('div', { class: 'cyc-q-col cyc-q-grow' }, el('div', { class: 'cyc-q-actions' }, same), status, panel('readout', ro), stair)),
      result,
      el('div', { class: 'cyc-q-actions' }, lock, fresh));

    function setVariant(v) {
      variant = v;
      vLuma.classList.toggle('on', v === 'luma'); vChroma.classList.toggle('on', v === 'chroma');
      st = { r: rng(ctx.seed * 7 + (v === 'luma' ? 1 : 2)), delta: J.START, trial: 0, deltas: [], answers: [], side: null, final: null };
      result.style.display = 'none'; lock.disabled = true;
      nextTrial();
    }
    function render(side) {
      const d = variant === 'luma' ? J.degrade(base, st.delta) : J.degradeChroma(ycc, st.delta);
      const cleanImg = variant === 'luma' ? base : J.ycbcrToRgb(ycc);
      const dirtyImg = variant === 'luma' ? d.img : J.ycbcrToRgb(d.ycc);
      drawImageData(left, side === 'L' ? dirtyImg : cleanImg, 0, 0, 256, 256);
      drawImageData(right, side === 'R' ? dirtyImg : cleanImg, 0, 0, 256, 256);
      return d.mse;
    }
    function nextTrial() {
      st.side = st.r.next() < 0.5 ? 'L' : 'R';
      st.mse = render(st.side);
      ro.set('trial', `${st.trial + 1} / ${J.TRIALS}`);
      ro.set('delta', `${fmt(st.delta, 1)} (quantizer MSE ${fmt(st.mse, 1)})`, 'hi');
      status.textContent = st.trial === 0 ? 'One image is DCT-quantized. Click it, or SAME TO ME.' : status.textContent;
      drawStair();
    }
    function answer(side) {
      if (!st || st.final) return;
      const correct = side !== null && side === st.side;
      st.deltas.push(st.delta); st.answers.push({ delta: st.delta, side, correct });
      const nd = J.nextDelta(st.delta, correct);
      status.textContent = side === null ? `Trial ${st.trial + 1}: gave up at Δ ${fmt(st.delta, 1)} → harsher (Δ ${fmt(nd, 1)}). It was ${st.side === 'L' ? 'A' : 'B'}.`
        : correct ? `Trial ${st.trial + 1}: correct at Δ ${fmt(st.delta, 1)} → gentler (Δ ${fmt(nd, 1)}).`
          : `Trial ${st.trial + 1}: wrong at Δ ${fmt(st.delta, 1)} → harsher (Δ ${fmt(nd, 1)}). It was ${st.side === 'L' ? 'A' : 'B'}.`;
      status.className = 'cyc-q-status ' + (correct ? 'ok' : 'err');
      ro.set('last', correct ? 'correct' : side === null ? 'same to me' : 'wrong', correct ? 'good' : 'bad');
      st.delta = nd; st.trial++;
      if (st.trial >= J.TRIALS) complete(); else nextTrial();
    }
    function complete() {
      const thr = J.threshold(st.deltas);
      const planes = variant === 'luma' ? [base] : [ycc.Cb, ycc.Cr];
      const lossless = J.blockCost(planes, 1), yours = J.blockCost(planes, thr), saved = J.bitsSaved(thr);
      const degr = variant === 'luma' ? J.degrade(base, thr) : J.degradeChroma(ycc, thr);
      st.final = { thr, lossless, yours, saved, mse: degr.mse };
      render(null);
      ro.set('thr', fmt(thr, 1), 'hi');
      ro.set('delta', '—');
      drawStair(thr);
      result.style.display = '';
      result.innerHTML = '';
      result.append(
        el('div', { html: `<b>Threshold Δ ≈ ${fmt(thr, 1)}</b> (${variant.toUpperCase()}: geometric mean of the last 6 step sizes).` }),
        el('div', { text: `Bits saved ≈ log₂(range/Δ) vs log₂(range/1) = log₂ Δ = ${fmt(saved)} bits per coefficient compared with Δ = 1.` }),
        el('div', { text: `At this Δ a lossless block costs ${fmt(lossless, 0)} bits, yours costs ${fmt(yours, 0)} (order-0 entropy of the quantized ${variant === 'luma' ? 'luma' : 'Cb/Cr'} coefficients; ${fmt(100 * (1 - yours / lossless), 0)}% smaller). PSNR at threshold ${fmt(psnr(degr.mse), 1)} dB.` }),
        el('div', { text: variant === 'luma' ? 'Now try CHROMA: the same staircase on Cb/Cr only.' : 'Compare with LUMA: the eye tolerates a far coarser step in chroma.' }));
      status.textContent = 'Staircase finished. Both images now show the original.';
      status.className = 'cyc-q-status ok';
      lock.disabled = false;
    }
    function finish() {
      if (!st || !st.final) return;
      lock.disabled = true;
      const f = st.final;
      ctx.complete({ score: 100, summary: `${variant.toUpperCase()} threshold Δ ≈ ${fmt(f.thr, 1)}: lossless block ${fmt(f.lossless, 0)} bits, yours ${fmt(f.yours, 0)} (${fmt(f.saved)} bits/coef saved vs Δ = 1).`, stats: { variant, threshold: f.thr, deltas: st.deltas.slice(), answers: st.answers.map((a) => a.correct), losslessBlockBits: f.lossless, yourBlockBits: f.yours, bitsSavedPerCoef: f.saved } });
    }
    function drawStair(thr) {
      clearCanvas(stair);
      const g = stair.g, qc = cssVar(root, '--qc', TK.gold);
      const x0 = 40, y0 = stair.cssH - 18, w = stair.cssW - x0 - 10, hgt = y0 - 10;
      const all = st.deltas.concat(st.final ? [] : [st.delta]);
      const lo = Math.log(Math.min(4, ...all)) - 0.2, hi = Math.log(Math.max(60, ...all)) + 0.2;
      const X = (i) => x0 + (w * i) / Math.max(1, J.TRIALS - 1), Y = (d) => y0 - hgt * (Math.log(d) - lo) / (hi - lo);
      g.font = `11px ${TK.mono}`; g.fillStyle = TK.ink2; g.textAlign = 'right'; g.textBaseline = 'middle';
      for (const d of [4, 8, 16, 32, 64]) if (Math.log(d) > lo && Math.log(d) < hi) { g.fillText(String(d), x0 - 4, Y(d)); g.strokeStyle = hexA(TK.line, 0.4); g.beginPath(); g.moveTo(x0, Y(d)); g.lineTo(x0 + w, Y(d)); g.stroke(); }
      g.strokeStyle = qc; g.lineWidth = 2; g.beginPath();
      all.forEach((d, i) => { if (i) g.lineTo(X(i), Y(d)); else g.moveTo(X(i), Y(d)); }); g.stroke();
      st.answers.forEach((a, i) => { g.fillStyle = a.correct ? TK.good : TK.bad; g.fillRect(X(i) - 4, Y(a.delta) - 4, 8, 8); });
      if (!st.final) { g.fillStyle = TK.ink; g.fillRect(X(st.deltas.length) - 4, Y(st.delta) - 4, 8, 8); }
      if (thr) { g.strokeStyle = TK.teal; g.setLineDash([5, 4]); g.beginPath(); g.moveTo(x0, Y(thr)); g.lineTo(x0 + w, Y(thr)); g.stroke(); g.setLineDash([]); g.fillStyle = TK.teal; g.textAlign = 'left'; g.fillText(`threshold ${fmt(thr, 1)}`, x0 + 4, Y(thr) - 8); }
      g.fillStyle = TK.ink2; g.textAlign = 'left'; g.textBaseline = 'bottom'; g.fillText('Δ per trial (log scale); green = seen, red = missed', x0, stair.cssH - 2);
    }
    setVariant('luma');
  }

  // ---- phrasebook: Train the phrasebook
  function renderPhrasebook(ctx) {
    const P = calc.phrasebook, root = ctx.root;
    const selected = []; // fragment texts in pick order
    const fragBtns = new Map(); // text -> [buttons]

    const sampleEls = P.SAMPLES.map((s, i) => {
      const row = el('div', { class: 'cyc-q-frags', id: `quest-phrasebook-sample-${i}` });
      for (const t of P.tokenize(s)) {
        if (/^\s+$/.test(t.text)) { row.appendChild(el('span', { text: t.text })); continue; }
        const b = el('button', { type: 'button', class: 'cyc-q-frag', title: `${t.kind}: ${t.text.length} byte${t.text.length === 1 ? '' : 's'}`, text: t.text, onclick: () => toggle(t.text) });
        if (!fragBtns.has(t.text)) fragBtns.set(t.text, []);
        fragBtns.get(t.text).push(b);
        row.appendChild(b);
      }
      return row;
    });
    const dictLabel = el('div', { class: 'cyc-q-budget', id: 'quest-phrasebook-budget' });
    const dictText = el('code', { id: 'quest-phrasebook-dict' });
    const clear = btn('quest-phrasebook-clear', 'CLEAR', () => { selected.length = 0; updateDict(); });
    const compress = btn('quest-phrasebook-compress', 'COMPRESS', doCompress, 'primary');
    const lock = btn('quest-phrasebook-lock', 'LOCK IN', finish, 'primary'); lock.disabled = true;
    const fresh = btn('quest-phrasebook-newround', 'RESET', () => ctx.newRound());
    const status = el('div', { class: 'cyc-q-status', id: 'quest-phrasebook-status', text: 'Cost model: literal 9 bits, match 20 bits (LZ77, window 4096, min match 3).' });
    const out = el('div', { class: 'cyc-q-col', id: 'quest-phrasebook-output' });

    root.append(
      panel('six sample messages (click fragments to add them to the dictionary)', ...sampleEls),
      panel('your dictionary', dictLabel, dictText, el('div', { class: 'cyc-q-actions' }, clear, compress), status),
      out,
      el('div', { class: 'cyc-q-actions' }, lock, fresh));

    function dictString() { return selected.join(''); }
    function toggle(text) {
      const i = selected.indexOf(text);
      if (i >= 0) selected.splice(i, 1);
      else if (dictString().length + text.length > P.BUDGET) { status.textContent = `"${text}" does not fit: ${dictString().length} + ${text.length} > ${P.BUDGET} bytes.`; status.className = 'cyc-q-status err'; return; }
      else selected.push(text);
      status.className = 'cyc-q-status';
      updateDict();
    }
    function updateDict() {
      const d = dictString();
      dictLabel.textContent = `${d.length} / ${P.BUDGET} bytes · ${selected.length} fragment${selected.length === 1 ? '' : 's'} (deduped)`;
      dictLabel.className = 'cyc-q-budget ' + (d.length > P.BUDGET ? 'bad' : d.length ? 'good' : '');
      dictText.textContent = d || '(empty: the LZ window starts cold)';
      for (const [text, bs] of fragBtns) { const on = selected.includes(text); const fits = on || d.length + text.length <= P.BUDGET; for (const b of bs) { b.classList.toggle('on', on); b.disabled = !fits; } }
    }
    function spans(res) {
      const wrap = el('div', { class: 'cyc-q-toks mono' });
      for (const t of res.tokens) wrap.appendChild(el('span', { class: 'cyc-q-tok ' + (t.type === 'lit' ? 'lit' : t.dict ? 'dict' : 'match'), text: t.text, title: t.type === 'lit' ? `literal: ${res.litBits} bits` : `match ← ${t.dist} back, ${t.len} chars, ${res.matchBits} bits (${t.dict ? 'from the dictionary' : 'from the message itself'})` }));
      return wrap;
    }
    const sizeLine = (label, sv) => el('div', { html: `${label}: <b>${sv.withDict.bits} bits</b> (${sv.withDict.bytes} B; ${sv.withDict.lits} literals, ${sv.withDict.matches} matches, ${sv.withDict.dictMatches} from the dictionary) vs ${sv.without.bits} bits cold → saves ${sv.saved} bits (${fmt(sv.pct, 0)}%)` });

    let final = null;
    function doCompress() {
      const dict = dictString();
      const yours = P.savings(P.TEST, dict), trainer = P.train(P.SAMPLES, P.BUDGET), tr = P.savings(P.TEST, trainer.dict);
      const fy = P.savings(P.FOREIGN, dict), ft = P.savings(P.FOREIGN, trainer.dict);
      const score = P.score(yours.saved, tr.saved);
      final = { dict, yours, trainer, tr, fy, ft, score };
      out.innerHTML = '';
      out.append(
        panel('test message with your dictionary (gold = dictionary match, green = self match, dim = literal)', spans(yours.withDict), sizeLine('yours', yours), el('div', { class: 'cyc-q-status', text: `Raw: ${P.TEST.length} bytes = ${P.TEST.length * 8} bits.` })),
        panel(`greedy trainer's pick (${trainer.dict.length} / ${P.BUDGET} bytes)`,
          el('code', { text: trainer.dict }),
          el('div', { class: 'cyc-q-status', text: `Scored (occurrences − 1) × length over all substrings of length 3–24 seen in ≥ 2 samples; best first, skipping what the dictionary already covers. Top picks: ${trainer.picks.slice(0, 5).map((p) => `"${p.s}" (${p.occ}× → ${p.score})`).join(', ')}.` }),
          spans(tr.withDict), sizeLine('trainer', tr)),
        panel('the foreign message (a different schema from another service)',
          el('div', { html: `yours: ` }, spans(fy.withDict)), sizeLine('yours', fy),
          sizeLine('trainer', ft),
          el('div', { class: 'cyc-q-status', text: 'The dictionary encodes the JSON stream’s habits; on a different stream those habits are almost all wasted bytes.' })),
        el('div', { class: 'cyc-q-result', html: `<b>Score ${Math.round(score)} / 100</b> = 100 × your savings (${yours.saved} bits) / trainer’s savings (${tr.saved} bits), capped at 100.` }));
      status.textContent = 'Re-pick and COMPRESS again, or LOCK IN.'; status.className = 'cyc-q-status ok';
      lock.disabled = false;
    }
    function finish() {
      if (!final) return;
      lock.disabled = true;
      ctx.complete({ score: Math.round(final.score), summary: `Your ${final.dict.length}-byte dictionary saved ${fmt(final.yours.pct, 0)}% on the test message (trainer ${fmt(final.tr.pct, 0)}%); on the foreign message ${fmt(final.fy.pct, 0)}% / ${fmt(final.ft.pct, 0)}%.`, stats: { dict: final.dict, yourBits: final.yours.withDict.bits, coldBits: final.yours.without.bits, trainerBits: final.tr.withDict.bits, trainerDict: final.trainer.dict, foreignPct: final.fy.pct, foreignTrainerPct: final.ft.pct } });
    }
    updateDict();
  }

  // ---- safehouses: VQ codebook placement
  function renderSafehouses(ctx) {
    const V = calc.safehouses, root = ctx.root;
    const pts = V.points(ctx.seed);
    const COLORS = [TK.rose, TK.teal, TK.sage, TK.gold];
    const st = { cents: [], phase: 'place', drag: -1, before: null, after: null, best: null, anim: null };
    const SIZE = 360, PAD = 14;
    const map = mkCanvas(SIZE, SIZE, 'quest-safehouses-map');
    map.style.cursor = 'crosshair'; map.style.touchAction = 'none';
    const vor = document.createElement('canvas'); vor.width = 90; vor.height = 90;
    const toPx = (v) => PAD + ((v + 1) / 2) * (SIZE - 2 * PAD), toMap = (p) => ((p - PAD) / (SIZE - 2 * PAD)) * 2 - 1;

    const ro = readout([['n', 'code vectors placed'], ['dist', 'distortion Σ‖x−c‖²'], ['bits', 'bits per point'], ['book', 'codebook'], ['km', 'after k-means'], ['best', 'best of 10 restarts']]);
    const status = el('div', { class: 'cyc-q-status', id: 'quest-safehouses-status', text: 'Click the map to place 4 code vectors. Drag to move.' });
    const run = btn('quest-safehouses-kmeans', 'RUN K-MEANS', runKmeans, 'primary'); run.disabled = true;
    const best = btn('quest-safehouses-bestof', 'BEST OF 10', runBest); best.disabled = true;
    const reset = btn('quest-safehouses-reset', 'RESET', () => { stopAnim(); Object.assign(st, { cents: [], phase: 'place', before: null, after: null, best: null }); result.style.display = 'none'; status.textContent = 'Cleared.'; update(); });
    const lock = btn('quest-safehouses-lock', 'LOCK IN', finish, 'primary'); lock.disabled = true;
    const fresh = btn('quest-safehouses-newround', 'NEW ROUND', () => { stopAnim(); ctx.newRound(); });
    const result = el('div', { class: 'cyc-q-result', id: 'quest-safehouses-result', style: 'display:none' });

    root.append(
      el('div', { class: 'cyc-q-row' },
        el('div', { class: 'cyc-q-col' }, map),
        el('div', { class: 'cyc-q-col cyc-q-grow' }, panel('readout', ro), status, el('div', { class: 'cyc-q-actions' }, run, best, reset))),
      result,
      el('div', { class: 'cyc-q-actions' }, lock, fresh));
    ctx.onDispose(stopAnim);

    const centArr = () => Float64Array.from(st.cents.flat());
    function hit(x, y) { for (let j = st.cents.length - 1; j >= 0; j--) { const dx = toPx(st.cents[j][0]) - x, dy = toPx(st.cents[j][1]) - y; if (dx * dx + dy * dy < 144) return j; } return -1; }
    function pos(e) { const r = map.getBoundingClientRect(); return [((e.clientX - r.left) / r.width) * SIZE, ((e.clientY - r.top) / r.height) * SIZE]; }
    map.addEventListener('pointerdown', (e) => {
      if (st.phase !== 'place') return;
      const [x, y] = pos(e), j = hit(x, y);
      const capture = () => { try { map.setPointerCapture(e.pointerId); } catch (err) { /* synthetic or unsupported pointer */ } };
      if (j >= 0) { st.drag = j; capture(); }
      else if (st.cents.length < V.K) { st.cents.push([clamp(toMap(x), -1, 1), clamp(toMap(y), -1, 1)]); st.drag = st.cents.length - 1; capture(); update(); }
      else { status.textContent = 'All 4 placed: drag one to move it, or RUN K-MEANS.'; }
    });
    map.addEventListener('pointermove', (e) => { if (st.drag < 0 || st.phase !== 'place') return; const [x, y] = pos(e); st.cents[st.drag] = [clamp(toMap(x), -1, 1), clamp(toMap(y), -1, 1)]; update(); });
    const endDrag = () => { st.drag = -1; };
    map.addEventListener('pointerup', endDrag); map.addEventListener('pointercancel', endDrag);

    function update(assignOverride, centsOverride) {
      const cents = centsOverride || centArr(), k = cents.length / 2;
      const a = k ? (assignOverride || V.assign(pts, cents)) : null;
      const d = k ? V.distortion(pts, cents, a) : NaN;
      ro.set('n', `${k} / ${V.K}`);
      ro.set('dist', k ? fmt(d, 3) : '—', 'hi');
      ro.set('bits', `${V.bitsPerPoint} (vs ${V.rawBitsPerPoint} raw) → ${V.N * V.bitsPerPoint} bits for the stream`);
      ro.set('book', `${V.codebookBytes} bytes (4 × 2 coords × float32)`);
      ro.set('km', st.after ? `${fmt(st.after.distortion, 3)} after ${st.after.iterations} iteration${st.after.iterations === 1 ? '' : 's'}${st.after.converged ? '' : ' (not converged)'}` : '—');
      ro.set('best', st.best ? `${fmt(st.best.distortion, 3)} (runs: ${st.best.runs.map((x) => fmt(x, 2)).join(' ')})` : '—');
      run.disabled = !(st.phase === 'place' && k === V.K);
      best.disabled = st.phase !== 'kmeans-done';
      draw(cents, a);
    }
    function draw(cents, a) {
      clearCanvas(map);
      const g = map.g, k = cents.length / 2;
      if (k) {
        const vg = vor.getContext('2d'), id = vg.createImageData(vor.width, vor.height);
        for (let py = 0; py < vor.height; py++) for (let px = 0; px < vor.width; px++) {
          const x = (px + 0.5) / vor.width * 2 - 1, y = (py + 0.5) / vor.height * 2 - 1;
          let bj = 0, bd = Infinity;
          for (let j = 0; j < k; j++) { const dx = x - cents[2 * j], dy = y - cents[2 * j + 1], dd = dx * dx + dy * dy; if (dd < bd) { bd = dd; bj = j; } }
          const c = COLORS[bj], o = (py * vor.width + px) * 4;
          id.data[o] = parseInt(c.slice(1, 3), 16) * 0.22 + 7; id.data[o + 1] = parseInt(c.slice(3, 5), 16) * 0.22 + 4; id.data[o + 2] = parseInt(c.slice(5, 7), 16) * 0.22 + 15; id.data[o + 3] = 255;
        }
        vg.putImageData(id, 0, 0);
        g.imageSmoothingEnabled = false; g.drawImage(vor, PAD, PAD, SIZE - 2 * PAD, SIZE - 2 * PAD);
      }
      g.strokeStyle = hexA(TK.line, 0.4); g.lineWidth = 1;
      for (let i = 0; i <= 10; i++) { const p = PAD + (i / 10) * (SIZE - 2 * PAD); g.beginPath(); g.moveTo(p, PAD); g.lineTo(p, SIZE - PAD); g.stroke(); g.beginPath(); g.moveTo(PAD, p); g.lineTo(SIZE - PAD, p); g.stroke(); }
      for (let i = 0; i < V.N; i++) {
        const x = toPx(pts[2 * i]), y = toPx(pts[2 * i + 1]);
        if (a) { g.strokeStyle = COLORS[a[i]]; g.globalAlpha = 0.35; g.beginPath(); g.moveTo(x, y); g.lineTo(toPx(cents[2 * a[i]]), toPx(cents[2 * a[i] + 1])); g.stroke(); g.globalAlpha = 1; }
        g.fillStyle = a ? COLORS[a[i]] : TK.ink; g.fillRect(x - 3, y - 3, 6, 6); g.strokeStyle = TK.ink; g.strokeRect(x - 3, y - 3, 6, 6);
      }
      if (st.best && st.phase !== 'place') for (let j = 0; j < V.K; j++) { const x = toPx(st.best.centroids[2 * j]), y = toPx(st.best.centroids[2 * j + 1]); g.strokeStyle = TK.ink; g.lineWidth = 2; g.setLineDash([3, 3]); g.strokeRect(x - 9, y - 9, 18, 18); g.setLineDash([]); }
      for (let j = 0; j < k; j++) {
        const x = toPx(cents[2 * j]), y = toPx(cents[2 * j + 1]);
        g.fillStyle = COLORS[j]; g.fillRect(x - 8, y - 8, 16, 16); g.strokeStyle = TK.ink; g.lineWidth = 2; g.strokeRect(x - 8, y - 8, 16, 16);
        g.fillStyle = TK.ink; g.font = `bold 11px ${TK.mono}`; g.textAlign = 'center'; g.textBaseline = 'middle'; g.fillText(String(j), x, y + 1);
      }
      g.fillStyle = TK.ink2; g.font = `11px ${TK.mono}`; g.textAlign = 'left'; g.textBaseline = 'bottom';
      g.fillText(st.phase === 'place' ? `${k}/${V.K} code vectors` : st.phase === 'kmeans' ? 'Lloyd iterations…' : 'dashed = best of 10', PAD + 2, SIZE - 2);
    }
    function stopAnim() { if (st.anim) { clearTimeout(st.anim); st.anim = null; } }
    function runKmeans() {
      if (st.cents.length !== V.K) return;
      const start = centArr();
      st.before = { centroids: start, distortion: V.distortion(pts, start) };
      const res = V.lloyd(pts, start, 20);
      st.phase = 'kmeans'; run.disabled = true;
      let i = 0;
      const tick = () => {
        const c = res.history[i];
        update(null, c);
        status.textContent = `Lloyd iteration ${i} / ${res.history.length - 1}: distortion ${fmt(V.distortion(pts, c), 3)}`;
        if (i < res.history.length - 1) { i++; st.anim = setTimeout(tick, 320); }
        else {
          st.anim = null; st.after = res; st.cents = Array.from({ length: V.K }, (_, j) => [res.centroids[2 * j], res.centroids[2 * j + 1]]);
          st.phase = 'kmeans-done';
          status.textContent = `k-means ${res.converged ? 'converged' : 'stopped'} after ${res.iterations} iteration${res.iterations === 1 ? '' : 's'}: ${fmt(st.before.distortion, 3)} → ${fmt(res.distortion, 3)}. Now BEST OF 10.`;
          status.className = 'cyc-q-status ok';
          update();
        }
      };
      tick();
    }
    function runBest() {
      st.best = V.bestOf(pts, V.K, 10, ctx.seed + 99);
      st.phase = 'done';
      const scoreAfter = V.score(st.best.distortion, st.after.distortion), scoreBefore = V.score(st.best.distortion, st.before.distortion);
      result.style.display = '';
      result.innerHTML = '';
      result.append(
        el('div', { html: `<b>Best of 10 restarts: ${fmt(st.best.distortion, 3)}</b>. Yours after k-means: ${fmt(st.after.distortion, 3)} (score ${Math.round(scoreAfter)}); your hand placement before k-means: ${fmt(st.before.distortion, 3)} (score ${Math.round(scoreBefore)}).` }),
        el('div', { text: `Restart distortions: ${st.best.runs.map((x) => fmt(x, 2)).join(', ')}. The spread is the local-minimum problem; the dashed squares are the winner's codebook.` }),
        el('div', { text: `Each point now costs ${V.bitsPerPoint} bits instead of ${V.rawBitsPerPoint}; the ${V.codebookBytes}-byte codebook ships once.` }));
      status.textContent = 'LOCK IN to record the score.';
      update();
      lock.disabled = false;
    }
    function finish() {
      if (!st.best || !st.after) return;
      lock.disabled = true;
      const scoreAfter = V.score(st.best.distortion, st.after.distortion), scoreBefore = V.score(st.best.distortion, st.before.distortion);
      ctx.complete({ score: Math.round(scoreAfter), summary: `Distortion ${fmt(st.after.distortion, 3)} after k-means (hand placement ${fmt(st.before.distortion, 3)}); best of 10 restarts ${fmt(st.best.distortion, 3)}.`, stats: { before: st.before.distortion, after: st.after.distortion, best: st.best.distortion, scoreBefore, iterations: st.after.iterations, converged: st.after.converged, runs: st.best.runs } });
    }
    update();
  }

  // ---- blockpack: Pack the block
  function renderBlockpack(ctx) {
    const B = calc.blockpack, root = ctx.root;
    const w = B.weights(ctx.seed);
    const ref = B.quantize(w, { b: 4, n: 32, min: false, rot: false });
    const o = { b: 4, n: 32, min: false, rot: false };

    const bitBtns = B.BITS.map((b) => btn(`quest-blockpack-bits-${b}`, String(b), () => { o.b = b; update(); }));
    const blkBtns = B.BLOCKS.map((n) => btn(`quest-blockpack-block-${n}`, String(n), () => { o.n = n; update(); }));
    const minCb = el('input', { type: 'checkbox', id: 'quest-blockpack-min', onchange: () => { o.min = minCb.checked; update(); } });
    const rotCb = el('input', { type: 'checkbox', id: 'quest-blockpack-rot', onchange: () => { o.rot = rotCb.checked; update(); } });
    const strips = mkCanvas(512, 120, 'quest-blockpack-strips');
    const budget = el('div', { class: 'cyc-q-budget', id: 'quest-blockpack-bpw' });
    const ro = readout([['bpw', 'bits per weight'], ['max', 'max |error|'], ['rms', 'RMS error'], ['ref', 'Q4_0 reference RMS'], ['blocks', 'blocks']]);
    const status = el('div', { class: 'cyc-q-status', id: 'quest-blockpack-status' });
    const lock = btn('quest-blockpack-lock', 'LOCK IN', finish, 'primary');
    const fresh = btn('quest-blockpack-newround', 'NEW ROUND', () => ctx.newRound());
    const result = el('div', { class: 'cyc-q-result', id: 'quest-blockpack-result', style: 'display:none' });

    root.append(
      el('div', { class: 'cyc-q-row' },
        panel('format',
          el('div', { class: 'cyc-q-seg' }, el('span', { text: 'bits b' }), ...bitBtns),
          el('div', { class: 'cyc-q-seg' }, el('span', { text: 'block n' }), ...blkBtns),
          el('div', { class: 'cyc-q-seg' }, el('label', {}, minCb, ' min / zero-point (+16 bits per block)'), el('label', {}, rotCb, ' Hadamard rotation (orthogonal: error preserved)')),
          budget),
        el('div', { class: 'cyc-q-col cyc-q-grow' }, panel('readout', ro))),
      el('div', { class: 'cyc-q-col' }, el('div', { class: 'cyc-q-title', text: 'original · reconstructed · |error| ×4 (colour scale clipped at ±0.08; outliers saturate)' }), strips),
      status, result,
      el('div', { class: 'cyc-q-actions' }, lock, fresh));

    let cur = null;
    function update() {
      bitBtns.forEach((b, i) => b.classList.toggle('on', B.BITS[i] === o.b));
      blkBtns.forEach((b, i) => b.classList.toggle('on', B.BLOCKS[i] === o.n));
      cur = B.quantize(w, o);
      const over = cur.bpw > B.BUDGET + 1e-9;
      budget.textContent = `${fmt(cur.bpw, 3)} bpw = (16 scale + ${o.min ? 16 : 0} min + ${o.n} × ${o.b}) / ${o.n} → ${over ? 'OVER' : 'within'} the 4.5 bpw budget`;
      budget.className = 'cyc-q-budget ' + (over ? 'bad' : 'good');
      ro.set('bpw', fmt(cur.bpw, 3), over ? 'bad' : 'good');
      ro.set('max', fmt(cur.maxErr, 4), 'hi');
      ro.set('rms', fmt(cur.rms, 5), 'hi');
      ro.set('ref', `${fmt(ref.rms, 5)} (b=4, n=32, no min, no rotation)`);
      ro.set('blocks', `${cur.blocks} × ${o.n} weights`);
      status.textContent = o.rot ? 'Rotated: the outliers are smeared across all 256 slots, so no single block scale is hijacked.' : o.n >= 128 ? 'Large blocks: one outlier sets the scale for the whole block and its neighbours round to zero.' : `Score preview: ${over ? '0 (over budget)' : Math.round(B.score(ref.rms, cur.rms, cur.bpw)) + ' / 100 = Q4_0 RMS / your RMS'}.`;
      draw();
    }
    function draw() {
      clearCanvas(strips);
      const g = strips.g, Wd = strips.cssW, rowH = 32, CLIP = 0.08;
      const mix = (c0, c1, k) => `rgb(${c0.map((x, n) => Math.round(x + (c1[n] - x) * k)).join(',')})`;
      const col = (v) => { const k = clamp(Math.abs(v) / CLIP, 0, 1); return mix(hexRGB(TK.bg), hexRGB(v >= 0 ? TK.rose : TK.teal), k); };
      const rows = [[w, 'orig'], [cur.recon, 'recon'], [null, 'err']];
      rows.forEach((rw, ri) => {
        const y = ri * (rowH + 8) + 2;
        for (let i = 0; i < w.length; i++) {
          const v = rw[0] ? rw[0][i] : 4 * (w[i] - cur.recon[i]);
          g.fillStyle = rw[0] ? col(v) : mix(hexRGB(TK.bg), hexRGB(TK.ink), clamp(Math.abs(v) / CLIP, 0, 1));
          g.fillRect((i * Wd) / w.length, y, Wd / w.length + 0.5, rowH);
        }
        if (ri < 2) for (let s = 0; s < w.length; s += o.n) { g.strokeStyle = hexA(TK.ink, 0.35); g.beginPath(); g.moveTo((s * Wd) / w.length, y); g.lineTo((s * Wd) / w.length, y + rowH); g.stroke(); }
        g.fillStyle = hexA(TK.bg, 0.8); g.fillRect(0, y, 42, 14); g.fillStyle = TK.ink; g.font = `11px ${TK.mono}`; g.textAlign = 'left'; g.textBaseline = 'top'; g.fillText(rw[1], 3, y + 1);
      });
    }
    function finish() {
      const name = B.formatName(o.b, o.n, o.min, o.rot), gb = B.modelBytes(cur.bpw) / 1e9, score = B.score(ref.rms, cur.rms, cur.bpw);
      result.style.display = '';
      result.innerHTML = '';
      result.append(
        el('div', { html: `<b>${name}</b>. A 7B model at ${fmt(cur.bpw, 2)} bpw weighs ${fmt(gb, 2)} GB (7e9 × bpw / 8).` }),
        el('div', { text: score === 0 ? `Over budget: ${fmt(cur.bpw, 2)} bpw > 4.5. Score 0. Spend bits on weights, not on scales.` : `RMS error ${fmt(cur.rms, 5)} vs plain Q4_0 ${fmt(ref.rms, 5)} → score ${Math.round(score)} / 100.` }));
      lock.disabled = true;
      ctx.complete({ score: Math.round(score), summary: `${name}; RMS ${fmt(cur.rms, 5)} (Q4_0 ${fmt(ref.rms, 5)}), max |err| ${fmt(cur.maxErr, 4)}, 7B model ${fmt(gb, 2)} GB.`, stats: { b: o.b, n: o.n, min: o.min, rot: o.rot, bpw: cur.bpw, rms: cur.rms, maxErr: cur.maxErr, refRms: ref.rms, format: name, modelGB: gb } });
    }
    update();
  }

  // ---- raceaccess: Random access race
  function renderRaceaccess(ctx) {
    const R = calc.raceaccess, root = ctx.root;
    const sym = R.symbols(ctx.seed), an = R.analyze(sym);
    const r = rng(ctx.seed + 5);
    const st = { si: 2, log: [], anim: null, cur: null };

    const lanes = mkCanvas(560, 150, 'quest-raceaccess-lanes');
    const slider = el('input', { type: 'range', min: 0, max: R.SPACINGS.length - 1, step: 1, value: st.si, id: 'quest-raceaccess-spacing', 'aria-label': 'checkpoint spacing' });
    const spLabel = el('span', { class: 'mono', id: 'quest-raceaccess-spacing-label' });
    slider.addEventListener('input', () => { st.si = parseInt(slider.value, 10); st.log = []; st.cur = null; stopAnim(); status.textContent = 'Spacing changed: fetch log reset.'; update(); });
    const fetchBtn = btn('quest-raceaccess-fetch', 'FETCH', doFetch, 'primary');
    const ro = readout([['code', 'Huffman code'], ['bytes', 'bytes (payload + checkpoints)'], ['last', 'last fetch'], ['steps', 'decode steps, cumulative'], ['fetches', 'fetches']]);
    const status = el('div', { class: 'cyc-q-status', id: 'quest-raceaccess-status', text: 'Pick a spacing, then FETCH ten times.' });
    const lock = btn('quest-raceaccess-lock', 'LOCK IN', finish, 'primary'); lock.disabled = true;
    const fresh = btn('quest-raceaccess-newround', 'NEW ROUND', () => { stopAnim(); ctx.newRound(); });
    const result = el('div', { class: 'cyc-q-result', id: 'quest-raceaccess-result', style: 'display:none' });

    root.append(
      el('div', { class: 'cyc-q-row' },
        el('div', { class: 'cyc-q-col cyc-q-grow' },
          panel('checkpoint spacing (16-bit offset each)', slider, spLabel),
          el('div', { class: 'cyc-q-actions' }, fetchBtn)),
        el('div', { class: 'cyc-q-col cyc-q-grow' }, panel('readout', ro))),
      lanes, status, result,
      el('div', { class: 'cyc-q-actions' }, lock, fresh));
    ctx.onDispose(stopAnim);

    const spacing = () => R.SPACINGS[st.si];
    function update() {
      const row = R.row(an.totalBits, spacing());
      spLabel.textContent = `every ${row.label === 'none' ? '— (no checkpoints)' : row.label + ' elements'} → ${row.checkpoints} checkpoints = ${row.checkpointBytes} bytes; mean decode ${fmt(row.meanSteps, 1)} steps`;
      ro.set('code', `${fmt(an.meanBits, 2)} bits/symbol mean (H = ${fmt(an.H, 2)}, ${an.alphabet} symbols, lengths ${Math.min(...an.lengths.filter((l) => l > 0))}–${Math.max(...an.lengths)})`);
      ro.set('bytes', `fixed ${row.fixedBytes} · Huffman ${row.huffmanBytes} (${row.payloadBytes} + ${row.checkpointBytes})`, row.smaller ? 'good' : 'bad');
      const tf = st.log.length, tH = st.log.reduce((s, f) => s + f.steps, 0);
      ro.set('steps', `fixed ${tf} · Huffman ${tH}${tf ? ` (mean ${fmt(tH / tf, 1)})` : ''}`, tf && tH / tf <= 64 ? 'good' : tf ? 'bad' : '');
      ro.set('fetches', `${tf} / 10`);
      ro.set('last', st.cur ? `element ${st.cur.index.toLocaleString('en-US')}: fixed 1 step · Huffman from checkpoint ${st.cur.checkpoint.toLocaleString('en-US')}: ${st.cur.steps.toLocaleString('en-US')} steps` : '—', 'hi');
      lock.disabled = tf === 0;
      draw(st.cur, 1);
    }
    function stopAnim() { if (st.anim) { cancelAnimationFrame(st.anim); st.anim = null; } }
    function doFetch() {
      stopAnim();
      const index = r.int(R.N), s = R.steps(index, spacing());
      st.cur = { index, checkpoint: s.checkpoint, steps: s.steps };
      st.log.push(st.cur);
      status.textContent = `Fetch ${st.log.length}: element ${index.toLocaleString('en-US')} → fixed-length reads byte ${index.toLocaleString('en-US')} directly; Huffman decodes ${s.steps.toLocaleString('en-US')} symbol${s.steps === 1 ? '' : 's'} from checkpoint ${s.checkpoint.toLocaleString('en-US')}.`;
      status.className = 'cyc-q-status';
      update(); // readout and final lane state first; the animation below is cosmetic
      const t0 = performance.now(), dur = 650;
      const frame = (now) => { const t = clamp((now - t0) / dur, 0, 1); draw(st.cur, t); if (t < 1) st.anim = requestAnimationFrame(frame); else st.anim = null; };
      st.anim = requestAnimationFrame(frame);
    }
    function draw(cur, t) {
      clearCanvas(lanes);
      const g = lanes.g, x0 = 96, w = lanes.cssW - x0 - 14, X = (i) => x0 + (w * i) / R.N, qc = cssVar(root, '--qc', TK.gold);
      const laneY = [42, 108], sp = spacing(), ncp = R.checkpoints(sp);
      g.font = `12px ${TK.mono}`; g.textBaseline = 'middle';
      ['fixed 8-bit', 'Huffman'].forEach((name, li) => {
        const y = laneY[li];
        g.fillStyle = TK.ink2; g.textAlign = 'right'; g.fillText(name, x0 - 8, y);
        g.fillStyle = TK.sunk; g.fillRect(x0, y - 10, w, 20); g.strokeStyle = TK.ink; g.strokeRect(x0, y - 10, w, 20);
        if (li === 1 && ncp > 0) {
          if (ncp <= 64) { g.strokeStyle = TK.teal; g.lineWidth = 1; for (let c = 0; c < R.N; c += sp) { g.beginPath(); g.moveTo(X(c), y - 14); g.lineTo(X(c), y + 14); g.stroke(); } }
          else { g.fillStyle = hexA(TK.teal, 0.25); g.fillRect(x0, y - 10, w, 20); }
          g.fillStyle = TK.teal; g.textAlign = 'left'; g.fillText(`${ncp.toLocaleString('en-US')} checkpoint${ncp === 1 ? '' : 's'}${ncp > 64 ? ' (too dense to draw)' : ''}`, x0 + 4, y + 24);
        }
      });
      g.fillStyle = TK.ink2; g.textAlign = 'left'; g.fillText('0', x0, 14); g.textAlign = 'right'; g.fillText('4095', x0 + w, 14);
      if (!cur) return;
      const xi = X(cur.index);
      g.strokeStyle = qc; g.lineWidth = 2; g.setLineDash([4, 3]); g.beginPath(); g.moveTo(xi, laneY[0] - 16); g.lineTo(xi, laneY[1] + 16); g.stroke(); g.setLineDash([]);
      g.fillStyle = qc; g.textAlign = 'center'; g.fillText(`element ${cur.index}`, clamp(xi, x0 + 40, x0 + w - 40), laneY[1] + 30);
      if (t >= 0.25) { g.fillStyle = TK.good; g.fillRect(xi - 5, laneY[0] - 7, 10, 14); g.fillStyle = TK.ink; g.textAlign = 'left'; g.fillText('1 step', Math.min(xi + 10, x0 + w - 50), laneY[0]); }
      const prog = clamp((t - 0.15) / 0.8, 0, 1), xc = X(cur.checkpoint), xh = xc + (xi - xc) * prog;
      g.fillStyle = hexA(TK.bad, 0.45); g.fillRect(xc, laneY[1] - 7, Math.max(2, xh - xc), 14);
      g.fillStyle = TK.bad; g.fillRect(xh - 5, laneY[1] - 7, 10, 14);
      g.fillStyle = TK.ink; g.textAlign = 'left'; g.fillText(`${Math.round(cur.steps * prog).toLocaleString('en-US')} steps`, Math.min(xh + 10, x0 + w - 70), laneY[1]);
    }
    function finish() {
      stopAnim();
      const table = R.table(an.totalBits), chosen = table[st.si];
      const tbl = el('table', { id: 'quest-raceaccess-table' },
        el('thead', {}, el('tr', {}, ...['spacing', 'checkpoints', 'Huffman bytes', 'fixed bytes', 'mean steps', 'verdict'].map((h) => el('th', { text: h })))),
        el('tbody', {}, ...table.map((row) => el('tr', { class: row === chosen ? 'chosen' : '' },
          el('td', { text: row.label }), el('td', { text: row.checkpoints.toLocaleString('en-US') }), el('td', { text: row.huffmanBytes.toLocaleString('en-US') }), el('td', { text: row.fixedBytes.toLocaleString('en-US') }), el('td', { text: fmt(row.meanSteps, 1) }),
          el('td', { text: row.score === 100 ? 'smaller and ≤ 64 steps' : row.score === 60 ? 'smaller but slow' : 'bigger than fixed' })))));
      const tf = st.log.length, tH = st.log.reduce((s, f) => s + f.steps, 0);
      result.style.display = '';
      result.innerHTML = '';
      result.append(
        el('div', { html: `<b>Spacing ${chosen.label}: Huffman ${chosen.huffmanBytes.toLocaleString('en-US')} bytes vs fixed ${chosen.fixedBytes.toLocaleString('en-US')}; mean decode ${fmt(chosen.meanSteps, 1)} steps vs 1.</b> Your ${tf} fetches: ${tH.toLocaleString('en-US')} Huffman steps vs ${tf} fixed.` }),
        tbl,
        el('div', { text: `Score ${chosen.score} / 100 (100: smaller and mean steps ≤ 64; 60: smaller but slower; 30: not smaller).` }));
      lock.disabled = true; fetchBtn.disabled = true; slider.disabled = true;
      ctx.complete({ score: chosen.score, summary: `Spacing ${chosen.label}: Huffman ${chosen.huffmanBytes} bytes vs fixed ${chosen.fixedBytes}, mean ${fmt(chosen.meanSteps, 1)} decode steps (${tf} fetches, ${tH} steps).`, stats: { spacing: chosen.spacing, huffmanBytes: chosen.huffmanBytes, fixedBytes: chosen.fixedBytes, meanSteps: chosen.meanSteps, meanBits: an.meanBits, entropy: an.H, fetches: tf, totalSteps: tH, table } });
    }
    update();
  }

  // ---- siamese: Joined at the hip
  function renderSiamese(ctx) {
    const S = calc.siamese, root = ctx.root;
    const round = (((ctx.seed - BASE_SEED) % S.SLOPES.length) + S.SLOPES.length) % S.SLOPES.length;
    const slope = S.SLOPES[round];
    const P = S.pairs(ctx.seed, slope), w = S.weights(P.a);
    const st = { mode: 'scalar', weighted: false, res: { scalar: null, joint: null }, resW: { scalar: null, joint: null }, ent: null };

    const map = mkCanvas(320, 320, 'quest-siamese-map');
    const mScalar = btn('quest-siamese-mode-scalar', 'SCALAR', () => { st.mode = 'scalar'; update(); });
    const mJoint = btn('quest-siamese-mode-joint', 'JOINT', () => { st.mode = 'joint'; update(); });
    const run = btn('quest-siamese-run', 'RUN', doRun, 'primary');
    const ent = btn('quest-siamese-entropy', 'ENTROPY', doEntropy);
    const weighted = btn('quest-siamese-weighted', 'WEIGHTED', () => { st.weighted = !st.weighted; setStatus(st.weighted ? 'WEIGHTED on: RUN also fits centroids with sample weights w = |a| + 0.1.' : 'WEIGHTED off.'); update(); });
    const ro = readout([['data', 'data'], ['scalar', 'SCALAR MSE (two 1-bit axes)'], ['joint', 'JOINT MSE (one 2-bit codebook)'], ['ratio', 'scalar / joint'], ['Ha', 'H(a)'], ['Hb', 'H(b)'], ['Hab', 'H(a,b)'], ['I', 'I = H(a)+H(b)−H(a,b)']]);
    const wtable = el('div', { id: 'quest-siamese-weighted-table' });
    const status = el('div', { class: 'cyc-q-status', id: 'quest-siamese-status', text: 'Pick a mode and press RUN; do both, then ENTROPY.' });
    const lock = btn('quest-siamese-lock', 'LOCK IN', finish, 'primary'); lock.disabled = true;
    const fresh = btn('quest-siamese-newround', 'NEW ROUND', () => ctx.newRound());
    const result = el('div', { class: 'cyc-q-result', id: 'quest-siamese-result', style: 'display:none' });
    root.append(
      el('div', { class: 'cyc-q-row' },
        el('div', { class: 'cyc-q-col' }, map),
        el('div', { class: 'cyc-q-col cyc-q-grow' },
          el('div', { class: 'cyc-q-actions' }, el('span', { text: 'mode:' }), mScalar, mJoint, run, ent, weighted),
          status, panel('readout', ro), wtable)),
      result,
      el('div', { class: 'cyc-q-actions' }, lock, fresh));

    function setStatus(t, cls) { status.textContent = t; status.className = 'cyc-q-status' + (cls ? ' ' + cls : ''); }
    function doRun() {
      const r = st.mode === 'scalar' ? S.scalar(P) : S.joint(P, ctx.seed + 11);
      st.res[st.mode] = r;
      let extra = '';
      if (st.weighted) { const rw = st.mode === 'scalar' ? S.scalar(P, w) : S.joint(P, ctx.seed + 11, w); st.resW[st.mode] = rw; extra = ` Weighted fit: plain MSE ${fmt(rw.mse, 4)} (up from ${fmt(r.mse, 4)}), weighted MSE ${fmt(rw.wmse, 4)} (down from ${fmt(r.wmse, 4)}).`; }
      setStatus(`${st.mode.toUpperCase()}: MSE ${fmt(r.mse, 4)} per value at 2 bits per pair.${extra}`, 'ok');
      update();
    }
    function doEntropy() {
      st.ent = S.entropy(P);
      setStatus(`16 bins of width 0.5 over [−4, 4]: I = ${fmt(st.ent.I)} bits of the H(a)+H(b) = ${fmt(st.ent.Ha + st.ent.Hb)} that two marginal codes pay.`);
      update();
    }
    function update() {
      mScalar.classList.toggle('on', st.mode === 'scalar'); mJoint.classList.toggle('on', st.mode === 'joint'); weighted.classList.toggle('on', st.weighted);
      const s = st.res.scalar, j = st.res.joint, e = st.ent;
      ro.set('data', `b = ${fmt(slope, 1)}·a + 0.1·N(0,1), ${S.N} pairs (round ${round + 1} of ${S.SLOPES.length})`);
      ro.set('scalar', s ? `${fmt(s.mse, 4)} (levels a: ${s.ca.map((v) => fmt(v, 2)).join(' / ')}, b: ${s.cb.map((v) => fmt(v, 2)).join(' / ')})` : '— press RUN in SCALAR', s ? 'hi' : '');
      ro.set('joint', j ? fmt(j.mse, 4) : '— press RUN in JOINT', j ? 'hi' : '');
      ro.set('ratio', s && j ? `${fmt(s.mse / j.mse)}× → score ${Math.round(S.score(s.mse, j.mse))} / 100` : '—', s && j ? 'good' : '');
      ro.set('Ha', e ? `${fmt(e.Ha)} bits` : '— press ENTROPY'); ro.set('Hb', e ? `${fmt(e.Hb)} bits` : '—'); ro.set('Hab', e ? `${fmt(e.Hab)} bits` : '—'); ro.set('I', e ? `${fmt(e.I)} bits` : '—', e ? 'hi' : '');
      wtable.innerHTML = '';
      for (const m of ['scalar', 'joint']) if (st.resW[m] && st.res[m]) {
        wtable.appendChild(el('table', {},
          el('thead', {}, el('tr', {}, el('th', { text: `${m.toUpperCase()} centroids` }), el('th', { text: 'plain MSE' }), el('th', { text: 'weighted MSE (w = |a| + 0.1)' }))),
          el('tbody', {},
            el('tr', {}, el('td', { text: 'unweighted fit' }), el('td', { text: fmt(st.res[m].mse, 4) }), el('td', { text: fmt(st.res[m].wmse, 4) })),
            el('tr', {}, el('td', { text: 'weighted fit' }), el('td', { text: fmt(st.resW[m].mse, 4) }), el('td', { text: fmt(st.resW[m].wmse, 4) })))));
      }
      lock.disabled = !(s && j);
      draw();
    }
    function draw() {
      clearCanvas(map);
      const g = map.g, qc = cssVar(root, '--qc', TK.gold), X = (v) => ((v + 4) / 8) * map.cssW, Y = (v) => map.cssH - ((v + 4) / 8) * map.cssH;
      g.strokeStyle = hexA(TK.line, 0.4); g.lineWidth = 1;
      for (let v = -4; v <= 4; v += 2) { g.beginPath(); g.moveTo(X(v), 0); g.lineTo(X(v), map.cssH); g.stroke(); g.beginPath(); g.moveTo(0, Y(v)); g.lineTo(map.cssW, Y(v)); g.stroke(); }
      g.fillStyle = hexA(TK.ink2, 0.55);
      for (let i = 0; i < P.n; i++) g.fillRect(X(P.a[i]), Y(P.b[i]), 1.5, 1.5);
      const drawPts = (pts, color, hollow) => {
        for (const [x, y] of pts) {
          if (hollow) { g.strokeStyle = color; g.lineWidth = 2; g.strokeRect(X(x) - 6, Y(y) - 6, 12, 12); }
          else { g.fillStyle = color; g.fillRect(X(x) - 5, Y(y) - 5, 10, 10); g.strokeStyle = TK.ink; g.lineWidth = 1; g.strokeRect(X(x) - 5, Y(y) - 5, 10, 10); }
        }
      };
      if (st.res.scalar) {
        const s = st.res.scalar;
        g.strokeStyle = qc; g.lineWidth = 1; g.setLineDash([3, 3]);
        for (const v of s.ca) { g.beginPath(); g.moveTo(X(v), 0); g.lineTo(X(v), map.cssH); g.stroke(); }
        for (const v of s.cb) { g.beginPath(); g.moveTo(0, Y(v)); g.lineTo(map.cssW, Y(v)); g.stroke(); }
        g.setLineDash([]);
        drawPts(s.points, qc, true);
      }
      if (st.res.joint) drawPts(st.res.joint.points, TK.teal, false);
      if (st.weighted) { if (st.resW.scalar) drawPts(st.resW.scalar.points, TK.rose, true); if (st.resW.joint) drawPts(st.resW.joint.points, TK.rose, false); }
      g.font = `11px ${TK.mono}`; g.fillStyle = TK.ink2; g.textAlign = 'left'; g.textBaseline = 'top';
      g.fillText('b ↑  a →', 4, 4);
      g.textBaseline = 'bottom';
      g.fillText(`hollow = SCALAR grid · filled = JOINT centroids${st.weighted ? ' · orange = weighted' : ''}`, 4, map.cssH - 4);
    }
    function finish() {
      const s = st.res.scalar, j = st.res.joint;
      if (!s || !j) return;
      const score = S.score(s.mse, j.mse), e = st.ent || S.entropy(P);
      result.style.display = ''; result.innerHTML = '';
      result.append(
        el('div', { html: `<b>SCALAR MSE ${fmt(s.mse, 4)} vs JOINT MSE ${fmt(j.mse, 4)}: ${fmt(s.mse / j.mse)}× better for the same 2 bits per pair.</b>` }),
        el('div', { text: `H(a) = ${fmt(e.Ha)}, H(b) = ${fmt(e.Hb)}, H(a,b) = ${fmt(e.Hab)} bits: two marginal codes pay the mutual information I = ${fmt(e.I)} bits twice. Score ${Math.round(score)} / 100.` }));
      lock.disabled = true;
      ctx.complete({ score: Math.round(score), summary: `Slope ${fmt(slope, 1)}: scalar MSE ${fmt(s.mse, 4)} vs joint ${fmt(j.mse, 4)} (${fmt(s.mse / j.mse)}×); I = ${fmt(e.I)} bits.`, stats: { slope, scalarMse: s.mse, jointMse: j.mse, Ha: e.Ha, Hb: e.Hb, Hab: e.Hab, I: e.I, weighted: st.weighted } });
    }
    update();
  }

  // ---- anchorchain: Anchor or chain?
  function renderAnchorchain(ctx) {
    const A = calc.anchorchain, root = ctx.root;
    const x = A.matrix(ctx.seed);
    const modes = ['none', 'chain', 'anchor'];
    const st = { mode: 'anchor', gi: 2 };
    const mBtns = modes.map((m) => btn(`quest-anchorchain-mode-${m}`, m.toUpperCase(), () => { st.mode = m; update(); }));
    const slider = el('input', { type: 'range', min: 0, max: A.G.length - 1, step: 1, value: st.gi, id: 'quest-anchorchain-g', 'aria-label': 'anchor spacing g' });
    slider.addEventListener('input', () => { st.gi = parseInt(slider.value, 10); update(); });
    const gLabel = el('span', { class: 'mono', id: 'quest-anchorchain-g-label' });
    const wave = mkCanvas(560, 236, 'quest-anchorchain-wave');
    const ro = readout([['H', 'entropy of the symbols'], ['bits', 'bits / element (raw tokens at 8)'], ['depth', 'decode dependency depth'], ['err', 'worst-case |error|']]);
    const table = el('div', { id: 'quest-anchorchain-table' });
    const status = el('div', { class: 'cyc-q-status', id: 'quest-anchorchain-status', text: 'Uniform quantizer Δ = 1 on 64 tokens × 16 channels; g only matters for ANCHOR.' });
    const lock = btn('quest-anchorchain-lock', 'LOCK IN', finish, 'primary');
    const fresh = btn('quest-anchorchain-newround', 'NEW ROUND', () => ctx.newRound());
    const result = el('div', { class: 'cyc-q-result', id: 'quest-anchorchain-result', style: 'display:none' });
    root.append(
      el('div', { class: 'cyc-q-row' },
        el('div', { class: 'cyc-q-col cyc-q-grow' }, el('div', { class: 'cyc-q-actions' }, el('span', { text: 'predictor:' }), ...mBtns), panel('anchor spacing g', slider, gLabel), status),
        el('div', { class: 'cyc-q-col cyc-q-grow' }, panel('readout', ro))),
      wave, table, result,
      el('div', { class: 'cyc-q-actions' }, lock, fresh));

    let cur = null, all = null;
    function update() {
      const g = A.G[st.gi];
      mBtns.forEach((b, i) => b.classList.toggle('on', modes[i] === st.mode));
      gLabel.textContent = `g = ${g} → ${Math.ceil(A.T / g)} anchors among ${A.T} tokens`;
      all = { none: A.code(x, 'none', g), chain: A.code(x, 'chain', g), anchor: A.code(x, 'anchor', g) };
      cur = all[st.mode];
      ro.set('H', `${fmt(cur.H)} bits/symbol over ${cur.nSymbols} symbols`, 'hi');
      ro.set('bits', `${fmt(cur.bitsPerElement)} = (${cur.raw} raw × 8 + ${cur.nSymbols} × H) / ${A.T * A.C}`, 'hi');
      ro.set('depth', `${cur.depth} sequential step${cur.depth === 1 ? '' : 's'} to reach token 63`, cur.depth <= 2 ? 'good' : 'bad');
      ro.set('err', fmt(cur.maxErr, 2), cur.maxErr <= 0.5 + 1e-9 ? 'good' : 'bad');
      table.innerHTML = '';
      table.appendChild(el('table', {},
        el('thead', {}, el('tr', {}, ...['predictor', 'H (bits/symbol)', 'bits/element', 'depth', 'worst |error|'].map((h) => el('th', { text: h })))),
        el('tbody', {}, ...modes.map((m) => el('tr', { class: m === st.mode ? 'chosen' : '' },
          el('td', { text: m.toUpperCase() + (m === 'anchor' ? ` (g = ${g})` : '') }), el('td', { text: fmt(all[m].H) }), el('td', { text: fmt(all[m].bitsPerElement) }), el('td', { text: String(all[m].depth) }), el('td', { text: fmt(all[m].maxErr, 2) }))))));
      draw(g);
    }
    function draw(g) {
      clearCanvas(wave);
      const G = wave.g, x0 = 52, w = wave.cssW - x0 - 10, top = 16, h = 128, X = (t) => x0 + (w * t) / (A.T - 1);
      const ch = (arr) => Array.from({ length: A.T }, (_, t) => arr[t * A.C]);
      const o = ch(x), c = ch(all.chain.recon), a = ch(all.anchor.recon);
      const lo = Math.min(...o, ...c, ...a) - 1, hi = Math.max(...o, ...c, ...a) + 1, Y = (v) => top + h - (h * (v - lo)) / (hi - lo);
      G.font = `11px ${TK.mono}`; G.fillStyle = TK.ink2; G.textAlign = 'right'; G.textBaseline = 'middle';
      G.fillText(fmt(hi, 0), x0 - 4, Y(hi)); G.fillText(fmt(lo, 0), x0 - 4, Y(lo));
      if (st.mode === 'anchor') { G.strokeStyle = hexA(TK.teal, 0.35); G.lineWidth = 1; for (let t = 0; t < A.T; t += g) { G.beginPath(); G.moveTo(X(t), top); G.lineTo(X(t), top + h); G.stroke(); } }
      const line = (arr, color, width) => { G.strokeStyle = color; G.lineWidth = width; G.beginPath(); arr.forEach((v, t) => (t ? G.lineTo(X(t), Y(v)) : G.moveTo(X(t), Y(v)))); G.stroke(); };
      line(c, TK.bad, 2); line(a, TK.teal, 2); line(o, TK.ink, 1);
      G.textAlign = 'left'; G.textBaseline = 'top';
      G.fillStyle = TK.ink; G.fillText('channel 0: original', x0 + 4, top); G.fillStyle = TK.bad; G.fillText('CHAIN recon', x0 + 160, top); G.fillStyle = TK.teal; G.fillText(`ANCHOR recon (g = ${g})`, x0 + 260, top);
      const rows = [['NONE', all.none], ['CHAIN', all.chain], ['ANCHOR', all.anchor]], ry0 = top + h + 10, rh = 14;
      const emax = Math.max(0.5, ...rows.map((r) => r[1].maxErr));
      rows.forEach(([name, res], i) => {
        const y = ry0 + i * (rh + 4);
        G.fillStyle = TK.ink2; G.textAlign = 'right'; G.textBaseline = 'middle'; G.fillText(name, x0 - 4, y + rh / 2);
        for (let t = 0; t < A.T; t++) { const e = res.errByToken[t] / emax; G.fillStyle = hexA(TK.bad, 0.1 + 0.9 * e); G.fillRect(x0 + (w * t) / A.T, y, w / A.T + 0.5, rh); }
      });
      G.fillStyle = TK.ink2; G.textAlign = 'left'; G.textBaseline = 'top';
      G.fillText(`|error| per token, max over the 16 channels (black 0 → red ${fmt(emax, 1)}); CHAIN drifts, ANCHOR is bounded`, x0, ry0 + 3 * (rh + 4) + 2);
    }
    function finish() {
      const g = A.G[st.gi], score = A.score(st.mode, g);
      result.style.display = ''; result.innerHTML = '';
      result.append(
        el('div', { html: `<b>${st.mode.toUpperCase()}${st.mode === 'anchor' ? ` with g = ${g}` : ''}: ${fmt(cur.bitsPerElement)} bits/element, dependency depth ${cur.depth}, worst-case error ${fmt(cur.maxErr, 2)}.</b>` }),
        el('div', { text: `CHAIN ${fmt(all.chain.bitsPerElement)} bits / depth 64 / error ${fmt(all.chain.maxErr, 2)} · ANCHOR ${fmt(all.anchor.bitsPerElement)} / 2 / ${fmt(all.anchor.maxErr, 2)} · NONE ${fmt(all.none.bitsPerElement)} / 1 / ${fmt(all.none.maxErr, 2)}. Score ${score} / 100.` }));
      lock.disabled = true;
      ctx.complete({ score, summary: `${st.mode.toUpperCase()}${st.mode === 'anchor' ? ` g = ${g}` : ''}: ${fmt(cur.bitsPerElement)} bits/element, dependency depth ${cur.depth}, worst-case error ${fmt(cur.maxErr, 2)}.`, stats: { mode: st.mode, g, H: cur.H, bitsPerElement: cur.bitsPerElement, depth: cur.depth, maxErr: cur.maxErr, chain: { bits: all.chain.bitsPerElement, maxErr: all.chain.maxErr }, anchor: { bits: all.anchor.bitsPerElement, maxErr: all.anchor.maxErr }, none: { bits: all.none.bitsPerElement, maxErr: all.none.maxErr } } });
    }
    update();
  }

  // ---- rankkeys: Rank the keys
  function renderRankkeys(ctx) {
    const R = calc.rankkeys, root = ctx.root;
    const M = R.matrices(ctx.seed), aK = R.analyzeK(M.K, M.Q), aV = R.analyzeV(M.V);
    const st = { which: 'K', r: 16 };
    const slider = el('input', { type: 'range', min: 1, max: R.D, step: 1, value: st.r, id: 'quest-rankkeys-r', 'aria-label': 'rank r' });
    slider.addEventListener('input', () => { st.r = parseInt(slider.value, 10); update(); });
    const rLabel = el('span', { class: 'mono', id: 'quest-rankkeys-r-label' });
    const sw = btn('quest-rankkeys-switch', 'SWITCH TO V', () => { st.which = st.which === 'K' ? 'V' : 'K'; update(); });
    const bars = mkCanvas(560, 170, 'quest-rankkeys-bars');
    const ro = readout([['m', 'matrix'], ['store', 'storage (numbers)'], ['err', 'error'], ['ov', 'top-16 overlap']]);
    const status = el('div', { class: 'cyc-q-status', id: 'quest-rankkeys-status' });
    const lock = btn('quest-rankkeys-lock', 'LOCK IN', finish, 'primary');
    const fresh = btn('quest-rankkeys-newround', 'NEW ROUND', () => ctx.newRound());
    const result = el('div', { class: 'cyc-q-result', id: 'quest-rankkeys-result', style: 'display:none' });
    root.append(
      el('div', { class: 'cyc-q-row' },
        el('div', { class: 'cyc-q-col cyc-q-grow' }, panel('rank r', slider, rLabel), el('div', { class: 'cyc-q-actions' }, sw), status),
        el('div', { class: 'cyc-q-col cyc-q-grow' }, panel('readout', ro))),
      bars, result,
      el('div', { class: 'cyc-q-actions' }, lock, fresh));

    function update() {
      const r = st.r, s = R.storage(r), isK = st.which === 'K';
      sw.textContent = isK ? 'SWITCH TO V' : 'SWITCH TO K';
      rLabel.textContent = `r = ${r} of ${R.D}`;
      ro.set('m', isK ? 'K: 256 tokens × 64 dims, singular values 8·0.85^i, plus noise' : 'V: 256 tokens × 64 dims, flat singular values (2), plus noise');
      ro.set('store', `L·r + r·d = ${s.lowrank.toLocaleString('en-US')} vs L·d = ${s.full.toLocaleString('en-US')} (crossover at r = ${fmt(s.crossover, 1)})`, s.lowrank < s.full ? 'good' : 'bad');
      if (isK) {
        ro.set('err', `Σ_t |softmax(QKᵀ/√d) − softmax(QK_rᵀ/√d)| = ${aK.scoreErr[r].toPrecision(3)} (${fmt((100 * aK.scoreErr[r]) / aK.scoreErr[1], 0)}% of the r = 1 error; mean over 8 queries)`, 'hi');
        ro.set('ov', `${fmt(aK.overlap[r], 1)} of 16 (mean over 8 queries)`, aK.overlap[r] >= 15 ? 'good' : 'bad');
        status.textContent = aK.overlap[r] >= 15 ? 'Overlap holds at this rank. Is there a smaller rank where it still holds?' : 'Overlap below 15 of 16: the truncated keys change who gets attended to.';
      } else {
        ro.set('err', `relative Frobenius error ‖V − V_r‖ / ‖V‖ = ${fmt(aV.relErr[r], 3)}`, 'hi');
        ro.set('ov', 'n/a: attention scores do not involve V');
        status.textContent = 'Storage falls identically, but V has no decay: the dropped directions carry as much as the kept ones.';
      }
      draw(isK ? aK.sigma : aV.sigma, r);
    }
    function draw(sigma, r) {
      clearCanvas(bars);
      const g = bars.g, qc = cssVar(root, '--qc', TK.gold), x0 = 36, y0 = bars.cssH - 22, w = bars.cssW - x0 - 10, hgt = y0 - 14;
      const smax = Math.max(...sigma), bw = w / R.D, Y = (v) => y0 - (hgt * v) / smax;
      g.font = `11px ${TK.mono}`; g.textBaseline = 'middle'; g.textAlign = 'right'; g.fillStyle = TK.ink2;
      g.fillText(fmt(smax, 1), x0 - 4, Y(smax)); g.fillText('0', x0 - 4, y0);
      sigma.forEach((v, i) => { g.fillStyle = i < r ? qc : TK.line; g.fillRect(x0 + i * bw + 1, Y(v), Math.max(1, bw - 2), y0 - Y(v)); });
      g.strokeStyle = TK.ink; g.lineWidth = 1; g.beginPath(); g.moveTo(x0 + r * bw, 14); g.lineTo(x0 + r * bw, y0); g.stroke();
      g.textAlign = 'left'; g.textBaseline = 'top'; g.fillStyle = TK.ink; g.fillText(`kept r = ${r}`, Math.min(x0 + r * bw + 4, bars.cssW - 90), 14);
      g.fillStyle = TK.ink2; g.fillText(`singular values of ${st.which} (σ₁…σ₆₄); ${st.which === 'K' ? 'geometric decay: energy compaction along the token axis' : 'flat: nothing to truncate'}`, x0, y0 + 6);
    }
    function finish() {
      const r = st.r, sc = R.score(r, aK.overlap), s = R.storage(r);
      result.style.display = ''; result.innerHTML = '';
      result.append(
        el('div', { html: `<b>Rank ${r}: K top-16 overlap ${fmt(aK.overlap[r], 1)} / 16, storage ${s.lowrank.toLocaleString('en-US')} of ${s.full.toLocaleString('en-US')} numbers (${fmt((100 * s.lowrank) / s.full, 0)}%).</b> The smallest rank with overlap ≥ 15 is r = ${sc.rStar}.` }),
        el('div', { text: `At the same rank V keeps only ${fmt(100 * (1 - aV.relErr[r] ** 2), 0)}% of its energy (relative error ${fmt(aV.relErr[r], 2)}): keys are low-rank, values are not. Score ${sc.score} / 100${st.which === 'V' ? ' (scored on the key matrix)' : ''}.` }));
      lock.disabled = true;
      ctx.complete({ score: sc.score, summary: `Rank ${r}: K overlap ${fmt(aK.overlap[r], 1)}/16 (smallest rank holding 15/16: ${sc.rStar}), storage ${s.lowrank} vs ${s.full}; V relative error ${fmt(aV.relErr[r], 2)}.`, stats: { r, which: st.which, overlap: aK.overlap[r], scoreErr: aK.scoreErr[r], relErrV: aV.relErr[r], rStar: sc.rStar, storage: s } });
    }
    update();
  }

  // ---- pagepurge: Rented by the room
  function renderPagepurge(ctx) {
    const Pg = calc.pagepurge, root = ctx.root;
    const { w, sinks } = Pg.weights(ctx.seed), wmax = Math.max(...w);
    const st = { evicted: new Uint8Array(Pg.N), mode: 'holefill', comp: null };
    const qc = cssVar(root, '--qc', TK.gold);
    const rgb = (() => { const m = /^#([0-9a-f]{6})$/i.exec(qc.trim()); const n = m ? parseInt(m[1], 16) : 0xb8892e; return [n >> 16, (n >> 8) & 255, n & 255]; })();
    const shade = (i) => `rgba(${rgb[0]},${rgb[1]},${rgb[2]},${(0.12 + 0.88 * Math.sqrt(w[i] / wmax)).toFixed(3)})`;

    const grid = el('div', { class: 'cyc-q-grid16', id: 'quest-pagepurge-grid' });
    const tokBtns = [], rowLabels = [];
    for (let p = 0; p < Pg.PAGES; p++) {
      const lab = el('span', { class: 'cyc-q-pagelabel', text: `p${p}` });
      rowLabels.push(lab); grid.appendChild(lab);
      for (let i = p * Pg.PAGE; i < (p + 1) * Pg.PAGE; i++) {
        const b = el('button', { type: 'button', class: 'cyc-q-tok', id: `quest-pagepurge-tok-${i}`, title: `token ${i}: ${(100 * w[i]).toFixed(2)}% of attention${sinks.includes(i) ? ' (sink)' : i >= Pg.N - Pg.PAGE ? ' (recent)' : ''}`, onclick: () => { st.evicted[i] ^= 1; st.comp = null; update(); } });
        b.style.background = shade(i);
        tokBtns.push(b); grid.appendChild(b);
      }
    }
    const after = el('div', { class: 'cyc-q-grid16', id: 'quest-pagepurge-after' });
    const afterWrap = el('div', { class: 'cyc-q-col', style: 'display:none' }, el('div', { class: 'cyc-q-title', text: 'after compaction (green outline = copied)' }), after);
    const auto = btn('quest-pagepurge-auto', 'AUTO', () => { st.evicted = Pg.auto(w); st.comp = null; setStatus('AUTO: evicted the 192 least important tokens.'); update(); });
    const mHole = btn('quest-pagepurge-mode-holefill', 'HOLE-FILL', () => { st.mode = 'holefill'; st.comp = null; update(); });
    const mOrder = btn('quest-pagepurge-mode-inorder', 'IN-ORDER', () => { st.mode = 'inorder'; st.comp = null; update(); });
    const compact = btn('quest-pagepurge-compact', 'COMPACT', doCompact, 'primary');
    const ro = readout([['kept', 'tokens kept'], ['mass', 'attention mass kept'], ['freed', 'PAGES FREED'], ['need', 'pages needed'], ['comp', 'after compaction']]);
    const status = el('div', { class: 'cyc-q-status', id: 'quest-pagepurge-status', text: 'Click tokens to evict them, or press AUTO. A page is freed only when all 16 of its tokens are gone.' });
    const lock = btn('quest-pagepurge-lock', 'LOCK IN', finish, 'primary');
    const fresh = btn('quest-pagepurge-newround', 'NEW ROUND', () => ctx.newRound());
    const result = el('div', { class: 'cyc-q-result', id: 'quest-pagepurge-result', style: 'display:none' });
    root.append(
      el('div', { class: 'cyc-q-row' },
        el('div', { class: 'cyc-q-col cyc-q-grow' }, grid, afterWrap),
        el('div', { class: 'cyc-q-col cyc-q-grow' },
          el('div', { class: 'cyc-q-actions' }, auto, el('span', { text: 'compaction:' }), mHole, mOrder, compact),
          status, panel('readout', ro))),
      result,
      el('div', { class: 'cyc-q-actions' }, lock, fresh));

    function setStatus(t, cls) { status.textContent = t; status.className = 'cyc-q-status' + (cls ? ' ' + cls : ''); }
    function doCompact() {
      st.comp = st.mode === 'holefill' ? Pg.holeFill(st.evicted) : Pg.inOrder(st.evicted);
      setStatus(`${st.mode === 'holefill' ? 'HOLE-FILL' : 'IN-ORDER'}: ${st.comp.copies} copies, ${st.comp.pagesFreed} pages freed (minimum copies for this eviction set: ${st.comp.minCopies}).`, 'ok');
      update();
    }
    function update() {
      mHole.classList.toggle('on', st.mode === 'holefill'); mOrder.classList.toggle('on', st.mode === 'inorder');
      const s = Pg.stats(w, st.evicted);
      tokBtns.forEach((b, i) => b.classList.toggle('ev', !!st.evicted[i]));
      rowLabels.forEach((lab, p) => { lab.textContent = `p${p}·${s.perPage[p]}`; lab.style.color = s.perPage[p] === 0 ? TK.good : ''; });
      ro.set('kept', `${s.kept} / ${Pg.N} (budget ${Pg.BUDGET})`, s.kept <= Pg.BUDGET ? 'good' : 'bad');
      ro.set('mass', `${fmt(100 * s.mass, 1)}%`, 'hi');
      ro.set('freed', `${s.pagesFreed} of ${Pg.PAGES} (${s.pagesUsed} still occupied)`, s.pagesFreed ? 'good' : 'bad');
      ro.set('need', `${s.pagesNeeded} = ceil(${s.kept} / 16)`);
      ro.set('comp', st.comp ? `${st.comp.copies} copies → ${st.comp.pagesUsed} pages used, ${st.comp.pagesFreed} freed (min copies ${st.comp.minCopies})` : '— press COMPACT', st.comp ? (st.comp.pagesUsed === s.pagesNeeded ? 'good' : 'bad') : '');
      after.innerHTML = '';
      if (st.comp) {
        afterWrap.style.display = '';
        const lay = st.comp.layout;
        for (let p = 0; p < Pg.PAGES; p++) {
          after.appendChild(el('span', { class: 'cyc-q-pagelabel', text: `p${p}·${st.comp.perPage[p]}`, style: st.comp.perPage[p] === 0 ? 'color:var(--qgood)' : '' }));
          for (let slot = p * Pg.PAGE; slot < (p + 1) * Pg.PAGE; slot++) {
            const tok = lay[slot];
            const cell = el('span', { class: 'cyc-q-tok' + (tok < 0 ? ' empty' : tok !== slot ? ' moved' : ''), title: tok < 0 ? 'empty' : `token ${tok}${tok !== slot ? ` copied from slot ${tok}` : ''}` });
            if (tok >= 0) cell.style.background = shade(tok);
            after.appendChild(cell);
          }
        }
      } else afterWrap.style.display = 'none';
    }
    function finish() {
      const s = Pg.stats(w, st.evicted), c = st.comp;
      const pagesUsed = c ? c.pagesUsed : s.pagesUsed, copies = c ? c.copies : 0, minC = Pg.minCopies(st.evicted);
      const score = Pg.score(s.kept, pagesUsed, copies, minC);
      result.style.display = ''; result.innerHTML = '';
      result.append(
        el('div', { html: `<b>${s.kept} survivors (${fmt(100 * s.mass, 1)}% of attention mass) in ${pagesUsed} pages after ${copies} copies.</b> Eviction alone freed ${s.pagesFreed} pages; ${s.pagesNeeded} pages are needed; the fewest copies that get there is ${minC}.` }),
        el('div', { text: score === 0 ? (s.kept > Pg.BUDGET ? `Over budget: ${s.kept} > ${Pg.BUDGET} tokens. Score 0.` : 'Nothing kept. Score 0.') : score === 40 ? `Pages not minimal (${pagesUsed} used, ${s.pagesNeeded} needed): VRAM is unchanged until you compact. Score 40.` : score === 70 ? `Pages minimal but ${copies} copies vs ${minC} minimum: try HOLE-FILL. Score 70.` : `Pages minimal with ${copies} copies (≤ 1.2 × ${minC}). Score 100.` }));
      lock.disabled = true;
      ctx.complete({ score, summary: `${s.kept} tokens kept (${fmt(100 * s.mass, 1)}% of attention mass): ${s.pagesFreed} pages freed by eviction, ${pagesUsed} pages after ${copies} copies (${c ? (st.mode === 'holefill' ? 'HOLE-FILL' : 'IN-ORDER') : 'no compaction'}; minimum ${minC}).`, stats: { kept: s.kept, mass: s.mass, pagesFreedByEviction: s.pagesFreed, pagesUsed, pagesNeeded: s.pagesNeeded, copies, minCopies: minC, mode: c ? st.mode : null } });
    }
    update();
  }

  // ======================================================================= public API

  const renderers = { shannon: renderShannon, hufftree: renderHufftree, waterfill: renderWaterfill, jnd: renderJnd, phrasebook: renderPhrasebook, safehouses: renderSafehouses, blockpack: renderBlockpack, raceaccess: renderRaceaccess,
    siamese: renderSiamese, anchorchain: renderAnchorchain, rankkeys: renderRankkeys, pagepurge: renderPagepurge };

  function has(questId) { return Object.prototype.hasOwnProperty.call(renderers, questId); }

  function render(container, questId, opts) {
    opts = opts || {};
    if (!has(questId)) throw new Error('CYC.Quests: unknown quest "' + questId + '"');
    if (typeof document === 'undefined') throw new Error('CYC.Quests.render needs a DOM');
    injectCSS();
    const q = (CYC.SIDEQUESTS || []).find((x) => x.id === questId) || { id: questId };
    container.innerHTML = '';
    const root = el('div', { class: 'cyc-quest cyc-quest-' + questId, id: 'quest-' + questId, 'data-quest': questId });
    const color = roleColorFor(q);
    if (color) root.style.setProperty('--quest-color', color);
    container.appendChild(root);
    refreshTokens(root);

    let done = false, disposers = [];
    const ctx = {
      root, quest: q, seed: BASE_SEED,
      onDispose(fn) { disposers.push(fn); },
      complete(result) {
        if (done) return;
        done = true;
        root.dispatchEvent(new CustomEvent('cyc-quest-complete', { bubbles: true, detail: { questId, result } }));
        if (typeof opts.onComplete === 'function') opts.onComplete(result);
      },
      newRound() {
        for (const fn of disposers) { try { fn(); } catch (e) { /* ignore */ } }
        disposers = []; done = false; ctx.seed += 1;
        root.innerHTML = '';
        renderers[questId](ctx);
      },
    };
    renderers[questId](ctx);
    return { newRound: ctx.newRound, root };
  }

  CYC.Quests = { render, has, calc, ids: () => Object.keys(renderers) };
})();
