// Choose Your Codec — Algorithm labs: one worked numeric example per algorithm.
// Contract (docs/DEMOS.md): CYC.Demos.render(container, fighterId), CYC.Demos.has(id),
// pure compute functions on CYC.Demos.calc.* (testable in node without a DOM).
// Plain ES2020, no libraries. DOM is only touched inside render().

window.CYC = window.CYC || {};

(function () {
  'use strict';

  var CYC = window.CYC;

  // ===========================================================================
  // Pure math (no DOM)
  // ===========================================================================
  var calc = {};

  function rng(seed) { // mulberry32
    var s = (seed >>> 0) || 1;
    return function () {
      s = (s + 0x6D2B79F5) >>> 0;
      var t = s;
      t = Math.imul(t ^ (t >>> 15), t | 1);
      t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }
  function gauss(r) {
    var u = 1 - r(), v = r();
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
  }
  function sum(a) { var s = 0; for (var i = 0; i < a.length; i++) s += a[i]; return s; }
  function sq(x) { return x * x; }
  function log2(x) { return Math.log(x) / Math.LN2; }
  function dist2(p, q) { var s = 0; for (var i = 0; i < p.length; i++) s += sq(p[i] - q[i]); return s; }
  calc.rng = rng;
  calc.mse = function (a, b) { var s = 0; for (var i = 0; i < a.length; i++) s += sq(a[i] - b[i]); return s / a.length; };
  calc.energy = function (a) { return sum(a.map(sq)); };
  calc.entropy = function (probs) { var h = 0; probs.forEach(function (p) { if (p > 0) h -= p * log2(p); }); return h; };

  // ---- transform -----------------------------------------------------------
  calc.dct = function (x) { // orthonormal DCT-II
    var N = x.length, X = [];
    for (var k = 0; k < N; k++) {
      var s = 0;
      for (var n = 0; n < N; n++) s += x[n] * Math.cos(Math.PI * (2 * n + 1) * k / (2 * N));
      X.push(s * Math.sqrt((k === 0 ? 1 : 2) / N));
    }
    return X;
  };
  calc.idct = function (X) {
    var N = X.length, x = [];
    for (var n = 0; n < N; n++) {
      var s = 0;
      for (var k = 0; k < N; k++) s += Math.sqrt((k === 0 ? 1 : 2) / N) * X[k] * Math.cos(Math.PI * (2 * n + 1) * k / (2 * N));
      x.push(s);
    }
    return x;
  };
  calc.dctKeep = function (row, k) {
    var X = calc.dct(row);
    var Xk = X.map(function (v, i) { return i < k ? v : 0; });
    var rec = calc.idct(Xk);
    return { coeffs: X, kept: Xk, rec: rec, mse: calc.mse(row, rec) };
  };

  calc.mdctWindow = function (N) { // sine window over 2N samples
    var w = [];
    for (var n = 0; n < 2 * N; n++) w.push(Math.sin(Math.PI * (n + 0.5) / (2 * N)));
    return w;
  };
  calc.mdctDelayMs = function (N, fs) { return N / (fs || 48000) * 1000; };
  calc.princenBradley = function (N) { // w[n]^2 + w[n+N]^2 for n in [0,N)
    var w = calc.mdctWindow(N), out = [];
    for (var n = 0; n < N; n++) out.push(sq(w[n]) + sq(w[n + N]));
    return out;
  };

  calc.haar = function (a) {
    var avg = [], det = [];
    for (var i = 0; i + 1 < a.length; i += 2) { avg.push((a[i] + a[i + 1]) / 2); det.push((a[i] - a[i + 1]) / 2); }
    return { avg: avg, det: det };
  };
  calc.haarPyramid = function (a, levels) {
    var out = [], cur = a;
    for (var l = 0; l < levels && cur.length >= 2; l++) { var h = calc.haar(cur); out.push(h); cur = h.avg; }
    return out;
  };

  calc.rgb2ycc = function (r, g, b) {
    var y = 0.299 * r + 0.587 * g + 0.114 * b;
    return { y: y, cb: 128 + 0.564 * (b - y), cr: 128 + 0.713 * (r - y) };
  };
  calc.ycc2rgb = function (y, cb, cr) {
    return { r: y + (cr - 128) / 0.713, g: y - (0.114 / 0.564 * (cb - 128) + 0.299 / 0.713 * (cr - 128)) / 0.587, b: y + (cb - 128) / 0.564 };
  };
  calc.midSide = function (l, r) { return { m: (l + r) / 2, s: l - r }; };

  calc.pcaPoints = function (n, seed) {
    var r = rng(seed || 11), pts = [];
    for (var i = 0; i < n; i++) { var x = r() * 2 - 1; pts.push([x, 0.8 * x + 0.1 * gauss(r)]); }
    return pts;
  };
  calc.pca = function (pts) {
    var n = pts.length, mx = 0, my = 0;
    pts.forEach(function (p) { mx += p[0]; my += p[1]; }); mx /= n; my /= n;
    var sxx = 0, syy = 0, sxy = 0;
    pts.forEach(function (p) { sxx += sq(p[0] - mx); syy += sq(p[1] - my); sxy += (p[0] - mx) * (p[1] - my); });
    sxx /= n; syy /= n; sxy /= n;
    var tr = sxx + syy, det = sxx * syy - sxy * sxy, disc = Math.sqrt(Math.max(0, tr * tr / 4 - det));
    var l1 = tr / 2 + disc, l2 = tr / 2 - disc;
    var angle = 0.5 * Math.atan2(2 * sxy, sxx - syy);
    return { mean: [mx, my], varX: sxx, varY: syy, cov: sxy, lambda: [l1, l2], angle: angle, axis: [Math.cos(angle), Math.sin(angle)] };
  };
  calc.rotate = function (pts, angle) {
    var c = Math.cos(angle), s = Math.sin(angle);
    return pts.map(function (p) { return [c * p[0] + s * p[1], -s * p[0] + c * p[1]]; });
  };
  calc.gridQuant = function (pts, bits) { // per-axis uniform quantizer with 2^bits levels spanning the axis range
    var L = Math.pow(2, bits), axes = [], err = 0;
    for (var d = 0; d < 2; d++) {
      var lo = Infinity, hi = -Infinity;
      pts.forEach(function (p) { lo = Math.min(lo, p[d]); hi = Math.max(hi, p[d]); });
      var step = (hi - lo) / L || 1e-9, edges = [];
      for (var i = 0; i <= L; i++) edges.push(lo + i * step);
      axes.push({ lo: lo, hi: hi, step: step, edges: edges });
    }
    var rec = pts.map(function (p) {
      return p.map(function (v, d) {
        var a = axes[d], idx = Math.min(L - 1, Math.max(0, Math.floor((v - a.lo) / a.step)));
        var r = a.lo + (idx + 0.5) * a.step; err += sq(v - r); return r;
      });
    });
    return { axes: axes, rec: rec, err: err };
  };

  calc.bwt = function (word) {
    var n = word.length, rots = [];
    for (var i = 0; i < n; i++) rots.push({ s: word.slice(i) + word.slice(0, i), i: i });
    var sorted = rots.slice().sort(function (a, b) { return a.s < b.s ? -1 : a.s > b.s ? 1 : 0; });
    var last = sorted.map(function (r) { return r.s[n - 1]; }).join('');
    var primary = sorted.findIndex(function (r) { return r.i === 0; });
    return { rotations: rots.map(function (r) { return r.s; }), sorted: sorted.map(function (r) { return r.s; }), last: last, primary: primary };
  };
  calc.runs = function (s) { var r = 0; for (var i = 0; i < s.length; i++) if (i === 0 || s[i] !== s[i - 1]) r++; return r; };

  calc.neuralEncWave = function (n, seed) {
    var r = rng(seed || 3), out = [];
    for (var i = 0; i < n; i++) out.push(0.6 * Math.sin(i * 0.11) + 0.3 * Math.sin(i * 0.37 + 1) + 0.08 * gauss(r));
    return out;
  };

  // ---- predict -------------------------------------------------------------
  // Greedy LZ77. Returns tokens for `text` only; `preload` is history already in the window.
  calc.lz77 = function (text, window, minMatch, preload) {
    preload = preload || ''; minMatch = minMatch || 3;
    var full = preload + text, start = preload.length, tokens = [], i = start;
    while (i < full.length) {
      var bestLen = 0, bestDist = 0;
      for (var j = Math.max(0, i - window); j < i; j++) {
        var len = 0;
        while (i + len < full.length && full[j + len] === full[i + len]) len++;
        if (len > bestLen || (len === bestLen && len > 0 && i - j < bestDist)) { bestLen = len; bestDist = i - j; }
      }
      if (bestLen >= minMatch) { tokens.push({ type: 'match', pos: i - start, dist: bestDist, len: bestLen, text: full.substr(i, bestLen) }); i += bestLen; }
      else { tokens.push({ type: 'lit', pos: i - start, ch: full[i] }); i++; }
    }
    var matches = tokens.filter(function (t) { return t.type === 'match'; });
    var covered = sum(matches.map(function (t) { return t.len; }));
    // a (dist,len) pair is charged 2 bytes in this toy, a literal 1 byte
    var cost = (tokens.length - matches.length) + 2 * matches.length;
    return { tokens: tokens, matches: matches.length, covered: covered, cost: cost, saved: text.length - cost };
  };

  calc.lzw = function (input) {
    var dict = {}, next = 256, w = '', codes = [], steps = [], added = [];
    for (var c = 0; c < 256; c++) dict[String.fromCharCode(c)] = c;
    for (var i = 0; i < input.length; i++) {
      var ch = input[i], wc = w + ch;
      if (dict.hasOwnProperty(wc)) { steps.push({ w: w, c: ch, out: null, add: null }); w = wc; }
      else {
        codes.push(dict[w]); dict[wc] = next; added.push({ code: next, phrase: wc });
        steps.push({ w: w, c: ch, out: dict[w], add: next + '=' + wc }); next++; w = ch;
      }
    }
    if (w !== '') { codes.push(dict[w]); steps.push({ w: w, c: '(end)', out: dict[w], add: null }); }
    return { codes: codes, steps: steps, added: added };
  };

  calc.delta = function (x, order) {
    var out = x.slice();
    for (var o = 0; o < order; o++) for (var i = x.length - 1; i > o; i--) out[i] = out[i] - out[i - 1];
    return out;
  };
  calc.bitsUnsigned = function (vals) { var m = 0; vals.forEach(function (v) { m = Math.max(m, Math.abs(v)); }); return Math.max(1, Math.ceil(log2(m + 1))); };
  calc.bitsSigned = function (vals) { var m = 0; vals.forEach(function (v) { m = Math.max(m, Math.abs(v)); }); return Math.ceil(log2(m + 1)) + 1; };

  calc.lpcSignal = function (n) {
    var x = [];
    for (var i = 0; i < (n || 16); i++) x.push(Math.round(100 * Math.pow(0.9, i) * Math.cos(0.6 * i)));
    return x;
  };
  calc.lpcFit2 = function (x) { // least squares for x[n] ≈ a1 x[n-1] + a2 x[n-2]
    var r11 = 0, r12 = 0, r22 = 0, b1 = 0, b2 = 0;
    for (var n = 2; n < x.length; n++) {
      r11 += x[n - 1] * x[n - 1]; r12 += x[n - 1] * x[n - 2]; r22 += x[n - 2] * x[n - 2];
      b1 += x[n] * x[n - 1]; b2 += x[n] * x[n - 2];
    }
    var det = r11 * r22 - r12 * r12;
    return { a1: (b1 * r22 - b2 * r12) / det, a2: (r11 * b2 - r12 * b1) / det };
  };
  calc.lpcResidual = function (x, a1, a2) {
    var r = [];
    for (var n = 2; n < x.length; n++) r.push(x[n] - (a1 * x[n - 1] + a2 * x[n - 2]));
    return r;
  };

  calc.pngFilters = function (block) {
    var H = block.length, W = block[0].length;
    function at(r, c) { return (r < 0 || c < 0) ? 0 : block[r][c]; }
    function paeth(a, b, c) { var p = a + b - c, pa = Math.abs(p - a), pb = Math.abs(p - b), pc = Math.abs(p - c); return (pa <= pb && pa <= pc) ? a : (pb <= pc ? b : c); }
    var res = { sub: [], up: [], paeth: [] }, sums = { sub: 0, up: 0, paeth: 0 };
    for (var r = 0; r < H; r++) {
      res.sub.push([]); res.up.push([]); res.paeth.push([]);
      for (var c = 0; c < W; c++) {
        var x = block[r][c], a = at(r, c - 1), b = at(r - 1, c), cc = at(r - 1, c - 1);
        var s = x - a, u = x - b, p = x - paeth(a, b, cc);
        res.sub[r].push(s); res.up[r].push(u); res.paeth[r].push(p);
        sums.sub += Math.abs(s); sums.up += Math.abs(u); sums.paeth += Math.abs(p);
      }
    }
    var winner = Object.keys(sums).reduce(function (a, b) { return sums[b] < sums[a] ? b : a; });
    return { residuals: res, sums: sums, winner: winner };
  };

  calc.frameWithSquare = function (n, x, y, size, val) {
    var f = [];
    for (var r = 0; r < n; r++) { f.push([]); for (var c = 0; c < n; c++) f[r].push((r >= y && r < y + size && c >= x && c < x + size) ? val : 0); }
    return f;
  };
  calc.shiftFrame = function (f, dx, dy) {
    var n = f.length, out = [];
    for (var r = 0; r < n; r++) { out.push([]); for (var c = 0; c < n; c++) { var sr = r - dy, sc = c - dx; out[r].push((sr >= 0 && sr < n && sc >= 0 && sc < n) ? f[sr][sc] : 0); } }
    return out;
  };
  calc.frameDiff = function (a, b) { return a.map(function (row, r) { return row.map(function (v, c) { return v - b[r][c]; }); }); };
  calc.motionSearch = function (prev, cur, range) {
    var best = null;
    for (var dy = -range; dy <= range; dy++) for (var dx = -range; dx <= range; dx++) {
      var pred = calc.shiftFrame(prev, dx, dy), sad = 0;
      for (var r = 0; r < cur.length; r++) for (var c = 0; c < cur.length; c++) sad += Math.abs(cur[r][c] - pred[r][c]);
      if (!best || sad < best.sad) best = { dx: dx, dy: dy, sad: sad, pred: pred };
    }
    best.residual = calc.frameDiff(cur, best.pred);
    return best;
  };
  calc.countNonzero = function (m) { var n = 0; m.forEach(function (row) { row.forEach(function (v) { if (v !== 0) n++; }); }); return n; };

  calc.contextCounts = function (text, ctx) {
    var counts = {}, total = 0;
    for (var i = ctx.length; i < text.length; i++) if (text.substr(i - ctx.length, ctx.length) === ctx) { counts[text[i]] = (counts[text[i]] || 0) + 1; total++; }
    return { counts: counts, total: total };
  };
  calc.laplace = function (count, total, alphabet) { return (count + 1) / (total + alphabet); };
  calc.cost = function (p) { return -log2(p); };

  // ---- quantize ------------------------------------------------------------
  function roundSym(v) { return Math.sign(v) * Math.round(Math.abs(v)); } // half away from zero
  calc.uniformQ = function (xs, delta, range) {
    var q = xs.map(function (x) { return roundSym(x / delta); });
    var rec = q.map(function (v) { return v * delta; });
    var levels = 2 * Math.floor((range || 9) / delta) + 1;
    return { q: q, rec: rec, err: xs.map(function (x, i) { return rec[i] - x; }), mse: calc.mse(xs, rec), theory: delta * delta / 12, levels: levels, bits: Math.ceil(log2(levels)) };
  };

  calc.JPEG_Q50 = [
    [16, 11, 10, 16, 24, 40, 51, 61], [12, 12, 14, 19, 26, 58, 60, 55], [14, 13, 16, 24, 40, 57, 69, 56], [14, 17, 22, 29, 51, 87, 80, 62],
    [18, 22, 37, 56, 68, 109, 103, 77], [24, 35, 55, 64, 81, 104, 113, 92], [49, 64, 78, 87, 103, 121, 120, 101], [72, 92, 95, 98, 112, 100, 103, 99]];
  calc.WIKI_DCT = [ // Wikipedia JPEG article, DCT of the 8×8 example block
    [-415.38, -30.19, -61.20, 27.24, 56.13, -20.10, -2.39, 0.46], [4.47, -21.86, -60.76, 10.25, 13.15, -7.09, -8.54, 4.88],
    [-46.83, 7.37, 77.13, -24.56, -28.91, 9.93, 5.42, -5.65], [-48.53, 12.07, 34.10, -14.76, -10.24, 6.30, 1.83, 1.95],
    [12.12, -6.55, -13.20, -3.95, -1.87, 1.75, -2.79, 3.14], [-7.73, 2.91, 2.38, -5.94, -2.38, 0.94, 4.30, 1.85],
    [-1.03, 0.18, 0.42, -2.42, -0.88, -3.02, 4.12, -0.66], [-0.17, 0.14, -1.07, -4.19, -1.17, -0.10, 0.50, 1.68]];
  calc.quantizeBlock = function (coeffs, table, scale) {
    var zeros = 0, q = coeffs.map(function (row, r) { return row.map(function (v, c) { var s = Math.round(v / (table[r][c] * scale)); if (s === 0) zeros++; return s; }); });
    return { q: q, zeros: zeros };
  };

  calc.mulaw = function (x) { return Math.sign(x) * Math.log(1 + 255 * Math.abs(x)) / Math.log(256); };
  calc.mulawInv = function (y) { return Math.sign(y) * (Math.pow(256, Math.abs(y)) - 1) / 255; };
  calc.mulawQuant = function (x) { // sign + 7-bit magnitude
    var code = Math.round(calc.mulaw(x) * 127), rec = calc.mulawInv(code / 127);
    return { code: code, rec: rec, relErr: Math.abs(rec - x) / Math.abs(x), worst: (Math.pow(256, 1 / 254) - 1) };
  };
  calc.uniformQuant8 = function (x) { // 256 levels over [-1,1), Δ = 1/128
    var d = 1 / 128, q = Math.round(x / d), rec = q * d;
    return { code: q, rec: rec, relErr: Math.abs(rec - x) / Math.abs(x), worst: (d / 2) / Math.abs(x), delta: d };
  };

  calc.cityPoints = function (seed) {
    var r = rng(seed || 42), centers = [[-0.6, -0.5], [0.45, 0.6], [0.6, -0.55]], pts = [];
    for (var k = 0; k < 3; k++) for (var i = 0; i < 20; i++) pts.push([centers[k][0] + 0.13 * gauss(r), centers[k][1] + 0.13 * gauss(r)]);
    return pts;
  };
  calc.kmeans = function (pts, K, iters, seed) {
    var r = rng(seed || 7), cents = [pts[Math.floor(r() * pts.length)].slice()];
    while (cents.length < K) { // k-means++ seeding, deterministic
      var d2 = pts.map(function (p) { var m = Infinity; cents.forEach(function (c) { m = Math.min(m, dist2(p, c)); }); return m; });
      var t = r() * sum(d2), idx = d2.length - 1;
      for (var i = 0; i < d2.length; i++) { t -= d2[i]; if (t <= 0) { idx = i; break; } }
      cents.push(pts[idx].slice());
    }
    var assign = pts.map(function () { return 0; });
    for (var it = 0; it < (iters || 10); it++) {
      pts.forEach(function (p, i) { var b = 0, bd = Infinity; cents.forEach(function (c, k) { var d = dist2(p, c); if (d < bd) { bd = d; b = k; } }); assign[i] = b; });
      var acc = cents.map(function () { return [0, 0, 0]; });
      pts.forEach(function (p, i) { var a = acc[assign[i]]; a[0] += p[0]; a[1] += p[1]; a[2]++; });
      acc.forEach(function (a, k) { if (a[2] > 0) cents[k] = [a[0] / a[2], a[1] / a[2]]; });
    }
    var dist = 0;
    pts.forEach(function (p, i) { var b = 0, bd = Infinity; cents.forEach(function (c, k) { var d = dist2(p, c); if (d < bd) { bd = d; b = k; } }); assign[i] = b; dist += bd; });
    return { centroids: cents, assign: assign, distortion: dist };
  };
  calc.rvq = function (pts, stages, K) {
    var res = pts.map(function (p) { return p.slice(); }), out = [];
    for (var s = 0; s < stages; s++) {
      var km = calc.kmeans(res, K, 10, 7 + s);
      var next = res.map(function (p, i) { var c = km.centroids[km.assign[i]]; return [p[0] - c[0], p[1] - c[1]]; });
      out.push({ stage: s + 1, centroids: km.centroids, assign: km.assign, input: res, distortion: sum(next.map(function (p) { return dist2(p, [0, 0]); })), bits: 2 * (s + 1), effective: Math.pow(K, s + 1), stored: K * (s + 1) });
      res = next;
    }
    return out;
  };
  calc.PQ_BOOKS = [[[-0.7, -0.5], [0.8, 0.2], [0.1, 0.9], [-0.3, 0.4]], [[0.6, 0.6], [-0.4, 0.7], [0.3, -0.8], [-0.8, -0.2]]];
  calc.pqEncode = function (vec, books) {
    var halves = [vec.slice(0, 2), vec.slice(2, 4)], idx = [], rec = [];
    halves.forEach(function (h, b) { var best = 0, bd = Infinity; books[b].forEach(function (c, i) { var d = dist2(h, c); if (d < bd) { bd = d; best = i; } }); idx.push(best); rec = rec.concat(books[b][best]); });
    return { halves: halves, idx: idx, rec: rec, err: dist2(vec, rec) };
  };

  calc.blockQuant = function (x) { // absmax int4: s = max|x|/7, q in [-8,7]
    var amax = Math.max.apply(null, x.map(Math.abs)), s = amax / 7;
    var q = x.map(function (v) { return Math.max(-8, Math.min(7, Math.round(v / s))); });
    var rec = q.map(function (v) { return v * s; });
    return { s: s, q: q, rec: rec, err: x.map(function (v, i) { return rec[i] - v; }), sse: sum(x.map(function (v, i) { return sq(rec[i] - v); })) };
  };
  calc.hadamard8 = function (x) { // orthonormal Sylvester Hadamard, its own inverse
    var out = [];
    for (var i = 0; i < 8; i++) { var s = 0; for (var j = 0; j < 8; j++) { var b = i & j, pop = 0; while (b) { pop += b & 1; b >>= 1; } s += (pop % 2 ? -1 : 1) * x[j]; } out.push(s / Math.sqrt(8)); }
    return out;
  };
  calc.blockQuantRotated = function (x) {
    var y = calc.hadamard8(x), bq = calc.blockQuant(y), rec = calc.hadamard8(bq.rec);
    return { rotated: y, s: bq.s, q: bq.q, recRotated: bq.rec, rec: rec, err: x.map(function (v, i) { return rec[i] - v; }), sse: sum(x.map(function (v, i) { return sq(rec[i] - v); })) };
  };

  // ---- entropy -------------------------------------------------------------
  calc.huffman = function (freqs) { // freqs: {sym: weight}; deterministic tie-break by insertion order
    var nodes = Object.keys(freqs).map(function (s, i) { return { sym: s, w: freqs[s], id: i }; }), nid = nodes.length;
    if (nodes.length === 1) nodes[0] = { w: nodes[0].w, left: nodes[0], right: null, id: nid++ };
    while (nodes.length > 1) {
      nodes.sort(function (a, b) { return a.w - b.w || a.id - b.id; });
      var a = nodes.shift(), b = nodes.shift();
      nodes.push({ w: a.w + b.w, left: a, right: b, id: nid++ });
    }
    var tree = nodes[0], codes = {};
    (function walk(n, code) { if (n.sym !== undefined) { codes[n.sym] = code || '0'; return; } if (n.left) walk(n.left, code + '0'); if (n.right) walk(n.right, code + '1'); })(tree, '');
    var total = sum(Object.keys(freqs).map(function (s) { return freqs[s]; })), mean = 0, probs = [], lengths = {};
    Object.keys(freqs).forEach(function (s) { var p = freqs[s] / total; probs.push(p); lengths[s] = codes[s].length; mean += p * codes[s].length; });
    return { tree: tree, codes: codes, lengths: lengths, mean: mean, entropy: calc.entropy(probs) };
  };

  calc.arith = function (msg, model) { // model: [{sym, p}] in fixed order
    var lo = 0, hi = 1, steps = [{ lo: 0, hi: 1, sym: null }], bits = 0;
    for (var i = 0; i < msg.length; i++) {
      var w = hi - lo, cum = 0, picked = null;
      for (var k = 0; k < model.length; k++) { if (model[k].sym === msg[i]) { picked = { lo: lo + cum * w, hi: lo + (cum + model[k].p) * w, p: model[k].p }; break; } cum += model[k].p; }
      bits += -log2(picked.p); lo = picked.lo; hi = picked.hi;
      steps.push({ lo: lo, hi: hi, sym: msg[i], p: picked.p });
    }
    return { steps: steps, width: hi - lo, bits: -log2(hi - lo), sumBits: bits, model: model };
  };

  calc.ransEncode = function (msg, freq, cum, M, x0) {
    var x = x0, steps = [{ x: x, sym: null }];
    for (var i = 0; i < msg.length; i++) {
      var s = msg[i], f = freq[s], nx = Math.floor(x / f) * M + cum[s] + (x % f);
      steps.push({ sym: s, x: nx, from: x, f: f, cum: cum[s], gain: log2(nx) - log2(x) });
      x = nx;
    }
    return { x: x, steps: steps };
  };
  calc.ransDecode = function (x, n, freq, cum, M) {
    var out = [], steps = [], syms = Object.keys(freq);
    for (var i = 0; i < n; i++) {
      var slot = x % M, s = syms.find(function (k) { return slot >= cum[k] && slot < cum[k] + freq[k]; });
      var nx = freq[s] * Math.floor(x / M) + slot - cum[s];
      steps.push({ from: x, slot: slot, sym: s, x: nx }); out.unshift(s); x = nx;
    }
    return { symbols: out, steps: steps, x: x };
  };

  calc.rice = function (n, k) {
    var q = n >> k, low = k > 0 ? (n & ((1 << k) - 1)).toString(2).padStart(k, '0') : '';
    return { q: q, unary: '1'.repeat(q) + '0', low: low, bits: q + 1 + k };
  };
  calc.riceTotal = function (vals, k) { return sum(vals.map(function (v) { return calc.rice(v, k).bits; })); };

  calc.zigzag = function (n) {
    var order = [];
    for (var s = 0; s < 2 * n - 1; s++) {
      var cells = [];
      for (var r = 0; r < n; r++) { var c = s - r; if (c >= 0 && c < n) cells.push([r, c]); }
      if (s % 2 === 0) cells.reverse(); // even diagonals go up-right (0,0) (1,0) (0,1) (0,2) (1,1) (2,0) ...
      order = order.concat(cells);
    }
    return order;
  };
  calc.rowMajor = function (n) { var o = []; for (var r = 0; r < n; r++) for (var c = 0; c < n; c++) o.push([r, c]); return o; };
  calc.rlePairs = function (seq) {
    var pairs = [], run = 0, lastNz = -1;
    seq.forEach(function (v, i) { if (v !== 0) lastNz = i; });
    for (var i = 0; i <= lastNz; i++) { if (seq[i] === 0) run++; else { pairs.push({ run: run, val: seq[i] }); run = 0; } }
    pairs.push({ run: null, val: 'EOB' });
    return pairs;
  };

  // ---- perceptual ----------------------------------------------------------
  calc.maskingCurve = function (nBands, maskerBand, level, bg) {
    var curve = [], free = [];
    for (var b = 0; b < nBands; b++) {
      var d = b - maskerBand, m = level - 12 - (d < 0 ? 25 * -d : 10 * d);
      curve.push(m); free.push(m > bg);
    }
    return { curve: curve, free: free, freeCount: free.filter(Boolean).length };
  };

  calc.subsample420 = function (block) { // block: 4 [r,g,b]
    var ycc = block.map(function (p) { return calc.rgb2ycc(p[0], p[1], p[2]); });
    var cb = sum(ycc.map(function (c) { return c.cb; })) / 4, cr = sum(ycc.map(function (c) { return c.cr; })) / 4;
    var rec = ycc.map(function (c) { var p = calc.ycc2rgb(c.y, cb, cr); return [p.r, p.g, p.b]; });
    var err = rec.map(function (p, i) { return Math.sqrt(dist2(p, block[i]) / 3); });
    return { ycc: ycc, cb: cb, cr: cr, rec: rec, err: err, maxErr: Math.max.apply(null, err) };
  };

  calc.ganWave = function () {
    var w = [];
    for (var n = 0; n < 32; n++) { var win = 0.5 - 0.5 * Math.cos(2 * Math.PI * n / 31); w.push(win * (Math.sin(2 * Math.PI * 3 * n / 32) + 0.5 * Math.sin(2 * Math.PI * 7 * n / 32))); }
    return w;
  };
  calc.smooth = function (w, taps) {
    var h = Math.floor(taps / 2);
    return w.map(function (_, i) { var s = 0, c = 0; for (var k = -h; k <= h; k++) { var j = i + k; if (j >= 0 && j < w.length) { s += w[j]; c++; } } return s / c; });
  };
  calc.shiftWave = function (w, k) { return w.map(function (_, i) { return (i - k >= 0 && i - k < w.length) ? w[i - k] : 0; }); };

  // ---- prior ---------------------------------------------------------------
  calc.DEFLATE_FIXED = [
    { range: '0–143', bits: 8, codes: '00110000 … 10111111' }, { range: '144–255', bits: 9, codes: '110010000 … 111111111' },
    { range: '256–279', bits: 7, codes: '0000000 … 0010111' }, { range: '280–287', bits: 8, codes: '11000000 … 11000111' }];
  calc.tableBreakEven = function (payloadBytes, tableBytes, gainBitsPerByte) {
    var gainBytes = payloadBytes * gainBitsPerByte / 8;
    return { gainBytes: gainBytes, net: gainBytes - tableBytes, breakEven: tableBytes * 8 / gainBitsPerByte, pays: gainBytes > tableBytes };
  };
  calc.codebookCost = function (K, d) { return { sideBytes: K * d * 2, bitsPerVector: log2(K) }; };
  calc.decoderNumbers = function () {
    var pcm = 24000 * 16, tok = 75 * 8 * 10, bytesPerSec = tok / 8, weights = 20e6;
    return { pcmBps: pcm, tokenBps: tok, ratio: pcm / tok, bytesPerSec: bytesPerSec, weights: weights, priorRatio: weights / bytesPerSec };
  };

  // ---- level 5: KV cache ---------------------------------------------------
  // Generic absmax integer quantizer: levels [-(L+1), L] with L = 2^(bits-1)-1, s = max|x|/L.
  calc.absmaxQuant = function (xs, bits) {
    var L = Math.pow(2, bits - 1) - 1, amax = Math.max.apply(null, xs.map(Math.abs)), s = amax / L || 1e-12;
    var q = xs.map(function (v) { return Math.max(-L - 1, Math.min(L, roundSym(v / s))); }), rec = q.map(function (v) { return v * s; });
    var err = xs.map(function (v, i) { return rec[i] - v; }), sse = sum(err.map(sq));
    return { s: s, q: q, rec: rec, err: err, sse: sse, rms: Math.sqrt(sse / xs.length), levels: 2 * L + 2 };
  };
  calc.rope = function (pair, m, theta) { var c = Math.cos(m * theta), s = Math.sin(m * theta); return [c * pair[0] - s * pair[1], s * pair[0] + c * pair[1]]; };
  calc.unropeLab = function (key, n, theta, bits) {
    var rotated = [], rewound = [];
    for (var m = 0; m < n; m++) { var r = calc.rope(key, m, theta); rotated.push(r); rewound.push(calc.rope(r, m, -theta)); }
    function chan(pts, d) { return pts.map(function (p) { return p[d]; }); }
    var qb = { x: calc.absmaxQuant(chan(rotated, 0), bits), y: calc.absmaxQuant(chan(rotated, 1), bits) };
    var qa = { x: calc.absmaxQuant(chan(rewound, 0), bits), y: calc.absmaxQuant(chan(rewound, 1), bits) };
    function range(a) { return Math.max.apply(null, a) - Math.min.apply(null, a); }
    return { rotated: rotated, rewound: rewound, before: qb, after: qa, rangeBefore: [range(chan(rotated, 0)), range(chan(rotated, 1))], rangeAfter: [range(chan(rewound, 0)), range(chan(rewound, 1))] };
  };

  calc.symEig = function (A) { // Jacobi; returns eigenvalues desc and eigenvectors (vectors[k] is the k-th eigenvector)
    var n = A.length, a = A.map(function (r) { return r.slice(); }), V = [], i, j, k;
    for (i = 0; i < n; i++) { V.push([]); for (j = 0; j < n; j++) V[i].push(i === j ? 1 : 0); }
    for (var sweep = 0; sweep < 60; sweep++) {
      var off = 0; for (i = 0; i < n; i++) for (j = i + 1; j < n; j++) off += a[i][j] * a[i][j];
      if (off < 1e-24) break;
      for (var p = 0; p < n; p++) for (var q = p + 1; q < n; q++) {
        if (Math.abs(a[p][q]) < 1e-300) continue;
        var th = (a[q][q] - a[p][p]) / (2 * a[p][q]), t = (th >= 0 ? 1 : -1) / (Math.abs(th) + Math.sqrt(th * th + 1)), c = 1 / Math.sqrt(t * t + 1), s = t * c;
        for (k = 0; k < n; k++) { var akp = a[k][p], akq = a[k][q]; a[k][p] = c * akp - s * akq; a[k][q] = s * akp + c * akq; }
        for (k = 0; k < n; k++) { var apk = a[p][k], aqk = a[q][k]; a[p][k] = c * apk - s * aqk; a[q][k] = s * apk + c * aqk; }
        for (k = 0; k < n; k++) { var vkp = V[k][p], vkq = V[k][q]; V[k][p] = c * vkp - s * vkq; V[k][q] = s * vkp + c * vkq; }
      }
    }
    var idx = []; for (i = 0; i < n; i++) idx.push(i);
    idx.sort(function (x, y) { return a[y][y] - a[x][x]; });
    return { values: idx.map(function (i) { return a[i][i]; }), vectors: idx.map(function (i) { return V.map(function (row) { return row[i]; }); }) };
  };
  calc.singularValues = function (M) { // M: m×n with m ≥ n
    var n = M[0].length, G = [];
    for (var i = 0; i < n; i++) { G.push([]); for (var j = 0; j < n; j++) { var s = 0; M.forEach(function (row) { s += row[i] * row[j]; }); G[i].push(s); } }
    var e = calc.symEig(G);
    return { sigma: e.values.map(function (v) { return Math.sqrt(Math.max(0, v)); }), V: e.vectors, total: sum(e.values.map(function (v) { return Math.max(0, v); })) };
  };
  calc.lowRank = function (M, r) {
    var sv = calc.singularValues(M), tail = 0;
    for (var i = r; i < sv.sigma.length; i++) tail += sv.sigma[i] * sv.sigma[i];
    var stored = M.length * r + M[0].length * r;
    return { sigma: sv.sigma, relErr: Math.sqrt(tail / sv.total), stored: stored, full: M.length * M[0].length };
  };
  calc.svdLabMatrices = function () {
    var u1 = [1, 0.5, -0.3, 0.8], u2 = [-0.2, 0.9, 0.4, -0.1], coef = [[1.0, 0.2], [0.8, -0.5], [-0.6, 0.9], [1.2, 0.4], [-0.9, -0.3], [0.3, 1.1], [0.7, -0.8], [-1.1, 0.6]];
    var r1 = rng(21), K = coef.map(function (c) { return u1.map(function (v, j) { return Math.round((c[0] * v + c[1] * u2[j] + 0.05 * gauss(r1)) * 100) / 100; }); });
    var r2 = rng(22), Vm = coef.map(function () { return [0, 0, 0, 0].map(function () { return Math.round(gauss(r2) * 100) / 100; }); });
    return { K: K, V: Vm };
  };

  calc.mla = function () {
    var per = { mha: 128 * 128 * 2, gqa8: 8 * 128 * 2, mla: 512 + 64 }, tokens = 131072;
    var presets = [
      { name: 'Llama-3-70B (GQA-8)', layers: 80, perLayer: per.gqa8 },
      { name: 'DeepSeek-V3 (MLA)', layers: 61, perLayer: per.mla }];
    presets.forEach(function (p) { p.bytesPerToken = p.layers * p.perLayer * 2; p.at128k = p.bytesPerToken * tokens; });
    return { per: per, ratio: per.mha / per.mla, presets: presets, tokens: tokens };
  };

  calc.anchorDelta = function (x, delta) {
    var chain = x.map(function (v, i) { return i === 0 ? v : v - x[i - 1]; }), anchor = x.map(function (v, i) { return i === 0 ? v : v - x[0]; });
    var qc = chain.map(function (d, i) { return i === 0 ? d : roundSym(d / delta) * delta; }), qa = anchor.map(function (d, i) { return i === 0 ? d : roundSym(d / delta) * delta; });
    var recC = [], recA = [];
    for (var i = 0; i < x.length; i++) { recC.push(i === 0 ? x[0] : recC[i - 1] + qc[i]); recA.push(i === 0 ? x[0] : x[0] + qa[i]); }
    var errC = recC.map(function (v, i) { return v - x[i]; }), errA = recA.map(function (v, i) { return v - x[i]; });
    return { chain: chain, anchor: anchor, qChain: qc, qAnchor: qa, recChain: recC, recAnchor: recA, errChain: errC, errAnchor: errA, maxChain: Math.max.apply(null, errC.map(Math.abs)), maxAnchor: Math.max.apply(null, errA.map(Math.abs)) };
  };

  calc.COUPLED_PAIRS = [[0.82, 0.71], [-0.45, -0.38], [1.21, 1.12], [-1.05, -0.99], [0.33, 0.35], [-0.78, -0.66], [1.52, 1.34], [-0.21, -0.24], [0.64, 0.55], [-1.33, -1.17], [0.97, 0.91], [-0.58, -0.49], [0.15, 0.09], [-0.92, -0.86], [1.08, 0.99], [-0.37, -0.30]];
  calc.kmeansInit = function (pts, init, iters) {
    var cents = init.map(function (c) { return c.slice(); }), assign = pts.map(function () { return 0; }), dist = 0;
    function assignAll() { dist = 0; pts.forEach(function (p, i) { var b = 0, bd = Infinity; cents.forEach(function (c, k) { var d = dist2(p, c); if (d < bd) { bd = d; b = k; } }); assign[i] = b; dist += bd; }); }
    for (var it = 0; it < iters; it++) {
      assignAll();
      var acc = cents.map(function () { return [0, 0, 0]; });
      pts.forEach(function (p, i) { var a = acc[assign[i]]; a[0] += p[0]; a[1] += p[1]; a[2]++; });
      acc.forEach(function (a, k) { if (a[2] > 0) cents[k] = [a[0] / a[2], a[1] / a[2]]; });
    }
    assignAll();
    return { centroids: cents, assign: assign, distortion: dist, mse: dist / pts.length };
  };
  calc.signQuant2 = function (pts) {
    var ma = sum(pts.map(function (p) { return Math.abs(p[0]); })) / pts.length, mb = sum(pts.map(function (p) { return Math.abs(p[1]); })) / pts.length;
    var rec = pts.map(function (p) { return [Math.sign(p[0]) * ma, Math.sign(p[1]) * mb]; }), d = sum(pts.map(function (p, i) { return dist2(p, rec[i]); }));
    return { levels: [ma, mb], rec: rec, distortion: d, mse: d / pts.length, points: [[ma, mb], [ma, -mb], [-ma, mb], [-ma, -mb]] };
  };
  calc.binEntropy = function (pts, bins) { // marginal and joint entropies over equal-width bins per axis
    function edges(d) { var lo = Infinity, hi = -Infinity; pts.forEach(function (p) { lo = Math.min(lo, p[d]); hi = Math.max(hi, p[d]); }); return { lo: lo, w: (hi - lo) / bins || 1 }; }
    var ex = edges(0), ey = edges(1), ha = {}, hb = {}, hab = {};
    pts.forEach(function (p) {
      var i = Math.min(bins - 1, Math.floor((p[0] - ex.lo) / ex.w)), j = Math.min(bins - 1, Math.floor((p[1] - ey.lo) / ey.w));
      ha[i] = (ha[i] || 0) + 1; hb[j] = (hb[j] || 0) + 1; hab[i + ',' + j] = (hab[i + ',' + j] || 0) + 1;
    });
    function H(h) { return calc.entropy(Object.keys(h).map(function (k) { return h[k] / pts.length; })); }
    return { Ha: H(ha), Hb: H(hb), Hab: H(hab) };
  };

  calc.waterfill = function (vars, theta) {
    var bits = vars.map(function (v) { return Math.max(0, 0.5 * log2(v / theta)); }), dist = vars.map(function (v) { return Math.min(v, theta); });
    var R = sum(bits), eq = R / vars.length, eqDist = sum(vars.map(function (v) { return v * Math.pow(2, -2 * eq); }));
    return { bits: bits, dist: dist, totalBits: R, totalDist: sum(dist), equalBits: eq, equalDist: eqDist };
  };

  calc.FP4_GRID = (function () { var g = [0]; [0.5, 1, 1.5, 2, 3, 4, 6].forEach(function (v) { g.push(v); g.push(-v); }); return g.sort(function (a, b) { return a - b; }); })();
  calc.FP4_BLOCK = [0.031, -0.012, 0.045, 0.008, -0.027, 0.019, -0.041, 0.004, 0.42, -0.015, 0.036, -0.033, 0.022, -0.006, 0.048, -0.029];
  calc.fp4Quant = function (xs) {
    var s = Math.max.apply(null, xs.map(Math.abs)) / 6, q = xs.map(function (v) { var best = 0, bd = Infinity; calc.FP4_GRID.forEach(function (g) { var d = Math.abs(v / s - g); if (d < bd) { bd = d; best = g; } }); return best; });
    var rec = q.map(function (v) { return v * s; }), err = xs.map(function (v, i) { return rec[i] - v; });
    return { s: s, q: q, rec: rec, err: err, sse: sum(err.map(sq)), bits: (16 * 4 + 8) / 16 };
  };
  calc.int4Block = function (xs) { var r = calc.absmaxQuant(xs, 4); r.bits = (32 * 4 + 16) / 32; return r; };

  calc.escape = function (xs, bits) {
    var plain = calc.absmaxQuant(xs, bits), oi = 0;
    xs.forEach(function (v, i) { if (Math.abs(v) > Math.abs(xs[oi])) oi = i; });
    var rest = xs.filter(function (_, i) { return i !== oi; }), qr = calc.absmaxQuant(rest, bits), rec = [], k = 0;
    xs.forEach(function (v, i) { rec.push(i === oi ? v : qr.rec[k++]); });
    var err = xs.map(function (v, i) { return rec[i] - v; });
    var idxBits = Math.ceil(log2(xs.length));
    return { plain: plain, outlierIndex: oi, restScale: qr.s, rec: rec, err: err, sse: sum(err.map(sq)), bitsPlain: bits, bitsEscape: bits + (16 + idxBits) / xs.length, idxBits: idxBits, bitsOnePercent: bits + 0.01 * (16 + idxBits) };
  };

  calc.EVICT_W = [0.30, 0.02, 0.05, 0.03, 0.01, 0.06, 0.02, 0.04, 0.03, 0.05, 0.04, 0.05, 0.075, 0.075, 0.075, 0.075];
  calc.evict = function (w, k, pageSize) {
    var order = w.map(function (v, i) { return i; }).sort(function (a, b) { return w[b] - w[a] || a - b; }), kept = order.slice(0, k), keep = w.map(function () { return false; });
    kept.forEach(function (i) { keep[i] = true; });
    var pages = [];
    for (var p = 0; p < w.length / pageSize; p++) { var n = 0; for (var i = 0; i < pageSize; i++) if (keep[p * pageSize + i]) n++; pages.push({ kept: n, freed: n === 0 }); }
    return { keep: keep, mass: sum(w.filter(function (_, i) { return keep[i]; })), memory: k / w.length, pages: pages, pagesFreed: pages.filter(function (p) { return p.freed; }).length };
  };

  calc.sinkWeights = function () {
    var r = rng(9), mid = [], w = [];
    for (var i = 0; i < 27; i++) mid.push(0.3 + r());
    var ms = sum(mid);
    for (i = 0; i < 32; i++) w.push(i === 0 ? 0.35 : i >= 28 ? 0.075 : 0.35 * mid[i - 1] / ms);
    return w;
  };
  calc.sinkguard = function (w, exempt, e) {
    var exemptSet = {}; (exempt || []).forEach(function (i) { exemptSet[i] = true; });
    var err = w.map(function (v, i) { return exemptSet[i] ? 0 : e; });
    return { weighted: sum(w.map(function (v, i) { return v * err[i]; })), exemptMass: sum(w.filter(function (_, i) { return exemptSet[i]; })), exempt: exemptSet };
  };

  calc.spotKeys = function () {
    var r = rng(77), keys = [];
    for (var i = 0; i < 32; i++) { var x = 8 * r() - 4, y = 8 * r() - 4; if (x + 0.4 * y > 1.5) x = -x; keys.push([Math.round(x * 100) / 100, Math.round(y * 100) / 100]); }
    keys[5] = [3.8, 1.6]; keys[13] = [4.1, 1.9]; keys[20] = [3.6, 2.1]; keys[27] = [4.0, 1.3];
    return keys;
  };
  calc.softmax = function (s) { var m = Math.max.apply(null, s), e = s.map(function (v) { return Math.exp(v - m); }), z = sum(e); return e.map(function (v) { return v / z; }); };
  calc.spotlight = function (keys, q, k) {
    var exact = calc.softmax(keys.map(function (kk) { return q[0] * kk[0] + q[1] * kk[1]; }));
    var proxyKeys = keys.map(function (kk) { var m = (Math.abs(kk[0]) + Math.abs(kk[1])) / 2; return [Math.sign(kk[0]) * m, Math.sign(kk[1]) * m]; });
    var proxy = calc.softmax(proxyKeys.map(function (kk) { return q[0] * kk[0] + q[1] * kk[1]; }));
    function topk(p) { return p.map(function (v, i) { return i; }).sort(function (a, b) { return p[b] - p[a] || a - b; }).slice(0, k); }
    var te = topk(exact), tp = topk(proxy), overlap = te.filter(function (i) { return tp.indexOf(i) >= 0; }).length;
    return { exact: exact, proxy: proxy, proxyKeys: proxyKeys, topExact: te, topProxy: tp, overlap: overlap, massExact: sum(te.map(function (i) { return exact[i]; })), massProxy: sum(tp.map(function (i) { return exact[i]; })) };
  };

  calc.basisBytes = function (d, layers) { var one = d * d * 2; return { shared: one, perLayer: one * layers, kvtcParams: 0.024 * 70e9, kvtcBytes: 0.024 * 70e9 * 2 }; };
  calc.tiered = function (B, concurrency) {
    var miss = { 2048: 0.30, 4096: 0.134, 8192: 0.067 }[B], layers = 78, k = 2048, entry = 656, step = 0.030;
    var perToken = layers * k * entry * miss, bw = perToken / step * concurrency;
    return { miss: miss, perToken: perToken, bandwidth: bw, pcie: 64e9, gather: 37e9, gpuBytes: B * layers * entry, dense128k: 13e9, layers: layers, k: k, entry: entry, step: step };
  };

  // ===========================================================================
  // DOM helpers (only called from render)
  // ===========================================================================
  var CSS = [
    '.cyc-demo{font-family:inherit;color:var(--fg,var(--ink,#2a211b));line-height:1.3;font-size:max(13px,.84em);max-width:100%;overflow:hidden}',
    '.cyc-demo *{box-sizing:border-box}',
    '.cyc-demo .cyc-row{display:flex;flex-wrap:wrap;gap:8px 12px;align-items:flex-start;max-width:100%}',
    '.cyc-demo .cyc-row>*{min-width:0;max-width:100%}',
    '.cyc-demo .cyc-col{display:flex;flex-direction:column;gap:6px;min-width:0}',
    '.cyc-demo .cyc-scroll{overflow-x:auto;max-width:100%;scrollbar-width:thin}',
    '.cyc-demo .cyc-h{font-size:.8em;letter-spacing:.08em;font-variant:small-caps;color:var(--ink-2,#5a4a3d);margin:0}',
    '.cyc-demo table.cyc-strip{border-collapse:collapse;font-variant-numeric:tabular-nums;font-family:var(--font-mono,"JetBrains Mono",ui-monospace,Menlo,monospace)}',
    '.cyc-demo table.cyc-strip td,.cyc-demo table.cyc-strip th{padding:1px 5px;text-align:right;white-space:nowrap;font-weight:normal}',
    '.cyc-demo table.cyc-strip th{text-align:left;opacity:.7}',
    '.cyc-demo table.cyc-strip td.hi{color:var(--role-color);font-weight:bold}',
    '.cyc-demo table.cyc-strip td.dim{opacity:.4}',
    '.cyc-demo .cyc-matrix{display:inline-grid;gap:2px;font-variant-numeric:tabular-nums;font-family:var(--font-mono,"JetBrains Mono",ui-monospace,Menlo,monospace)}',
    '.cyc-demo .cyc-cell{padding:1px 4px;text-align:right;min-width:2.2em;background:var(--paper-3,#e0d2b6);position:relative}',
    '.cyc-demo .cyc-cell.zero{opacity:.3}',
    '.cyc-demo .cyc-cell.hi{outline:1px solid var(--role-color);color:var(--role-color)}',
    '.cyc-demo canvas{display:block;max-width:100%}',
    '.cyc-demo .cyc-ctrl{display:flex;align-items:center;gap:8px;margin:6px 0;flex-wrap:wrap}',
    '.cyc-demo .cyc-ctrl label{opacity:.85}',
    '.cyc-demo input[type=range]{accent-color:var(--role-color);width:150px;margin:0}',
    '.cyc-demo .cyc-readout{font-variant-numeric:tabular-nums;min-width:3.5em;color:var(--role-color)}',
    '.cyc-demo .cyc-toggle button{font:inherit;background:transparent;color:inherit;border:1px solid currentColor;opacity:.55;padding:0 8px;cursor:pointer;margin-right:4px;border-radius:2px}',
    '.cyc-demo .cyc-toggle button.on{opacity:1;border-color:var(--role-color);color:var(--role-color)}',
    '.cyc-demo .cyc-note{margin-top:8px;font-size:.9em;opacity:.8}',
    '.cyc-demo .cyc-spans{font-family:var(--font-mono,"JetBrains Mono",ui-monospace,Menlo,monospace);line-height:1.9;word-break:break-all}',
    '.cyc-demo .cyc-span{display:inline-block;padding:0 1px;white-space:pre}',
    '.cyc-demo .cyc-span.lit{opacity:.5}',
    '.cyc-demo .cyc-span.match{background:var(--paper-3,#e0d2b6);outline:1px solid var(--role-color);margin:0 2px}',
    '.cyc-demo .cyc-badge{font-size:.7em;vertical-align:super;color:var(--role-color);margin-left:1px}',
    '.cyc-demo .cyc-kv{display:grid;grid-template-columns:auto auto;gap:1px 12px;font-variant-numeric:tabular-nums;align-content:start}',
    '.cyc-demo .cyc-kv span:nth-child(odd){opacity:.75}',
    '.cyc-demo .cyc-kv b{color:var(--role-color);font-weight:normal}',
    '.cyc-demo .cyc-box{border:1px solid var(--role-color);padding:2px 8px;border-radius:2px;white-space:nowrap}',
    '.cyc-demo .cyc-arrow{opacity:.6;align-self:center}'
  ].join('\n');
  var styleInjected = false;
  function injectStyle() {
    if (styleInjected || !document.getElementById) return;
    if (!document.getElementById('cyc-demo-style')) {
      var st = document.createElement('style'); st.id = 'cyc-demo-style'; st.textContent = CSS; document.head.appendChild(st);
    }
    styleInjected = true;
  }

  function fmt(v, d) {
    if (typeof v !== 'number') return String(v);
    if (!isFinite(v)) return v > 0 ? '∞' : v < 0 ? '-∞' : 'NaN';
    if (d === undefined) d = Number.isInteger(v) ? 0 : 2;
    var s = v.toFixed(d);
    if (/^-0(\.0+)?$/.test(s)) s = s.slice(1);
    return s;
  }
  function signed(v, d) { return (v > 0 ? '+' : '') + fmt(v, d); }
  function el(tag, cls, text) { var e = document.createElement(tag); if (cls) e.className = cls; if (text !== undefined) e.textContent = text; return e; }
  function row() { var r = el('div', 'cyc-row'); for (var i = 0; i < arguments.length; i++) if (arguments[i]) r.appendChild(arguments[i]); return r; }
  function col() { var c = el('div', 'cyc-col'); for (var i = 0; i < arguments.length; i++) if (arguments[i]) c.appendChild(arguments[i]); return c; }
  function h(text) { return el('p', 'cyc-h', text); }
  function note(text) { return el('div', 'cyc-note', text); }
  function kv(pairs) {
    var g = el('div', 'cyc-kv');
    pairs.forEach(function (p) { g.appendChild(el('span', null, p[0])); var v = el('span'); var b = el('b', null, typeof p[1] === 'number' ? fmt(p[1], p[2]) : String(p[1])); v.appendChild(b); g.appendChild(v); });
    return g;
  }
  // rows: {label, values, hi?: boolean[], dim?: boolean[], d?: decimals}
  function numStrip(rows) {
    var t = el('table', 'cyc-strip');
    rows.forEach(function (r) {
      var tr = el('tr'); tr.appendChild(el('th', null, r.label));
      r.values.forEach(function (v, i) {
        var td = el('td', null, typeof v === 'number' ? fmt(v, r.d) : String(v));
        if (r.hi && r.hi[i]) td.className = 'hi'; else if (r.dim && r.dim[i]) td.className = 'dim';
        tr.appendChild(td);
      });
      t.appendChild(tr);
    });
    var wrap = el('div', 'cyc-scroll'); wrap.appendChild(t);
    return wrap;
  }
  // values: number[][]; opts {heat, zeroDim, hi: fn(v,r,c) -> bool, d, color}
  function matrix(values, opts) {
    opts = opts || {};
    var g = el('div', 'cyc-matrix'); g.style.gridTemplateColumns = 'repeat(' + values[0].length + ', auto)';
    if (opts.small) { g.style.fontSize = '.8em'; }
    var max = 0; if (opts.heat) values.forEach(function (r) { r.forEach(function (v) { max = Math.max(max, Math.abs(v)); }); });
    values.forEach(function (r, ri) { r.forEach(function (v, ci) {
      var c = el('div', 'cyc-cell', fmt(v, opts.d));
      if (opts.zeroDim && v === 0) c.classList.add('zero');
      if (opts.heat && max > 0) { var a = Math.abs(v) / max; c.style.background = hexA(opts.color, 0.08 + 0.55 * a); }
      if (opts.hi && opts.hi(v, ri, ci)) c.classList.add('hi');
      g.appendChild(c);
    }); });
    var wrap = el('div', 'cyc-scroll'); wrap.appendChild(g);
    return wrap;
  }
  function hexA(hex, a) {
    hex = (hex || '#2a211b').trim();
    if (hex[0] !== '#' || (hex.length !== 7 && hex.length !== 4)) return 'rgba(42,33,27,' + a + ')';
    if (hex.length === 4) hex = '#' + hex[1] + hex[1] + hex[2] + hex[2] + hex[3] + hex[3];
    return 'rgba(' + parseInt(hex.slice(1, 3), 16) + ',' + parseInt(hex.slice(3, 5), 16) + ',' + parseInt(hex.slice(5, 7), 16) + ',' + a + ')';
  }
  function makeCanvas(w, h, T) {
    var dpr = window.devicePixelRatio || 1, c = el('canvas');
    c.width = Math.round(w * dpr); c.height = Math.round(h * dpr); c.style.width = w + 'px'; c.style.height = h + 'px';
    var ctx = c.getContext('2d'); ctx.scale(dpr, dpr);
    ctx.fillStyle = T.bg; ctx.fillRect(0, 0, w, h);
    ctx.font = '12px ' + T.font; ctx.textBaseline = 'middle';
    return { canvas: c, ctx: ctx, w: w, h: h };
  }
  function slider(id, label, min, max, step, value, onChange, fmtFn) {
    var wrap = el('div', 'cyc-ctrl'), lab = el('label', null, label), inp = el('input'), out = el('span', 'cyc-readout');
    inp.type = 'range'; inp.min = min; inp.max = max; inp.step = step; inp.value = value; inp.id = id; lab.htmlFor = id;
    var show = function () { out.textContent = fmtFn ? fmtFn(+inp.value) : fmt(+inp.value); };
    inp.addEventListener('input', function () { show(); onChange(+inp.value); });
    show(); wrap.appendChild(lab); wrap.appendChild(inp); wrap.appendChild(out);
    return wrap;
  }
  function toggle(id, label, options, onChange, initial) {
    var wrap = el('div', 'cyc-ctrl cyc-toggle'); wrap.id = id; if (label) wrap.appendChild(el('label', null, label));
    var btns = options.map(function (o, i) {
      var b = el('button', i === (initial || 0) ? 'on' : '', o); b.type = 'button'; b.id = id + '-' + i;
      b.addEventListener('click', function () { btns.forEach(function (x) { x.classList.remove('on'); }); b.classList.add('on'); onChange(i); });
      wrap.appendChild(b); return b;
    });
    return wrap;
  }
  function spans(tokens) {
    var box = el('div', 'cyc-spans');
    tokens.forEach(function (t) {
      if (t.type === 'lit') box.appendChild(el('span', 'cyc-span lit', t.ch));
      else { var s = el('span', 'cyc-span match', t.text); s.appendChild(el('span', 'cyc-badge', '(' + t.dist + ',' + t.len + ')')); box.appendChild(s); }
    });
    return box;
  }
  // Design tokens (docs/DESIGN.md): teal, sage, rose, plum, gold, bad, copper, teal-2. Refreshed from CSS variables in render().
  var PALETTE_VARS = ['--teal', '--sage', '--rose', '--plum', '--gold', '--bad', '--copper', '--teal-2'];
  var PALETTE = ['#246b66', '#6f8f4e', '#b05a4c', '#6b4a7a', '#b8892e', '#a7403a', '#9c6b3f', '#3f8f88'];
  var NEG_COLOR = '#a7403a';

  // ---- canvas primitives ---------------------------------------------------
  // map: coordinates -1..1. o: {points, assign?, centroids?, arrows?, lines?, grid?: {x:[], y:[]}, labels?}
  function map(cv, o, T) {
    var ctx = cv.ctx, w = cv.w, hh = cv.h, pad = 10;
    var sx = function (x) { return pad + (x + 1) / 2 * (w - 2 * pad); }, sy = function (y) { return hh - pad - (y + 1) / 2 * (hh - 2 * pad); };
    ctx.strokeStyle = T.dim; ctx.lineWidth = 1; ctx.beginPath(); ctx.moveTo(sx(-1), sy(0)); ctx.lineTo(sx(1), sy(0)); ctx.moveTo(sx(0), sy(-1)); ctx.lineTo(sx(0), sy(1)); ctx.stroke();
    if (o.grid) {
      ctx.strokeStyle = hexA(T.color, 0.45); ctx.setLineDash([3, 3]); ctx.beginPath();
      (o.grid.x || []).forEach(function (x) { ctx.moveTo(sx(x), sy(-1)); ctx.lineTo(sx(x), sy(1)); });
      (o.grid.y || []).forEach(function (y) { ctx.moveTo(sx(-1), sy(y)); ctx.lineTo(sx(1), sy(y)); });
      ctx.stroke(); ctx.setLineDash([]);
    }
    (o.lines || []).forEach(function (l) { ctx.strokeStyle = l.color || T.color; ctx.lineWidth = l.width || 1.5; ctx.beginPath(); ctx.moveTo(sx(l.x1), sy(l.y1)); ctx.lineTo(sx(l.x2), sy(l.y2)); ctx.stroke(); });
    (o.points || []).forEach(function (p, i) {
      ctx.fillStyle = o.assign ? PALETTE[o.assign[i] % PALETTE.length] : (o.pointColor || T.fg);
      ctx.beginPath(); ctx.arc(sx(p[0]), sy(p[1]), o.r || 2.5, 0, Math.PI * 2); ctx.fill();
    });
    (o.arrows || []).forEach(function (a) {
      ctx.strokeStyle = a.color || T.color; ctx.lineWidth = 1.2; ctx.beginPath(); ctx.moveTo(sx(a.from[0]), sy(a.from[1])); ctx.lineTo(sx(a.to[0]), sy(a.to[1])); ctx.stroke();
      var ang = Math.atan2(sy(a.to[1]) - sy(a.from[1]), sx(a.to[0]) - sx(a.from[0]));
      ctx.fillStyle = a.color || T.color; ctx.beginPath(); ctx.moveTo(sx(a.to[0]), sy(a.to[1]));
      ctx.lineTo(sx(a.to[0]) - 6 * Math.cos(ang - 0.4), sy(a.to[1]) - 6 * Math.sin(ang - 0.4)); ctx.lineTo(sx(a.to[0]) - 6 * Math.cos(ang + 0.4), sy(a.to[1]) - 6 * Math.sin(ang + 0.4)); ctx.fill();
    });
    (o.centroids || []).forEach(function (c, k) {
      ctx.fillStyle = o.centroidColor || PALETTE[k % PALETTE.length]; ctx.strokeStyle = T.bg; ctx.lineWidth = 1.5;
      ctx.beginPath(); ctx.rect(sx(c[0]) - 5, sy(c[1]) - 5, 10, 10); ctx.fill(); ctx.stroke();
      ctx.fillStyle = T.fg; ctx.fillText(String(k), sx(c[0]) + 7, sy(c[1]) - 6);
    });
    (o.labels || []).forEach(function (l) { ctx.fillStyle = l.color || T.fg; ctx.fillText(l.text, sx(l.x), sy(l.y)); });
  }
  function tree(cv, root, T) {
    var ctx = cv.ctx, leaves = [], depth = 0;
    (function walk(n, d) { if (!n) return; depth = Math.max(depth, d); if (n.sym !== undefined) { n._x = leaves.length; leaves.push(n); return; } walk(n.left, d + 1); walk(n.right, d + 1); n._x = (n.left._x + (n.right ? n.right._x : n.left._x)) / 2; })(root, 0);
    var padX = 24, padY = 14, dx = (cv.w - 2 * padX) / Math.max(1, leaves.length - 1), dy = (cv.h - 2 * padY) / Math.max(1, depth);
    var X = function (n) { return padX + n._x * dx; };
    (function draw(n, d) {
      if (!n || n.sym !== undefined) return;
      [n.left, n.right].forEach(function (ch, bit) {
        if (!ch) return;
        ctx.strokeStyle = T.dim; ctx.lineWidth = 1.5; ctx.beginPath(); ctx.moveTo(X(n), padY + d * dy); ctx.lineTo(X(ch), padY + (d + 1) * dy); ctx.stroke();
        ctx.fillStyle = T.color; ctx.textAlign = 'center'; ctx.fillText(String(bit), (X(n) + X(ch)) / 2 + (bit ? 7 : -7), padY + (d + 0.5) * dy);
        draw(ch, d + 1);
      });
    })(root, 0);
    (function nodes(n, d) {
      if (!n) return; var x = X(n), y = padY + d * dy;
      if (n.sym !== undefined) { ctx.fillStyle = T.color; ctx.fillRect(x - 9, y - 8, 18, 16); ctx.fillStyle = T.bg; ctx.textAlign = 'center'; ctx.fillText(n.sym, x, y + 1); }
      else { ctx.fillStyle = T.fg; ctx.beginPath(); ctx.arc(x, y, 3, 0, Math.PI * 2); ctx.fill(); nodes(n.left, d + 1); nodes(n.right, d + 1); }
    })(root, 0);
    ctx.textAlign = 'left';
  }
  // steps: [{lo,hi,sym}] — each row shows the interval of step i zoomed to full width, with the sub-interval chosen next highlighted.
  function intervals(cv, steps, model, T) {
    var ctx = cv.ctx, padL = 66, padR = 56, rowH = cv.h / steps.length, W = cv.w - padL - padR;
    steps.forEach(function (s, i) {
      var y = rowH * i + rowH / 2, width = s.hi - s.lo, X = function (v) { return padL + (v - s.lo) / width * W; };
      ctx.strokeStyle = T.dim; ctx.lineWidth = 2; ctx.beginPath(); ctx.moveTo(padL, y); ctx.lineTo(padL + W, y); ctx.stroke();
      if (i + 1 < steps.length) {
        var cum = 0;
        model.forEach(function (m) {
          var a = s.lo + cum * width, b = s.lo + (cum + m.p) * width, chosen = m.sym === steps[i + 1].sym;
          ctx.fillStyle = chosen ? T.color : hexA(T.fg, 0.08); ctx.fillRect(X(a), y - 6, X(b) - X(a), 12);
          ctx.fillStyle = chosen ? T.bg : T.fg; ctx.textAlign = 'center'; ctx.fillText(m.sym, (X(a) + X(b)) / 2, y + 1);
          if (chosen && i + 2 < steps.length + 1) { ctx.strokeStyle = hexA(T.color, 0.5); ctx.setLineDash([2, 3]); ctx.beginPath(); ctx.moveTo(X(a), y + 6); ctx.lineTo(padL, y + rowH - 6); ctx.moveTo(X(b), y + 6); ctx.lineTo(padL + W, y + rowH - 6); ctx.stroke(); ctx.setLineDash([]); }
          cum += m.p;
        });
      } else { ctx.fillStyle = T.color; ctx.fillRect(padL, y - 6, W, 12); }
      ctx.fillStyle = T.fg; ctx.textAlign = 'right'; ctx.fillText(fmt(s.lo, 6).replace(/\.?0+$/, ''), padL - 4, y); ctx.textAlign = 'left'; ctx.fillText(fmt(s.hi, 6).replace(/\.?0+$/, ''), padL + W + 4, y);
    });
    ctx.textAlign = 'left';
  }
  // curve(cv, fns: [{fn, color}], domain [x0,x1,y0,y1], marks [{x,y,label,color}])
  function curve(cv, fns, dom, marks, T) {
    var ctx = cv.ctx, pad = 24, W = cv.w - 2 * pad, H = cv.h - 2 * pad;
    var sx = function (x) { return pad + (x - dom[0]) / (dom[1] - dom[0]) * W; }, sy = function (y) { return pad + H - (y - dom[2]) / (dom[3] - dom[2]) * H; };
    ctx.strokeStyle = T.dim; ctx.lineWidth = 1; ctx.beginPath();
    if (dom[2] <= 0 && dom[3] >= 0) { ctx.moveTo(sx(dom[0]), sy(0)); ctx.lineTo(sx(dom[1]), sy(0)); }
    if (dom[0] <= 0 && dom[1] >= 0) { ctx.moveTo(sx(0), sy(dom[2])); ctx.lineTo(sx(0), sy(dom[3])); }
    ctx.stroke();
    fns.forEach(function (f) {
      ctx.strokeStyle = f.color || T.color; ctx.lineWidth = f.width || 1.5; if (f.dash) ctx.setLineDash(f.dash); ctx.beginPath();
      for (var i = 0; i <= 200; i++) { var x = dom[0] + (dom[1] - dom[0]) * i / 200, y = f.fn(x); if (!isFinite(y)) continue; var px = sx(x), py = sy(Math.max(dom[2], Math.min(dom[3], y))); if (i === 0) ctx.moveTo(px, py); else ctx.lineTo(px, py); }
      ctx.stroke(); ctx.setLineDash([]);
    });
    (marks || []).forEach(function (m) {
      ctx.fillStyle = m.color || T.fg; ctx.beginPath(); ctx.arc(sx(m.x), sy(m.y), 3.5, 0, Math.PI * 2); ctx.fill();
      if (m.label) ctx.fillText(m.label, sx(m.x) + 6, sy(m.y) - 8);
    });
    ctx.fillStyle = T.fg; ctx.globalAlpha = 0.6; ctx.fillText(fmt(dom[0]), sx(dom[0]), cv.h - 8); ctx.textAlign = 'right'; ctx.fillText(fmt(dom[1]), sx(dom[1]), cv.h - 8); ctx.textAlign = 'left'; ctx.fillText(fmt(dom[3]), 2, pad); ctx.fillText(fmt(dom[2]), 2, pad + H); ctx.globalAlpha = 1;
    return { sx: sx, sy: sy };
  }
  // bars(cv, values, labels, opts {max, min, colors[], overlay: fn(ctx, sx, sy), hiline})
  function bars(cv, values, labels, opts, T) {
    opts = opts || {}; var ctx = cv.ctx, pad = 18, padB = labels ? 18 : 8, W = cv.w - 2 * pad, H = cv.h - pad - padB, n = values.length;
    var max = opts.max !== undefined ? opts.max : Math.max.apply(null, values), min = opts.min !== undefined ? opts.min : Math.min(0, Math.min.apply(null, values));
    var bw = W / n, sx = function (i) { return pad + i * bw; }, sy = function (v) { return pad + H - (v - min) / (max - min) * H; };
    values.forEach(function (v, i) {
      ctx.fillStyle = (opts.colors && opts.colors[i]) || T.color; var y0 = sy(Math.max(min, 0)), y1 = sy(v);
      ctx.fillRect(sx(i) + bw * 0.12, Math.min(y0, y1), bw * 0.76, Math.max(1, Math.abs(y0 - y1)));
      if (opts.values) { ctx.fillStyle = T.fg; ctx.textAlign = 'center'; ctx.fillText(opts.values[i], sx(i) + bw / 2, Math.min(y0, y1) - 8); }
      if (labels && labels[i] !== undefined) { ctx.fillStyle = T.fg; ctx.globalAlpha = 0.7; ctx.textAlign = 'center'; ctx.fillText(String(labels[i]), sx(i) + bw / 2, cv.h - 8); ctx.globalAlpha = 1; }
    });
    ctx.textAlign = 'left';
    if (opts.overlay) opts.overlay(ctx, function (i) { return sx(i) + bw / 2; }, sy);
    return { sx: sx, sy: sy, bw: bw };
  }
  function wave(cv, series, T, opts) { // series: [{data, color, dash}], data in roughly -1..1
    opts = opts || {}; var ctx = cv.ctx, pad = 6, W = cv.w - 2 * pad, H = cv.h - 2 * pad, amp = opts.amp || 1;
    ctx.strokeStyle = T.dim; ctx.beginPath(); ctx.moveTo(pad, cv.h / 2); ctx.lineTo(cv.w - pad, cv.h / 2); ctx.stroke();
    series.forEach(function (s) {
      var n = s.data.length; ctx.strokeStyle = s.color || T.color; ctx.lineWidth = s.width || 1.5; if (s.dash) ctx.setLineDash(s.dash); ctx.beginPath();
      s.data.forEach(function (v, i) { var x = pad + i / (n - 1) * W, y = cv.h / 2 - v / amp * H / 2; if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y); });
      ctx.stroke(); ctx.setLineDash([]);
      if (s.dots) s.data.forEach(function (v, i) { ctx.fillStyle = s.color || T.color; ctx.beginPath(); ctx.arc(pad + i / (n - 1) * W, cv.h / 2 - v / amp * H / 2, 2, 0, Math.PI * 2); ctx.fill(); });
    });
  }
  function grid(cv, m, T, opts) { // small pixel grid with values; opts {max, signed, label}
    opts = opts || {}; var ctx = cv.ctx, n = m.length, cell = Math.min(cv.w, cv.h - (opts.label ? 14 : 0)) / n, top = opts.label ? 14 : 0;
    var max = opts.max || Math.max(1, Math.max.apply(null, m.map(function (r) { return Math.max.apply(null, r.map(Math.abs)); })));
    m.forEach(function (r, ri) { r.forEach(function (v, ci) {
      var a = Math.abs(v) / max; ctx.fillStyle = v === 0 ? hexA(T.fg, 0.05) : (v < 0 ? hexA(NEG_COLOR, 0.2 + 0.7 * a) : hexA(T.color, 0.2 + 0.7 * a));
      ctx.fillRect(ci * cell + 1, top + ri * cell + 1, cell - 2, cell - 2);
    }); });
    if (opts.label) { ctx.fillStyle = T.fg; ctx.fillText(opts.label, 0, 6); }
  }

  // ===========================================================================
  // Labs — each: function (box, T, f) where box is the lab body, T the theme
  // ===========================================================================
  var LABS = {};

  // ---- transform -----------------------------------------------------------
  LABS.dct = function (box, T) {
    var rowIn = [52, 55, 61, 66, 70, 61, 64, 73], view = el('div');
    function draw(k) {
      var r = calc.dctKeep(rowIn, k); view.textContent = '';
      view.appendChild(numStrip([
        { label: 'pixels x', values: rowIn },
        { label: 'DCT X[k]', values: r.coeffs, d: 1, hi: r.coeffs.map(function (_, i) { return i < k; }) },
        { label: 'kept', values: r.kept, d: 1, dim: r.kept.map(function (v) { return v === 0; }) },
        { label: 'rebuilt x̂', values: r.rec, d: 1 },
        { label: 'error', values: r.rec.map(function (v, i) { return v - rowIn[i]; }), d: 1 }]));
      view.appendChild(kv([['coefficients kept', k + ' of 8'], ['MSE', r.mse, 2], ['RMS error (grey levels)', Math.sqrt(r.mse), 2]]));
    }
    box.appendChild(slider('demo-dct-k', 'keep first k coefficients', 1, 8, 1, 3, draw));
    box.appendChild(view); draw(3);
    box.appendChild(note('X[k] = c(k)·Σ x[n]·cos(π(2n+1)k/16), orthonormal DCT-II. Energy compaction: k=3 of 8 already rebuilds the row to within a few grey levels; the high-frequency coefficients are tiny.'));
  };

  LABS.mdct = function (box, T) {
    var sizes = [64, 128, 256, 512, 1024, 1152], cv = makeCanvas(T.w, 110, T), info = el('div');
    function drawWindows() {
      var ctx = cv.ctx, N = 64, w = calc.mdctWindow(N), pb = calc.princenBradley(N), pad = 6, W = cv.w - 2 * pad, H = cv.h - 2 * pad;
      var X = function (n) { return pad + n / (3 * N) * W; }, Y = function (v) { return pad + H - v * H; };
      ctx.strokeStyle = T.dim; ctx.beginPath(); ctx.moveTo(pad, Y(0)); ctx.lineTo(cv.w - pad, Y(0)); ctx.stroke();
      [0, N].forEach(function (off, i) { ctx.strokeStyle = i ? PALETTE[0] : T.color; ctx.lineWidth = 1.5; ctx.beginPath(); w.forEach(function (v, n) { if (n === 0) ctx.moveTo(X(n + off), Y(v)); else ctx.lineTo(X(n + off), Y(v)); }); ctx.stroke(); });
      ctx.strokeStyle = PALETTE[4]; ctx.setLineDash([4, 3]); ctx.beginPath(); pb.forEach(function (v, n) { if (n === 0) ctx.moveTo(X(n + N), Y(v)); else ctx.lineTo(X(n + N), Y(v)); }); ctx.stroke(); ctx.setLineDash([]);
      ctx.fillStyle = T.fg; ctx.fillText('frame t', X(N / 4) - 20, Y(0.25)); ctx.fillText('frame t+1', X(2.75 * N) - 30, Y(0.25)); ctx.fillStyle = PALETTE[4]; ctx.fillText('w²[n] + w²[n+N] = ' + fmt(pb[Math.floor(N / 3)], 4), X(1.5 * N) - 60, Y(0.88));
    }
    drawWindows();
    function show(i) {
      var N = sizes[i], pb = calc.princenBradley(N), maxDev = Math.max.apply(null, pb.map(function (v) { return Math.abs(v - 1); }));
      info.textContent = ''; info.appendChild(kv([['frame size N', N + ' samples'], ['window length', 2 * N + ' samples (50% overlap)'], ['algorithmic delay at 48 kHz', fmt(calc.mdctDelayMs(N, 48000), 2) + ' ms'], ['max |w²+w²−1| over the overlap', maxDev.toExponential(1)]]));
    }
    box.appendChild(cv.canvas);
    box.appendChild(slider('demo-mdct-n', 'frame size N', 0, sizes.length - 1, 1, 5, show, function (i) { return String(sizes[i]); }));
    box.appendChild(info); show(5);
    box.appendChild(note('w[n] = sin(π(n+½)/2N). Princen–Bradley: the squared windows sum to 1 in the overlap, so overlap-add cancels the aliasing exactly. Delay = N/48000 s: the decoder cannot finish frame t until frame t+1 arrives.'));
  };

  LABS.dwt = function (box, T) {
    var input = [10, 12, 14, 16, 100, 98, 96, 94], view = el('div');
    function draw(levels) {
      var pyr = calc.haarPyramid(input, levels), rows = [{ label: 'input', values: input }];
      pyr.forEach(function (lv, i) {
        rows.push({ label: 'L' + (i + 1) + ' avg (a+b)/2', values: lv.avg });
        rows.push({ label: 'L' + (i + 1) + ' det (a−b)/2', values: lv.det, hi: lv.det.map(function (v) { return Math.abs(v) > 5; }) });
      });
      view.textContent = ''; view.appendChild(numStrip(rows));
      var kept = pyr.length ? pyr[pyr.length - 1].avg : input, details = [];
      pyr.forEach(function (lv) { details = details.concat(lv.det); });
      view.appendChild(kv([['coefficients', input.length + ' → ' + kept.length + ' averages + ' + details.length + ' details'], ['details with |d| ≤ 2', details.filter(function (v) { return Math.abs(v) <= 2; }).length + ' of ' + details.length], ['largest detail', Math.max.apply(null, details.map(Math.abs))]]));
    }
    box.appendChild(toggle('demo-dwt-levels', 'levels', ['1', '2', '3'], function (i) { draw(i + 1); }, 1));
    box.appendChild(view); draw(2);
    box.appendChild(note('One Haar level: pairwise average and half-difference. Details are tiny on the smooth parts and large only at the edge (the 16→100 jump). Recurse on the averages for a multiresolution pyramid; the averages are a thumbnail.'));
  };

  LABS.prism = function (box, T) {
    var px = [[200, 180, 170], [205, 182, 172], [198, 179, 168], [202, 181, 171]], ycc = px.map(function (p) { return calc.rgb2ycc(p[0], p[1], p[2]); });
    function spread(a) { return Math.max.apply(null, a) - Math.min.apply(null, a); }
    var Y = ycc.map(function (c) { return c.y; }), Cb = ycc.map(function (c) { return c.cb; }), Cr = ycc.map(function (c) { return c.cr; });
    box.appendChild(numStrip([
      { label: 'R', values: px.map(function (p) { return p[0]; }).concat(['spread ' + spread(px.map(function (p) { return p[0]; }))]) },
      { label: 'G', values: px.map(function (p) { return p[1]; }).concat(['spread ' + spread(px.map(function (p) { return p[1]; }))]) },
      { label: 'B', values: px.map(function (p) { return p[2]; }).concat(['spread ' + spread(px.map(function (p) { return p[2]; }))]) },
      { label: 'Y', values: Y.concat(['spread ' + fmt(spread(Y), 1)]), d: 1, hi: [1, 1, 1, 1, 1] },
      { label: 'Cb', values: Cb.concat(['spread ' + fmt(spread(Cb), 1)]), d: 1 },
      { label: 'Cr', values: Cr.concat(['spread ' + fmt(spread(Cr), 1)]), d: 1 }]));
    var ms = calc.midSide(1000, 980);
    box.appendChild(numStrip([{ label: 'stereo L, R', values: [1000, 980] }, { label: 'M=(L+R)/2, S=L−R', values: [ms.m, ms.s], hi: [true, true] }]));
    box.appendChild(note('Y = 0.299R + 0.587G + 0.114B, Cb = 128 + 0.564(B−Y), Cr = 128 + 0.713(R−Y). Brightness varies across the four pixels; both chroma channels sit near 128 with a spread of about one level. The correlated part moved into one channel.'));
  };

  LABS.pca = function (box, T) {
    var pts = calc.pcaPoints(40, 11), p = calc.pca(pts), view = el('div');
    function draw(rotated) {
      view.textContent = '';
      var P = rotated ? calc.rotate(pts, p.angle) : pts, gq = calc.gridQuant(P, 2), gq0 = calc.gridQuant(pts, 2), gqR = calc.gridQuant(calc.rotate(pts, p.angle), 2);
      var cv = makeCanvas(170, 170, T), lines = rotated ? [] : [{ x1: -p.axis[0], y1: -p.axis[1], x2: p.axis[0], y2: p.axis[1], color: T.color, width: 2 }];
      map(cv, { points: P, lines: lines, grid: { x: gq.axes[0].edges, y: gq.axes[1].edges } }, T);
      var stats = rotated
        ? kv([['axes', 'principal (rotated by ' + fmt(p.angle * 180 / Math.PI, 1) + '°)'], ['variance axis 1 (λ1)', p.lambda[0], 4], ['variance axis 2 (λ2)', p.lambda[1], 4], ['2-bit grid error Σ|x−x̂|²', gq.err, 3], ['vs un-rotated', fmt(gq0.err, 3) + ' (' + fmt((1 - gqR.err / gq0.err) * 100, 0) + '% less)']])
        : kv([['axes', 'original x, y'], ['var x', p.varX, 4], ['var y', p.varY, 4], ['2-bit grid error Σ|x−x̂|²', gq.err, 3], ['after rotation it would be', gqR.err, 3]]);
      view.appendChild(row(cv.canvas, stats));
    }
    box.appendChild(toggle('demo-pca-rotate', '', ['original axes', 'rotate to principal axes'], function (i) { draw(i === 1); }));
    box.appendChild(view); draw(false);
    box.appendChild(note('40 points of y = 0.8x + N(0, 0.1²). The 2-bit grid gives 4 levels per axis, spanning that axis\'s range. Rotation moves nothing, but afterwards one axis is nearly flat, so its 4 levels are spent on a tiny range and the quantization error drops.'));
  };

  LABS.bwt = function (box, T) {
    var view = el('div');
    function draw(word) {
      var b = calc.bwt(word); view.textContent = '';
      var left = el('table', 'cyc-strip'), right = el('table', 'cyc-strip');
      b.rotations.forEach(function (r) { var tr = el('tr'); tr.appendChild(el('td', null, r)); left.appendChild(tr); });
      b.sorted.forEach(function (r, i) { var tr = el('tr'); var td = el('td', null, r.slice(0, -1)); td.style.textAlign = 'left'; tr.appendChild(td); var last = el('td', 'hi', r[r.length - 1]); tr.appendChild(last); if (i === b.primary) tr.appendChild(el('td', null, '← row ' + i + ' is the original')); right.appendChild(tr); });
      left.querySelectorAll('td').forEach(function (td) { td.style.textAlign = 'left'; td.style.letterSpacing = '.15em'; });
      right.querySelectorAll('td').forEach(function (td) { td.style.letterSpacing = '.15em'; });
      view.appendChild(row(col(h('rotations'), left), el('div', 'cyc-arrow', 'sort →'), col(h('sorted, last column'), right)));
      view.appendChild(kv([['output', b.last + '  (primary index ' + b.primary + ')'], ['runs of equal letters', calc.runs(word) + ' in the input → ' + calc.runs(b.last) + ' in the output']]));
    }
    box.appendChild(toggle('demo-bwt-word', 'word', ['banana$', 'abracadabra$'], function (i) { draw(i ? 'abracadabra$' : 'banana$'); }));
    box.appendChild(view); draw('banana$');
    box.appendChild(note('Sort all rotations, keep the last column plus the row index of the original. Same letters cluster because they share the context that follows them; MTF + RLE then eats the runs. Fully reversible.'));
  };

  LABS.neural_enc = function (box, T) {
    var w = calc.neuralEncWave(320, 3), cv = makeCanvas(150, 60, T); wave(cv, [{ data: w, width: 1 }], T, { amp: 1.1 });
    var latent = [0.81, -1.20, 0.07, 2.31, -0.45, 0.92, -1.77, 0.33];
    var lat = el('div', 'cyc-box', latent.map(function (v) { return fmt(v, 2); }).join(' ')); lat.style.fontVariantNumeric = 'tabular-nums';
    box.appendChild(row(col(h('320 samples × 16 bit'), cv.canvas), el('div', 'cyc-arrow', '→'), col(h('encoder'), el('div', 'cyc-box', 'conv / transformer')), el('div', 'cyc-arrow', '→'), col(h('one 8-float latent'), lat), el('div', 'cyc-arrow', '→'), col(h('RVQ'), el('div', 'cyc-box', '80 bits'))));
    var bitsIn = 320 * 16, bitsOut = 80;
    box.appendChild(kv([['bits in', bitsIn + ' (320 × 16)'], ['bits out', bitsOut], ['ratio before any entropy coding', fmt(bitsIn / bitsOut, 0) + ':1']]));
    box.appendChild(note('Stand-in numbers: the real EnCodec latent is 512-dim at 75 Hz. The analysis transform is learned jointly with the quantizer and decoder, so the latent is shaped for the quantizer that follows.'));
  };

  // ---- predict -------------------------------------------------------------
  LABS.lz77 = function (box, T) {
    var text = 'the_cat_sat_on_the_mat_the_cat_sat', view = el('div');
    function draw(win) {
      var r = calc.lz77(text, win, 3); view.textContent = ''; view.appendChild(spans(r.tokens));
      var longest = r.tokens.filter(function (t) { return t.type === 'match'; }).sort(function (a, b) { return b.len - a.len; })[0];
      view.appendChild(kv([['window', win + ' bytes'], ['matches', r.matches], ['bytes covered by matches', r.covered + ' of ' + text.length], ['toy cost (literal 1 B, match 2 B)', r.cost + ' B → saves ' + r.saved + ' B'], ['longest match', longest ? '(' + longest.dist + ',' + longest.len + ') = "' + longest.text + '"' : 'none']]));
    }
    box.appendChild(slider('demo-lz77-window', 'window size', 4, 32, 1, 32, draw));
    box.appendChild(view); draw(32);
    box.appendChild(note('Greedy LZ77, minimum match 3. A (distance,length) pair says "copy from distance back": a prediction with certainty 1. Shrink the window and the long match to the first "the_cat_sat" falls out of reach.'));
  };

  LABS.lzw = function (box, T) {
    var r = calc.lzw('ABABABA'), t = el('table', 'cyc-strip'), head = el('tr');
    ['w', 'next', 'output', 'dictionary add'].forEach(function (x) { head.appendChild(el('th', null, x)); }); t.appendChild(head);
    r.steps.forEach(function (s) { var tr = el('tr'); [s.w || '∅', s.c, s.out === null ? '' : s.out, s.add || ''].forEach(function (v, i) { var td = el('td', i === 2 && v !== '' ? 'hi' : null, String(v)); td.style.textAlign = 'left'; tr.appendChild(td); }); t.appendChild(tr); });
    box.appendChild(row(t, kv([['input', 'ABABABA (7 symbols)'], ['output codes', r.codes.join(' ')], ['codes sent', r.codes.length], ['dictionary gained', r.added.map(function (a) { return a.code + '=' + a.phrase; }).join(' ')]])));
    box.appendChild(note('w grows while w+c is in the table; when it is not, emit code(w), add w+c as a new entry, restart from c. The decoder rebuilds the identical table from the codes alone, so nothing extra travels.'));
  };

  LABS.delta = function (box, T) {
    var x = [100, 102, 105, 107, 108, 108, 107, 105], view = el('div');
    function draw(order) {
      var r = calc.delta(x, order), tail = r.slice(1); view.textContent = '';
      view.appendChild(numStrip([{ label: 'samples x', values: x }, { label: 'residuals', values: r.map(function (v, i) { return i === 0 ? v : signed(v); }), hi: r.map(function (_, i) { return i > 0; }) }]));
      view.appendChild(kv([['raw range', Math.min.apply(null, x) + '..' + Math.max.apply(null, x) + ' → ' + calc.bitsUnsigned(x) + ' bits each'], ['residual range (after the first)', Math.min.apply(null, tail) + '..' + Math.max.apply(null, tail) + ' → ' + calc.bitsSigned(tail) + ' bits with sign'], ['Σ|residual| after the first', sum(tail.map(Math.abs))]]));
    }
    box.appendChild(toggle('demo-delta-order', 'order', ['1: x[n] − x[n−1]', '2: delta of delta'], function (i) { draw(i + 1); }));
    box.appendChild(view); draw(1);
    box.appendChild(note('The first sample is sent raw; every other one is replaced by its difference from the prediction. Order 2 predicts "same slope" and squeezes the smooth run to ±1. Zero parameters, zero side information.'));
  };

  LABS.lpc = function (box, T) {
    var x = calc.lpcSignal(16), fit = calc.lpcFit2(x), rFixed = calc.lpcResidual(x, 2, -1), rFit = calc.lpcResidual(x, fit.a1, fit.a2), raw = x.slice(2);
    var cv = makeCanvas(T.w, 70, T); wave(cv, [{ data: x, color: T.fg, width: 1, dots: true }, { data: [0, 0].concat(rFixed), color: PALETTE[2], dash: [3, 3] }, { data: [0, 0].concat(rFit), color: T.color }], T, { amp: 100 });
    box.appendChild(cv.canvas);
    box.appendChild(numStrip([{ label: 'x[n]', values: x }, { label: 'fixed 2x[n−1]−x[n−2]', values: ['', ''].concat(rFixed), d: 0 }, { label: 'fitted a1,a2', values: ['', ''].concat(rFit), d: 1, hi: [0, 0].concat(rFit.map(function () { return true; })) }]));
    box.appendChild(kv([['fitted coefficients', 'a1 = ' + fmt(fit.a1, 3) + ', a2 = ' + fmt(fit.a2, 3)], ['Σr² raw (n ≥ 2)', calc.energy(raw), 0], ['Σr² fixed predictor', calc.energy(rFixed), 0], ['Σr² fitted predictor', calc.energy(rFit), 0]]));
    box.appendChild(note('x[n] = round(100·0.9ⁿ·cos 0.6n), a damped resonance. The fixed order-2 predictor assumes a straight line and fails on the curve; two least-squares coefficients follow it. a1, a2 are side info sent once per block.'));
  };

  LABS.spatial = function (box, T) {
    var block = [[10, 11, 12, 13], [20, 21, 22, 23], [30, 31, 32, 33], [40, 41, 42, 43]], r = calc.pngFilters(block);
    var names = { sub: 'Sub (left)', up: 'Up (above)', paeth: 'Paeth' }, items = [col(h('block'), matrix(block))];
    Object.keys(names).forEach(function (k) { var c = col(h(names[k] + ' Σ|r| = ' + r.sums[k]), matrix(r.residuals[k], { zeroDim: true })); if (k === r.winner) c.firstChild.style.color = T.color; items.push(c); });
    box.appendChild(row.apply(null, items));
    box.appendChild(kv([['winner', names[r.winner] + ' with Σ|residual| = ' + r.sums[r.winner]]]));
    box.appendChild(note('Residual = pixel − prediction from already-decoded neighbours (out-of-image neighbours count as 0). Sub kills the horizontal gradient, Up the vertical, Paeth picks per pixel whichever of left/up/upper-left is closest to left+up−upper-left. PNG picks per row; H.264 picks a direction per block.'));
  };

  LABS.motion = function (box, T) {
    var f1 = calc.frameWithSquare(8, 2, 2, 3, 9), f2 = calc.frameWithSquare(8, 3, 2, 3, 9), mv = calc.motionSearch(f1, f2, 2), view = el('div');
    function draw(mc) {
      view.textContent = '';
      var res = mc ? mv.residual : calc.frameDiff(f2, f1), nz = calc.countNonzero(res);
      var c1 = makeCanvas(90, 104, T), c2 = makeCanvas(90, 104, T), c3 = makeCanvas(90, 104, T);
      grid(c1, f1, T, { max: 9, label: 'frame 1' }); grid(c2, f2, T, { max: 9, label: 'frame 2' }); grid(c3, res, T, { max: 9, label: 'residual' });
      view.appendChild(row(c1.canvas, c2.canvas, c3.canvas, kv(mc
        ? [['search range', '±2 (25 candidates)'], ['motion vector', '(' + signed(mv.dx) + ', ' + signed(mv.dy) + ')'], ['SAD at best vector', mv.sad], ['nonzero residual cells', nz + ' of 64']]
        : [['prediction', 'frame 1 as is'], ['motion vector', 'none'], ['nonzero residual cells', nz + ' of 64'], ['Σ|residual|', sum(res.map(function (r) { return sum(r.map(Math.abs)); }))]])));
    }
    box.appendChild(toggle('demo-motion-mc', '', ['motion compensation', 'no motion compensation'], function (i) { draw(i === 0); }));
    box.appendChild(view); draw(true);
    box.appendChild(note('A 3×3 square at (2,2) moves to (3,2). The encoder tries every shift in ±2 and keeps the one with the smallest SAD: (+1, 0). The vector costs a few bits; the residual costs nothing. Without it, both the old and the new position have to be coded.'));
  };

  LABS.cm = function (box, T) {
    var text = 'the_theme_of_the_thesis', A = 27, contexts = ['th', 'he', 'e_'], view = el('div');
    var tables = row.apply(null, contexts.map(function (c) {
      var cc = calc.contextCounts(text, c), t = el('table', 'cyc-strip'), syms = Object.keys(cc.counts).sort(function (a, b) { return cc.counts[b] - cc.counts[a]; });
      var head = el('tr'); ['after "' + c + '"', 'n', 'P', 'bits'].forEach(function (x) { head.appendChild(el('th', null, x)); }); t.appendChild(head);
      syms.forEach(function (s) { var p = calc.laplace(cc.counts[s], cc.total, A), tr = el('tr'); [s === '_' ? '␣' : s, cc.counts[s], fmt(p, 3), fmt(calc.cost(p), 2)].forEach(function (v, i) { tr.appendChild(el('td', i === 0 ? 'hi' : null, String(v))); }); t.appendChild(tr); });
      var p0 = calc.laplace(0, cc.total, A), tr0 = el('tr'); ['other', 0, fmt(p0, 3), fmt(calc.cost(p0), 2)].forEach(function (v) { tr0.appendChild(el('td', 'dim', String(v))); }); t.appendChild(tr0);
      return t;
    }));
    box.appendChild(tables);
    var th = calc.contextCounts(text, 'th');
    function draw(extra) {
      var n = th.counts.e + extra, p = calc.laplace(n, th.total + extra, A); view.textContent = '';
      view.appendChild(kv([['"the" seen', n + ' times'], ['P(e | th)', fmt(p, 3) + ' = (' + n + '+1)/(' + (th.total + extra) + '+' + A + ')'], ['cost of the next e', fmt(calc.cost(p), 2) + ' bits']]));
    }
    box.appendChild(slider('demo-cm-extra', 'more "the" sightings', 0, 300, 10, 0, draw, function (v) { return '+' + v; }));
    box.appendChild(view); draw(0);
    box.appendChild(note('Text: ' + text + '. Order-2 counts with Laplace smoothing over ' + A + ' symbols: P = (n+1)/(N+' + A + '). The coder pays −log₂P bits; the model\'s job is to make P large, and it only gets confident with evidence.'));
  };

  LABS.neural_prior = function (box, T) {
    var dist = [['Paris', 0.90], ['Lyon', 0.03], ['London', 0.02], ['other', 0.05]], cv = makeCanvas(200, 120, T);
    bars(cv, dist.map(function (d) { return d[1]; }), dist.map(function (d) { return d[0]; }), { max: 1, values: dist.map(function (d) { return fmt(d[1], 2); }) }, T);
    var t = numStrip([{ label: 'token', values: dist.map(function (d) { return d[0]; }) }, { label: 'p', values: dist.map(function (d) { return d[1]; }), d: 2 }, { label: '−log₂p bits', values: dist.map(function (d) { return calc.cost(d[1]); }), d: 2, hi: [true, false, true, false] }, { label: 'fixed code', values: dist.map(function () { return 8; }) }]);
    box.appendChild(row(col(h('P(next | "The capital of France is")'), cv.canvas), t));
    box.appendChild(note('Any model of P(next symbol) is a compressor once an arithmetic coder charges −log₂p: the expected token costs ' + fmt(calc.cost(0.9), 2) + ' bits, a surprise ' + fmt(calc.cost(0.02), 1) + '. A 7B LLM gets English to about 0.8 bits/char this way; the catch is running the model identically at both ends.'));
  };

  // ---- quantize ------------------------------------------------------------
  LABS.uniform = function (box, T) {
    var xs = [3.7, -1.2, 0.4, 8.9, -5.5, 2.1], view = el('div');
    function draw(d) {
      var r = calc.uniformQ(xs, d, 9); view.textContent = '';
      view.appendChild(numStrip([{ label: 'x', values: xs, d: 2 }, { label: 'q = round(x/Δ)', values: r.q, hi: r.q.map(function () { return true; }) }, { label: 'x̂ = q·Δ', values: r.rec, d: 2 }, { label: 'error', values: r.err, d: 2 }]));
      view.appendChild(kv([['levels in ±9', r.levels + ' → ' + r.bits + ' bits per value'], ['MSE measured', r.mse, 3], ['Δ²/12 (high-resolution theory)', r.theory, 3]]));
    }
    box.appendChild(slider('demo-uniform-delta', 'step Δ', 0.25, 4, 0.25, 1, draw));
    box.appendChild(view); draw(1);
    box.appendChild(note('Doubling Δ halves the number of levels (−1 bit per value) and quadruples the error. The Δ²/12 rule assumes the values are spread evenly inside each bin, which they are not for six samples.'));
  };

  LABS.qmatrix = function (box, T) {
    var view = el('div');
    function draw(scale) {
      var tab = calc.JPEG_Q50.map(function (r) { return r.map(function (v) { return Math.round(v * scale); }); }), r = calc.quantizeBlock(calc.WIKI_DCT, tab, 1); view.textContent = '';
      view.appendChild(row(col(h('Q table × ' + fmt(scale, 2)), matrix(tab, { heat: true, color: T.color })), col(h('round(DCT / Q)'), matrix(r.q, { zeroDim: true })), kv([['zeros', r.zeros + ' of 64'], ['nonzero', 64 - r.zeros], ['DC step', tab[0][0]], ['highest-frequency step', tab[7][7]]])));
    }
    box.appendChild(slider('demo-qmatrix-scale', 'table scale', 0.5, 3, 0.25, 1, draw, function (v) { return fmt(v, 2) + '×'; }));
    box.appendChild(view); draw(1);
    box.appendChild(note('JPEG Annex K luma table at quality 50 applied to the Wikipedia example block\'s DCT (DC = −415.38). Coarse steps where the eye is blind (bottom-right), so most high-frequency coefficients round to exactly zero for the run-length coder.'));
  };

  LABS.companding = function (box, T) {
    var cv = makeCanvas(170, 150, T), view = el('div');
    function draw(a) {
      var m = calc.mulawQuant(a), u = calc.uniformQuant8(a); view.textContent = '';
      cv.ctx.fillStyle = T.bg; cv.ctx.fillRect(0, 0, cv.w, cv.h);
      curve(cv, [{ fn: calc.mulaw }, { fn: function (x) { return x; }, color: T.dim, dash: [3, 3] }], [-1, 1, -1, 1], [{ x: a, y: calc.mulaw(a), label: 'x=' + fmt(a, 3), color: T.fg }], T);
      view.appendChild(numStrip([
        { label: '', values: ['8-bit μ-law', '8-bit uniform'] },
        { label: 'code', values: [m.code, u.code] },
        { label: 'x̂', values: [fmt(m.rec, 5), fmt(u.rec, 5)] },
        { label: 'relative error', values: [fmt(m.relErr * 100, 2) + '%', fmt(u.relErr * 100, 1) + '%'], hi: [true, false] },
        { label: 'worst case at this x', values: [fmt(m.worst * 100, 1) + '%', fmt(u.worst * 100, 1) + '%'] }]));
    }
    box.appendChild(row(cv.canvas, view));
    box.appendChild(slider('demo-companding-amp', 'input amplitude', -3, 0, 0.05, -2, function (v) { draw(Math.pow(10, v)); }, function (v) { return fmt(Math.pow(10, v), 3); }));
    draw(0.01);
    box.appendChild(note('F(x) = sgn(x)·ln(1+255|x|)/ln 256, then 7 magnitude bits plus sign; uniform uses Δ = 1/128 on [−1,1]. Quiet samples: μ-law\'s step in the log domain keeps the relative error near 2% at worst, uniform\'s Δ/2 is a fixed 0.0039 so its relative error explodes as x shrinks.'));
  };

  LABS.vq = function (box, T) {
    var pts = calc.cityPoints(42), view = el('div');
    function draw(K) {
      var km = calc.kmeans(pts, K, 10, 7), k1 = calc.kmeans(pts, 1, 10, 7), cv = makeCanvas(170, 170, T); view.textContent = '';
      map(cv, { points: pts, assign: km.assign, centroids: km.centroids }, T);
      view.appendChild(row(cv.canvas, kv([['K', K + ' safehouses'], ['bits per point', fmt(log2(K), 2) + ' (log₂K)'], ['total squared distortion', km.distortion, 3], ['vs K=1 (one centroid)', k1.distortion, 3], ['codebook (side info)', K + ' × 2 floats = ' + K * 2 * 4 + ' B']])));
    }
    box.appendChild(slider('demo-vq-k', 'codebook size K', 2, 8, 1, 3, draw));
    box.appendChild(view); draw(3);
    box.appendChild(note('60 points in 3 clusters; k-means with k-means++ seeding, 10 iterations. Each point is replaced by the index of its nearest centroid. The codebook is side information the decoder must already hold.'));
  };

  LABS.rvq = function (box, T) {
    var pts = calc.cityPoints(42), stages = calc.rvq(pts, 3, 4), view = el('div');
    function draw(n) {
      view.textContent = '';
      var st = stages[n - 1], cv = makeCanvas(170, 170, T), arrows = [];
      if (n === 1) map(cv, { points: pts, assign: st.assign, centroids: st.centroids }, T);
      else {
        var scale = 3, inp = st.input.map(function (p) { return [p[0] * scale, p[1] * scale]; }), cents = st.centroids.map(function (c) { return [c[0] * scale, c[1] * scale]; });
        for (var i = 0; i < 6; i++) arrows.push({ from: inp[i * 10], to: cents[st.assign[i * 10]] });
        map(cv, { points: inp, assign: st.assign, centroids: cents, arrows: arrows, labels: [{ x: -0.95, y: 0.9, text: 'residuals × ' + scale }] }, T);
      }
      var rows = stages.slice(0, n).map(function (s) { return 'stage ' + s.stage + ': ' + fmt(s.distortion, 3); });
      view.appendChild(row(cv.canvas, kv([['stages used', n], ['bits per point', st.bits + ' (2 per stage)'], ['distortion after each stage', rows.join('  ')], ['effective reconstruction points', st.effective + ' (4ⁿ)'], ['centroids stored', st.stored + ' (4n)']])));
    }
    box.appendChild(toggle('demo-rvq-stages', 'stages', ['1', '2', '3'], function (i) { draw(i + 1); }, 1));
    box.appendChild(view); draw(2);
    box.appendChild(note('Stage 1 quantizes the points with K=4; stage 2 runs k-means on the leftovers (point − centroid) with another K=4; stage 3 on the leftovers of that. Arrows show a few residuals snapping to their stage centroid. 4·n centroids give 4ⁿ reconstruction points.'));
  };

  LABS.pq = function (box, T) {
    var vec = [0.9, 0.1, -0.3, 0.8], books = calc.PQ_BOOKS, r = calc.pqEncode(vec, books), maps = [];
    r.halves.forEach(function (hv, b) {
      var cv = makeCanvas(120, 120, T);
      map(cv, { points: [hv], pointColor: T.fg, r: 4, centroids: books[b], arrows: [{ from: hv, to: books[b][r.idx[b]] }] }, T);
      maps.push(col(h('half ' + (b + 1) + ': codebook ' + books[b].map(function (c, i) { return i + ':(' + fmt(c[0], 1) + ',' + fmt(c[1], 1) + ')'; }).join(' ')), row(cv.canvas, numStrip([{ label: 'sub-vector', values: hv, d: 1 }, { label: 'nearest', values: [r.idx[b]], hi: [true] }, { label: 'x̂', values: books[b][r.idx[b]], d: 1 }]))));
    });
    box.appendChild(row.apply(null, maps));
    box.appendChild(kv([['vector', '[' + vec.join(', ') + ']'], ['code (i, j)', '(' + r.idx.join(', ') + ') = 2 + 2 bits'], ['reconstruction', '[' + r.rec.map(function (v) { return fmt(v, 1); }).join(', ') + ']'], ['squared error', r.err, 3]]));
    box.appendChild(note('Each half picks its nearest entry independently. 4 × 4 = 16 effective centroids from 8 stored. The distance from a query to a stored code is the sum of two table lookups, one per half: search runs on the codes without decoding.'));
  };

  LABS.blockint = function (box, T) {
    var x = [0.012, -0.031, 0.004, 0.550, 0.020, -0.008, 0.015, -0.027], view = el('div');
    function draw(rot) {
      view.textContent = '';
      var plain = calc.blockQuant(x), r = rot ? calc.blockQuantRotated(x) : plain, rows = [{ label: 'x', values: x, d: 3 }];
      if (rot) rows.push({ label: 'Hx/√8 (rotated)', values: r.rotated, d: 3 });
      rows.push({ label: 'q = round(·/s)', values: r.q, hi: r.q.map(function (v) { return Math.abs(v) <= 1; }) });
      rows.push({ label: 'x̂', values: r.rec, d: 3 }); rows.push({ label: 'error', values: r.err, d: 3 });
      view.appendChild(numStrip(rows));
      var small = x.map(function (v, i) { return i !== 3 ? Math.abs(r.err[i]) / Math.abs(v) : null; }).filter(function (v) { return v !== null; });
      view.appendChild(kv([['scale s = max|·|/7', r.s, 4], ['values that became 0 or ±1', r.q.filter(function (v) { return Math.abs(v) <= 1; }).length + ' of 8'], ['Σ error²', r.sse.toExponential(2)], ['vs un-rotated Σ error²', plain.sse.toExponential(2)], ['mean relative error of the 7 small values', fmt(sum(small) / small.length * 100, 0) + '%'], ['bits', '8 × 16 = 128 → 8 × 4 + 16 (fp16 scale) = 48']]));
    }
    box.appendChild(toggle('demo-blockint-rotate', '', ['absmax int4', 'rotate (Hadamard) then int4'], function (i) { draw(i === 1); }));
    box.appendChild(view); draw(false);
    box.appendChild(note('One outlier sets the scale, so the seven small values round to zero (100% error). An orthonormal Hadamard rotation spreads the 0.55 as ±0.19 over all eight slots before quantizing; undo it after, and the total error drops. Rotation is free at decode: one tiny matmul.'));
  };

  // ---- entropy -------------------------------------------------------------
  LABS.huffman = function (box, T) {
    var base = { A: 45, B: 13, C: 12, D: 16, E: 9, F: 5 }, view = el('div');
    function draw(pct) {
      var pA = pct / 100, f = {}, scale = (1 - pA) / 0.55; // others keep their proportions and share 1 − P(A)
      Object.keys(base).forEach(function (s) { f[s] = s === 'A' ? pA : base[s] / 100 * scale; });
      var hf = calc.huffman(f), cv = makeCanvas(190, 150, T); tree(cv, hf.tree, T); view.textContent = '';
      var syms = Object.keys(base);
      view.appendChild(row(cv.canvas, col(numStrip([{ label: 'symbol', values: syms }, { label: 'P', values: syms.map(function (s) { return f[s]; }), d: 3 }, { label: 'code', values: syms.map(function (s) { return hf.codes[s]; }), hi: syms.map(function () { return true; }) }, { label: 'length', values: syms.map(function (s) { return hf.lengths[s]; }) }]),
        kv([['mean length', fmt(hf.mean, 3) + ' bits/symbol'], ['entropy H', fmt(hf.entropy, 3) + ' bits/symbol'], ['gap', fmt(hf.mean - hf.entropy, 3) + ' bits (' + fmt((hf.mean / hf.entropy - 1) * 100, 1) + '%)']]))));
    }
    box.appendChild(slider('demo-huffman-skew', 'P(A)', 45, 95, 1, 45, function (v) { draw(v); }, function (v) { return fmt(v / 100, 2); }));
    box.appendChild(view); draw(45);
    box.appendChild(note('CLRS frequencies A:45 B:13 C:12 D:16 E:9 F:5. Merge the two rarest repeatedly; 0 to the left child, 1 to the right. Optimal among integer-length codes, but A can never cost less than 1 bit, so past P(A)=0.5 the gap to H grows.'));
  };

  LABS.arith = function (box, T) {
    var model = [{ sym: 'A', p: 0.5 }, { sym: 'B', p: 0.25 }, { sym: 'C', p: 0.25 }], view = el('div');
    function draw(msg) {
      var r = calc.arith(msg, model), cv = makeCanvas(T.w, 34 * r.steps.length, T); intervals(cv, r.steps, model, T); view.textContent = '';
      view.appendChild(cv.canvas);
      view.appendChild(kv([['message', msg], ['final interval', '[' + fmt(r.steps[r.steps.length - 1].lo, 6) + ', ' + fmt(r.steps[r.steps.length - 1].hi, 6) + ')'], ['width', fmt(r.width, 6) + ' = 1/' + fmt(1 / r.width, 0)], ['bits = −log₂ width', fmt(r.bits, 2) + ' = Σ −log₂p = ' + r.steps.slice(1).map(function (s) { return fmt(-log2(s.p), 0); }).join('+')]]));
    }
    box.appendChild(toggle('demo-arith-msg', 'message', ['ABAC', 'AAAA'], function (i) { draw(i ? 'AAAA' : 'ABAC'); }));
    box.appendChild(view); draw('ABAC');
    box.appendChild(note('Model P(A)=0.5, P(B)=P(C)=0.25, fixed order A|B|C. Each symbol shrinks the interval to its share; the whole message is one number inside the final interval, so symbols can cost fractional bits.'));
  };

  LABS.ans = function (box, T) {
    var freq = { A: 3, B: 1 }, cum = { A: 0, B: 3 }, M = 4, msg = ['A', 'A', 'B'], enc = calc.ransEncode(msg, freq, cum, M, 4), dec = calc.ransDecode(enc.x, msg.length, freq, cum, M);
    var te = el('table', 'cyc-strip'), hd = el('tr'); ['push', 'x', 'x÷f', 'x mod f', "x'", 'Δlog₂x', '−log₂p'].forEach(function (x) { hd.appendChild(el('th', null, x)); }); te.appendChild(hd);
    enc.steps.slice(1).forEach(function (s) { var tr = el('tr'); [s.sym, s.from, Math.floor(s.from / s.f), s.from % s.f, s.x, fmt(s.gain, 3), fmt(-log2(s.f / M), 3)].forEach(function (v, i) { tr.appendChild(el('td', i === 4 ? 'hi' : null, String(v))); }); te.appendChild(tr); });
    var td = el('table', 'cyc-strip'), hd2 = el('tr'); ['pop', 'x', 'x mod M', 'symbol', "x'"].forEach(function (x) { hd2.appendChild(el('th', null, x)); }); td.appendChild(hd2);
    dec.steps.forEach(function (s, i) { var tr = el('tr'); [i + 1, s.from, s.slot, s.sym, s.x].forEach(function (v, j) { tr.appendChild(el('td', j === 3 ? 'hi' : null, String(v))); }); td.appendChild(tr); });
    var teWrap = el('div', 'cyc-scroll'); teWrap.appendChild(te); var tdWrap = el('div', 'cyc-scroll'); tdWrap.appendChild(td);
    box.appendChild(row(col(h('encode A A B from x = 4'), teWrap), col(h('decode by popping from x = ' + enc.x), tdWrap)));
    box.appendChild(kv([['M = 4, f_A = 3, f_B = 1, cum_A = 0, cum_B = 3', ''], ['final state', enc.x + ' (log₂ = ' + fmt(log2(enc.x), 2) + ' bits, started at log₂4 = 2)'], ['decoded (reversed)', dec.symbols.join(' ') + (dec.symbols.join('') === msg.join('') ? ' ✓ round-trips' : ' ✗')]]));
    box.appendChild(note("encode: x' = (x ÷ f_s)·M + cum_s + (x mod f_s). decode: s = symbol whose [cum, cum+f) holds x mod M, then x' = f_s·(x ÷ M) + (x mod M) − cum_s. The state grows by about −log₂p per symbol. LIFO: decoding runs backwards."));
  };

  LABS.rice = function (box, T) {
    var vals = [3, 5, 2, 37, 1, 0, 8], view = el('div'), totals = [0, 1, 2, 3, 4, 5].map(function (k) { return calc.riceTotal(vals, k); }), bestK = totals.indexOf(Math.min.apply(null, totals));
    function draw(k) {
      view.textContent = '';
      var codes = vals.map(function (v) { return calc.rice(v, k); });
      view.appendChild(numStrip([{ label: 'n', values: vals }, { label: 'q = n >> ' + k + ' (unary)', values: codes.map(function (c) { return c.unary; }), hi: codes.map(function (c) { return c.q > 4; }) }, { label: 'low ' + k + ' bits', values: codes.map(function (c) { return c.low || '·'; }) }, { label: 'bits', values: codes.map(function (c) { return c.bits; }) }]));
      view.appendChild(kv([['total for k=' + k, totals[k] + ' bits'], ['totals k=0..5', totals.join(' / ')], ['best k', bestK + ' (' + totals[bestK] + ' bits)' + (k === bestK ? ' ← this one' : '')], ['raw 6-bit fixed', vals.length * 6 + ' bits']]));
    }
    box.appendChild(slider('demo-rice-k', 'Rice parameter k', 0, 5, 1, 2, draw));
    box.appendChild(view); draw(2);
    box.appendChild(note('Each value: q = n >> k ones then a zero, then the k low bits raw. Optimal for geometric distributions, which is what prediction residuals look like. The 37 shows the long unary tail that escape codes exist for.'));
  };

  LABS.rle = function (box, T) {
    var block = [[12, 3, 0, 0], [-2, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]], view = el('div');
    function draw(zig) {
      view.textContent = '';
      var order = zig ? calc.zigzag(4) : calc.rowMajor(4), seq = order.map(function (rc) { return block[rc[0]][rc[1]]; }), pairs = calc.rlePairs(seq);
      var cv = makeCanvas(120, 120, T), cell = 30, ctx = cv.ctx;
      block.forEach(function (r, ri) { r.forEach(function (v, ci) { ctx.fillStyle = v === 0 ? hexA(T.fg, 0.05) : hexA(T.color, 0.35); ctx.fillRect(ci * cell + 1, ri * cell + 1, cell - 2, cell - 2); ctx.fillStyle = T.fg; ctx.textAlign = 'center'; ctx.fillText(String(v), ci * cell + cell / 2, ri * cell + cell / 2); }); });
      ctx.strokeStyle = T.color; ctx.lineWidth = 1.5; ctx.beginPath(); order.forEach(function (rc, i) { var x = rc[1] * cell + cell / 2 + 8, y = rc[0] * cell + cell / 2 + 8; if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y); }); ctx.stroke(); ctx.textAlign = 'left';
      var lastNz = seq.reduce(function (a, v, i) { return v !== 0 ? i : a; }, -1);
      view.appendChild(row(cv.canvas, col(numStrip([{ label: 'scan', values: seq, dim: seq.map(function (v, i) { return v === 0 && i > lastNz; }) }]), kv([['run/value pairs', pairs.map(function (p) { return p.run === null ? 'EOB' : '(' + p.run + ',' + p.val + ')'; }).join(' ')], ['symbols after RLE', pairs.length + ' instead of 16'], ['zero runs before EOB', pairs.filter(function (p) { return p.run > 0; }).length]]))));
    }
    box.appendChild(toggle('demo-rle-scan', 'scan', ['zigzag', 'row-major'], function (i) { draw(i === 0); }));
    box.appendChild(view); draw(true);
    box.appendChild(note('Quantized coefficients live top-left; zigzag visits low frequencies first so every zero lands in one tail ended by EOB. Row-major scan interleaves zeros and values, producing runs in the middle that each cost a symbol.'));
  };

  // ---- perceptual ----------------------------------------------------------
  LABS.psychoacoustic = function (box, T) {
    var nB = 24, bg = 30, masker = 8, view = el('div');
    function draw(level) {
      view.textContent = '';
      var m = calc.maskingCurve(nB, masker, level, bg), levels = []; for (var b = 0; b < nB; b++) levels.push(b === masker ? level : bg);
      var cv = makeCanvas(T.w, 150, T), colors = levels.map(function (_, b) { return b === masker ? T.color : (m.free[b] ? hexA(PALETTE[4], 0.6) : hexA(T.fg, 0.35)); });
      bars(cv, levels, levels.map(function (_, b) { return b % 4 === 0 ? b : ''; }), { max: 100, min: 0, colors: colors, overlay: function (ctx, sx, sy) { ctx.strokeStyle = PALETTE[4]; ctx.lineWidth = 1.5; ctx.beginPath(); m.curve.forEach(function (v, b) { if (b === 0) ctx.moveTo(sx(b), sy(Math.max(0, v))); else ctx.lineTo(sx(b), sy(Math.max(0, v))); }); ctx.stroke(); ctx.fillStyle = PALETTE[4]; ctx.fillText('masking threshold', sx(masker + 1), sy(level - 12) - 8); } }, T);
      view.appendChild(cv.canvas);
      view.appendChild(kv([['masker', 'band ' + masker + ' at ' + level + ' dB; background ' + bg + ' dB'], ['threshold peak', (level - 12) + ' dB, −25 dB/band below, −10 dB/band above'], ['free bands (background under the curve)', m.freeCount + ' of ' + nB + ': ' + m.free.map(function (f, b) { return f ? b : null; }).filter(function (v) { return v !== null; }).join(',')]]));
    }
    box.appendChild(slider('demo-psychoacoustic-level', 'masker level', 50, 90, 1, 80, draw, function (v) { return v + ' dB'; }));
    box.appendChild(view); draw(80);
    box.appendChild(note('Quantization noise in a tinted band sits under the masking threshold and is inaudible, so the quantizer may be brutal there for free. The model is encoder-only; the decoder never sees it.'));
  };

  LABS.chroma = function (box, T) {
    var blocks = [[[200, 150, 100], [204, 152, 102], [198, 149, 99], [202, 151, 101]], [[255, 0, 0], [0, 0, 255], [255, 0, 0], [0, 0, 255]]], view = el('div');
    function draw(bi) {
      view.textContent = '';
      var b = blocks[bi], r = calc.subsample420(b), sw = el('div'); sw.style.display = 'grid'; sw.style.gridTemplateColumns = '28px 28px'; sw.style.gap = '2px';
      b.forEach(function (p) { var s = el('div'); s.style.height = '28px'; s.style.background = 'rgb(' + p.join(',') + ')'; sw.appendChild(s); });
      var sw2 = el('div'); sw2.style.display = 'grid'; sw2.style.gridTemplateColumns = '28px 28px'; sw2.style.gap = '2px';
      r.rec.forEach(function (p) { var s = el('div'); s.style.height = '28px'; s.style.background = 'rgb(' + p.map(function (v) { return Math.round(Math.max(0, Math.min(255, v))); }).join(',') + ')'; sw2.appendChild(s); });
      view.appendChild(row(col(h('RGB in'), sw), numStrip([
        { label: 'pixel', values: [1, 2, 3, 4] }, { label: 'RGB', values: b.map(function (p) { return p.join(','); }) },
        { label: 'Y', values: r.ycc.map(function (c) { return c.y; }), d: 1, hi: [1, 1, 1, 1] }, { label: 'Cb', values: r.ycc.map(function (c) { return c.cb; }), d: 1 }, { label: 'Cr', values: r.ycc.map(function (c) { return c.cr; }), d: 1 },
        { label: 'RGB rebuilt', values: r.rec.map(function (p) { return p.map(function (v) { return fmt(v, 0); }).join(','); }) }, { label: 'RMS error', values: r.err, d: 1, hi: r.err.map(function (e) { return e > 10; }) }]), col(h('4:2:0 out'), sw2)));
      view.appendChild(kv([['kept', '4 Y + mean Cb ' + fmt(r.cb, 1) + ' + mean Cr ' + fmt(r.cr, 1) + ' = 6 numbers instead of 12'], ['largest per-pixel error', fmt(r.maxErr, 1) + ' levels']]));
    }
    box.appendChild(toggle('demo-chroma-block', 'block', ['smooth skin tone', 'sharp red/blue edge'], draw));
    box.appendChild(view); draw(0);
    box.appendChild(note('4:2:0 keeps every Y and one Cb, Cr per 2×2 block: 2:1 on raw samples. On smooth colour the eye barely notices; on a sharp red/blue edge the shared chroma turns both pixels purple.'));
  };

  LABS.gan = function (box, T) {
    var w = calc.ganWave(), sm = calc.smooth(w, 5), sh = calc.shiftWave(w, 2), W = Math.min(T.w, 360);
    var c1 = makeCanvas(W, 70, T), c2 = makeCanvas(W, 70, T);
    wave(c1, [{ data: w, color: T.dim, width: 1 }, { data: sm, color: PALETTE[2], width: 2 }], T, { amp: 1.4 });
    wave(c2, [{ data: w, color: T.dim, width: 1 }, { data: sh, color: T.color, width: 2 }], T, { amp: 1.4 });
    box.appendChild(row(col(h('"MSE-trained": 5-tap smoothing, MSE ' + fmt(calc.mse(w, sm), 3)), c1.canvas), col(h('"adversarial": sharp but shifted 2 samples, MSE ' + fmt(calc.mse(w, sh), 3)), c2.canvas)));
    box.appendChild(kv([['original', '32-sample two-tone burst (grey)'], ['MSE smoothed / shifted', fmt(calc.mse(w, sm), 3) + ' / ' + fmt(calc.mse(w, sh), 3)], ['the one that sounds right', 'the shifted one, despite ' + fmt(calc.mse(w, sh) / calc.mse(w, sm), 1) + '× the MSE']]));
    box.appendChild(note('Minimizing MSE rewards the blurry average of plausible outputs. The sharp version keeps the texture but is misaligned, so MSE punishes it. A discriminator judges plausibility, not alignment; it is training-time only.'));
  };

  // ---- prior ---------------------------------------------------------------
  LABS.tables = function (box, T) {
    var t = el('table', 'cyc-strip'), hd = el('tr'); ['lit/length', 'bits', 'codes'].forEach(function (x) { hd.appendChild(el('th', null, x)); }); t.appendChild(hd);
    calc.DEFLATE_FIXED.forEach(function (r) { var tr = el('tr'); [r.range, r.bits, r.codes].forEach(function (v, i) { var td = el('td', i === 1 ? 'hi' : null, String(v)); td.style.textAlign = 'left'; tr.appendChild(td); }); t.appendChild(tr); });
    var view = el('div'), tableBytes = 80, gain = 0.3;
    function draw(payload) {
      var r = calc.tableBreakEven(payload, tableBytes, gain), lo = calc.tableBreakEven(payload, 40, gain), hi = calc.tableBreakEven(payload, 120, gain); view.textContent = '';
      var cv = makeCanvas(160, 110, T);
      bars(cv, [tableBytes, r.gainBytes], ['table cost', 'gain'], { max: Math.max(tableBytes, r.gainBytes) * 1.3, colors: [hexA(T.fg, 0.4), T.color], values: [tableBytes + ' B', fmt(r.gainBytes, 0) + ' B'] }, T);
      view.appendChild(row(cv.canvas, kv([['payload', payload + ' B'], ['dynamic table cost (typical 40–120 B)', tableBytes + ' B'], ['gain at 0.3 bits/byte', fmt(r.gainBytes, 1) + ' B'], ['net', (r.net >= 0 ? '+' : '') + fmt(r.net, 1) + ' B → ' + (r.pays ? 'dynamic tables pay' : 'fixed table wins')], ['break-even payload', fmt(r.breakEven, 0) + ' B (' + fmt(lo.breakEven, 0) + '–' + fmt(hi.breakEven, 0) + ' B for 40–120 B tables)']])));
    }
    box.appendChild(row(col(h('DEFLATE fixed Huffman code (block type 1)'), t)));
    box.appendChild(slider('demo-tables-payload', 'payload size', 100, 10000, 100, 1000, draw, function (v) { return v >= 1000 ? fmt(v / 1000, 1) + ' KB' : v + ' B'; }));
    box.appendChild(view); draw(1000);
    box.appendChild(note('The fixed table ships with every decoder and costs 0 header bytes; a dynamic table fits the data better but must travel. Stated assumption: the custom tree saves 0.3 bits per payload byte. Break-even = table cost × 8 / 0.3.'));
  };

  LABS.dictionary = function (box, T) {
    var msg = '{"id":4821,"type":"click","ok":true,"ts":1717}', dict = '{"id":1000,"type":"view","ok":false,"ts":1700}', win = 128;
    var a = calc.lz77(msg, win, 3), b = calc.lz77(msg, win, 3, dict);
    box.appendChild(col(h('empty window: ' + a.matches + ' matches, ' + a.tokens.length + ' tokens, toy cost ' + a.cost + ' B'), spans(a.tokens)));
    box.appendChild(col(h('window preloaded with ' + dict), el('div', 'cyc-h', 'preloaded: ' + b.matches + ' matches, ' + b.tokens.length + ' tokens, toy cost ' + b.cost + ' B'), spans(b.tokens)));
    box.appendChild(kv([['message', msg.length + ' B'], ['tokens empty / preloaded', a.tokens.length + ' / ' + b.tokens.length], ['bytes covered by matches', a.covered + ' / ' + b.covered]]));
    box.appendChild(note('Same greedy LZ77 (min match 3, window ' + win + '), different history. A small message has nothing to point back at; a trained dictionary is history both sides already hold. zstd dictionaries carry an ID so the decoder knows which one.'));
  };

  LABS.codebook = function (box, T) {
    var book = [[-0.8, -0.6], [-0.2, 0.7], [0.5, 0.1], [0.9, -0.7]], idx = 2, cv = makeCanvas(120, 120, T);
    map(cv, { centroids: book, labels: [{ x: book[idx][0] - 0.55, y: book[idx][1] - 0.28, text: 'index 2 ↗', color: T.color }] }, T);
    var t = numStrip([{ label: 'index', values: [0, 1, 2, 3], hi: [0, 0, 1, 0] }, { label: 'x', values: book.map(function (c) { return c[0]; }), d: 1 }, { label: 'y', values: book.map(function (c) { return c[1]; }), d: 1 }]);
    box.appendChild(row(cv.canvas, col(t, kv([['stream carries', 'index ' + idx + ' = "' + idx.toString(2).padStart(2, '0') + '" (2 bits)'], ['decoder looks up', '(' + book[idx].join(', ') + ')']]))));
    var view = el('div'), d = 8;
    function draw(e) {
      var K = Math.pow(2, e), c = calc.codebookCost(K, d), cv2 = makeCanvas(170, 100, T); view.textContent = '';
      var pts = []; for (var i = 2; i <= 12; i++) pts.push({ x: i, y: log2(calc.codebookCost(Math.pow(2, i), d).sideBytes) });
      curve(cv2, [{ fn: function (x) { return log2(Math.pow(2, x) * d * 2); } }], [2, 12, 0, 18], [{ x: e, y: log2(c.sideBytes), color: T.fg }], T);
      cv2.ctx.fillStyle = T.fg; cv2.ctx.globalAlpha = 0.6; cv2.ctx.fillText('x: bits/vector  y: log₂ bytes', 26, 10); cv2.ctx.globalAlpha = 1;
      view.appendChild(row(cv2.canvas, kv([['K', K + ' entries, d = ' + d], ['bits per vector', fmt(c.bitsPerVector, 0) + ' (log₂K)'], ['side info the decoder holds', (c.sideBytes >= 1024 ? fmt(c.sideBytes / 1024, 1) + ' KB' : c.sideBytes + ' B') + ' (K × d × 2 B)']])));
    }
    box.appendChild(slider('demo-codebook-k', 'codebook size K', 2, 12, 1, 8, draw, function (e) { return String(Math.pow(2, e)); }));
    box.appendChild(view); draw(8);
    box.appendChild(note('A VQ index is a pointer into a table the decoder must own. Every extra bit per vector doubles the codebook: ratio in the stream is bought with bytes in the decoder.'));
  };

  LABS.decoder_net = function (box, T) {
    var n = calc.decoderNumbers(), cv = makeCanvas(T.w, 90, T), ctx = cv.ctx, pad = 10, W = cv.w - 2 * pad;
    var full = Math.log10(n.weights), small = Math.log10(n.bytesPerSec);
    ctx.fillStyle = hexA(T.fg, 0.35); ctx.fillRect(pad, 14, W * small / full, 22); ctx.fillStyle = T.fg; ctx.fillText('stream: ' + n.bytesPerSec + ' B/s', pad + W * small / full + 6, 25);
    ctx.fillStyle = T.color; ctx.fillRect(pad, 50, W, 22); ctx.fillStyle = T.bg; ctx.fillText('decoder weights: ' + fmt(n.weights / 1e6, 0) + ' MB   (log scale)', pad + 6, 61);
    box.appendChild(cv.canvas);
    box.appendChild(kv([['PCM in', '24 kHz × 16 bit = ' + fmt(n.pcmBps / 1000, 0) + ' kbps'], ['tokens out', '75/s × 8 codebooks × 10 bits = ' + fmt(n.tokenBps / 1000, 0) + ' kbps'], ['ratio', fmt(n.ratio, 0) + ':1'], ['bitstream per second', n.bytesPerSec + ' B'], ['decoder weights', fmt(n.weights / 1e6, 0) + ' MB'], ['decoder : one second of stream', fmt(n.priorRatio, 0).replace(/\B(?=(\d{3})+(?!\d))/g, ',') + ':1']]));
    box.appendChild(note('The prior is enormous and the message is tiny; that is the whole trick. Those weights know what speech sounds like, so 750 bytes per second only have to say which plausible waveform. The receiver must already hold the exact matching decoder.'));
  };

  // ---- level 5: KV cache ---------------------------------------------------
  LABS.unrope = function (box, T) {
    var key = [8.0, 0.3], view = el('div');
    function draw(theta) {
      var r = calc.unropeLab(key, 8, theta, 3), cv = makeCanvas(150, 150, T); view.textContent = '';
      map(cv, { points: r.rotated.map(function (p) { return [p[0] / 9, p[1] / 9]; }), pointColor: hexA(T.fg, 0.7), r: 3, centroids: [[key[0] / 9, key[1] / 9]], centroidColor: T.color }, T);
      var ms = [0, 1, 2, 3, 4, 5, 6, 7];
      view.appendChild(row(col(cv.canvas, el('div', 'cyc-h', 'dots: RoPE\'d pairs m = 0..7 (÷9) · square: all 8 rewound')), col(numStrip([
        { label: 'm', values: ms }, { label: 'x_m', values: r.rotated.map(function (p) { return p[0]; }), d: 2 }, { label: 'y_m', values: r.rotated.map(function (p) { return p[1]; }), d: 2 },
        { label: 'rewound x', values: r.rewound.map(function (p) { return p[0]; }), d: 2, hi: ms.map(function () { return true; }) }, { label: 'rewound y', values: r.rewound.map(function (p) { return p[1]; }), d: 2, hi: ms.map(function () { return true; }) }]))));
      view.appendChild(kv([['range of x, y (RoPE\'d)', fmt(r.rangeBefore[0], 2) + ', ' + fmt(r.rangeBefore[1], 2)], ['range of x, y (rewound)', fmt(r.rangeAfter[0], 2) + ', ' + fmt(r.rangeAfter[1], 2)],
        ['int3 absmax error of y, RMS', fmt(r.before.y.rms, 3) + ' (step ' + fmt(r.before.y.s, 2) + ') → ' + fmt(r.after.y.rms, 3) + ' (step ' + fmt(r.after.y.s, 3) + ')'], ['int3 absmax error of x, RMS', fmt(r.before.x.rms, 3) + ' → ' + fmt(r.after.x.rms, 3)]]));
    }
    box.appendChild(slider('demo-unrope-theta', 'RoPE angle θ', 0.1, 1.5, 0.05, 0.5, draw, function (v) { return fmt(v, 2) + ' rad'; }));
    box.appendChild(view); draw(0.5);
    box.appendChild(note('One channel pair of a key, k = (8.0, 0.3), rotated by m·θ at positions m = 0..7. RoPE smears the stable outlier across both channels, so the small channel y inherits an 8-wide range and a coarse step. Rotating back by −m·θ costs 0 bits: the decoder knows m.'));
  };

  LABS.svd_lowrank = function (box, T) {
    var M = calc.svdLabMatrices(), view = el('div');
    function draw(r) {
      var a = calc.lowRank(M.K, r), b = calc.lowRank(M.V, r); view.textContent = '';
      view.appendChild(row(col(h('K: 8 tokens × 4 dims'), matrix(M.K, { d: 2, small: true })), col(h('V: flat spectrum'), matrix(M.V, { d: 2, small: true }))));
      view.appendChild(kv([['σ of K', a.sigma.map(function (v) { return fmt(v, 2); }).join('  ')], ['σ of V', b.sigma.map(function (v) { return fmt(v, 2); }).join('  ')], ['stored at rank ' + r, (8 * r) + ' + ' + (4 * r) + ' = ' + a.stored + ' numbers vs ' + a.full],
        ['relative error ‖K−K_r‖/‖K‖', fmt(a.relErr * 100, 1) + '%'], ['same rank on V', fmt(b.relErr * 100, 1) + '%']]));
    }
    box.appendChild(slider('demo-svd_lowrank-r', 'rank r', 1, 4, 1, 2, draw));
    box.appendChild(view); draw(2);
    box.appendChild(note('K is built from two fixed basis rows plus N(0, 0.05²) noise, so two singular values carry almost everything and rank 2 reproduces it with 24 numbers instead of 32. V (Gaussian entries) has no such structure: the same rank throws away a large share. Pre-RoPE keys are low-rank, values are not.'));
  };

  LABS.mla = function (box, T) {
    var m = calc.mla(), cv = makeCanvas(T.w, 120, T), view = el('div');
    bars(cv, [m.per.mha, m.per.gqa8, m.per.mla], ['MHA 128 heads', 'GQA-8', 'MLA'], { values: [m.per.mha, m.per.gqa8, m.per.mla].map(function (v) { return v.toLocaleString('en-US'); }), colors: [hexA(T.fg, 0.4), hexA(T.fg, 0.6), T.color] }, T);
    box.appendChild(col(h('elements cached per token per layer'), cv.canvas));
    function draw(i) {
      var p = m.presets[i]; view.textContent = '';
      view.appendChild(kv([['model', p.name], ['per token', p.layers + ' layers × ' + p.perLayer.toLocaleString('en-US') + ' × 2 B = ' + fmt(p.bytesPerToken / 1024, 1) + ' KiB'], ['at 128k tokens (131,072)', fmt(p.at128k / Math.pow(2, 30), 1) + ' GiB'], ['MHA : MLA elements', fmt(m.ratio, 1) + ' : 1 (32,768 / 576)']]));
    }
    box.appendChild(toggle('demo-mla-preset', 'preset', ['Llama-3-70B', 'DeepSeek-V3'], draw));
    box.appendChild(view); draw(0);
    box.appendChild(note('MHA: 128 heads × 128 dims × (K and V). GQA shares 8 KV heads. MLA caches one 512-dim latent plus a 64-dim positional key: 576 elements, decided before pretraining, zero cost at inference. RoPE cannot pass through the low-rank map, hence the small separate key.'));
  };

  LABS.anchor_delta = function (box, T) {
    var x = [100, 103, 105, 104, 108, 110, 109, 113, 115, 114], view = el('div');
    function draw(delta) {
      var r = calc.anchorDelta(x, delta), all = x.map(function () { return true; }); view.textContent = '';
      view.appendChild(numStrip([
        { label: 'x[t]', values: x }, { label: 'CHAIN x[t]−x[t−1]', values: r.chain.map(function (v, i) { return i ? signed(v) : v; }) }, { label: 'ANCHOR x[t]−x[0]', values: r.anchor.map(function (v, i) { return i ? signed(v) : v; }) },
        { label: 'chain x̂ (Δ=' + delta + ')', values: r.recChain }, { label: 'chain error', values: r.errChain.map(function (v) { return signed(v); }), hi: r.errChain.map(function (v) { return Math.abs(v) > delta / 2; }) },
        { label: 'anchor x̂', values: r.recAnchor }, { label: 'anchor error', values: r.errAnchor.map(function (v) { return signed(v); }), hi: r.errAnchor.map(function (v) { return Math.abs(v) > delta / 2; }) }]));
      view.appendChild(kv([['max |error| chain / anchor', r.maxChain + ' / ' + r.maxAnchor + '  (Δ/2 = ' + delta / 2 + ')'], ['dependency depth', 'chain 9 (each token needs the previous)  ·  anchor 1 (all 9 decode in parallel)']]));
    }
    box.appendChild(toggle('demo-anchor_delta-step', 'quantization step Δ', ['1', '2', '4'], function (i) { draw([1, 2, 4][i]); }, 2));
    box.appendChild(view); draw(4);
    box.appendChild(note('Deltas are quantized with round-half-away-from-zero. The chain adds quantized deltas one onto another, so rounding errors pile up along the row; the anchor adds each delta to the exact x[0], so every token\'s error stays within Δ/2. CacheGen puts an anchor every 10 tokens.'));
  };

  LABS.coupled = function (box, T) {
    var pts = calc.COUPLED_PAIRS, sc = calc.signQuant2(pts), km = calc.kmeansInit(pts, [[1.2, 1.1], [-1.2, -1.1], [0.4, 0.35], [-0.4, -0.35]], 10), ent = calc.binEntropy(pts, 4);
    var cv = makeCanvas(170, 170, T), S = 1.6;
    map(cv, { points: pts.map(function (p) { return [p[0] / S, p[1] / S]; }), pointColor: hexA(T.fg, 0.8), r: 3,
      lines: sc.points.map(function (p) { return { x1: p[0] / S - 0.04, y1: p[1] / S, x2: p[0] / S + 0.04, y2: p[1] / S, color: PALETTE[2], width: 3 }; }),
      centroids: km.centroids.map(function (c) { return [c[0] / S, c[1] / S]; }), centroidColor: T.color }, T);
    box.appendChild(row(col(cv.canvas, el('div', 'cyc-h', 'dashes: scalar ±mean levels · squares: joint K=4')), col(numStrip([{ label: 'a', values: pts.map(function (p) { return p[0]; }), d: 2 }, { label: 'b', values: pts.map(function (p) { return p[1]; }), d: 2 }]),
      kv([['SCALAR: sign × mean|·| per channel', '2 × 1 bit, levels ±' + fmt(sc.levels[0], 2) + ', ±' + fmt(sc.levels[1], 2) + ' → MSE ' + fmt(sc.mse, 3)], ['JOINT: k-means K=4 on pairs', '2 bits per pair → MSE ' + fmt(km.mse, 3)],
        ['H(a) + H(b)', fmt(ent.Ha, 2) + ' + ' + fmt(ent.Hb, 2) + ' = ' + fmt(ent.Ha + ent.Hb, 2) + ' bits'], ['H(a, b)', fmt(ent.Hab, 2) + ' bits (4 bins per axis, 16-sample estimate)']]))));
    box.appendChild(note('16 pairs with b ≈ 0.9a. Both quantizers spend 2 bits per pair, but the scalar one pays for a and b as if independent (half its four levels sit where no data is); the joint codebook puts all four centroids on the line the data lives on. The gap H(a)+H(b) − H(a,b) is the mutual information paid twice.'));
  };

  LABS.waterline = function (box, T) {
    var vars = [64, 32, 16, 8, 4, 2, 1, 0.5], view = el('div');
    function draw(e) {
      var theta = Math.pow(2, e), w = calc.waterfill(vars, theta), cv = makeCanvas(T.w, 120, T); view.textContent = '';
      bars(cv, vars.map(function (v) { return log2(v) + 1; }), vars.map(function (v) { return v; }), { max: 8, min: 0, colors: vars.map(function (v) { return v > theta ? T.color : hexA(T.fg, 0.3); }), values: w.bits.map(function (b) { return fmt(b, 1) + 'b'; }), overlay: function (ctx, sx, sy) { var y = sy(log2(theta) + 1); ctx.strokeStyle = PALETTE[0]; ctx.setLineDash([4, 3]); ctx.beginPath(); ctx.moveTo(sx(0) - 20, y); ctx.lineTo(sx(7) + 20, y); ctx.stroke(); ctx.setLineDash([]); ctx.fillStyle = PALETTE[0]; ctx.fillText('θ = ' + fmt(theta, 2), sx(7) - 30, y - 8); } }, T);
      view.appendChild(cv.canvas);
      view.appendChild(numStrip([{ label: 'σ²', values: vars }, { label: 'R_i bits', values: w.bits, d: 2, hi: w.bits.map(function (b) { return b > 0; }) }, { label: 'D_i', values: w.dist, d: 2 }]));
      view.appendChild(kv([['water-filling: ΣR, ΣD', fmt(w.totalBits, 2) + ' bits, ' + fmt(w.totalDist, 2)], ['equal allocation, same ΣR', fmt(w.equalBits, 3) + ' bits each → ΣD = ' + fmt(w.equalDist, 2)], ['distortion ratio equal / water-filling', fmt(w.equalDist / w.totalDist, 1) + '×']]));
    }
    box.appendChild(slider('demo-waterline-theta', 'water level θ', -2, 6, 0.25, 1, draw, function (e) { return fmt(Math.pow(2, e), 2); }));
    box.appendChild(view); draw(1);
    box.appendChild(note('Bars show log₂σ² (+1) per coordinate. R_i = max(0, ½·log₂(σ²_i/θ)), D_i = min(σ²_i, θ): coordinates under water get 0 bits and keep their whole variance as error. Equal allocation spends the same bits on the 0.5-variance tail as on the 64. KVTC solves this by dynamic programming per PCA coordinate.'));
  };

  LABS.fp4 = function (box, T) {
    var xs = calc.FP4_BLOCK, f = calc.fp4Quant(xs), i4 = calc.int4Block(xs), cv = makeCanvas(T.w, 44, T), ctx = cv.ctx, pad = 14, W = cv.w - 2 * pad;
    var X = function (v) { return pad + (v + 6) / 12 * W; };
    ctx.strokeStyle = T.dim; ctx.beginPath(); ctx.moveTo(pad, 22); ctx.lineTo(pad + W, 22); ctx.stroke();
    calc.FP4_GRID.forEach(function (g, i) { ctx.fillStyle = T.color; ctx.fillRect(X(g) - 1.5, 15, 3, 14); ctx.fillStyle = T.fg; ctx.textAlign = 'center'; ctx.fillText(String(g), X(g), i % 2 ? 7 : 38); });
    ctx.textAlign = 'left';
    box.appendChild(col(h('E2M1 grid: ' + calc.FP4_GRID.length + ' values, non-uniform spacing'), cv.canvas));
    box.appendChild(numStrip([{ label: 'x', values: xs, d: 3 }, { label: 'x/s → E2M1', values: f.q, d: 1 }, { label: 'NVFP4 x̂', values: f.rec, d: 3, hi: xs.map(function () { return true; }) }, { label: 'error', values: f.err, d: 3 }, { label: 'int4 x̂', values: i4.rec, d: 3 }, { label: 'error', values: i4.err, d: 3 }]));
    box.appendChild(kv([['NVFP4 scale s = max|x|/6', fmt(f.s, 4) + ' (real format: FP8 E4M3)'], ['int4 scale s = max|x|/7', fmt(i4.s, 4)], ['Σ error² NVFP4 / int4', f.sse.toExponential(2) + ' / ' + i4.sse.toExponential(2)], ['smallest nonzero step near 0', 'NVFP4 0.5·s = ' + fmt(0.5 * f.s, 4) + '  ·  int4 1·s = ' + fmt(i4.s, 4)], ['bits per element', 'NVFP4 (16×4+8)/16 = ' + fmt(f.bits, 1) + '  ·  int4 (32×4+16)/32 = ' + fmt(i4.bits, 1)]]));
    box.appendChild(note('16 values, mostly within ±0.05 and one at 0.42. The floating grid has steps of 0.5 near zero and 2 near the top, so the small values land closer than on int4\'s uniform grid: a tiny μ-law, decoded by the tensor core itself.'));
  };

  LABS.escape = function (box, T) {
    var xs = [0.012, -0.031, 0.004, 0.550, 0.020, -0.008, 0.015, -0.027], r = calc.escape(xs, 3), view = el('div');
    function draw(esc) {
      view.textContent = '';
      var rec = esc ? r.rec : r.plain.rec, err = esc ? r.err : r.plain.err;
      view.appendChild(numStrip([{ label: 'x', values: xs, d: 3 }, { label: esc ? 'x̂ (escape)' : 'x̂ (3-bit absmax)', values: rec, d: 3, hi: xs.map(function (_, i) { return esc && i === r.outlierIndex; }) }, { label: 'error', values: err, d: 3 }]));
      view.appendChild(kv(esc
        ? [['outlier', 'x[' + r.outlierIndex + '] = ' + xs[r.outlierIndex] + ' → fp16 side list + ' + r.idxBits + '-bit index'], ['step for the other 7', 's = ' + fmt(Math.max.apply(null, xs.filter(function (_, i) { return i !== r.outlierIndex; }).map(Math.abs)), 3) + '/3 = ' + fmt(r.restScale, 4)], ['Σ error²', r.sse.toExponential(2)], ['bits per element', '3 + (16 + ' + r.idxBits + ')/8 = ' + fmt(r.bitsEscape, 2) + ' here; with 1% outliers ≈ ' + fmt(r.bitsOnePercent, 2)]]
        : [['step', 's = 0.55/3 = ' + fmt(r.plain.s, 4)], ['values that collapsed to 0', r.plain.q.filter(function (q) { return q === 0; }).length + ' of 8'], ['Σ error²', r.plain.sse.toExponential(2)], ['bits per element', '3']]));
    }
    box.appendChild(toggle('demo-escape-mode', '', ['no escape', 'escape the outlier'], function (i) { draw(i === 1); }));
    box.appendChild(view); draw(false);
    box.appendChild(note('A 3-bit grid is sized by its extreme: one 0.55 forces a step of 0.18 on values of 0.01. Pull it into a sparse full-precision list and the rest get a step 18× finer. In a real tensor the list holds 1% of the values, so the per-element cost is about 3.3 bits.'));
  };

  LABS.evict = function (box, T) {
    var w = calc.EVICT_W, view = el('div');
    function draw(k) {
      var r = calc.evict(w, k, 8); view.textContent = '';
      var pages = el('div'); pages.style.display = 'grid'; pages.style.gridTemplateColumns = 'repeat(8, 1fr)'; pages.style.gap = '3px'; pages.style.fontVariantNumeric = 'tabular-nums';
      w.forEach(function (v, i) { var c = el('div', 'cyc-cell', fmt(v, 3).replace(/^0/, '')); c.style.textAlign = 'center'; c.style.fontSize = '.7em'; c.style.minWidth = '0'; c.style.padding = '2px 0'; if (r.keep[i]) { c.style.background = hexA(T.color, 0.45); } else { c.style.opacity = '.3'; } pages.appendChild(c); });
      view.appendChild(pages);
      view.appendChild(kv([['kept', k + ' of 16 tokens, top-k by w'], ['attention mass kept Σw', fmt(r.mass, 3)], ['memory kept', fmt(r.memory * 100, 0) + '%'],
        ['pages freed (8 tokens each)', r.pagesFreed + ' of 2: ' + r.pages.map(function (p, i) { return 'page ' + i + ' ' + (p.freed ? 'FREED' : p.kept + ' kept'); }).join(', ')], ['memory actually reclaimed', fmt(r.pagesFreed * 8 / 16 * 100, 0) + '%']]));
    }
    box.appendChild(slider('demo-evict-k', 'tokens kept k', 1, 16, 1, 5, draw));
    box.appendChild(view); draw(5);
    box.appendChild(note('Rows are pages: 8 tokens each. Token 0 carries 0.30 of the attention, tokens 12–15 another 0.30, the rest is spread thin. Keeping the top 5 keeps 60% of the mass at 31% of the tokens, but a paged allocator only frees a page when all 8 of its tokens are gone. The score is the perceptual-model stage\'s job; the pages are the system\'s.'));
  };

  LABS.sinkguard = function (box, T) {
    var w = calc.sinkWeights(), exemptIdx = [0, 28, 29, 30, 31], view = el('div');
    function draw(exempt) {
      var r = calc.sinkguard(w, exempt ? exemptIdx : [], 0.25), cv = makeCanvas(T.w, 110, T); view.textContent = '';
      bars(cv, w, w.map(function (_, i) { return i % 4 === 0 ? i : ''; }), { max: 0.4, colors: w.map(function (_, i) { return r.exempt[i] ? T.color : hexA(T.fg, 0.4); }) }, T);
      view.appendChild(cv.canvas);
      view.appendChild(kv([['weights', 'w₀ = ' + fmt(w[0], 2) + ' (sink), w₂₈..₃₁ = ' + fmt(w[28], 3) + ' each, the other 27 share ' + fmt(sum(w.slice(1, 28)), 2)], ['quantization error per quantized position', '0.25 (simulated 2-bit)'], ['exempt', exempt ? 'positions 0 and 28–31 (' + fmt(r.exemptMass * 100, 0) + '% of the mass, 5 of 32 tokens)' : 'none'],
        ['weighted error Σ wᵢ·eᵢ', fmt(r.weighted, 4)]]));
    }
    box.appendChild(toggle('demo-sinkguard-exempt', '', ['quantize everything', 'exempt sink + newest 4'], function (i) { draw(i === 1); }));
    box.appendChild(view); draw(false);
    box.appendChild(note('Attention is the reader: error on a position is felt in proportion to its weight. Five of 32 tokens carry 65% of this query\'s attention; keeping them at full precision cuts the weighted error from 0.25 to 0.0875 while 27 of 32 tokens still sit at 2 bits. KIVI keeps the last 128 in fp16; KVQuant the sink.'));
  };

  LABS.spotlight = function (box, T) {
    var keys = calc.spotKeys(), q = [1, 0.4], view = el('div'), S = 4.5;
    function draw(k) {
      var r = calc.spotlight(keys, q, k), cv = makeCanvas(170, 170, T); view.textContent = '';
      var lit = {}; r.topExact.forEach(function (i) { lit[i] = true; });
      map(cv, { points: keys.map(function (p) { return [p[0] / S, p[1] / S]; }), assign: keys.map(function (_, i) { return lit[i] ? 4 : 7; }), r: 3, arrows: [{ from: [0, 0], to: [q[0] / S * 2, q[1] / S * 2], color: T.fg }], labels: [{ x: q[0] / S * 2 + 0.05, y: q[1] / S * 2, text: 'q' }] }, T);
      r.topProxy.forEach(function (i) { var ctx = cv.ctx, x = 10 + (keys[i][0] / S + 1) / 2 * (cv.w - 20), y = cv.h - 10 - (keys[i][1] / S + 1) / 2 * (cv.h - 20); ctx.strokeStyle = T.color; ctx.lineWidth = 1.5; ctx.beginPath(); ctx.arc(x, y, 6, 0, Math.PI * 2); ctx.stroke(); });
      view.appendChild(row(cv.canvas, kv([['exact top-' + k, r.topExact.join(', ')], ['proxy top-' + k, r.topProxy.join(', ')], ['overlap', r.overlap + ' of ' + k], ['attention mass in exact top-' + k, fmt(r.massExact, 3)], ['mass the proxy\'s picks actually carry', fmt(r.massProxy, 3)]])));
      view.appendChild(el('div', 'cyc-h', 'dots: keys (yellow = exact top-k)  ·  rings: proxy picks'));
    }
    box.appendChild(slider('demo-spotlight-k', 'top-k', 1, 8, 1, 4, draw));
    box.appendChild(view); draw(4);
    box.appendChild(note('32 keys, q = (1, 0.4), exact weights softmax(q·k). The proxy is a 1-bit copy of each key: sign per coordinate times that key\'s mean |coordinate|, scored the same way. Four keys sit near q and take most of the mass; distortion on the rest would cost nothing. The proxy is the address book: cheap, read in full every step, never compressed further.'));
  };

  LABS.basis = function (box, T) {
    var dims = [128, 512, 1024, 4096, 8192], view = el('div');
    function fmtBytes(b) { return b >= Math.pow(2, 30) ? fmt(b / Math.pow(2, 30), 2) + ' GiB' : b >= Math.pow(2, 20) ? fmt(b / Math.pow(2, 20), 1) + ' MiB' : fmt(b / 1024, 1) + ' KiB'; }
    function draw(i) {
      var d = dims[i], b = calc.basisBytes(d, 80); view.textContent = '';
      view.appendChild(kv([['basis', d + ' × ' + d + ' bf16 = ' + fmtBytes(b.shared)], ['one shared across all layers', fmtBytes(b.shared)], ['one per layer (× 80)', fmtBytes(b.perLayer)],
        ['Hadamard rotation', '0 bytes: defined by a formula'], ['DCT', '0 bytes: in the standard'], ['KVTC reported', '2.4% of a 70B model ≈ ' + fmt(b.kvtcParams / 1e9, 2) + ' B parameters ≈ ' + fmt(b.kvtcBytes / 1e9, 1) + ' GB in bf16']]));
    }
    box.appendChild(slider('demo-basis-d', 'basis size d', 0, dims.length - 1, 1, 2, draw, function (i) { return String(dims[i]); }));
    box.appendChild(view); draw(2);
    box.appendChild(note('A fitted orthonormal transform is side information exactly like a VQ codebook: fitted once on calibration data, shipped with the model, never in the stream. The bytes grow as d²; sharing one basis across layers is what keeps PCA affordable.'));
  };

  LABS.tiered = function (box, T) {
    var sizes = [2048, 4096, 8192], state = { B: 4096, c: 1 }, view = el('div');
    function draw() {
      var r = calc.tiered(state.B, state.c), cv = makeCanvas(T.w, 110, T); view.textContent = '';
      bars(cv, [r.bandwidth / 1e9, r.gather / 1e9, r.pcie / 1e9], ['needed', 'gather kernel', 'PCIe 5 ×16'], { max: 70, values: [fmt(r.bandwidth / 1e9, 2), '37', '64'].map(function (v) { return v + ' GB/s'; }), colors: [r.bandwidth > r.gather ? PALETTE[5] : T.color, hexA(T.fg, 0.4), hexA(T.fg, 0.25)] }, T);
      view.appendChild(cv.canvas);
      view.appendChild(kv([['GPU buffer B', state.B + ' entries/layer → miss rate ' + fmt(r.miss * 100, 1) + '% (measured on GLM-5.1, HiSparse)'], ['fetched per token', '78 × 2048 × 656 B × ' + fmt(r.miss, 3) + ' = ' + fmt(r.perToken / 1e6, 1) + ' MB'], ['bandwidth per request', fmt(r.perToken / r.step / 1e9, 2) + ' GB/s at 30 ms/token'], ['× ' + state.c + ' concurrent', fmt(r.bandwidth / 1e9, 2) + ' GB/s' + (r.bandwidth > r.gather ? ' — over the 37 GB/s a gather kernel achieves' : '')],
        ['GPU memory per request', 'B × 78 × 656 B = ' + fmt(r.gpuBytes / 1e9, 2) + ' GB (vs ' + fmt(r.dense128k / 1e9, 0) + ' GB dense at 128k)']]));
    }
    box.appendChild(slider('demo-tiered-buffer', 'GPU buffer B', 0, 2, 1, 1, function (i) { state.B = sizes[i]; draw(); }, function (i) { return String(sizes[i]); }));
    box.appendChild(slider('demo-tiered-concurrency', 'concurrency', 1, 16, 1, 1, function (v) { state.c = v; draw(); }));
    box.appendChild(view); draw();
    box.appendChild(note('The lossless cache lives in host DRAM; the GPU keeps a small LRU buffer of exact entries and pulls misses over PCIe. Fixed: 78 layers, 2048 picks per layer, 656 B per entry, 30 ms per token. A hot shelf twice the pick list misses one in eight; concurrency multiplies the traffic until it no longer hides behind compute.'));
  };

  // ===========================================================================
  // Public API
  // ===========================================================================
  function cssVar(elm, name, fallback) {
    try { var v = getComputedStyle(elm).getPropertyValue(name).trim(); return v || fallback; } catch (e) { return fallback; }
  }
  function render(container, fighterId) {
    if (!container) return;
    container.textContent = '';
    var lab = LABS[fighterId]; if (!lab) return;
    injectStyle();
    var f = (CYC.FIGHTERS || []).find(function (x) { return x.id === fighterId; }) || {};
    var roleColor = (CYC.ROLES && CYC.ROLES[f.role] && CYC.ROLES[f.role].color) || '#246b66';
    var root = el('div', 'cyc-demo'); root.dataset.fighter = fighterId;
    container.appendChild(root);
    var T = {
      color: cssVar(container, '--role-color', roleColor),
      fg: cssVar(container, '--fg', cssVar(container, '--ink', '#2a211b')),
      bg: cssVar(container, '--panel', cssVar(container, '--paper-2', '#ebdfc8')),
      font: cssVar(container, '--font-mono', '"JetBrains Mono", ui-monospace, monospace'),
      w: Math.max(220, Math.min(380, (container.clientWidth || 380) - 8))
    };
    T.dim = hexA(T.fg, 0.3);
    PALETTE = PALETTE_VARS.map(function (v, i) { return cssVar(container, v, PALETTE[i]); });
    NEG_COLOR = cssVar(container, '--bad', NEG_COLOR);
    root.style.setProperty('--role-color', T.color);
    lab(root, T, f);
  }
  function has(id) { return Object.prototype.hasOwnProperty.call(LABS, id); }

  CYC.Demos = { render: render, has: has, calc: calc, ids: Object.keys(LABS) };
})();
