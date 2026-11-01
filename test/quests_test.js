// Node tests for js/quests.js (docs/SIDEQUESTS.md "Tests"). Run: node test/quests_test.js
'use strict';
const assert = require('assert');
const path = require('path');

global.window = global;
require(path.join(__dirname, '..', 'js', 'data.js'));
require(path.join(__dirname, '..', 'js', 'quests_data.js'));
require(path.join(__dirname, '..', 'js', 'quests.js'));

const Q = window.CYC.Quests;
const C = Q.calc;
const SEED = 20240601;

let passed = 0, failed = 0;
function test(name, fn) {
  try { fn(); passed++; console.log('ok   ' + name); }
  catch (e) { failed++; console.log('FAIL ' + name + '\n     ' + (e && e.message ? e.message : e)); }
}
const near = (a, b, eps, msg) => assert.ok(Math.abs(a - b) <= eps, `${msg || ''} expected ${b} ± ${eps}, got ${a}`);

// ---- contract
test('contract: render/has/calc exist and every CYC.SIDEQUESTS id has a renderer', () => {
  assert.strictEqual(typeof Q.render, 'function');
  assert.strictEqual(typeof Q.has, 'function');
  assert.strictEqual(typeof Q.calc, 'object');
  const ids = window.CYC.SIDEQUESTS.map((q) => q.id);
  assert.strictEqual(ids.length, 12);
  for (const id of ids) assert.ok(Q.has(id), 'missing renderer for ' + id);
  assert.ok(!Q.has('nope'));
  assert.strictEqual(typeof document, 'undefined', 'test must run without a DOM');
});

test('prng: mulberry32 is deterministic and in [0,1)', () => {
  const a = C.mulberry32(42), b = C.mulberry32(42);
  for (let i = 0; i < 1000; i++) { const x = a(); assert.strictEqual(x, b()); assert.ok(x >= 0 && x < 1); }
});

// ---- shared math
test('dct8/idct8 round-trip and orthonormality (Parseval)', () => {
  const r = C.rng(1), blk = new Float64Array(64);
  for (let i = 0; i < 64; i++) blk[i] = r.range(0, 255);
  const coef = C.dct8(blk), back = C.idct8(coef);
  for (let i = 0; i < 64; i++) near(back[i], blk[i], 1e-9, 'roundtrip');
  let e1 = 0, e2 = 0;
  for (let i = 0; i < 64; i++) { e1 += blk[i] * blk[i]; e2 += coef[i] * coef[i]; }
  near(e2, e1, 1e-6, 'Parseval');
  const flat = new Float64Array(64).fill(100);
  near(C.dct8(flat)[0], 800, 1e-9, 'DC of a flat block = 8 × mean');
});

test('zigzag is a permutation of 0..63 starting 0, 1, 8, 16, 9, 2', () => {
  assert.deepStrictEqual(C.ZIGZAG.slice(0, 6), [0, 1, 8, 16, 9, 2]);
  assert.deepStrictEqual([...C.ZIGZAG].sort((a, b) => a - b), [...Array(64).keys()]);
});

test('entropy and huffman basics', () => {
  near(C.entropyOfCounts([1, 1, 1, 1]), 2, 1e-12);
  near(C.entropyOfCounts([1, 0, 0, 0]), 0, 1e-12);
  assert.deepStrictEqual(C.huffmanLengths([1, 1, 1, 1]), [2, 2, 2, 2]);
  assert.deepStrictEqual(C.huffmanLengths([0, 5, 0]), [0, 1, 0]);
});

// ---- hufftree
test('hufftree: real Huffman on A45 B13 C12 D16 E9 F5 gives mean 2.24; H ≈ 2.22', () => {
  const opt = C.hufftree.optimal();
  near(opt.mean, 2.24, 1e-9, 'mean');
  assert.deepStrictEqual(opt.lengths, [1, 3, 3, 3, 4, 4]);
  // The spec quotes H ≈ 2.196; the real value of −Σ p log2 p for these counts is 2.2199 (the game shows the real one).
  near(C.hufftree.entropy(), 2.2199, 1e-3, 'H');
  assert.ok(opt.mean > C.hufftree.entropy(), 'Huffman pays the integer-length tax');
  near(C.hufftree.kraft(opt.lengths), 1, 1e-12, 'Kraft');
  assert.strictEqual(C.hufftree.score(2.24, 2.24), 100);
  near(C.hufftree.score(2.34, 2.24), 75, 1e-6, 'score −25 per 0.1 bit');
  // a bad tree: merge A first
  near(C.hufftree.meanLength([3, 3, 3, 2, 2, 2]), (45 * 3 + 13 * 3 + 12 * 3 + 16 * 2 + 9 * 2 + 5 * 2) / 100, 1e-12);
});

// ---- waterfill
test('waterfill: greedy (water-filling) allocation beats the uniform allocation in MSE under the same budget', () => {
  const W = C.waterfill;
  const img = W.image(SEED);
  assert.strictEqual(img.w, 32);
  const an = W.analyze(img);
  assert.strictEqual(an.blocks.length, 16);
  assert.strictEqual(an.variances.length, 8);
  for (let k = 1; k < 8; k++) assert.ok(an.variances[k] < an.variances[0], 'DC band has the most energy');
  const uni = W.uniform(), opt = W.optimal(an);
  assert.strictEqual(W.blockBits(uni), W.BUDGET);
  assert.strictEqual(W.blockBits(opt), W.BUDGET);
  assert.ok(opt.every((b) => b >= 0 && b <= 8));
  const mseUni = W.reconstruct(an, uni).mse, mseOpt = W.reconstruct(an, opt).mse;
  assert.ok(mseOpt <= mseUni, `optimal ${mseOpt} should be ≤ uniform ${mseUni}`);
  assert.ok(mseOpt < mseUni * 0.9, `expected a clear win, got ${mseOpt} vs ${mseUni}`);
  const th = W.theta(an.variances, W.BUDGET);
  near(th.rates.reduce((s, x) => s + 8 * x, 0), W.BUDGET, 1e-6, 'Σ 8·R_k = budget');
  assert.strictEqual(W.score(mseOpt, mseUni) < 100, true);
  assert.strictEqual(W.score(mseOpt, mseOpt), 100);
  const zero = W.reconstruct(an, [0, 0, 0, 0, 0, 0, 0, 0]);
  assert.ok(zero.mse > mseUni, 'zero bits is worse than uniform');
});

// ---- jnd
test('jnd: staircase direction, threshold, degrade monotone in Δ, entropy cost', () => {
  const J = C.jnd;
  near(J.nextDelta(24, true), 24 / 1.25, 1e-12, 'seen → gentler');
  near(J.nextDelta(24, false), 24 * 1.25, 1e-12, 'missed → harsher');
  near(J.threshold([1, 1, 1, 1, 1, 1, 4, 4, 4, 4, 4, 4]), 4, 1e-9);
  near(J.threshold([2, 8, 2, 8, 2, 8]), 4, 1e-9, 'geometric mean');
  const img = J.image(SEED);
  assert.strictEqual(img.w, 64);
  const a = J.degrade(img, 4).mse, b = J.degrade(img, 16).mse, c = J.degrade(img, 64).mse;
  assert.ok(a < b && b < c, `MSE should grow with Δ: ${a} ${b} ${c}`);
  const lossless = J.blockCost([img], 1), coarse = J.blockCost([img], 16);
  assert.ok(lossless > coarse && coarse > 0, `${lossless} > ${coarse}`);
  near(J.bitsSaved(16), 4, 1e-12);
  const ycc = J.makeChroma(img), d = J.degradeChroma(ycc, 16);
  assert.strictEqual(d.ycc.Y, ycc.Y, 'luma untouched in the chroma variant');
  const rgb = J.ycbcrToRgb(ycc);
  assert.strictEqual(rgb.data.length, 64 * 64 * 4);
});

// ---- phrasebook
test('phrasebook: trainer dictionary saves ≥ 30% on the test message; foreign message saves < 10%', () => {
  const P = C.phrasebook;
  const tr = P.train(P.SAMPLES, P.BUDGET);
  assert.ok(tr.dict.length <= P.BUDGET, 'under budget');
  assert.ok(tr.picks.length >= 2);
  const sv = P.savings(P.TEST, tr.dict);
  assert.ok(sv.pct >= 30, `trainer should save ≥ 30% on the test message, got ${sv.pct.toFixed(1)}% (${sv.without.bits} → ${sv.withDict.bits} bits)`);
  const fo = P.savings(P.FOREIGN, tr.dict);
  assert.ok(fo.pct < 10, `foreign message should save < 10%, got ${fo.pct.toFixed(1)}% (${fo.without.bits} → ${fo.withDict.bits} bits)`);
  assert.strictEqual(P.score(sv.saved, sv.saved), 100);
  assert.strictEqual(P.score(sv.saved / 2, sv.saved), 50);
});

test('phrasebook: tokenizer yields keys/values/numbers/punctuation and LZ77 flags dictionary matches in gold', () => {
  const P = C.phrasebook;
  const toks = P.tokenize(P.SAMPLES[0]);
  assert.deepStrictEqual(toks.map((t) => t.text), P.SAMPLES[0].match(/"[^"]*":|"[^"]*"|-?\d+(?:\.\d+)?|[A-Za-z_]+|[^\w"\s]+|\s+/g));
  assert.ok(toks.some((t) => t.kind === 'key' && t.text === '"type":'));
  assert.ok(toks.some((t) => t.kind === 'str' && t.text === '"click"'));
  assert.ok(toks.some((t) => t.kind === 'num' && t.text === '4821'));
  assert.ok(toks.some((t) => t.kind === 'word' && t.text === 'true'));
  assert.ok(toks.some((t) => t.kind === 'punct' && t.text === '{'));
  assert.strictEqual(toks.map((t) => t.text).join(''), P.SAMPLES[0], 'tokens cover the message');
  const res = C.lz77('abcabcabc', 'xyz');
  assert.strictEqual(res.bits, 3 * 9 + 20, 'abc literal, then one match of 6');
  const withDict = C.lz77('abcabcabc', 'abc');
  assert.strictEqual(withDict.tokens[0].type, 'match');
  assert.strictEqual(withDict.tokens[0].dict, true);
  assert.strictEqual(withDict.bits, 20);
});

// ---- safehouses
test('safehouses: best-of-10 k-means distortion ≤ a single run from a bad init; Lloyd never increases distortion', () => {
  const V = C.safehouses;
  const pts = V.points(SEED);
  assert.strictEqual(pts.length, 120);
  for (const v of pts) assert.ok(v >= -1 && v <= 1);
  const bad = new Float64Array([-0.55, -0.35, -0.5, -0.3, -0.6, -0.4, -0.52, -0.38]); // all four in one cluster
  const single = V.lloyd(pts, bad, 20);
  const best = V.bestOf(pts, V.K, 10, SEED + 99);
  assert.ok(best.distortion <= single.distortion + 1e-12, `best ${best.distortion} ≤ bad-init ${single.distortion}`);
  for (let i = 1; i < single.history.length; i++) assert.ok(V.distortion(pts, single.history[i]) <= V.distortion(pts, single.history[i - 1]) + 1e-9, 'monotone');
  assert.strictEqual(best.runs.length, 10);
  assert.strictEqual(V.score(best.distortion, best.distortion), 100);
  assert.ok(V.score(best.distortion, single.distortion) <= 100);
  assert.strictEqual(V.bitsPerPoint, 2);
  assert.strictEqual(V.codebookBytes, 32);
});

// ---- blockpack
test('blockpack: bpw(4,32,no min) = 4.5, bpw(8,32) = 8.5; Hadamard is orthogonal; rotation reduces RMS at b=4, n=256', () => {
  const B = C.blockpack;
  assert.strictEqual(B.bpw(4, 32, false), 4.5);
  assert.strictEqual(B.bpw(8, 32, false), 8.5);
  assert.strictEqual(B.bpw(4, 32, true), 5.0);
  const w = B.weights(SEED);
  assert.strictEqual(w.length, 256);
  const outliers = Array.from(w).filter((v) => Math.abs(v) >= 0.3);
  assert.strictEqual(outliers.length, 3, 'three outliers');
  const h = B.hadamard(w), hh = B.hadamard(h);
  let n1 = 0, n2 = 0;
  for (let i = 0; i < 256; i++) { n1 += w[i] * w[i]; n2 += h[i] * h[i]; near(hh[i], w[i], 1e-12, 'H·H = I'); }
  near(n2, n1, 1e-12, 'norm preserved');
  const plain = B.quantize(w, { b: 4, n: 256, min: false, rot: false });
  const rot = B.quantize(w, { b: 4, n: 256, min: false, rot: true });
  assert.ok(rot.rms < plain.rms, `rotation should reduce RMS: ${rot.rms} vs ${plain.rms}`);
  const q40 = B.quantize(w, { b: 4, n: 32, min: false, rot: false });
  near(q40.bpw, 4.5, 1e-12);
  assert.ok(q40.rms < plain.rms, 'small blocks contain the outliers');
  const q80 = B.quantize(w, { b: 8, n: 32, min: false, rot: false });
  assert.ok(q80.rms < q40.rms);
  assert.strictEqual(B.score(q40.rms, q80.rms, q80.bpw), 0, 'over budget → 0');
  assert.strictEqual(B.score(q40.rms, q40.rms, 4.5), 100);
  assert.strictEqual(B.formatName(4, 32, false, false), 'Q4_0: 4.5 bpw');
  assert.strictEqual(B.formatName(4, 32, true, false), 'Q4_1: 5.0 bpw');
  assert.strictEqual(B.formatName(8, 32, false, false), 'Q8_0: 8.5 bpw');
  assert.ok(B.formatName(4, 64, true, false).startsWith('≈ Q4_K'));
  assert.ok(B.formatName(4, 32, false, true).includes('QuaRot'));
  near(B.modelBytes(4.5) / 1e9, 3.9375, 1e-9, '7B at 4.5 bpw');
});

// ---- raceaccess
test('raceaccess: Huffman total bits < 4096×8 and ≥ 4096×H; table verdicts', () => {
  const R = C.raceaccess;
  const sym = R.symbols(SEED);
  assert.strictEqual(sym.length, 4096);
  const an = R.analyze(sym);
  assert.ok(an.totalBits < 4096 * 8, `${an.totalBits} < ${4096 * 8}`);
  assert.ok(an.totalBits >= an.entropyBits, `${an.totalBits} ≥ ${an.entropyBits}`);
  assert.ok(an.meanBits > 4.5 && an.meanBits < 5.8, `mean ${an.meanBits} should be ~5 bits`);
  assert.deepStrictEqual(R.steps(3187, 64), { checkpoint: 3136, steps: 52 });
  assert.deepStrictEqual(R.steps(3187, null), { checkpoint: 0, steps: 3188 });
  assert.deepStrictEqual(R.steps(7, 1), { checkpoint: 7, steps: 1 });
  const t = R.table(an.totalBits);
  assert.strictEqual(t.length, 6);
  const by = Object.fromEntries(t.map((r) => [r.label, r]));
  assert.strictEqual(by['1'].score, 30, 'a checkpoint per element is bigger than fixed');
  assert.strictEqual(by['16'].score, 100);
  assert.strictEqual(by['64'].score, 100);
  assert.strictEqual(by['256'].score, 60);
  assert.strictEqual(by['none'].score, 60);
  near(by['64'].meanSteps, 32.5, 1e-9);
  assert.strictEqual(by['none'].checkpoints, 0);
  assert.strictEqual(by['16'].huffmanBytes, Math.ceil(an.totalBits / 8) + 2 * 256);
});

// ---- shannon
test('shannon: cost and score formulas', () => {
  const S = C.shannon;
  assert.strictEqual(S.cost(1), 0);
  assert.strictEqual(S.cost(2), 1);
  assert.strictEqual(S.cost(4), 2);
  assert.strictEqual(S.score(1.0), 100);
  assert.strictEqual(S.score(1.5), 100);
  assert.strictEqual(S.score(4.7), 0);
  near(S.score(3.1), 50, 1e-9);
  assert.strictEqual(S.ALPHABET.length, 27);
  for (const s of S.SENTENCES) for (const ch of s) assert.ok(S.ALPHABET.includes(ch), `sentence char "${ch}" in alphabet`);
});

// ---- level 5: siamese
test('siamese: at slope 0.9 joint MSE < scalar MSE / 2 and I > 1 bit', () => {
  const S = C.siamese;
  const P = S.pairs(SEED, 0.9);
  assert.strictEqual(P.n, 4000);
  const sc = S.scalar(P), jt = S.joint(P, SEED + 11);
  assert.strictEqual(sc.points.length, 4, '2×2 grid');
  assert.strictEqual(jt.points.length, 4, '4 centroids');
  assert.ok(jt.mse < sc.mse / 2, `joint ${jt.mse} should be < scalar/2 ${sc.mse / 2}`);
  const e = S.entropy(P);
  assert.ok(e.I > 1, `I = ${e.I} should exceed 1 bit (H(a)=${e.Ha}, H(b)=${e.Hb}, H(a,b)=${e.Hab})`);
  assert.ok(e.Hab <= e.Ha + e.Hb + 1e-9 && e.Hab >= Math.max(e.Ha, e.Hb) - 1e-9, 'joint entropy bounds');
  // weighted fit: weighted MSE goes down, plain MSE goes up (or equal)
  const w = S.weights(P.a), scw = S.scalar(P, w);
  assert.ok(scw.wmse <= sc.wmse + 1e-12, `weighted fit should lower weighted MSE: ${scw.wmse} vs ${sc.wmse}`);
  assert.ok(scw.mse >= sc.mse - 1e-12, `weighted fit should not lower plain MSE: ${scw.mse} vs ${sc.mse}`);
  assert.strictEqual(S.score(3, 1), 100);
  assert.strictEqual(S.score(1, 1), 20, 'minimum 20 once both runs are done');
  near(S.score(1.75, 1), 50, 1e-9);
  // slope 0: a and b nearly independent → little mutual information
  assert.ok(S.entropy(S.pairs(SEED, 0.0)).I < 0.2);
});

// ---- level 5: anchorchain
test('anchorchain: CHAIN entropy ≤ ANCHOR(g=10) entropy ≤ NONE entropy; CHAIN worst error ≥ ANCHOR worst error', () => {
  const A = C.anchorchain;
  const x = A.matrix(SEED);
  assert.strictEqual(x.length, 64 * 16);
  const none = A.code(x, 'none', 10), chain = A.code(x, 'chain', 10), anchor = A.code(x, 'anchor', 10);
  assert.ok(chain.H <= anchor.H && anchor.H <= none.H, `H: chain ${chain.H} ≤ anchor ${anchor.H} ≤ none ${none.H}`);
  assert.ok(chain.bitsPerElement <= anchor.bitsPerElement && anchor.bitsPerElement <= none.bitsPerElement, `bits/element: chain ${chain.bitsPerElement} ≤ anchor ${anchor.bitsPerElement} ≤ none ${none.bitsPerElement}`);
  assert.ok(chain.maxErr >= anchor.maxErr, `chain worst ${chain.maxErr} ≥ anchor worst ${anchor.maxErr}`);
  assert.ok(anchor.maxErr <= 0.5 + 1e-9 && none.maxErr <= 0.5 + 1e-9, 'closed-loop residuals keep |error| ≤ Δ/2');
  assert.ok(chain.maxErr > 0.5, 'open-loop chain drifts beyond one rounding step');
  assert.deepStrictEqual([none.depth, anchor.depth, chain.depth], [1, 2, 64]);
  assert.strictEqual(anchor.raw, 16 * Math.ceil(64 / 10), 'anchors every g tokens per channel');
  assert.strictEqual(A.score('anchor', 10), 100);
  assert.strictEqual(A.score('anchor', 4), 70);
  assert.strictEqual(A.score('chain', 10), 40);
  assert.strictEqual(A.score('none', 10), 20);
});

// ---- level 5: rankkeys
test('rankkeys: K overlap at r=16 ≥ 15/16 and storage(16) < L·d; V relative error at r=16 > 0.5; SVD is exact', () => {
  const R = C.rankkeys;
  const M = R.matrices(SEED);
  assert.strictEqual(M.K.length, 256 * 64);
  // SVD sanity: V orthonormal, Σσ² = ‖M‖_F², M = P·Vᵀ
  const s = R.svd(M.K);
  let fro = 0; for (const v of M.K) fro += v * v;
  near(s.sigma.reduce((a, v) => a + v * v, 0), fro, 1e-6 * fro, 'Σσ² = ‖K‖²');
  for (let i = 1; i < 64; i++) assert.ok(s.sigma[i] <= s.sigma[i - 1] + 1e-12, 'sorted');
  for (let i = 0; i < 64; i += 7) for (let j = 0; j < 64; j += 5) { let d = 0; for (let k = 0; k < 64; k++) d += s.V[i][k] * s.V[j][k]; near(d, i === j ? 1 : 0, 1e-8, 'V orthonormal'); }
  for (let t = 0; t < 256; t += 37) for (let k = 0; k < 64; k += 9) { let v = 0; for (let i = 0; i < 64; i++) v += s.P[t * 64 + i] * s.V[i][k]; near(v, M.K[t * 64 + k], 1e-8, 'K = P Vᵀ'); }
  assert.ok(s.sigma[0] > 7 && s.sigma[0] < 9, `top singular value ≈ 8, got ${s.sigma[0]}`);
  const aK = R.analyzeK(M.K, M.Q);
  assert.ok(aK.overlap[16] >= 15, `overlap at r=16 should be ≥ 15, got ${aK.overlap[16]}`);
  assert.ok(aK.overlap[64] >= 15.999, 'full rank reproduces the exact top-16');
  assert.ok(aK.scoreErr[64] < 1e-9 && aK.scoreErr[1] > aK.scoreErr[16], 'score error falls with rank');
  const st = R.storage(16);
  assert.ok(st.lowrank < st.full, `${st.lowrank} < ${st.full}`);
  near(st.crossover, 51.2, 1e-9);
  const aV = R.analyzeV(M.V);
  assert.ok(aV.relErr[16] > 0.5, `V relative error at r=16 should be > 0.5, got ${aV.relErr[16]}`);
  assert.ok(aV.relErr[64] < 1e-9);
  const sc = R.score(16, aK.overlap);
  assert.ok(sc.rStar >= 1 && sc.rStar <= 16);
  assert.strictEqual(R.score(sc.rStar, aK.overlap).score, 100);
  assert.strictEqual(R.score(sc.rStar + 2, aK.overlap).score, 100);
  assert.strictEqual(R.score(64, aK.overlap).score, Math.max(0, 100 - 10 * (64 - sc.rStar - 2)));
});

// ---- level 5: pagepurge
test('pagepurge: AUTO frees ≤ 2 pages before compaction; hole-fill reaches 4 pages with copies ≤ 64', () => {
  const P = C.pagepurge;
  const { w, sinks } = P.weights(SEED);
  assert.strictEqual(w.length, 256);
  near(w.reduce((a, b) => a + b, 0), 1, 1e-9, 'normalized');
  assert.strictEqual(sinks.length, 3);
  const ev = P.auto(w);
  const st = P.stats(w, ev);
  assert.strictEqual(st.kept, 64);
  assert.ok(st.pagesFreed <= 2, `AUTO freed ${st.pagesFreed} pages`);
  assert.strictEqual(st.pagesNeeded, 4);
  assert.ok(st.mass > 0.5, 'the 64 most important tokens carry most of the mass');
  const hf = P.holeFill(ev);
  assert.strictEqual(hf.pagesUsed, 4, `hole-fill should reach 4 pages, got ${hf.pagesUsed}`);
  assert.ok(hf.copies <= 64, `copies ${hf.copies} ≤ 64`);
  assert.strictEqual(hf.copies, hf.minCopies, 'hole-fill is copy-optimal');
  assert.strictEqual(hf.layout.filter((t) => t >= 0).length, 64, 'no survivor lost');
  const io = P.inOrder(ev);
  assert.strictEqual(io.pagesUsed, 4);
  assert.ok(io.copies >= hf.copies, `in-order ${io.copies} ≥ hole-fill ${hf.copies}`);
  assert.strictEqual(P.score(64, 4, hf.copies, hf.minCopies), 100);
  assert.strictEqual(P.score(64, 4, 2 * hf.minCopies + 5, hf.minCopies), 70);
  assert.strictEqual(P.score(64, st.pagesUsed, 0, hf.minCopies), st.pagesUsed === 4 ? 100 : 40);
  assert.strictEqual(P.score(65, 5, 0, 0), 0);
});

console.log(`\n${passed} passed, ${failed} failed`);
process.exit(failed ? 1 : 0);
