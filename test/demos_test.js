// node test/demos_test.js — pure-compute checks for js/demos.js (no DOM needed).
'use strict';
global.window = global;
require('../js/data.js');
require('../js/demos.js');

const calc = CYC.Demos.calc;
let failures = 0, passes = 0;
function check(name, cond, detail) {
  if (cond) { passes++; console.log('ok   ' + name + (detail !== undefined ? '  (' + detail + ')' : '')); }
  else { failures++; console.log('FAIL ' + name + (detail !== undefined ? '  (' + detail + ')' : '')); }
}
function near(a, b, eps) { return Math.abs(a - b) <= (eps === undefined ? 1e-9 : eps); }
function eqArr(a, b, eps) { return a.length === b.length && a.every((v, i) => near(v, b[i], eps)); }

// --- every fighter has a lab, every lab is a fighter -------------------------
const ids = CYC.FIGHTERS.map(f => f.id);
check('every fighter has a lab', ids.every(id => CYC.Demos.has(id)), ids.filter(id => !CYC.Demos.has(id)).join(',') || 'all 34');
check('every lab id is a fighter', CYC.Demos.ids.every(id => ids.includes(id)), CYC.Demos.ids.filter(id => !ids.includes(id)).join(',') || 'none stray');
check('one lab per fighter', CYC.Demos.ids.length === CYC.FIGHTERS.length, CYC.Demos.ids.length);
check('has() is false for unknown id', !CYC.Demos.has('nope'));

// --- Huffman (CLRS) ---------------------------------------------------------
const hf = calc.huffman({ A: 45, B: 13, C: 12, D: 16, E: 9, F: 5 });
check('Huffman CLRS lengths A1 B3 C3 D3 E4 F4',
  hf.lengths.A === 1 && hf.lengths.B === 3 && hf.lengths.C === 3 && hf.lengths.D === 3 && hf.lengths.E === 4 && hf.lengths.F === 4,
  JSON.stringify(hf.lengths));
check('Huffman CLRS mean 2.24', near(hf.mean, 2.24), hf.mean.toFixed(4));
check('Huffman CLRS entropy ≈ 2.22 (spec says ≈ 2.20)', near(hf.entropy, 2.22, 0.01), hf.entropy.toFixed(4));
check('Huffman codes are prefix-free', (() => { const cs = Object.values(hf.codes); return cs.every(a => cs.every(b => a === b || !b.startsWith(a))); })());

// --- Arithmetic coding -------------------------------------------------------
const model = [{ sym: 'A', p: 0.5 }, { sym: 'B', p: 0.25 }, { sym: 'C', p: 0.25 }];
const ar = calc.arith('ABAC', model);
check('arith ABAC final width 1/64', near(ar.width, 1 / 64), ar.width);
check('arith ABAC interval [0.296875, 0.3125)', near(ar.steps[4].lo, 0.296875) && near(ar.steps[4].hi, 0.3125), ar.steps[4].lo + '..' + ar.steps[4].hi);
check('arith ABAC costs 6 bits = 1+2+1+2', near(ar.bits, 6) && near(ar.sumBits, 6), ar.bits);
check('arith AAAA width 1/16, 4 bits', near(calc.arith('AAAA', model).width, 1 / 16) && near(calc.arith('AAAA', model).bits, 4));

// --- rANS ------------------------------------------------------------------
const freq = { A: 3, B: 1 }, cum = { A: 0, B: 3 };
const enc = calc.ransEncode(['A', 'A', 'B'], freq, cum, 4, 4);
const dec = calc.ransDecode(enc.x, 3, freq, cum, 4);
check('rANS encode AAB from 4 → 5, 6, 27', eqArr(enc.steps.map(s => s.x), [4, 5, 6, 27]), enc.steps.map(s => s.x).join(','));
check('rANS decode round-trips to AAB', dec.symbols.join('') === 'AAB', dec.symbols.join(''));
check('rANS decode returns to initial state 4', dec.x === 4, dec.x);
check('rANS state grows ≈ −log2 p (A: 0.415, B: 2 bits)',
  near(Math.log2(27) - Math.log2(4), 3 * -Math.log2(0.75) + -Math.log2(0.25), 0.5), (Math.log2(27) - 2).toFixed(3));

// --- Rice ------------------------------------------------------------------
check('Rice 37, k=3 → 8 bits', calc.rice(37, 3).bits === 8, calc.rice(37, 3).unary + '|' + calc.rice(37, 3).low);
check('Rice 37, k=3 unary is 11110 and low bits 101', calc.rice(37, 3).unary === '11110' && calc.rice(37, 3).low === '101');
const vals = [3, 5, 2, 37, 1, 0, 8];
const totals = [0, 1, 2, 3, 4, 5].map(k => calc.riceTotal(vals, k));
const bestK = totals.indexOf(Math.min(...totals));
check('Rice best k for the lab values is 1 or 2', bestK === 1 || bestK === 2, 'totals k=0..5: ' + totals.join(',') + ' best ' + bestK);

// --- LZW -------------------------------------------------------------------
const lzw = calc.lzw('ABABABA');
check('LZW ABABABA → [65,66,256,258]', eqArr(lzw.codes, [65, 66, 256, 258]), lzw.codes.join(','));
check('LZW dictionary gains 256=AB 257=BA 258=ABA', lzw.added.map(a => a.code + '=' + a.phrase).join(' ') === '256=AB 257=BA 258=ABA', lzw.added.map(a => a.code + '=' + a.phrase).join(' '));

// --- Haar ------------------------------------------------------------------
const haar = calc.haar([10, 12, 14, 16, 100, 98, 96, 94]);
check('Haar level 1 averages [11,15,99,95]', eqArr(haar.avg, [11, 15, 99, 95]), haar.avg.join(','));
check('Haar level 1 details [-1,-1,1,1]', eqArr(haar.det, [-1, -1, 1, 1]), haar.det.join(','));
const pyr = calc.haarPyramid([10, 12, 14, 16, 100, 98, 96, 94], 2);
check('Haar level 2 → [13,97], [-2,2]', eqArr(pyr[1].avg, [13, 97]) && eqArr(pyr[1].det, [-2, 2]), pyr[1].avg + ' | ' + pyr[1].det);

// --- LZ77 ------------------------------------------------------------------
const lz = calc.lz77('the_cat_sat_on_the_mat_the_cat_sat', 32, 3);
const longest = Math.max(...lz.tokens.filter(t => t.type === 'match').map(t => t.len));
check('LZ77 toy finds a match of length ≥ 8', longest >= 8, 'longest ' + longest + ', tokens: ' + lz.tokens.filter(t => t.type === 'match').map(t => '(' + t.dist + ',' + t.len + ')').join(' '));
check('LZ77 tokens reproduce the text', lz.tokens.map(t => t.type === 'lit' ? t.ch : t.text).join('') === 'the_cat_sat_on_the_mat_the_cat_sat');
const msg = '{"id":4821,"type":"click","ok":true,"ts":1717}', dict = '{"id":1000,"type":"view","ok":false,"ts":1700}';
const cold = calc.lz77(msg, 128, 3), warm = calc.lz77(msg, 128, 3, dict);
check('dictionary lab: preloaded window finds more matches', warm.matches > cold.matches && warm.tokens.length < cold.tokens.length, cold.matches + ' → ' + warm.matches + ' matches, ' + cold.tokens.length + ' → ' + warm.tokens.length + ' tokens');
check('dictionary lab: preloaded tokens reproduce the message', warm.tokens.map(t => t.type === 'lit' ? t.ch : t.text).join('') === msg);

// --- μ-law -----------------------------------------------------------------
const mu = calc.mulawQuant(0.01), un = calc.uniformQuant8(0.01);
check('μ-law 8-bit relative error at 0.01 < 5%', mu.relErr < 0.05, (mu.relErr * 100).toFixed(2) + '%');
check('μ-law worst-case relative error ≈ 2%', mu.worst < 0.03, (mu.worst * 100).toFixed(2) + '%');
check('uniform 8-bit at 0.01 is far worse than μ-law', un.relErr > mu.relErr && un.worst > 0.3, 'uniform actual ' + (un.relErr * 100).toFixed(1) + '%, worst ' + (un.worst * 100).toFixed(1) + '%');
check('μ-law inverse undoes forward', near(calc.mulawInv(calc.mulaw(0.37)), 0.37, 1e-12));

// --- k-means on the city map -------------------------------------------------
const pts = calc.cityPoints(42);
check('city map has 60 points', pts.length === 60);
const k1 = calc.kmeans(pts, 1, 10, 7), k3 = calc.kmeans(pts, 3, 10, 7);
check('k-means K=3 distortion below K=1 by ≥ 5×', k1.distortion / k3.distortion >= 5, 'K=1 ' + k1.distortion.toFixed(3) + ', K=3 ' + k3.distortion.toFixed(3) + ', ratio ' + (k1.distortion / k3.distortion).toFixed(1));
const rvq = calc.rvq(pts, 3, 4);
check('RVQ distortion falls stage by stage', rvq[0].distortion > rvq[1].distortion && rvq[1].distortion > rvq[2].distortion, rvq.map(s => s.distortion.toFixed(3)).join(' > '));
check('RVQ effective points 4,16,64 vs stored 4,8,12', eqArr(rvq.map(s => s.effective), [4, 16, 64]) && eqArr(rvq.map(s => s.stored), [4, 8, 12]));

// --- DCT -------------------------------------------------------------------
const rowIn = [52, 55, 61, 66, 70, 61, 64, 73];
check('DCT round-trips', eqArr(calc.idct(calc.dct(rowIn)), rowIn, 1e-9));
check('DCT is orthonormal (energy preserved)', near(calc.energy(calc.dct(rowIn)), calc.energy(rowIn), 1e-6));
check('DCT keep k=3 gives RMS error < 4 grey levels', Math.sqrt(calc.dctKeep(rowIn, 3).mse) < 4, 'RMS ' + Math.sqrt(calc.dctKeep(rowIn, 3).mse).toFixed(2));

// --- MDCT ------------------------------------------------------------------
check('MDCT delay 1152 @ 48 kHz = 24 ms', near(calc.mdctDelayMs(1152, 48000), 24));
check('Princen–Bradley: w²[n]+w²[n+N] = 1', calc.princenBradley(64).every(v => near(v, 1, 1e-12)));

// --- Prism -----------------------------------------------------------------
const ycc = [[200, 180, 170], [205, 182, 172], [198, 179, 168], [202, 181, 171]].map(p => calc.rgb2ycc(...p));
const spread = a => Math.max(...a) - Math.min(...a);
check('Prism: chroma spread tiny vs luma spread', spread(ycc.map(c => c.cb)) < 2 && spread(ycc.map(c => c.cr)) < 2 && spread(ycc.map(c => c.y)) > 2,
  'Y ' + spread(ycc.map(c => c.y)).toFixed(2) + ' Cb ' + spread(ycc.map(c => c.cb)).toFixed(2) + ' Cr ' + spread(ycc.map(c => c.cr)).toFixed(2));
check('Prism: M/S of 1000, 980 → 990, 20', calc.midSide(1000, 980).m === 990 && calc.midSide(1000, 980).s === 20);
const back = calc.ycc2rgb(ycc[0].y, ycc[0].cb, ycc[0].cr);
check('YCbCr inverse recovers RGB', near(back.r, 200, 1e-9) && near(back.g, 180, 1e-9) && near(back.b, 170, 1e-9));

// --- PCA -------------------------------------------------------------------
const pp = calc.pcaPoints(40, 11), pca = calc.pca(pp);
check('PCA axis ≈ slope 0.8 (angle ≈ 38.7°)', near(pca.angle * 180 / Math.PI, Math.atan(0.8) * 180 / Math.PI, 4), (pca.angle * 180 / Math.PI).toFixed(1) + '°');
check('PCA λ1 ≫ λ2', pca.lambda[0] > 10 * pca.lambda[1], pca.lambda.map(v => v.toFixed(4)).join(', '));
const gq0 = calc.gridQuant(pp, 2), gqR = calc.gridQuant(calc.rotate(pp, pca.angle), 2);
check('2-bit grid error lower after rotation', gqR.err < gq0.err, gq0.err.toFixed(3) + ' → ' + gqR.err.toFixed(3));

// --- BWT -------------------------------------------------------------------
const bwt = calc.bwt('banana$');
check('BWT banana$ → annb$aa', bwt.last === 'annb$aa', bwt.last + ' primary ' + bwt.primary);
check('BWT primary index 4', bwt.primary === 4);

// --- Delta -----------------------------------------------------------------
const x = [100, 102, 105, 107, 108, 108, 107, 105];
check('delta order 1', eqArr(calc.delta(x, 1), [100, 2, 3, 2, 1, 0, -1, -2]), calc.delta(x, 1).join(','));
check('delta order 2', eqArr(calc.delta(x, 2), [100, 2, 1, -1, -1, -1, -1, -1]), calc.delta(x, 2).join(','));
check('raw needs 7 bits, order-1 residuals 3 bits with sign', calc.bitsUnsigned(x) === 7 && calc.bitsSigned(calc.delta(x, 1).slice(1)) === 3);

// --- LPC -------------------------------------------------------------------
const sig = calc.lpcSignal(16), fit = calc.lpcFit2(sig);
const eRaw = calc.energy(sig.slice(2)), eFixed = calc.energy(calc.lpcResidual(sig, 2, -1)), eFit = calc.energy(calc.lpcResidual(sig, fit.a1, fit.a2));
check('LPC fitted predictor beats fixed beats raw', eFit < eFixed && eFit < eRaw, 'raw ' + eRaw + ' fixed ' + eFixed.toFixed(0) + ' fitted ' + eFit.toFixed(0) + ' a1=' + fit.a1.toFixed(3) + ' a2=' + fit.a2.toFixed(3));
check('LPC fit ≈ 2·0.9·cos0.6, −0.81 (theory for the damped cosine)', near(fit.a1, 2 * 0.9 * Math.cos(0.6), 0.1) && near(fit.a2, -0.81, 0.1));

// --- Spatial ---------------------------------------------------------------
const pf = calc.pngFilters([[10, 11, 12, 13], [20, 21, 22, 23], [30, 31, 32, 33], [40, 41, 42, 43]]);
check('PNG filter sums Sub 112, Up 166, Paeth 52; Paeth wins', pf.sums.sub === 112 && pf.sums.up === 166 && pf.sums.paeth === 52 && pf.winner === 'paeth', JSON.stringify(pf.sums));

// --- Motion ----------------------------------------------------------------
const f1 = calc.frameWithSquare(8, 2, 2, 3, 9), f2 = calc.frameWithSquare(8, 3, 2, 3, 9), mv = calc.motionSearch(f1, f2, 2);
check('motion vector (+1, 0), residual all zero', mv.dx === 1 && mv.dy === 0 && calc.countNonzero(mv.residual) === 0, '(' + mv.dx + ',' + mv.dy + ') sad ' + mv.sad);
check('no-MC residual has 6 nonzero cells (two 3×3 squares overlapping in 2 columns)', calc.countNonzero(calc.frameDiff(f2, f1)) === 6, calc.countNonzero(calc.frameDiff(f2, f1)));

// --- Context model ---------------------------------------------------------
const th = calc.contextCounts('the_theme_of_the_thesis', 'th');
check('P(e|th) = 5/31 ≈ 0.16 → 2.6 bits', th.counts.e === 4 && th.total === 4 && near(calc.laplace(4, 4, 27), 5 / 31) && near(calc.cost(5 / 31), 2.63, 0.01), calc.cost(5 / 31).toFixed(3));

// --- Uniform / qmatrix -----------------------------------------------------
const uq = calc.uniformQ([3.7, -1.2, 0.4, 8.9, -5.5, 2.1], 1, 9);
check('uniform Δ=1: q = round(x)', eqArr(uq.q, [4, -1, 0, 9, -6, 2]) && uq.levels === 19 && uq.bits === 5, uq.q.join(',') + ' levels ' + uq.levels);
const qb = calc.quantizeBlock(calc.WIKI_DCT, calc.JPEG_Q50, 1);
check('qmatrix: Wikipedia block DC → −26, zeros ≥ 40 of 64', qb.q[0][0] === -26 && qb.zeros >= 40, 'zeros ' + qb.zeros);

// --- PQ / blockint ---------------------------------------------------------
const pq = calc.pqEncode([0.9, 0.1, -0.3, 0.8], calc.PQ_BOOKS);
check('PQ code and reconstruction', pq.idx.length === 2 && pq.rec.length === 4 && pq.err < 0.1, '(' + pq.idx + ') err ' + pq.err.toFixed(3));
const bx = [0.012, -0.031, 0.004, 0.550, 0.020, -0.008, 0.015, -0.027], bq = calc.blockQuant(bx), br = calc.blockQuantRotated(bx);
check('blockint: 7 of 8 become 0 or ±1', bq.q.filter(v => Math.abs(v) <= 1).length === 7, bq.q.join(','));
check('blockint: Hadamard rotation lowers Σerror²', br.sse < bq.sse, bq.sse.toExponential(2) + ' → ' + br.sse.toExponential(2));
check('Hadamard is its own inverse', eqArr(calc.hadamard8(calc.hadamard8(bx)), bx, 1e-12));

// --- RLE / zigzag ----------------------------------------------------------
const block = [[12, 3, 0, 0], [-2, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]];
const zz = calc.zigzag(4).map(rc => block[rc[0]][rc[1]]);
check('zigzag scan starts 12,3,-2,0', eqArr(zz.slice(0, 4), [12, 3, -2, 0]), zz.join(','));
check('zigzag RLE pairs (0,12)(0,3)(0,-2) EOB', calc.rlePairs(zz).map(p => p.run === null ? 'EOB' : '(' + p.run + ',' + p.val + ')').join('') === '(0,12)(0,3)(0,-2)EOB');
check('row-major scan needs a mid-run', calc.rlePairs(calc.rowMajor(4).map(rc => block[rc[0]][rc[1]])).some(p => p.run > 0));

// --- Perceptual ------------------------------------------------------------
const mk = calc.maskingCurve(24, 8, 80, 30);
check('masking at 80 dB frees bands 7..11', mk.free.map((f, b) => f ? b : null).filter(v => v !== null).join(',') === '7,8,9,10,11', mk.freeCount);
const smooth = calc.subsample420([[200, 150, 100], [204, 152, 102], [198, 149, 99], [202, 151, 101]]), edge = calc.subsample420([[255, 0, 0], [0, 0, 255], [255, 0, 0], [0, 0, 255]]);
check('4:2:0 error small on smooth block, large on red/blue edge', smooth.maxErr < 3 && edge.maxErr > 50, smooth.maxErr.toFixed(2) + ' vs ' + edge.maxErr.toFixed(1));
const w = calc.ganWave();
check('gan: shifted waveform has higher MSE than smoothed', calc.mse(w, calc.shiftWave(w, 2)) > calc.mse(w, calc.smooth(w, 5)), calc.mse(w, calc.smooth(w, 5)).toFixed(3) + ' < ' + calc.mse(w, calc.shiftWave(w, 2)).toFixed(3));

// --- Prior -----------------------------------------------------------------
check('tables break-even for 80 B table at 0.3 bits/B ≈ 2133 B', near(calc.tableBreakEven(1000, 80, 0.3).breakEven, 2133.3, 0.1));
check('codebook K=256, d=8 → 4096 B side info, 8 bits/vector', calc.codebookCost(256, 8).sideBytes === 4096 && calc.codebookCost(256, 8).bitsPerVector === 8);
const dn = calc.decoderNumbers();
check('decoder_net: 384 kbps → 6 kbps (64:1), 750 B/s, ~27000:1', dn.pcmBps === 384000 && dn.tokenBps === 6000 && dn.ratio === 64 && dn.bytesPerSec === 750 && near(dn.priorRatio, 26667, 1), dn.priorRatio.toFixed(0));

// --- Level 5: KV cache -------------------------------------------------------
const L5 = ['unrope', 'svd_lowrank', 'mla', 'anchor_delta', 'coupled', 'waterline', 'fp4', 'escape', 'evict', 'sinkguard', 'spotlight', 'basis', 'tiered'];
check('level 5: all 13 labs present', L5.every(id => CYC.Demos.has(id)), L5.filter(id => !CYC.Demos.has(id)).join(',') || '13');
check('level 5: lab count is 47', CYC.Demos.ids.length === 47 && CYC.FIGHTERS.length === 47, CYC.Demos.ids.length + ' labs, ' + CYC.FIGHTERS.length + ' fighters');

const ur = calc.unropeLab([8.0, 0.3], 8, 0.5, 3);
check('unrope: rewound pairs all equal (8.0, 0.3) within 1e-9', ur.rewound.every(p => near(p[0], 8.0) && near(p[1], 0.3)), ur.rewound.map(p => p.map(v => v.toFixed(6)).join(',')).join(' '));
check('unrope: y-channel int3 error drops ≥ 5×', ur.before.y.rms / Math.max(ur.after.y.rms, 1e-15) >= 5, ur.before.y.rms.toFixed(3) + ' → ' + ur.after.y.rms.toExponential(2));
check('unrope: rotated y range is wide, rewound is 0', ur.rangeBefore[1] > 5 && near(ur.rangeAfter[1], 0), ur.rangeBefore[1].toFixed(2));

const mats = calc.svdLabMatrices(), lrK = calc.lowRank(mats.K, 2), lrV = calc.lowRank(mats.V, 2);
check('svd_lowrank: K has two large and two tiny singular values', lrK.sigma[1] > 5 * lrK.sigma[2], lrK.sigma.map(v => v.toFixed(2)).join(','));
check('svd_lowrank: rank-2 error small for K, large for V', lrK.relErr < 0.1 && lrV.relErr > 0.3, (lrK.relErr * 100).toFixed(1) + '% vs ' + (lrV.relErr * 100).toFixed(1) + '%');
check('svd_lowrank: rank 4 is exact, storage 8r+4r', near(calc.lowRank(mats.K, 4).relErr, 0, 1e-6) && lrK.stored === 24 && lrK.full === 32);
const eigA = [[4, 1, 0.5], [1, 3, 0.2], [0.5, 0.2, 1]], eg = calc.symEig(eigA);
check('symEig reconstructs A = V diag(λ) Vᵀ', eigA.every((row, i) => row.every((v, j) => near(v, eg.values.reduce((s, l, k) => s + l * eg.vectors[k][i] * eg.vectors[k][j], 0), 1e-9))));

const mla = calc.mla();
check('mla: 32768 / 576 = 56.9', mla.per.mha === 32768 && mla.per.mla === 576 && near(mla.ratio, 56.9, 0.05), mla.ratio.toFixed(2));
check('mla: Llama-3-70B 320 KiB/token, 40 GiB at 131072 tokens', mla.presets[0].bytesPerToken === 320 * 1024 && near(mla.presets[0].at128k / 2 ** 30, 40), (mla.presets[0].at128k / 2 ** 30).toFixed(2));
check('mla: DeepSeek-V3 68.6 KiB/token (spec: 8.4 GiB at 128k; computed at 131072 tokens)', near(mla.presets[1].bytesPerToken / 1024, 68.6, 0.05), (mla.presets[1].at128k / 2 ** 30).toFixed(2) + ' GiB');

const ad1 = calc.anchorDelta([100, 103, 105, 104, 108, 110, 109, 113, 115, 114], 1), ad4 = calc.anchorDelta([100, 103, 105, 104, 108, 110, 109, 113, 115, 114], 4);
check('anchor_delta: chain deltas', eqArr(ad1.chain, [100, 3, 2, -1, 4, 2, -1, 4, 2, -1]), ad1.chain.join(','));
check('anchor_delta: anchor deltas', eqArr(ad1.anchor, [100, 3, 5, 4, 8, 10, 9, 13, 15, 14]), ad1.anchor.join(','));
check('anchor_delta at Δ=4: chain max error > anchor max error, anchor ≤ Δ/2', ad4.maxChain > ad4.maxAnchor && ad4.maxAnchor <= 2, 'chain ' + ad4.maxChain + ' anchor ' + ad4.maxAnchor);

const cp = calc.COUPLED_PAIRS, sq2 = calc.signQuant2(cp), kmc = calc.kmeansInit(cp, [[1.2, 1.1], [-1.2, -1.1], [0.4, 0.35], [-0.4, -0.35]], 10), be = calc.binEntropy(cp, 4);
check('coupled: joint K=4 beats scalar sign quantizer at 2 bits/pair', kmc.mse < sq2.mse, sq2.mse.toFixed(3) + ' → ' + kmc.mse.toFixed(3));
check('coupled: H(a,b) < H(a) + H(b)', be.Hab < be.Ha + be.Hb, be.Ha.toFixed(2) + '+' + be.Hb.toFixed(2) + ' vs ' + be.Hab.toFixed(2));

const wf = calc.waterfill([64, 32, 16, 8, 4, 2, 1, 0.5], 2);
check('waterline θ=2: bits [2.5,2,1.5,1,0.5,0,0,0]', eqArr(wf.bits, [2.5, 2, 1.5, 1, 0.5, 0, 0, 0]), wf.bits.join(','));
check('waterline θ=2: ΣD = 13.5', near(wf.totalDist, 13.5), wf.totalDist);
check('waterline: equal allocation is worse', wf.equalDist > wf.totalDist, wf.equalDist.toFixed(2));

const grid = calc.FP4_GRID;
check('fp4: grid has 15 distinct values incl. 0 and ±6', new Set(grid).size === 15 && grid.length === 15 && grid.includes(0) && grid.includes(6) && grid.includes(-6), grid.join(','));
const f4 = calc.fp4Quant(calc.FP4_BLOCK), i4 = calc.int4Block(calc.FP4_BLOCK);
check('fp4: NVFP4 lower error than int4 on the block, both 4.5 bits', f4.sse < i4.sse && f4.bits === 4.5 && i4.bits === 4.5, f4.sse.toExponential(2) + ' vs ' + i4.sse.toExponential(2));

const es = calc.escape([0.012, -0.031, 0.004, 0.550, 0.020, -0.008, 0.015, -0.027], 3);
check('escape: without escape 7 values collapse to 0', es.plain.q.filter(q => q === 0).length === 7 && near(es.plain.s, 0.55 / 3));
check('escape: outlier is x[3], rest step 0.031/3, error drops', es.outlierIndex === 3 && near(es.restScale, 0.031 / 3) && es.sse < es.plain.sse / 10, es.plain.sse.toExponential(2) + ' → ' + es.sse.toExponential(2));
check('escape: bits 3 vs 3 + 19/8 = 5.375', es.bitsPlain === 3 && near(es.bitsEscape, 5.375), es.bitsEscape);

check('evict: weights sum to 1', near(calc.EVICT_W.reduce((a, b) => a + b, 0), 1, 1e-9));
const ev5 = calc.evict(calc.EVICT_W, 5, 8);
check('evict at k=5 keeps ≥ 0.60 mass', ev5.mass >= 0.6 - 1e-9, ev5.mass.toFixed(3));
check('evict: pages free only when all 8 tokens are gone', calc.evict(calc.EVICT_W, 1, 8).pagesFreed === 1 && ev5.pagesFreed === 0);

const sw = calc.sinkWeights();
check('sinkguard: weights sum to 1, w0 = 0.35, tail 0.075 each', near(sw.reduce((a, b) => a + b, 0), 1, 1e-9) && sw[0] === 0.35 && sw[31] === 0.075);
check('sinkguard: weighted error 0.25 → 0.0875', near(calc.sinkguard(sw, [], 0.25).weighted, 0.25, 1e-9) && near(calc.sinkguard(sw, [0, 28, 29, 30, 31], 0.25).weighted, 0.0875, 1e-9), calc.sinkguard(sw, [0, 28, 29, 30, 31], 0.25).weighted.toFixed(4));

const sp = calc.spotlight(calc.spotKeys(), [1, 0.4], 4);
check('spotlight: exact top-4 carries ~0.9 of the mass', sp.massExact > 0.8, sp.massExact.toFixed(3));
check('spotlight: proxy top-4 overlaps exact top-4 on ≥ 3 keys', sp.overlap >= 3, sp.overlap + ' of 4; proxy mass ' + sp.massProxy.toFixed(3));

check('basis: 1024² bf16 = 2 MiB, 8192² = 128 MiB', calc.basisBytes(1024, 80).shared === 2 * 2 ** 20 && calc.basisBytes(8192, 80).shared === 128 * 2 ** 20);
const td = calc.tiered(4096, 1);
check('tiered: B=4096 → 13.4% miss, ≈14.0 MB/token, ≈0.47 GB/s per request', near(td.miss, 0.134) && near(td.perToken / 1e6, 14.04, 0.01) && near(td.bandwidth / 1e9, 0.468, 0.001), (td.perToken / 1e6).toFixed(2) + ' MB, ' + (td.bandwidth / 1e9).toFixed(3) + ' GB/s');
check('tiered: bandwidth scales with concurrency; 16 × B=2048 ≈ 16.8 GB/s (26% of PCIe 5 nominal)', near(calc.tiered(2048, 16).bandwidth, 16 * calc.tiered(2048, 1).bandwidth) && near(calc.tiered(2048, 16).bandwidth / 1e9, 16.77, 0.01), (calc.tiered(2048, 16).bandwidth / 1e9).toFixed(2) + ' GB/s');

console.log('\n' + passes + ' passed, ' + failures + ' failed');
process.exit(failures ? 1 : 0);
