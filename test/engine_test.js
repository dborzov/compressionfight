#!/usr/bin/env node
// Engine self-test: node test/engine_test.js
'use strict';
const path = require('path');
global.window = global;
require(path.join(__dirname, '..', 'js', 'data.js'));
require(path.join(__dirname, '..', 'js', 'engine.js'));

const { CYC } = global;
const E = CYC.Engine, I = E._internals;
let failures = 0;
function check(cond, msg) { if (!cond) { failures++; console.error('  FAIL: ' + msg); } }
const bytesEq = (a, b) => a.length === b.length && a.every((v, i) => v === b[i]);

// ---------- primitives: byte-exact round trips
{
  for (const kind of ['json', 'prose', 'image', 'screenshot', 'audio', 'weights']) {
    const p = I.getPayload(kind); const ser = I.serializePlanes(p.planes); const bytes = ser.bytes;
    for (const dict of [null, I.dictionaryBytes(kind)]) {
      const tokens = I.lz77(bytes, dict, 4096, 3, 258);
      check(bytesEq(I.lz77Decode(tokens, dict), bytes), `LZ77 round trip (${kind}, dict=${!!dict})`);
    }
    const z = I.lzw(bytes, 12); check(bytesEq(I.lzwDecode(z.codes, 12), bytes), `LZW round trip (${kind})`);
    if (kind === 'json' || kind === 'prose') {
      const { last, primary } = I.bwt(bytes); const mtf = I.mtfEncode(last);
      check(bytesEq(I.bwtInverse(I.mtfDecode(mtf), primary), bytes), `BWT+MTF round trip (${kind})`);
    }
  }
  // Huffman: Kraft equality and optimality sanity on a skewed histogram
  const hist = new Map([[0, 50], [1, 25], [2, 12], [3, 13]]); const len = I.huffmanLengths(hist);
  let kraft = 0; for (const l of len.values()) kraft += Math.pow(2, -l);
  check(Math.abs(kraft - 1) < 1e-9, 'Huffman code is complete (Kraft sum = 1)');
  check(len.get(0) === 1 && len.get(1) === 2, 'Huffman gives the shortest codes to the most frequent symbols');
  // KT adaptive coder never beats the empirical entropy
  const syms = Int32Array.from({ length: 2000 }, (_, i) => (i * 7919) % 5 === 0 ? 1 : 0);
  const H = I.entropyOf(syms).H * syms.length;
  check(I.adaptiveOrder0Bits(syms, 2) >= H, 'adaptive order-0 code length >= n*H');
  // fp16 round trip
  for (const v of [0, 1, -1, 0.02, 65504, 1e-5, -0.3]) check(Math.abs(I.halfBitsToFloat(I.toHalfBits(v)) - v) <= Math.abs(v) * 1e-3 + 1e-7, `fp16 round trip ${v}`);
}

// ---------- every codec squad on its heist payload
const rows = [];
// screenshot payload + public accessor
{
  const ratio = (kind, pl) => { const r = E.run(kind, pl); return r.rawBits / r.finalBits; };
  const PNG = { predict: ['spatial', 'lz77'], entropy: ['huffman'] }, FLAC = { transform: ['prism'], predict: ['lpc'], entropy: ['rice'] }, DEFLATE = { predict: ['lz77'], entropy: ['huffman'] };
  const pl = E.payload('screenshot');
  check(E.PAYLOAD_KINDS.includes('screenshot'), 'PAYLOAD_KINDS includes screenshot');
  check(pl.bytes.length === 3072 && pl.bytesWidth === 1 && pl.domain === 'image' && pl.planes.length === 3 && pl.text === null, 'payload(screenshot): 3072 one-byte samples, 3 planes');
  pl.planes[0].data[0] = -1; check(E.payload('screenshot').planes[0].data[0] !== -1, 'payload() returns copies (cache not corrupted)');
  check(typeof E.payload('json').text === 'string' && E.payload('json').label === E.payloadLabel('json'), 'payload(json): text and label');
  check(ratio('screenshot', PNG) > ratio('screenshot', FLAC), 'PNG beats FLAC on screenshot');
  check(ratio('screenshot', PNG) > ratio('screenshot', DEFLATE), 'PNG beats DEFLATE on screenshot');
  check(ratio('screenshot', PNG) > ratio('screenshot', { predict: ['lpc'], entropy: ['huffman'] }), 'PNG beats LPC+Huffman on screenshot');
  check(ratio('screenshot', DEFLATE) >= 2, 'DEFLATE alone gets >= 2x on screenshot');
  check(ratio('audio', FLAC) > ratio('audio', PNG), 'FLAC beats PNG on audio');
  check(ratio('json', DEFLATE) > ratio('json', FLAC), 'DEFLATE beats FLAC on json');
}

const kinds = E.PAYLOAD_KINDS;
for (const codec of CYC.CODECS) {
  const heist = CYC.HEISTS.find((h) => h.id === codec.heist);
  let r; const t0 = process.hrtime.bigint();
  try { r = E.run(heist.payload, codec.squad); } catch (e) { failures++; console.error(`  FAIL: ${codec.id} threw: ${e.stack}`); continue; }
  const ms = Number(process.hrtime.bigint() - t0) / 1e6;
  check(r.finalBits > 0, `${codec.id}: finalBits > 0`);
  check(r.rawBits > 0 && r.entropyBoundBits > 0, `${codec.id}: raw/bound > 0`);
  check(Array.isArray(r.stages) && r.stages.length > 0, `${codec.id}: has stages`);
  for (const st of r.stages) {
    check(typeof st.caption === 'string' && /\d/.test(st.caption), `${codec.id}/${st.fighterId}: caption has numbers`);
    check(st.viz && typeof st.viz.type === 'string', `${codec.id}/${st.fighterId}: viz present`);
    const arrs = [];
    if (st.viz.type === 'grid') st.viz.planes.forEach((p) => arrs.push(p.values));
    if (st.viz.type === 'heatmap') arrs.push(st.viz.values);
    if (st.viz.type === 'waveform') st.viz.series.forEach((s) => arrs.push(s));
    if (st.viz.type === 'scatter') arrs.push(st.viz.points);
    for (const a of arrs) check(a.length <= 4096, `${codec.id}/${st.fighterId}: viz array ${a.length} <= 4096`);
    if (st.viz.type === 'histogram') check(st.viz.bins.length === st.viz.labels.length && st.viz.bins.length <= 64, `${codec.id}/${st.fighterId}: histogram shape`);
    if (st.viz.type === 'text') check(st.viz.spans.every((s) => ['lit', 'match', 'dict', 'run'].includes(s.cls)), `${codec.id}/${st.fighterId}: span classes`);
  }
  const ent = codec.squad.entropy || [];
  const usesCtx = (codec.squad.predict || []).some((id) => id === 'cm' || id === 'neural_prior');
  if (r.lossless && !usesCtx && ent.some((id) => id === 'arith' || id === 'ans')) check(r.finalBits >= 0.9 * r.entropyBoundBits, `${codec.id}: arith/ans lossless final ${r.finalBits} >= 0.9×bound ${r.entropyBoundBits}`);
  if (r.lossless) check(r.distortion === null, `${codec.id}: lossless => distortion null`);
  else check(r.distortion && typeof r.distortion.value === 'number', `${codec.id}: lossy => distortion present`);
  const hasQ = (codec.squad.quantize || []).length > 0 || (codec.squad.perceptual || []).includes('chroma') || (codec.squad.transform || []).some((id) => ['neural_enc', 'svd_lowrank', 'mla'].includes(id));
  check(r.backingCopy === !!(codec.squad.prior || []).includes('tiered'), `${codec.id}: backingCopy flag`);
  check(r.lossless === !hasQ, `${codec.id}: lossless flag ${r.lossless} matches squad (quantizer/chroma/neural_enc present: ${hasQ})`);
  check(ms < 50, `${codec.id}: run time ${ms.toFixed(1)} ms < 50 ms`);
  const r2 = E.run(heist.payload, codec.squad);
  check(r2.finalBits === r.finalBits && r2.entropyBoundBits === r.entropyBoundBits, `${codec.id}: deterministic`);
  rows.push({ codec: codec.id, payload: heist.payload, raw: r.rawBits, bound: r.entropyBoundBits, final: r.finalBits, ratio: (r.rawBits / r.finalBits).toFixed(2), side: r.sideInfoBytes, lossless: r.lossless, distortion: r.distortion ? r.distortion.label : '-', ms: ms.toFixed(1), warnings: r.warnings.length });
}
console.log('\ncodec            payload      rawBits   boundBits   finalBits   ratio   sideInfo     lossless  distortion                          ms   warn');
for (const r of rows) console.log(`${r.codec.padEnd(16)} ${r.payload.padEnd(11)} ${String(r.raw).padStart(8)} ${String(r.bound).padStart(11)} ${String(r.final).padStart(11)} ${String(r.ratio).padStart(7)}   ${String(r.side).padStart(9)}   ${String(r.lossless).padEnd(8)}  ${r.distortion.padEnd(34)} ${r.ms.padStart(5)}   ${r.warnings}`);

// ---------- level 5: the KV-cache stories must hold on the activations payload
{
  const err = (squad) => E.run('activations', squad).distortion.value;
  const bits = (squad) => E.run('activations', squad).finalBits;
  const b0 = err({ quantize: ['blockint'] }), bU = err({ transform: ['unrope'], quantize: ['blockint'] }), bH = err({ transform: ['pca'], quantize: ['blockint'] });
  check(bits({ quantize: ['blockint'] }) === bits({ transform: ['unrope'], quantize: ['blockint'] }), 'unrope changes no bits');
  check(bU < b0, `unrope + blockint (${(bU * 100).toFixed(2)}%) beats blockint alone (${(b0 * 100).toFixed(2)}%)`);
  check(bH < b0, `pca (Hadamard) + blockint (${(bH * 100).toFixed(2)}%) beats blockint alone (${(b0 * 100).toFixed(2)}%)`);
  const c0 = err({ quantize: ['coupled'], prior: ['codebook'] }), cU = err({ transform: ['unrope'], quantize: ['coupled'], prior: ['codebook'] });
  check(cU < c0, `unrope + coupled (${(cU * 100).toFixed(2)}%) beats coupled alone (${(c0 * 100).toFixed(2)}%)`);
  const rW = E.run('activations', { quantize: ['waterline'] }); check(rW.warnings.some((w) => /flat ground/.test(w)), 'waterline on raw channels warns');
  const rK = E.run('activations', CYC.CODECS.find((c) => c.id === 'kvtc').squad); check(!rK.warnings.some((w) => /flat ground/.test(w)), 'waterline after unrope+pca+basis does not warn');
  const rS = E.run('activations', { perceptual: ['spotlight'], quantize: ['evict'] }); check(/attention-weighted/.test(rS.distortion.label), 'spotlight makes distortion attention-weighted');
  const rE = E.run('activations', { quantize: ['evict'] }); check(rE.finalBits < 0.6 * rE.rawBits, 'evict halves the stream');
  const rT = E.run('activations', { prior: ['tiered'] }); check(rT.backingCopy === true && rT.sideInfoBytes === rT.rawBits / 8, 'tiered: backing copy flag + fp16 copy as side info');
  const rA = E.run('activations', { predict: ['anchor_delta'], entropy: ['arith'] }); check(rA.lossless && rA.finalBits > 0, 'anchor_delta is lossless');
  console.log(`\nlevel-5 checks: blockint ${(b0 * 100).toFixed(2)}% | +unrope ${(bU * 100).toFixed(2)}% | +hadamard ${(bH * 100).toFixed(2)}% ; coupled ${(c0 * 100).toFixed(2)}% | +unrope ${(cU * 100).toFixed(2)}%`);
  for (const sq of [{ transform: ['unrope'], quantize: ['blockint'], perceptual: ['sinkguard'] }, { transform: ['pca'], quantize: ['blockint'], perceptual: ['sinkguard'] }, { quantize: ['fp4'] }, { quantize: ['escape'] }, { transform: ['unrope'], quantize: ['escape'] }, { transform: ['svd_lowrank'] }, { transform: ['unrope', 'svd_lowrank'] }, { transform: ['mla'] }])
    { const r = E.run('activations', sq); console.log(`  ${JSON.stringify(sq).padEnd(78)} final=${String(r.finalBits).padStart(6)} ${r.distortion ? r.distortion.label : 'lossless'}`); }
}

// ---------- every fighter in every role on every payload must not throw
const roles = Object.keys(CYC.ROLES); const ids = CYC.FIGHTERS.map((f) => f.id).concat(['none']);
let runs = 0; const tAll = process.hrtime.bigint();
for (const kind of kinds) for (const role of roles) for (const id of ids) {
  const squad = { [role]: [id] };
  try { const r = E.run(kind, squad); runs++; check(r.finalBits > 0 && Number.isFinite(r.finalBits), `${kind}/${role}/${id}: finalBits finite > 0`); check(Number.isFinite(r.entropyBoundBits), `${kind}/${role}/${id}: bound finite`); }
  catch (e) { failures++; console.error(`  FAIL: ${kind}/${role}/${id} threw: ${e.stack.split('\n').slice(0, 3).join(' | ')}`); }
}
// each fighter in its own role, paired with each entropy coder, on each payload
for (const kind of kinds) for (const f of CYC.FIGHTERS) for (const ent of ['none', 'huffman', 'arith', 'ans', 'rice', 'rle']) {
  const squad = { [f.role]: [f.id], entropy: f.role === 'entropy' ? [f.id, ent] : [ent] };
  try { const r = E.run(kind, squad); runs++; check(r.finalBits > 0, `${kind}/${f.id}+${ent}: finalBits > 0`); if (r.lossless && !['cm', 'neural_prior'].includes(f.id) && (ent === 'arith' || ent === 'ans')) check(r.finalBits >= 0.9 * r.entropyBoundBits, `${kind}/${f.id}+${ent}: final ${r.finalBits} >= 0.9×bound ${r.entropyBoundBits}`); for (const st of r.stages) check(/\d/.test(st.caption), `${kind}/${f.id}+${ent}/${st.fighterId}: caption has numbers`); }
  catch (e) { failures++; console.error(`  FAIL: ${kind}/${f.id}+${ent} threw: ${e.stack.split('\n').slice(0, 3).join(' | ')}`); }
}
// the full heist 'ideal' squads, every pair of level-5 fighters with every quantizer, and every duel
for (const h of CYC.HEISTS) { try { const r = E.run(h.payload, h.ideal); runs++; check(r.finalBits > 0, `${h.id} ideal squad`); } catch (e) { failures++; console.error(`  FAIL: heist ${h.id} ideal threw: ${e.stack}`); } }
const l5 = CYC.FIGHTERS.filter((f) => f.level >= 5);
for (const a of l5) for (const b of CYC.FIGHTERS) {
  if (a === b) continue; const squad = {}; squad[a.role] = [a.id]; squad[b.role] = (squad[b.role] || []).concat([b.id]);
  for (const kind of ['activations', 'vectors', 'audio']) { try { const r = E.run(kind, squad); runs++; check(Number.isFinite(r.finalBits) && r.finalBits > 0, `${kind}/${a.id}+${b.id}: finalBits`); if (r.distortion) check(Number.isFinite(r.distortion.value) || r.distortion.value === Infinity, `${kind}/${a.id}+${b.id}: distortion finite`); } catch (e) { failures++; console.error(`  FAIL: ${kind}/${a.id}+${b.id} threw: ${e.stack.split('\n').slice(0, 3).join(' | ')}`); } }
}
for (const d of (CYC.DUELS || [])) for (const id of [d.a, d.b]) { const squad = JSON.parse(JSON.stringify(d.squadBase)); const role = d.role || (CYC.byId[id] && CYC.byId[id].role); if (!role) continue; squad[role] = (squad[role] || []).concat([id]); try { E.run(d.payload, squad); runs++; } catch (e) { failures++; console.error(`  FAIL: duel ${d.a} vs ${d.b} (${id}) threw: ${e.stack.split('\n').slice(0, 3).join(' | ')}`); } }
const totalMs = Number(process.hrtime.bigint() - tAll) / 1e6;
console.log(`\n${runs} extra runs in ${totalMs.toFixed(0)} ms (${(totalMs / runs).toFixed(2)} ms/run)`);
for (const k of kinds) console.log(`  ${k.padEnd(12)} ${E.payloadLabel(k)}`);

if (failures) { console.error(`\n${failures} check(s) failed`); process.exit(1); }
console.log('\nall checks passed');
