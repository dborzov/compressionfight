#!/usr/bin/env node
// Data sheet self-test: node test/datasheet_test.js  (pure calc.* only, no DOM)
'use strict';
const path = require('path');
global.window = global;
require(path.join(__dirname, '..', 'js', 'data.js'));
require(path.join(__dirname, '..', 'js', 'engine.js'));
require(path.join(__dirname, '..', 'js', 'datasheet.js'));

const { CYC } = global;
const E = CYC.Engine, C = CYC.DataSheet.calc;
let failures = 0, checks = 0;
function check(cond, msg) { checks++; if (!cond) { failures++; console.error('  FAIL: ' + msg); } }
const near = (a, b, eps) => Math.abs(a - b) <= (eps || 1e-9);
const bytesOf = (s) => Uint8Array.from(Array.from(s).map((c) => c.charCodeAt(0)));

// ---------- entropy
{
  const u = C.entropy([0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3]);
  check(near(u.H, 2) && u.alphabet === 4, 'entropy of uniform 4-symbol array is 2 bits, alphabet 4');
  const k = C.entropy([7, 7, 7, 7, 7]);
  check(k.H === 0 && k.alphabet === 1, 'entropy of a constant array is 0');
}

// ---------- diffs
{
  const ramp = Array.from({ length: 10 }, (_, i) => i);
  const d1 = C.diffs(ramp, 1), d2 = C.diffs(ramp, 2);
  check(d1[0] === 0 && Array.from(d1).slice(1).every((v) => v === 1), 'order-1 residual of a ramp is 1 after index 0');
  check(Array.from(d2).slice(2).every((v) => v === 0), 'order-2 residual of a ramp is 0 after index 1');
}

// ---------- rowDiffs: 3x2 plane, rows [1,2,4] and [5,5,9]
{
  const plane = { w: 3, h: 2, data: Float64Array.from([1, 2, 4, 5, 5, 9]) };
  const up = Array.from(C.rowDiffs(plane, 'up')), sub = Array.from(C.rowDiffs(plane, 'sub'));
  check(JSON.stringify(up) === JSON.stringify([1, 2, 4, 4, 3, 5]), 'rowDiffs up: ' + up);
  check(JSON.stringify(sub) === JSON.stringify([1, 1, 2, 5, 0, 4]), 'rowDiffs sub: ' + sub);
}

// ---------- repeats
{
  const r = C.repeats(bytesOf('abcabcabc'));
  check(r.coverage > 0.5, 'repeats coverage on abcabcabc > 0.5 (' + r.coverage + ')');
  check(r.matches.some((m) => m.dist === 3), 'repeats finds a match at distance 3');
  const z = C.repeats(bytesOf('abcdefghi'));
  check(z.coverage === 0 && z.matches.length === 0, 'repeats on 9 distinct bytes: coverage 0, no matches');
}

// ---------- literals
{
  const got = C.pyBytes(Uint8Array.from([65, 10, 39, 92, 0, 200]));
  const want = "b'A\\n\\'\\\\\\x00\\xc8'";
  check(got === want, `pyBytes: got ${got}, want ${want}`);
  check(C.pyList([1, 2, 3], 2) === '[1, 2, …]' && C.pyList([1, 2, 3], 2).endsWith('…]'), 'pyList truncates with …]');
  check(C.pyList([1, 2, 3]) === '[1, 2, 3]', 'pyList without truncation');
}

// ---------- profiles
{
  const j = C.profile(E.payload('json'));
  check(j.byteAlphabet >= 30 && j.byteAlphabet <= 120, 'json byteAlphabet in 30..120 (' + j.byteAlphabet + ')');
  check(j.byteEntropy > 3 && j.byteEntropy < 7, 'json byteEntropy in 3..7 (' + j.byteEntropy + ')');
  check(j.diffEntropy > j.byteEntropy, 'json: neighbour differences widen the entropy');
  check(j.lz.coverage > 0.5, 'json LZ coverage > 0.5 (' + j.lz.coverage + ')');
  check(j.lz.top.length > 0 && j.lz.top[0].len >= 20, 'json longest repeat >= 20 bytes');

  const s = C.profile(E.payload('screenshot'));
  check(s.zeroUp > 0.4, 'screenshot zeroUp > 0.4 (' + s.zeroUp + ')');
  check(s.zeroSub > 0.4, 'screenshot zeroSub > 0.4 (' + s.zeroSub + ')');
  check(s.lz.coverage > 0.8, 'screenshot LZ coverage > 0.8 (' + s.lz.coverage + ')');

  const im = C.profile(E.payload('image'));
  check(im.lz.coverage < 0.4, 'photo LZ coverage < 0.4 (' + im.lz.coverage + ')');
  check(im.nearUp > im.zeroUp, 'photo: near-equal neighbours far outnumber exact ones');

  const a = C.profile(E.payload('audio'));
  check(a.distinct > 1000, 'audio distinct sample values > 1000 (' + a.distinct + ')');
  check(a.d2Entropy < a.d1Entropy && a.d1Entropy < a.sampleEntropy, `audio entropies: d2 ${a.d2Entropy} < d1 ${a.d1Entropy} < raw ${a.sampleEntropy}`);
  check(a.d2Within256 > 0.7, 'audio order-2 residuals mostly within +-256 (' + a.d2Within256 + ')');
  check(a.lz.coverage < 0.1, 'audio LZ coverage < 0.1 (' + a.lz.coverage + ')');
}

// ---------- every payload kind: facts and prose
{
  for (const kind of E.PAYLOAD_KINDS) {
    let prof = null; try { prof = C.profile(E.payload(kind)); } catch (e) { check(false, `profile(${kind}) threw: ${e.message}`); continue; }
    check(Array.isArray(prof.facts) && prof.facts.length > 0, `${kind}: facts non-empty`);
    check(prof.facts.every((f) => typeof f.n === 'string' && typeof f.t === 'string' && f.n && f.t), `${kind}: every fact has string n and t`);
    check(!!(CYC.DATAKINDS && CYC.DATAKINDS[kind]), `${kind}: has an entry in CYC.DATAKINDS`);
  }
}

// ---------- codec x dataset matrix: the diagonal wins
{
  const ids = ['deflate', 'png', 'flac'];
  const pipelines = ids.map((id) => {
    const c = CYC.CODECS.find((x) => x.id === id); const hz = CYC.HEISTS.find((x) => x.id === c.heist);
    return { id, name: c.name, squad: c.squad, home: hz.payload };
  });
  const kinds = ['json', 'screenshot', 'audio'];
  check(pipelines.map((p) => p.home).join() === kinds.join(), 'codec homes are json, screenshot, audio: ' + pipelines.map((p) => p.home));
  const rows = C.crossMatrix(kinds, pipelines, E.run);
  check(rows.length === 3, 'crossMatrix has one row per kind');
  for (const r of rows) {
    check(r.cells.length === 3, `${r.kind}: one cell per pipeline`);
    check(r.cells.filter((c) => c.best).length === 1, `${r.kind}: exactly one best cell`);
    check(r.cells.every((c) => c.lossless === true), `${r.kind}: every cell lossless`);
    const home = r.cells.find((c) => c.home);
    check(home && home.best, `${r.kind}: home cell (${home && home.pipeline}) is the best; ratios ${r.cells.map((c) => c.pipeline + '=' + c.ratio.toFixed(2)).join(' ')}`);
  }
}

if (failures) { console.error(`${failures} of ${checks} checks FAILED`); process.exit(1); }
console.log(`${checks} checks; all checks passed`);
