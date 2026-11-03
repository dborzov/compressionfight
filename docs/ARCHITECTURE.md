# Choose Your Codec — architecture contract

Static site, no build step. Open `index.html` directly (file://) or serve the folder.
Plain ES2020 JavaScript, no frameworks, no bundler. Everything hangs off `window.CYC`.

```
index.html            page shell, loads css + js in order
css/style.css         all styling (90s arcade fighting-game aesthetic)
js/data.js            ROSTER: roles, families, algorithms, tasks, codecs  (done, read it first)
js/engine.js          CYC.Engine — toy but REAL pipeline math on sample payloads
js/rules.js           CYC.Rules  — scoring: requirements of the job vs properties of the pipeline
js/datasheet.js       CYC.DataSheet — the sample data printed as the bytes it is, measured, and the codec × dataset matrix
js/viz.js             CYC.Viz    — canvas renderers for the generic viz descriptors
js/ui.js              CYC.UI     — screens, modes, levels, catalog, pipeline panel, results
img/fighters/<id>.png 256×256 portraits, one per algorithm id in data.js (+ none.png)
img/heists/<id>.png   384×256 posters, one per task id
```

Additional files (added after the first draft of this document):

```
js/curriculum.js      CYC.TUTOR (the tutor's lines by trigger), CYC.LESSONS[level], CYC.QUIZZES[level],
                      CYC.TRADEOFF_AXES / CYC.TRADEOFF_PRESETS
js/quests_data.js     CYC.SIDEQUESTS — exercise content (howto / watch / debrief)
js/quests.js          CYC.Quests.render(container, questId, {onComplete}) — exercise mechanics (see docs/SIDEQUESTS.md)
js/demos.js           CYC.Demos.render(container, fighterId) — per-algorithm labs (see docs/DEMOS.md)
js/datasheet.js       CYC.DataSheet — data sheet, mini preview, codec × dataset matrix (see DataSheet contract below)
```
data.js also carries game mechanics: `CYC.METAPHORS` (task-metaphor labels for each requirement stat),
`CYC.COMBOS` / `CYC.ANTICOMBOS` (synergies and conflicts; `CYC.Rules.activeCombos(squad, heist)` and
`activeAnticombos(squad, heist, roles)` evaluate them), `CYC.DUELS` (head-to-head presets).

Script load order in index.html: data.js, curriculum.js, quests_data.js, engine.js, rules.js, datasheet.js, demos.js, quests.js, viz.js, ui.js.

## Vocabulary

- **Stage** (= slot = role): `transform, predict, quantize, entropy, perceptual, prior`. Pipeline order is
  the key order of `CYC.ROLES`.
- **Algorithm**: one compression technique. `CYC.FIGHTERS`, lookup `CYC.byId[id]`. `CYC.EMPTY` (id `none`) can sit in any stage.
- **Pipeline**: `{ transform: [ids], predict: [ids], quantize: [ids], entropy: [ids], perceptual: [ids], prior: [ids] }`.
  Each role holds 0..2 algorithms (ordered). Missing key or empty array == pass-through. `['none']` == explicitly empty.
- **Task**: an application with a requirements profile `req` and sample `payload` kind.
- **Level**: `CYC.LEVELS[i].roles` lists which stages are visible; algorithms with `level <= current` are in the roster.

## Engine contract (`js/engine.js`)

```js
CYC.Engine.run(payloadKind, pipeline) -> Result
```

`payloadKind` is one of `json, prose, image, screenshot, audio, video, vectors, weights, activations` (the `payload` field of a task).
The engine generates the sample payload deterministically (seeded), so repeated runs are identical.

Execution order (encoder direction), skipping empty stages:

1. `prior`      — setup only: preload LZ window with a dictionary, mark codebooks/tables/weights as side info.
2. `transform`  — algorithms in family `channel` (Prism) run here, before anything else.
3. `perceptual` — may reduce samples (chroma subsampling), or set thresholds the quantizer uses (masking).
4. `predict`    — residuals / LZ tokens / motion compensation.
5. `transform`  — remaining transform algorithms (DCT, MDCT, DWT, PCA, BWT, learned encoder).
6. `quantize`   — the only lossy stage.
7. `entropy`    — the only stage that produces actual bits. Chain allowed: `rle` then `huffman`.

```ts
Result = {
  payloadKind: string,
  rawBits: number,              // n_samples × original bit depth (text 8/char, audio 16, image 8/channel, vectors 32, weights/activations 16)
  finalBits: number,            // honest bit count of the output stream incl. estimated header/table overhead
  entropyBoundBits: number,     // order-0 Shannon entropy of the symbol stream that reached the entropy stage × n (what a perfect coder would pay)
  sideInfoBytes: number,        // dictionary / codebook / tables / decoder weights the decoder must already hold (NOT counted in finalBits)
  lossless: boolean,            // true iff no quantizer ran (and perceptual stage removed nothing)
  distortion: null | { metric: 'PSNR'|'SNR'|'MSE'|'rel-err', value: number, label: string },
  stages: Stage[],
  warnings: string[],           // e.g. "Huffman on a symbol with p=0.82: paying 1 bit for 0.29 bits of information"
}
Stage = {
  role: string, fighterId: string, title: string,
  caption: string,              // one or two sentences WITH NUMBERS: what this stage did to this data
  metrics: { symbols?: number, alphabet?: number, entropyBitsPerSymbol?: number, bits?: number, sideInfoBytes?: number, distortion?: Result['distortion'] },
  viz: Viz,                     // see below
}
```

Viz descriptors (engine emits, viz.js draws; keep arrays small, ≤ 4096 values):

```ts
{ type: 'text',      spans: { text: string, cls: 'lit'|'match'|'dict'|'run', title?: string }[] }
{ type: 'waveform',  series: Float32Array[] /* each normalized to -1..1 */, labels: string[] }
{ type: 'grid',      planes: { name: string, w: number, h: number, values: Float32Array /* 0..1 gray */ }[], rgb?: { w, h, data: Uint8ClampedArray } }
{ type: 'heatmap',   w: number, h: number, values: Float32Array /* 0..1, log-scaled magnitude */, zeroFrac: number, blockSize?: number }
{ type: 'histogram', bins: number[], labels: string[], entropyBitsPerSymbol: number, fixedBits: number }
{ type: 'scatter',   points: Float32Array /* xy pairs, -1..1 */, centroids?: Float32Array, assignments?: Uint16Array }
{ type: 'bits',      raw: number, out: number, bound: number, sideInfoBytes: number }
```

Also export for the UI: `CYC.Engine.payloadLabel(kind)` → human label (e.g. "8 KB of JSON events").

`screenshot` is a 32×32 RGB payload, noise-free by construction: flat fills, a repeated 6×6 icon, rows of text, a
gradient button (the PNG task's data; `image` stays the noisy photograph).

`CYC.Engine.payload(kind)` → a copy of the sample the engine runs on (mutating it does not corrupt the cache), so the
UI can print the data without re-deriving it:

```js
{ kind, domain: 'text'|'image'|'audio'|'video'|'floats', bitDepth, label, n /* values */, rawBits,
  planes: [{ name, w, h, data /* typed array */ }], text: string|null /* text payloads only */,
  bytes: Uint8Array /* the serialized samples the byte-oriented stages see */, bytesWidth: number /* bytes per sample */ }
```

Engine must be real where it is cheap to be real: real LZ77 (hash-chain or brute force, window ≤ 4 KB), real
Huffman code lengths (tree build), real order-0/order-1 adaptive entropy estimate for arithmetic/ANS, real
Golomb-Rice bit counts with best k, real 8×8 DCT-II, real k-means (few iterations) for VQ/PQ, real block-wise
int4 with per-32 fp16 scale, real delta/LPC (order-2 fixed predictor is enough), real RLE. Approximations
are allowed and must be labelled in the caption: MDCT → DCT-IV on 32-sample frames; wavelet → one Haar level;
learned encoder → PCA on patches (stand-in); context mixing / neural prior → order-2 adaptive model;
motion compensation → block search between two synthesized frames; adversarial loss → no size effect, caption only.

## DataSheet contract (`js/datasheet.js`)

```js
CYC.DataSheet.render(container, kind, opts)   // full sheet: sample printed as what it is, picture, measured facts, what fits / fails
CYC.DataSheet.renderMini(container, kind)     // two-line preview (literal + one measured fact) for task cards
CYC.DataSheet.renderCross(container, { codecs: [codecId] | kinds + pipelines: [{id,name,squad,home}], onPick(kind, pipelineId), caption })
CYC.DataSheet.profile(kind)                   // measured facts for a payload kind (cached): { byteEntropy, lz:{coverage,top,…}, facts:[{n,t,bad?}], … }
CYC.DataSheet.calc.*                          // pure, DOM-free, tested in node: histogram, entropy, diffs, rowDiffs, shareWithin,
                                              // distinct, repeats, pyChar, pyBytes, pyList, hexRow, profile, crossMatrix
```

Printing: text as a Python-style `bytes` literal, images as pixel rows with the PNG "Up"/"Sub" filter residuals, audio as
int16 samples with the FLAC order-2 residual. Rule: every number shown is computed from the sample by the stated formula
(Shannon entropy, a greedy LZ77 pass for exact-repeat coverage, neighbour-residual statistics, live `CYC.Engine.run` calls
for the matrix); nothing is hard-coded. The prose (structure, what fits, what does not, real-world use) comes from
`CYC.DATAKINDS[kind]` in `js/data.js`, one entry per payload kind. `renderCross` runs every listed codec's squad on every
listed dataset and marks the best cell per row and each codec's home dataset; with `onPick` the cells are buttons.

Lessons in `js/curriculum.js` may carry `datasheets: [kinds]` (data sheets embedded in the lesson) and
`crossmatrix: { codecs: [codecIds] }` (a codec × dataset matrix embedded in the lesson).

## Rules contract (`js/rules.js`)

```js
CYC.Rules.score(task, pipeline, result?) -> {
  score: number /* 0..100 */, rank: 'S'|'A'|'B'|'C'|'D',
  verdicts: { role: string, fighterId: string|null, ok: 'ideal'|'accept'|'meh'|'wrong', text: string }[],
  headline: string,
}
```

## UI contract (`js/ui.js`, `js/viz.js`)

`CYC.Viz.draw(canvas, viz)` draws any Viz descriptor into a canvas, sized to the canvas. Pixel look
(`image-rendering: pixelated`) where it fits.

The UI has two top-level modes and must keep state in the page (plus `localStorage` for progression, wrapped in try/catch):

**TASK mode** — screens: level select → task select (posters, requirement bars) → pipeline room
(stages by role, click a stage → CHOOSE YOUR ALGORITHM screen: nested catalog family → algorithms, portraits,
stat bars, key idea, used-in, weakness; pick up to 2 per stage) → live pipeline panel updates on every change
(`Engine.run`) → RUN THE TASK → results (rank letter, per-stage verdicts, size meter raw→bound→final, side-info meter,
distortion, then the `reveal` text naming the real codec). Rank B or better unlocks the next task; finishing all
tasks of a level unlocks the next level. An "unlock everything" button exists.
The task screen shows a data preview (`DataSheet.renderMini`) on each task card; the level screen shows the codec × dataset
matrix (`DataSheet.renderCross`) once every task of a level is lossless; the pipeline builder shows the full data sheet
(`DataSheet.render`) above the stages.

**EXPLORE mode** — tabs: Codecs (every `CYC.CODECS` entry as a pre-filled pipeline; click to load it into the
same pipeline panel on its task's payload), Roster (every algorithm as a character card, grouped by role → family,
with the level badge), Job board (all tasks' requirement profiles side by side), and Matrix (algorithms × requirement
dimensions: which properties satisfy which requirements, derived from stats).

Portrait fallback: if `img/fighters/<id>.png` fails to load, show a colored tile with the algorithm's initials in
the role color. Same for task posters.
