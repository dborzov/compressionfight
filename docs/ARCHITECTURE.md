# Choose Your Codec — architecture contract

Static site, no build step. Open `index.html` directly (file://) or serve the folder.
Plain ES2020 JavaScript, no frameworks, no bundler. Everything hangs off `window.CYC`.

```
index.html            page shell, loads css + js in order
css/style.css         all styling (90s arcade fighting-game aesthetic)
js/data.js            ROSTER: roles, families, fighters, heists, codecs  (done, read it first)
js/engine.js          CYC.Engine — toy but REAL pipeline math on sample payloads
js/rules.js           CYC.Rules  — scoring: requirements of the job vs properties of the squad
js/viz.js             CYC.Viz    — canvas renderers for the generic viz descriptors
js/ui.js              CYC.UI     — screens, modes, levels, catalog, pipeline panel, results
img/fighters/<id>.png 256×256 portraits, one per fighter id in data.js (+ none.png)
img/heists/<id>.png   384×256 posters, one per heist id
```

Additional files (added after the first draft of this document):

```
js/curriculum.js      CYC.TUTOR (the Fixer's lines by trigger), CYC.LESSONS[level], CYC.QUIZZES[level],
                      CYC.TRADEOFF_AXES / CYC.TRADEOFF_PRESETS
js/quests_data.js     CYC.SIDEQUESTS — side-quest content (howto / watch / debrief)
js/quests.js          CYC.Quests.render(container, questId, {onComplete}) — side-quest mechanics (see docs/SIDEQUESTS.md)
js/demos.js           CYC.Demos.render(container, fighterId) — per-fighter Special Move labs (see docs/DEMOS.md)
```
data.js also carries game mechanics: `CYC.METAPHORS` (heist-metaphor labels for each requirement stat),
`CYC.COMBOS` / `CYC.ANTICOMBOS` (tag teams and friendly fire; `CYC.Rules.activeCombos(squad, heist)` and
`activeAnticombos(squad, heist, roles)` evaluate them), `CYC.DUELS` (head-to-head presets).

Script load order in index.html: data.js, curriculum.js, quests_data.js, engine.js, rules.js, demos.js, quests.js, viz.js, ui.js.

## Vocabulary

- **Role** (= slot = chair): `transform, predict, quantize, entropy, perceptual, prior`. Pipeline order is
  the key order of `CYC.ROLES`.
- **Fighter**: an algorithm. `CYC.FIGHTERS`, lookup `CYC.byId[id]`. `CYC.EMPTY` (id `none`) can sit in any chair.
- **Squad**: `{ transform: [ids], predict: [ids], quantize: [ids], entropy: [ids], perceptual: [ids], prior: [ids] }`.
  Each role holds 0..2 fighters (ordered). Missing key or empty array == empty chair. `['none']` == explicitly empty.
- **Heist**: an application with a requirements profile `req` and sample `payload` kind.
- **Level**: `CYC.LEVELS[i].roles` lists which chairs are visible; fighters with `level <= current` are in the roster.

## Engine contract (`js/engine.js`)

```js
CYC.Engine.run(payloadKind, squad) -> Result
```

`payloadKind` is one of `json, prose, image, audio, video, vectors, weights, activations` (the `payload` field of a heist).
The engine generates the sample payload deterministically (seeded), so repeated runs are identical.

Execution order (encoder direction), skipping empty chairs:

1. `prior`      — setup only: preload LZ window with a dictionary, mark codebooks/tables/weights as side info.
2. `transform`  — fighters in family `channel` (Prism) run here, before anything else.
3. `perceptual` — may reduce samples (chroma subsampling), or set thresholds the quantizer uses (masking).
4. `predict`    — residuals / LZ tokens / motion compensation.
5. `transform`  — remaining transform fighters (DCT, MDCT, DWT, PCA, BWT, learned encoder).
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

Engine must be real where it is cheap to be real: real LZ77 (hash-chain or brute force, window ≤ 4 KB), real
Huffman code lengths (tree build), real order-0/order-1 adaptive entropy estimate for arithmetic/ANS, real
Golomb-Rice bit counts with best k, real 8×8 DCT-II, real k-means (few iterations) for VQ/PQ, real block-wise
int4 with per-32 fp16 scale, real delta/LPC (order-2 fixed predictor is enough), real RLE. Approximations
are allowed and must be labelled in the caption: MDCT → DCT-IV on 32-sample frames; wavelet → one Haar level;
learned encoder → PCA on patches (stand-in); context mixing / neural prior → order-2 adaptive model;
motion compensation → block search between two synthesized frames; adversarial loss → no size effect, caption only.

## Rules contract (`js/rules.js`)

```js
CYC.Rules.score(heist, squad, result?) -> {
  score: number /* 0..100 */, rank: 'S'|'A'|'B'|'C'|'D',
  verdicts: { role: string, fighterId: string|null, ok: 'ideal'|'accept'|'meh'|'wrong', text: string }[],
  headline: string,
}
```

## UI contract (`js/ui.js`, `js/viz.js`)

`CYC.Viz.draw(canvas, viz)` draws any Viz descriptor into a canvas, sized to the canvas. Pixel look
(`image-rendering: pixelated`) where it fits.

The UI has two top-level modes and must keep state in the page (plus `localStorage` for progression, wrapped in try/catch):

**HEIST mode** — screens: level select → heist select (posters, requirement bars) → squad room
(chairs by role, click a chair → CHOOSE YOUR FIGHTER screen: nested catalog family → fighters, portraits,
stat bars, signature move, used-in, weakness; pick up to 2 per chair) → live pipeline panel updates on every change
(`Engine.run`) → RUN THE HEIST → results (rank letter, per-chair verdicts, size meter raw→bound→final, side-info meter,
distortion, then the `reveal` text naming the real codec). Rank B or better unlocks the next heist; finishing all
heists of a level unlocks the next level. An "unlock everything" button exists.

**EXPLORE mode** — tabs: Codecs (every `CYC.CODECS` entry as a pre-filled squad; click to load it into the
same pipeline panel on its heist's payload), Roster (every fighter as a character card, grouped by role → family,
with the level badge), Job board (all heists' requirement profiles side by side), and Matrix (fighters × requirement
dimensions: which properties satisfy which requirements, derived from stats).

Portrait fallback: if `img/fighters/<id>.png` fails to load, show a colored tile with the fighter's initials in
the role color. Same for heist posters.
