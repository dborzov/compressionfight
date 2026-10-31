# Special Move Training — per-fighter worked examples (`js/demos.js`)

Contract: `CYC.Demos.render(container: HTMLElement, fighterId: string)` renders one interactive mini-lab into the
container (clearing it first) and returns nothing. `CYC.Demos.has(fighterId) -> boolean`. Each lab is one
concrete, numeric, data-specific example: the user sees actual numbers go in and actual numbers come out,
optionally with one slider/toggle. No lab may lie: every displayed number is computed live from the stated
input by the stated formula. Keep each lab to roughly 180–360 px tall; it sits inside the fighter detail card.

Shared primitives to build once and reuse (plain DOM + canvas, styled via existing CSS variables; the UI
agent's stylesheet defines `--role-color` on the container, pixel fonts, dark theme):
- `numStrip(rows: {label, values: (number|string)[], hi?: boolean[]}[])` — aligned rows of numbers, highlighted cells.
- `matrix(values: number[][], opts {heat?: boolean, zeroDim?: boolean})` — small grid with the numbers written in.
- `map(canvas, {points, centroids?, assign?, arrows?, axes?})` — 2D scatter "city map", coordinates in -1..1.
- `tree(canvas, huffmanTree)` — binary tree with 0/1 edge labels and leaf symbols.
- `intervals(canvas, steps: {lo, hi, label}[])` — successive zoomed number lines.
- `curve(canvas, fn, domain, marks)` — function plot with marked points.
- `bars(canvas, values, labels, opts)`.
- `slider(label, min, max, step, value, onChange)` and `toggle(label, options, onChange)`.
- `note(text)` — one explanatory line under the lab, with the formula in plain text (no LaTeX library needed).

## Labs (34). Inputs are fixed; `⟲` marks the interactive control.

### transform
- **dct** — input row `[52,55,61,66,70,61,64,73]` (first row of the Wikipedia JPEG block). Show the 8 DCT-II coefficients (orthonormal). ⟲ slider "keep first k coefficients" (1..8): reconstruct with the rest zeroed, show reconstruction row and MSE. Note: energy compaction: `k=3` already gives small error.
- **mdct** — draw two consecutive sine windows over 2N samples overlapping by N, and their squared sum (should be flat = 1, Princen–Bradley). ⟲ slider frame size N ∈ {64,128,256,512,1024,1152} → show algorithmic delay = N/48000·1000 ms at 48 kHz (1152 → 24 ms). Note: the decoder cannot finish frame t until frame t+1 arrives.
- **dwt** — input `[10,12,14,16,100,98,96,94]`. One Haar level: averages `(a+b)/2 → [11,15,99,95]`, details `(a−b)/2 → [-1,-1,1,1]`. Then a second level on the averages: `[13,97]`, `[-2,2]`. Show as a pyramid; highlight that details are tiny except at the edge. ⟲ toggle levels 1/2/3.
- **prism** — four RGB pixels `(200,180,170) (205,182,172) (198,179,168) (202,181,171)` → Y, Cb, Cr using JPEG's `Y=0.299R+0.587G+0.114B`, `Cb=128+0.564(B−Y)`, `Cr=128+0.713(R−Y)`. Show that Y varies while Cb/Cr sit near 128 with tiny spread. Second row: stereo `L=1000, R=980 → M=(L+R)/2=990, S=L−R=20`. Note: the correlated part moves into one channel.
- **pca** — 40 2D points from `y = 0.8x + noise(σ=0.1)`, x uniform in -1..1 (seeded). Map with the principal axis drawn. ⟲ toggle "rotate to principal axes": show variances per axis before `(var x, var y)` and after `(λ1, λ2)`, and overlay a 2-bit uniform grid on both; show total quantization error before/after. Note: rotation moves nothing, but it lets an axis-aligned quantizer waste fewer levels.
- **bwt** — word `banana$`. Table of all 7 rotations, then sorted, with the last column highlighted → `annb$aa`, primary index. Note: same letters cluster because they share the context that follows them; MTF+RLE then eats the runs. ⟲ toggle a second word `abracadabra$`.
- **neural_enc** — schematic strip: 320 waveform samples (draw) → encoder box → one 8-float latent `[0.81,-1.20,0.07,2.31,-0.45,0.92,-1.77,0.33]` → "RVQ → 80 bits". Numbers: 320×16 = 5120 bits in, 80 bits out, 64:1 before any entropy coding. Note: stand-in numbers; the real latent is 512-dim at 75 Hz.

### predict
- **lz77** — text `the_cat_sat_on_the_mat_the_cat_sat` ; greedy LZ77, min match 3. Render spans: literals dim, matches highlighted with `(dist,len)` badges. ⟲ slider window size 4..32: count matches and bytes saved. Note: the longest match, `(23,11)`, is a prediction with certainty 1. (Greedy LZ77 with min match 3 finds `(4,3) (15,4) (11,3) (23,11)` on this text.)
- **lzw** — input `ABABABA`. Table of steps: w, next char, output code, dictionary add. Expected output codes `65 66 256 258` (4 codes for 7 symbols); dictionary gains `256=AB 257=BA 258=ABA`. Note: both sides build the same table, nothing extra travels.
- **delta** — `[100,102,105,107,108,108,107,105]` → residuals `[100,+2,+3,+2,+1,0,-1,-2]`. Show raw range (needs 7 bits) vs residual range after the first (needs 3 bits with sign). ⟲ toggle order 1 (delta) / order 2 (delta of delta: `[100,2,+1,-1,-1,-1,-1,-1]`).
- **lpc** — 16 samples of `x[n] = 100·0.9^n·cos(0.6n)` rounded to integers. Residuals for the fixed order-2 predictor `x̂ = 2x[n−1] − x[n−2]` and for the fitted 2-tap predictor (solve least squares for a1, a2 on these 16 samples; display a1, a2). Show residual energy (Σr²) for raw / fixed / fitted. Note: fitted coefficients are side info, sent once per block.
- **spatial** — 4×4 block `[[10,11,12,13],[20,21,22,23],[30,31,32,33],[40,41,42,43]]`. Residual grids for Sub (left), Up (above), Paeth. Sum of |residual| per filter; highlight the winner. Note: PNG picks per row; H.264 picks a direction per block.
- **motion** — frame 1: 8×8 with a bright 3×3 square at (2,2); frame 2: same square at (3,2) (shifted right by 1). Show both grids, the found motion vector `(+1, 0)` from a ±2 block search, and the residual grid (all zeros). ⟲ toggle "no motion compensation": residual = frame2 − frame1 (6 nonzero cells: the two squares overlap in 2 of 3 columns). Note: the vector costs a few bits; the residual costs nothing.
- **cm** — text `the_theme_of_the_thesis`. Order-2 context table: after `th` the counts of the next byte (`e`: 4, others 0 → with Laplace smoothing over 27 symbols P(e|th) = 5/31 ≈ 0.16 → 2.6 bits; ⟲ slider "more sightings" 0..300: after 10 more it is only 15/41 ≈ 0.37 → 1.45 bits; reaching 0.9 takes ~230 sightings. Laplace smoothing is slow to trust a context, which is why real CM uses adaptive-rate counters). Show the table for contexts `th`, `he`, `e_`. Note: the coder pays −log2 P; the model's job is to make P large.
- **neural_prior** — bar chart of a made-up next-token distribution for `The capital of France is`: Paris 0.90, Lyon 0.03, London 0.02, other 0.05. Under arithmetic coding the actual next token costs −log2 p: Paris 0.15 bits, London 5.6 bits. Fixed 8-bit code would cost 8 regardless. Note: a 7B LLM gets English to ~0.8 bits/char this way.

### quantize
- **uniform** — vector `[3.7, −1.2, 0.4, 8.9, −5.5, 2.1]`. ⟲ slider Δ 0.25..4. Rows: x, q = round(x/Δ) (round half away from zero, so −5.5/1 → −6), x̂ = q·Δ, error. Show bits per value = ceil(log2(number of distinct q levels in range ±9)), MSE, and the high-resolution theory Δ²/12 beside it. Note: doubling Δ saves ≈1 bit per value and quadruples the error.
- **qmatrix** — the JPEG luma table at quality 50 (standard Annex K values) as a heat matrix with numbers; the Wikipedia example DCT block coefficients (`[-415.38,-30.19,-61.20,27.24,56.13,-20.10,-2.39,0.46], [4.47,-21.86,-60.76,10.25,13.15,-7.09,-8.54,4.88], ...` use the full 8×8 from the Wikipedia JPEG article). Divide and round → quantized block; count zeros (44 of 64 at quality 50; DC = −26 as on Wikipedia). ⟲ slider quality scale 0.5×..3× of the table. Note: coarse where the eye is blind (bottom-right).
- **companding** — μ-law curve `F(x)=sgn(x)·ln(1+255|x|)/ln(256)` plotted. ⟲ slider input amplitude 0.001..1: 8-bit μ-law code, reconstruction, relative error; next to it, 8-bit uniform's relative error for the same input. Note: at amplitude 0.01 the worst-case error is 39% of the signal for 8-bit uniform (Δ/2 = 0.0039) vs ≈2.2% for μ-law; for x = 0.01 exactly the actual errors are 21.9% vs 0.10%. Show both actual and worst-case.
- **vq** — "city map": 60 seeded 2D points in 3 clusters. ⟲ slider K 2..8: run k-means (10 iterations, seeded init), draw safehouse centroids and assignments, show bits per point = log2 K, total squared distortion, codebook size K×2 floats. Note: codebook = side information.
- **rvq** — same points. Stage 1: K=4 centroids. Stage 2: k-means on the residuals, K=4; draw residual arrows for a few points. ⟲ toggle stages 1/2/3: show distortion and bits (2 bits per stage) and the effective number of reconstruction points (4, 16, 64) vs centroids stored (4, 8, 12).
- **pq** — 4-dim vector `[0.9, 0.1, −0.3, 0.8]` split into `[0.9,0.1]` and `[−0.3,0.8]`. Two small maps, each with a fixed 4-entry codebook (give the centroids), nearest index per half → code `(i, j)`, 2+2 bits, reconstruction and error. Note: 4×4 = 16 effective centroids from 8 stored; distance to a query is a sum of two table lookups.
- **blockint** — block of 8 values `[0.012, −0.031, 0.004, 0.550, 0.020, −0.008, 0.015, −0.027]` (one outlier). absmax scale `s = max|x|/7`, `q = round(x/s)` clamped to [−8,7], x̂ = q·s. Show that 7 of 8 become 0 or ±1 and the relative error of the small values. ⟲ toggle "rotate / isolate outlier": replace the outlier with a Hadamard-spread version (divide the 0.55 across all 8 as ±0.19 added to each) and redo; show the error drop. Bits: 8×16 = 128 → 8×4 + 16 = 48.

### entropy
- **huffman** — frequencies `A:45 B:13 C:12 D:16 E:9 F:5` (CLRS). Build the tree, draw it, list codes (expected lengths A1 B3 C3 D3 E4 F4, mean 2.24 bits/symbol vs H ≈ 2.22). ⟲ slider "skew": set P(A) from 0.45 to 0.95 (rescale others) → recompute: show mean length vs H; above 0.5 the gap grows (Huffman cannot go below 1 bit for A). Note: optimal among integer-length codes.
- **arith** — message `ABAC`, model P(A)=0.5 P(B)=0.25 P(C)=0.25 (fixed order). Draw the intervals: [0,1) → A [0,0.5) → B [0.25,0.375) → A [0.25,0.3125) → C [0.296875,0.3125). Final width 1/64 → 6 bits = Σ −log2 p = 1+2+1+2. ⟲ toggle a second message `AAAA` (width 1/16, 4 bits) to show cheap symbols. Note: fractional bits per symbol become possible because the whole message is one number.
- **ans** — rANS toy with M=4, freq A=3 B=1, cum A=0 B=3. Encode `A A B` from state x=4 (step table with `x' = (x ÷ f_s)·M + cum_s + (x mod f_s)`); then decode by popping: `s = symbol whose range contains x mod M`, `x = f_s·(x ÷ M) + (x mod M) − cum_s`. Show the state growing by ≈ −log2 p per symbol (log2 x increases by 0.415 for A, 2 for B). Note: LIFO: decoding runs backwards.
- **rice** — values `[3, 5, 2, 37, 1, 0, 8]`. ⟲ slider k 0..5: each value as unary(q=n>>k) + k low bits, total bits; highlight the best k (expected k=1 or 2 here; the 37 shows the long unary tail). Note: optimal for geometric distributions, which is what residuals look like.
- **rle** — 4×4 quantized coefficient block `[[12,3,0,0],[-2,0,0,0],[0,0,0,0],[0,0,0,0]]` → zigzag scan order drawn as a path → sequence `12,3,-2,0,0,...` → run/value pairs `(0,12) (0,3) (0,-2) EOB`. ⟲ toggle row-major scan vs zigzag to show why zigzag gives one tail instead of several runs.

### perceptual
- **psychoacoustic** — 24 spectral bands (bars), one loud tone at band 8 (level 80 dB), background around 30 dB. Draw the spreading/masking curve (triangle: −25 dB/band below, −10 dB/band above the masker, from the masker level −12 dB). Bands under the curve are "free" (quantization noise there is inaudible) — tint them. ⟲ slider masker level 50..90 dB: count free bands. Note: the model is encoder-only; the decoder never sees it.
- **chroma** — a 2×2 pixel RGB block (give values). Convert to Y Cb Cr; 4:2:0 keeps 4 Y and the mean Cb, mean Cr: 12 numbers → 6 numbers. Reconstruct RGB and show the per-pixel error (small). ⟲ toggle a second block with a sharp red/blue edge where the error is large. Note: 2:1 on raw samples, and the eye barely notices.
- **gan** — tiny waveform (32 samples of a two-tone burst). Two reconstructions: "MSE-trained" = low-pass smoothed version (plot, MSE value), "adversarial" = same waveform shifted by 2 samples (plot, higher MSE). Note: the sharp one has the worse MSE and the better sound; the discriminator judges plausibility, not alignment.

### prior
- **tables** — table excerpt of DEFLATE's fixed Huffman code (literals 0–143 → 8 bits starting 00110000, 144–255 → 9 bits, 256–279 → 7 bits, 280–287 → 8 bits) next to the cost of transmitting a dynamic table (typically 40–120 bytes). ⟲ slider payload size 100 B .. 10 KB: show when dynamic tables pay for themselves (break-even when the dynamic table's gain per byte × payload > table cost; use 0.3 bits/byte gain as the stated assumption).
- **dictionary** — message `{"id":4821,"type":"click","ok":true,"ts":1717}`; run the lz77 toy twice: with an empty window (expected: almost no matches) and with the window preloaded with `{"id":1000,"type":"view","ok":false,"ts":1700}`. Render both span views and the token counts. Note: same matcher, different history.
- **codebook** — a 4-entry 2D codebook with numbers (`0:(−0.8,−0.6) 1:(−0.2,0.7) 2:(0.5,0.1) 3:(0.9,−0.7)`); the stream carries the index `2` (2 bits); the decoder looks up `(0.5, 0.1)`. ⟲ slider K 4..4096: side-info bytes = K×d×2 (d=8 for this slider) vs bits/vector = log2 K, plotted against each other. Note: ratio in the stream is bought with bytes in the decoder.
- **decoder_net** — numbers only: 24 kHz × 16 bit = 384 kbps PCM in; tokens out 75/s × 8 codebooks × 10 bits = 6 kbps (64:1); decoder weights ≈ 20 MB that the receiver must already hold. Bar: bitstream per second (750 B) vs decoder (20 MB): ratio 27,000:1. Note: the prior is enormous and the message is tiny; that is the whole trick.

## Tests
`test/demos_test.js` (node, `global.window = global`, stub `document` is not needed: expose the pure compute
functions on `CYC.Demos.calc.*` and test those): Huffman CLRS lengths and mean 2.24; arithmetic `ABAC` width 1/64;
rANS encode-then-decode of `AAB` round-trips; Rice `37,k=3` → 8 bits; LZW `ABABABA` → `[65,66,256,258]`; Haar
first level averages `[11,15,99,95]`; LZ77 toy finds a match of length ≥ 8 in the lz77 lab text; μ-law 8-bit relative
error at 0.01 < 5%; k-means on the city map with K=3 reaches distortion below K=1 by ≥ 5×.
