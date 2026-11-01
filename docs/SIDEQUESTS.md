# Exercises — mechanics spec (`js/quests.js`)

Content (names, tutor text, debriefs) lives in `js/quests_data.js` as `CYC.SIDEQUESTS`; do not duplicate text,
read it from there. Implement:

```js
CYC.Quests.render(container, questId, { onComplete(result) })   // result = { score: 0..100, summary: string, stats: {...} }
CYC.Quests.has(questId)
CYC.Quests.calc.*                                              // pure functions, node-testable
```

Each quest screen (the UI agent provides the frame and the tutor's dialogue; you render the playfield inside
`container`): a playfield (canvas and/or DOM), the controls named below, a live readout, and a LOCK IN / finish
action that calls `onComplete`. Art Nouveau styling (docs/DESIGN.md tokens) comes from the page stylesheet via CSS variables
(`--role-color`, `--fg`, `--panel`, `--accent`); scope your own CSS under `.cyc-quest`. All randomness seeded
(mulberry32) so a quest replays identically unless a "NEW ROUND" control reseeds.

## shannon — Guess the next letter
Hidden sentences (pick one per round, seeded): "the quick brown fox jumps over the lazy dog", "compression is prediction plus arithmetic", "a back reference is a prediction that came true", "entropy is the floor and the model lowers it". Alphabet: a–z and space. Reveal progressively; the player types a letter (input + Enter, or on-screen keyboard). Count guesses g per letter; cost = log2(g) bits (g=1 → 0). Running total and bits/letter display; a comparison bar: 4.70 (fixed 5-bit code), 4.14 (letter frequencies), yours, ~1.0 (Shannon's human estimate). Crossed-out tried letters per position. Finish when the sentence is complete. Score: 100 if ≤ 1.5 bits/letter, linear to 0 at 4.7.

## hufftree — Build the tree
Symbols/counts: A 45, B 13, C 12, D 16, E 9, F 5 (so the optimal mean length is 2.24; H ≈ 2.196). Show nodes as tiles; click two to merge (parent count = sum; render as a growing tree on canvas). Code lengths = depth. Kraft bar: Σ2^−l over leaves (only defined when all are in one tree; show partial as "…"). Readout: mean length, optimal (compute real Huffman), H. Score: 100 if mean length equals optimal, minus 25 per 0.1 bit above. UNDO button.

## waterfill — The bit budget
Generate the engine-style 32×32 synthetic grayscale image (smooth gradient + disc + edge + noise, seeded). 8×8 DCT-II on 16 blocks. Band k (0..7) = coefficients with zigzag index in [8k, 8k+8). Per-band variance σ²_k over all blocks. Controls: 8 sliders, 0..8 bits per coefficient; budget shown as "B / 1024 bits per block" where the budget = 192 bits per block (3 bits/coef average); block total = Σ 8·b_k... careful: each band has 8 coefficients so block bits = 8·Σ b_k; cap at the budget (disable increases when full). Reconstruction: uniform quantizer per band with Δ_k = (max|coef in band|·2) / 2^{b_k} (b=0 → zero the band), inverse DCT, clamp, draw original vs reconstructed side by side (pixelated upscale), MSE readout. Optimal: reverse water-filling on σ²_k with the same budget: find θ by bisection so that Σ_k 8·max(0, ½log2(σ²_k/θ)) = budget, round to integers greedily (give bits one at a time to the band with the largest σ²_k/4^{b_k} until budget is spent). Show the optimal allocation and its MSE on LOCK IN. Score: 100 × (optimal MSE / your MSE), capped at 100.

## jnd — Spot the difference
Same synthetic image, 64×64 (upsample the generator). Variant toggle: LUMA (quantize the gray image) or CHROMA (make an RGB version with a colored gradient; quantize only Cb/Cr after YCbCr). Trial: pick a random side for the degraded image; degrade = 8×8 DCT, uniform quantizer step Δ on all coefficients, inverse. 1-up/1-down staircase on Δ in steps of ×1.25 starting at Δ=24: correct → Δ÷1.25 (gentler quantization, harder to see), wrong or SAME → Δ×1.25. 12 trials; threshold = geometric mean of the last 6 Δ values (the reversals). Convert: bits saved ≈ mean over coefficients of log2(dynamic range/Δ) compared to Δ=1, and print "at this Δ a lossless block costs X bits, yours costs Y" using a real order-0 entropy of the quantized coefficients. Score: informational only; onComplete with 100 if finished.

## phrasebook — Train the phrasebook
Sample messages (6), test message (1), foreign message (1): the same shape as the engine's json payload
(`{"id":4821,"type":"click","ok":true,"ts":1717,"user":"u_204","region":"eu-west"}` etc., types among click/view/purchase, regions among eu-west/us-east/ap-south; the foreign one is a different schema, e.g. `<event id="..." kind="..."/>` XML). Tokenize samples into clickable fragments: JSON keys with quotes and colon (`"type":`), string values with quotes, numbers, punctuation runs. Click toggles membership; budget 96 bytes (concatenate selected fragments, dedupe). COMPRESS: LZ77 (window 4096, min match 3) on the test message with the dictionary preloaded in the window vs empty window; cost model: literal 9 bits, match 20 bits (state this in the readout). Highlight matches (dict matches in gold). Then show the greedy trainer's pick: score every substring of length 3..24 occurring in ≥ 2 samples by (occurrences−1) × length, pick greedily under the budget skipping substrings already covered. Also run on the foreign message with both dictionaries. Score: 100 × (your savings / trainer's savings) capped at 100.

## `safehouses` — VQ codebook placement
60 points in 3 anisotropic Gaussian clusters (seeded), coordinates −1..1, drawn as a 2D map (grid lines). K=4; click to place up to 4 centroids, drag to move. Assignments by nearest; draw Voronoi cells (per-pixel nearest on a coarse canvas) tinted by safehouse. Readout: distortion Σ‖x−c‖², bits per point = 2, codebook bytes = 4×2×4. RUN K-MEANS: animate Lloyd iterations (≤ 20, stop when assignments stable) from the player's placement. Then BEST OF 10: 10 seeded random restarts, keep the lowest distortion; show both numbers. Score: 100 × (best / yours-after-kmeans) capped at 100; show also your pre-k-means score.

## blockpack — Pack the block
256 weights: Gaussian σ=0.02, 3 outliers at ±0.3..0.5 (seeded). Controls: bits b ∈ {2,3,4,5,6,8}; block n ∈ {8,16,32,64,128,256}; "min/zero-point" toggle (adds 16 bits per block, quantizes x−min with range max−min, otherwise symmetric absmax with 2^(b−1)−1 levels); "Hadamard rotation" toggle (apply a normalized Sylvester Hadamard of size 256 before quantizing and its transpose after; note it is orthogonal so error is preserved). Readout: bpw = (16 + (min?16:0) + n·b)/n, max abs error, RMS error, and "strip" visualization: original vs reconstructed values as two heat strips plus an error strip. Budget indicator: ≤ 4.5 bpw green, else red. LOCK IN: name the nearest real format (b=4,n=32,no min → "Q4_0: 4.5 bpw"; b=4,n=32,min → "Q4_1: 5.0 bpw"; b=8,n=32 → "Q8_0: 8.5 bpw"; b=4,n=64,min → "≈ Q4_K sub-block (real one uses 6-bit scales)"; anything with rotation → "+ QuaRot/SpinQuant-style rotation") and a 7B model's size at that bpw (7e9 × bpw / 8 bytes). Score: if bpw ≤ 4.5, 100 × (RMS error of plain Q4_0 / your RMS error) capped at 100; else 0 with the over-budget message.

## raceaccess — Random access race
4096 symbols drawn from a skewed 256-symbol distribution (geometric-ish, so Huffman gives ~5 bits). Compute real Huffman code lengths. Controls: checkpoint spacing ∈ {1, 16, 64, 256, 1024, none}; FETCH button draws a random index and animates two lanes: fixed-length (1 step) and Huffman (steps = index − last checkpoint ≤ index). Readout after each fetch and cumulative: bytes total (fixed = 4096; Huffman = ceil(Σ lengths/8) + 2 × number of checkpoints), decode steps total. Table of all spacings precomputed on LOCK IN. Score: 100 if the chosen spacing has Huffman bytes < fixed bytes and mean steps ≤ 64; 60 if smaller but slower; 30 otherwise.

## Tests (`test/quests_test.js`, node, `global.window = global`, require data.js, quests_data.js, quests.js)
Huffman on the hufftree counts → mean 2.24; waterfill: the greedy allocation's MSE ≤ the uniform allocation's MSE on the generated image; `safehouses`: best-of-10 k-means distortion ≤ a single run from a bad init; blockpack: bpw(4, 32, no min) = 4.5 and bpw(8, 32) = 8.5, and rotation reduces RMS error on the outlier data for b=4,n=256; raceaccess: Huffman total bits < 4096×8 and ≥ 4096×H; phrasebook: trainer dictionary saves ≥ 30% on the test message relative to no dictionary, and the foreign message saves < 10%.


---

# Level 5 exercises (KV cache)

Content is in `js/quests_data.js` (ids `siamese`, `anchorchain`, `rankkeys`, `pagepurge`). Same contract and style
as above. Button labels must match the `howto` text: RUN, ENTROPY, WEIGHTED, SCALAR/JOINT, NONE/CHAIN/ANCHOR, SWITCH TO V,
AUTO, COMPACT, HOLE-FILL / IN-ORDER, LOCK IN, NEW ROUND.

## siamese — Joined at the hip
4000 seeded pairs: a ~ N(0,1), b = 0.9·a + 0.1·N(0,1) (NEW ROUND cycles slope through 0.9, 0.6, 0.3, 0.0). Mode toggle SCALAR /
JOINT. SCALAR: 1-D k-means with K=2 on a alone and on b alone (10 iterations), reconstruction = (c_a[i], c_b[j]); draw the 2×2
grid points. JOINT: 2-D k-means K=4 (seeded init from 4 data points, 15 iterations). Both are 2 bits per pair; show MSE for
each after RUN. ENTROPY: bin a and b each into 16 equal-width bins over [−4,4]; print H(a), H(b), H(a,b) in bits and the
mutual information I = H(a)+H(b)−H(a,b). WEIGHTED toggle: k-means with sample weights w = |a| + 0.1 (weighted means);
show plain MSE and weighted MSE for weighted vs unweighted centroids. Score: 100 × (scalar MSE / joint MSE − 1) / 1.5 capped
at 100 (joint should be ~2.5× better at slope 0.9) with a minimum of 20 once both RUNs were done.

## anchorchain — Anchor or chain?
Matrix 64 tokens × 16 channels, seeded: x[t][c] = offset_c + walk_c[t] + noise, offset_c ~ U(−20, 20), walk a random walk
with step N(0, 0.6), noise N(0, 0.3). Quantizer: uniform step Δ = 1 (round). Modes NONE / CHAIN / ANCHOR with slider g ∈
{4, 8, 10, 16, 32} (anchor every g tokens coded raw at 8 bits = quantized with Δ but costed at 8 bits/element).
Entropy: order-0 H of the quantized symbols per mode (NONE: of q(x); CHAIN: of q(x[t]−x̂[t−1]) with x̂ the reconstruction
(closed-loop) — also show open-loop variant? No: use the simple open-loop residual q(x[t]−x[t−1]) and reconstruct by
accumulating the quantized deltas so the drift is real; ANCHOR: residual against the anchor's reconstruction). Dependency
depth: NONE 1, ANCHOR 2, CHAIN 64. Worst-case |error| over all elements after reconstruction. Draw channel 0 as three
overlaid waveforms (original, chain recon, anchor recon). LOCK IN: score 100 for ANCHOR with g ∈ {8, 10, 16}, 70 for other
ANCHOR g, 40 for CHAIN, 20 for NONE; summary quotes the three numbers.

## rankkeys — Rank the keys
Seeded: K = U·diag(s)·Vᵀ + 0.05·N, U (256×64), V (64×64) random orthonormal (QR of Gaussians), s_i = 8·0.85^i; V_vals =
U'·diag(flat s=2)·V'ᵀ + 0.05·N; Q = 8 × 64 Gaussian scaled by 1/√64. Compute the SVD of the chosen matrix once (power
iteration with deflation is fine for 64 dims, or Jacobi); slider r 1..64. Show: storage numbers = L·r + r·d vs L·d
(crossover r where equal), bar chart of singular values with the kept ones highlighted, score error
mean over queries of Σ_t |softmax(QKᵀ/√d)_t − softmax(QK_rᵀ/√d)_t|, top-16 overlap (mean over queries) for K; for V mode
show relative Frobenius reconstruction error instead (and note that attention scores do not involve V). SWITCH TO V
toggles. LOCK IN: score = 100 if r is the smallest rank with overlap ≥ 15/16 (±2), decreasing 10 per rank above, 0 if
overlap < 15/16.

## pagepurge — Rented by the room
Grid 16×16, importance scores seeded: each token w ~ Exp(1) normalized, with 3 "sink" tokens at ×10 and the last 16 ×3;
colour by w. Click toggles evicted. AUTO evicts the 192 lowest w. Readout: kept count, attention mass kept (Σw kept /
Σw), PAGES FREED (rows with 0 survivors), live. COMPACT with mode HOLE-FILL (repeatedly take a survivor from the row with
the fewest survivors and move it into the fullest non-full row; count each move as a copy) or IN-ORDER (write survivors
sequentially into fresh rows in token order; copies = number of survivors whose position changes). Show copies and pages
freed after compaction; also show "pages needed" = ceil(kept/16). LOCK IN: score 100 if kept ≤ 64, pages used = ceil(kept/16)
and copies ≤ 1.2 × minimum (hole-fill optimum), 70 if pages minimal but copies high, 40 if pages not minimal, 0 if kept > 64.

## Tests (append to `test/quests_test.js`)
siamese: at slope 0.9 joint MSE < scalar MSE / 2 and I > 1 bit; anchorchain: CHAIN entropy ≤ ANCHOR(g=10) entropy ≤ NONE
entropy, and CHAIN worst-case error ≥ ANCHOR worst-case error; rankkeys: for K, overlap at r=16 ≥ 15/16 and storage(r=16)
< L·d; for V, relative error at r=16 > 0.5; pagepurge: AUTO on the seeded grid frees ≤ 2 pages before compaction, hole-fill
reaches ceil(64/16)=4 pages with copies ≤ 64.
