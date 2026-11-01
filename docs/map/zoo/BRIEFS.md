# Article briefs for the Zoo

Each article is a Markdown file `docs/map/zoo/<slug>.md` written to `docs/map/style.md` (read it first, in full; it is
the authority on voice and structure). This file carries the principal content each article must land: the
question, the aha moments, a story frame, a toy-example seed, the must-cover items and formulas, the labs to embed,
figures to produce, connections and cross-links. The writer owns everything else.

## Shared conventions (every article)

Frontmatter:
```yaml
---
title: "…"                       # the name of the article, a noun phrase or the question it answers
slug: modeling-and-coding
chair: foundations | entropy | predict | transform | quantize | perceptual | prior | jobs
family: "Prefix codes"           # the family name as in map.md
summary: "one sentence"
fighters: [huffman, rice, rle]   # ids from js/data.js whose cards link here
related: [arithmetic-and-ans, modeling-and-coding]   # slugs for the Related box
reading_time: 12
---
```
Body: Markdown with `$…$` / `$$…$$` LaTeX (KaTeX), fenced ```python blocks (runnable, NumPy only, 5–25 lines each,
with the printed output shown in a comment or a following block), and these shortcodes:
- `{{< lab id="huffman" >}}` — embeds the game's interactive Special Move lab for that fighter id (js/demos.js). Use
  1–3 per article at the point in the narrative where the reader has just been handed the concept. Every lab id must
  exist in `CYC.FIGHTERS`.
- `{{< fig src="img/<slug>/<name>.png" alt="…" caption="…" >}}` — a figure you generated with Python (matplotlib via
  `uv run --with matplotlib --with numpy python script.py`; save scripts under `docs/map/zoo/src/<slug>/`, images under
  `docs/map/zoo/img/<slug>/`, PNG at 1600px wide, dark background #0b0714 with light text, consistent across articles).
- `{{< viz src="viz/<slug>-<name>.html" height="420" >}}` — an optional self-contained interactive HTML/canvas
  visualization you wrote (no external scripts; dark background), stored at `docs/map/zoo/viz/`.
- Cross-links to other articles as `[text](../<slug>/)` (site-relative to the zoo), to the game as
  `[GRID](/compressionfight/#fighter=uniform)`.
Length: 1800–3200 words. One or two toy examples with concrete numbers carried through the article. At least one
genuine aha moment. A short **Connections** section near the end. A final **In the game** line listing the fighters
and heists where this family appears. No bullet-list surveys; narrative prose. Match the style guide's rule on
terminology: introduce the proper term with a clear definition right after the toy example has made it concrete.

Facts and numbers must be true; where the brief gives a number, use it; where you add one, it must be one you are
confident of (or omit it). Do not invent paper results. Historical story framing may be invented where the brief
says so, and must then be visibly a story ("imagine…", invented names), never attributed to a real person.

---

## modeling-and-coding — Bits are surprise
Chair: foundations. Fighters: cm, neural_prior (as the extreme), huffman (as the first coder). Related: prefix-codes, arithmetic-and-ans, context-models, side-information.
Question: *Why is there a floor on how small a file can get, and what exactly decides where the floor is?*
Story frame: 1948, a young engineer at a telephone company (invented name) is asked how many telegraph symbols per second a line can carry if the operators "talk like humans". Walk through naive answers (count the letters; count distinct words), their failures, then the idea that the price of a symbol is how surprising it is.
Toy seed: a weather station sends one of {SUN, RAIN, SNOW, HAIL} every hour; p = (0.5, 0.25, 0.125, 0.125). Fixed code: 2 bits. The code 0, 10, 110, 111 averages 1.75 bits; show it is exactly $H = -\sum p \log_2 p$. Then a second station where p = (0.9, 0.05, 0.03, 0.02): H ≈ 0.62 bits but no prefix code beats 1.0. Then a third dataset coded with the first station's probabilities: cross-entropy $H(P,Q) = H(P) + D_{KL}(P\|Q)$, computed numerically.
Aha moments: (1) $-\log_2 p$ is a price, additive over symbols, and the average price is the entropy; the floor is a property of the probabilities, not of the coder. (2) The split: a *model* assigns p, a *coder* charges $-\log_2 p$; every compressor is these two halves, and all the cleverness is in the model. (3) Compressing with the wrong model costs exactly the KL divergence: "how much you knew" is a byte count.
Must cover: source coding theorem (statement, not proof), the noise test ("compressed output should look random"), modeling vs coding vocabulary, a 3-line Python entropy function, LLM + arithmetic coder as a compressor (≈0.8 bits/char vs zstd ≈2.5).
Labs: huffman, cm. Figures: bar chart of $-\log_2 p$ vs p; the three stations' costs.
Connections: perplexity = 2^(bits/token); Kelly betting and log-loss; Occam / MDL (a model is good if it compresses).

## anatomy-of-a-job — The five axes
Chair: foundations. Fighters: none specifically (link heists). Related: kv-cache-vaults, codec-dissections, side-information.
Question: *Why do MP3 and a phone call, both "compress audio", share almost no design?*
Story frame: two engineers (invented) in 1997 and 2012 given the same input file and wildly different constraint sheets. Contrast tables.
Toy seed: the same 10-second stereo clip; MP3 job (encode once, decode on a 60 MHz chip, latency irrelevant) vs VoIP job (20 ms frames, symmetric, UDP loss). Then a third: FLAC (lossless, seekable).
Aha moments: (1) The five axes (fidelity contract and consumer; redundancy profile; access pattern incl. symmetry, latency, random access, errors; compute envelope; where the prior lives) determine most of the design before any algorithm is chosen. (2) Every structural convenience, blocks, keyframes, packet independence, context resets, is bought with ratio; the dial is set by the job not the data. (3) Encode-once-decode-many means Moore's law is spent on the encoder (video generations).
Must cover: the one-sentence job template from the primer; latency budget arithmetic (frame + lookahead + overlap; 150 ms mouth-to-ear); random access vs variable-length codes (why GPU kernels want fixed-length); throughput tiers (GB/s, 100s MB/s, MB/s, KB/s).
Labs: mdct (delay slider), none else. Figures: a radar/stat-sheet comparison of 3 jobs; latency budget stacked bar.
Connections: the game's heist stat sheet is this article; Amdahl-style thinking; CAP-like trade-offs.

## rate-and-distortion — The one irreversible chair
Chair: foundations. Fighters: uniform, waterline, qmatrix. Related: scalar-quantization, block-transforms, perceptual-models.
Question: *If I may lose a little, how much do I save, and where should I lose it?*
Story frame: Bell Labs, early 1950s, an engineer (invented) digitizing telephone speech must pick a step size; invent the moment of discovering that halving the step costs exactly one bit per sample.
Toy seed: six values [3.7, −1.2, 0.4, 8.9, −5.5, 2.1], Δ = 1, 2, 4; q = round(x/Δ), x̂ = qΔ; MSE vs Δ²/12; bits vs log₂(range/Δ). Then two coordinates with variances 64 and 1: equal bits vs reverse water-filling; show the gain numerically.
Aha moments: (1) Error variance ≈ Δ²/12 and each doubling of Δ saves ≈1 bit: 6 dB per bit; the R(D) trade is a curve, and every lossy codec is a point on it. (2) Reverse water-filling: with several coordinates, spend bits where the variance is, drop coordinates below the water level; equal allocation is the one allocation that is certainly wrong. (3) Distortion is defined by the consumer: MSE is a stand-in, and the perceptual chair's job is to replace it.
Must cover: R(D) definition in words and the Gaussian formula $R(D) = \tfrac12 \log_2(\sigma^2/D)$; high-resolution approximation; dead zones (why zeros are cheap); 6.02 dB/bit; the link to transform coding gain.
Labs: uniform, waterline. Figures: R(D) curve for Gaussian with the operating points of Δ=1,2,4; water-filling bars.
Connections: JPEG quality slider = scaling one table; video CRF; the Shannon lower bound; variational autoencoders (rate term = KL).

## prefix-codes — Whole bits
Chair: entropy. Fighters: huffman, rice, rle, (none as the empty chair). Related: arithmetic-and-ans, modeling-and-coding, linear-prediction.
Question: *How do you assign bit strings so the stream decodes without separators, and how close to the floor can whole bits get?*
Story frame: David Huffman, 1951, MIT term paper (real and well known; keep to the known outline: Fano's problem, bottom-up merging); invented details only around the "night before" feeling.
Toy seed: counts A45 B13 C12 D16 E9 F5 (CLRS). Build the tree by merging the two smallest; code lengths 1,3,3,3,4,4; mean 2.24 vs H 2.22. Then the 90/10 source: Huffman pays 1.0 vs H 0.47. Then residuals [3,5,2,37,1,0,8] with Rice k=1,2,3 bit counts; then a 4×4 quantized DCT block with zigzag → (run,value) pairs.
Aha moments: (1) Kraft's inequality $\sum 2^{-l_i} \le 1$: code lengths are a budget, and an optimal prefix code spends it exactly; Huffman is greedy on that budget and provably optimal among integer lengths. (2) The one-bit tax: a symbol with p > 0.5 still costs a full bit; the cure within prefix codes is to change the alphabet (runs, pairs), which is what JPEG does. (3) Golomb-Rice: for a geometric source the optimal prefix code has a closed form, so residual coders need no table at all.
Must cover: prefix property and instantaneous decoding; canonical Huffman and table transmission; bounds $H \le L < H+1$ and the $p_{max}+0.086$ refinement; Rice parameter choice k ≈ log₂(mean); RLE + EOB in JPEG; fixed-length codes as a deliberate choice for random access.
Labs: huffman, rice, rle. Figures: the CLRS tree; Huffman mean length vs H as P(A) sweeps 0.1→0.95.
Connections: Morse code as a prefix-ish code tuned by frequency; prefix codes as binary trees as questions in twenty questions; DEFLATE's two trees.

## arithmetic-and-ans — Fractional bits
Chair: entropy. Fighters: arith, ans. Related: prefix-codes, context-models, modeling-and-coding.
Question: *How can a symbol cost 0.03 bits?*
Story frame: the 1976–79 IBM work (Rissanen, Pasco) can be named; the patent thicket (why JPEG's arithmetic mode went unused, why bzip2 chose Huffman) as the subplot; Jarek Duda's 2014 ANS as the culmination (real; keep claims modest).
Toy seed: message ABAC with P(A)=.5, P(B)=.25, P(C)=.25: intervals [0,1)→[0,.5)→[.25,.375)→[.25,.3125)→[.296875,.3125); width 1/64 → 6 bits = Σ −log₂p. Then AAAA → 4 bits. Then the rANS toy: M=4, f(A)=3, f(B)=1; encode A A B from x=4 with x′ = ⌊x/f⌋·M + cum + (x mod f): 4→5→6→27; decode by popping.
Aha moments: (1) Code the whole message as one number in [0,1): each symbol narrows the interval by its probability, and the final width is the product of probabilities, so the cost is additive and fractional. (2) Because the interval is narrowed one symbol at a time, the model may change every symbol: this is what unlocks adaptive and context modeling (the Oracle chair plugs straight in). (3) ANS: the same information in a single integer state, pushed like a stack; log₂x grows by −log₂p per symbol; table-driven variants make it as fast as Huffman, and decoding runs backwards.
Must cover: renormalization in words (finite precision), range coder vs arithmetic, binary coders (CABAC), rANS vs tANS/FSE, LIFO consequence for streaming, interleaved streams for SIMD, where each is used.
Labs: arith, ans. Figures: interval zoom diagram; bits paid vs p for Huffman vs arithmetic.
Connections: arithmetic coding = enumerative coding = a bijection between messages and reals; the link to Kelly/compounding; why LLM compression needs integer-exact models at both ends.

## lz-family — Say it again
Chair: predict. Fighters: lz77, lzw, dictionary. Related: side-information, prefix-codes, sorting-transforms, codec-dissections.
Question: *The data repeats itself; how do you say "the same as before" cheaply, and how far back is "before"?*
Story frame: Lempel and Ziv 1977/78 can be named; Phil Katz and PKZIP (real); the GIF/Unisys patent story (real, keep short). Invent a scene for the small-message problem (an engineer at a message-queue company watching gzip make 300-byte JSONs bigger).
Toy seed: `the_cat_sat_on_the_mat_the_cat_sat` → tokens (4,3) (15,4) (11,3) (23,11); bytes saved; then the same with a 4-byte window (nothing found). LZW on ABABABA → 65 66 256 258. Then a 300-byte JSON with an empty window vs a 600-byte preloaded dictionary.
Aha moments: (1) A back-reference is a prediction with certainty: the match says "the next 11 bytes are those", and only the pointer costs bits; the LZ output is still skewed, which is why an entropy coder follows (DEFLATE's two Huffman trees). (2) Two dials map onto the job: window size = memory and reach; parse effort = encoder time; decode is a memcpy regardless, which is the property that made LZ the substrate of distribution. (3) The window can be anything both sides share: a trained dictionary, the old version of a file on another machine (rsync), a content-addressed chunk store (dedup); all are LZ with a different window.
Must cover: LZ77 vs LZ78 explicit dictionaries; greedy vs lazy vs optimal parsing; zstd's FSE/Huffman back-end; LZ4 as "no back-end" for GB/s; dictionaries and the dictionary ID; content-defined chunking with a rolling hash (why fixed chunks fail on insertion).
Labs: lz77, lzw, dictionary. Figures: matches highlighted on the toy text; ratio vs window size on a synthetic log; bytes saved vs dictionary size.
Connections: suffix trees/arrays as match finders; dedup in backups; git packfiles; Brotli's built-in dictionary.

## linear-prediction — Guess, then send the miss
Chair: predict. Fighters: delta, lpc, spatial, anchor_delta. Related: prefix-codes, block-transforms, motion-compensation, codec-dissections.
Question: *If the next sample is almost the last one, why send it at all?*
Story frame: an invented time-series engineer in 2015 (Gorilla-style: Facebook's paper is real and can be cited) staring at timestamps spaced 60 s apart; then a FLAC-style audio archivist; then PNG's filter menu.
Toy seed: timestamps 1700000000, +60, +60, … → deltas 60,60,60 → delta-of-delta 0,0,0. Audio: x[n] = 100·0.9ⁿ cos(0.6n) for n=0..15: residuals with the fixed order-2 predictor x̂ = 2x[n−1]−x[n−2] vs least-squares a₁,a₂; residual energies. Image 4×4 gradient block with Sub/Up/Paeth residuals. KV: x=[100,103,105,104,108,110,109,113,115,114] chain vs anchor deltas with Δ=4: error drift vs bounded.
Aha moments: (1) Prediction does not change how many symbols there are; it changes their histogram from wide to a spike at zero, and that spike is what Rice/Huffman/arithmetic coders are good at (the modeling/coding split in action). (2) Signals are resonances: an all-pole filter driven by a small excitation; LPC estimates the filter per block and ships 2–32 coefficients as side information; speech codecs and FLAC are the same math at different fidelity contracts. (3) Chain vs anchor: predicting from the previous reconstruction gives the best entropy but a sequential decoder and drifting error; predicting from a fixed anchor gives parallel decode and bounded error (GOP, CacheGen).
Must cover: zigzag mapping of signed residuals; closed-loop vs open-loop prediction; Levinson-Durbin in one paragraph (optional); PNG per-row filter choice; intra prediction modes in video as "a menu of predictors chosen per block".
Labs: delta, lpc, spatial, anchor_delta. Figures: raw vs residual histograms; the resonance waveform and residual; chain-vs-anchor error rows.
Connections: Kalman filters and innovation sequences; ARIMA; DPCM in 1950s telephony; Parquet delta encoding.

## context-models — The mentalist
Chair: predict. Fighters: cm, neural_prior, arith (as partner). Related: arithmetic-and-ans, modeling-and-coding, side-information.
Question: *After "th", how sure are you the next letter is "e", and how do you cash that certainty?*
Story frame: Shannon's 1951 guessing game with Betty Shannon (real); PPM in the 1980s (Cleary & Witten, real); PAQ and the Hutter Prize (real); an invented newsroom scene for escapes ("we've never seen this word after that one").
Toy seed: text `the_theme_of_the_thesis`: order-2 counts after "th", Laplace-smoothed P(e|th) = 5/31 → 2.6 bits; after 10 more sightings 15/41 → 1.45 bits; show why smoothing is slow to trust and why adaptive-rate counters are used. Then mixing two predictions in the logistic domain for one symbol.
Aha moments: (1) The coder pays −log₂ P(actual | context); a model's whole job is to make that P large, and contexts are how. (2) Escapes/back-off: when a long context has never seen this symbol, fall back to a shorter one, paying a small escape cost; the sum over the ladder is the real cost. (3) Any predictor is a compressor: mixing many models with a tiny neural net (PAQ) or using a transformer (LLM) is just a better P; the decoder must run the identical model, which is why these are symmetric and slow, and why bit-exact arithmetic matters.
Must cover: order-k models and table growth; PPM escape mechanism; context mixing (logistic mixing); CABAC's contexts as a tiny context model inside video; LLM compression numbers (≈0.8 bits/char); the Hutter Prize's decompressor-size rule.
Labs: cm, neural_prior, arith. Figures: bits per character vs context order on a text sample (compute with a small script); escape ladder diagram.
Connections: n-gram language models; Kneser-Ney smoothing is the same problem; prequential/MDL; Solomonoff induction in one sentence.

## motion-compensation — The tracker
Chair: predict. Fighters: motion, spatial, anchor_delta. Related: linear-prediction, block-transforms, anatomy-of-a-job, codec-dissections.
Question: *Consecutive frames are nearly the same picture, shifted; how do you say "shifted" in a few bits, and what does it cost you to seek?*
Story frame: an invented broadcast engineer in 1988 (H.261 era) with a 64 kbit/s ISDN line and a talking head; the discovery that sending the shift beats sending the difference.
Toy seed: 8×8 frame with a bright 3×3 square at (2,2); frame 2 shifted right by 1; block search ±2 finds (+1,0); residual all zeros vs frame2−frame1 (6 nonzero cells). Then a GOP: I P P P I; seeking to frame 3 requires decoding 1–3; a dropped packet corrupts until the next I.
Aha moments: (1) Motion estimation is the encoder's expensive search; motion compensation is the decoder's free copy: the clearest example of asymmetry chosen by the job (one encode, a billion decodes). (2) The GOP length is a dial between ratio and random access, and every system with seek points (zstd frames, FLAC blocks, Parquet row groups, GGUF blocks, CacheGen anchors) is turning the same dial. (3) The residual still has spatial structure, so a transform and quantizer follow; the hybrid codec is prediction + transform coding, the same skeleton since 1988.
Must cover: block matching and SAD; sub-pixel interpolation in one paragraph; I/P/B frames; reference buffers and memory; in-loop deblocking in a sentence; rate control; low-latency configs (no B-frames, intra refresh, ACK-driven references).
Labs: motion, spatial, anchor_delta. Figures: the two frames + vector + residual; bits per frame across a GOP.
Connections: optical flow; video calls vs streaming configs; CacheGen's anchor tokens as a GOP for KV; delta sync.

## block-transforms — Which directions matter
Chair: transform. Fighters: dct, mdct, dwt. Related: rate-and-distortion, rotations-and-rank, scalar-quantization, perceptual-models.
Question: *64 numbers in, 64 numbers out, nothing saved; why is this the most important move in lossy coding?*
Story frame: Nasir Ahmed's DCT (1972–74, real); an invented scene of a JPEG committee member in 1988 watching blocking artifacts appear at low quality; the MDCT's overlap as the fix for clicks in audio.
Toy seed: row [52,55,61,66,70,61,64,73] → 8 DCT-II coefficients; keep 3 → reconstruction MSE small. A flat ramp [10,12,…,24] → DC + one tilt coefficient. Coding gain: variances of the 8 coefficients, arithmetic mean / geometric mean. Haar on [10,12,14,16,100,98,96,94] → averages [11,15,99,95], details [−1,−1,1,1].
Aha moments: (1) An orthonormal transform preserves total energy and total squared error (Parseval), so quantization error in the transform domain is error in the pixel domain; what it changes is where the energy sits, and if it piles into a few coefficients the rest can be zeroed for free. (2) Transform coding gain = arithmetic/geometric mean of coefficient variances: 1 for white noise, large for smooth signals; the transform's job is decorrelation and KLT/PCA is the optimum, DCT is its cheap fixed approximation for smooth data. (3) Block edges are the price of blocks; lapped transforms (MDCT, 50% overlap, TDAC) and wavelets each pay a different price (one frame of delay; whole-image memory).
Must cover: DCT-II formula; basis images; 8×8 choice (1990 memory); integer DCTs in video; MDCT window and delay (1152 samples at 44.1 kHz ≈ 26 ms); wavelets and progressive decoding; why JPEG 2000 lost.
Labs: dct, mdct, dwt. Figures: 8×8 DCT basis image grid; coefficient magnitudes of a photo block (log scale) vs a noise block; Princen-Bradley windows.
Connections: Fourier and the heat equation (the DCT diagonalizes a Markov-1 process asymptotically); JPEG XL's variable blocks; PCA is the data-adaptive version (next article).

## rotations-and-rank — Learned axes
Chair: transform. Fighters: pca, svd_lowrank, unrope, basis. Related: block-transforms, vector-quantization, block-float-formats, kv-cache-vaults.
Question: *A rotation changes nothing about the data; how can it halve the error of the same quantizer?*
Story frame: Karhunen/Loève and Hotelling can be named for PCA; invent a vector-search engineer in 2013 (OPQ era) and an LLM engineer in 2024 (QuaRot/KIVI era) hitting the same wall from two sides.
Toy seed: 40 points with y ≈ 0.8x; variances per axis before (≈ equal) and after rotation (λ₁ ≫ λ₂); a 2-bit grid on each; error before vs after. Key pair (8.0, 0.3) under RoPE angles m·0.5 for m=0..7: per-channel range balloons; undo the rotation and it collapses. A rank-2 8×4 matrix + noise: singular values and storage 8r+4r vs 32.
Aha moments: (1) An axis-aligned quantizer (scalar grid, product codebooks, int4 blocks) wastes levels when the data lies on a diagonal; rotate so variance (or outlier energy) lines up with the axes; PCA concentrates, Hadamard flattens, and which you want depends on the quantizer behind it. (2) Orthogonal rotations preserve dot products, so (Hq)·(Hk) = q·k: attention scores do not change, which is why QuaRot can rotate inside a transformer. (3) Low rank is energy compaction along the token axis: pre-RoPE keys live near a subspace, values do not, and RoPE's position-dependent angle destroys the structure, hence "undo RoPE first".
Must cover: KLT optimality in one paragraph; PCA via SVD; OPQ as PCA + balancing for product quantization; Hadamard cost O(n log n) and zero side-info; SVD truncation and the Eckart-Young theorem; ShadowKV rank 160/1024; a fitted basis is side information (KVTC 2.4% of parameters).
Labs: pca, svd_lowrank, unrope, basis. Figures: the before/after scatter with grids; RoPE wandering plot; singular-value decay for K vs V.
Connections: whitening in statistics; the DCT as the KLT of a Markov-1 process; random projections / JL lemma; LoRA as low rank.

## channel-and-layout — Group like statistics
Chair: transform. Fighters: prism. Related: block-transforms, perceptual-models, lz-family, codec-dissections.
Question: *Why does storing the same numbers in a different order make them compress 5× better?*
Story frame: an invented 1950s colour-TV engineer who must stay compatible with black-and-white sets (the real NTSC luma/chroma history can be mentioned without specifics); an invented data engineer discovering byte-shuffle on float arrays.
Toy seed: four RGB pixels (200,180,170)… → Y, Cb, Cr with JPEG's formulas: Y varies, Cb/Cr ≈ 128 with spread 3; stereo L=1000, R=980 → M=990, S=20. Float32 array of 8 slowly varying values: per-byte-plane entropies (exponent bytes ≈ 1 bit, mantissa bytes ≈ 8). A 5-column table row-wise vs column-wise: distinct values per column.
Aha moments: (1) R, G and B are three copies of brightness plus a little colour; mix them so one channel carries the shared part and two carry the small differences; integer-reversible versions (YCoCg-R, mid/side) make this fine for lossless too. (2) "Group bytes that share statistics, separate the ones that don't": byte shuffle, struct-of-arrays, columnar stores are one principle at three scales, and it works because entropy coders and LZ see cleaner distributions. (3) Blocking is not compression; it is plumbing that buys parallelism, seek and corruption containment, and it always costs ratio.
Must cover: YCbCr formulas; chroma subsampling as the follow-up (link perceptual); mid/side; YCoCg-R; byte shuffle (Blosc, HDF5); Parquet dictionary/RLE/delta per column; framing dials (bzip2 blocks, FLAC frames, GOP, row groups).
Labs: prism, chroma (from perceptual). Figures: RGB vs YCbCr plane energies; byte-plane entropies of a float array; row vs column layout diagram.
Connections: decorrelation again (PCA on RGB gives roughly luma/chroma); data-oriented design in games; database compression.

## sorting-transforms — The shuffler
Chair: transform. Fighters: bwt, rle. Related: lz-family, context-models, prefix-codes.
Question: *How can sorting a block of text, a reversible permutation, make it compress like a context model?*
Story frame: Michael Burrows and David Wheeler, 1994 (real); invented scene of Julian Seward writing bzip2 to dodge arithmetic-coding patents (he is real; keep the patent motive as the documented one and the scene invented).
Toy seed: `banana$` → rotations sorted → last column `annb$aa`; show inversion in a few lines of Python; MTF on `annb$aa`; then `abracadabra$`.
Aha moments: (1) Sorting rotations groups letters by the context that follows them, so identical contexts put identical letters next to each other: locality, not global skew. (2) MTF converts locality into small numbers, RLE converts runs into pairs, and only then does an entropy coder see skew: three links deep, each stage feeding the next's appetite (bzip2). (3) The same transform is a search index: the FM-index lets you count occurrences of a pattern in a genome in time proportional to the pattern, not the genome.
Must cover: BWT definition and the primary index; inverse via LF-mapping (sketch); block size vs ratio vs memory; why no streaming/random access; MTF; bzip2's stack; genomics (BWA/bowtie).
Labs: bwt, rle. Figures: the sorted rotation table; run-length profile before/after BWT on a text sample.
Connections: suffix arrays; the BWT as a compressed self-index; relation to PPM (BWT implicitly sorts by context).

## learned-transforms — The neuron
Chair: transform. Fighters: neural_enc, decoder_net, gan, mla, rvq. Related: vector-quantization, perceptual-models, side-information, kv-cache-vaults.
Question: *A fixed DCT assumes signals are sums of cosines; what if the transform could learn what speech is?*
Story frame: an invented 2020 researcher trying to get MP3-quality speech into 3 kbps for a call over a bad link; the blurry-output failure of MSE training; the discriminator fix.
Toy seed: a 320-sample frame → 8-float latent → RVQ 8×10 bits = 80 bits; 5120 bits in, 80 out, 64:1 before any entropy coding; then the bytes of the decoder (20 MB) vs the bytes per second on the wire (750 B). A 2-D VAE-style sketch: encoder, quantizer, decoder, and where the training loss sits.
Aha moments: (1) Encoder, quantizer and decoder trained jointly for the final metric: the latents are shaped for the codebooks and the codebooks for the decoder, so the pipeline is one object, and the decoder is a huge prior over the data domain. (2) MSE-trained decoders produce the average of plausible outputs, which is blurry; an adversarial loss forces one plausible output, so the fidelity contract changes from "accurate" to "plausible". (3) The same move at architecture level: MLA bakes a low-rank KV bottleneck into the model, GQA shares heads, V4 compresses along the token axis; zero cost at inference, zero choice afterwards.
Must cover: SoundStream/EnCodec pipeline (causal conv, RVQ, discriminators), bitrates (3–6 kbps vs MP3 128), EnCodec's optional LM + arithmetic coding, learned image codecs with hyperpriors (Ballé), the VAE in diffusion as a codec, failure modes (out-of-distribution, hallucination), MLA numbers (576 vs 32,768 per token per layer).
Labs: neural_enc, decoder_net, rvq, mla. Figures: pipeline diagram; bitrate vs quality sketch (qualitative, labelled as such).
Connections: autoencoders and PCA (a linear autoencoder learns PCA); audio tokens as the interface for AudioLM/MusicGen; rate term of the VAE ELBO is a bitrate.

## scalar-quantization — The grid
Chair: quantize. Fighters: uniform, companding, qmatrix, waterline, escape. Related: rate-and-distortion, block-transforms, block-float-formats, perceptual-models.
Question: *Rounding is easy; where should the grid be fine and where coarse?*
Story frame: Lloyd at Bell Labs (real person, 1957 memo; Max 1960) as in the style guide's example, with an invented scene around the telephone quantizer; μ-law as the 1960s answer; JPEG's table as the 1990 answer; KVTC's DP as the 2026 answer.
Toy seed: the six values with Δ slider; μ-law at x=0.01: uniform 8-bit relative error vs μ-law (worst case 39% vs 2.2%; actual 21.9% vs 0.10%); JPEG luma table on the Wikipedia block → 44 zeros of 64; 8 values with one outlier of 0.55 and a 3-bit absmax grid vs escaping the outlier.
Aha moments: (1) Uniform quantization with an entropy coder is within 0.25 bit/sample of the best scalar scheme at high rate; the design freedom is in Δ per coordinate and the dead zone, not in fancy level placement. (2) Non-uniform levels (companding, Lloyd-Max, NF4) match the grid to the data's density or to a relative-error criterion; μ-law is a logarithmic ear in 8 bits. (3) Outliers set the scale: one value of 30 forces a step of 4 on everything else; the cures are smaller blocks, escape codes, rotations, or a sparse side list.
Must cover: Δ²/12; dead-zone quantizers and why zeros are cheap; Lloyd-Max iteration; companding law; JPEG tables and the quality scale factor; water-filling as bit allocation across coordinates; RDO rounding (video, GPTQ as Hessian-aware rounding) in one paragraph.
Labs: uniform, companding, qmatrix, waterline, escape. Figures: Lloyd-Max levels for a Gaussian; μ-law curve; the JPEG table as a heat matrix.
Connections: A/D converters; dithering; histogram equalisation; NF4's quantile levels.

## vector-quantization — The codebook
Chair: quantize. Fighters: vq, rvq, pq, coupled, codebook. Related: rotations-and-rank, learned-transforms, side-information, scalar-quantization.
Question: *Why does quantizing two numbers together beat quantizing each alone, and what must the decoder own to make sense of a 2-bit index?*
Story frame: three little piggies' houses on a map (the style guide's example, run with it); a chicken family moves in (second cluster); then 768-dimensional embeddings at a billion scale (invented search engineer, 2017).
Toy seed: three 2-D points (houses), their centroid; six points in two clusters and K=2; the 4-point codebook {(−0.8,−0.6),(−0.2,0.7),(0.5,0.1),(0.9,−0.7)} and index 2; 16 correlated pairs: sign quantizer per axis (2 bits) vs 2-D k-means K=4 (2 bits), MSE both; H(a)+H(b) vs H(a,b). PQ: [0.9,0.1,−0.3,0.8] split in two halves with 4-entry codebooks → code (i,j), 16 effective centroids from 8 stored. RVQ: 3 stages × K=4.
Aha moments: (1) A scalar grid in d dimensions is a cube lattice and data rarely fills a cube; k-means puts levels where the data is, and the gain on correlated dimensions is exactly the mutual information a scalar quantizer pays twice (joint entropy ≤ sum of marginals). (2) Codebook size grows exponentially with bits per vector; residual and product quantization break the exponential by composing small codebooks (1024⁸ effective points from 8192 stored). (3) The codebook is side information: indices are meaningless without it; in CELP it is in the standard, in Faiss it is in the index file, in SoundStream it is in the checkpoint.
Must cover: Lloyd's algorithm and local minima/restarts; codebook collapse; asymmetric distance computation (search in the compressed domain); RVQ bitrate scalability; coupled quantization with Fisher weighting (1 bit/channel); palettes as VQ (GIF).
Labs: vq, rvq, pq, coupled, codebook. Figures: piggy map with centroids and Voronoi cells; PQ split diagram; scalar grid vs joint centroids on correlated data.
Connections: k-means in clustering; Voronoi cells; VQ-VAE and discrete tokens; the Vector Vault heist.

## block-float-formats — Blocky
Chair: quantize. Fighters: blockint, fp4, escape, evict (as the two-level case). Related: scalar-quantization, rotations-and-rank, kv-cache-vaults, attention-as-audience.
Question: *Why does a GPU want 4.5 bits per weight rather than 4, and why no entropy coder?*
Story frame: an invented engineer in 2023 trying to fit a 70B model on a laptop (llama.cpp era can be mentioned); the discovery that the kernel is the decoder.
Toy seed: 8 values with one outlier; absmax scale over 8 → small values collapse; block of 32 with fp16 scale: (16 + 32×4)/32 = 4.5 bpw; Q8_0 8.5; NVFP4 grid ±{0,.5,1,1.5,2,3,4,6} with a scale per 16: 4.5 bits; same block under int4 absmax; keys with 2 outlier channels: per-token vs per-channel grouping.
Aha moments: (1) The decoder is a kernel: one multiply per element, every thread addressing its own block; variable-length codes are impossible, so the format pays nominal width and buys random access. (2) Small blocks localise outliers at the cost of scale bits: bits/weight = (scale bits + n·b)/n; the trade is explicit and every format on Hugging Face is a point in it. (3) Keys and values differ: outlier channels want per-channel scales (KIVI) or a rotation first; eviction is the degenerate two-level quantizer, and paged memory frees nothing until a block is empty.
Must cover: Q4_0 / Q4_1 / Q4_K_M / Q8_0 layouts; absmax vs min-max (zero point); NVFP4 (E2M1 + E4M3 scale per 16 + fp32 tensor scale); MX formats in a sentence; AWQ/GPTQ as encoder-side cleverness with the same layout; KV int2/int4 (KIVI group 32, residual window); the "entropy-coded weights decompress to int4 before the kernel" caveat.
Labs: blockint, fp4, escape, evict. Figures: bits/weight vs block size for b=2..8; error vs block size on outlier data; the Q4_0 byte layout diagram.
Connections: block floating point in DSPs (1970s); floating point itself as a per-value block format; the "Random Access Race" quest.

## perceptual-models — The con artist
Chair: perceptual. Fighters: psychoacoustic, chroma, gan. Related: rate-and-distortion, block-transforms, learned-transforms, attention-as-audience.
Question: *If the error metric is wrong, every optimal codec is optimal for the wrong thing; what does the ear actually hear?*
Story frame: an invented Fraunhofer engineer in 1989 listening to a 12:1 codec and hearing nothing wrong; the pre-echo surprise on castanets (the real MP3 story can be referenced broadly).
Toy seed: 24 bands with a masker at band 8 at 80 dB; spreading function −25 dB/band below, −10 above; count bands under the curve ("free"); 2×2 RGB block → 4 Y + 1 Cb + 1 Cr (12 numbers → 6) with per-pixel error; a red/blue edge where it fails. A JND staircase sketch.
Aha moments: (1) Masking: a loud tone raises the audibility threshold around it, so quantization noise placed under the curve is free; MP3 at 128 kbps vs 1411 is mostly this. (2) Chroma subsampling is the cheapest 2:1 in all of image coding because the retina samples colour at half the resolution of brightness; quant tables encode the contrast-sensitivity curve. (3) A learned discriminator is a perceptual model you did not have to derive: it defines "plausible" and lets the decoder hallucinate detail; the price is confident wrongness.
Must cover: masking threshold, spreading, simultaneous vs temporal masking, pre-echo and block switching; CSF; 4:2:0; JND and the staircase method; encoder-only nature (decoders never run the model); PSNR vs SSIM vs perceptual metrics in a paragraph; adversarial/feature losses.
Labs: psychoacoustic, chroma, gan. Figures: masking curve over bands; CSF curve; 4:2:0 sampling grid.
Connections: Weber-Fechner law; the JPEG quality slider; loudness war; perceptual hashing.

## attention-as-audience — The softmax is the reader
Chair: perceptual. Fighters: sinkguard, spotlight, evict, tiered. Related: kv-cache-vaults, block-float-formats, perceptual-models, rotations-and-rank.
Question: *When the consumer of your data is a transformer, what does it not notice?*
Story frame: an invented inference engineer in 2024 with a 128k context that does not fit; the discovery (SpeCache's measurement is real: 0.5% of keys carry 90% of attention mass) that most tokens are never looked at; then the two systems ambushes: FlashAttention never writes scores; paged memory frees nothing until a block is empty.
Toy seed: 32 attention weights with a sink at position 0 (0.35) and a recent window (0.30): 2-bit error on everything = 0.25 weighted; exempt sink+recent → 0.0875. 32 keys in 2-D, a query, exact top-4 vs a 1-bit proxy's top-4. 16 tokens in 2 pages of 8: evict 12, free 0 pages.
Aha moments: (1) Distortion on a token the softmax weights at ~0 costs nothing: importance replaces masking as the perceptual model; sinks and the recent window are where error is amplified. (2) Selection vs eviction: a lossless backing copy one hop away (CPU) never forgets, eviction does; selection without offload saves compute, not memory. (3) The importance signal must be rebuilt from a proxy because FlashAttention leaves no receipts, and memory only comes back after compaction.
Must cover: attention sinks; KIVI residual window; DSA indexer (64 heads, 128 dims, top-k 2048) as the address book that stays dense; landmarks/PQ codes/1-bit copies as proxies; H2O vs SnapKV vs StreamingLLM vs TriAttention; paged attention block granularity; HiSparse miss rates (top-k only 30%, LRU 4k 13.4%).
Labs: sinkguard, spotlight, evict, tiered. Figures: attention mass vs rank-of-token curve; the paged grid before/after compaction.
Connections: cache replacement policies (LRU, Bélády); perceptual coding; sparse attention as a codec decision.

## side-information — The inside man
Chair: prior. Fighters: tables, dictionary, codebook, decoder_net, basis, tiered. Related: lz-family, vector-quantization, learned-transforms, modeling-and-coding.
Question: *What does the decoder know before the first byte arrives, who shipped it, and what happens when it was trained on the wrong data?*
Story frame: an invented message-queue team in 2016 (zstd's dictionary feature was built for this; Facebook is real) and an invented versioning disaster ("dictionary #7 was deleted").
Toy seed: a 300-byte JSON compressed with an empty window (bigger than the input) vs a 600-byte dictionary (half the size); DEFLATE fixed tables vs a dynamic tree header of ~60 bytes on a 200-byte payload; a 4-entry codebook and index 2; decoder weights 20 MB vs 750 B/s of tokens; cross-entropy with a mismatched model computed numerically on two tiny distributions.
Aha moments: (1) The stream is half the story; the prior is the other half, and the field's words for it are dictionary, codebook, tables, model, checkpoint. (2) Ratio on the wire is bought with bytes installed at the receiver plus versioning and specificity; the spectrum runs adaptive → header → shared dictionary → trained weights, and small inputs need the right end of it. (3) A model trained on A and used on B pays H(P,Q) = H(P) + D_KL(P‖Q): the gain of a prior is the information it shares with the data, and it decodes correctly either way (the ID matches), it just saves nothing.
Must cover: static vs semi-static vs adaptive; zstd --train (COVER), dictionary IDs; Brotli's 120 KB dictionary; HTTP compression dictionaries; standard tables (JPEG Annex K, MP3's 32 tables); codebooks in Faiss indexes and checkpoints; fitted bases (KVTC 2.4% of parameters, SpinQuant 0.26%); tiered copies as out-of-band side information in space.
Labs: dictionary, tables, codebook, decoder_net, basis, tiered. Figures: bytes saved vs dictionary size; the knowledge spectrum diagram; cross-entropy vs mismatch.
Connections: Slepian-Wolf / Wyner-Ziv (side information at the decoder, in theory); rsync; mutual information as "what the prior already knew".

## kv-cache-vaults — Same bytes, five vaults
Chair: jobs. Fighters: unrope, coupled, spotlight, tiered, waterline, anchor_delta, fp4, evict, mla. Related: anatomy-of-a-job, attention-as-audience, block-float-formats, rotations-and-rank, side-information.
Question: *One tensor, 40 GB; why does it need five different codecs depending on where it sits?*
Story frame: an invented serving team walking the same cache through five rooms of the building: the GPU, the parking lot (PCIe to host DRAM), cold storage (NVMe), the courier (network), the furnace (eviction), and the architects' office (GQA/MLA decided before pretraining).
Toy seed: bytes/token = 2 × layers × kv_heads × head_dim × 2 B; Llama-3-70B: 80×8×128×2×2 = 320 KiB → 40 GiB at 131,072 tokens; at 4.8 TB/s that is ~9 ms per token before any weight is read; DeepSeek-V3 MLA 61×576×2 = 68.6 KiB; PCIe gather at 37 GB/s vs 0.79 GB/s with per-entry copies; HiSparse miss rates.
Aha moments: (1) Bandwidth is the stat: decode re-reads the cache every token, so bytes per token is time per token and capacity is batch size. (2) The Packer's chair is empty inside HBM in every method and fills the moment the cache is stored or shipped (KVTC +1.23×, CacheGen arithmetic coding); random access is the switch. (3) The biggest shrinks happened at model-design time (GQA 8×, MLA ~5×, V4 ~9×); the codec chairs work on what the architects left.
Must cover: the five jobs as stat sheets (latency, enc/dec budget, random access, bandwidth, memory, prior allowed); keys vs values (outlier channels, per-channel vs per-token, pre-RoPE low rank); KIVI/KVQuant/QuaRot; CQ/CommVQ; ShadowKV/SpeCache/HiSparse; KVTC/TensorRT-LLM cold pages; CacheGen; SnapKV/StreamingLLM/TriAttention; the paged-memory and FlashAttention constraints; MLA/CSA numbers.
Labs: unrope, coupled, tiered, waterline, anchor_delta. Figures: bytes/token table as a bar chart; the five stat sheets side by side; a tier diagram with bandwidths.
Connections: everything else on the map; this is the capstone and should link to at least eight other articles.

## codec-dissections — Worked anatomies
Chair: jobs. Fighters: (all, by codec). Related: everything.
Question: *Can you read a codec spec as a squad sheet?*
Format exception: this article is a sequence of twelve short dissections (DEFLATE, PNG, bzip2, JPEG, JPEG 2000/WebP, FLAC, MP3, Opus, SoundStream, H.264, zstd, GGUF + Faiss PQ), each ~150–250 words: the job's stat sheet in one sentence, the chairs in pipeline order with the fighter in each, the one design decision that only makes sense given the job, and the number worth remembering. Keep the narrative voice, but no detective arc per codec; one framing paragraph and one closing paragraph suffice. Embed no labs; link each dissection to the family articles and to the game's heist.
Must cover: the composition rules from the primer (entropy coder last, lossy first, never two eaters of the same redundancy, structural conveniences cost ratio) as the framing.
Figures: one pipeline strip diagram per codec (generate programmatically with matplotlib: boxes in a row coloured by chair).
