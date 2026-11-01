# KV-cache compression: the quantization / transform-coding family

Research notes for Choose Your Codec. Scope: methods that shrink the KV cache by *re-representing numbers*
(quantizers, rotations, codebooks, transform coding), as opposed to eviction/merging/low-rank-retrieval.
Sources were read in full where possible (arXiv HTML/PDF); items marked **[unverified]** come from abstracts or
secondary pages only. Numbers are quoted from the papers' own tables.

The game already has: `pca` (AXIS: PCA/OPQ/Hadamard rotation), `blockint` (BLOCKY: int8/int4 + per-block scale,
cites KIVI), `vq`/`rvq`/`pq` (CATALOG/MATRYOSHKA/SPLITTER), `codebook` (KEYMASTER, prior), `companding` (cites NF4),
`qmatrix`, and heist `kvcache` (THE CONTEXT WINDOW: online, symmetric, ra 5, reveal names KIVI + QuaRot/SpinQuant).
Everything below either confirms those entries or proposes additions.

---

## 0. Shared background: why K and V are quantized differently

**Key outlier channels.** In every decoder-only LLM studied, a handful of fixed channels of the key tensor carry
magnitudes far larger than the rest, and the pattern persists across tokens; the value tensor shows no channel
pattern (KIVI Fig. 2/3; KVQuant Fig. 2; RotateKV notes the outlier channels differ per head). Consequence: a
quantizer whose scale is shared *across channels of one token* (per-token) is wrecked by the outlier channel, while a
scale shared *across tokens of one channel* (per-channel) sees a stationary magnitude. KIVI's ablation on
Llama-2-13B at 2 bits, GSM8K (fp16 = 22.67): K per-channel/V per-token (fake quant, no window) 12.21; K per-token/V
per-token 4.55; any configuration with V per-channel 0.00–0.08; KIVI-2 with its fp16 window 20.77
(KIVI Table 3, arXiv 2402.02750).

**RoPE scrambles the outliers.** RoPE rotates channel pairs by a position-dependent angle, so after RoPE the outlier
magnitude leaks between the two channels of each pair and the per-channel statistics are no longer stationary
(KVQuant §3.2, Fig. 2). Two responses exist in the literature: (a) quantize the *pre-RoPE* key and apply RoPE after
dequantization (KVQuant, CQ, KVTC, TRT-LLM's `skip_rope_quantization`), or (b) rotate Q and K by the *same* orthogonal
matrix after RoPE so attention scores are unchanged (QuaRot/SpinQuant R3), or (c) build a codebook that commutes with
RoPE (CommVQ). In codec terms: RoPE is a known, invertible transform applied by the model; the coder either undoes it
before the Butcher or chooses a Butcher that commutes with it.

**Attention sinks.** The first token(s) attract huge attention mass and are disproportionately sensitive to error;
KVQuant keeps token 0 in fp16 (2-bit LLaMA-7B perplexity 8.47 → 7.23); KVTC leaves 4 sink tokens uncompressed.

**Infra reality (NVIDIA blog, Han group, 2026).** Two facts constrain every KV method: FlashAttention never writes
the attention-score matrix to HBM, so methods that need scores force a fall-back to eager attention; and paged KV
allocators free a block only when it is completely empty ("evicting 14,400 of 16,000 tokens leaves 1,600 survivors
spread over roughly 1,000 blocks, and almost none are freed"). Quantization is the family that sidesteps both: it
needs no scores and keeps the page layout, which is why it is what production stacks (TensorRT-LLM, SGLang, vLLM)
actually ship. Source: research.nvidia.com/labs/eai/blogs/kv-cache-compression-and-its-infra-problems/.

---

## 1. KIVI (Liu et al., ICML 2024, arXiv 2402.02750) — the online baseline

1. **What:** K and V, 2-bit (or 4-bit), as stored by the framework. The paper never mentions RoPE; keys are quantized
   as cached, i.e. post-RoPE **[inferred, not stated]**.
2. **Chairs:** `quantize` = asymmetric uniform integer, zero-point + scale per group of G=32: for keys the group runs
   along the *token* axis within one channel, for values along the *channel* axis within one token. `perceptual`-like
   rule: the newest R=128 tokens ("residual") stay fp16 until a full group of 32 tokens has arrived, then are quantized
   in a block; the fp16 sliding window is "crucial for hard generation tasks such as mathematical reasoning"
   (§4.2.1). `prior`: none. "Tuning-free": no calibration at all. `entropy`: none (fixed-length). Novelty in
   information terms: choose the quantization axis along which the source is stationary; a trivially cheap "which way
   do I slice" decision buys 3 orders of magnitude on GSM8K.
3. **Profile:** online, per token, symmetric, inside the decode loop; must be random access (attention reads every
   token every step); GPU HBM resident; prefill and decode; zero latency except that keys wait ≤32 tokens in fp16.
4. **Numbers:** 2.6× less peak memory incl. weights, up to 4× batch, 2.35×–3.47× throughput (abstract, Fig. 5,
   Llama-2-7B on A100). Llama-2-7B CoQA/TruthfulQA/GSM8K fp16 63.88/30.76/13.50 vs KIVI-2 63.05/33.95/12.74;
   Mistral-7B GSM8K 38.36 → 36.01 (Table 3). Effective bits are above 2: KVQuant's Table 1 and CommVQ's Table 1 count
   KIVI-2 at ~3.0 bits/element once the fp16 window and per-32 scale/zero-point are included.
5. **Limits (admitted):** Falcon-7B (multi-query attention, already 1 KV head) loses too much at 2 bits and needs
   KIVI-4 (§4.2.2). Residual length matters: R=64 anomalously worse than 32/96/128 in the ablation (§4.2.3).
   Needle-in-a-haystack retrieval degrades at 2 bits on Llama-3-8B (Fig. 4, qualitative).
6. **Game:** already covered by `blockint` + heist `kvcache`. Suggest tightening `blockint.usedIn` to
   "KIVI (2-bit, G=32 along tokens for K / along channels for V, newest 128 tokens fp16)". The fp16 window is a
   candidate fighter (see `sinkguard`, §9).

---

## 2. KVQuant (Hooper et al., NeurIPS 2024, arXiv 2401.18079) — the calibrated scalar quantizer

1. **What:** K pre-RoPE, V post; per-channel K scales and per-token V, 4/3/2 bits. RoPE is re-applied on the fly
   after dequantization in the attention kernel (Appendix R mentions the RoPE cost inside the key matvec).
2. **Chairs:**
   - `transform`: undo RoPE (store the un-rotated key). A rotation is a transform; this one is already known to the
     decoder for free. Worth 0.82 perplexity at 3 bits on LLaMA-7B (7.05 → 6.23).
   - `quantize`: non-uniform scalar levels ("nuqX"), k-means on normalized [-1,1] channels, **weighted by diagonal
     Fisher information** (sensitivity-weighted k-means), one datatype per layer, rescaled per vector. Worth 0.29
     perplexity over uniform 3-bit, up to 0.33 over NormalFloat. Plus **dense-and-sparse decomposition**: per-vector
     outlier thresholds; the top 1% of magnitudes are stored as fp16 in a sparse CSR/CSC matrix (CSC for K, CSR for V
     so appends are cheap). 3-bit LLaMA-7B: no outliers 5.94; 0.1% per-vector 5.82; 1% per-vector 5.75 (fp16 5.68).
     In codec terms this is an *escape code*: the quantizer's alphabet has an "out of range, see side table" symbol,
     exactly as JPEG/H.264 escape large coefficients.
   - `perceptual`-like: attention-sink-aware — token 0 fp16, excluded from calibration.
   - `prior`: per-layer datatypes, per-channel K scales/zero-points, calibrated on 16 × 2K WikiText-2 tokens. Fisher
     info + k-means take "a few minutes" per layer for LLaMA-65B, ≤ 6 h sequential.
3. **Profile:** GPU HBM resident, random access, decode-bound; calibrated offline once per model; needs custom kernels
   (nuq lookup + sparse matvec). Prefill not benchmarked (Appendix S).
4. **Numbers (Table 1, WikiText-2, LLaMA 7B/13B/30B/65B, fp16 5.68/5.09/4.10/3.53):** KVQuant-3bit-1%
   5.75/5.14/4.15/3.57 ("<0.1 degradation"); KVQuant-2bit-1% 6.01/5.36/4.35/3.70; int2 11779/69965/1470/7272; nf2
   3210/5786/2044/5367. Memory 3.7×/4.8×/6.9× at 4/3/2 bits. LLaMA-7B 1M tokens on one A100-80GB at nuq2 (~64 GB);
   10M tokens on 8 GPUs (fp16 would be ~4,883 GB, nuq2 ~612 GB, Table 8). Kernel: 1.7× (K) / 1.6× (V) over fp16 matvec
   at 16K on an A6000 (Table 6). Passkey retrieval at 32K: nuq2-1% ≈ perfect vs KIVI-2-gs32-r128 ≈ 0.7.
5. **Limits (Appendix S):** sparse-matrix memory allocation is inefficient on append; latency only measured for the
   memory-bound decode phase; 32-bit indices for the sparse entries at long context. Dense-and-sparse costs a sparse
   matvec on GPU (CQ's stated reason to avoid it).
6. **Game:** confirms `companding.usedIn` ("NF4 / non-uniform levels") and `pca` for "undo the rotation". Proposed
   fighters `unrope` (transform) and `escape` (quantize) in §9.

---

## 3. QuaRot (Ashkboos et al., NeurIPS 2024, arXiv 2404.00456) and SpinQuant (Liu et al., ICLR 2025, arXiv 2405.16406) — rotation as a transform

1. **What:** whole model to 4-bit weights/activations/KV. For the KV cache: values rotated per head by a Hadamard fused
   into W_v/W_out; **queries and keys rotated by the same per-head Hadamard *after* RoPE**, so q·k is unchanged
   ("computational invariance"). KV quantized asymmetric, group 128, clip ratio 0.95. QuaRot needs "1½ Hadamard
   transforms per layer" online, ≤ 7% overhead on the 4-bit linear layer.
2. **Chairs:** `transform` = Hadamard rotation (fixed, random-signed): smears the outlier channel's energy across all
   128 channels of the head so one shared scale fits; the Hadamard is O(d log d) and needs no stored matrix. SpinQuant:
   R1 (residual stream) and R2 (per-head, V and out-proj input) are **learned** on the Stiefel manifold by Cayley SGD
   from a Hadamard init (800 WikiText-2 samples, 100 iterations, ~30 min for LLaMA-3 8B, 3.5 h for 70B; R1+R2 = 0.26%
   of weight size); R3 (Q/K for KV cache) and R4 (MLP) stay fixed Hadamards because they "cannot be absorbed into the
   weight matrix". So SpinQuant adds a `prior` (learned rotations shipped with the checkpoint) on top of QuaRot.
   Information-theoretic novelty: an orthogonal transform changes no entropy of the vector but equalizes the
   *marginal* variances, which is exactly what a scalar quantizer with a shared scale needs (the dual of PCA, which
   concentrates variance to let you drop coordinates).
3. **Profile:** online, GPU HBM, random access, decode and prefill; QuaRot needs no calibration; SpinQuant needs one
   offline optimization per model.
4. **Numbers:** QuaRot KV-only (Table 6, WikiText-2, LLaMA-2 7B/13B/70B, fp16 5.47/4.88/3.32): K4V4 5.51/4.91/3.33;
   K3V3 5.68/5.02/3.39; K2V2 9.23/7.07/4.13; K4V2 5.75/5.09/3.43 vs K2V4 8.06/6.42/3.89 (keys are the sensitive half).
   Full W4A4KV4 LLaMA-2-70B: 3.79 vs 3.32 (loss 0.47; 0.29 with 128-group), ≥ 98.6% zero-shot retained; prefill up to
   3.33× (70B, batch 32) and 2.16× (7B, batch 64); decode memory 3.89× (70B, §5.2; the conclusion's "3.39×" conflicts).
   Random orthogonal matrices are worse than Hadamard (7.45 vs 6.10 on 7B). SpinQuant: best-vs-worst of 100 random
   rotations differ by 13 zero-shot points (random Hadamards by ~6); LLaMA-2 7B W4A4KV4 64.0 vs 66.9 fp (gap 2.9);
   LLaMA-3 8B gap reduced "up to 45.1% relative to QuaRot"; online Hadamard costs 58.88 → 63.90 ms/token on an M1 Pro.
5. **Limits:** 2-bit KV still fails (K2V2 9.23); Llama-3 far more sensitive than Llama-2; rotations help a few layers a
   lot and most layers little (SpinQuant Fig. on per-layer SNR); first-token outliers left as open question.
6. **Game:** confirms `pca` (AXIS) and combo `rotateclamp` exactly. Suggest `pca.why` note: "Hadamard needs no stored
   matrix (0 side info); a *learned* rotation (SpinQuant) is 0.26% of the weights and becomes an Inside Man".

---

## 4. Coupled Quantization (Zhang, Yi, Xu, Shrivastava; NeurIPS 2024, arXiv 2405.03917) — small-group VQ

1. **What:** K and V, both; keys **pre-RoPE** ("RoPE adds outliers and makes quantization harder"). Groups of c
   contiguous channels (c = 2, 4, 8) share one b-bit code: CQ-4c8b = 4 channels, 256 centroids, 2 bits/channel;
   CQ-8c8b = 1 bit/channel; CQ-8c10b = 1.25 bits/channel.
2. **Chairs:** `quantize` = vector quantization on tiny vectors (c ≤ 8). `prior` = codebook of 2^b c-dimensional
   centroids per channel group per layer, learned once by **Fisher-weighted k-means** (k-means++ init, 100 GPU
   iterations, 16 × 2048 WikiText-2 tokens; 53 min for LLaMA-7B 2c8b, 14 min at 8c8b). Centroids total 67.11M params
   for LLaMA-7B ≈ 0.996% of model weights. `entropy`: none; fixed-length codes, so every token stays addressable.
   Information-theoretic argument (Fig. 1, LLaMA-7B K/V on 262k WikiText-2 tokens, 16 bins/channel): the sum of
   marginal entropies grows linearly in c, the **joint entropy grows sub-linearly**, i.e. channels carry mutual
   information that a scalar quantizer cannot cash in. The ablation makes this concrete (Table 4, 2 bits, WikiText-2,
   Mistral-7B fp16 4.76): uniform k-means centroids c=1/2/4 → 97.76/16.29/5.42; Fisher-guided → 5.34/5.20/5.11.
   Fisher weighting *raises* raw MSE but lowers perplexity: a perceptual-style weighting of the distortion measure for a
   machine reader rather than a human.
3. **Profile:** GPU HBM, random access, decode; offline calibration required; no CQ kernel is described in the arXiv
   version; the Stevens publication record claims 1.4–3.5× throughput **[unverified; not in the arXiv text]**.
4. **Numbers (Table 1, WikiText-2; LLaMA-7B / LLaMA-2-13B / Mistral-7B; fp16 5.68 / 4.57 / 4.76):** CQ-2c8b (4 bit)
   5.70 / 4.59 / 4.79; CQ-4c8b (2 bit) 5.97 / 4.81 / 5.11 vs KVQuant-2b (2.00 bit, no sparse) 8.17 / 29.25 / 7.33 and
   KVQuant-2b-1% (2.32 bit) 6.06 / 4.92 / 5.16; CQ-8c8b (1 bit) 8.09 / 6.55 / 7.25 vs KVQuant-1b 321.58 / 4709.83 /
   203.73; CQ-8c10b (1.25 bit) 6.78 / 5.47 / 5.90. Zero-shot LLaMA-7B (WinoGrande/PIQA/ARC-C fp16 69.93/78.67/41.72):
   CQ-4c8b 67.48/76.11/38.48; CQ-8c8b 56.51/71.16/30.20.
5. **Limits:** bit counts exclude centroids and scales; 1-bit is "reasonably well", not lossless (perplexity +2.4 on
   LLaMA-7B); calibration per model; no latency numbers; no long-context/NIAH evaluation; no KIVI comparison.
6. **Game:** the `vq` fighter's "correlated dimensions are VQ's home turf" line is this paper's thesis. Proposed
   fighter `coupled` (§9); the toy-math side quest in §11 is built on Table 4.

---

## 5. CommVQ (Li et al., Apple + UMass, ICML 2025, arXiv 2506.18879) — additive codebooks that commute with RoPE

1. **What:** K and V, per token, across all KV heads of a layer at once (LLaMA-3.1-8B: 8 heads × 128 = 1024 dims).
   Keys are stored unrotated and RoPE is applied *after* codebook decoding (Eq. 12) — effectively pre-RoPE storage.
2. **Chairs:**
   - `quantize` (values): **additive quantization with a binary code**: a lightweight encoder (linear → activation →
     linear, Gumbel-softmax for training) emits s ∈ {0,1}^Nc; the value is decoded as s·C (one matmul). Nc = 1024 for
     1-bit, 2048 for 2-bit on LLaMA-3.1-8B (Table 8) — so "bits per element" is literally Nc / 1024. This is an
     `rvq`-like idea taken to the extreme: 2^Nc effective centroids from Nc stored vectors.
   - `quantize` (keys): residual additive VQ over RoPE's 2-D subspaces, codebooks of Nc′ = 64 entries, R residual
     rounds (R = 11 for 1-bit, 21 for 2-bit), g = 64 sub-vectors share a code; avg bits = R·log2 64 / g (1.03 for R=11;
     the paper reports 2.00 for R=21, which the formula gives as 1.97 — **minor inconsistency in the paper**).
   - `transform`: the structural trick. Every key-codebook entry is a 2×2 block of the form [[x, y], [−y, x]]
     (scaled rotation, i.e. a complex number). Such blocks commute with RoPE's 2×2 rotations R(mθ), so
     R(θ)·C = C·R(θ) (Eq. 8) and the score q·R_t·(s_i C_K R_i)^T can be regrouped so that (q R_t)·C_K^T is computed
     **once per query** and only the per-token part varies (Eq. 19). Values are decoded as Softmax(A)·S_V·C_V, dropping
     cost from O(d·Nc·N) to O(Nc·N + d·Nc). Naive decoding would cost Nc× vanilla attention; after reordering about
     (R+1)/2 ×. In information terms: the codebook is constrained to a sub-algebra that commutes with the model's own
     positional transform, so the dequantizer can be folded into the attention matmul instead of materializing fp16.
   - `prior`: codebooks trained by EM with soft assignments and temperature annealing (M-step closed form) on
     FineWeb-Edu; 2.00/4.00 MB (values, 1/2-bit) and 2.75/5.25 MB (keys) in fp16 (Table 7). Encoder weights too.
3. **Profile:** GPU HBM resident, random access, decode-bound long context; calibrated once per model; needs its own
   attention kernel. Designed for "128K context of LLaMA-3.1 8B on one RTX 4090".
4. **Numbers:** 2-bit = "87.5% reduction of FP16 KV" (= 2/16). LongBench avg, LLaMA-3.1-8B-Instruct (fp16 48.05):
   CommVQ-2 (2.00 bit) 47.98; KIVI-2 (3.00 bit) 47.62; KVQuant-2 (2.33) 45.35; CommVQ-1 (1.03 bit) 44.94; KIVI-1
   16.70; KVQuant-1 5.88. GSM8K (fp16 76.27): CommVQ-2 76.04, KIVI-2 73.69; CommVQ-1 66.57 vs KIVI-1 2.20.
   InfiniteBench R.KV (key-value retrieval) collapses: 0.00 at 1-bit, 12.20 at 2-bit. 120K context: 60 GB fp16 → 20 GB
   (H100, §5.6; the intro figure's 88 GB at 128K/batch 2 is not reconciled). Per-layer per-token latency, naive →
   optimized: 36.6 → 3.8 ms at 128K (9.6×); no fp16 comparison given.
5. **Limits (stated):** EM training "was not stable and would frequently fail to optimize" before soft assignment;
   decode overhead grows with R, i.e. with lower bit width; exact-match retrieval dies at 1 bit; only LLaMA-3.1-8B fully
   evaluated; in-domain perplexity rises most (+1.37 on FineWeb-Edu).
6. **Game:** proposed fighter `commvq` (§9), proposed combo with `codebook`.

---

## 6. KVTC — KV Cache Transform Coding (Staniszewski & Łańcucki, NVIDIA; ICLR 2026, arXiv 2511.01815) — a JPEG for KV

1. **What:** K and V separately, **pre-RoPE** (RoPE undone before PCA; "positional embeddings distort the low-rank
   structure"), 4 sink tokens and the newest w = 128 tokens left uncompressed. Compression runs *between* inference
   phases (every c = 16 tokens in the simulated multi-turn setup); the model never attends over compressed data.
2. **Chairs** — the full classical pipeline, which is why it matters for the game:
   - `transform`: PCA with **one joint basis per model** for keys (and another for values) built by concatenating all
     l·h heads (p = l·h·d_head, e.g. 32 × 8 × 128 = 32,768 for Llama 3.1 8B) over n sampled token positions;
     randomized SVD on GPU; 160K–200K calibration tokens (50/50 FineWeb ≥0.95 quality + OpenR1-Math), 86 s for the PCA,
     < 10 min total for a 12B model. Ablation: GSM8K at kvtc8× with the basis spanning 1/2/4/8/16/32 layers = 27.5 /
     44.8 / 53.3 / 55.9 / 56.0 / 56.9 — cross-layer redundancy is real and keys "benefit more from global
     concatenation". The basis is 2.4% of Llama 3.3 70B's parameters.
   - `quantize`: uniform channel-wise in the PCA domain (KIVI-style), but with **adaptive bit allocation by dynamic
     programming**: a global bit budget is spread over PCA coordinates to minimize Frobenius error (orthonormality makes
     PCA-domain error equal original-domain error, Eq. 5); coordinates are grouped (group sizes {1,16,64,256,1024}),
     each group gets a 16-bit shift+scale and its own bit width; many trailing components get 0 bits, so V can be
     truncated. This is reverse water-filling / JPEG's quantization matrix for a machine reader, computed once per
     (model, ratio) in < 8 min.
   - `entropy`: DEFLATE via nvCOMP on GPU, lossless, content-dependent, averaging **1.23× on top of quantization**.
     Ablation (Mistral NeMo 12B, kvtc32×, Table 14, min–max ratio): identity 29.6–32.4; ANS 32.8–37.6; DEFLATE
     34.7–42.9; GDeflate 34.6–42.8; Zstandard 34.6–46.3; LZ4 29.5–35.5; Snappy 29.4–34.5. The Packer earns ~20–30%
     here because the DP leaves many near-zero low-bit symbols.
   - `prior`: PCA matrices, bit-allocation table, group scales (model-specific side info).
   Information-theoretic novelty: treats the KV cache as a stationary vector source and applies textbook transform
   coding (decorrelate → allocate bits by variance → entropy code). Nothing in it is new to image codecs; what is new
   is noticing that this works across layers and heads and that the storage tier, not attention, is the customer.
3. **Profile:** **storage and transfer tier**: GPU HBM (hot, for longer cache lifetime), CPU DRAM, NVMe, and
   prefill→decode node transfers; multi-turn chat and shared-prefix reuse; latency tolerant (hundreds of ms per
   compress/decompress call, Table 5); batch-friendly; random access only at the whole-cache (or 16-token chunk)
   granularity; not usable inside attention without decompression.
4. **Numbers (Table 2, Llama 3.1 8B; vanilla GSM8K 56.8, MMLU 60.5, QASPER 40.4, LITM 99.4, RULER-VT 99.8):** kvtc16×
   (realized 18–22× after DEFLATE) 56.9 / 60.1 / 40.7 / 99.3 / 99.1; kvtc32× (34–44×) 57.8 / 60.6 / 39.4 / 99.1 / 98.9;
   kvtc64× (60–88×) 57.2 / 60.7 / 37.8 / 90.2 / 95.9. Baselines: KIVI-2 (~5×) GSM8K 52.8, LITM 88.8; GEAR (~5×) LITM
   96.9; H2O/TOVA (~8×) MMLU 44.3/44.8, LITM 20.2/1.2; FP8 E4M3 (2×) GSM8K 55.2. Llama 3.3 70B MATH-500: 75.6 → 74.4 at
   10×, 72.6 at 20×, NIAH/LITM 100.0. Reasoning (R1-Qwen 7B, kvtc16× at 18–21×): AIME24 50.9 → 50.9, LiveCodeBench
   36.7 → 31.6. Latency (Mistral NeMo 12B, H100, bf16, Transformers): compress 379 ms / decompress 267 ms at
   BS 8 × 8K; TTFT recompute 3098 ms vs decompress 380 ms (8.2×); 1780 vs 208 ms at 16K/BS 2 (8.6×). Separate K/V
   ratios (Table 7): values tolerate more compression for retrieval (K32×/V256×: LITM 71.9) but less for GSM8K/MMLU.
5. **Limits (stated):** not an attention-compute method (must decompress; PCA-domain attention is future work);
   Frobenius error is a proxy that "does not guarantee task-level gains"; multi-turn is simulated; 200K-token
   calibration ceiling; kernels unoptimized; DEFLATE gain varies with content; combination with eviction untested.
   Observed: QASPER and LITM degrade first at 64×.
6. **Game:** this is the heist that finally seats all six chairs on an ML payload. Proposed heist `coldkv` and
   fighters `waterline`, `basis` (§9, §10).

---

## 7. TensorRT-LLM KV cache compression (product docs) and NVFP4 — the tier boundary in production

Source: nvidia.github.io/TensorRT-LLM/features/kv-cache-compression.html (read 2026-10-31).

1. **What:** two mutually exclusive methods per LLM instance. (a) **Cold-page quantization**: a GPU "hot" page is
   encoded to **NVFP4** as it migrates to host DRAM or disk and decoded back to the runtime KV dtype (FP16/BF16/FP8) on
   onboard; "Attention does not consume the cold representation." GDN/SSM/Conv state, DSA side buffers and the
   DeepSeek-V4 indexer cache are "skipped by quantization and preserved losslessly". (b) TriAttention: eviction, not
   quantization (out of this slice). Separately, `KvCacheConfig(dtype="nvfp4")` keeps the *active* GPU KV in NVFP4.
2. **Chairs:** `quantize` = `blockint` in its newest clothing. NVFP4 = E2M1 4-bit values, one FP8 E4M3 scale per block
   of **16**, plus a per-tensor FP32 scale (NVIDIA "Pushing intelligence to 4-bit"; LMSYS NVFP4-KV post): 8 bytes of
   data + 1 byte scale per 16 values = **4.5 bits/element, 56% of FP8**. Decoder: a multiply per element, tensor-core
   native on Blackwell (SM100/103/107 required). RoPE handling at the tier boundary: `skip_rope_quantization`
   (default false; DeepSeek-V4, GLM-5, Qwen3.5 only) keeps the RoPE part of the key at hot precision; DeepSeek-V4's
   CSA cache uses "NoPE NVFP4; RoPE 2FP4 by default" with `nvfp4_residual_dim` 0 or 64 — the KVQuant lesson
   (rotated channels quantize badly) surfacing as a config flag. Docs call the RoPE-preserving formats "exploratory"
   and warn "preserving RoPE increases cold-cache storage relative to single NVFP4". No entropy coding, no prior
   beyond ModelOpt KV scales (`scale_checkpoint_path`; for DeepSeek-V4 "only the per-layer K scale is applied to the
   CSA cache; the V scale is not used").
3. **Profile:** offload path (GPU → host/disk), KVCM V2 native C++ backend, `host_cache_size` ≥ 1; benefit "largest
   when cold-tier capacity or Page migration is a bottleneck". Per-page granularity (128 or 256 tokens per block for
   DeepSeek-V4), so random access at page level; encode/decode on the migration path, off the attention critical path.
4. **Numbers:** the TRT-LLM page gives **no accuracy numbers** ("must be validated before deployment"). Nearest
   measured data, SGLang's NVFP4 KV (LMSYS blog 2026-09-16, RTX PRO 6000 Blackwell): Qwen3.8-27B FP8→NVFP4 KV: GSM8K
   −0.31, GPQA-Diamond −1.01, AIME 2025 equal (98.33%), SWE-bench Verified 389/500 → 381/500 (−1.6 pt); Qwen3.5-397B
   GSM8K differs by one answer; capacity ×1.78 ideal; iso-capacity peak decode throughput +37% (32K) to +78% (1M);
   TTFT +0.2–0.4% at matched concurrency. Treat as indicative for TRT-LLM, not identical **[different stack]**.
5. **Limits:** Blackwell only; lossy ("may trade some output quality"); one method per instance; FP8 `fp8_ds_mla`
   footer-scale layout not covered; tested-model lists "not exhaustive".
6. **Game:** confirms `blockint` and the `kvcache` heist's "no Packer" reveal. Proposed fighter `fp4` as a `blockint`
   sibling (§9); the `coldkv` heist's `accept` squad.

---

## 8. Cross-cutting map onto the six chairs

| Work | transform | predict | quantize | entropy | perceptual (steers Butcher) | prior |
|---|---|---|---|---|---|---|
| KIVI | — | — | int2, axis per tensor, G=32 | — | fp16 window R=128 | — (tuning-free) |
| KVQuant | undo RoPE | — | Fisher-k-means scalar levels + 1% fp16 escapes | — | sink token fp16 | per-layer datatypes, K scales |
| QuaRot / SpinQuant | Hadamard / learned rotation (Q,K post-RoPE; V) | — | int4 asym, G=128 | — | — | learned R1,R2 (0.26% of weights) |
| CQ | undo RoPE | — | VQ on 2–8 channels | — | Fisher-weighted distortion | centroids (~1% of weights) |
| CommVQ | RoPE-commuting codebook structure | — | binary additive VQ (V), residual VQ (K) | — | — | codebooks 2–5 MB + encoder |
| KVTC | undo RoPE, global PCA | — | DP bit allocation, uniform per group | DEFLATE (1.23×) | sinks + 128 recent raw | PCA basis (2.4%), bit table |
| TRT-LLM cold page | (RoPE part kept hot) | — | NVFP4, 16-block E4M3 scale | — | — | ModelOpt scales |

Two requirement profiles emerge, and they want different squads:

- **Resident** (attention reads it every step): latency 0, encBudget 0–1, decBudget 0–1, ra 5, decMem 1. Fixed-length
  codes only; no Packer. Winners: KIVI/NVFP4 (no calibration) or CQ/CommVQ (calibration allowed, 1–2 bits).
- **Stored** (reuse later, maybe elsewhere): latency 3–4, encBudget 2, decBudget 2, ra 1 (whole cache or 16-token
  chunk), bandwidth 5, decMem 3. The Packer comes back, the Lens matters, and side info is fine: KVTC, cold-page NVFP4.

---

## 9. Proposed FIGHTER entries (shape of js/data.js; stats 0..5; ids are new)

```js
{
  id: 'unrope', name: 'REWIND', real: 'Pre-RoPE storage (undo the positional rotation before quantizing)', role: 'transform', family: 'rotation', level: 4,
  stats: { enc: 0, dec: 1, ratio: 1, latency: 0, mem: 0, ra: 5, lossless: true },
  domains: ['activations'],
  move: 'Unspin the Key: RoPE rotated each channel pair by a position-dependent angle; rotate it back, quantize the still vector, re-apply RoPE on the way out.',
  usedIn: ['KVQuant (pre-RoPE keys: 0.82 perplexity at 3-bit)', 'Coupled Quantization', 'KVTC (PCA on un-rotated keys)', 'TensorRT-LLM skip_rope_quantization / NoPE+RoPE split for DeepSeek-V4'],
  why: 'Keys have a few outlier channels whose magnitude is stable across tokens. RoPE smears each outlier between two channels by an angle that changes every token, so per-channel statistics stop being stationary. The rotation is known to the decoder for free: undo it, and the Butcher sees a clean source again. Costs one tiny rotation per token at decode, inside the attention kernel.',
  weakness: 'Needs a custom attention kernel that applies RoPE after dequantization. Useless for values (RoPE never touches them).',
  persona: 'a clock-faced fighter spinning its hands backwards, gears on the shoulders, cyan and brass',
},
{
  id: 'escape', name: 'ESCAPE HATCH', real: 'Dense-and-sparse decomposition (top 1% outliers kept in fp16, sparse CSR)', role: 'quantize', family: 'scalar', level: 4,
  stats: { enc: 1, dec: 2, ratio: 3, latency: 0, mem: 1, ra: 4, lossless: false },
  domains: ['activations', 'weights'],
  move: 'Let the Big Ones Out: every vector gets its own threshold; values beyond it skip the grid and ride in a side list at full precision.',
  usedIn: ['KVQuant (1% per-vector outliers: 3-bit LLaMA-7B 5.94 -> 5.75 ppl)', 'SqueezeLLM (weights)', 'JPEG / H.264 escape codes for large coefficients (the same idea, 30 years earlier)'],
  why: 'A low-bit grid is sized by its extremes: one value of 30 forces a step of 4 on everything else. Pull the 1% largest out into a sparse fp16 list and the remaining 99% fit a 3-bit grid with a tenth of the step. Bits: 3 + 0.01 x (16 + index) is about 3.3 per element.',
  weakness: 'A sparse matvec on a GPU is slow and appending to CSR is awkward; Coupled Quantization exists partly to avoid this. Index cost grows with sequence length (32-bit indices at 1M tokens).',
  persona: 'a stocky firefighter holding open a glowing trapdoor while a few oversized glowing numbers leap through, yellow helmet, red suspenders',
},
{
  id: 'coupled', name: 'SIAMESE', real: 'Coupled quantization (VQ on groups of 2-8 channels, Fisher-weighted centroids)', role: 'quantize', family: 'vector', level: 4,
  stats: { enc: 2, dec: 1, ratio: 5, latency: 0, mem: 2, ra: 5, lossless: false },
  domains: ['activations'],
  move: 'Joined at the Hip: 8 channels share one 8-bit code. 256 centroids in 8-D, 1 bit per channel, and the model still talks.',
  usedIn: ['Coupled Quantization (NeurIPS 2024): CQ-8c8b = 1 bit/channel, LLaMA-2-13B ppl 6.55 vs 4.57 fp16 where 1-bit scalar gives 4709'],
  why: 'Channels of a key or value vector are correlated: their joint entropy grows slower than the sum of their marginals. A scalar quantizer pays for each channel as if it were independent; a codebook over 4 or 8 channels pays only for the joint. Centroids are k-means weighted by Fisher information, so the error lands where the loss does not care. Fixed-length codes, so attention still reads any token alone.',
  weakness: 'Codebook per layer and channel group, trained per model (about 1% of the weights, 15-55 minutes). Group size is capped by codebook size: 8 channels at 8 bits is the practical ceiling. Pre-RoPE keys, so it needs the Rewind kernel too.',
  persona: 'two acrobats in one oversized jacket moving as a single fighter, four arms, matching masks, teal and magenta',
},
{
  id: 'commvq', name: 'COMMUTER', real: 'Commutative additive VQ (codebooks that commute with RoPE, binary additive codes)', role: 'quantize', family: 'vector', level: 4,
  stats: { enc: 2, dec: 2, ratio: 5, latency: 0, mem: 2, ra: 5, lossless: false },
  domains: ['activations'],
  move: 'Swap Places: a codebook built from 2x2 scaled rotations commutes with RoPE, so "rotate then decode" equals "decode then rotate" and the decode folds into the attention matmul.',
  usedIn: ['CommVQ (Apple / UMass, ICML 2025): 2-bit = 87.5% smaller KV, LongBench 47.98 vs 48.05 fp16; 1.03-bit GSM8K 66.57 where KIVI-1 scores 2.20'],
  why: 'Additive quantization: each value token becomes a binary vector s of length Nc (1024 or 2048 for LLaMA-3.1-8B) and decodes as s x C, one matmul, 2^Nc effective centroids. Keys use residual codebooks over RoPE\'s channel pairs with entries of the form [[x, y], [-y, x]], which behave like complex numbers and commute with the rotation; the query-side product with the codebook is then computed once per step instead of once per token. Codebooks are trained by EM and are 2-5 MB.',
  weakness: 'Exact key-value retrieval collapses at 1 bit (InfiniteBench R.KV 0.00). Decode overhead grows with the number of residual rounds (about (R+1)/2 x vanilla attention). EM training was unstable until soft assignments were added.',
  persona: 'a commuter in a trench coat mid-stride between two revolving doors that spin in sync, briefcase glowing with a 2x2 matrix, grey and amber',
},
{
  id: 'waterline', name: 'WATERLINE', real: 'Adaptive bit allocation across transform coefficients (reverse water-filling by dynamic programming)', role: 'quantize', family: 'scalar', level: 4,
  stats: { enc: 2, dec: 0, ratio: 4, latency: 0, mem: 0, ra: 4, lossless: false },
  domains: ['activations', 'vectors', 'image'],
  move: 'Flood the Valleys: coefficients with big variance get 8 bits, small ones get 2, the tail gets 0. The level is set once so the total hits the budget with the least error.',
  usedIn: ['KVTC (NVIDIA, ICLR 2026): DP over PCA coordinates, groups of {1,16,64,256,1024} with a 16-bit scale each', 'JPEG quantization matrices (the hand-tuned version)', 'MP3 bit reservoir / AAC noise allocation'],
  why: 'After a decorrelating transform the coordinates have wildly different variances. Spending the same bits on each is the one allocation that is certainly wrong; rate-distortion theory says allocate so every coordinate ends at the same error ("water level") and drop coordinates below it. Because the transform is orthonormal, error in the transform domain is error in the original domain, so the allocator can be run once on calibration data and reused.',
  weakness: 'Minimizes squared error, which "does not guarantee task-level gains" (the KVTC authors\' own words). Needs a transform in front of it; on raw channels every coordinate looks the same.',
  persona: 'a lock-keeper standing on a dam of stacked bar-chart columns, water filling up to a glowing line across them, blue overalls, lantern',
},
{
  id: 'fp4', name: 'QUARTER', real: 'NVFP4 (E2M1 4-bit floats, FP8 E4M3 scale per 16 values, FP32 per-tensor scale)', role: 'quantize', family: 'blockint', level: 4,
  stats: { enc: 0, dec: 0, ratio: 3, latency: 0, mem: 0, ra: 5, lossless: false },
  domains: ['weights', 'activations'],
  move: 'Sixteen at a Time: a 1-byte floating scale per 16 four-bit floats. 4.5 bits per element, 56% of FP8, decoded by the tensor core itself.',
  usedIn: ['TensorRT-LLM cold-page KV (encode on GPU->host/disk migration, decode on return)', 'TensorRT-LLM / SGLang active NVFP4 KV cache (Blackwell)', 'NVFP4 weights for Blackwell inference'],
  why: 'The hardware decoder is the whole point: Blackwell tensor cores read NVFP4 natively, so the only question is accuracy. A 16-element block with a floating (not integer) scale localizes outliers tighter than int4\'s 32-block and spends 1 bit per element on scale to do it. On Qwen3.8-27B, FP8 -> NVFP4 KV costs 0.3 GSM8K, 1.0 GPQA, 1.6 SWE-bench points (SGLang measurement).',
  weakness: 'Blackwell only. Keys with RoPE still hurt: production stacks keep the RoPE part of the key at hot precision ("skip_rope_quantization"), and the RoPE-preserving formats are labelled exploratory.',
  persona: 'a slim robot built from 16 translucent green cubes with one bright gold cube at the top, coin-slot visor, pixel sparkle',
},
{
  id: 'basis', name: 'BLUEPRINT', real: 'Trained transform basis shipped to the decoder (PCA matrices, learned rotations)', role: 'prior', family: 'trained', level: 4,
  stats: { enc: 0, dec: 1, ratio: 2, latency: 0, mem: 2, ra: 5, lossless: false },
  domains: ['activations', 'vectors'],
  move: 'Hold the Axes: the orthonormal matrix the Lens needs. Without it the coefficients are numbers in an unknown frame.',
  usedIn: ['KVTC PCA bases (2.4% of Llama 3.3 70B parameters, one for K, one for V, shared across all layers and heads)', 'SpinQuant learned rotations R1/R2 (0.26% of weights)', 'Faiss OPQ rotation stored in the index'],
  why: 'A data-fitted transform is side information exactly like a VQ codebook: fit once on calibration data (KVTC: 160K-200K tokens, 86 seconds of randomized SVD), ship with the model, never in the stream. Contrast with a Hadamard (QuaRot), which costs 0 bytes because it is defined by a formula, and with a DCT, which is in the standard. The game question: who trains it, how big is it, what happens when the model is fine-tuned?',
  weakness: 'Per model, per dtype. Change the model, refit. A single global basis assumes all layers share directions; KVTC shows this is mostly true (GSM8K 27.5 -> 56.9 going from a 1-layer to a 32-layer basis) but it is an assumption.',
  persona: 'an architect unrolling a glowing blueprint covered in orthogonal axes, pencil behind the ear, round glasses, blue-print blue and white',
},
{
  id: 'sinkguard', name: 'BOUNCER', real: 'Importance exemptions: attention sinks and the newest tokens stay full precision', role: 'perceptual', family: 'saliency', level: 4,
  stats: { enc: 0, dec: 0, ratio: 1, latency: 1, mem: 1, ra: 5, lossless: false },
  domains: ['activations'],
  move: 'Not Those Two: the first tokens (attention sinks) and the latest 128 never go through the Butcher.',
  usedIn: ['KIVI residual window R=128 (GSM8K on Llama-2-13B: 12.21 without it, 20.77 with it)', 'KVQuant attention-sink-aware quantization (2-bit LLaMA-7B ppl 8.47 -> 7.23)', 'KVTC: 4 sinks + 128 recent tokens uncompressed'],
  why: 'The perceptual chair for a reader that is a transformer. Attention puts enormous weight on the first token and on the recent window, so error there is amplified; everything older is averaged over thousands of tokens and tolerates 2 bits. Costs a few hundred fp16 tokens of memory and a one-group delay before keys are quantized.',
  weakness: 'Only a few hundred tokens of mercy; it does nothing for the long middle. The recent window delays per-channel key quantization until a full group has arrived.',
  persona: 'a huge doorman in a tuxedo with a velvet rope, waving two glowing tokens past while a long queue waits, night-club neon',
},
```

Needs a new perceptual family: `'saliency': { role: 'perceptual', name: 'Machine-reader importance', note: 'Which
tokens or coordinates the downstream model is sensitive to. The perceptual model when the audience is a transformer.' }`.

Proposed combos: `['unrope','blockint']` "UNSPIN & CLAMP" (KVQuant's 0.82 ppl); `['pca','waterline','lz77']`
or `['pca','waterline','huffman']` "TRANSFORM CODER" (KVTC: the JPEG recipe on activations; entropy adds 1.23×);
`['commvq','codebook']` "KEYS THAT COMMUTE". Anti-combo: `waterline` without any transform fighter ("WATER ON FLAT
GROUND": every raw channel has similar variance; nothing to allocate). Anti-combo: any `entropy` fighter in the
resident `kvcache` heist is already penalized by `ideal: entropy ['none']`.

---

## 10. Proposed HEIST entries

```js
{
  id: 'coldkv', level: 4, name: 'THE COLD STORAGE', codec: 'KV-cache transform coding (KVTC) / TensorRT-LLM cold-page NVFP4',
  domain: 'activations', payload: 'activations',
  brief: 'A chat agent will come back to this 32k-token conversation in twenty minutes, or never. Park its KV cache in CPU DRAM or on NVMe instead of recomputing it. Shrink it 10-20x; a few hundred milliseconds to pack and unpack is fine.',
  req: { lossless: false, latency: 3, encBudget: 2, decBudget: 2, ra: 1, bandwidth: 5, decMem: 3, priorOK: true },
  reqNotes: [
    'Nobody attends over the stored form: it is unpacked back to bf16 before use. So the GPU-kernel constraints of the Context Window are gone.',
    'Random access only at whole-cache or chunk level. Variable-length codes are allowed again.',
    'The decoder is the same server that encoded it, so model-specific side information (a PCA basis, bit tables) is free.',
    'Bandwidth is the whole job: PCIe, network between prefill and decode nodes, DRAM capacity. 1,000 lines of code on a 70B model is 1.6 GiB of 8-bit KV.',
  ],
  ideal: { transform: ['unrope', 'pca'], quantize: ['waterline'], entropy: ['lz77', 'huffman'], perceptual: ['sinkguard'], prior: ['basis'] },
  accept: { quantize: ['fp4', 'blockint'], entropy: ['ans', 'none'], transform: ['none'] },
  reveal: 'KVTC (NVIDIA, ICLR 2026) = undo RoPE, one PCA basis across all layers and heads (Blueprint), DP bit allocation per principal component (Waterline), uniform quantization, then DEFLATE on the GPU (nvCOMP). 18-22x on Llama 3.1 8B with GSM8K 56.9 vs 56.8; the Packer alone is worth 1.23x. Time to first token: 380 ms to decompress vs 3,098 ms to recompute. The production shortcut is TensorRT-LLM cold pages: NVFP4 (Quarter) applied as a page leaves the GPU, decoded as it comes back, no transform, no Packer, 3.5x against FP16. Same data as the Context Window; only the moment of use changed, and with it every chair.',
},
{
  id: 'onebit', level: 4, name: 'THE ONE-BIT VAULT', codec: 'Coupled Quantization / CommVQ (calibrated vector quantization of KV)',
  domain: 'activations', payload: 'activations',
  brief: 'Serve a 128k-token context on a single 24 GB consumer GPU. The KV cache must drop to 1-2 bits per element and attention still reads every token every step. You may spend an hour of calibration per model, once.',
  req: { lossless: false, latency: 0, encBudget: 1, decBudget: 1, ra: 5, bandwidth: 5, decMem: 1, priorOK: true },
  reqNotes: [
    'Resident in HBM like the Context Window, but the budget is 8x tighter: 2 bits where int2 scalar quantization gives perplexity in the thousands.',
    'Random access total, so fixed-length codes only. The ratio has to come from the Butcher and the Inside Man, not the Packer.',
    'Inside Man allowed: codebooks trained once per model (1% of the weights) ship with the checkpoint.',
    'Decode budget is small but not zero: one table lookup or one small matmul per token is acceptable if it fuses into attention.',
  ],
  ideal: { transform: ['unrope'], quantize: ['coupled', 'commvq'], prior: ['codebook'], entropy: ['none'], perceptual: ['sinkguard'] },
  accept: { quantize: ['escape', 'blockint'], transform: ['pca'] },
  reveal: 'Coupled Quantization = VQ on groups of 8 pre-RoPE channels with Fisher-weighted centroids: 1 bit/channel, LLaMA-2-13B perplexity 6.55 where 1-bit scalar gives 4,709. CommVQ = additive binary codes for values and RoPE-commuting residual codebooks for keys: at 2 bits LongBench 47.98 vs 48.05 fp16, LLaMA-3.1 8B at 128k on one RTX 4090. Both: no Packer (random access), a Keymaster holding a few MB of codebooks, and a kernel that decodes inside attention. The scalar fighters are not wrong here, they are just out of bits: KIVI at 1 bit scores 2.2 on GSM8K.',
},
```

---

## 11. Quiz-grade facts (with source)

1. KIVI quantizes **keys per-channel and values per-token**, group size 32, keeping the newest 128 tokens in fp16;
   reversing the axes (K per-token, V per-channel) at 2 bits gives GSM8K **0.00** on Llama-2-13B vs 20.77 for KIVI-2
   (KIVI Table 3, arXiv 2402.02750).
2. KIVI's headline systems numbers: **2.6× less peak memory, up to 4× batch, 2.35–3.47× throughput** (abstract).
3. KVQuant: quantizing keys **before RoPE** improves 3-bit LLaMA-7B perplexity from 7.05 to **6.23** (−0.82); keeping
   just **1%** of values as fp16 outliers takes 3-bit perplexity from 5.94 to 5.75 (fp16 5.68) (KVQuant §3, Table 1).
4. KVQuant serves LLaMA-7B at **1M tokens on one A100-80GB** (nuq2, ~64 GB) and 10M on 8 GPUs; fp16 at 10M would need
   ~4,883 GB (Table 8).
5. QuaRot rotates Q and K by the **same Hadamard after RoPE** so attention scores are unchanged; 2-bit keys are still
   the sensitive half: K2V4 8.06 vs K4V2 5.75 on LLaMA-2-7B (fp16 5.47) (QuaRot Table 6).
6. SpinQuant: 100 random rotations differ by up to **13 zero-shot points**; the KV-cache rotation R3 stays a fixed
   Hadamard because it "cannot be absorbed into the weight matrix" (SpinQuant §3).
7. Coupled Quantization: the **joint entropy of c channels grows sub-linearly** in c while the sum of marginals grows
   linearly (Fig. 1); CQ-8c8b stores **1 bit per channel** with LLaMA-2-13B perplexity 6.55 where 1-bit KVQuant gives
   4709.83 (Table 1); Fisher-weighted centroids turn 2-bit Mistral-7B from 97.76 (uniform, c=1) to 5.11 (c=4) (Table 4).
8. CommVQ: codebook entries of the form **[[x, y], [−y, x]] commute with RoPE's 2×2 rotations**, so the query-codebook
   product is computed once per step; 2-bit = **87.5%** KV reduction; GSM8K at 1.03 bits = 66.57 vs KIVI-1 = 2.20
   (CommVQ Tables 1, 3).
9. KVTC is literally transform coding: **PCA → DP bit allocation → DEFLATE**; the lossless stage adds ~**1.23×**, and
   Zstandard beats DEFLATE (34.6–46.3× vs 34.7–42.9× at kvtc32×, Table 14). Decompressing a stored cache takes 380 ms
   vs 3,098 ms to recompute it (8.2× TTFT, Table 5). One global PCA basis spanning 32 layers vs 1 layer: GSM8K 56.9 vs
   27.5 (Appendix).
10. NVFP4 = 4-bit E2M1 + one FP8 E4M3 scale per **16** values + an FP32 tensor scale = **4.5 bits/element** (56% of
    FP8); TensorRT-LLM applies it to KV pages **as they migrate to host/disk**, and attention never reads the compressed
    form (TRT-LLM docs; LMSYS NVFP4-KV post).
11. Why eviction is harder than quantization in production: paged allocators free a block only when it is empty, so
    "evicting 14,400 of 16,000 tokens leaves 1,600 survivors spread over roughly 1,000 blocks" (NVIDIA EAI blog, 2026).

---

## 12. Side-quest ideas (hands-on toy math)

**A. "Joined at the hip" — joint vs marginal entropy (teaches CQ).** Generate 4,000 2-D samples (a, b) with
b = 0.9·a + 0.1·noise (correlation ≈ 0.99). Step 1: quantize each coordinate independently to 2 levels (1 bit each,
2 bits per pair) by k-means on each axis; show MSE and the 2×2 grid of reconstruction points, three of which sit in
empty space. Step 2: run 2-D k-means with 4 centroids (also 2 bits per pair); the centroids line up along the
diagonal; MSE drops by roughly the ratio of the minor to major variance. Step 3: histogram both coordinates into 16
bins, compute H(a) + H(b) and H(a, b) from counts; display the gap as "mutual information you were paying for twice".
Optional: reweight the k-means by a per-sample "sensitivity" (e.g. |a|) and show MSE goes *up* while a weighted error
goes down — the Fisher-weighting lesson. Uses the existing `scatter` viz (points + centroids + assignments).

**B. "Rotate then quantize, or quantize then rotate?" — RoPE commutation (teaches CommVQ/KVQuant/QuaRot).** Take
2-D key vectors k and a per-position rotation R(mθ). (i) Scalar grid: quantize k, then rotate; vs rotate, then
quantize. Show the two results differ and the error depends on position m (post-RoPE outliers wander). (ii) Build a
toy codebook of 8 entries, each a 2×2 matrix [[x, y], [−y, x]] applied to a fixed seed; verify numerically that
R·C_j = C_j·R for every entry (max abs difference ~1e-12), so decode-then-rotate equals rotate-then-decode; then
replace one entry with a generic 2×2 matrix and watch the commutator blow up. (iii) QuaRot variant: rotate both q and k
by the same random orthogonal 2×2 Q and confirm q·k is unchanged while the coordinate ranges equalize. Viz: `scatter`
of the key cloud before/after rotation, and a `bits`/`histogram` of per-coordinate ranges.

(Optional C, if a third fits: "Water-filling" — 8 PCA coordinates with variances 64, 32, 16, ..., 0.5; let the player
drag a water level and watch bits per coordinate = max(0, ½·log2(σ²/D)) and total bits; compare total MSE against
equal allocation at the same budget. This is KVTC's DP in one picture and JPEG's quant matrix in disguise.)

---

## 13. Verification notes / things not confirmed

- KIVI: the paper never states pre- vs post-RoPE; "post-RoPE" is inferred from its HF-cache implementation.
- CQ: the "1.4–3.5× throughput" claim appears only in a publication-record abstract (Stevens), not in arXiv v1; the
  OpenReview PDF was behind a verification wall. No CQ kernel is described in the text I read.
- CommVQ: internal inconsistencies noted by the paper's own numbers (bit formula 1.97 vs reported 2.00; 60 GB vs
  88 GB for ~128K; "LLaMA-2-8B" vs "7B").
- KVTC: how RoPE is re-applied after decompression is not spelled out in the sections I could read (the ICLR version
  may differ from arXiv v2). Appendix B.16 (vLLM/LMCache latency) was not retrieved.
- TensorRT-LLM: the docs give no accuracy numbers at all; the NVFP4 accuracy figures quoted are from SGLang's
  implementation (LMSYS blog) and from NVIDIA's general NVFP4 format description, not from TRT-LLM measurements.
- QuaRot's conclusion contradicts its own body on decode memory (3.39× vs 3.89×) and on which model the 2.16× prefill
  refers to; body values are used above.
- The NVIDIA EAI blog's middle section was unreadable (embedded base64 images); its quantization discussion (Quant
  VideoGen "residuals from group averages, 7×, <4% overhead"; LongLive 2.0 NVFP4 KV, 1.84× throughput, fused parallel
  dequantization <2% overhead) comes from the first 100K characters and figure captions only.
