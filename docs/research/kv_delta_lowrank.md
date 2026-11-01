# KV-cache compression: the delta / residual / low-rank / retrieval family

Research notes for Choose Your Codec. Scope: methods that shrink or offload the KV cache by exploiting
*similarity* (between tokens, between layers, between the key matrix and a low-rank subspace) or that
treat the cache as a *vector database* and only decode what attention asks for. Quantization-only
methods (KIVI, KVQuant, FP4) are the sibling slice and are referenced only as baselines.

Sources read in full: CacheGen (SIGCOMM 2024 PDF), SpeCache (PMLR v267 PDF), DeltaKV (arXiv HTML v1),
ShadowKV (arXiv HTML v2), xKV (arXiv HTML v2 incl. appendices), PQCache (arXiv HTML v2), DeepSeek-V2
(arXiv HTML v5, MLA section). Numbers below are quoted from those texts; anything I derived myself is
marked "derived". Unverified items are collected in the last section.

## 0. The map in one table

| Work | Redundancy exploited | Where the saving lands | Decoder = ? |
| --- | --- | --- | --- |
| CacheGen | nearby tokens have similar K/V; early layers are loss-sensitive; per-(layer,channel) statistics are skewed | network bytes (KV shipped between machines) | GPU arithmetic decoder, pipelined with transfer |
| DeltaKV | a token's K/V is close to the mean of a few similar earlier tokens; attention only needs the selected tokens | GPU HBM (no offload) | small MLP / linear layer, run only on selected tokens |
| ShadowKV | pre-RoPE keys are low-rank; attention is sparse and temporally local | GPU HBM (values go to CPU) | rank-r matmul + RoPE for selected chunks, PCIe fetch for values |
| xKV | adjacent layers share dominant singular vectors (even when token-wise cosine is low) | GPU HBM | rank-r matmul, optionally only for selected rows |
| SpeCache | attention is sparse (0.5% of keys carry 90% of mass) and the sparsity pattern can be guessed one step ahead | GPU HBM (full copy on CPU) | 1-2 bit dequant for the index, PCIe prefetch of top-k fp16 pairs |
| PQCache | top-k attention is maximum-inner-product search over keys | GPU HBM (full copy on CPU) | PQ lookup tables for scores, PCIe fetch of top-k |
| MLA (DeepSeek-V2) | K and V of all heads are linear functions of one low-dim latent, learned at pretraining | everything (the cache is born small) | nothing: up-projections are absorbed into W_Q and W_O |

Information-theoretic reading: CacheGen is a classic *predictive + transform-free + entropy-coded* codec
(the only one in this family that produces real variable-length bits). DeltaKV is *lossy LZ for vectors*
(nearest-reference prediction with a learned residual coder). ShadowKV, xKV and MLA are *transform coding
by rank truncation* (KLT / SVD along different axes: tokens, layers, heads). SpeCache and PQCache are
*scalable / layered coding*: a coarse base layer (low-bit copy or PQ codes) is enough to *decide*, the
enhancement layer (fp16) is fetched only where the decision says so. The common novelty vs. image/audio
codecs: the "receiver" is a softmax, and a softmax is extremely tolerant of errors on tokens it was going
to ignore anyway. That is the family's perceptual model.

## 1. CacheGen (Liu et al., UChicago / Microsoft / Stanford, SIGCOMM 2024, arXiv 2310.07240)

**What is compressed, what redundancy.** A precomputed KV cache of a long reused context (RAG documents,
chat history), sent from a storage server to the serving GPU over single-digit Gbps links. Three empirical
insights (Llama-7B/13B, LongChat 9.2K-9.6K-token contexts):
1. *Token-wise locality*: within one layer and channel, consecutive tokens have similar values; the
   deltas have 2.4-2.9x lower variance than the raw values and are concentrated around zero (Fig. 3).
2. *Layer-wise sensitivity*: rounding noise in shallow layers hurts accuracy much more than the same
   noise in deep layers (Fig. 4).
3. *Grouping*: entropy per element drops much more when values are grouped by (layer, channel) than when
   grouped by token position (Fig. 5, roughly 5.4 -> 4.8 bits/element on Llama-7B).

**Pipeline on the six chairs.**
- Oracle (predict): split tokens into groups of 10; the first token is the *anchor*, coded on its own;
  the other 9 are coded as deltas against the anchor (not the previous token). The paper says this is
  chosen so that encode and decode of all tokens in a group run in parallel; it is explicitly compared
  to a video GOP with one I-frame and 9 P-frames that all reference the I-frame.
- Butcher (quantize): uniform ("vectorwise", LLM.int8-style) quantization of the deltas with a bin
  size per layer group. Three equal layer groups, bins 0.5 / 1 / 1.5 from early to late (App. C.2).
  Anchor tokens stay at 8 bit because their error shifts the whole group's delta distribution.
- Packer (entropy): arithmetic coding with a *separate probability table per (layer, channel)* for
  deltas and another for anchors, profiled offline once per model and reused for all contexts.
  Grouped tables cut the bitstream by up to 53% vs one global table (Sec. 7.5).
- Inside Man (prior): those per-model probability tables, shipped to both ends.
- Transform and Con Artist chairs are empty. No rotation, no low-rank.
- Streaming layer: the context is cut into 1.5K-token chunks, each pre-encoded at several quality levels
  (default medium level: 140 MB per chunk for Llama-7B). The sender measures throughput on the previous
  chunk and picks the highest-quality level that fits the TTFT SLO; if even the lowest level is too slow
  it sends the chunk as *text* and lets the LLM recompute that chunk's KV. Decoding is a CUDA
  arithmetic decoder (one thread per token) pipelined with the transfer of the next chunk.

**Novelty in IT terms.** The first paper to drop the tensor format of the KV cache and emit a real
bitstream; the three-stage predictive/quantize/AC structure is textbook but the *context model*
(per layer-channel tables) and the *anchor-group* predictor are tuned to how K/V are laid out. The
layer-wise bin size is a crude rate allocation, i.e. a "perceptual" weighting by loss sensitivity.

**Requirements profile.** Offline encode (once per context, ~200 ms), online decode on the GPU before
prefill. Random access at *chunk* granularity (1.5K tokens) and at group granularity (10 tokens) inside a
chunk; not per element. Decode budget: moderate, but it happens once per request, not per decoding step.
Memory tier: object storage / another server -> network -> GPU. Network streaming is the whole point.
No interaction with sparse attention; the decoded cache is a normal dense fp16 tensor.

**Headline numbers.** Mistral-7B, LongChat 9.4K tokens: 8-bit quantized KV 622 MB, CacheGen 176 MB at
accuracy 0.98 vs 1.00 (Table 1). Across Mistral-7B / Llama-34B / Llama-70B on four datasets: KV size
3.5-4.3x smaller than the uniform-quantization baseline at matched quality; TTFT 3.2-3.7x lower than
quantization and 3.1-4.7x lower than sending text at 3 Gbps; 1.67-1.81x lower TTFT even vs 8-bit.
Quality cost: <2% accuracy, <0.1 F1 point, <0.1 perplexity. Streaming with adaptation: SLO-violation
rate 81% -> 8% at a 1 s SLO. Derived: Mistral-7B fp16 KV is 128 KB/token, so 9.4K tokens ~1.2 GB; 176 MB
is ~15% of fp16, i.e. roughly 2.3 bits/element including anchors.

**Admitted limits.** Gains shrink above ~20 Gbps and against very fast GPUs (recompute wins). Not tested
on free-text generation tasks or >70B models. Does not handle incremental/scalable streaming (named as
future work, with Scalable Video Coding as the model). Volatile contexts (web search results) are never
reused so never cached. Only the *prefill* cache is handled; nothing about decoding-time memory.

## 2. DeltaKV (Hao et al., HIT Shenzhen, arXiv 2602.08005, unreviewed; code: Sparse-vLLM)

**What is compressed, what redundancy.** The GPU-resident cache of the *sparse-attention layers* of a
model running OmniKV-style selective attention. Two redundancies: a token's pre-RoPE K/V is close to the
mean of a few similar earlier tokens (inter-token similarity, chosen by nearest neighbour rather than by
position), and sparse attention only ever touches a small fraction of tokens, so most tokens never need
to be decoded (decode-on-select; related work says <= 10% of tokens get reconstructed).

**Pipeline on the six chairs.**
- Oracle: *references* are a strided subset (stride s = 10) of earlier tokens, kept uncompressed. For
  each token, find its k = 4 nearest references by L2 distance on pre-RoPE K/V and predict the token as
  their mean. This is lossy LZ77 for vectors: a match is a prediction that came true.
- Lens: a trained compressor (GeLU MLP, 2·d_k -> d_c with d_c = 25% of 2·d_k, hidden 4096) maps token
  and references into a latent space; the residual is taken *in the latent space*.
- Butcher: 4-bit token-wise (KIVI-style) quantization of the latent residuals only; references, sink
  tokens, recent window and full-attention ("filter") layers stay BF16.
- Decoder: an MLP, or a bias-free linear layer in the "light" variant, applied only to tokens the sparse
  mask selected.
- Inside Man: the compressor/decoder weights (trained on 160M tokens, ~8 GPU-hours for a 7-8B model).
- Packer: none. Con Artist: the sparse-attention mask itself.

**Novelty.** Residual coding against a *content-addressed* reference set (nearest earlier tokens) instead
of a positional one (previous token, anchor), plus the observation that in a sparse-attention model the
decoder only has to run on the selected rows, so decoder cost scales with the attention budget, not with
the context.

**Requirements profile.** Online, per token, inside the forward pass; needs a nearest-reference search per
new token. Random access per token (reference + residual + decoder, no chain). Decode budget: small MLP
per selected token; the paper measures 37.3 ms of a 91.0 ms step at batch 16 going to reconstruction,
because no fused kernel exists yet. Tier: GPU HBM only, explicitly avoids offload. Requires a
sparse-attention model (OmniKV layout) to make sense.

**Headline numbers.** Keep-ratio KR = L_full/L + (L_sparse/L)(1/s + d_c/2d_k); memory down to 29% of
the original with 4-bit residuals (43-48% without). LongBench Llama-3.1-8B: 50.0 -> 50.2 (KR 45%);
Qwen2.5-32B 48.4 -> 47.7. SCBench Llama-3.1-8B 50.4 -> 45.0 (lightweight decoder 46.3); Retrieval-KV
subtask 79.0 -> 58.0. AIME (R1-Distill-Qwen-7B) 50.0 -> 43.3 (SnapKV 33.3). Decode throughput on one
RTX PRO 6000: 512K context 33.1 tok/s (vLLM full) -> 67.7 (DeltaKV light); 128K: 143.2 -> 148.4 (187.0
light) but OmniKV alone reaches 216.7.

**Admitted limits.** Python-level bookkeeping and CUDA sync overhead; no fused reconstruct+attend kernel.
Accuracy drops on retrieval of random-looking strings (distribution mismatch for a trained compressor).
References and dense layers are not compressed at all. Slower than OmniKV alone at 128K.

## 3. ShadowKV (Sun et al., ByteDance Seed / CMU, arXiv 2410.21465; ICML 2025 spotlight per reading list)

**What is compressed, what redundancy.** Decoding-time KV of long-prefill workloads. Four observations:
(a) *pre-RoPE* keys have a sharp singular-value decay (rank 160 of 1024 recovers near full-rank accuracy
on Llama-3.1-8B); post-RoPE keys and values do not; (b) the subspace is sequence-specific (shared within a
prompt and its continuation, not across prompts); (c) post-RoPE keys in chunks of 8 tokens are close to
their chunk mean, except 0.2-0.3% outlier chunks; (d) selected chunks repeat across adjacent decoding
steps, so caching them saves >60% of fetch traffic.

**Pipeline on the six chairs.**
- Lens: SVD of the pre-RoPE key matrix per layer and KV head during prefill (asynchronously on CPU, or
  cached with the prefix); keep A (L x r) and B (r x d), r = 160. Keys are reconstructed on demand as
  (A B) then RoPE'd. Values are *not* transformed; they are offloaded to CPU in full.
- Con Artist: landmarks (chunk means of post-RoPE keys, chunk size 8) kept on GPU; each step the query is
  scored against landmarks, pooled over GQA heads by max, top K = 256 chunks per KV head are selected
  (derived: 2048 tokens, 1.56% of 128K). Outlier chunks (48) are always kept in full.
- Inside Man (stretch): the full value cache on CPU, reachable over PCIe, and the per-sequence SVD
  factors. CUDA multi-streams overlap key reconstruction with value fetch.
- Butcher, Packer: empty. Oracle: none.

**Novelty.** Separating K and V by their statistics: keys get a KLT-style rank truncation, values get
tiering, and the sparse selector runs on *exact* landmarks of the uncompressed keys so selection quality
does not degrade with compression. Keeping the low-rank factors on GPU turns "fetch K from CPU" into
"recompute K on GPU", trading PCIe bandwidth for FLOPs the decoder has to spare.

**Requirements profile.** Prefill-time SVD (one-off), decode-time rank-r matmul per selected chunk.
Random access per chunk of 8. Tier: GPU for factors + landmarks + outliers, CPU for values. Interaction
with sparse attention is intrinsic. Generated tokens are not covered by the base method (ShadowKV+
projects them into the existing subspace).

**Headline numbers.** >6x GPU KV footprint reduction (theory 7.08x for 128K, r=160, K=256, O=48).
Batch size up to 6x (Llama-3-8B-1M at 60K: full KV OOMs past batch 8, ShadowKV reaches 48). Throughput
on A100: up to 3.04x (Llama-3.1-8B, 122K: 245.90 vs 80.78 tok/s); 2.97x / 2.66x / 2.56x on the other
models. Equivalent bandwidth 7.2 TB/s, 3.6x the A100's 2 TB/s. RULER 128K: Llama-3.1-8B 85.53 -> 83.57,
Llama-3-8B-1M 86.68 -> 86.88, GLM-4-9B-1M 86.82 -> 85.62, Yi-9B-200K 65.73 -> 65.53. LongBench budget
256: Llama-3.1-8B 48.96 -> 48.13.

**Admitted limits.** Accuracy falls when chunk size > 8. Models with 4 KV heads gain less (GLM-4 falls to
2.23x at 244K). Chunk hit rate is ~60% regardless of chunk size, so 40% of the fetch is new every step.
Decode-only; chunked prefill not discussed. Values are not compressed at all on GPU.

## 4. xKV (Chang et al., Cornell abdelfattah-lab, arXiv 2503.18893)

**What is compressed, what redundancy.** Prefill KV cache, keys (pre-RoPE) and values symmetrically.
Observation: *token-wise* cosine similarity between adjacent layers is low (so MiniCache-style per-token
merging is weak), but Centered Kernel Alignment between adjacent layers' caches is consistently high: the
*dominant singular vectors* are shared. The rank needed for 95% cumulative eigenvalue mass, relative to
total size, drops as more layers are grouped (Fig. 2c).

**Pipeline on the six chairs.**
- Lens: split layers into windows of W = 4; concatenate the W caches horizontally (L x W·d); one rank-r
  SVD (torch.svd_lowrank, online during prefill, ~3.9% of prefill time at 128K) gives a *shared token
  basis* A_k = U Sigma (L x r) and a layer-specific B_l (r x d) per layer. Storage O(W·L·d) ->
  O(L·r + W·r·d). Training-free, no calibration set.
- Con Artist (optional, "Selective Reconstruction"): ShadowKV-style landmarks + top-k chunks at a 2K
  token budget; only the selected rows A_k[S,:] B_l are reconstructed. Dense reconstruction is
  compute-bound (0.4x kernel speed at 122K), so SR is what makes it fast.
- Butcher / Packer / Oracle: empty. Inside Man: the per-sequence factors (not a shipped prior).

**Novelty.** Cross-layer KLT: the rank saving comes from sharing the *token-side* basis across layers,
which is the axis MLA shares at training time. The paper shows the same post-hoc trick works on an MLA
model's already-latent cache (DeepSeek-Coder-V2-Lite, ~3x on RepoBench, ~3.5x on LCC "without
compromising accuracy", App. E), i.e. even a learned latent still has cross-layer redundancy.

**Requirements profile.** Prefill-time SVD per prompt; random access per row (one r-dim dot product per
layer); decode budget small if SR is used; GPU only (SR variant offloads nothing but counts landmark
memory). Generated tokens stay uncompressed.

**Headline numbers.** Llama-3.1-8B-Instruct RULER 64K: xKV 88.50 at 8.03x vs 91.89 full; single-layer SVD
45.71 at 8.40x; MiniCache 45.04 at 1.30x. Window ablation (RULER): W=1 45.71, W=2 75.15, W=4 88.50, W=8
88.91. LongBench average: full 43.20, xKV 8.03x 42.27, KIVI-2 7.10x 41.47, SnapKV 8x 42.83. With SR on
A100: up to 4.23x end-to-end throughput at 122K vs FlashAttention-2 full attention, ~30% above ShadowKV
at similar accuracy (LongBench: xKV-SR 42.69 vs ShadowKV 40.51).

**Admitted limits.** Long-prefill setting only; long generations are future work. xK-SR throughput is
slightly below ShadowKV's because cross-layer reconstruction has a larger inner dimension. Compression
rates assume 64K context. Some internal inconsistencies in the text (see last section).

## 5. SpeCache (Jie et al., Peking U / Huawei Noah's Ark / BUPT, ICML 2025, PMLR 267:27917)

**What is compressed, what redundancy.** Decoding-time VRAM. Observations (Llama-3-8B, PG19 8K): 0.5% of
keys cover 90% of a query's attention mass; the pattern is query-dependent (greedy eviction a la H2O has a
much lower hit rate than per-query top-k); decoding is memory-IO bound so a second token in the batch is
nearly free; a top-1% fetch over PCIe costs 95% less than fetching everything even though sparse copies
are ~5x slower per byte.

**Pipeline on the six chairs.**
- Butcher: a KIVI-quantized copy of the whole cache at 2 bit or 1 bit (group 32 or 64) lives in VRAM.
  For 1 bit the zero-point and scale are redefined (z = (3 min + max)/4, s = (max - min)/2) so the two
  levels sit at interval midpoints instead of at min/max.
- Inside Man (stretch): the full 16-bit cache in CPU RAM, written layer by layer during prefill.
- Con Artist: at each step two tokens are decoded in parallel: the real output token (using the low-bit
  cache with its top-k entries replaced by prefetched 16-bit pairs) and a *speculative* next token whose
  attention scores on the low-bit cache name the top-k (k = 64) pairs to prefetch for the next step.
  Prefetch overlaps with the remaining layers' compute.
- Lens, Oracle, Packer: empty.

**Novelty.** Layered coding with a one-step-ahead *request predictor*: the base layer is good enough to
decide what to fetch, and the enhancement layer is fetched before it is needed. Nothing is ever evicted,
so there is no irreversible forgetting.

**Requirements profile.** Online, training-free, per-token quantization at prefill. Random access per
token on both tiers. Decode budget: one extra token per step (free when IO-bound) plus a top-k. Tiers:
GPU + CPU over PCIe. Interaction with sparse attention: the method *is* a sparse-attention selector with a
lossy fallback for the unselected tokens.

**Headline numbers.** Mistral-7B-Instruct-v0.2, LongBench avg: full 42.3; 1-bit KIVI g=64 alone 28.9 at
0.10x size; + SpeCache 40.4 at the same 0.10x. Llama-3-8B-Instruct: full 42.7; 1-bit g=64 24.7 -> 41.9
with SpeCache (0.11x). Beats H2O / InfLLM / StreamingLLM at 0.13-0.25x while using 0.10-0.17x.
Needle-in-a-haystack at 1 bit is restored to near 16-bit. Throughput (A6000, Mistral-7B, 32K): batch 3 ->
36, 10.3 -> 47.3 tok/s (4.6x). Speculative vs. non-speculative fetch: 643 vs 877 ms/step at 2 bit,
accuracy 42.5 vs 42.6. Even k = 16 helps.

**Admitted limits.** PyTorch multi-stream copy, not optimal kernels. k must be tuned per deployment. At
large batch the PCIe load becomes the dominant cost. The paper does not discuss prefill memory or
multi-GPU.

## 6. PQCache (Zhang et al., Peking U / Baichuan, arXiv 2407.12820; SIGMOD 2025 per reading list)

**What is compressed, what redundancy.** Decoding-time VRAM, framed as "KV storage and retrieval is an
embedding-retrieval problem": picking top-k keys for a query is ANNS / maximum inner-product search.

**Pipeline on the six chairs.**
- Butcher: during prefill, keys of each layer and head are split into m sub-vectors and k-means gives 2^b
  centroids per sub-vector; m = 2, b = 6 (LongBench) or m = 4, b = 8 (InfiniteBench). Derived: for
  d = 128 that is 12 or 32 bits per key vs 2048 bits in fp16, matching the paper's "1/128" and "1/64" of
  key memory. K-means runs on the *CPU*, overlapped with GPU prefill, with the iteration count capped by a
  fitted cost model so it never stalls the GPU.
- Inside Man: the centroids (about 16 MB for a 7B GQA model at m=2, b=6) stay on GPU; the codebook is
  built per sequence, so this is side information created online, not shipped.
- Con Artist: at each step the query times the centroids gives lookup tables; summing table entries
  through each token's codes gives approximate scores; the top 1/5 or 1/10 of tokens are fetched from CPU
  together with always-resident initial and local tokens. A block-level GPU cache (128-token blocks, LRU
  or LFU, 32 blocks = 4K tokens, hit rate ~0.6) cuts TPOT by 26-33%. PQ codes for the next layer are
  prefetched one layer ahead.
- Lens, Oracle, Packer: empty. No rotation before PQ (no OPQ step mentioned).

**Novelty.** Reuses the vector-database stack (PQ + asymmetric distance computation) unchanged, but as
an *index for attention* rather than as the stored representation: the codes are never dequantized into
keys, they only rank tokens. Decompression cost is zero; the "decoder" is a table lookup.

**Requirements profile.** Prefill-time k-means on CPU; decode-time ADC lookups plus a PCIe fetch of the
selected KV. Random access per token (PQ codes are fixed-length). Tiers: GPU for codes + centroids +
initial/local tokens + block cache, CPU for everything else. Interaction with sparse attention: it *is*
the selector.

**Headline numbers.** Llama-3.1-8B LongBench: full 47.37, PQCache 47.29 at 1/5 and 47.19 at 1/10 tokens,
gap to oracle top-k < 0.70%. InfiniteBench: full 47.63, PQCache 47.31 / 46.80. Llama-3.1-70B on 4x A800:
52.89 full vs 52.88. NIAH to 131K near full/oracle where InfLLM often fails. TPOT stays below human
reading speed and flat in sequence length. Hardware: RTX 4090 24 GB, 2x Xeon Gold 6330, 500 GB RAM,
PCIe 1.0 x16.

**Admitted limits.** PQ quality is bounded by how many k-means iterations fit in the prefill shadow; a
fast GPU with a slow CPU gets worse codes. Output tokens get nearest-centroid codes from a codebook
trained on the input; long generations would need periodic re-clustering. Multi-turn strategies proposed
but not evaluated. SparQ slightly beats it at a 1/16 communication budget (with much higher latency).

## 7. Context: DeepSeek-V2 MLA (arXiv 2405.04434) as the trained endpoint

MLA is what the whole family converges to when the transform is allowed into the training loop. Per
token, one latent c_KV = W_DKV h_t of dimension d_c = 512 is cached, plus one decoupled RoPE key k_R of
dimension d_R = 64 shared by all heads. Keys and values of all n_h = 128 heads (d_h = 128) are
k_C = W_UK c_KV and v_C = W_UV c_KV; at inference W_UK is absorbed into W_Q and W_UV into W_O so the cache
is never expanded. RoPE cannot be absorbed (it sits between W_UK and the dot product), hence the separate
small positional key. Cache per token per layer: d_c + d_R = 576 elements vs 2·n_h·d_h = 32,768 for MHA
(derived: 56.9x), "equal to GQA with only 2.25 groups" (Table 1); the paper claims MLA is *stronger* than
MHA, not merely cheaper, and quotes a 93.3% KV reduction vs DeepSeek 67B. In chair terms: Lens = learned
linear down-projection (a trained KLT), Inside Man = the model weights themselves, Butcher/Packer empty.
xKV shows the latent still has ~3x of cross-layer redundancy left. TensorRT-LLM's FP4 KV codec (reading
list item 12) is the Butcher that then sits on top of MLA in production; not covered here.

## 8. Requirements profiles side by side (game stat scale, 0..5)

| Job | latency | encBudget | decBudget | ra | bandwidth | decMem | lossless | priorOK | Packer chair |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Ship a reused KV cache over the network (CacheGen) | 2 (SLO ~1 s) | 3 (offline, once) | 2 (GPU AC decoder, once) | 2 (chunk / group) | 5 | 3 | no | yes (per-model tables) | **full** (arith) |
| Fit decoding in HBM with CPU as backing store (ShadowKV / SpeCache / PQCache) | 0 | 2 (prefill-time SVD / k-means) | 1 | 5 (per token or 8-token chunk) | 4 (PCIe is the channel) | 2 | no (but a lossless copy exists) | yes (per-sequence) | empty |
| Fit decoding in HBM, no offload (DeltaKV / xKV / KIVI) | 0 | 1 | 1 | 5 | 3 | 1 | no | yes (trained MLP) / no | empty |
| Design the model so the cache is small (MLA) | 0 | 0 | 0 | 5 | 3 | 1 | n/a | yes (the model) | empty |

The key teaching contrast: the moment the KV cache leaves the GPU and crosses a *network*, the Packer's
chair fills up (CacheGen is arithmetic-coded), because random access per element stops mattering and
every byte on the wire costs latency. Inside HBM, the Packer's chair stays empty in every method, and
the saving comes from transforms, indices and tiering instead.

## 9. Proposed game content

### 9.1 Fighters (shape of js/data.js; `level: 4`; new domain tag `activations` already exists)

```js
{
  id: 'anchor_delta', name: 'ANCHORMAN', real: 'Anchor-token delta coding (group-of-pictures for KV)', role: 'predict', family: 'linear', level: 4,
  stats: { enc: 0, dec: 0, ratio: 2, latency: 1, mem: 0, ra: 4, lossless: true },
  domains: ['activations', 'audio', 'video'],
  move: 'Hold the Anchor: every 10th token is coded on its own at 8 bit; the other 9 are coded as differences from it, never from each other.',
  usedIn: ['CacheGen (SIGCOMM 2024) KV streaming', 'I-frame + P-frames that all reference the I-frame', 'Parquet delta-against-block-min'],
  why: 'Consecutive tokens in one layer and channel are similar: CacheGen measured 2.4-2.9x lower variance in the deltas than in the raw values. Referencing a fixed anchor instead of the previous token costs a little ratio but makes every token in the group decodable in parallel on a GPU thread, and stops quantization error from accumulating down a chain.',
  weakness: 'Only exploits short-range (10-token) similarity. The anchor must stay high precision or the whole group drifts.',
  persona: 'a stocky news anchor in a sharp suit gripping a glowing ship anchor like a mace, nine translucent copies of himself fanned out behind him, teal and gold',
},
{
  id: 'ref_residual', name: 'DOPPELGANGER', real: 'Nearest-reference residual coding (lossy LZ for vectors)', role: 'predict', family: 'dictionary', level: 4,
  stats: { enc: 2, dec: 1, ratio: 3, latency: 0, mem: 2, ra: 3, lossless: false },
  domains: ['activations', 'vectors'],
  move: 'Find Your Twins: keep every 10th token as a reference, find the 4 nearest references to the new token, predict it as their mean, code only the residual.',
  usedIn: ['DeltaKV (2026) with a trained residual MLP and 4-bit residuals', 'the idea behind MiniCache / CaM token merging'],
  why: 'An LZ77 match is a prediction that came true; here the match is approximate and chosen by L2 distance in key space instead of by position. Only tokens the sparse-attention mask selects are ever reconstructed, so decoder cost follows the attention budget, not the context length. DeltaKV reaches 29% of the original cache on Llama-3.1-8B with LongBench unchanged (50.0 -> 50.2).',
  weakness: 'A trained residual coder mis-predicts out-of-distribution content: retrieval of random IDs dropped 79 -> 58. References themselves stay uncompressed.',
  persona: 'a sly figure in a mirrored hall, four faint look-alikes leaning in to whisper, each slightly different, violet and silver',
},
{
  id: 'svd_lowrank', name: 'PROJECTOR', real: 'Rank-r SVD truncation of the key matrix (per-sequence KLT)', role: 'transform', family: 'rotation', level: 4,
  stats: { enc: 3, dec: 1, ratio: 3, latency: 3, mem: 1, ra: 4, lossless: false },
  domains: ['activations', 'vectors'],
  move: 'Flatten the Keys: SVD the L x 1024 pre-RoPE key matrix once after prefill, keep A (L x 160) and B (160 x 1024), rebuild any row on demand as A[i]·B then apply RoPE.',
  usedIn: ['ShadowKV (ICML 2025): rank 160 keys on GPU, values offloaded', 'xKV: one shared token basis across 4 layers (8x)', 'MLA (DeepSeek-V2) as the trained version'],
  why: 'Pre-RoPE keys of a long prompt live near a low-dimensional subspace; values do not. Truncating at rank 160 of 1024 keeps RULER within 2 points on Llama-3.1-8B. Reconstruction is a tiny matmul the GPU has FLOPs to spare for, so storing factors and recomputing keys beats fetching them over PCIe. Needs the whole prefix before it can fit the basis.',
  weakness: 'Values are not low-rank. Post-RoPE keys are not low-rank. Tokens generated after prefill fall outside the fitted subspace. Dense reconstruction of every row is compute-bound (0.4x at 122K): pair with a selector.',
  persona: 'a lanky projectionist in a vintage cinema booth, a beam of light flattening a stack of film reels into one thin glowing sheet, cyan and amber',
},
{
  id: 'xlayer_svd', name: 'LAYERCAKE', real: 'Cross-layer shared low-rank subspace (grouped-layer SVD)', role: 'transform', family: 'rotation', level: 4,
  stats: { enc: 3, dec: 1, ratio: 4, latency: 3, mem: 1, ra: 4, lossless: false },
  domains: ['activations'],
  move: 'Stack and Slice: concatenate 4 layers side by side, one SVD, one shared L x r token basis and four small r x d layer maps.',
  usedIn: ['xKV (2025): 8.03x on Llama-3.1-8B RULER 64K at 88.5 vs 91.9', 'applies on top of MLA latents (3-3.5x on DeepSeek-Coder-V2-Lite)'],
  why: 'Adjacent layers disagree token by token (low cosine similarity) but agree on which directions matter (high CKA). Sharing the token-side basis across a window of layers is what turns 45.7 (single-layer SVD at 8.4x) into 88.5 at 8.03x. Training-free, computed per prompt in about 4% of prefill time.',
  weakness: 'Prefill-only; generated tokens stay uncompressed. Larger inner dimension than per-layer SVD, so reconstruction is a little slower.',
  persona: 'a pastry-chef brawler balancing a four-tier glowing layer cake on one hand, slicing it with a long knife that leaves one shared bright line through every tier, pink and gold',
},
{
  id: 'spotlight', name: 'SPOTLIGHT', real: 'Top-k attention selector (landmarks, low-bit scores or PQ scores decide which tokens matter)', role: 'perceptual', family: 'attention', level: 4,
  stats: { enc: 0, dec: 1, ratio: 4, latency: 0, mem: 1, ra: 5, lossless: false },
  domains: ['activations'],
  move: 'Light the Few: score the query against a cheap stand-in (chunk means, a 1-bit copy, PQ codes), keep the top 1-2% of tokens exact, let everything else stay coarse or stay on the CPU.',
  usedIn: ['ShadowKV (256 chunks of 8 = 1.56% at 128K)', 'SpeCache (top-64 fp16 pairs prefetched one step ahead)', 'PQCache (top 1/5 via PQ lookup tables)', 'Quest, SparQ, InfLLM, DeepSeek DSA'],
  why: 'The softmax is the audience and it only watches the stage: 0.5% of keys carry 90% of a query\'s attention mass (SpeCache, Llama-3-8B). That is the perceptual model of this domain: distortion on tokens that would have received ~0 weight costs nothing. Which tokens matter changes every step, so eviction (H2O) forgets; selection with a lossless backing copy never does.',
  weakness: 'Needs a cheap, decently accurate score proxy on the GPU and a fast path to the exact data. Hit rates sit around 60%, so each step still fetches new material. Query-dependent: nothing can be decided before the query arrives.',
  persona: 'a theatre stagehand in black swinging a huge spotlight, a crowd of grey figures in the dark and three lit in full colour, yellow beam on indigo',
},
{
  id: 'tiered', name: 'UNDERSTUDY', real: 'Two-tier cache: lossy stand-in on the GPU, lossless original on the CPU, fetch on demand', role: 'prior', family: 'tiered', level: 4,
  stats: { enc: 0, dec: 2, ratio: 4, latency: 1, mem: 5, ra: 5, lossless: true },
  domains: ['activations'],
  move: 'Call the Original: the GPU holds a 1-bit copy, PQ codes or low-rank factors to make decisions; the full fp16 tensor waits in CPU RAM one PCIe hop away and is pulled only where the decision says so.',
  usedIn: ['SpeCache (full copy on CPU, 1-2 bit copy on GPU)', 'PQCache (codes on GPU, KV on CPU)', 'ShadowKV (keys as factors on GPU, values on CPU)', 'FlexGen, InfLLM, vLLM CPU offload'],
  why: 'Side information the decoder already holds, measured in HBM: the CPU copy is out of band. Layered coding, like SVC in video: the base layer is enough to rank, the enhancement layer is fetched on demand. SpeCache restores Mistral-7B from 28.9 to 40.4 LongBench at 0.10x GPU size this way; the price is PCIe traffic (~31 GB/s) which must be hidden behind compute by prefetching.',
  weakness: 'PCIe is 60x slower than HBM; everything depends on overlapping fetch with compute. Non-contiguous gathers are ~5x slower per byte. CPU RAM must hold the whole cache.',
  persona: 'a nervous understudy in half-finished stage makeup holding the script on stage, while the real star in full costume waits in the wings behind a heavy curtain, warm backstage reds',
},
{
  id: 'mla', name: 'BOTTLENECK', real: 'Multi-head Latent Attention: learned low-rank KV latent baked into the model', role: 'transform', family: 'learned', level: 4,
  stats: { enc: 0, dec: 0, ratio: 5, latency: 0, mem: 0, ra: 5, lossless: false },
  domains: ['activations'],
  move: 'Born Small: cache one 512-dim latent plus a 64-dim positional key per token instead of 128 heads x 128 dims x 2; the up-projections are absorbed into W_Q and W_O so nothing is ever expanded.',
  usedIn: ['DeepSeek-V2 / V3 / R1', 'Kimi K2 and other MLA models'],
  why: 'The transform is trained with the model, so it is the optimal linear bottleneck for this network rather than a post-hoc SVD. 576 elements per token per layer vs 32,768 for MHA (56.9x), equal to GQA with 2.25 groups, and DeepSeek reports it is stronger than MHA, not weaker. RoPE cannot pass through a low-rank map, hence the small separate positional key: the one piece of structure that refused to compress.',
  weakness: 'Only available if you pretrained with it. Zero encode cost, zero decode cost, zero choice.',
  persona: 'a slim fighter squeezing through the neck of a giant glass bottle, body stretched into a thin glowing line in the middle and whole on both sides, green glass and white light',
},
```

New family entries needed: `'attention': { role: 'perceptual', name: 'Attention-sparsity models', note: 'The audience is a softmax. It only watches the top-k tokens; the rest can be coarse or far away.' }` and `'tiered': { role: 'prior', name: 'Tiered caches', note: 'A lossless original one hop away, not counted in the fast memory. Out-of-band in space, not in time.' }`. Existing fighters to extend: `pq` add `'activations'` to domains and `'PQCache: 12-bit PQ codes as an attention index (m=2, b=6)'` to usedIn; `blockint` add `'CacheGen anchor tokens (8 bit) / SpeCache 1-bit GPU copy'`; `arith` add `'CacheGen per-(layer,channel) tables, CUDA decoder'`.

### 9.2 Heists

```js
{
  id: 'kvcourier', level: 4, name: 'THE COURIER', codec: 'CacheGen (KV cache streaming)',
  domain: 'activations', payload: 'activations',
  brief: 'A 10k-token document\'s KV cache (1.2 GB in fp16) sits on a storage server. A query arrives at a GPU 3 Gbps away. Get the cache there and decoded faster than recomputing it from text, with less than 2% accuracy loss, and adapt when the link drops to 0.2 Gbps.',
  req: { lossless: false, latency: 2, encBudget: 3, decBudget: 2, ra: 2, bandwidth: 5, decMem: 3, priorOK: true },
  reqNotes: [
    'The channel is a network, not HBM: every byte on the wire is latency. Variable-length codes are welcome here; this is the one KV job where the Packer sits down.',
    'Encode once, offline, per context. Decode once per request on the GPU, pipelined with the download.',
    'Random access only at chunk level (1.5k tokens) so chunks can be sent at different quality levels or recomputed from text.',
    'Early layers are fragile, late layers are not: spend bits where the model is sensitive.',
  ],
  ideal: { predict: ['anchor_delta'], quantize: ['blockint'], entropy: ['arith'], prior: ['tables'] },
  accept: { predict: ['delta'], quantize: ['uniform'], entropy: ['ans', 'huffman'] },
  reveal: 'CacheGen (SIGCOMM 2024): groups of 10 tokens, 8-bit anchor + deltas, bin size 0.5 / 1 / 1.5 for early / middle / late layers, arithmetic coding with one probability table per (layer, channel) profiled offline per model, CUDA decoder with one thread per token. 3.5-4.3x smaller than uniform quantization at matched quality, TTFT 3.2-3.7x lower; falls back to sending text when the link is too slow. Mistral-7B, 9.4k tokens: 622 MB at 8 bit -> 176 MB.',
},
{
  id: 'overflow', level: 4, name: 'THE OVERFLOW', codec: 'ShadowKV / SpeCache / PQCache (tiered sparse KV)',
  domain: 'activations', payload: 'activations',
  brief: 'A 128k-token prompt, batch 48, one A100. The full KV cache does not fit in HBM. Keep something small enough on the GPU to decide which 1-2% of tokens each step needs, pull exactly those from CPU RAM over PCIe, and never let accuracy slip more than 2 points on RULER.',
  req: { lossless: false, latency: 0, encBudget: 2, decBudget: 1, ra: 5, bandwidth: 4, decMem: 2, priorOK: true },
  reqNotes: [
    'Online decode per token step; a one-off SVD or k-means during prefill is allowed.',
    'Random access is per token or per 8-token chunk, on both tiers. Fixed-length representations only; the Packer stays home.',
    'The scarce channel is PCIe (~31 GB/s vs 2 TB/s HBM): the GPU-side stand-in must be good enough to pick the right tokens, and fetches must hide behind compute.',
    'A lossless copy exists on the CPU: nothing is forgotten, unlike eviction. Decoder memory means HBM, the CPU copy is out of band.',
  ],
  ideal: { transform: ['svd_lowrank'], perceptual: ['spotlight'], prior: ['tiered'], entropy: ['none'] },
  accept: { quantize: ['pq', 'blockint'], transform: ['xlayer_svd', 'none'] },
  reveal: 'ShadowKV: rank-160 pre-RoPE keys + chunk-mean landmarks on GPU, values on CPU, 256 chunks of 8 fetched per step, >6x less HBM, 3.04x throughput. SpeCache: 1-bit KIVI copy on GPU, fp16 on CPU, a speculative token guesses next step\'s top-64 and prefetches them. PQCache: 12-bit PQ codes rank tokens with lookup tables, top 1/5 fetched. Three stand-ins, one trick: the softmax only needs the few tokens it will actually weight.',
},
```

Codec entries for Explore mode: `cachegen` (heist kvcourier), `shadowkv`, `specache`, `pqcache` (heist overflow), `mla` (heist kvcache, squad transform: ['mla'] with everything else empty: "the Butcher's chair is empty because the model was trained to need no Butcher; TensorRT-LLM later adds FP4 on top").

Combos: `['svd_lowrank', 'spotlight']` "SHADOW PLAY" (reconstruct only the lit rows: compute-bound dense reconstruction becomes free); `['spotlight', 'tiered']` "FETCH ON DEMAND"; `['anchor_delta', 'arith']` "GROUP OF TOKENS". Anticombos: `['arith']` on heists with ra 5 (variable-length codes break per-token random access, the existing kvcache / weightsvault lesson); `['delta', 'spotlight']` (chain deltas must decode in order, the selector wants row i alone: use the anchor variant).

## 10. Quiz-grade facts

1. CacheGen measured that deltas between consecutive tokens (same layer, same channel) have 2.4-2.9x lower variance than the raw K/V values, and that grouping by (layer, channel) lowers entropy far more than grouping by token position. (CacheGen Sec. 5.1, Figs. 3 and 5)
2. CacheGen codes groups of 10 tokens against a fixed 8-bit anchor, with quantization bins 0.5 / 1 / 1.5 for the early / middle / late third of layers, then arithmetic-codes with one probability table per (layer, channel). Result: 3.5-4.3x smaller than uniform quantization at the same quality. (CacheGen Secs. 5.2, 7.2, App. C.2)
3. Pre-RoPE keys are low-rank, post-RoPE keys and values are not; ShadowKV keeps rank 160 of 1024 and recovers near full-rank accuracy, 6x fewer GPU bytes for the key side. (ShadowKV Sec. 3)
4. ShadowKV fetches 256 chunks of 8 tokens per step at 128K, about 1.56% of the context, and reports an equivalent memory bandwidth of 7.2 TB/s on an A100 whose HBM delivers 2 TB/s. (ShadowKV Sec. 5; the 1.56% is derived)
5. xKV: adjacent layers have low token-wise cosine similarity but high CKA; grouping 4 layers under one SVD basis raises Llama-3.1-8B RULER-64K from 45.71 (single-layer SVD at 8.4x) to 88.50 at 8.03x, vs 91.89 uncompressed. (xKV Table 1, Table 8)
6. xKV compresses an MLA model's already-latent KV cache a further ~3x (RepoBench) to ~3.5x (LCC) on DeepSeek-Coder-V2-Lite. (xKV App. E)
7. SpeCache: 0.5% of keys cover 90% of a query's attention mass on Llama-3-8B/PG19; a 1-bit KIVI copy alone scores 28.9 on LongBench (Mistral-7B) but 40.4 with top-64 fp16 pairs prefetched from CPU, at the same 0.10x GPU footprint (full: 42.3). (SpeCache Sec. 3.1, Table 1)
8. SpeCache decodes a second, speculative token each step purely to predict which KV pairs to prefetch; because decoding is memory-IO bound the extra token is nearly free, and speculative prefetch cuts step latency 877 -> 643 ms vs fetching after the fact. (SpeCache Sec. 3.3, Table 4)
9. PQCache stores 12 bits of PQ code per key (m=2, b=6) instead of 2048 bits of fp16 (1/128 of key memory, derived) and uses them only to rank tokens; fetching the top 1/5 keeps LongBench at 47.29 vs 47.37 full on Llama-3.1-8B. (PQCache Secs. 4, 6)
10. MLA caches 576 elements per token per layer (512 latent + 64 RoPE key) vs 32,768 for 128-head MHA, "equal to GQA with 2.25 groups"; the up-projections are absorbed into W_Q and W_O so the cache is never expanded, and RoPE is the one operation that forced a separate uncompressed key. (DeepSeek-V2 Sec. 2.1, Table 1)

## 11. Side-quest ideas (real toy math)

**A. "Anchor or chain?"** Synthesize a 64-token x 16-channel "KV" matrix: per-channel offset + slow
random walk + small noise (mimics CacheGen Fig. 3). Player picks a predictor: none, previous-token delta
(GHOST), or anchor delta with group size g (ANCHORMAN). Show three live numbers: order-0 entropy of the
quantized symbols (bits/element, with a fixed bin), decode dependency depth (chain: 63 sequential steps;
anchor: 1), and worst-case error after quantization (chain: error random-walks and grows with distance
from the start; anchor: bounded by one step). Debrief: previous-token delta wins ~0.3 bit on entropy, the
anchor wins parallel decode and bounded drift; CacheGen chose the anchor, video codecs chose both (I, P,
B frames). Optional second knob: anchor precision 8 bit vs 4 bit to show the whole group drifting.

**B. "Rank the keys."** Synthesize a key matrix K (256 tokens x 64 dims) = U diag(s) V^T with s decaying
geometrically plus isotropic noise, a query set Q (8 x 64), and a value matrix V with flat singular
values. Slider r = 1..64. Live: storage L·r + r·d vs L·d (crossover around r = 51 for these sizes),
attention-score error ||softmax(QK^T) - softmax(Q K_r^T)||_1, and top-16 overlap between exact and
truncated attention. Then flip the slider onto V instead of K: storage falls the same way but the
reconstruction error does not, because V has no decay (the ShadowKV asymmetry). Debrief: energy
compaction = fast singular value decay; rank truncation is the KLT the Lens chair already teaches, applied
along the token axis. Stretch: two "layers" K1, K2 = K1 R + noise with R a rotation; show that stacking
[K1 K2] and sharing U saves vs. two separate SVDs (the xKV point) even though row-wise cosine(K1_i, K2_i)
is near zero.

(Optional C, if a third fits: "PQ as the index": PQ-code the toy keys with m=2, b=4; compare top-k recall
of PQ-approximated scores vs exact at k = 8, 16, 32; show that recall, not reconstruction MSE, is the
metric that matters here.)

## 12. Flags: what I could not verify or found inconsistent

- Venues: ShadowKV "ICML 2025 spotlight" and PQCache "SIGMOD 2025" are taken from the reading list; the
  arXiv pages I read do not state a venue. SpeCache ICML 2025 (PMLR 267) and CacheGen SIGCOMM 2024 are
  confirmed from the papers themselves. DeltaKV is an unreviewed preprint.
- CacheGen never states a bits-per-element figure for its output; the ~2.3 bits/element above is my
  derivation from 176 MB vs a 128 KB/token fp16 cache for Mistral-7B over ~9.4K tokens.
- CacheGen's per-(layer, channel) AC tables: Sec. 5.2 says they are profiled offline per model, Sec. 6
  says probabilities are obtained "by counting the frequencies of quantized symbols in the KV feature for
  the corresponding context". Both may be true (offline default, per-context refinement); the text does
  not reconcile them.
- ShadowKV: the 1.56% budget is my arithmetic (256 chunks x 8 / 128K); the paper gives K = 256 and chunk
  size 8 but not the percentage. I did not read the last ~15% of the HTML (appendix).
- xKV: the text has internal inconsistencies (Qwen2.5-7B vs 14B in Table 1 vs prose; 3.5x vs 3.6x
  attention speedup; a "1:1.5" key/value rank ratio that is never spelled out). No numeric CKA values are
  given anywhere in the text. MLA accuracy for App. E is only in a figure I could not read.
- DeltaKV: the "<= 10% of tokens reconstructed" figure comes from its related-work framing, not from a
  measured table; the last ~15% of the HTML (appendix) was not read.
- DeepSeek-V2's "93.3% KV reduction vs DeepSeek 67B": per-token element arithmetic with DeepSeek 67B as
  GQA-8 / 95 layers (194,560 elements) vs V2's 60 x 576 = 34,560 gives ~82%, not 93.3%; the paper's
  accounting (possibly bytes, possibly a different 67B head configuration) is not shown in the text I
  read. Treat 93.3% as the paper's claim, 56.9x-vs-MHA as the derived per-layer figure.
- PQCache bits-per-key (12 / 32) are derived from m and b; the paper states m, b and the "1/128", "1/64"
  communication fractions, which agree with the derivation for d = 128.
- Not covered on purpose: KVTC (transform coding, NVIDIA), Coupled Quantization, CommVQ, TensorRT-LLM
  FP4, ESS, HiSparse. They belong to the quantization / systems slices.
