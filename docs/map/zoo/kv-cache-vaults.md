---
title: "The five vaults of the KV cache: same bytes, five jobs"
slug: kv-cache-vaults
chair: jobs
family: "Jobs: the KV cache"
summary: "One tensor of keys and values, 40 GiB for a long context, needs five different codecs depending on whether it sits in HBM, crosses PCIe, parks on NVMe, travels over a network or gets burned; bandwidth is the stat, random access is the switch that empties or fills the Packer's chair, and the biggest shrinks happened before any codec, at model-design time."
fighters: [unrope, coupled, spotlight, tiered, waterline, anchor_delta, fp4, evict, mla]
related: [anatomy-of-a-job, attention-as-audience, block-float-formats, rotations-and-rank, side-information]
reading_time: 20
---

## One tensor, 40 GiB

Imagine a serving team in 2025, at a company we will call Lantern. They run a 70-billion-parameter model with 80 layers and 8 key-value heads of 128 dimensions each, in bf16, and a customer wants 128k-token contexts. The lead engineer, Teo, has the whole problem on one line of a whiteboard before the meeting starts:

$$
\text{bytes per token} = 2 \times \text{layers} \times \text{kv\_heads} \times \text{head\_dim} \times 2\ \text{B} = 2 \times 80 \times 8 \times 128 \times 2 = 327{,}680\ \text{B} = 320\ \text{KiB}.
$$

That is what every token leaves behind, once it has been read, for the rest of the conversation: its key and its value in every layer. At 131,072 tokens it is 40 GiB, for one user, with the model's weights not yet counted. Teo's team has one tensor and, over the next six months, will walk it through six rooms of the building: the GPU, the parking lot, cold storage, the courier's van, the furnace, and finally the architects' office. In each room the same bytes get a different codec, and by the end of the walk every chair on [the map](../../) will have been sat in and vacated at least once. The question is the one the [anatomy of a job](../anatomy-of-a-job/) article promised would be answerable: *one tensor; why does it need five different codecs depending on where it sits?*

Before the rooms, the number that rules them all.

## Bandwidth is the stat

Here is what a decode step does. To produce one new token, every layer's attention reads *every* cached key and value, computes the softmax, and takes the weighted sum. Then the next token does it again. The cache is not read once; it is re-read per generated token. So the bytes per token of cache are not a storage cost, they are a time cost, paid every step:

```python
KiB, GiB = 1024, 1024**3
def bytes_per_token(layers, kv_heads, head_dim, nbytes=2): return 2 * layers * kv_heads * head_dim * nbytes
b70 = bytes_per_token(80, 8, 128)                       # Llama-3-70B, bf16
ctx = 131072
cache = b70 * ctx
print("per token: %d B = %d KiB;  128k context: %.0f GiB" % (b70, b70 // KiB, cache / GiB))
print("time to read it once at 4.8 TB/s (H200): %.1f ms  -> the floor on time-per-token, before any weight is read" % (cache / 4.8e12 * 1e3))
print("bandwidth needed for 50 tok/s with this cache: %.1f TB/s per request" % (cache * 50 / 1e12))
free = 141e9 - 70e9                                      # H200 HBM minus 70B weights at 8 bits
print("tokens of cache that fit beside the weights: %d at bf16, %d at 4 bits" % (free / b70, free / (b70 / 4)))
mla = 61 * 576 * 2                                       # DeepSeek-V3: one 576-element latent per layer, shared by K and V
print("DeepSeek-V3 (MLA): %d B = %.1f KiB per token, %.1f GiB at 128k" % (mla, mla / KiB, mla * ctx / GiB))
```
```
per token: 327680 B = 320 KiB;  128k context: 40 GiB
time to read it once at 4.8 TB/s (H200): 8.9 ms  -> the floor on time-per-token, before any weight is read
bandwidth needed for 50 tok/s with this cache: 2.1 TB/s per request
tokens of cache that fit beside the weights: 216674 at bf16, 866699 at 4 bits
DeepSeek-V3 (MLA): 70272 B = 68.6 KiB per token, 8.6 GiB at 128k
```

Nine milliseconds just to stream the cache past the compute units, on a GPU with 4.8 TB/s of memory bandwidth; a single 128k request at 50 tokens per second would want 2.1 TB/s of HBM to itself. This is the first aha, and everything in the five rooms follows from it:

> **Bandwidth is the stat.** Decode re-reads the cache every token, so bytes per token *is* time per token. And because the GPU's memory is shared by every concurrent request, bytes per token also caps how many users fit: capacity is batch size, and batch size is throughput.

{{< fig src="img/kv-cache-vaults/bytes_per_token.png" alt="Horizontal bar chart on a log scale of KV-cache bytes per token: Llama-3-70B with all 64 heads would be 2,560 KiB; Llama-2-7B 512 KiB; Llama-3-70B with GQA 320 KiB; Llama-3-8B 128 KiB; GLM-5.1 about 100 KB (reported); DeepSeek-V3 MLA 68.6 KiB; Llama-3-70B at 2-bit about 60 KiB; DeepSeek-V3.2 FP8 latent 39 KiB; DeepSeek-V4 about 10 KiB (derived from 9.62 GiB per million tokens). A top axis converts to GiB at 131,072 tokens." caption="Bytes per token across models and formats, derived from the public formula and configs except where labelled reported. The two largest drops, 2,560 to 320 KiB and 320 to 69 KiB, are architecture decisions, not codecs." >}}

One more thing the figure shows, and it will matter in the last room. The gap between the top bar and the bars below it is not made by any codec. It is the difference between keeping 64 key-value heads and keeping 8, and between keeping 8 heads and keeping one 576-element latent. Hold that thought.

## Six rooms, six stat sheets

The [anatomy of a job](../anatomy-of-a-job/) article says that five axes pin a compression problem down before any algorithm is chosen: the fidelity contract and its consumer, the redundancy profile, the access pattern (symmetry, latency, random access, errors), the compute envelope, and where the prior lives. The KV cache is the best demonstration of that claim on the whole map, because the data never changes and the stat sheet changes completely from room to room.

{{< fig src="img/kv-cache-vaults/stat_sheets.png" alt="Seven small bar charts side by side, one per job, each showing latency tolerance, encoder budget, decoder budget, random access, bandwidth pressure and decoder memory on a 0 to 5 scale, with lossless and side-information flags: the Architect's Cut (all zero except random access 5, bandwidth 4), the Context Window (random access 5, bandwidth 3, no side info), the One-Bit Vault (random access 5, bandwidth 5, side info allowed), the Warehouse Run (latency 1, encoder 2, random access 5, bandwidth 5, memory 2), the Cold Storage (latency 3, encoder 2, decoder 2, random access 1, bandwidth 5, memory 3), the Courier (latency 2, encoder 3, decoder 2, random access 2, bandwidth 5, memory 3) and the Purge (random access 5, bandwidth 3, no side info)." caption="The game's requirement stats for the KV-cache heists, from js/data.js. Same payload in every column. Watch the random-access row: it is 5 in every room where attention reads the stored form and drops to 1 or 2 the moment the cache is parked or shipped." >}}

Read the random-access row first. In five of the seven columns it is pinned at 5: attention must read any token's key and value by index, inside a GPU kernel, without decoding its neighbours. In two it collapses to 1 or 2. That stat is the switch this article keeps returning to; walk the rooms and watch it flip.

{{< fig src="img/kv-cache-vaults/tiers.png" alt="Horizontal bars on a log time axis showing how long it takes to move the whole 40 GiB cache through each pipe: HBM at 4.8 TB/s, 8.9 ms; NVLink-C2C at 900 GB/s, 48 ms; PCIe Gen5 nominal 64 GB/s, 0.67 s; PCIe with a GPU gather kernel at 37 GB/s, 1.2 s; NVMe at 6 GB/s, 7.2 s; a 10 Gbit/s network, 34 s; PCIe with per-entry cudaMemcpyAsync at 0.79 GB/s, 54 s. Dashed lines mark a 30 ms decode-step deadline and the 3.1 s KVTC reports for recomputing a batch of prefixes on a 12B model." caption="The same 40 GiB through five pipes. Only HBM beats the per-step deadline; everything else must either move a tiny selected fraction per step or move the whole cache once, off the critical path, and that choice is the job." >}}

## Room 1: the GPU (the Context Window)

The cache lives in HBM. New tokens arrive one at a time, each must be encoded the instant it is produced, and attention reads every token every step. Stat sheet: latency 0, encoder budget 0, decoder budget 0, random access 5. The decoder here is not a program; it is the attention kernel, and a GPU thread must find its element by arithmetic on an index. That one fact bans every variable-length code on the map. No Huffman, no arithmetic coder, no ANS, no Rice. The Packer's chair is empty, by law, in this room; the [block formats](../block-float-formats/) article explains why fixed-length codes with a per-block scale are what survive.

So the whole game in HBM is played by the Lens and the Butcher, and the first thing Teo learns is that keys and values are different animals. The key tensor has a handful of channels, fixed per head, whose magnitudes are far larger than the rest and stable from token to token; the value tensor has no such pattern. A quantizer whose scale is shared across the channels of one token is wrecked by the outlier channel; a scale shared across tokens of one channel sees a stationary magnitude. KIVI's ablation on Llama-2-13B at 2 bits says it in one row of a table: keys per-channel and values per-token gives GSM8K 12.21 (20.77 with its fp16 window for the newest 128 tokens, which is the Bouncer from [attention as the audience](../attention-as-audience/)); keys per-token drops to 4.55; put values per-channel and the score is 0.00. The cheapest decision in the field, *which way to slice*, is worth three orders of magnitude.

The second thing he learns is that the model itself has scrambled the keys. RoPE rotates each pair of key channels by an angle that depends on the token's position, so the outlier channel's energy leaks between the two channels of its pair by a different amount at every position, and the per-channel statistics stop being stationary. Three fixes exist, all from the [rotations and rank](../rotations-and-rank/) article. Undo the rotation before quantizing and re-apply it inside the kernel: KVQuant stores pre-RoPE keys and takes 3-bit LLaMA-7B perplexity from 7.05 to 6.23 with that move alone. Rotate queries and keys by the same Hadamard *after* RoPE, which leaves every dot product unchanged and smears the outlier across all 128 channels so one scale fits: QuaRot and SpinQuant. Or build a codebook whose entries commute with the rotation, so decoding and rotating can be swapped: CommVQ.

{{< lab id="unrope" >}}

Those three fixes also show the room's one allowed luxury. A Hadamard costs zero bytes of side information; a calibrated set of per-layer levels (KVQuant) or a trained codebook (Coupled Quantization, CommVQ) is an Inside Man, fine when the serving system owns both ends, and it is what gets the budget from 4 bits down to 1 or 2. That is the One-Bit Vault: the same room with a tighter van. Coupled Quantization quantizes groups of eight pre-RoPE channels together with Fisher-weighted centroids, cashing in the mutual information between channels that a scalar grid pays for twice (see [vector quantization](../vector-quantization/)); at one bit per channel LLaMA-2-13B lands at perplexity 6.55 where one-bit scalar quantization gives 4,709. CommVQ at 2 bits scores 47.98 on LongBench against 48.05 in fp16 and serves LLaMA-3.1-8B at 128k on one RTX 4090. Still no Packer; codebooks of a few megabytes, looked up inside the attention kernel.

The systems win of all this is batch, exactly as the first aha said. KIVI reports 2.6× less peak memory, up to 4× the batch size and 2.35–3.47× the throughput. Nothing got faster per byte; there were fewer bytes.

## Room 2: the parking lot (the Warehouse Run)

Sparse attention arrived in 2025 and, as Teo's team discovers, saved the wrong resource. DeepSeek's Sparse Attention runs a small indexer, 64 heads of 128 dimensions in FP8, over every cached token and lets the real attention read only the top 2,048: at 128k the per-step read of the big latents drops sixty-fold. Memory drops by nothing, because the indexer needs the whole context in HBM to choose from, and the chosen 2,048 change every step. Selection without offload saves compute, not memory.

But it makes a new room possible. If only 2,048 of 131,072 entries per layer are touched per step, the other 129,024 can sit in host DRAM and be fetched on demand over PCIe. The stat sheet changes: latency 1 (the fetch must land inside one layer's attention critical path, about 60 μs per layer on an H200 for GLM-5.2, or be prefetched a few layers ahead), bandwidth 5, and the scarce channel is no longer HBM but a 64 GB/s PCIe link shared by every request on the GPU. Random access is still 5, now at the granularity of a single token's layer-local record: 656 bytes in FlashMLA's FP8 layout, 2,048 of them at arbitrary host addresses per layer, chosen microseconds before the read.

```python
entry, topk, layers = 656, 2048, 61                 # FlashMLA FP8 latent entry; DSA top-k; DeepSeek-V3.2 layers
gather = entry * topk * layers                       # bytes per generated token per request if every pick misses
print("all-miss gather per token: %.1f MB" % (gather / 1e6))
for name, bw in (("GPU gather kernel over pinned memory", 37e9), ("cudaMemcpyAsync per entry", 0.79e9)):
    print("  %-38s %7.2f ms" % (name + ":", gather / bw * 1e3))
glm = 200e6                                          # GLM-5.1: ~200 MB per token all-miss (HiSparse)
for buf, miss in ((2048, 0.30), (4096, 0.134), (8192, 0.067)):
    print("  LRU buffer %5d entries: %4.1f%% misses -> %5.1f MB/token = %4.2f GB/s per request at 30 ms/token" % (buf, miss * 100, glm * miss / 1e6, glm * miss / 0.03 / 1e9))
```
```
all-miss gather per token: 82.0 MB
  GPU gather kernel over pinned memory:     2.21 ms
  cudaMemcpyAsync per entry:              103.74 ms
  LRU buffer  2048 entries: 30.0% misses ->  60.0 MB/token = 2.00 GB/s per request at 30 ms/token
  LRU buffer  4096 entries: 13.4% misses ->  26.8 MB/token = 0.89 GB/s per request at 30 ms/token
  LRU buffer  8192 entries:  6.7% misses ->  13.4 MB/token = 0.45 GB/s per request at 30 ms/token
```

Two things in that output decide whether the room exists at all. The first is the difference between 2 ms and 104 ms for the same bytes: ESS measured 37 GB/s when a GPU kernel does address-based loads from pinned host memory and 0.79 GB/s when a copy engine is asked to move 656-byte entries one `cudaMemcpyAsync` at a time. Scattered small transfers are a different animal from bulk copies, and a design that cannot gather in one kernel per layer is not slow, it is impossible. The second is the miss rate. Consecutive steps select mostly the same tokens, so the GPU-side buffer is a cache and LRU is the right policy: HiSparse measured 30% misses keeping only the current top-k, 13.4% with a 4,096-entry buffer, 6.7% at 8,192, against Bélády's clairvoyant 8.2% at 4,096. With the buffer the real draw is about a gigabyte per second per request, and HiSparse gets 2.1–4.7× throughput at 32k–200k contexts with roughly 0.4 GB of GPU memory per 128k request instead of 13 GB.

Notice which chairs are occupied. The Butcher is light (FP8, or a block format on the host copy); there is still no Packer, because a gather of 2,048 entries by index is random access in its purest form. The work is done by the Con Artist, the Spotlight that ranks tokens with a proxy, and by the Inside Man in its strangest costume: the Understudy, a lossless copy one hop away. ShadowKV keeps rank-160 factors of the pre-RoPE keys on the GPU and the values on the CPU, recomputing keys instead of fetching them; SpeCache keeps a 1-bit copy on the GPU and decodes a speculative extra token each step purely to guess next step's top-64. The [side information](../side-information/) article calls this out-of-band in space, and the [attention as the audience](../attention-as-audience/) article explains why a 1-bit stand-in is enough to decide.

{{< lab id="tiered" >}}

## Room 3: cold storage (the Cold Storage)

A chat agent pauses. The user will come back in twenty minutes, or never. Teo can recompute the 32k-token prefix when they return, which costs a prefill, or park its cache somewhere cheaper than HBM and reload it. The stat sheet turns inside out. Latency 3: a few hundred milliseconds to pack and unpack is fine. Encoder budget 2: the whole cache is on the table, so look-ahead across tokens, per-layer statistics and a transform fitted to this model are all allowed. Decoder budget 2, because it runs once per prefix, not once per step. And random access drops to 1: nobody attends over the stored form; it is unpacked back into bf16 pages before use.

That last change is the second aha of the article, and it is the switch the stat sheets were pointing at:

> **The Packer's chair is empty inside HBM in every method, and it fills the moment the cache is stored or shipped.** Random access is the switch. Where attention reads the stored form by index, fixed-length codes only; where the cache is decoded once in bulk, variable-length codes are back, and so is every chair they bring with them.

KVTC, NVIDIA's "transform coding for KV", is what the full classical pipeline looks like on this payload, and it is literally JPEG's recipe with a machine as the viewer. Undo RoPE. Fit one PCA basis per model for keys and one for values, concatenating all layers and heads (32,768 dimensions for Llama 3.1 8B) over 160k–200k calibration tokens, 86 seconds of randomized SVD; the basis is the Blueprint, 2.4% of Llama 3.3 70B's parameters, and the ablation that spans the basis over 1, 2, 4, 8, 16 or 32 layers takes GSM8K at 8× from 27.5 to 56.9, which is the cross-layer redundancy of [rotations and rank](../rotations-and-rank/) made measurable. Then allocate bits across the PCA coordinates by dynamic programming to minimise squared error for a global budget: reverse water-filling from [rate and distortion](../rate-and-distortion/), with groups of coordinates sharing a 16-bit scale and many trailing coordinates getting zero bits. Keep four sink tokens and the newest 128 raw. Quantize uniformly. Then, because this room allows it, DEFLATE on the GPU via nvCOMP: 1.23× on top of the quantizer, from the many near-zero low-bit symbols the allocator leaves behind, and their ablation finds Zstandard does slightly better still. Net: 18–22× on Llama 3.1 8B with GSM8K at 56.9 against 56.8 uncompressed; on a 12B model, time to first token is 380 ms by decompression against 3,098 ms by recompute.

{{< lab id="waterline" >}}

The production shortcut in the same room skips the Lens and the Packer and keeps only the Butcher: TensorRT-LLM encodes a KV page to NVFP4, four-bit floats with an FP8 scale per 16 values, 4.5 bits per element, as the page migrates from the GPU to host memory or disk, and decodes it back to the runtime dtype when it returns; attention never consumes the cold form. No transform, no entropy coder, about 3.5× against fp16, and tellingly the docs keep the RoPE part of the key at hot precision for the newest models, which is the KVQuant lesson surfacing as a config flag.

## Room 4: the courier's van (the Courier)

Now the cache has to cross a network. A 10k-token document's cache, 1.2 GB in fp16 for a 7B model, sits on a storage server, and a query arrives at a GPU 3 Gbit/s away. Sending the text and recomputing is the baseline; the codec exists to beat it. The stat sheet is the cold-storage sheet with the van made smaller: every byte on the wire is latency, so bandwidth is 5 and encoder budget is 3, because encoding happens once per context, offline, and decoding once per request, pipelined with the download. Random access 2, at the granularity of 1.5k-token chunks that can be sent at different quality levels or, if the link is too slow for even the lowest, sent as text to be recomputed.

CacheGen is the one KV codec with a real bitstream, and its squad is a video codec's. Tokens are grouped in tens; the first is an anchor coded on its own at 8 bits, the other nine are coded as deltas from the anchor, never from each other, so that every token in a group decodes on its own GPU thread and quantization error cannot accumulate down a chain. That is a group of pictures with one I-frame and nine P-frames all referencing it, the anchor variant of [linear prediction](../linear-prediction/) and the GOP dial of [motion compensation](../motion-compensation/), chosen for parallel decode rather than for ratio; they measured 2.4–2.9× lower variance in the deltas than in the raw values. The quantizer's bin size is 0.5, 1 and 1.5 for the early, middle and late third of the layers, because rounding noise in shallow layers hurts accuracy far more than the same noise deep in the network: a crude rate allocation by loss sensitivity. Then an arithmetic coder, with one probability table per (layer, channel) profiled offline per model, which is the Rulebook shipped once and the Fencer of [arithmetic and ANS](../arithmetic-and-ans/) doing the cashing-in; grouped tables cut the bitstream by up to 53% against one global table. Result on Mistral-7B with 9.4k tokens: 622 MB at 8 bits becomes 176 MB, roughly 2.3 bits per element by my arithmetic; across models, 3.5–4.3× smaller than uniform quantization at matched quality and 3.2–3.7× lower time to first token.

## Room 5: the furnace (the Purge)

The cheapest codec is the one that keeps nothing. Roughly 20% of tokens collect 80% of the attention mass, so drop the rest: a quantizer with two levels, all bits or zero, chosen per token. The stat sheet looks like the Context Window's again (latency 0, random access 5, no side information), with the distortion measured in benchmark accuracy: SnapKV at a 2,048-token budget scores 34.6% on AIME-24 where the dense model scores 57.1%.

The [attention as the audience](../attention-as-audience/) article is about this room, so only the two ambushes need naming here. The importance scores an eviction policy wants are never written down, because FlashAttention does not materialise the attention matrix, so the policy must rebuild them from an observation window (SnapKV), from key geometry (TriAttention) or give up on scores entirely (StreamingLLM's sinks plus window). And paged memory frees a 16-token block only when every token in it is gone; evict 14,400 of 16,000 tokens and the 1,600 survivors can still occupy 1,000 blocks, so memory comes back only after compaction, which costs copies. TriAttention, which does both, matches dense AIME-25 accuracy at 10.7× less KV memory. The Reaper is a legitimate fighter; it just needs a list and a broom.

## Room 6: the architects' office (the Architect's Cut)

Teo's last meeting is with the people who trained the model, and it rearranges his sense of the previous five rooms. Go back to the bytes-per-token figure. The model they serve has 64 query heads but only 8 key-value heads, because the architects chose grouped-query attention before pretraining; with 64 KV heads the cache would be 2.5 MiB per token and the 128k context 320 GiB. That is an 8× that no codec in the five rooms came close to, and it cost nothing at inference: fewer things to store, rather than smaller things.

DeepSeek-V2's Multi-head Latent Attention goes further. Instead of keys and values for every head, each token caches one 512-element latent plus a 64-element positional key, 576 numbers per layer, and the up-projections that would turn the latent back into 128 heads of keys and values are absorbed into the query and output matrices, so the latent is attended to directly and never expanded. Against 128-head MHA that is 32,768 elements per layer reduced to 576, 57×, "equal to GQA with 2.25 groups"; DeepSeek-V3's 61 layers cost 68.6 KiB per token, a fifth of the 70B model's, with ten times the parameters. The one piece of structure that refused to compress is RoPE, which cannot pass through a low-rank map and so gets its own small key, the same obstacle the Rewind fighter meets in Room 1 from the other side. And DeepSeek-V4 compresses the token axis itself: 30 layers cache a learned weighted sum of eight tokens at stride four, 31 layers one entry per 128 tokens, with a 128-token sliding window kept raw; 9.62 GiB per million-token sequence against 83.9 GiB for a V3.2-style stack, 8.7×.

```python
mha, gqa8, mla = 2 * 128 * 128, 2 * 8 * 128, 512 + 64     # elements per token per layer
print("MHA 128 heads: %5d   GQA 8 heads: %5d (%.0fx)   MLA latent: %3d (%.1fx vs MHA, %.1fx vs GQA-8)" % (mha, gqa8, mha / gqa8, mla, mha / mla, gqa8 / mla))
print("GQA 'groups' equivalent to MLA: %.2f heads" % (mla / (2 * 128)))
print("DeepSeek-V4 vs V3.2 per 1M tokens: %.1f GiB vs %.1f GiB = %.1fx" % (9.62, 83.9, 83.9 / 9.62))
```
```
MHA 128 heads: 32768   GQA 8 heads:  2048 (16x)   MLA latent: 576 (56.9x vs MHA, 3.6x vs GQA-8)
GQA 'groups' equivalent to MLA: 2.25 heads
DeepSeek-V4 vs V3.2 per 1M tokens: 9.6 GiB vs 83.9 GiB = 8.7x
```

In chair terms MLA is a [learned transform](../learned-transforms/), a linear bottleneck trained with the model rather than fitted afterwards by SVD, with the "decoder" folded into the weights; CSA and HCA are the same move along the token axis. The stat sheet is all zeros except random access, because nothing runs at inference and nothing can be changed afterwards. Third aha:

> **The biggest shrinks happened at model-design time** (GQA 8×, MLA several times more, V4 nearly 9× again), by people who never thought of themselves as writing a codec. The five codec rooms work on what the architects left.

## The walk, in one table of prose

Put the six rooms side by side and the chairs sort themselves. In HBM, the Lens (undo RoPE, a Hadamard, a PCA) and the Butcher (block integers, NVFP4, coupled or commuting codebooks) do everything, the Con Artist exempts the sinks and the recent window, the Inside Man is allowed only if the serving system owns both ends, and the Packer is forbidden. In the parking lot the Butcher goes light, the Con Artist becomes a ranking proxy that is read in full every step and must never be compressed, and the Inside Man is a lossless copy in host RAM. In cold storage and in the van the Packer sits down, the Lens gets a fitted basis, the Butcher gets a water-filled bit allocation, the Oracle gets an anchor predictor, and the Inside Man ships bases and probability tables with the model. In the furnace the Butcher has two levels and the Con Artist is the only chair that matters. In the architects' office there is only the Lens, and it was trained.

Two constraints never change across the rooms, and they are not about compression at all: FlashAttention never writes attention scores to HBM, so any importance model must predict from the data before the kernel runs; and paged allocators hand out memory in blocks of 16 or 256 tokens, so a quantizer's block should tile the page and headers belong to pages, not tokens. A codec that violates either is not a worse codec; it does not run.

## Connections

Video has walked this corridor before. A GOP is an anchor group; scalable video coding is a base layer good enough to decide plus an enhancement layer fetched where needed, which is the Understudy; low-latency call configurations that ban B-frames and refresh intra blocks gradually are the Context Window's "encode each token the instant it arrives" rule. CacheGen's authors say so themselves, naming I-frames and P-frames in the paper.

Database buffer pools and CPU caches are the parking lot without the softmax. The hot-buffer size against miss rate, LRU against Bélády, prefetching a few layers ahead: every one of these is a 1970s operating-systems result with a transformer attached.

And JPEG is KVTC's direct ancestor: a decorrelating transform, a quantization matrix that spends bits where the variance is, an entropy coder last. Thirty-four years apart, the same three chairs in the same order, because the stat sheet of "store it, decode it once, every byte is money" has not changed. What changed is the viewer. For JPEG it was a retina with a contrast-sensitivity curve; for the KV cache it is a softmax with an attention curve, and the one chair that had to be reinvented from scratch was the Con Artist.

**In the game:** this article is the whole of level 5, The Vault: [The Context Window](/compressionfight/#heist=kvcache) and [The One-Bit Vault](/compressionfight/#heist=onebit) (Rewind, Siamese, Quarter, Bouncer), [The Warehouse Run](/compressionfight/#heist=overflow) (Spotlight, Understudy, Projector), [The Cold Storage](/compressionfight/#heist=coldkv) (Blueprint, Waterline, Echo and Woodcutter), [The Courier](/compressionfight/#heist=kvcourier) (Anchorman, Fencer, Rulebook), [The Purge](/compressionfight/#heist=purge) (Reaper) and [The Architect's Cut](/compressionfight/#heist=bornsmall) (Bottleneck); the Fixer's line that ends the level is the third aha: you work on what the architects left you.
