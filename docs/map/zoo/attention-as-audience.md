---
title: "Attention as the audience: the softmax is the reader"
slug: attention-as-audience
chair: perceptual
family: "A transformer as the audience"
summary: "When the consumer of your data is a transformer, the perceptual model is the attention weight: distortion on a token the softmax ignores costs nothing, error on a sink or the recent window is amplified, and the importance signal has to be rebuilt from a proxy because FlashAttention never writes it down."
fighters: [sinkguard, spotlight, evict, tiered]
related: [kv-cache-vaults, block-float-formats, perceptual-models, rotations-and-rank]
reading_time: 16
---

## A context that does not fit

Imagine an inference engineer in the spring of 2024. Call her Mara. Her team serves a chat product on eight GPUs, and a customer has just asked for 128,000-token contexts: whole codebases, whole contracts, pasted in. Mara does the arithmetic on a whiteboard. For the 70-billion-parameter model they run, every token leaves behind 320 KiB of keys and values across the 80 layers; 131,072 tokens of that is 40 GiB, for one user, before a single weight is read. The product manager wants sixteen concurrent users per GPU.

Mara has read the [perceptual models](../perceptual-models/) article. She knows how MP3 gets 11:1 out of audio: not by being clever about bits, but by knowing what the ear cannot hear, and putting the quantization noise there. A loud tone at band 8 raises the threshold of hearing around it, and anything under that curve is free. So she asks the obvious question, and it turns out to be the whole article:

*When the consumer of my data is not an ear or an eye but a transformer, what does it not notice?*

She tries the naive thing first, because it is easy: round every key and value to two bits, the way [block formats](../block-float-formats/) do for weights. The cache drops 8×. The model's answers on grade-school math drop too, badly. This is the quantization stage without a perceptual-model stage: a quantizer with no model of the consumer, spending its error budget uniformly over data the consumer reads anything but uniformly. The MP3 engineers of 1989 would have recognised the mistake immediately. What Mara lacks is the equivalent of the masking curve.

## Measuring the audience

The ear has a masking curve because physiology says so. A transformer has something much simpler, and it is sitting right there in the forward pass. Every layer, every head, every generated token computes a softmax over all cached keys and then takes a weighted average of the cached values:

$$
w_i = \frac{\exp(q \cdot k_i)}{\sum_j \exp(q \cdot k_j)}, \qquad o = \sum_i w_i\, v_i .
$$

Those weights $w_i$ are the audience's attention, literally. Mara logs them for one head, one query, in a 32-token toy context, and finds the pattern that every paper in this field reports in one form or another. Token 0, the beginning-of-sequence marker, gets 0.35 of the mass even though it carries no content. The last four tokens, the ones just written, get another 0.30 between them. The 27 tokens in the middle share the remaining 0.35, about 0.013 each.

{{< fig src="img/attention-as-audience/mass_vs_rank.png" alt="Left: cumulative attention mass versus the fraction of keys kept, ranked by weight, for a synthetic softmax over 4,096 keys; 2.5% of keys carry 90% of the mass in the toy, with the reported SpeCache (0.5% to 90%) and NVIDIA (20% to 80%) points marked. Right: the 32-token toy with a sink at position 0 holding 0.35, a recent window of four tokens holding 0.30, and 27 middle tokens at 0.013 each." caption="The audience watches a tiny part of the stage. Left: a synthetic softmax over 4,096 keys with one sink; the curve is a toy, the two diamonds are reported measurements (SpeCache: 0.5% of keys carry 90% of a query's mass on Llama-3-8B; NVIDIA's blog: roughly 20% of tokens collect 80%). Right: the 32-token toy used throughout this article." >}}

Now redo the 2-bit experiment with a pencil. Suppose 2-bit quantization puts an error of size 0.25 (in whatever units the values live in) on every token's value vector. The error that reaches the output is not 0.25; it is $\sum_i w_i \cdot 0.25$, because the output is a weighted average. With every token quantized that is $0.25 \times 1.0 = 0.25$. Now exempt the sink and the four recent tokens, leaving them in 16-bit:

```python
import numpy as np
w = np.zeros(32); w[0] = 0.35; w[-4:] = 0.075; w[1:-4] = 0.35 / 27   # sums to 1
eps = 0.25                                       # 2-bit error on a value vector
err_all = (w * eps).sum()
exempt = np.zeros(32, bool); exempt[0] = True; exempt[-4:] = True
err_exempt = (w[~exempt] * eps).sum()
print("quantize everything :", err_all)
print("exempt sink + recent:", round(err_exempt, 4), "with", exempt.sum(), "of 32 tokens kept in fp16")
# the R(D) view: minimise sum w_i 2^-b_i for a fixed bit total -> b_i = const + log2(w_i)
bits = 134 / 32 + np.log2(w) - np.log2(w).mean()
print("bits per token the weights 'ask for': sink %.1f, recent %.1f, middle %.1f" % (bits[0], bits[-1], bits[1]))
```
```
quantize everything : 0.25
exempt sink + recent: 0.0875 with 5 of 32 tokens kept in fp16
bits per token the weights 'ask for': sink 8.5, recent 6.3, middle 3.7
```

Five tokens out of thirty-two, 16% of the memory, and the error the audience actually experiences drops by a factor of 2.9. That is the first aha of this article, and it deserves to be stated as plainly as the masking curve was:

> **Distortion on a token the softmax weights at ~0 costs nothing.** Importance replaces masking. The perceptual model of a transformer is its own attention weight, and the places where error is amplified rather than averaged away are the attention sinks and the recent window.

The last three lines of the snippet say something sharper. Treat $w_i$ as a distortion weight and ask [rate-distortion theory](../rate-and-distortion/) how to spend a fixed budget of bits: minimise $\sum_i w_i 2^{-b_i}$ subject to $\sum_i b_i = B$. The Lagrangian gives $b_i = \text{const} + \log_2 w_i$. Bits should follow the logarithm of attention mass: the sink wants about five more bits than a middle token, the recent window about two and a half more. Engineering rounds this to its bluntest form: sinks and the newest tokens skip the quantization stage entirely, everything else gets the low-bit grid. KIVI keeps the newest 128 tokens in fp16, and its ablation on Llama-2-13B at 2 bits puts GSM8K at 12.21 without that window and 20.77 with it. KVQuant keeps token 0 in fp16 and takes 2-bit LLaMA-7B perplexity from 8.47 to 7.23 with that one exemption. KVTC leaves four sink tokens and 128 recent tokens raw. The game calls this algorithm the sink & window exemption.

{{< lab id="sinkguard" >}}

Two words need proper definitions now that the toy has made them concrete. An **attention sink** is a token, almost always the first one or two in the sequence, that receives a large share of attention mass from most queries regardless of content; the softmax needs somewhere to put weight when nothing is relevant, and the model learns to dump it there. The **recent window** is the last few dozen to few hundred positions, which recent queries attend to heavily because language is local. Neither is a quirk of one model; both appear in every decoder-only transformer measured so far, and both are places where the gradient of the output with respect to the stored vector is large.

Why is the sink's error *amplified* rather than merely weighted? Look at where a key error goes. Perturb the key $k_i$ by $\delta$ and the logit changes by $q \cdot \delta$; the weight changes by roughly $w_i (1 - w_i)\, q \cdot \delta$. For a middle token with $w_i = 0.013$ that factor is 0.013. For the sink at 0.35 it is 0.23, eighteen times larger. And because the softmax is normalised, a shift in the sink's weight shifts everyone else's; a wrong sink key is a wrong denominator for the whole row. The sink is where a small error becomes a large one, which is exactly what "masking in reverse" should mean.

## If nobody reads it, why store it?

Mara's measurement raises a more radical thought. If 0.5% of the keys carry 90% of the mass, as SpeCache measured on Llama-3-8B, why keep the other 99.5% in HBM at all? Drop them. This is the token eviction, token **eviction**: a quantizer with exactly two levels, all bits or zero bits, chosen per token by an importance score. Everything from the [rate-and-distortion](../rate-and-distortion/) article still applies; the distortion is just measured in benchmark accuracy instead of mean squared error.

The simplest version needs no scores at all. StreamingLLM keeps the sinks and a recent window and throws away the middle; it runs forever in constant memory, and it forgets everything that was in the middle, which is fine for a chatbot that rambles and fatal for a question about page 40 of the contract. H2O keeps a running sum of the attention each token has received and evicts the lowest. SnapKV looks at a short observation window of queries at the end of prefill, scores every token once, and keeps the top budget; at a 2,048-token budget it gets 34.6% on AIME-24 where the dense model gets 57.1%.

That last number is the problem in one line. The importance of a token is **query-dependent**: the token that matters for the next question is not the token that mattered for the last one. An eviction policy is a bet on the future made with evidence from the past, and once it loses the bet the token is gone. A dropped token at step 1,000 cannot be recalled at step 5,000.

Which suggests the second aha, and it is a systems idea dressed as a compression idea:

> **Selection is not eviction.** If a lossless copy of every token sits one hop away, in host DRAM over PCIe, then the GPU only needs enough to *decide* which tokens this query wants, and can fetch exactly those. Nothing is ever forgotten. The price moves from accuracy to bandwidth.

This is the tiered cache from the [side information](../side-information/) article, side information measured in space rather than time: the full fp16 tensor is out of band, not counted in HBM, and fetched on demand. SpeCache keeps a 1-bit copy of the whole cache on the GPU and the 16-bit original on the CPU; at 0.10× the GPU footprint, Mistral-7B scores 40.4 on LongBench where the 1-bit copy alone scores 28.9 and the full cache 42.3. ShadowKV keeps rank-160 factors of the pre-RoPE keys on the GPU (see [rotations and rank](../rotations-and-rank/)) and the values on the CPU, fetching 256 chunks of 8 tokens per step, about 1.6% of a 128k context. HiSparse, the production version in SGLang, keeps the full cache in pinned host memory and a small per-layer buffer on the GPU.

And the mirror image is just as important: **selection without offload saves compute, not memory.** Sparse attention that picks the top 2,048 tokens per step but keeps the full context in HBM so that it *can* pick has shrunk the per-step read by 60× and the memory by nothing. That is precisely why the offload designs exist, and why the [KV cache](../kv-cache-vaults/) article treats "the warehouse run" as its own job.

{{< lab id="spotlight" >}}

## Pitfall one: FlashAttention leaves no receipts

Every selection or eviction method above needs a score per token. Where do the scores come from? Mara's first instinct is to read them off the attention matrix. The attention matrix does not exist.

FlashAttention, which is how every serious serving stack computes attention, never materialises the $N \times N$ matrix of weights. It streams blocks of keys through on-chip memory, keeps running maxima and sums, and writes only the output. The weights are computed and discarded inside the kernel; writing them to HBM would cost more bandwidth than the attention itself. H2O's reference implementation, which needs cumulative scores, falls back to eager attention and loses more than it saves.

So the importance signal must be **rebuilt from a proxy** that is cheap enough to read in full every step. The proxies in use are a family: chunk means of the keys (ShadowKV's landmarks), a 1-bit or 2-bit copy of the keys (SpeCache), product-quantization codes of the keys with lookup-table scoring (PQCache, 12 bits per key instead of 2,048), or a separately trained small indexer. DeepSeek's Sparse Attention is the last kind: a "lightning indexer" with 64 heads of dimension 128, kept in FP8, scores every cached token and hands the top 2,048 to the real attention. Note what this does to the cache: there are now two caches per token per layer, the big latent that only 2,048 tokens' worth are read per step, and the small indexer key that is read for *all* tokens every step. The indexer is the address book. You compress what it points to; you do not compress the address book.

How good does a proxy have to be? Here are 32 keys in 16 dimensions, a query, and a stand-in that keeps one sign bit per coordinate plus one scale per key:

```python
import numpy as np
rng = np.random.default_rng(0)
K = rng.normal(size=(32, 16)); q = rng.normal(size=16)
s = K @ q                                               # exact scores
K1 = np.sign(K) * np.abs(K).mean(axis=1, keepdims=True)  # 1 bit per coordinate + a scale per key
sp = K1 @ q                                             # proxy scores
exact4 = set(np.argsort(-s)[:4].tolist()); proxy4 = set(np.argsort(-sp)[:4].tolist()); proxy8 = set(np.argsort(-sp)[:8].tolist())
w = np.exp(s - s.max()); w /= w.sum()
print("exact top-4:", sorted(exact4), " proxy top-4:", sorted(proxy4), " overlap:", len(exact4 & proxy4))
print("attention mass in exact top-4: %.2f   in proxy top-4: %.2f" % (w[list(exact4)].sum(), w[list(proxy4)].sum()))
print("exact top-4 inside proxy top-8:", exact4 <= proxy8)
```
```
exact top-4: [3, 15, 18, 26]  proxy top-4: [8, 15, 18, 26]  overlap: 3
attention mass in exact top-4: 0.96   in proxy top-4: 0.69
exact top-4 inside proxy top-8: True
```

{{< fig src="img/attention-as-audience/proxy_topk.png" alt="Scatter of proxy score against exact score for 32 keys; the exact top-4 are filled yellow, the proxy's top-4 and top-8 are ringed in cyan; three of the four exact picks are in the proxy's top-4, and all four are in its top-8." caption="A 1-bit stand-in ranks well enough to be useful and badly enough to need slack. The proxy's top-4 misses the key carrying 0.27 of the mass; its top-8 catches it." >}}

The proxy gets three of four right and misses the one that carried 0.27 of the mass, which is the difference between 0.96 and 0.69 of the audience's attention. But its top-8 contains the exact top-4. The lesson every system in this family learned the hard way: **fetch more than you need, and keep what you fetched.** Consecutive decode steps select mostly the same tokens, so the GPU-side buffer is a cache in the operating-systems sense and the right policy is LRU. HiSparse's measurement on GLM-5.1 at top-k 2,048 makes this quantitative: keep only the current top-k and you miss 30% of the next step's picks; keep an LRU buffer of 4,096 entries and the miss rate is 13.4%; at 8,192 it is 6.7%, against Bélády's clairvoyant optimum of 8.2% at 4,096. This is the third aha, and it is why this article is about the audience and not just about quantizers:

> **The importance signal must be rebuilt from a proxy, because the kernel that knows it does not write it down; and the proxy's mistakes are paid for in bandwidth, so a hot buffer with slack is part of the codec.**

## Pitfall two: memory is allocated in blocks

Mara's eviction experiment produces a second surprise. She drops 75% of the tokens, checks the GPU's memory, and nothing has changed.

Serving systems allocate KV memory in pages: vLLM's default is a block of 16 tokens, and DeepSeek-V4's recipe uses a logical block of 256 positions. A page is freed when it is *empty*. Evicting a token leaves a hole in a page that still holds survivors, and the allocator cannot hand a hole to another request. Here is the toy, 16 tokens in two pages of eight, keeping the sink, two middle tokens and the newest one:

```python
import numpy as np
PAGE = 8
tokens = np.arange(16); page_of = tokens // PAGE
keep = np.array([0, 7, 8, 15])                      # sink, two middle tokens, the newest
print("evicted", 16 - len(keep), "of 16; pages still occupied:", np.unique(page_of[keep]).tolist(), "-> freed 0")
slots = np.arange(len(keep))                         # compaction: move survivors to the front
print("after compaction: copies", int((slots != keep).sum()), "; pages occupied:", np.unique(slots // PAGE).tolist(), "-> freed 1")
```
```
evicted 12 of 16; pages still occupied: [0, 1] -> freed 0
after compaction: copies 3 ; pages occupied: [0] -> freed 1
```

{{< fig src="img/attention-as-audience/paged_grid.png" alt="Top row: 16 tokens in two pages of 8; after evicting 12 both pages remain allocated; after compacting the four survivors into page 0 with three copies, page 1 is free. Bottom row: 256 tokens in 16 pages of 16; evicting 192 by importance frees 0 of 16 pages; compacting the 64 survivors forward frees 12." caption="Paged memory frees nothing until a page is empty. Top: the toy. Bottom: 256 tokens in 16 pages, 75% evicted by a heavy-tailed importance score; zero pages come back until the survivors are physically moved." >}}

At scale this is not a corner case; it is the typical case, because the important tokens are scattered. NVIDIA's blog on the infrastructure problems of KV compression gives the example of evicting 14,400 of 16,000 tokens and watching the 1,600 survivors sit in roughly 1,000 of the original 1,000 blocks: VRAM usage unchanged. Memory comes back only after **compaction**, physically copying survivors into fresh pages, which costs copies and must be scheduled; TriAttention, which scores tokens by pre-RoPE key geometry rather than attention scores and compacts every so often, is the one eviction method in that report that addresses both pitfalls, matching dense AIME-25 accuracy at 10.7× less KV memory on Qwen3-8B. For codec designers the rule that falls out is simple: a quantizer's block should tile the page, and headers belong to pages, not tokens.

{{< lab id="evict" >}}

## The stage, properly

Step back and name what Mara built. In every lossy codec on the map the perceptual-model stage holds a model of the consumer that tells the quantization stage where to be brutal. For audio it is a masking curve derived from physiology; for images, a contrast-sensitivity function; for a neural codec, a trained discriminator. For a transformer it is the **importance** of each cached token, which is, up to the $w_i(1-w_i)$ factor for keys, the attention weight itself. The stage then has three ways to act, and the four algorithms of this family are those three ways plus the thing that makes the second one safe.

It can **exempt**: leave the sinks and the recent window at full precision and quantize the rest (the sink & window exemption, inside KIVI, KVQuant, KVTC). It can **select**: rank tokens by a cheap proxy and spend the bandwidth only on the few the query wants (the top-k selection, inside DSA, ShadowKV, SpeCache, PQCache, Quest). It can **evict**: quantize the unimportant tokens to zero bits (the token eviction, inside StreamingLLM, H2O, SnapKV, TriAttention), and then it must compact. And the tiered cache, the lossless copy one hop away, is what turns selection from a gamble into a cache.

Three constraints from the serving system rule the designs in or out, and they are worth memorising because none of them is about compression. The decoder is a GPU kernel, so whatever the stage leaves behind must be readable by index, which bans variable-length codes inside HBM (the [block formats](../block-float-formats/) article is about this). FlashAttention never writes scores, so the stage must predict importance from the data before the kernel runs. And paged memory frees whole pages, so dropping must be followed by moving.

## Connections

The hot buffer in front of the host copy is a cache, and its policy question is the one operating systems settled in the 1960s. LRU, FIFO and random were all measured by HiSparse (13.4%, 17.2% and 16.1% misses at a 4,096-entry buffer on GLM-5.1), and Bélády's 1966 algorithm, evict the entry whose next use is furthest in the future, gives the unreachable lower bound of 8.2%. SpeCache's trick of decoding a speculative extra token purely to predict next step's top-k is an attempt to approximate Bélády by looking one step into the future; since decoding is memory-bound, the extra token is nearly free.

The whole article is [perceptual coding](../perceptual-models/) with the ear replaced by a softmax. Masking says a loud tone hides quiet neighbours in frequency; importance says a heavy key hides the error on light ones in the weighted sum. The MP3 encoder runs an FFT alongside the MDCT to compute the masking threshold; the KV codec runs an indexer alongside attention to compute importance. Both are encoder-side models the decoder never sees.

Sparse attention itself is a codec decision in disguise. Choosing top-k 2,048 out of 128k is choosing an operating point on a rate-distortion curve whose rate is bytes read per step and whose distortion is benchmark accuracy; DSA's indexer is the perceptual-model stage, the gather is the decoder, and the fact that it reduced compute rather than memory is the kind of thing only the [anatomy of a job](../anatomy-of-a-job/) makes visible. Scalable video coding is the closest classical relative of the tiered designs: a base layer good enough to decide, an enhancement layer fetched where the decision says so.

**In the game:** the sink & window exemption (sinkguard), the top-k selection (spotlight), the token eviction (evict) and the tiered cache (tiered) are the perceptual-model stage and shared-prior stages of the level-5 tasks [the KV cache resident in HBM task](/compressionfight/#heist=kvcache), [the KV cache offload to host memory task](/compressionfight/#heist=overflow) and [the KV token eviction task](/compressionfight/#heist=purge); the synergies 'Top-k selection + tiered cache' and 'Token eviction + top-k selection' and the conflicts 'Token eviction without an importance model' and 'Tiered cache without a selector' are this article's pitfalls in card form.
