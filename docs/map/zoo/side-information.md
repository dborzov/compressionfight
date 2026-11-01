---
title: "Side information: what the decoder knew before the first byte"
slug: side-information
chair: prior
family: "Side information"
summary: "The compressed stream is half the story; the other half is already installed at the receiver as tables, a dictionary, a codebook, a fitted basis or a trained decoder. Ratio on the wire is bought with bytes at the receiver plus versioning and specificity, and a prior trained on the wrong data decodes correctly and saves exactly nothing."
fighters: [tables, dictionary, codebook, decoder_net, basis, tiered]
related: [lz-family, vector-quantization, learned-transforms, modeling-and-coding]
reading_time: 15
---

## Ten million small messages

Imagine a message-queue team at a payments company in 2016. Call the lead engineer Priya. Every order placed anywhere in the system becomes a JSON event, around 220 bytes, and ten million of them a second flow through her queue into a log that must be kept for seven years. Each event must be readable on its own, because consumers subscribe to arbitrary slices of the stream; she cannot compress the day's log as one file and hand out byte offsets into the middle of a gzip stream. Here is one event:

```
{"event": "order.placed", "version": 3, "customer_id": 29031462, "sku": "SKU-96717",
 "quantity": 6, "currency": "EUR", "amount": 131.55, "warehouse": "FRA-2",
 "ts": "2016-03-07T09:42:41Z", "trace_id": "795dcafaf33fe492"}
```

Her storage bill says: compress these. Her first attempt is the obvious one, gzip each event. The result is the small-message disaster that everyone in this business meets once. Raw DEFLATE turns the 220 bytes into 170; add gzip's 18-byte header and trailer and the "compressed" event is 188 bytes, a saving of 15%. A toy LZ77 of the kind the [LZ family](../lz-family/) article builds does even worse, as we will see in a moment: its output is *larger* than its input.

The reason is not that the data lacks redundancy. Look at the event again: `"event": "order.placed"`, `"version": 3`, `"currency": "EUR"`, the key names, the date prefix, all of it is identical across ten million events. The redundancy is enormous. It just is not *inside* any one message. A compressor that sees one event at a time has, at the moment it reads the first byte, no history to match against, no statistics to code against, nothing. So here is the question this article is about, and it is a question about the decoder, not the encoder:

*What does the decoder know before the first byte arrives? Who shipped that knowledge to it? And what happens when the knowledge was trained on the wrong data?*

## The stream is half the story

Priya tries something that cannot work in production, just to see the ceiling. She concatenates 64 events and compresses them as one stream: 14,120 bytes become 2,520, about 39 bytes per event, five times better than compressing each alone. The redundancy is real and LZ77 can see it, when it is allowed to see across messages. Random access was the only thing in the way.

Then the idea. LZ77 looks for repeats in a window of recent bytes. Nothing in the algorithm says the window has to start empty. What if both the writer and every reader of the queue agreed, once, on a few hundred bytes of typical content, and *pre-loaded the window* with it before compressing or decompressing any event? The matcher would then find `"event": "order.placed", "version": 3, "customer_id": ` in its window on the very first byte of the very first message. Here is that experiment with a 20-line LZ77 whose literals cost one byte, whose matches cost three bytes (two for the distance, one for the length), and whose window is pre-loaded with two earlier events, 440 bytes in all:

```python
m  = b'{"event": "order.placed", "version": 3, "customer_id": 29031462, "sku": "SKU-96717", "quantity": 6, "currency": "EUR", "amount": 131.55, "warehouse": "FRA-2", "ts": "2016-03-07T09:42:41Z", "trace_id": "795dcafaf33fe492"}'
d  = b'{"event": "order.placed", "version": 3, "customer_id": 79534122, "sku": "SKU-91558", "quantity": 5, "currency": "EUR", "amount": 181.24, "warehouse": "FRA-2", "ts": "2016-03-17T07:20:27Z", "trace_id": "0668c437647a9ad8"}'
d += b'{"event": "order.placed", "version": 3, "customer_id": 19170460, "sku": "SKU-64177", "quantity": 6, "currency": "EUR", "amount": 456.94, "warehouse": "AMS-1", "ts": "2016-03-03T10:15:01Z", "trace_id": "2bf9154ba63f96b4"}'
def lz77(data, window=b"", minlen=4):
    buf = window + data; i = len(window); out = []
    while i < len(buf):
        best = (0, 0)
        for j in range(i):                                   # search the whole window (toy speed)
            L = 0
            while i + L < len(buf) and buf[j + L] == buf[i + L]: L += 1
            if L > best[1]: best = (i - j, L)
        if best[1] >= minlen: out.append(("M",) + best); i += best[1]   # (distance, length)
        else: out.append(("L", buf[i])); i += 1
    return out
def cost(tokens):   # 1 byte per literal, 3 per match, 1 flag bit per token
    return sum(1 if t[0] == "L" else 3 for t in tokens) + (len(tokens) + 7) // 8
for name, win in (("empty window        ", b""), ("440-byte dictionary ", d)):
    t = lz77(m, win); print(name, f"{sum(x[0]=='L' for x in t):3d} literals, {sum(x[0]=='M' for x in t):2d} matches ->", cost(t), "bytes of", len(m))
```
```
empty window         171 literals, 11 matches -> 227 bytes of 220
440-byte dictionary   43 literals,  6 matches -> 68 bytes of 220
```

With an empty window the toy coder finds eleven short repeats inside the message, pays three bytes for each, and ends up 7 bytes *bigger* than the input. With 440 bytes of history it finds six long matches, the longest 55 bytes (`{"event": "order.placed", "version": 3, "customer_id": ` in one go), and the output is 68 bytes: 3.2× smaller, with the same algorithm. Real DEFLATE with the same 440 bytes passed as a dictionary goes from 170 to 57.

{{< fig src="img/side-information/dict_size.png" alt="Left: bytes on the wire for one 220-byte JSON event against the size of the dictionary installed at the receiver, for the toy LZ77 (227 with no dictionary, 73 with one previous message, 68 with two, 62 with 32) and for real raw DEFLATE (170 down to 55); horizontal lines mark the message size and the 39 bytes per message achievable by compressing 64 together. Right: DEFLATE output against payload length with fixed versus dynamic Huffman tables; zlib picks fixed below about 150 bytes and dynamic wins from about 192." caption="Left: what the decoder already knows is what the wire does not carry. One previous message as a dictionary gets most of the gain; the rest is diminishing. Right: the cheaper end of the same spectrum, DEFLATE's choice between tables in the spec and tables in the header." >}}

Look at the shape of that curve. One previous message as the dictionary is worth almost everything (73 bytes); two messages get to 68; thirty-two messages, seven kilobytes, get to 62. The gain saturates fast because the shared part of these events, the schema, is small and the per-event part, the IDs and amounts and timestamps, is genuinely new information that no dictionary can know. This is the first aha, and it is a reframing rather than an algorithm:

> **The stream is half the story.** Every decoder holds knowledge it did not receive in the stream, and the field has a different word for that knowledge in every family: *tables* in a prefix coder, *dictionary* in an LZ coder, *codebook* in a vector quantizer, *basis* in a transform coder, *model* or *checkpoint* in a learned codec. They are one stage.

Define it properly. **Side information** is any data the decoder needs to reconstruct the output that is not counted in the compressed stream: it was agreed in advance, distributed out of band, and is assumed present at both ends. The **prior** is the same thing seen from the modelling side: what both ends believe about the data before seeing it. A format is **self-describing** when it carries all of its side information in its own header, which is the "pass-through" of this family: nothing to install, nothing to version, and nothing known before the first byte.

{{< lab id="dictionary" >}}

## The day dictionary #7 was deleted

Priya ships it. zstd, which Facebook released that year with exactly her problem in mind, has a `--train` mode: feed it a few thousand sample events and it runs an algorithm called COVER that selects the segments of the samples most often reused, packs about 100 KB of them into a dictionary, pre-computes entropy-coder tables from the same samples, and stamps the dictionary with a 32-bit ID that is written into every compressed frame. Both the producers and the consumers of the queue load dictionary #7. Compressed events drop to a fifth of their size.

Eight months later, so the story goes, a cleanup script deletes old build artefacts from the object store that happens to also hold the dictionaries. The queue keeps working; everyone has #7 cached in memory. Then a consumer restarts, fails to load the dictionary, and reads the ID in the frame header of an event that it can no longer decode. Seven years of retention policy, and the data for the last eight months is unreadable by any process that was not already running. The dictionary is restored from a developer's laptop. The post-mortem introduces a rule that every compression engineer eventually writes down: **the dictionary is part of the data.**

This is the cost side of the stage, and it is why the algorithm cards in this family list versioning as a weakness rather than a footnote. Everything on the spectrum below buys ratio on the wire with three things at the receiver: bytes installed, a versioning problem, and specificity, meaning a decoder that only understands one kind of data.

## The spectrum

Line the priors up by how much the receiver has to hold, and every family on the map falls somewhere on one axis.

{{< fig src="img/side-information/spectrum.png" alt="Six stations on an arrow from left to right: adaptive (learned from the stream; PPM, CABAC, PAQ), header (DEFLATE dynamic trees, FLAC LPC coefficients), standard tables (DEFLATE fixed block, JPEG Annex K, MP3's 32 tables, mu-law), shared dictionary (zstd --train, Brotli 120 KB, HTTP compression dictionaries), codebook or basis (Faiss PQ, CELP, KVTC PCA at 2.4% of parameters, SpinQuant at 0.26%), trained decoder (SoundStream, EnCodec, LLM plus arithmetic coder); below each, what is installed at the receiver, from zero bytes to 20 MB." caption="The knowledge spectrum. Further right: more installed at the receiver, more to version, more specific to one kind of data, fewer bytes on the wire, and small inputs become compressible at all." >}}

At the left end the decoder knows nothing and learns everything from the stream: **adaptive** models like PPM, CABAC's contexts and PAQ update their probabilities symbol by symbol from decoded history, identically on both sides (see [context models](../context-models/)). Zero bytes installed, nothing to version, best ratio on long inputs, and a cold-start tax: the first kilobyte is coded by a model that has seen nothing, which is precisely Priya's problem in another costume.

One step right, the encoder measures the data and ships what it learned in the **header**: DEFLATE's dynamic blocks send the two Huffman trees before the symbols; FLAC sends its LPC coefficients per block; a JPEG can carry its own quantization tables. This is **semi-static** coding, and it has a crossover. The right-hand panel of the figure above measures it with zlib on the JSON events: below about 150 bytes zlib itself decides the header is not worth sending and emits a fixed-table block instead; dynamic tables only win from about 192 bytes of payload. Headers are bad for tiny inputs for the same reason an empty window is.

Then **standard tables**, constants written into the specification so the stream need not carry them: DEFLATE's fixed Huffman block type, JPEG's Annex K default tables, MP3's set of 32 Huffman tables of which the stream names one by index, the μ-law curve of G.711. A few kilobytes installed in every decoder on earth, versioned with the format itself (which is to say, never), tuned for typical data and therefore never optimal for yours. The standard tables.

{{< lab id="tables" >}}

Then the **shared dictionary**, Priya's answer: data-specific, out of band, with an ID. zstd's trained dictionaries; Brotli's built-in 120 KB dictionary of common web text and HTML fragments, which is a shared dictionary promoted to a standard table; and HTTP's compression-dictionary transport, where a browser that has already downloaded version 41 of a JavaScript bundle tells the server so, and receives version 42 compressed against it. A hundred kilobytes per schema, a versioning problem per schema, 3–5× on small JSON.

Then **codebooks** and **bases**. A [vector quantizer](../vector-quantization/) sends indices; the indices mean nothing without the table they point into. Here is the smallest possible one:

```python
import numpy as np
codebook = np.array([[-0.8, -0.6], [-0.2, 0.7], [0.5, 0.1], [0.9, -0.7]])  # 4 entries: 2 bits per vector
x = np.array([0.42, 0.18])
idx = int(np.argmin(((codebook - x) ** 2).sum(axis=1)))
print("stream carries index", idx, "(2 bits); decoder looks up", codebook[idx], "; error", np.round(codebook[idx] - x, 2))
print("the same 2 bits with no codebook:", format(idx, "02b"), "-> meaningless")
```
```
stream carries index 2 (2 bits); decoder looks up [0.5 0.1] ; error [ 0.08 -0.08]
the same 2 bits with no codebook: 10 -> meaningless
```

Where the codebook lives is the design question of every VQ system, and the answers span the spectrum. In CELP speech codecs it is printed in the standard (a table). In a Faiss index it is trained on the indexed vectors and stored in the index file next to the codes (a dictionary with no ID problem, because the file is the unit). In SoundStream it is a tensor in the model checkpoint. A fitted transform basis is the same object for the transform stage: KVTC's PCA matrices for keys and values, fitted on 160k–200k calibration tokens and shipped with the model, are 2.4% of Llama 3.3 70B's parameters; SpinQuant's learned rotations are 0.26% of the weights. Contrast the DCT, whose basis is defined by a formula and lives in the standard, and a Hadamard rotation, which costs zero bytes for the same reason. The [rotations and rank](../rotations-and-rank/) article covers what the basis does; here the point is only who holds it.

{{< lab id="codebook" >}}

At the far right the prior *is* the decoder. A neural audio codec like SoundStream or EnCodec ships tens of megabytes of weights to both ends; the wire then carries 6 kbps, which is 750 bytes per second, saying only *which* plausible speech, because the decoder already knows what speech sounds like. Twenty megabytes installed to save 15 kilobytes a second against MP3: the exchange rate is absurd until you remember that the weights are paid once and the seconds are paid forever. The same arithmetic makes a large language model plus an arithmetic coder a text compressor at roughly 0.8 bits per character, against zstd's 2.5, with a prior measured in gigabytes ([modeling and coding](../modeling-and-coding/) works through why). The [learned transforms](../learned-transforms/) article is about how such decoders are built; the cost shows up in the algorithm card as a matched pair that only works on the data it was trained on.

That is the second aha, now with all six stations in view:

> **Ratio on the wire is bought with bytes installed at the receiver, plus versioning and specificity.** The spectrum runs adaptive → header → standard tables → shared dictionary → codebook and basis → trained weights, and the smaller your inputs, the further right you must go, because a small input cannot afford to teach its decoder anything.

## What a prior saves, exactly

Priya's team expands into Brazil. The new region's events are the same schema with different values: `"currency": "BRL"`, different warehouse codes, different ID ranges. The producers there load dictionary #7, trained on European traffic. Everything decodes correctly, the IDs match, nothing is corrupt, and the compression ratio is noticeably worse. How much worse, and why, has an exact answer, and it is the sharpest thing in this article.

Strip the situation down to two tiny distributions. The data in the new region produces four symbols with probabilities $P = (0.5, 0.25, 0.125, 0.125)$. The prior, trained elsewhere, believes $Q = (0.9, 0.05, 0.03, 0.02)$. An ideal entropy coder driven by the prior charges $-\log_2 Q(x)$ bits for symbol $x$, so its average cost on data distributed as $P$ is the **cross-entropy**:

$$
H(P, Q) = -\sum_x P(x) \log_2 Q(x) = H(P) + D_{\mathrm{KL}}(P \,\|\, Q), \qquad D_{\mathrm{KL}}(P\,\|\,Q) = \sum_x P(x) \log_2 \frac{P(x)}{Q(x)} .
$$

```python
import numpy as np
P = np.array([0.5, 0.25, 0.125, 0.125])      # what the data actually does
Q = np.array([0.9, 0.05, 0.03, 0.02])        # what the shipped prior believes
H   = -(P * np.log2(P)).sum()
HPQ = -(P * np.log2(Q)).sum()
KL  =  (P * np.log2(P / Q)).sum()
print("H(P)       = %.3f bits/symbol  (floor with the right prior)" % H)
print("H(P,Q)     = %.3f bits/symbol  (what the wrong prior pays)" % HPQ)
print("D_KL(P||Q) = %.3f bits/symbol  = H(P,Q) - H(P): %.3f" % (KL, HPQ - H))
print("per-symbol prices under Q:", np.round(-np.log2(Q), 2), " under P:", -np.log2(P))
```
```
H(P)       = 1.750 bits/symbol  (floor with the right prior)
H(P,Q)     = 2.494 bits/symbol  (what the wrong prior pays)
D_KL(P||Q) = 0.744 bits/symbol  = H(P,Q) - H(P): 0.744
per-symbol prices under Q: [0.15 4.32 5.06 5.64]  under P: [1. 2. 3. 3.]
```

{{< fig src="img/side-information/cross_entropy.png" alt="Left: bars of the price in bits charged for each of four symbols under the right prior (1, 2, 3, 3 bits) and under the shipped prior (0.15, 4.32, 5.06, 5.64 bits); the averages are 1.75 and 2.49 bits per symbol. Right: bits per symbol actually paid as the prior is interpolated from the wrong distribution to the right one, falling from 2.49 to the floor of 1.75, with the shaded gap labelled as the KL divergence and a dotted line at 2 bits for a fixed code with no prior." caption="A prior trained on A and used on B pays H(P) plus the KL divergence, and here that is worse than no prior at all: 2.49 bits against the 2.0 of a plain fixed-length code." >}}

Read the per-symbol prices. The prior thinks the first symbol is overwhelmingly likely and prices it at 0.15 bits; the data produces it only half the time. The prior thinks the fourth symbol is rare and prices it at 5.64 bits; the data produces it one time in eight. The average comes to 2.494 bits per symbol, against a floor of 1.750 with the right prior, and against 2.000 for a plain fixed-length code with no prior at all. The dictionary is not merely useless in Brazil; it is a tax. And the stream is still perfectly decodable, because the ID matched and both ends ran the same tables. Correctness and ratio have come apart.

> **A prior trained on A and used on B pays $H(P,Q) = H(P) + D_{\mathrm{KL}}(P\|Q)$.** The gain of a prior is the information it shares with the data, nothing more; when the data moves, the prior still decodes correctly, it just stops saving anything, and past a point it costs.

The right-hand panel shows the whole relationship: interpolate the prior from the wrong distribution to the right one and the cost slides down the KL divergence to the floor. Every point on the knowledge spectrum has a version of this failure. A JPEG default table on a medical image; Brotli's web-English dictionary on a Chinese page; a Faiss codebook trained on last year's embeddings applied to this year's model; a neural speech codec on music; a KVTC basis on a fine-tuned model. Each is a prior drifting away from its data, and each pays in exactly these units. The practical rule follows directly: train the prior on the data it will see, measure the mismatch (it is one line of NumPy against a sample), and retrain when it grows, which is why every dictionary has an ID and every codebook has a version.

## Side information in space

One more station belongs on the spectrum, and it looks nothing like a table until you see it as one. In the KV-cache systems of 2025, a GPU holds a 1-bit copy, or PQ codes, or low-rank factors of a tensor whose exact, lossless original sits in host DRAM one PCIe hop away. The decoder on the GPU reconstructs what it can from the cheap copy, decides which few entries it actually needs, and *fetches those from the original*. Nothing travelled through a compressed stream; the original was simply there, not counted in the fast memory.

That is side information measured in HBM rather than in bytes on a wire: out of band in **space** rather than in time. The tiered cache algorithm card collects these designs, and the [attention as the audience](../attention-as-audience/) article explains why a cheap copy is enough to decide what to fetch. Here the lesson is only that the stage is the same: a cost at the receiver (host RAM that holds everything, PCIe bandwidth to reach it), bought to make what the fast path carries small.

## Connections

Information theory anticipated all of this in 1973. Slepian and Wolf showed that two correlated sources can be coded separately and decoded jointly at a total rate equal to their joint entropy, as if the encoders had cooperated; Wyner and Ziv extended it to lossy coding with side information at the decoder only. In their language the dictionary, the previous version of a file, the host-RAM copy of a tensor, are all "side information $Y$ available at the decoder", and the rate needed for $X$ drops from $H(X)$ to $H(X|Y)$. The saving, $H(X) - H(X|Y) = I(X;Y)$, is the **mutual information** between the data and what the decoder already knew. The cross-entropy result above is the same fact from the other side: a prior saves exactly the information it shares with the data.

rsync is the purest engineering example. To update a file on a remote machine, the receiver's old copy is the side information; the sender learns which blocks the receiver already has from rolling checksums and transmits only the rest. The "dictionary" is a whole file, on the other machine, that was never shipped at all because it was already there. Deduplicating backup systems extend the same window to terabytes, and git's packfiles delta-code each object against a similar one already in the pack.

And the game's shared-prior stage exists because every task eventually asks the question Priya's post-mortem asked. Can both ends pre-agree on data-specific knowledge? A zip file for strangers cannot, and so DEFLATE is self-describing. A message queue you control can, and so zstd has dictionary IDs. A model you ship with its own kernels can go all the way to a trained decoder. Where a codec sits on the spectrum is an answer to that one question, and it is decided by the job before any algorithm is chosen.

**In the game:** the standard tables (tables), the trained dictionary (dictionary), the VQ codebook (codebook), the learned decoder (decoder_net), the trained basis (basis) and the tiered cache (tiered) fill the shared-prior stage in [the Small JSON messages task](/compressionfight/#heist=logpipe), [the Billion-scale vector search task](/compressionfight/#heist=vectorvault), [the Real-time speech task](/compressionfight/#heist=phonecall), [the KV cache cold storage task](/compressionfight/#heist=coldkv) and [the KV cache offload to host memory task](/compressionfight/#heist=overflow); the synergy 'Trained dictionary + LZ77' and the conflicts 'Trained dictionary without an LZ matcher' and 'Vector quantization without a codebook' are the lessons of this article in card form.
