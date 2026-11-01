---
title: "The anatomy of a compression task: the five axes"
slug: anatomy-of-a-job
chair: foundations
family: "Foundations"
summary: "Why MP3 and a phone call, both 'compressing audio', share almost no design: the five axes that pin down any compression job before an algorithm is chosen, the latency ledger of a call, why random access and variable-length codes cannot coexist, and why Moore's law gets spent on the encoder."
fighters: []
related: [kv-cache-vaults, codec-dissections, side-information, modeling-and-coding, rate-and-distortion]
reading_time: 14
---

## Two desks, one file

Imagine two engineers who never met. Both are invented.

The first, call him Henrik Lund, sits in a consumer-electronics lab in 1997 with a prototype on the bench: a portable music player the size of a cigarette packet, built around a 60 MHz digital signal processor and a battery that must last a commute. His brief is a sentence: *fit a CD onto this at 128 kilobits per second, and make it sound like the CD.* The encoding will happen on a desktop computer the night before. The player only has to decode, in real time, for hours, on a chip that cannot afford a floating-point unit.

The second, call her Mira Castellanos, sits in a video-calling startup in 2012. Her brief is also a sentence: *carry a conversation between two phones over a mobile link that drops one packet in twenty, and make it feel like a conversation.* Both phones encode and decode at once. Nothing may be buffered for longer than a couple of hundredths of a second, because the person on the other end is waiting to reply.

Give both of them the same input file, a ten-second stereo clip, and ask them to compress it. They will build machines that share almost nothing. Henrik's will use long overlapping windows, an elaborate model of hearing, and a Huffman coder reading from a table of thirty-two standard trees. Mira's will use a linear predictor of the vocal tract, frames of twenty milliseconds, a packet format in which every packet can be decoded without the previous one, and a redundant coarse copy of each frame tucked into the next in case it goes missing. Neither design is better. Each would fail at the other's job: Henrik's long windows alone would blow Mira's latency budget before a single byte reached the network, and Mira's packet-loss machinery would be dead weight in a file that is read from a disk.

Both of them "compress audio". So the question this article is about is:

> *Why do MP3 and a phone call, both "compress audio", share almost no design?*

The answer is that "compress audio" was never the job. The job is the input *plus a constraint sheet*, and the constraint sheet has five lines on it. Once you can write those five lines for a new problem, most of the design is already decided, before any algorithm has been named.

## The same ten seconds, three times

Start by pricing the input, because every job begins with the uncompressed size. Ten seconds of CD-quality stereo is 44,100 samples per second, two channels, 16 bits each:

```python
import numpy as np
fs, ch, bits, sec = 44_100, 2, 16, 10
raw = fs * ch * bits * sec                      # the same 10-second stereo clip, three jobs
print(f"raw PCM: {raw/8/1e6:.3f} MB, {fs*ch*bits/1000:.0f} kbit/s")
print(f"MP3 @128 kbit/s : {128_000*sec/8/1e3:6.0f} KB  ratio {raw/(128_000*sec):.1f}:1   frame = 1152 samples = {1152/fs*1e3:.1f} ms")
print(f"FLAC @58%       : {raw*0.58/8/1e3:6.0f} KB  ratio {1/0.58:.2f}:1   block = 4096 samples = {4096/fs*1e3:.1f} ms")
voip_fs, kbps, frame_ms = 16_000, 24, 20       # wideband speech, Opus-class bitrate
print(f"VoIP @{kbps} kbit/s : {kbps*1000*sec/8/1e3:6.0f} KB  ratio {raw/(kbps*1000*sec):.1f}:1   "
      f"frame = {voip_fs*frame_ms//1000} samples = {frame_ms} ms = {kbps*frame_ms/8:.0f} bytes per packet")
# the latency ledger of the call (one direction), in ms
ledger = {"capture buffer (1 frame)": 20, "encoder algorithmic (lookahead+overlap)": 6.5, "encode compute": 3,
          "network propagation": 40, "jitter buffer": 40, "decode compute": 2, "playout buffer (1 frame)": 20}
total = sum(ledger.values()); print(f"mouth-to-ear: {total} ms of the 150 ms budget; "
      f"an MP3-style 1152-sample frame + overlap alone would be {2*1152/fs*1e3:.0f} ms")
```
```
raw PCM: 1.764 MB, 1411 kbit/s
MP3 @128 kbit/s :    160 KB  ratio 11.0:1   frame = 1152 samples = 26.1 ms
FLAC @58%       :   1023 KB  ratio 1.72:1   block = 4096 samples = 92.9 ms
VoIP @24 kbit/s :     30 KB  ratio 58.8:1   frame = 320 samples = 20 ms = 60 bytes per packet
mouth-to-ear: 131.5 ms of the 150 ms budget; an MP3-style 1152-sample frame + overlap alone would be 52 ms
```

Three answers for one file, and the third one, FLAC, is a job neither Henrik nor Mira was given: an archivist who wants the clip back bit for bit and wants to seek to any second of it on a 2005 portable player. 160 KB, 1,023 KB and 30 KB are not three levels of skill. They are three different contracts about what "back" means and who is waiting.

Lay the three constraint sheets side by side. This table is the article; everything after it is commentary on its rows.

| Axis | **Music distribution task** (MP3, 1997) | **Real-time speech task** (VoIP, 2012) | **Lossless audio task** (FLAC) |
|---|---|---|---|
| Fidelity contract, consumer | Lossy; the ear; transparent at ~128–192 kbps | Lossy; intelligibility first, naturalness second | Lossless, MD5-checked; the consumer is a diff |
| Redundancy profile | Full-band music, anything | One voice, band-limited, long silences | Music; local smoothness and stereo correlation only |
| Symmetry | Encode once on a desktop, decode on millions of players | Both directions, both ends, live | Encode once (try eight predictors), decode on a tiny CPU |
| Latency | Irrelevant | Hard: ~20 ms frames, minimal lookahead | Irrelevant |
| Random access, errors | Frame-level seek; reliable file | Per-packet independence; UDP loss | Seek to any block; reliable file |
| Compute envelope | Decoder on a 60 MHz chip, encoder unbounded | Real-time both ways on a phone, on battery | Integer-only decoder, fixed point |
| Where the prior lives | Psychoacoustic model and tables in the spec | Speech priors; neural variants ship a voice model | Nothing shipped; predictor coefficients per block |

{{< fig src="img/anatomy-of-a-job/stat_sheets.png" alt="Three radar charts, one per job, over six axes: latency tolerance, encoder compute, decoder compute, random access, bandwidth pressure, decoder memory; the MP3 job is wide on latency tolerance and encoder compute, the phone call is a narrow hourglass pinned to zero latency and maximum bandwidth pressure, FLAC is wide on random access and latency" caption="The three jobs as the game's task stat sheets, 0 to 5 on each axis. Same clip, three silhouettes. The phone call's hourglass is the shape of a job with no slack: no latency, every bit counts, and nothing may be assumed about the decoder's memory." >}}

Notice what the table does *not* contain: the name of a single algorithm. That is the first thing to take from this article, and it is worth stating as a claim rather than an observation. **The five axes determine most of the design before any algorithm is chosen.** Given the Music distribution task row, you can deduce long transform windows (latency is free, and frequency resolution is what a masking model needs), an expensive encoder-side analysis (one encode, millions of decodes), a table-driven decoder (60 MHz), and standard tables shipped in the spec (nothing can be installed on a player after it leaves the factory). Given the Real-time speech task row, you can deduce short frames, a predictor that exploits "this is a voice", symmetric cost, and per-packet independence with redundancy for loss. You have not picked the MDCT or LPC yet; you have excluded almost everything else.

Let us name the axes properly, because the table used them loosely. A compression **job** is specified by: the **fidelity contract** (lossless, or lossy with a named consumer whose indifference defines what may be discarded); the **redundancy profile** (which kinds of structure the data has: statistical skew, exact repeats, context correlation, local smoothness, cross-channel correlation, perceptual irrelevance, numeric structure); the **access pattern** (encode/decode symmetry, the latency budget, random access and seekability, the error environment, the update pattern); the **compute envelope** (where the decoder runs, its memory window, its throughput tier, its energy budget); and the location of **prior knowledge** (learned from the stream, shipped in a header, a shared dictionary, or baked into trained weights). The [primer](../../primer/) compresses the whole thing into one fill-in-the-blank sentence:

> *Compress data with redundancy profile ⟨R⟩ under fidelity contract ⟨F⟩, with access pattern ⟨A⟩, inside compute envelope ⟨C⟩, with prior knowledge living at ⟨K⟩.*

Four of the five rows deserve a closer look, because each hides an arithmetic that engineers get wrong until they have done it once.

## The latency ledger

Mira's hardest line is latency, and the first thing to understand about it is that it is a *budget*, not a wish. Telephone engineers have known since the days of satellite links that a one-way delay beyond about 150 milliseconds (the figure in ITU-T recommendation G.114) makes two people start talking over each other; the conversation turns into a walkie-talkie exchange. That 150 ms is the whole budget, mouth to ear, and the codec is only one of the people spending it.

{{< fig src="img/anatomy-of-a-job/latency_budget.png" alt="Two stacked horizontal bars of latency components against a dashed 150 ms line: an Opus-class design with 20 ms frames totals 132 ms; an MP3-class design with 1152-sample frames, one frame of overlap and one of lookahead totals 189 ms" caption="One direction of a call, as a ledger. Network propagation and the jitter buffer that absorbs its variation are fixed costs of about 80 ms that no codec can touch. What the codec controls is its frame length, and the frame is paid three times: once to fill the capture buffer, once as algorithmic delay inside the encoder, once to fill the playout buffer. Swap in an MP3-sized frame with its overlap and lookahead and the budget is gone before the packet leaves the phone." >}}

Read the ledger from the right. The network costs what it costs, say 40 ms across a continent, and because packets arrive unevenly the receiver must hold a jitter buffer of a few frames, another 40. That is 80 ms spent before the codec is consulted. What the codec controls is its frame length, and here is the arithmetic that matters: the frame length is paid *three times*. The microphone must fill one frame before the encoder can start; the encoder's own algorithmic delay is at least its lookahead plus, for an overlapping transform, a second frame (the MDCT cannot finish reconstructing frame $t$ until frame $t+1$ has arrived, which is the MDCT algorithm's weakness in the game); and the loudspeaker must hold one frame before it can play. With 20 ms frames and 6.5 ms of lookahead, which adds up to the 26.5 ms algorithmic delay Opus documents for its default frame size, the codec's share is about 50 ms and the total lands at 132. With MP3's 1,152-sample frame at 44.1 kHz, 26 ms, paid three times plus the overlap, the codec alone is 104 ms and the total is 189, over budget with the phone still in your hand.

{{< lab id="mdct" >}}

This is why Mira cannot use Henrik's transform *at Henrik's frame size*, even though the transform is excellent. The constraint is not a tuning parameter. It is structural, and it cuts both ways: at 20 ms frames the transform's frequency resolution is too poor for the fine-grained masking model Henrik relies on, which is why the phone call's codec leans on a predictor of the voice instead. Opus's CELT layer does use an MDCT, shrunk to frames of 2.5 to 20 ms; the transform survived, the job shrank it.

## Random access and the variable-length code

The FLAC archivist's hardest line is "seek to any second", and it looks harmless until you ask what a seek means for a decoder that reads a stream of Huffman codes. A variable-length code has no fixed position for element seven million. To find it you must decode elements one through 6,999,999, because each code's length is only known once it has been read. The same is true of an adaptive context model, whose probabilities at byte $n$ depend on all bytes before it, and of a chain of predictions in which sample $n$ is a delta from sample $n-1$.

The cure is always the same, and it is always paid for. Cut the stream into **independent blocks**: reset the entropy coder's state, start the predictor fresh, and write a table of where each block begins. FLAC does this with blocks of a few thousand samples and a seek table; video does it with keyframes every couple of seconds, so that a seek lands on a frame that was coded without reference to its predecessors; zstd does it with frames; Parquet with row groups; a GPU weight format with blocks of 32 values, each carrying its own scale. In every case the cut costs ratio. The predictor loses its context at each boundary, the LZ window is empty for the first bytes of each block, the entropy coder spends bits describing a fresh table, and the keyframe, coded with no reference to its neighbours, is many times the size of the predicted frames around it. The block size is a dial between ratio and seek granularity, and this is the second thing to take from the article: **every structural convenience is bought with ratio, and the dial is set by the job, not by the data.** The data does not care whether it is cut into blocks of 4,096 or 65,536 samples. The 2005 portable player with its seek bar cares, and it sets the dial.

The extreme end of this axis is where the entropy-coding stage is left empty on purpose. A GPU kernel that multiplies a 4-bit weight by an activation runs ten thousand threads at once, each addressing its own block of the tensor; the thread cannot wait for its nine thousand neighbours to finish decoding a Huffman stream first. So weights on a GPU use fixed-length codes, 4 bits each plus a 16-bit scale per 32, which is 4.5 bits per weight, and accept that no entropy coder will ever squeeze the half bit of skew out of those integers. A billion-vector nearest-neighbour index does the same, 96 fixed bytes per vector, so that a query can scan millions of codes with arithmetic on the codes themselves. The half bit per weight and the skew in the PQ codes are the price of random access, paid knowingly.

## The compute envelope and the throughput tiers

Henrik's hardest line is the 60 MHz chip, and it decides something that is easy to miss: *where* the cleverness lives. MP3's decoder is an inverse transform and a table lookup. Everything expensive, the FFT that runs alongside the transform to find masking thresholds, the iterative search for scale factors that fit the frame into its bit budget, the lookahead that decides when to switch to short blocks for a drum hit, is in the encoder, which runs on a desktop with no deadline. Asymmetry is not a property of the algorithm. It is a property of the job: *encode once, decode many* means every unit of effort moved from decoder to encoder is multiplied by the number of decodes, and the economics of a million players make that number enormous.

The same line also fixes the throughput tier, and it is worth memorising the tiers because they sort the entire lossless toolkit. On a storage or network path, where compression must be cheaper than the I/O it saves, you want gigabytes per second per core and you get LZ4 and zstd at its fastest levels, which is to say a matcher with a cheap or absent entropy coder. For general files you want a few hundred megabytes per second and you get zstd's defaults and gzip. For archives you tolerate a few megabytes per second and you get xz and PPM, with huge windows and adaptive arithmetic coding. For record-setting you tolerate kilobytes per second and you get context mixing and neural models. The data is the same text in every tier; only the compute envelope moved, and it moved the choice of algorithm across the whole map.

## Where the prior lives

The quietest line on the sheet is the last one, and Mira's successors in the 2020s pushed it to an extreme Henrik could not have imagined. A neural speech codec puts tens of megabytes of trained decoder weights on both phones in advance, so that the wire need carry only a few kilobits per second of "which speech"; the [side information article](../side-information/) is about this spectrum, from adaptive models that learn from the stream, through tables in a header, to shared dictionaries, to a decoder that *is* the prior. The axis interacts with the others: Henrik could ship tables in the spec but nothing data-specific, because a player cannot be updated after it leaves the shop; a message-queue team that controls both ends can ship a trained dictionary and win three to five times on tiny messages; a lossless archive for strangers can ship nothing at all and must describe itself.

## Moore's law goes to the encoder

The last thing to take from this article follows from the symmetry axis and explains a pattern that otherwise looks like stagnation. Video codecs have had the same skeleton since H.261 in 1988: partition, predict from the previous frame, transform the residual, quantize, entropy-code. H.264 in 2003, HEVC in 2013, AV1 in 2018 and VVC in 2020 each bought roughly 30 to 50 percent fewer bits at equal quality, and each paid for it with roughly five to ten times the encoder complexity of its predecessor. The decoder barely moved, because the decoder is a 200 mW hardware block in a billion phones and televisions and must not. Every generation of transistors was spent on searching harder for motion vectors, trying more prediction modes, and running rate-distortion optimisation over more candidate decisions, all on the encoder, all once per title, amortised over every viewer. **Encode-once-decode-many means Moore's law is spent on the encoder.** Henrik's night-long encode in 1997 was the first instalment of the same bargain.

## Connections

The game's **task stat sheet** is this article: latency tolerance, encoder compute and decoder compute (the two halves of the compute envelope), random access, bandwidth pressure and decoder memory are the six axes, "lossless required" is the fidelity contract and "side information allowed" is where the shared prior may live. A task is a job written as a constraint sheet, and the pipeline that wins it is the one the sheet already implied.

**Amdahl's law** is the asymmetry axis in a different costume: the speedup you can buy by optimising a stage is bounded by the fraction of total time that stage occupies. For a title decoded a billion times, the encoder is a vanishing fraction of total compute, so it may be made arbitrarily slow; for a backup system that encodes everything once and decodes almost nothing, the opposite holds and the encoder must be greedy.

The axes trade off the way the **CAP** properties of a distributed store do: you may not have the best ratio, random access and zero latency at once, and asking for two of them fixes what you must surrender on the third. The block-size dial, the frame-length dial and the symmetry dial are the same dial seen from three jobs.

## In the game

This article has no algorithm of its own; it is the sheet every algorithm is measured against. Play the General-purpose file archive task, the Music distribution task, the Real-time speech task and the Lossless audio task back to back and watch the same six stats reshape the pipeline; then read the [codec dissections](../codec-dissections/) to see each real codec written as its constraint sheet, and the [KV cache walkthrough](../kv-cache-vaults/) to see one tensor pass through five different sheets without changing a single byte.
