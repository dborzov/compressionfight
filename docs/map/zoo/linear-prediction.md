---
title: "Guess, then send the miss: linear prediction"
slug: linear-prediction
chair: predict
family: "Linear and spatial predictors"
summary: "A predictor does not change how many numbers you send; it changes their histogram into a spike at zero, and the choice of what to predict from decides whether the decoder can run in parallel."
fighters: [delta, lpc, spatial, anchor_delta]
related: [prefix-codes, block-transforms, motion-compensation, codec-dissections]
reading_time: 15
---

# Guess, then send the miss: linear prediction

*If the next sample is almost the last one, why send it at all?*

## The question

Imagine an engineer in 2015, call her Mara, on the team that keeps a large company's monitoring database alive. Every server reports every metric every sixty seconds; the database ingests tens of millions of points a second and keeps the last day in RAM so that dashboards answer instantly. Each point is a 64-bit timestamp and a 64-bit float. Sixteen bytes a point, times a billion points, does not fit. Mara prints a few rows from one series:

```
1700000000  1700000060  1700000120  1700000180  1700000240  1700000300
```

Every one of those timestamps is 64 bits wide, and every one of them is the previous one plus sixty. She is spending 64 bits a point to transmit a number that anyone could have guessed to the second.

This is the whole question of the predictor's chair. The data is *smooth*: not identical, so a copy will not do ([the LZ family](../lz-family/) finds exact repeats and nothing else), but each value is close to something the decoder already knows. What is the right way to spend bits on a number the receiver can nearly guess?

## Deltas, and deltas of deltas

The first move anyone makes is to send differences.

```python
import numpy as np
np.set_printoptions(linewidth=120)
ts = 1700000000 + 60 * np.arange(6)
d1 = np.diff(ts); d2 = np.diff(d1)
print(ts); print(d1); print(d2)
```
```
[1700000000 1700000060 1700000120 1700000180 1700000240 1700000300]
[60 60 60 60 60]
[0 0 0 0]
```

The five deltas are all 60, and the four deltas of the deltas are all zero. Mara's database does the second thing: it predicts that *the spacing stays the same*, and sends the miss, which is zero for a server that reports on schedule and a small number for one that reported a second late. This is the timestamp scheme of Facebook's Gorilla database, published in 2015, and the paper reports that the overwhelming majority of its timestamps end up costing a single bit: the bit that says "the delta-of-delta was zero". Sixty-four bits became one, and no information was lost, because the decoder adds the miss back to its own guess and gets the exact timestamp.

Pin the vocabulary down on this example before it grows. A **predictor** is a function $\hat{x}[n]$ of the samples the decoder already has, $x[n-1], x[n-2], \dots$. The **residual** is the miss, $e[n] = x[n] - \hat{x}[n]$. The encoder sends $e[n]$; the decoder computes the same $\hat{x}[n]$ from its own past output and recovers $x[n] = \hat{x}[n] + e[n]$ exactly. "Delta coding" is the predictor $\hat{x}[n] = x[n-1]$; "delta-of-delta" is $\hat{x}[n] = 2x[n-1] - x[n-2]$, which is a straight line through the last two points extended one step. Both are **fixed** predictors: no parameters, nothing to send, and both sides agree on the formula in advance.

## Aha 1: the predictor changes the histogram, not the count

Here is the thing to be clear about, because it is the point where the predictor's chair meets the [modeling and coding](../modeling-and-coding/) split. Delta coding sends exactly as many numbers as there were samples. Six timestamps in, one timestamp and five residuals out. The *count* of symbols is untouched. What changed is which symbols they are: six values spread over a 64-bit range became one such value and five copies of `60`, or, one level up, five copies of `0`. The histogram went from flat and wide to a single spike.

A spike is exactly what an entropy coder is for. A [Golomb-Rice code](../prefix-codes/) with a small parameter spends about one bit on a zero; a Huffman code built on the residual histogram does the same; an arithmetic coder does fractionally better. None of them could do anything with the raw timestamps, whose histogram is flat. The predictor saved no bits by itself. It *manufactured skew*, and the coder cashed it in. Every predictor in this article does only that, and the way to judge one is to look at the residual histogram it leaves behind.

{{< fig src="img/linear-prediction/histograms.png" alt="Four histograms of the same 65,534-sample signal: raw, after delta, after a fixed order-2 predictor, after a fitted order-2 predictor, with entropies 10.94, 9.23, 7.94 and 7.36 bits" caption="One resonant signal, four predictors. The sample count is identical in every panel; the order-0 entropy falls from 10.9 to 7.4 bits as the predictor improves." >}}

The figure runs the same 65,534-sample signal through four predictors and reports the order-0 entropy of what is left. Raw: 10.9 bits per sample. Plain delta: 9.2. The line-extrapolating order-2 predictor: 7.9. A predictor fitted to the signal: 7.4. Three and a half bits per sample, a third of the file, for the price of two multiplies per sample. The signal in that figure was chosen to be a resonance, and what a resonance is, and why fitting beats extrapolating, is the next scene.

{{< lab id="delta" >}}

## Aha 2: signals are resonances

Switch heroes: an audio archivist, 2001, trying to store a CD collection losslessly. Delta coding helps a little on music, and the line-extrapolating predictor helps a little more, but neither gets near half the size. Take a tiny piece of a real sound: a plucked string decaying, sampled sixteen times.

```python
n = np.arange(16)
x = np.round(100 * 0.9**n * np.cos(0.6 * n)).astype(int)
print(x)
fixed = x[2:] - (2 * x[1:-1] - x[:-2])            # x̂[n] = 2x[n-1] - x[n-2]
print(fixed, "energy", (fixed**2).sum(), "vs raw", (x[2:]**2).sum())
```
```
[100  74  29 -17 -48 -58 -48 -23   4  25  33  30  17   1 -12 -19]
[-19  -1  15  21  20  15   2  -6 -13 -11 -10  -3   3   6] energy 2137 vs raw 13056
```

The sample is $x[n] = 100 \cdot 0.9^n \cos(0.6n)$, rounded to integers the way a 16-bit sample would be: a tone that rings and dies. The fixed order-2 predictor cuts the energy from 13,056 to 2,137, a factor of six, and the residual still has shape: it is itself a wobbling sinusoid, lagging the signal. The predictor assumed a straight line; the signal is curving. What predictor would assume the right thing?

Ask what the signal *is*. A string, a vocal tract, a guitar body, a room: each is a resonator, a system that, hit once, rings at a frequency and decays. A damped cosine $r^n \cos(\omega n)$ obeys an exact two-term recurrence, which you can check with the cosine addition formula:

$$x[n] = 2r\cos\omega \cdot x[n-1] - r^2 \cdot x[n-2].$$

For $r = 0.9$ and $\omega = 0.6$ that is $x[n] = 1.4856\,x[n-1] - 0.81\,x[n-2]$, *exactly*, for every $n \ge 2$. The straight-line predictor was the special case $r = 1, \omega = 0$. The right predictor for this signal is still a weighted sum of the last two samples; only the weights are different. And we do not need to know $r$ and $\omega$ to find them. We can ask the block of samples which weights fit it best, by least squares:

```python
A = np.column_stack([x[1:-1], x[:-2]]).astype(float)     # rows: (x[n-1], x[n-2])
a, *_ = np.linalg.lstsq(A, x[2:].astype(float), rcond=None)
res = x[2:] - A @ a
print("a =", a.round(4), " theory:", round(2 * 0.9 * np.cos(0.6), 4), -0.81)
print("residual", res.round(2), "energy", (res**2).sum().round(1))
```
```
a = [ 1.4849 -0.8091]  theory: 1.4856 -0.81
residual [ 0.03 -0.19  0.71 -0.48 -0.71  1.35 -0.68  0.45 -0.89  1.23 -0.85  0.03  0.27 -0.37] energy 7.0
```

Fourteen equations, two unknowns, and the fit recovers the resonance's coefficients to three decimals from integer-rounded data. The residual energy is 7.0 against the fixed predictor's 2,137: what is left is the rounding noise we injected and nothing else. The predictor did not merely dent the signal. It explained it.

{{< fig src="img/linear-prediction/resonance.png" alt="Three stem plots: the damped cosine, the fixed order-2 residual (energy 2137) and the fitted residual (energy 7)" caption="The toy signal, the fixed predictor's miss, and the fitted predictor's miss. Two numbers fitted to the block turn a ringing tone into rounding noise." >}}

Now the general statement. A **linear predictor of order $p$** is

$$\hat{x}[n] = \sum_{k=1}^{p} a_k\, x[n-k],$$

and **linear predictive coding** (LPC) chooses the $a_k$ to minimise the residual energy $\sum_n e[n]^2$ over a block. Setting the gradient to zero gives the normal equations $\mathbf{R}\mathbf{a} = \mathbf{r}$, where $\mathbf{R}$ is the $p \times p$ matrix of autocorrelations $R(|i-j|)$ and $\mathbf{r}$ the vector $R(1), \dots, R(p)$. Because $\mathbf{R}$ is Toeplitz, the Levinson-Durbin recursion solves it in $O(p^2)$ instead of $O(p^3)$, building the order-$p$ solution from the order-$(p-1)$ one and producing along the way a set of *reflection coefficients* that are bounded in $[-1, 1]$ and quantise gracefully; that is the form speech codecs ship. The quantity to remember is the **prediction gain** $G = \sigma_x^2 / \sigma_e^2$, the ratio of signal variance to residual variance; at high resolution each factor of 4 in $G$ is one bit per sample saved, the same $6$ dB-per-bit exchange rate as in [rate and distortion](../rate-and-distortion/).

The view that makes LPC more than a curve fit is the *source-filter* model. A signal with a flat residual is one that can be regenerated by driving an **all-pole filter** $1 / (1 - \sum a_k z^{-k})$ with a small, white excitation. The coefficients describe the resonator; the residual describes what hit it. In FLAC, that view is applied at the lossless fidelity contract: per block of 4,096 samples the encoder tries fixed predictors of orders 0 to 4 and fitted predictors up to order 32, quantises the chosen coefficients to integers, predicts from integers so that the decoder reproduces every residual bit-exactly, and sends 2 to 32 coefficients as side information ahead of Rice-coded residuals. In a speech codec the *same* math is applied at a different contract: estimate ten to sixteen coefficients per 20 ms frame, send those, and then send only a coarse description of the excitation, because the ear cares about the resonances (the vowel) far more than about the exact pulse train that excited them. FLAC and CELP are the same chair with different butchers behind it.

{{< lab id="lpc" >}}

## The same move sideways: pixels and their neighbours

Switch the axis from time to space. A pixel is nearly its left neighbour and nearly its upper neighbour. PNG, designed in 1996 for exactly the lossless-graphics job, does not fit predictors; it offers a menu of five fixed ones and lets the encoder pick one per row. Here is a $4 \times 4$ block that is mostly a gradient, with the pixel to the left ($a$), above ($b$) and above-left ($c$) read as zero outside the block, as PNG does at an image edge:

```python
B = np.array([[10, 13, 16, 19],
              [15, 18, 21, 25],
              [20, 23, 27, 29],
              [25, 29, 31, 34]])
left = np.zeros_like(B); left[:, 1:] = B[:, :-1]         # a: pixel to the left (0 outside)
up = np.zeros_like(B);   up[1:, :]   = B[:-1, :]         # b: pixel above
ul = np.zeros_like(B);   ul[1:, 1:]  = B[:-1, :-1]       # c: above-left
def paeth(a, b, c):
    p = a + b - c                                        # the "plane" guess
    pa, pb, pc = abs(p - a), abs(p - b), abs(p - c)
    return a if pa <= pb and pa <= pc else (b if pb <= pc else c)
P = np.vectorize(paeth)(left, up, ul)
for name, pred in (("None", 0 * B), ("Sub", left), ("Up", up), ("Average", (left + up) // 2), ("Paeth", P)):
    print(f"{name:8s} sum|residual| = {np.abs(B - pred).sum():4d}")
print(B - P)
```
```
None     sum|residual| =  355
Sub      sum|residual| =  107
Up       sum|residual| =  119
Average  sum|residual| =  114
Paeth    sum|residual| =   62
[[10  3  3  3]
 [ 5  3  3  4]
 [ 5  3  4  2]
 [ 5  4  2  3]]
```

**Sub** predicts from the left, **Up** from above, **Average** from their mean. **Paeth** (after Alan Paeth, 1991) first computes $a + b - c$, the value the pixel would have if the three neighbours lay on a plane, and then returns whichever of $a$, $b$, $c$ is closest to that plane value. On the gradient, every filter except None collapses the interior to the small step sizes 2 to 4, and Paeth, by picking the neighbour that best continues the local plane, does best, 62 against 107. PNG's reference encoder chooses the filter per row with exactly this heuristic, the smallest sum of absolute residuals, and spends one byte per row to say which it chose. The residual rows then go to DEFLATE, whose Huffman stage sees a spike of small values where there had been a gradient.

Video codecs take the menu idea further. An H.264 intra block chooses among nine prediction directions for $4 \times 4$ blocks (vertical, horizontal, DC, four diagonals, and two in between), HEVC among thirty-five; the encoder tries them, picks the one that leaves the cheapest residual, and sends the mode index. It is a menu of predictors chosen per block, PNG's filter byte with more entries, and the residual it leaves goes to a transform and a quantizer rather than to a lossless coder. The [motion compensation](../motion-compensation/) article is the same chair again with the prediction taken from the previous frame instead of the neighbouring pixels.

{{< lab id="spatial" >}}

## Residuals have a sign; coders want a magnitude

One mechanical detail before the last scene, because every residual coder needs it. Residuals are signed and centred on zero; a Rice code, and the bit-packing in Parquet, expect small *unsigned* integers. The **zigzag mapping** interleaves the two signs so that small magnitude becomes small index:

```python
r = np.array([0, -1, 1, -2, 2, -3, 3])
print(np.where(r >= 0, 2 * r, -2 * r - 1))
```
```
[0 1 2 3 4 5 6]
```

Zero stays zero, $-1$ becomes 1, $+1$ becomes 2, and so on; in two's complement it is the one-liner `(r << 1) ^ (r >> 31)`. FLAC, Protocol Buffers, Parquet's delta encoding and Gorilla all do some version of this before handing residuals to a code that charges by magnitude.

## Aha 3: what you predict from decides what the decoder can do

Back to Mara, but with a 2024 problem. A transformer's KV cache is a long sequence of tokens, and in any one layer and channel, consecutive tokens have similar values. She wants to delta-code them before streaming the cache across a network, and the deltas will be *quantised*, because this is a lossy job. Ten values from one channel:

```python
x = np.array([100, 103, 105, 104, 108, 110, 109, 113, 115, 114]); D = 4
q = lambda v: D * np.round(v / D)                        # uniform quantizer, step 4
open_chain = np.concatenate([[x[0]], x[0] + np.cumsum(q(np.diff(x)))])
closed = [x[0]]
for v in x[1:]: closed.append(closed[-1] + q(v - closed[-1]))
anchor = np.concatenate([[x[0]], x[0] + q(x[1:] - x[0])])
for name, r in (("open-loop chain", open_chain), ("closed-loop chain", np.array(closed)), ("anchor", anchor)):
    print(f"{name:18s} error {(r - x).astype(int)}  max {int(np.abs(r - x).max())}")
print("chain deltas ", np.diff(x), "   anchor deltas", x[1:] - x[0])
```
```
open-loop chain    error [ 0  1 -1  0  0 -2 -1 -1 -3 -2]  max 3
closed-loop chain  error [ 0  1 -1  0  0 -2 -1 -1  1  2]  max 2
anchor             error [ 0  1 -1  0  0 -2 -1 -1  1  2]  max 2
chain deltas  [ 3  2 -1  4  2 -1  4  2 -1]    anchor deltas [ 3  5  4  8 10  9 13 15 14]
```

Three ways to do the obvious thing, and they are not the same.

The **open-loop chain** takes the differences between consecutive *original* values, quantises each with step $\Delta = 4$, and lets the decoder add them up. Each quantised delta is off by up to $\Delta/2 = 2$, and the errors *add*: by the ninth token the reconstruction is off by 3, more than any single quantisation could cause, and over a long run the error is a random walk. The **closed-loop chain** fixes that by having the encoder predict from the previous *reconstructed* value, the same number the decoder will have, so each step's error is a fresh $\pm\Delta/2$ and nothing accumulates; this is what every lossy predictive codec since DPCM has done, and it is why a video encoder contains a full decoder. But look at what it costs: the encoder must now reconstruct token $n$ before it can code token $n+1$, and the decoder must do the same. The chain is sequential at both ends.

The **anchor** scheme keeps the first value exactly and codes the others as quantised differences from *it*. Error bounded at $\Delta/2$, like the closed loop, but with no chain: every token's reconstruction is $x_0 + q(x_n - x_0)$, computable on its own thread without waiting for its neighbour. The price is in the last line of output: the chain's deltas are small (3, 2, −1, 4, …) and the anchor's grow with distance from the anchor (3, 5, 4, 8, 10, …, 14). Over these ten values the order-0 entropy is 1.9 bits for the chain's deltas and 3.2 bits for the anchor's. The anchor pays about a bit per value for parallel decode and bounded error.

{{< fig src="img/linear-prediction/anchor.png" alt="Reconstruction error over 200 tokens: the open-loop chain walks away to 21, the closed-loop chain and the anchor scheme stay within ±2" caption="Two hundred tokens of a wandering channel, step Δ = 4. Open-loop deltas drift to an error of 21; the closed loop and the anchor both stay inside ±2, and only the anchor does so without a sequential decoder." >}}

That is the dial behind a whole row of systems. A video GOP is an anchor scheme in time: an I-frame coded on its own, then frames predicted from it, so that a seek lands on an anchor instead of replaying a chain. CacheGen, streaming KV caches in 2024, groups tokens in tens, stores the first at 8 bits and the other nine as deltas from it, and decodes one token per CUDA thread; it measured 2.4 to 2.9 times lower variance in the deltas than in the raw values, which is the entropy saving, and it chose the anchor over the chain precisely so that the decoder would not be a loop. Parquet's delta encoding against a block minimum is the same choice made for a column store. Predicting from the previous reconstruction gives the smallest residuals and a decoder that must run in order; predicting from an anchor gives slightly larger residuals, a decoder that runs in parallel, and an error that cannot walk away.

{{< lab id="anchor_delta" >}}

## Connections

The residual has an older name in statistics: the **innovation**. A Kalman filter predicts the next state from the current one, observes the measurement, and updates on the difference; the innovation sequence of an optimal filter is white, which is the same statement as "a good predictor leaves a flat residual". An ARIMA model is a linear predictor with a name for each part: AR is exactly the $\sum a_k x[n-k]$ above, I is delta coding (difference until stationary), and MA models the residual's own short memory. The time-series field and the audio-coding field solved the same normal equations under different names: Yule-Walker in one, autocorrelation-method LPC in the other.

The chair is also older than digital audio. Differential PCM, the closed-loop chain with a quantizer, was patented at Bell Labs in 1952 for telephony; the vocoder work of the 1960s and 70s turned LPC into the standard description of speech, and the GSM phone in a 1990s pocket ran an LPC analysis every 20 ms. Today the same code paths live in FLAC, in Opus's SILK layer, in PNG, in Parquet's `DELTA_BINARY_PACKED` columns, and in the delta stage of a KV-cache courier.

And one connection forward: a linear predictor of order $p$ and a $p$-tap decorrelating transform are two answers to the same question, how to make correlated samples independent. [Block transforms](../block-transforms/) do it with a rotation applied to a whole block at once, which is the natural move when the samples are all available together and lossy; prediction does it one sample at a time, which is the natural move when the decoder must stay causal and exact. Video uses both, in that order: predict, then transform the miss.

**In the game:** [GHOST](/compressionfight/#fighter=delta), [PROPHET](/compressionfight/#fighter=lpc), [NEIGHBOR](/compressionfight/#fighter=spatial) and [ANCHORMAN](/compressionfight/#fighter=anchor_delta) hold the Oracle's chair in Studio Master (FLAC), Pixel Perfect (PNG), Prime Time (H.264's intra modes), The Showroom (WebP) and The Courier (CacheGen); the [codec dissections](../codec-dissections/) show where each one sits in its pipeline.
