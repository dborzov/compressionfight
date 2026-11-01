---
title: "Block transforms: which directions matter"
slug: block-transforms
chair: transform
family: "Frequency transforms"
summary: "The DCT turns 64 pixels into 64 coefficients and saves nothing; the reason it is the most important move in lossy coding is that it moves the energy, and an orthonormal move lets you throw away what is left for free."
fighters: [dct, mdct, dwt]
related: [rate-and-distortion, rotations-and-rank, scalar-quantization, perceptual-models]
reading_time: 14
---

## A transform that saves nothing

In 1972 Nasir Ahmed, an electrical engineer at Kansas State University, asked the National Science Foundation to fund a small idea: a cosine-based transform for compressing pictures. The reviewers were unimpressed and declined. He did the work anyway, with T. Natarajan and K. R. Rao, and published it in January 1974. Today that transform runs inside every JPEG, every video call, every streamed film, and in integer form inside the silicon of every phone.

Here is what makes the idea hard to sell, then and now. The discrete cosine transform takes a block of 64 pixel values and returns 64 other numbers. Not 63. Not 10. Sixty-four real numbers in, sixty-four real numbers out, and the operation is exactly invertible, so by any honest accounting no information was removed and no bits were saved. A reviewer in 1972 could reasonably say: you have rotated the data. So what?

So the question this article has to answer is the reviewer's question. *Sixty-four numbers in, sixty-four numbers out, nothing saved; why is this the most important move in lossy coding?* Hold on to the fact that it is a fair question. The answer has three parts, and each one is an idea you can carry far beyond images.

## Eight pixels from a photograph

Take one row of eight pixels from a grey photograph of something smooth, say a cheek or a patch of sky with a cloud edge arriving at the right:

```
x = [52, 55, 61, 66, 70, 61, 64, 73]
```

Eight numbers between 0 and 255, eight bits each, 64 bits total. Suppose we wanted to store this row with fewer bits and did not mind a little error. The obvious thing is to quantize the pixels themselves: round each to the nearest multiple of 8, say. Every pixel then carries about the same error, roughly 8²/12 ≈ 5.3 of mean squared error (the [rate and distortion](../rate-and-distortion/) article derives that Δ²/12), and we have saved three bits per pixel. There is no way to do better with pixels, because every pixel looks like every other: each one is a number near 60 with a few units of wobble.

Now run the row through the DCT-II instead. The transform is a matrix $C$ whose rows are sampled cosines of increasing frequency, scaled so that $C$ is orthonormal, which means $C^{\top} C = I$ and the inverse is simply the transpose.

```python
import numpy as np
N = 8
n = np.arange(N)
C = np.cos(np.pi * (2*n[None, :] + 1) * n[:, None] / (2*N)) * np.sqrt(2/N)
C[0] /= np.sqrt(2)                       # the DC row gets the 1/sqrt(2) factor
print("orthonormal:", np.allclose(C @ C.T, np.eye(N)))

x = np.array([52, 55, 61, 66, 70, 61, 64, 73], float)
X = C @ x                                # forward DCT-II
print("X =", X.round(2))
print("energy in / out:", (x**2).sum(), (X**2).sum().round(6))

keep = np.zeros(N); big = np.argsort(-abs(X))[:3]; keep[big] = X[big]
x_hat = C.T @ keep                       # inverse DCT = transpose
print("kept", sorted(big.tolist()), "->", x_hat.round(2))
print("MSE pixels:", ((x - x_hat)**2).mean().round(4),
      " MSE coefficients:", ((X - keep)**2).mean().round(4))
```
```
orthonormal: True
X = [177.48 -14.43  -5.66  -6.74   7.07  -3.08  -0.72   2.41]
energy in / out: 31852.0 31852.0
kept [0, 1, 4] -> [58.17 54.25 56.24 63.84 66.66 64.26 66.25 72.33]
MSE pixels: 11.6592  MSE coefficients: 11.6592
```

Look at the eight coefficients. One of them is 177.48. The next largest is −14.43. The other six are all smaller than 8 in magnitude, and three of them are smaller than 4. The eight pixels were all alike; the eight coefficients are wildly unalike. The first coefficient, $X_0$, is $\sqrt{8}$ times the mean of the row (177.48 / 2.83 = 62.75, the average pixel), and it is called the DC coefficient. The rest are AC coefficients, and they measure how much of each cosine shape is present in the wobble around the mean.

The script then does something a pixel quantizer could never do: it keeps three of the eight numbers and sets the other five to zero. The reconstruction is off by a mean squared error of 11.66. Three numbers instead of eight, at an error comparable to the pixel quantizer that saved three bits per pixel. And we have not even quantized the three survivors yet.

{{< fig src="img/block-transforms/row.png" alt="The eight-sample row and its reconstruction from three DCT coefficients, beside the bar chart of all eight coefficients" caption="The row [52, 55, 61, 66, 70, 61, 64, 73] through the DCT-II. One coefficient carries the mean, one carries the overall tilt, and the other six are small. Zeroing five of them costs an MSE of 11.66, and that number is the same whether you measure it in pixels or in coefficients." >}}

## Aha one: Parseval, or why error in one domain is error in the other

Look again at the last line of output. The MSE measured in the pixel domain and the MSE measured in the coefficient domain are the same number, 11.6592, to four decimals. That is not a coincidence of this row. It is the whole reason transform coding works, so let us ask the question properly.

When we zero a coefficient we commit an error in the transform domain. The consumer of the picture never sees the transform domain; they see pixels. What does a transform-domain error of size $e$ look like once we invert the transform? For an arbitrary matrix the answer could be anything: a small change in coefficients could blow up into a large change in pixels, or shrink to nothing, depending on the direction. But an orthonormal matrix is a rotation (possibly with reflections), and rotations preserve lengths. If $\hat X = X + e$ then $\hat x = C^{\top}\hat X = x + C^{\top} e$, and

$$\|\hat x - x\|^2 = \|C^{\top} e\|^2 = e^{\top} C C^{\top} e = \|e\|^2 .$$

This is Parseval's identity for the DCT: the transform preserves the total energy of the signal (31852 in, 31852 out, as the script printed) and, by the same algebra, the total energy of any error. **Whatever squared error you introduce by quantizing coefficients is exactly the squared error the viewer gets in pixels.** You may do all your rate-distortion bookkeeping in the transform domain, where the numbers are easy to reason about, and the pixels will honour it.

Now the first half of the reviewer's question dissolves. Yes, the transform saves nothing by itself. But it moves the energy. In the pixel domain the energy is spread evenly over eight numbers, so every number matters about equally and nothing can be dropped. In the coefficient domain the same energy is piled into one or two numbers, and the rest are nearly empty. Dropping an empty coefficient costs nothing, and Parseval guarantees that "nothing" in the coefficient domain is "nothing" in the picture. The transform is a *setup move*: it changes where the energy sits so that a downstream quantizer with a dead zone can zero most of the block for free. In JPEG the quantizer that follows is the [frequency-weighted table](../scalar-quantization/) and the zeros it manufactures are what the zigzag run-length coder and Huffman stage then eat.

{{< lab id="dct" >}}

A ramp makes the point even more sharply. Run `[10, 12, 14, 16, 18, 20, 22, 24]` through the same matrix and you get `[48.08, −12.88, 0.00, −1.35, 0.00, −0.40, 0.00, −0.10]`: a DC term, one tilt term, and crumbs. The total energy of that ramp is 2480, and the first two coefficients hold 48.08² + 12.88² = 2477.6 of it. Sixteen bytes became, for all practical purposes, two numbers.

## Aha two: coding gain, and why the DCT and not some other rotation

So far one row. Any transform that happened to match this row would have looked good. The real question is the second one: *for a whole population of blocks, which rotation piles up the most energy, and how much does it save?*

To answer it we need a model of what a "smooth row" is. The simplest one that captures the photograph is a first-order Markov process: each sample is $\rho$ times its neighbour plus fresh noise, with $\rho$ close to 1. The covariance between samples $i$ and $j$ is then $\rho^{|i-j|}$, which for $\rho = 0.95$ means adjacent pixels are 95% correlated and pixels seven apart are still 70% correlated. Pass that covariance through the DCT and read off the variance of each coefficient:

```python
import numpy as np
N = 8; n = np.arange(N)
C = np.cos(np.pi*(2*n[None,:]+1)*n[:,None]/(2*N))*np.sqrt(2/N); C[0] /= np.sqrt(2)

rho = 0.95                                     # neighbour correlation of a smooth row
R = rho ** abs(n[:, None] - n[None, :])        # covariance of 8 consecutive samples
var_dct = np.diag(C @ R @ C.T)                 # coefficient variances under the DCT
gain = lambda v: v.mean() / np.exp(np.log(v).mean())   # arithmetic / geometric mean
print("DCT coefficient variances:", var_dct.round(3))
print("DCT coding gain:", gain(var_dct).round(3), "=", (10*np.log10(gain(var_dct))).round(2), "dB")
lam = np.linalg.eigvalsh(R)                    # KLT: the eigenbasis of R itself
print("KLT coding gain:", gain(lam).round(3))
print("white noise   :", gain(np.diag(C @ np.eye(8) @ C.T)).round(3))
```
```
DCT coefficient variances: [7.025 0.575 0.173 0.082 0.051 0.037 0.03  0.027]
DCT coding gain: 7.631 = 8.83 dB
KLT coding gain: 7.667
white noise   : 1.0
```

The pixel variances were all 1.0 (the diagonal of $R$). After the DCT the first coefficient has variance 7.0 and the last has 0.027: a ratio of 260 to 1. The arithmetic mean of the eight variances is still 1.0, because an orthonormal transform preserves the trace of the covariance. What changed is the *geometric* mean, which fell from 1.0 to 0.131.

Why is the geometric mean the thing to watch? Because of how bits are spent. The [rate and distortion](../rate-and-distortion/) article shows that for a Gaussian coordinate of variance $\sigma^2$ coded to distortion $D$ the rate is $\tfrac12 \log_2(\sigma^2 / D)$, and that the best way to split a bit budget across coordinates is reverse water-filling: give every coordinate the same distortion $D$. Sum the rates over $N$ coordinates and you get $\tfrac{N}{2}\log_2\!\big((\prod_k \sigma_k^2)^{1/N} / D\big)$. The total rate at a fixed distortion depends on the geometric mean of the variances, and only on that. Pixels, with every variance equal to 1, cost $\tfrac{N}{2}\log_2(1/D)$. The DCT coefficients cost $\tfrac{N}{2}\log_2(0.131/D)$. The difference per sample is $\tfrac12 \log_2 7.63 = 1.47$ bits.

That ratio has a name:

$$G_{\mathrm{TC}} \;=\; \frac{\frac1N \sum_{k} \sigma_k^2}{\big(\prod_{k} \sigma_k^2\big)^{1/N}} ,$$

the **transform coding gain**, arithmetic mean over geometric mean of the coefficient variances. It is 1 when the variances are equal, which is what happens to white noise under any rotation (the last line of output), and it grows as the transform makes the variances unequal. Ten times $\log_{10}$ of it is the gain in decibels, and since every 6.02 dB is one bit, 8.83 dB is the 1.47 bits per pixel we just computed. For a smooth row, the DCT lets you spend one and a half bits less per pixel at the same error, out of the eight you started with.

Which rotation maximises the gain? The arithmetic mean is fixed, so the question is which rotation minimises the geometric mean, and the answer is the one that makes the coefficients uncorrelated: the eigenbasis of the covariance matrix, which in this field is called the Karhunen-Loève transform and in statistics is PCA. The script computes it: the KLT gain is 7.667, the DCT gain 7.631. The DCT, a fixed matrix of cosines that knows nothing about the data, gets 99.5% of the gain of the transform fitted exactly to it. This is the deep reason the DCT exists. As $\rho \to 1$ the eigenvectors of the Markov-1 covariance converge to cosines, so for anything smooth the DCT *is* the KLT to within a rounding error, and unlike the KLT it needs no covariance estimate, no side information, and one fixed fast algorithm. The [next article](../rotations-and-rank/) is about what to do when the data is not smooth in this sense and the fixed basis stops being good enough.

{{< fig src="img/block-transforms/compaction.png" alt="An 8x8 block from a picture and an 8x8 block of noise, their DCT coefficient grids on a log scale, and the sorted coefficient magnitudes" caption="Energy compaction is a property of the data. The picture block puts 99% of its energy into two of 64 coefficients; the noise block needs 38. The DCT did exactly the same thing to both, and Parseval guarantees the totals match the pixel energies." >}}

Here is the general definition, now that the example has done its work. The 8-point **DCT-II** is

$$X_k \;=\; c_k \sqrt{\tfrac{2}{N}} \sum_{n=0}^{N-1} x_n \cos\!\left[\frac{\pi\,(2n+1)\,k}{2N}\right], \qquad c_0 = \tfrac{1}{\sqrt{2}},\; c_{k>0} = 1,$$

and the 2-D version applies it along rows and then along columns, $\mathbf X = C\,\mathbf x\,C^{\top}$, so that a block is a weighted sum of 64 **basis images** $C_u^{\top} C_v$, each a product of a vertical and a horizontal cosine. The $(2n+1)$ in the argument is the detail that makes it a cosine transform rather than a Fourier transform: it treats the block as if it were mirrored at its edges, so a block whose left and right ends differ does not look like a discontinuity, and the coefficients decay faster than a DFT's would.

{{< fig src="img/block-transforms/basis.png" alt="The 64 basis images of the 8x8 DCT arranged in a grid by vertical and horizontal frequency" caption="The 64 basis images of the 8×8 DCT-II. Every block in every JPEG is a weighted sum of these pictures. Smooth blocks live in the top-left corner; the quantization table that follows charges more for every step away from it." >}}

## Aha three: the price of the block

Imagine, because this scene is invented, a member of the JPEG committee in 1988, a few months before the standard is frozen, running the reference encoder at quality 10 on a portrait to see what the low end looks like. She expects blur. What she sees is a mosaic: the face is tiled into visible 8×8 squares, each one smooth inside, with a sharp step at every boundary. She knows immediately why. Each block was transformed and quantized on its own. Inside a block, quantization error is a smooth combination of cosines. At the boundary, two independently quantized blocks meet, and their errors have no reason to agree.

Why 8×8 at all, then? Because the transform must be computed by something. The 2-D DCT of a $B \times B$ block costs on the order of $B^3$ multiplies with the separable algorithm, and the coding gain of a Markov-1 source saturates quickly: most of the gain is in by $B = 8$, and a decoder of that era could not afford to buffer whole rows of larger blocks. The block was a compromise between gain, arithmetic, memory and random access (a corrupted byte ruins one block, not the picture), and the committee bought it with visible seams at low quality. Video codecs made the arithmetic cheaper rather than the block larger: H.264 replaced the real-valued DCT with 4×4 and 8×8 *integer* transforms whose coefficients are small whole numbers, so that encoder and decoder produce bit-identical results on every chip, and AV1 and JPEG XL pick the block size per region instead of fixing it.

The audio people met the same seam, and it hurt more. A click at a block boundary in a picture is a visible edge; a click at a block boundary in audio is a click, every 20 milliseconds, forever. Their fix is the **modified discrete cosine transform**. Instead of cutting the signal into disjoint blocks of $N$ samples, take windows of $2N$ samples that *overlap by half*, so each sample sits in two windows. That sounds like it should double the data, and the trick of the MDCT is that it does not: it maps each $2N$-sample window to only $N$ coefficients. The information that goes missing is a time-domain aliasing term, a folded copy of the signal, and the design is arranged so that the aliasing produced by one window is exactly cancelled by the aliasing of its neighbour when the two are overlapped and added. The acronym is TDAC, time-domain aliasing cancellation. The condition on the window for this to work is the Princen-Bradley condition $w^2[n] + w^2[n+N] = 1$, and the sine window $w[n] = \sin\!\big(\tfrac{\pi (n + 1/2)}{2N}\big)$ satisfies it.

{{< fig src="img/block-transforms/windows.png" alt="Overlapping sine windows for the MDCT, and the sum of their squares, which is flat at 1" caption="MDCT windows are 2N long and start every N samples. The Princen–Bradley condition makes the squared windows sum to exactly 1 wherever two overlap, so the folded halves reassemble the signal with no seam and no extra coefficients." >}}

The price is not zero; it just moved. Because the second half of each window belongs to the next frame, the decoder cannot finish reconstructing the current frame until the next one has arrived. The MDCT costs **one frame of delay**, always. In MP3 a frame is 1152 samples, which at 44.1 kHz is 26 ms, and for a file on disk nobody cares. In a phone call, 26 ms of codec delay on top of everything else is a noticeable chunk of the ~150 ms mouth-to-ear budget, which is why Opus runs the identical transform on frames of 2.5 to 20 ms. Same stage, same algorithm, different job; the [anatomy of a job](../anatomy-of-a-job/) article is about exactly this kind of dial.

{{< lab id="mdct" >}}

Wavelets pay a third price for the same seam. The simplest wavelet, the Haar transform, replaces each pair of samples by their average and half their difference:

```python
import numpy as np
x = np.array([10, 12, 14, 16, 100, 98, 96, 94], float)
avg = (x[0::2] + x[1::2]) / 2          # low-pass: pairwise means
det = (x[0::2] - x[1::2]) / 2          # high-pass: pairwise half-differences
print("averages:", avg, " details:", det)
avg2, det2 = (avg[0::2] + avg[1::2]) / 2, (avg[0::2] - avg[1::2]) / 2
print("level 2  :", avg2, det2)
# exact inverse: x[2i] = avg + det, x[2i+1] = avg - det
back = np.empty(8); back[0::2] = avg + det; back[1::2] = avg - det
print("inverse ok:", np.array_equal(back, x))
```
```
averages: [11. 15. 99. 95.]  details: [-1. -1.  1.  1.]
level 2  : [13. 97.] [-2.  2.]
inverse ok: True
```

The row has a sharp edge in the middle, from 16 to 100, which would spray energy across every DCT coefficient of an 8-point block. Haar handles it differently: the four averages are a half-resolution copy of the row (the edge survives, as 15 → 99), and the four details are tiny, ±1, because within each pair the row is smooth. Apply the same step to the averages and you get a quarter-resolution copy `[13, 97]` plus details `[−2, 2]`. Keep going and you have a pyramid: the signal at every scale, nested in one array of the same length as the input, with the detail coefficients small wherever the signal is smooth *at that scale* and large only right at the edge. There are no blocks, so there are no block seams; at low quality a wavelet image blurs rather than tiles, and the coarsest level is a free thumbnail, which gives progressive decoding for nothing. JPEG 2000 is built on this, with longer filters than Haar. What it costs is memory and locality: a multi-level transform touches the whole image, or at least whole bands of it, so the decoder cannot work in 64-pixel bites, and it needs every scale before it can finish any pixel. In the market this lost. JPEG 2000 was around 20% better than JPEG and far more complex, arrived with patent uncertainty, and faced a decoder already installed on every computer on Earth; it lives on in cinema distribution and medical archives, where the decoder is not the bottleneck.

{{< lab id="dwt" >}}

## Connections

**Fourier and the heat equation.** Joseph Fourier introduced the cosine series in 1807 to solve the flow of heat in a bar, and the reason it worked there is the reason the DCT works here. Cosines are the eigenfunctions of the second-derivative operator with reflecting boundaries, and the heat equation damps each eigenfunction at a rate proportional to the square of its frequency: high frequencies die first, low frequencies linger. A signal that has been smoothed, by diffusion or by a camera lens or by the physics of skin and sky, is a signal whose energy has drained out of the high-frequency cosines. The DCT does not create energy compaction; it reads off the compaction that smoothness already put there. The Markov-1 computation above is the discrete version of the same statement: as $\rho \to 1$ the covariance matrix becomes a discrete second-derivative operator, and its eigenvectors become cosines.

**PCA is the data-adaptive version.** The KLT in the coding-gain script is PCA. The DCT is what you use when you can afford neither the covariance estimate nor the side information to ship the basis, and your data is smooth enough that cosines are within half a percent of optimal. When the data is not smooth in that sense, learned rotations come back, with a basis that must be stored or sent; that is [the next article](../rotations-and-rank/). The error metric the whole chain minimises is squared error, which is a stand-in for what the eye sees; the [perceptual models](../perceptual-models/) article replaces the stand-in, and it does so by scaling exactly these coefficients.

**The same move at other scales.** JPEG XL's VarDCT picks block sizes from 8×8 up to 256×256 per region, which is the committee member's seam problem solved by making the block fit the content. Video codecs transform the residual left after motion compensation rather than the pixels, and the integer transforms they use are DCT approximations chosen so that $C$ has entries like 1, 2 and 1/2 and the decoder needs no floating point at all.

**In the game:** DCT (block DCT), MDCT (MDCT) and Wavelet (DWT) (wavelets) fill the transform stage in the Photographs task (JPEG), the Music distribution task (MP3), the Streaming video task (H.264 / AV1), the Modern web images task (WebP) and, with 2.5 ms frames, the Real-time speech task (Opus). Every one of them is followed by a quantizer, because the transform stage saves nothing on its own.
