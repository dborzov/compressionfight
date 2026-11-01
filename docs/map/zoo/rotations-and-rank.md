---
title: "Rotations and rank: learned axes"
slug: rotations-and-rank
chair: transform
family: "Rotations, projections and rank"
summary: "A rotation changes nothing about the data, yet it can cut the error of the same quantizer several times over; which rotation you want depends on the quantizer behind it, orthogonality keeps attention scores intact, and low rank is the same compaction along the token axis."
fighters: [pca, svd_lowrank, unrope, basis]
related: [block-transforms, vector-quantization, block-float-formats, kv-cache-vaults]
reading_time: 15
---

## The same wall, from two sides

Imagine two engineers, eleven years apart, who never meet.

The first, in 2013, runs the nearest-neighbour index of a large image search service. She has a hundred million 128-dimensional descriptors and no memory to hold them in float, so she compresses each one with a product quantizer: cut the vector into eight sub-vectors of 16 dimensions, train a 256-entry codebook for each, store eight bytes per vector. It works, and it is wrong in a way she can measure. Some sub-vectors carry most of the variance and their codebooks are overworked; others are nearly constant and their 256 entries are wasted. Worse, the sub-vectors are correlated with each other, so the codebooks keep rediscovering the same structure. The data is not lying along her axes. It is lying along a diagonal that her slicing cannot see.

The second, in 2024, is squeezing the KV cache of a language model into four bits per element so a longer context fits on the GPU. Keys quantized with one scale per token fall apart long before values quantized the same way do. When he plots the keys he sees why. A handful of channels, the same handful in every token, are an order of magnitude louder than the rest. A four-bit grid wide enough to hold the loud channel has a step so coarse that every quiet channel rounds to the same two or three values. The data is not lying along his axes either; it has a few spikes, and a shared grid serves spikes badly.

Both of them reach for the same tool, and that is the puzzle of this article. *A rotation changes nothing about the data: not its energy, not the distances between points, not the dot products. How can rotating it halve the error of the very same quantizer?* And a second question hides inside the first: the two engineers want *opposite* rotations, one that piles the variance up and one that spreads it flat. How can both be right?

## Forty points on a diagonal

Here is the smallest case that shows everything. Forty points in the plane, generated so that $y \approx 0.8x$: two measurements of nearly the same thing, like the red and green of a grey photograph, or two channels of a key that move together.

```python
import numpy as np
rng = np.random.default_rng(7)
x = rng.standard_normal(40) * 2.0
P = np.stack([x, 0.8*x + rng.standard_normal(40)*0.5], 1); P -= P.mean(0)
print("variance per axis:", P.var(0).round(3))
lam, V = np.linalg.eigh(np.cov(P.T, bias=True)); V, lam = V[:, ::-1], lam[::-1]
print("eigenvalues:", lam.round(3), " major axis angle:", np.degrees(np.arctan(V[1, 0] / V[0, 0])).round(1), "deg")
R = P @ V                                      # rotate so axes = eigenvectors

def grid(A, bits):                             # uniform mid-rise grid per axis, 2^b levels
    Q = np.zeros_like(A)
    for j, b in enumerate(bits):
        if b == 0: continue
        L = 2**b; s = 2*abs(A[:, j]).max()/L
        Q[:, j] = (np.clip(np.floor(A[:, j]/s), -L//2, L//2-1) + 0.5) * s
    return Q
mse = lambda A, Q: ((A - Q)**2).sum(1).mean()
print("original axes, 2+2 bits:", mse(P, grid(P, [2, 2])).round(3))
print("rotated axes,  2+2 bits:", mse(R, grid(R, [2, 2])).round(3))
print("rotated axes,  3+1 bits:", mse(R, grid(R, [3, 1])).round(3))
print("rotated axes,  4+0 bits:", mse(R, grid(R, [4, 0])).round(3), " (lambda_2 =", lam[1].round(3), ")")
```
```
variance per axis: [2.627 1.733]
eigenvalues: [4.236 0.124]  major axis angle: 38.7 deg
original axes, 2+2 bits: 0.516
rotated axes,  2+2 bits: 0.589
rotated axes,  3+1 bits: 0.241
rotated axes,  4+0 bits: 0.163  (lambda_2 = 0.124 )
```

Along the original axes the two variances are comparable, 2.63 and 1.73. The covariance matrix has eigenvalues 4.24 and 0.12, and its first eigenvector points at 38.7°, which is $\arctan 0.8$: the direction the points actually lie along. Rotating into the eigenbasis leaves the total variance alone (4.236 + 0.124 = 4.36 = 2.627 + 1.733) but moves nearly all of it onto one axis. That is the same energy compaction the [DCT](../block-transforms/) performed on a smooth row, except that this basis was computed from the data rather than assumed.

Now quantize. With a 2-bit grid on each axis, 16 cells per point, the error on the original axes is 0.516. Rotate and use the same 2-bit grid on each new axis and the error is 0.589. **The rotation made it worse.** This is the puzzle in its sharpest form, and it is worth sitting with before reading on. If a rotation changes nothing and the quantizer is identical, why would anyone expect a gain at all?

## Aha one: the rotation only pays when the quantizer can respond

Think about where the error of an axis-aligned grid comes from. Each axis is divided into $2^b$ steps of width $\Delta_j$, and at high resolution the error is $\sum_j \Delta_j^2/12$. With a fixed number of bits per axis and the step tied to the range of that axis, $\Delta_j$ scales with the spread $\sigma_j$ of the data along it, so the error scales with $\sum_j \sigma_j^2$. That sum is the trace of the covariance matrix, and a rotation does not change the trace. Two bits here plus two bits there costs about the same error in every orientation, up to the small-sample noise that made our rotated grid slightly worse. The rotation cannot help a quantizer that treats every axis the same.

What the rotation did change is the *distribution* of the variance across axes: 4.24 against 0.12 instead of 2.63 against 1.73. A quantizer that can respond to that, by spending its four bits unequally, suddenly has something to work with. Three bits on the major axis and one on the minor: 0.241. Four bits on the major axis and none at all on the minor, which means every point is snapped onto the 38.7° line and its distance from the line is simply discarded: 0.163, of which 0.124 is the variance of the dropped axis itself. Same points, same four bits, three times less error than the original grid. Try the 4+0 split on the original axes and you get 1.75: dropping the raw $y$ coordinate throws away a measurement that was not redundant, while dropping the rotated minor axis throws away something that was.

{{< fig src="img/rotations-and-rank/scatter.png" alt="Forty correlated points quantized with a 2-bit grid on the original axes, with a 2-bit grid on the rotated axes, and with 4 bits on the major axis only" caption="Same 40 points, same 4 bits per point. The rotation changes nothing until the bit allocation is allowed to follow the variance; then dropping the minor axis entirely gives 3.2× less error than the axis-aligned grid." >}}

This is the first half of the answer to the two-engineers puzzle. The quantity the rotation improves is the one from the [previous article](../block-transforms/): the arithmetic mean of the per-axis variances is invariant, but the geometric mean fell from $\sqrt{2.627 \times 1.733} = 2.13$ to $\sqrt{4.236 \times 0.124} = 0.72$, and the geometric mean is what the rate at a fixed distortion depends on, *provided the bits are allocated by reverse water-filling*. The rotation that minimises the geometric mean is the one that decorrelates the coordinates: the eigenbasis of the covariance, the **Karhunen-Loève transform**, which is PCA. Among all orthonormal transforms it is optimal for a Gaussian source in exactly this sense, and the DCT exists because for smooth data it approximates the KLT without having to be fitted. For data that is not smooth in the Markov sense, the image descriptors of the 2013 engineer, the fitted basis comes back.

That is what she did. **Optimized Product Quantization** finds a rotation $R$ and applies it to every vector before slicing into sub-vectors, choosing $R$ so that the variance is balanced across sub-vectors and the correlation between them is minimal: PCA to find the directions, then a permutation that deals the eigenvectors out to the eight sub-vectors so that each gets a fair share of the eigenvalues. Her codebooks stop fighting each other, and the [product quantizer](../vector-quantization/) behind the rotation loses far less at the same eight bytes. The rotation matrix is stored in the index, which is the first appearance in this article of a theme we will return to: a fitted basis is side information.

{{< lab id="pca" >}}

Here is the proper definition. For data vectors $x_1, \dots, x_L \in \mathbb{R}^d$ with sample covariance $\Sigma$, **PCA** is the orthonormal matrix $V$ whose columns are the eigenvectors of $\Sigma$ sorted by decreasing eigenvalue $\lambda_1 \ge \dots \ge \lambda_d$. The rotated coordinates $z = V^{\top} x$ are uncorrelated with variances $\lambda_j$, and keeping the first $r$ coordinates gives the rank-$r$ linear approximation of the data with the smallest possible total squared error, namely $\sum_{j > r} \lambda_j$. In practice $V$ is read off the SVD of the centred data matrix, $X = U S V^{\top}$, as the right singular vectors, with $\lambda_j = s_j^2 / L$.

## The other rotation: flatten, do not concentrate

Now the 2024 engineer. His quantizer is not a per-axis grid with a per-axis bit budget. It is an int4 block format: one shared scale for a whole vector of channels, sixteen levels for every channel, the same sixteen levels. The [block-float formats](../block-float-formats/) article explains why the GPU wants it that way; here it only matters what it does to a vector with one loud channel.

```python
import numpy as np
H = np.array([[1, 1, 1, 1], [1, -1, 1, -1], [1, 1, -1, -1], [1, -1, -1, 1]]) / 2
rng = np.random.default_rng(7)
K = rng.standard_normal((2000, 4)) * [8.0, 0.5, 0.5, 0.5]   # channel 0 is the outlier

def int4_shared(A):                     # one scale per vector, 16 levels
    s = 2 * abs(A).max(1, keepdims=True) / 16
    return (np.clip(np.floor(A / s), -8, 7) + 0.5) * s
err = lambda A: ((A - int4_shared(A))**2).sum(1).mean()
print("channel stds raw      :", K.std(0).round(2),  " MSE", err(K).round(3))
KH = K @ H.T
print("channel stds Hadamard :", KH.std(0).round(2), " MSE", err(KH).round(3))
q = rng.standard_normal(4)
print("q.k preserved:", np.allclose(K @ q, KH @ (H @ q)))
```
```
channel stds raw      : [7.92 0.49 0.49 0.5 ]  MSE 0.539
channel stds Hadamard : [3.99 3.97 3.99 3.97]  MSE 0.148
q.k preserved: True
```

Two thousand four-channel keys, channel 0 sixteen times louder than the others. The shared scale is set by channel 0, so the step is about one unit, and the three quiet channels, whose entire range is about ±1.5, each get two or three usable levels. Error 0.539. Then multiply every key by $H$, the 4×4 **Hadamard matrix** scaled by ½ so that it is orthonormal. Its rows are patterns of ±1, so every output channel is a sum and difference of all four inputs, and the loud channel's energy is shared out equally: four channels of standard deviation 4 instead of one of 8 and three of 0.5. The shared scale now serves everyone, the step halves, and the error drops to 0.148. Same quantizer, 3.6× less error.

{{< fig src="img/rotations-and-rank/hadamard.png" alt="Box plots of four channels before and after a Hadamard rotation, with one outlier channel before and four equal channels after" caption="Hadamard does the opposite of PCA. One loud channel sets a shared int4 scale that starves the other three; after the rotation all four have the same spread and the same scale fits them all." >}}

So the second half of the answer: **which rotation you want depends on the quantizer behind it.** A quantizer that can spend bits unevenly, per axis or per coordinate, wants the variance concentrated, and PCA. A quantizer forced to spend bits evenly, with one grid for every coordinate, wants the variance flattened, and Hadamard. The two engineers were not disagreeing; they were serving different quantizers.

The Hadamard has two further virtues that explain why it, and not some other flattening rotation, is the one in QuaRot and SpinQuant. It costs no side information at all: $H_{2n} = \tfrac{1}{\sqrt 2}\begin{pmatrix} H_n & H_n \\ H_n & -H_n \end{pmatrix}$ is defined by a formula, and a random sign flip per channel is the only thing to agree on. And it is fast: that recursive structure gives a butterfly algorithm in $O(n \log n)$ additions, no multiplies, which is what makes it affordable inside a forward pass. The choice is not arbitrary either: a random orthogonal matrix flattens too, but QuaRot reports it does measurably worse on LLaMA-2 7B (perplexity 7.45 against 6.10 at the same bit width). SpinQuant goes one step further and learns two of its rotations by gradient descent on the Stiefel manifold, starting from a Hadamard; the learned matrices are 0.26% of the weight size, and they ship with the checkpoint. The moment a rotation is fitted rather than formulaic it stops being free.

## Aha two: attention does not notice

There is something the 2024 engineer could do that the 2013 engineer could only approximate, and the script already showed it: `q.k preserved: True`. An orthonormal matrix preserves dot products,

$$(Hq)\cdot(Hk) \;=\; q^{\top} H^{\top} H\, k \;=\; q^{\top} k ,$$

so if you rotate every key by $H$ and also rotate every query by the same $H$, every attention score comes out exactly as before. The softmax, which only ever sees $q \cdot k$, cannot tell that anything happened. This is what lets QuaRot rotate *inside* a transformer: queries and keys are multiplied by the same per-head $H$ on the fly, the cache stores rotated keys, the kernel quantizes them with a shared scale, and the attention scores are unchanged up to the quantization error itself. (The rotation of the values, and its inverse on the output projection, can be folded into the weight matrices once and for all; the one on $q$ and $k$ cannot, because RoPE sits between the weights and the dot product, so it is applied online, which is where QuaRot's "1½ Hadamards per layer" comes from.) The 2013 engineer has the same identity, which is why she can rotate the database and the query with one matrix and run asymmetric distance computation in the rotated space; what she lacks is a way to make the rotation free, so hers lives in the index file.

This also tells you what a rotation *cannot* do. It cannot reduce the entropy of the vector or the information it carries; the dot products, the norms and the pairwise distances are all exactly preserved. All it changes are the marginal statistics of the coordinates: the variances of each channel, which is exactly, and only, what an axis-aligned quantizer looks at.

{{< lab id="basis" >}}

The side-information theme now has three points on it. A Hadamard costs 0 bytes, because it is a formula. SpinQuant's learned rotations cost 0.26% of the model. KVTC, which runs a full PCA on the keys and values of a model and quantizes in the eigenbasis with water-filling, ships a basis that is 2.4% of Llama 3.3 70B's parameters, fitted once on about 160k calibration tokens in 86 seconds of randomized SVD and shared by every layer and head. In each case the stream carries only coefficients, and the coefficients are meaningless to anyone who does not hold the axes. The [side information](../side-information/) article treats a fitted basis as exactly what it is: a codebook by another name.

## RoPE: a rotation the model already applied

The KV cache has one more rotation in it, and this one nobody chose for compression. Rotary position embeddings encode a token's position by rotating each pair of key channels $(k_{2i}, k_{2i+1})$ through an angle $m\theta_i$ that grows with the position $m$. The same rotation is applied to the query, and the dot product between a query at position $m$ and a key at position $m'$ then depends only on $m - m'$. It is elegant, and for a quantizer it is a disaster. Take one channel pair of a key, the loud channel 8.0 and a quiet one 0.3, and store it at eight consecutive positions:

```python
import numpy as np
k = np.array([8.0, 0.3])                      # one channel pair of a pre-RoPE key
rot = lambda v, t: np.array([np.cos(t)*v[0] - np.sin(t)*v[1], np.sin(t)*v[0] + np.cos(t)*v[1]])
stored = np.array([rot(k, m*0.5) for m in range(8)])   # the same key at positions 0..7
print(stored.round(2))
print("per-channel range after RoPE:", stored.min(0).round(2), "to", stored.max(0).round(2))
undone = np.array([rot(stored[m], -m*0.5) for m in range(8)])
print("after undoing:", undone.min(0).round(2), "to", undone.max(0).round(2))
q = np.array([1.0, 2.0])
print("q.k =", q @ k, " (Rq).(Rk) =", (rot(q, 1.3) @ rot(k, 1.3)).round(6))
```
```
[[ 8.    0.3 ]
 [ 6.88  4.1 ]
 [ 4.07  6.89]
 [ 0.27  8.  ]
 [-3.6   7.15]
 [-6.59  4.55]
 [-7.96  0.83]
 [-7.39 -3.09]]
per-channel range after RoPE: [-7.96 -3.09] to [8. 8.]
after undoing: [8.  0.3] to [8.  0.3]
q.k = 8.6  (Rq).(Rk) = 8.6
```

The content of the key never changed; only its position did. Yet the stored channel 0 wanders from +8 to −8 and the stored channel 1 from −3 to +8. A per-channel quantizer, the kind KIVI uses on keys because the outlier channels are stable from token to token, sees a channel whose magnitude is not stable at all; it has been smeared over its partner by an angle that depends on where the token sits. Undo the rotation, which the decoder can always do because it knows $m$, and both channels collapse back to constants.

{{< fig src="img/rotations-and-rank/rope.png" alt="The two channels of a key pair plotted against token position after RoPE, oscillating between -8 and 8, next to the circle the pair traces in the plane" caption="RoPE wandering. The key content is (8.0, 0.3) at every position; the stored values trace a circle of radius 8 as the position advances. Any statistic computed per channel across tokens sees the circle, not the point." >}}

This is the whole of the "undo RoPE first" idea: store the pre-RoPE key, quantize that, and let the attention kernel apply the rotation after dequantization. The rotation is invertible and known, so it costs no bits, only a 2×2 multiply per channel pair at read time. KVQuant measured what it is worth: 3-bit LLaMA-7B perplexity goes from 7.05 with post-RoPE keys to 6.23 with pre-RoPE keys, and Coupled Quantization, KVTC and TensorRT-LLM's `skip_rope_quantization` all do the same. QuaRot took the other road the dot-product identity allows: leave RoPE in, then rotate $q$ and $k$ by the same Hadamard *after* it, so the scores survive regardless. Values are never rotated by RoPE, which is one of several reasons keys and values want different treatment.

{{< lab id="unrope" >}}

## Aha three: low rank is compaction along the token axis

Everything so far rotated the *channels* of one vector. Turn the picture ninety degrees. A sequence of $L$ keys is an $L \times d$ matrix, and the question "do these vectors lie near a subspace" is the question "is this matrix nearly low rank". Here is an 8×4 matrix built to have rank 2, plus a little noise:

```python
import numpy as np
rng = np.random.default_rng(7)
M = rng.standard_normal((8, 2)) @ rng.standard_normal((2, 4))   # 8 tokens, 4 channels, rank 2
M += 0.05 * rng.standard_normal((8, 4))                           # plus a little noise
U, s, Vt = np.linalg.svd(M)
print("singular values:", s.round(3))
r = 2
A, B = U[:, :r] * s[:r], Vt[:r]                                   # 8x2 and 2x4 factors
print("rank-2 error (Frobenius):", np.linalg.norm(M - A @ B).round(4),
      " = sqrt(s3^2 + s4^2):", np.sqrt((s[r:]**2).sum()).round(4))
print("numbers stored:", A.size + B.size, "instead of", M.size)
print("row 5 rebuilt on demand:", (A[5] @ B).round(2), " stored row:", M[5].round(2))
```
```
singular values: [6.553 1.055 0.152 0.132]
rank-2 error (Frobenius): 0.2011  = sqrt(s3^2 + s4^2): 0.2011
numbers stored: 24 instead of 32
row 5 rebuilt on demand: [-1.26 -0.31 -1.34 -0.55]  stored row: [-1.25 -0.39 -1.34 -0.53]
```

The singular values drop from 6.55 and 1.06 to 0.15 and 0.13: two directions carry the signal, two carry the noise. Keep the first two and the matrix becomes a product $A B$ of an $8 \times 2$ and a $2 \times 4$ factor, 24 numbers instead of 32, and the error is exactly $\sqrt{s_3^2 + s_4^2}$. That exactness is the **Eckart-Young theorem**: truncating the SVD at rank $r$ gives the best rank-$r$ approximation in Frobenius norm, with error $\sqrt{\sum_{j>r} s_j^2}$, and no other rank-$r$ matrix does better. It is PCA again, read along the other axis: the columns of $V$ are the principal directions of the rows, and the rows of $A$ are the coordinates of each token in that basis. The storage is $Lr + rd$ against $Ld$, so at 8 tokens the saving is modest and at 128k tokens it is almost $d/r$.

This is why ShadowKV can keep the keys of a 128k-token prompt on the GPU. After prefill it takes the SVD of the pre-RoPE key matrix, $L \times 1024$ per layer, and keeps rank 160: an $L \times 160$ factor and a $160 \times 1024$ factor, 15.6% of the original, and rebuilds any key row it needs with a tiny matmul before applying RoPE. On Llama-3.1-8B the Golomb-Rice score at 128k moves from 85.5 to 83.6. Two things about this are the real lesson. First, it only works on *pre-RoPE* keys: RoPE multiplies each token's row by a different rotation, which scatters a shared subspace across positions just as it scattered the channel pair above, and the singular values of post-RoPE keys decay slowly. Second, it does not work on values at all. Values show no shared subspace; ShadowKV leaves them in full and offloads them to CPU. Keys are produced by a projection that the attention mechanism rewards for being discriminable along a few directions; values are what gets copied out, and they use their whole space.

{{< fig src="img/rotations-and-rank/svd.png" alt="Singular value decay and cumulative energy for synthetic pre-RoPE keys, post-RoPE keys and values" caption="Synthetic 1024×128 matrices built to mirror ShadowKV's observation: pre-RoPE keys with a 16-dimensional subspace reach 99% of their energy at rank 15; applying RoPE to the same keys pushes that to rank 94; values with no subspace need 124. The low-rank stage only has something to sit on before RoPE." >}}

The subspace is also *sequence-specific*: it is fitted to this prompt and serves this prompt's continuation, not other prompts, so it is not side information shipped with the model but a per-sequence factorization computed at prefill. xKV found that adjacent layers, which disagree token by token, nevertheless share the same important directions, so one basis fitted across four layers at once reaches 8× compression at a Golomb-Rice score of 88.5 against 91.9 for the full cache. And MLA, in the [learned transforms](../learned-transforms/) article, is what happens when the low-rank bottleneck is put into the model before pretraining instead of fitted afterwards.

{{< lab id="svd_lowrank" >}}

## Connections

**Whitening.** Statisticians have done the PCA-then-rescale move for a century under the name whitening: rotate to the eigenbasis and divide each coordinate by $\sqrt{\lambda_j}$, so the covariance becomes the identity. It is the limit of "flatten the variances", and the Hadamard is a cheap, side-information-free step in the same direction that does not need the eigenbasis.

**The DCT is the KLT of a Markov-1 process.** The fixed cosine basis of the [previous article](../block-transforms/) is what PCA converges to when the covariance is $\rho^{|i-j|}$ with $\rho$ near 1. The two articles are one idea: use the KLT when you can afford to fit and ship it, use a fixed approximation when the data is smooth enough that you do not have to.

**Random projections and the Johnson-Lindenstrauss lemma.** A random orthonormal matrix, like a Hadamard with random signs, preserves pairwise distances between $n$ points to within $1 \pm \epsilon$ even when you keep only $O(\log n / \epsilon^2)$ of its output coordinates. Flattening rotations are the first half of that construction; dimension reduction without PCA is the second.

**LoRA.** Fine-tuning a weight matrix with a product of a $d \times r$ and an $r \times d$ factor is the Eckart-Young picture applied to the *update* rather than the data: the bet that what changes during fine-tuning lies near a low-dimensional subspace, which is the same bet ShadowKV makes about keys.

**In the game:** PCA / Hadamard rotation (PCA / OPQ / Hadamard), Pre-RoPE storage (pre-RoPE keys), Low-rank SVD (rank-160 SVD) and Trained basis (the shipped basis) fill the transform stage and shared-prior stages in the Billion-scale vector search task (Faiss OPQ-PQ), the KV cache resident in HBM task (QuaRot / KIVI), the KV cache at 1–2 bits task (Coupled Quantization), the KV cache offload to host memory task (ShadowKV) and the KV cache cold storage task (KVTC).
