---
title: "Vector quantization: the codebook"
slug: vector-quantization
chair: quantize
family: "Vector quantizers"
summary: "Why quantizing two numbers together beats quantizing each alone (the gap is the mutual information a scalar grid pays twice), how product and residual codebooks dodge the exponential, and why a 2-bit index is meaningless until the decoder owns the codebook."
fighters: [vq, rvq, pq, coupled, codebook]
related: [rotations-and-rank, learned-transforms, side-information, scalar-quantization]
reading_time: 15
---

# Vector quantization: the codebook

Three little piggies live on an island. Their houses sit close together, because they are brothers and like to visit, and the island's one postman has to tell the mainland post office where to deliver. The telegraph line is expensive: he is allowed a handful of symbols per week, and the mainland clerk owns nothing but a blank map.

That is the whole problem of this article in miniature. Replace the houses with 768-dimensional embeddings and the postman with a search engineer, and the budget is still a handful of bits per item and the clerk still needs a map. Two questions decide everything: *why does describing several numbers together cost fewer bits than describing each one alone?* And *what must the decoder already own for a 2-bit message to mean anything at all?*

The [scalar quantizer](../scalar-quantization/) answered "how many bits?" one coordinate at a time. We are going to see that answering it for a whole vector at once is not a refinement of that; it is a different quantity being paid for.

## Three houses, one number

Here is the island as the postman sees it: kilometres east and north of the lighthouse.

```python
import numpy as np
houses = np.array([[1.0, 2.0], [1.4, 2.6], [1.8, 2.3]])   # piggy houses: km east, km north
c = houses.mean(axis=0)
print("centroid", c, " total squared distance", ((houses - c) ** 2).sum().round(3))
for guess in [np.array([1.0, 2.0]), np.array([1.5, 2.5])]:
    print("meet at", guess, "->", ((houses - guess) ** 2).sum().round(3))
```
```
centroid [1.4 2.3]  total squared distance 0.5
meet at [1. 2.] -> 1.25
meet at [1.5 2.5] -> 0.65
```

Six floats describe the three houses exactly. If the postman may send only one point, which point? The one that minimises how far, on average, a parcel delivered to it would have to be carried onward. With squared distance as the measure, that point is the mean: moving the meeting point to the first house costs 1.25, a reasonable-looking guess costs 0.65, the mean costs 0.5, and nothing beats it. This is the one calculus fact the article rests on:

$$
\mathbf{c}^\star = \arg\min_{\mathbf{c}} \sum_{i=1}^{n} \|\mathbf{x}_i - \mathbf{c}\|^2 = \frac{1}{n}\sum_{i=1}^{n} \mathbf{x}_i ,
$$

and we call $\mathbf{c}^\star$ the **centroid** of the set. Sending the centroid instead of the three houses is a lossy description with squared error 0.5, and it works *because* the brothers live near each other. Had they scattered across the island the centroid would be a useless point in the sea.

## The chickens move in

A family of chickens settles on the far side of the island. Six houses now; one centroid lands in the water between the two families (total squared error 36.2). Obviously the postman wants two meeting points, one per family, and one bit per house to say which. But he does not know in advance which houses belong together, and in 768 dimensions nobody can look at a map. He needs a procedure, and the natural one is to alternate two obvious rules.

```python
import numpy as np
houses = np.array([[1.0, 2.0], [1.4, 2.6], [1.8, 2.3],      # piggies
                   [6.1, 0.9], [6.5, 1.5], [5.8, 1.3]])     # chickens
C = np.array([[1.0, 2.0], [1.8, 2.3]])   # a bad start: both guesses inside the piggy cluster
for it in range(3):
    d = ((houses[:, None, :] - C[None, :, :]) ** 2).sum(-1)   # 6×2 table of squared distances
    a = d.argmin(1)                                            # nearest codeword for each house
    C = np.array([houses[a == k].mean(0) for k in range(2)])   # move each codeword to its crowd
    print(it, "assign", a, "codebook", C.round(3))
d = ((houses[:, None, :] - C[None, :, :]) ** 2).sum(-1)
print("total squared error K=2:", d.min(1).sum().round(3), " with one centroid:",
      ((houses - houses.mean(0)) ** 2).sum().round(3))
```
```
0 assign [0 1 1 1 1 1] codebook [[1.   2.  ]
 [4.32 1.72]]
1 assign [0 0 0 1 1 1] codebook [[1.4   2.3  ]
 [6.133 1.233]]
2 assign [0 0 0 1 1 1] codebook [[1.4   2.3  ]
 [6.133 1.233]]
total squared error K=2: 0.933  with one centroid: 36.247
```

Even from a deliberately bad start, two rounds of "assign each house to its nearest point, then move each point to the mean of its houses" find the two families. Now the vocabulary, properly.

A **vector quantizer** of dimension $d$ and size $K$ is a **codebook** $\mathcal{C} = \{\mathbf{c}_0, \dots, \mathbf{c}_{K-1}\} \subset \mathbb{R}^d$ together with the **nearest-neighbour rule** $q(\mathbf{x}) = \arg\min_k \|\mathbf{x} - \mathbf{c}_k\|^2$. The encoder sends the **index** $q(\mathbf{x})$, which costs $\lceil \log_2 K \rceil$ bits; the decoder outputs the **codeword** $\mathbf{c}_{q(\mathbf{x})}$. The set of inputs that map to codeword $k$ is its **Voronoi cell**. For a fixed codebook, the nearest-neighbour rule is the best possible encoder; for a fixed partition into cells, the centroid of each cell is the best possible codeword. **Lloyd's algorithm** (1957 for one dimension; **k-means** when the statisticians rediscovered it) alternates those two optimality conditions. Each step can only lower the total squared error, so it converges; to a local minimum, not necessarily the global one.

{{< fig src="img/vector-quantization/piggy-map.png" alt="Left: six houses on the island with two centroids and the dashed Voronoi boundary between them. Right: a four-entry codebook with its four Voronoi cells and a query point landing in cell 2." caption="Left: the island after Lloyd's algorithm. Every house pays one bit, the index of its family's centroid. Right: a four-entry codebook; every point in a cell decodes to that cell's X, so the index is all the decoder needs, provided it owns the X's." >}}

Two failure modes matter in practice. Start both codewords inside one family and a third family on the island might never attract either: the greedy update cannot move a codeword across a region with nobody in it. Random restarts and k-means++ seeding (pick each initial codeword with probability proportional to its squared distance from the ones chosen so far) are the standard cures. Worse, a codeword that is nearest to nothing never gets updated and stays dead forever; in neural codecs, where the codebook is trained by gradient descent, this is **codebook collapse**, and the usual fix is to re-seed dead entries from live data.

{{< lab id="vq" >}}

If you would rather watch the iteration than read it: the panel below runs Lloyd steps one at a time on three hundred points, with the scalar grid of the same bit budget drawn underneath for comparison. The correlated setting is the next section's argument in motion.

{{< viz src="viz/vector-quantization-kmeans.html" height="460" >}}

## What the decoder must own

Take the four-entry codebook from the lab, $\{(-0.8,-0.6), (-0.2,0.7), (0.5,0.1), (0.9,-0.7)\}$, and a fresh point.

```python
import numpy as np
codebook = np.array([[-0.8, -0.6], [-0.2, 0.7], [0.5, 0.1], [0.9, -0.7]])
x = np.array([0.4, 0.2])
d = ((codebook - x) ** 2).sum(1)
print("squared distances", d.round(2), " index", d.argmin(), " bits", int(np.log2(len(codebook))))
print("decoder output", codebook[d.argmin()], " error", (codebook[d.argmin()] - x).round(2))
```
```
squared distances [2.08 0.61 0.02 1.06]  index 2  bits 2
decoder output [0.5 0.1]  error [ 0.1 -0.1]
```

The entire transmission is the bit string `10`. Hand that to a decoder without the codebook and it is the integer two; hand it to one with the *wrong* codebook and it is a confident, wrong point. The codebook is **side information**: knowledge the decoder holds before the first index arrives, and the only thing that gives the indices meaning. Where it lives is a design decision with consequences. In the CELP speech codecs inside every mobile phone the codebooks are printed in the standard, so the decoder is born with them. In a Faiss index the codebooks are trained on your vectors and written into the index file, so a different index cannot read your codes. In SoundStream the codebooks are part of the model checkpoint, versioned with the network that produced the vectors. Change the data distribution and the codebook is stale on both ends at once. The [side information](../side-information/) article follows this thread across the whole map; here, keep asking of any VQ system: how big is the codebook, who trained it, and how does the decoder get it?

{{< lab id="codebook" >}}

## Why two numbers together beat two numbers alone

Now the question that gives this family its reason to exist. Suppose the two coordinates of each point are correlated: a weather station reports temperature at the top and bottom of a hill, or two channels of a key vector in a transformer move together. Here are sixteen such pairs, and a budget of two bits per pair. The scalar quantizer's natural move is one bit per coordinate: send the sign of $a$ and the sign of $b$, reconstruct each at its mean magnitude. The vector quantizer's natural move is a four-entry codebook over the pair.

```python
import numpy as np
a = np.array([-2.0,-1.5,-1.2,-0.9,-0.6,-0.4,-0.2,-0.1, 0.1, 0.3, 0.5, 0.7, 1.0, 1.3, 1.6, 2.1])
b = np.array([-1.8,-1.6,-1.0,-1.1,-0.5,-0.2,-0.4, 0.1,-0.1, 0.4, 0.3, 0.9, 0.8, 1.4, 1.5, 1.9])
P = np.stack([a, b], 1)                       # 16 correlated pairs
# scalar: one bit per axis, reconstruct at ±(mean magnitude)
rec = np.stack([np.where(v >= 0, np.abs(v).mean(), -np.abs(v).mean()) for v in (a, b)], 1)
print("sign grid, 2 bits/pair: MSE per pair", ((P - rec) ** 2).sum(1).mean().round(3))
cells, n = np.unique((P >= 0).astype(int), axis=0, return_counts=True)
print("grid cells actually used:", {tuple(int(v) for v in c): int(k) for c, k in zip(cells, n)})
# vector: one 2-bit index per pair, k-means with K=4
C = P[[0, 5, 10, 15]].astype(float)
for _ in range(20):
    asg = ((P[:, None, :] - C[None]) ** 2).sum(-1).argmin(1)
    C = np.array([P[asg == k].mean(0) for k in range(4)])
err = ((P[:, None, :] - C[None]) ** 2).sum(-1).min(1).mean()
print("k-means K=4, 2 bits/pair: MSE per pair", err.round(3), "\ncentroids", C.round(2))
```
```
sign grid, 2 bits/pair: MSE per pair 0.761
grid cells actually used: {(0, 0): 7, (0, 1): 1, (1, 0): 1, (1, 1): 7}
k-means K=4, 2 bits/pair: MSE per pair 0.164 
centroids [[-1.4  -1.38]
 [-0.24 -0.22]
 [ 0.62  0.6 ]
 [ 1.67  1.6 ]]
```

Same two bits per pair, and the error drops by a factor of 4.6. Look at where the scalar grid spent its budget: it has four cells, and fourteen of the sixteen points fell into two of them. The cells $(a<0, b\ge0)$ and $(a\ge0, b<0)$, the off-diagonal quadrants, each hold a single straggler. Half of the grid's resolution went to regions where the data almost never is. The codebook, by contrast, put all four of its codewords on the diagonal where the data lives, spaced out along it.

{{< fig src="img/vector-quantization/grid-vs-joint.png" alt="Two panels with the same sixteen correlated points. Left: the sign grid's four reconstruction points, two of them serving one point each. Right: four k-means centroids strung along the diagonal with their Voronoi cells." caption="Two bits per pair, two ways. A scalar grid in d dimensions is a cube lattice, and correlated data does not fill a cube; it fills a thin diagonal slab. k-means puts the levels where the data is." >}}

That picture is the first aha, and the entropy view makes it exact. Quantize each coordinate into four bins and count:

```python
import numpy as np
a = np.array([-2.0,-1.5,-1.2,-0.9,-0.6,-0.4,-0.2,-0.1, 0.1, 0.3, 0.5, 0.7, 1.0, 1.3, 1.6, 2.1])
b = np.array([-1.8,-1.6,-1.0,-1.1,-0.5,-0.2,-0.4, 0.1,-0.1, 0.4, 0.3, 0.9, 0.8, 1.4, 1.5, 1.9])
def H(labels):
    _, n = np.unique(labels, return_counts=True); p = n / n.sum()
    return -(p * np.log2(p)).sum()
qa, qb = np.digitize(a, [-0.7, 0, 0.7]), np.digitize(b, [-0.7, 0, 0.7])   # 4 bins per axis
print("H(a) =", H(qa).round(3), " H(b) =", H(qb).round(3), " H(a)+H(b) =", (H(qa) + H(qb)).round(3))
print("H(a,b) =", H(qa * 4 + qb).round(3), " mutual information =", (H(qa) + H(qb) - H(qa * 4 + qb)).round(3))
print("joint cells occupied:", len(np.unique(qa * 4 + qb)), "of 16")
```
```
H(a) = 1.977  H(b) = 1.977  H(a)+H(b) = 3.954
H(a,b) = 2.352  mutual information = 1.602
joint cells occupied: 6 of 16
```

An ideal entropy coder behind a scalar quantizer pays $H(A) + H(B) \approx 3.95$ bits per pair, because it codes the two indices as if they were independent (a context model could recover some of this, but a fixed-length kernel-friendly code cannot). A coder that sees the pair as one symbol pays $H(A,B) \approx 2.35$ bits. The gap is not a tuning detail; it is a named quantity,

$$
H(A) + H(B) - H(A,B) = I(A;B),
$$

the **mutual information** between the coordinates, 1.6 bits here, and it is exactly what a scalar quantizer pays twice. The joint entropy can never exceed the sum of the marginals, with equality only when the coordinates are independent. So the vector quantizer's gain on correlated data is bounded below by nothing and above by the mutual information, and "the channels of a key vector are correlated" is a precise statement about how much a scalar grid is overpaying. (There is a second, smaller gain that exists even for independent coordinates: hexagons tile a plane more efficiently than squares, and the best lattices in high dimension beat the cube by up to 1.53 dB, about a quarter of a bit per coordinate. The correlation gain is the one that matters.)

This is the thesis behind the game's newest vector algorithm. Coupled Quantization (Zhang et al., NeurIPS 2024) measured the key and value channels of LLaMA-7B and found what the sixteen pairs show: the sum of per-channel entropies grows linearly with the number of channels grouped, the joint entropy grows sub-linearly. So it quantizes groups of 4 or 8 contiguous channels with a 256-entry codebook per group per layer: 8 channels sharing 8 bits is **one bit per channel**. At that rate a scalar quantizer destroys the model (LLaMA-2-13B WikiText-2 perplexity 4,709 for 1-bit scalar, fp16 4.57); CQ-8c8b lands at 6.55. Two more things in that paper belong here. The codebooks are trained by k-means weighted by **Fisher information**, so that centroids sit where the *loss* is sensitive rather than where the squared error is: Fisher weighting raises raw MSE and lowers perplexity, which is the [perceptual stage's](../perceptual-models/) move, a distortion measure fitted to the consumer, applied to a reader that is a transformer. And the centroids total about 1% of the model's weights: a VQ codebook-sized codebook, shipped once per model, exactly as the previous section warned.

{{< lab id="coupled" >}}

## The exponential, and two ways around it

Back to the mainland. It is 2017 and imagine an engineer, call her Mara, who runs a search service over a billion image embeddings of 768 floats each. At 4 bytes a float that is 3,072 bytes per vector and 3 TB in total; her budget is one server with a few hundred gigabytes of RAM, so she needs about 32:1 and settles on 96 bytes per vector. Vector quantization is obviously the tool: the embeddings are heavily correlated across dimensions, which is precisely when the codebook beats the grid.

Then she writes the number down. 96 bytes is 768 bits per vector, which means a codebook of $2^{768}$ entries. Not storable, not trainable, not searchable. The codebook size grows exponentially in the bits per vector, and her budget is a rounding error compared to what full VQ would need. This is the second aha: she cannot have one codebook, so she must *compose* small ones, and there are two ways to compose.

**Product quantization** splits the vector into sub-vectors and gives each its own small codebook.

```python
import numpy as np
x = np.array([0.9, 0.1, -0.3, 0.8])
cb1 = np.array([[0.8, 0.2], [-0.7, 0.1], [0.1, 0.9], [0.0, -0.8]])   # codebook for dims 0,1
cb2 = np.array([[0.5, 0.5], [-0.4, 0.7], [0.6, -0.6], [-0.7, -0.5]])  # codebook for dims 2,3
i = ((cb1 - x[:2]) ** 2).sum(1).argmin()
j = ((cb2 - x[2:]) ** 2).sum(1).argmin()
xhat = np.concatenate([cb1[i], cb2[j]])
print("code (i, j) =", (int(i), int(j)), " 4 bits;  decoded", xhat, " error", (xhat - x).round(2))
print("stored centroids:", len(cb1) + len(cb2), " reachable 4-D points:", len(cb1) * len(cb2))
```
```
code (i, j) = (0, 1)  4 bits;  decoded [ 0.8  0.2 -0.4  0.7]  error [-0.1  0.1 -0.1 -0.1]
stored centroids: 8  reachable 4-D points: 16
```

Eight stored 2-D centroids give sixteen reachable 4-D points, because the decoder can combine any entry of the first codebook with any entry of the second. Mara's version: 768 dimensions cut into 96 sub-vectors of 8, each with a 256-entry codebook, one byte per sub-vector, 96 bytes per vector. She stores $96 \times 256 = 24{,}576$ centroids and can decode $256^{96} \approx 10^{231}$ distinct vectors. The exponential is still there; it has moved from the storage to the *reachable set*, which is where she wants it.

{{< fig src="img/vector-quantization/pq-split.png" alt="Diagram: a four-float vector split in two halves, each half matched against its own four-entry codebook, producing the code (0,1); below, the four-by-four grid of the sixteen vectors the decoder can produce." caption="Product quantization composes codebooks by concatenation. The price is an assumption: the sub-vectors are quantized independently, so correlation across the split is paid twice, exactly the sin of the scalar grid at a coarser scale. The cure is to rotate first so the split cuts along weakly correlated directions." >}}

Two consequences made PQ the substrate of every billion-scale vector index. First, the assumption it makes is the same one the scalar grid made: the sub-vectors are quantized independently, so whatever mutual information crosses the split is paid twice. The fix is a rotation chosen so that the split separates nearly independent groups and balances their variance; that is OPQ, and it is why the Billion-scale vector search task pipeline has the [rotation algorithm](../rotations-and-rank/) in front of the splitter. Second, Mara never needs to decode. For a query $\mathbf{q}$, she computes, once, a table of the 256 squared distances from each query sub-vector to its codebook's entries ($96 \times 256$ floats), and the distance to any stored vector is then 96 table lookups and additions: $\|\mathbf{q} - \hat{\mathbf{x}}\|^2 = \sum_{m=1}^{96} T_m[\text{code}_m]$. This is **asymmetric distance computation**: the query stays exact, the database stays compressed, and the search runs in the compressed domain at memory-bandwidth speed. Fixed-length codes are the point: the 3,187th vector's code is at byte offset $96 \times 3187$, with no entropy coder in the way.

{{< lab id="pq" >}}

**Residual vector quantization** composes the other way: by addition instead of concatenation. Quantize the vector with a small codebook, then quantize the *error* with a second codebook, then the error of that with a third.

```python
import numpy as np
rng = np.random.default_rng(7)
X = rng.normal(0, 1, (400, 2)) @ np.array([[1.0, 0.9], [0.0, 0.45]])   # correlated 2-D cloud
def kmeans(P, K, steps=30):
    C = P[:K].copy()
    for _ in range(steps):
        asg = ((P[:, None] - C[None]) ** 2).sum(-1).argmin(1)
        C = np.array([P[asg == k].mean(0) for k in range(K)])
    return C
resid, books, total = X.copy(), [], np.zeros_like(X)
for stage in range(3):                       # 3 stages × K=4 → 6 bits per vector
    C = kmeans(resid, 4); books.append(C)
    idx = ((resid[:, None] - C[None]) ** 2).sum(-1).argmin(1)
    total += C[idx]; resid = X - total
    print(f"after stage {stage+1}: MSE {((X-total)**2).sum(1).mean():.3f}  (bits so far {2*(stage+1)})")
print("centroids stored:", sum(len(b) for b in books), " reachable points:", 4 ** 3)
```
```
after stage 1: MSE 0.286  (bits so far 2)
after stage 2: MSE 0.104  (bits so far 4)
after stage 3: MSE 0.043  (bits so far 6)
centroids stored: 12  reachable points: 64
```

Twelve stored centroids, 64 reachable points, and a property PQ does not have: the stream is **scalable**. Stop reading after two bits and you have a coarse vector; after four, a better one; the later stages only refine. SoundStream and EnCodec run eight stages of 1,024 entries on each latent frame: 80 bits per frame from 8,192 stored centroids instead of $1024^8$, and a decoder that can drop the last stages when the network gets slow. Those 80-bit codes are the "audio tokens" that language models over sound are trained on, which is why the [learned transforms](../learned-transforms/) article treats this quantizer as the hinge of the neural codec. The cost is greed: each stage is optimal given the previous ones, not jointly, and the codebook-collapse problem returns at every stage.

{{< lab id="rvq" >}}

Mara's final design, incidentally, is three vector quantizers stacked by role: a coarse k-means with a few thousand centroids to pick a cell to search (a plain VQ used as an index), an OPQ rotation, then PQ codes inside each cell. The game calls that pipeline the Billion-scale vector search task.

## The oldest VQ you already use

A GIF has at most 256 colours. The encoder takes the image's RGB triples (3-D vectors of bytes), runs a clustering algorithm over them, writes the 256 winners into the file's palette, and replaces every pixel by an 8-bit index. That is vector quantization with $d = 3$, $K = 256$, a 24-bit vector sent as 8 bits, and the palette is the codebook shipped in the header: side information travelling with the stream rather than installed in advance. The reason it looks fine on a logo and bad on a photo is the same reason the piggy centroid lands in the sea when the brothers scatter: VQ works when the data clusters.

## Connections

The algorithm at the centre of this article is better known as **k-means clustering**, and the two fields want different things from the same iteration. Clustering cares about the assignment (which family is this house in?); quantization cares about the centroid (what do I send?). The failure modes are shared, which is why k-means++ and restarts show up in both literatures.

The Voronoi cells are the diagram you get by asking "which post office is nearest to every point on the map": the nearest-neighbour rule *is* a Voronoi partition, and the centroid rule asks that each site sit at the centre of mass of its own cell. A codebook that satisfies both is a centroidal Voronoi tessellation.

The [scalar quantizer](../scalar-quantization/) is the $d = 1$ case: Lloyd-Max levels for a Gaussian are a one-dimensional codebook found by the same algorithm, and the gain of VQ over it is the mutual-information term plus the small lattice term.

**VQ-VAE** and its descendants turn the codebook into a vocabulary. An encoder network maps an image or a sound to vectors, a codebook maps those to indices, and a transformer is trained to predict index sequences; the "tokens" of VQGAN-style image generators, AudioLM and MusicGen are vector-quantizer indices, and the codebook is the thing that makes a continuous signal discrete enough to put a language model on. Compression machinery became a representation.

And at the far end of the map, [coupled quantization](../kv-cache-vaults/) is this article run on a KV cache: small groups of correlated channels, codebooks of 256, one bit per channel, Fisher weights instead of plain squared error, and a few megabytes of codebook per model that the decoder must own.

**In the game:** Vector quantization (vq), Product quantization (pq), Residual VQ (rvq), Coupled quantization (coupled) and the VQ codebook (codebook) who holds their keys; they take the field in the Billion-scale vector search task (Faiss IVF-OPQ-PQ), the Real-time speech task (SoundStream's residual codebooks) and the KV cache at 1–2 bits task (Coupled Quantization / CommVQ), and the conflict 'Vector quantization without a codebook' is what happens when a VQ algorithm has no codebook in the shared-prior stage.
