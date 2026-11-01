---
title: "Rate and distortion: the one irreversible chair"
slug: rate-and-distortion
chair: foundations
family: "Foundations"
summary: "If you may lose a little, how much do you save and where should you lose it: the Δ²/12 law, 6 dB per bit, Shannon's R(D) floor, reverse water-filling across coordinates, and why the consumer, not the mathematician, defines what counts as error."
fighters: [uniform, waterline, qmatrix]
related: [scalar-quantization, block-transforms, perceptual-models, modeling-and-coding]
reading_time: 15
---

## The step size nobody could justify

Imagine Bell Laboratories in the early 1950s. The company has decided that telephone speech will, one day, travel as numbers rather than as a wobbling voltage, and a young engineer, call her Ruth Kessler, has been handed the least glamorous part of the plan: pick the step. Every 125 microseconds the voltage on the line will be measured, and the measurement will be rounded to the nearest multiple of some step Δ and sent as an integer. Her manager wants one number. "What step?"

The stakes are not small. A trunk line between two cities carries many conversations, and each conversation will now cost a fixed number of bits per sample, 8,000 samples per second. One bit per sample, over a million lines, is a second trunk. The telephone company owns the copper and the repeaters, and every bit per sample Ruth adds is copper she must buy. But every bit she removes is noise on the line, and the customers can hear noise.

Her first instinct is the one everyone has: make Δ small enough that nobody complains, then stop. The trouble is that this answer has no price attached. It cannot tell her whether 7 bits is extravagant or 6 bits is reckless, because it does not say what a bit buys. The question she actually needs to answer, and the one this article is about, is:

> *If I may lose a little, how much do I save, and where should I lose it?*

Nobody at the lab can say "how much" because nobody has written down the exchange rate between bits and error. Let us find it with six numbers, the way she might have on a lunch break.

## Six numbers on a grid

Take six measurements, in whatever units the voltmeter reads: $x = [3.7, -1.2, 0.4, 8.9, -5.5, 2.1]$. Quantizing them with step $\Delta$ means computing the integer $q = \operatorname{round}(x/\Delta)$, sending $q$, and letting the receiver rebuild $\hat x = q\Delta$. The error is $x - \hat x$. Here it is for three steps, each twice the last:

```python
import numpy as np
x = np.array([3.7, -1.2, 0.4, 8.9, -5.5, 2.1])
def rnd(v):                                  # round half away from zero
    return np.sign(v) * np.floor(np.abs(v) + 0.5)
rng = x.max() - x.min()                      # 14.4
for step in (1, 2, 4):
    q = rnd(x / step); xh = q * step
    mse = np.mean((x - xh) ** 2)
    levels = int(q.max() - q.min() + 1)      # distinct indices actually spanned
    print(f"Δ={step}: q={q.astype(int).tolist()}  x̂={xh.tolist()}")
    print(f"      MSE={mse:.4f}  Δ²/12={step**2/12:.4f}  range/Δ={rng/step:.1f}"
          f"  log2={np.log2(rng/step):.2f}  levels={levels} -> {int(np.log2(levels))} bits")
```
```
Δ=1: q=[4, -1, 0, 9, -6, 2]  x̂=[4.0, -1.0, 0.0, 9.0, -6.0, 2.0]
      MSE=0.0933  Δ²/12=0.0833  range/Δ=14.4  log2=3.85  levels=16 -> 4 bits
Δ=2: q=[2, -1, 0, 4, -3, 1]  x̂=[4.0, -2.0, 0.0, 8.0, -6.0, 2.0]
      MSE=0.3267  Δ²/12=0.3333  range/Δ=7.2  log2=2.85  levels=8 -> 3 bits
Δ=4: q=[1, 0, 0, 2, -1, 1]  x̂=[4.0, -0.0, 0.0, 8.0, -4.0, 4.0]
      MSE=1.3933  Δ²/12=1.3333  range/Δ=3.6  log2=1.85  levels=4 -> 2 bits
```

Read the two columns that matter. Each time Δ doubles, the number of levels the six values span halves, 16 to 8 to 4, so the integers need exactly one bit less: 4, 3, 2. And each time Δ doubles, the mean squared error roughly quadruples: 0.09, 0.33, 1.39. The toy values are too few to make it exact, which is why the next column, $\Delta^2/12$, is printed beside them: 0.083, 0.333, 1.333. That is the prediction, and six samples already track it.

Where does $\Delta^2/12$ come from? If the step is fine compared with how the values are spread, the rounding error $e = x - \hat x$ is just as likely to be anywhere in $[-\Delta/2, \Delta/2]$. Its mean square is then

$$
D = \int_{-\Delta/2}^{\Delta/2} \frac{e^2}{\Delta}\,de = \frac{\Delta^2}{12}.
$$

That is the whole derivation, and it is worth more than it looks. It says the error does not depend on the data at all, only on the step. The data decide the *rate*: the number of distinct integers you must be able to name is roughly $\text{range}/\Delta$, so the rate is about $\log_2(\text{range}/\Delta)$ bits per sample, and doubling $\Delta$ subtracts exactly one bit from it. Put the two facts together and you have Ruth's exchange rate, the moment this article was written to deliver:

> Halving the step costs one bit per sample and divides the error by four. In decibels, every bit buys $10\log_{10} 4 = 6.02$ dB of signal-to-noise ratio.

With this, "what step?" becomes a budget question with a price list, not a matter of taste. Seven bits instead of six is 6 dB quieter noise, and 8,000 bits per second more per conversation. Now the trunk and the ear can be weighed against each other on one scale.

{{< lab id="uniform" >}}

Let us now name what the toy showed. A **scalar quantizer** maps each real value to one of a finite set of reconstruction values; the **uniform quantizer** with **step size** $\Delta$ uses the multiples of $\Delta$ and rounds to the nearest. The **distortion** is the expected error under some measure, by default the **mean squared error** $D = \mathbb{E}[(X-\hat X)^2]$. The **rate** $R$ is the number of bits per sample the quantized stream needs. The **high-resolution approximation** is the regime where $\Delta$ is small relative to the spread of $X$, in which $D \approx \Delta^2/12$ and, if an entropy coder follows the quantizer, $R \approx h(X) - \log_2\Delta$, where $h(X)$ is the differential entropy of the source. The two formulas are what make "6 dB per bit" a law rather than a coincidence of six numbers.

## Is there a floor?

Ruth has an exchange rate, but a rate between two currencies does not tell you whether you are being swindled. Rounding to a grid is one way to turn numbers into bits. Could a cleverer scheme get the same error for fewer bits? Could it get it for *far* fewer? In the [previous foundations article](../modeling-and-coding/) the answer for lossless coding was a floor, the entropy, that no coder can beat. Is there a floor here too, now that we are allowed to be wrong?

There is, and it was Shannon again who wrote it down. Define the **rate-distortion function** $R(D)$ of a source as the smallest number of bits per sample for which *some* coding scheme, of any cleverness and any block length, achieves average distortion at most $D$. It is a floor in exactly the sense entropy is a floor: below $R(D)$ bits per sample, no scheme can keep the error under $D$; above it, some scheme can. The function depends only on the source's distribution and the distortion measure, never on the coder.

For a Gaussian source of variance $\sigma^2$ under squared error it has a closed form that will follow you around the field:

$$
R(D) = \frac{1}{2}\log_2\frac{\sigma^2}{D}, \qquad 0 \le D \le \sigma^2 .
$$

Read it back as a slope: cutting $D$ by four costs one bit. The floor has the same 6 dB per bit exchange rate that rounding had. That is not an accident; it is the first thing to check, and the second is how far above the floor rounding sits. The uniform quantizer with an ideal entropy coder behind it pays $h(X) - \log_2\Delta$ bits for distortion $\Delta^2/12$. For a Gaussian, $h = \frac12\log_2(2\pi e\sigma^2)$, and substituting $\Delta^2 = 12D$ gives

$$
R_{\text{uniform}}(D) = \frac12\log_2\frac{\sigma^2}{D} + \frac12\log_2\frac{2\pi e}{12} = R(D) + 0.254 \text{ bits}.
$$

A quarter of a bit per sample, at every step size, equivalently 1.53 dB. Here is the same claim measured rather than derived, on 200,000 Gaussian samples with the rate taken as the entropy of the integers the quantizer emits:

```python
import numpy as np
rng = np.random.default_rng(0)
x = rng.standard_normal(200_000)                 # unit-variance Gaussian source
def entropy_bits(q):
    _, c = np.unique(q, return_counts=True); p = c / c.sum()
    return -(p * np.log2(p)).sum()
print(" Δ     D=MSE   R_quant  R(D)=½log2(1/D)  gap")
for step in (0.125, 0.25, 0.5, 1.0):
    q = np.round(x / step); xh = q * step
    D = np.mean((x - xh) ** 2); R = entropy_bits(q); RD = 0.5 * np.log2(1 / D)
    print(f"{step:5.3f}  {D:7.5f}  {R:6.3f}   {RD:6.3f}          {R-RD:5.3f}")
```
```
 Δ     D=MSE   R_quant  R(D)=½log2(1/D)  gap
0.125  0.00130   5.050    4.794          0.256
0.250  0.00521   4.052    3.792          0.260
0.500  0.02082   3.064    2.793          0.271
1.000  0.08342   2.107    1.792          0.315
```

The gap is 0.256 bits at fine steps and drifts up only when $\Delta$ approaches the standard deviation and the high-resolution assumption starts to fray.

{{< fig src="img/rate-and-distortion/rd_curve.png" alt="Rate against distortion for a Gaussian source: the Shannon R(D) curve and, a constant 0.25 bit above it, the line of uniform quantizers with an entropy coder, with the Δ = 1, 2, 4 operating points marked" caption="The R(D) plane for a Gaussian with the toy's variance. The solid curve is the floor; the dashed line is what rounding to a grid and entropy coding the integers achieves, 0.25 bit above it at every step. The three dots are Δ = 1, 2, 4. Doubling Δ moves one bit down and one factor of four to the right." >}}

This picture is the first thing to carry away. It is a *plane*, not a number. The horizontal axis is how wrong you are allowed to be; the vertical axis is what it costs; the floor is a curve, and a lossy codec is a point. "Quality 85 JPEG", "128 kbps MP3", "CRF 23 x264" and "Q4_K_M" are not different kinds of object. They are coordinates. When a paper says codec A beats codec B, it means A's point is below and to the left of B's. When a product has a quality slider, the slider walks the point along a curve. Everything else in the lossy half of the map, transforms, perceptual models, vector codebooks, is a way of pushing the codec's curve down toward the floor, and the Butcher's chair is where the point is finally chosen.

## Zeros are cheap: the dead zone

Before leaving the single-coordinate world, one practical twist that every video codec uses and that only makes sense once you think in rate *and* distortion. Rounding to nearest is the choice that minimises distortion for a given step. It is not the choice that minimises distortion for a given *rate*, because not every integer costs the same to send. After a transform, coefficients pile up around zero, and zero is by far the most common symbol; an entropy coder charges it a fraction of a bit. A quantizer that nudges borderline values toward zero therefore pays a little distortion and saves a lot of rate.

The H.264 reference encoder does exactly this with a rounding offset $f$ below one half: $q = \operatorname{sgn}(x)\lfloor |x|/\Delta + f\rfloor$, with $f = 1/3$ for intra blocks and $1/6$ for inter blocks. The bin around zero becomes wider than the others, a **dead zone**. On a Laplacian source, which is what DCT coefficients look like, the comparison at equal rate is:

```python
import numpy as np
rng = np.random.default_rng(1)
x = rng.laplace(scale=1 / np.sqrt(2), size=200_000)   # unit-variance Laplacian (a DCT coefficient)
def H(q):
    _, c = np.unique(q, return_counts=True); p = c / c.sum(); return -(p * np.log2(p)).sum()
def quant(step, f):                                   # H.264-style: q = sgn(x)·⌊|x|/Δ + f⌋
    q = np.sign(x) * np.floor(np.abs(x) / step + f); return q, np.mean((x - q * step) ** 2)
q, D = quant(1.0, 1 / 3)                              # dead zone, Δ = 1
print(f"dead zone f=1/3, Δ=1.00: zeros={np.mean(q==0):.3f}  R={H(q):.3f} bits  D={D:.4f}")
target = H(q)                                         # now give plain rounding the same rate
steps = np.linspace(1.0, 1.6, 601)
rates = np.array([H(quant(s, 0.5)[0]) for s in steps])
s = steps[np.argmin(np.abs(rates - target))]
q2, D2 = quant(s, 0.5)
print(f"plain rounding,  Δ={s:.2f}: zeros={np.mean(q2==0):.3f}  R={H(q2):.3f} bits  D={D2:.4f}")
```
```
dead zone f=1/3, Δ=1.00: zeros=0.611  R=1.762 bits  D=0.1009
plain rounding,  Δ=1.21: zeros=0.575  R=1.762 bits  D=0.1120
```

Same 1.762 bits per coefficient; the dead-zone quantizer is 10% lower in error. Plain rounding had to coarsen its whole grid to 1.21 to reach that rate; the dead zone kept the fine grid for the values that matter and spent its savings on the ones near zero. This is the smallest instance of a pattern that recurs through the whole Butcher's chair: the question is never only "how big is the error" but "what does each error cost in bits", and the [scalar quantization article](../scalar-quantization/) takes it further.

## Two coordinates and the water level

Ruth's job had one coordinate: a voltage. Most lossy coding has many. A block of eight pixels through a DCT becomes eight coefficients; a key vector in a transformer has 128 channels; a stereo pair becomes a mid and a side. And the coordinates are not alike. Suppose there are two, Gaussian, with variances $\sigma_1^2 = 64$ and $\sigma_2^2 = 1$, and you have a budget of bits to split between them. Where does the error go?

The instinct is fairness: the same number of bits each. Let us price the instinct using the Gaussian formula, which for a single coordinate at rate $R$ gives $D = \sigma^2 2^{-2R}$, and compare it with the allocation that actually minimises the total error:

```python
import numpy as np
var = np.array([64.0, 1.0])                      # two Gaussian coordinates
for budget in (2, 4):
    # equal split
    R_eq = np.full(2, budget / 2)
    D_eq = var * 2.0 ** (-2 * R_eq)
    # reverse water-filling: find θ so that Σ max(0, ½log2(σ²/θ)) = budget
    lo, hi = 1e-6, var.max()
    for _ in range(100):
        th = np.sqrt(lo * hi)
        R = np.maximum(0, 0.5 * np.log2(var / th))
        lo, hi = (th, hi) if R.sum() > budget else (lo, th)
    D_wf = np.minimum(var, th)
    print(f"budget {budget} bits | equal: R={R_eq.tolist()} D={D_eq.round(4).tolist()} ΣD={D_eq.sum():.4f}")
    print(f"              | water: θ={th:.3f} R={R.round(3).tolist()} D={D_wf.round(4).tolist()} ΣD={D_wf.sum():.4f}"
          f"  gain={10*np.log10(D_eq.sum()/D_wf.sum()):.2f} dB")
```
```
budget 2 bits | equal: R=[1.0, 1.0] D=[16.0, 0.25] ΣD=16.2500
              | water: θ=4.000 R=[2.0, 0.0] D=[4.0, 1.0] ΣD=5.0000  gain=5.12 dB
budget 4 bits | equal: R=[2.0, 2.0] D=[4.0, 0.0625] ΣD=4.0625
              | water: θ=0.500 R=[3.5, 0.5] D=[0.5, 0.5] ΣD=1.0000  gain=6.09 dB
```

Look at the two-bit case first, because it contains the surprise. The best thing to do with the small coordinate is to send *nothing*: zero bits, reconstruct it as its mean, accept its entire variance of 1 as error. All two bits go to the big coordinate, and the total error drops from 16.25 to 5, a factor of three. Equal allocation spent a bit polishing a coordinate whose whole error was 1 while leaving an error of 16 on the other. With four bits the small coordinate earns half a bit and both end at the same distortion, 0.5 each, and the gain over equal allocation is a full 6 dB, one entire bit per coordinate of effective efficiency.

The structure of the optimum has a name that describes the picture exactly. Pick a **water level** $\theta$. Every coordinate whose variance pokes above the water gets enough bits to bring its distortion down to $\theta$; every coordinate below the water is drowned, gets no bits, and contributes its own variance as distortion. Raise or lower $\theta$ until the bits add up to the budget:

$$
R_i = \max\!\left(0,\ \tfrac12\log_2\frac{\sigma_i^2}{\theta}\right), \qquad D_i = \min(\sigma_i^2,\ \theta), \qquad R = \sum_i R_i .
$$

This is **reverse water-filling**, and it is the rate-distortion function of a set of independent Gaussian coordinates. The reason it is optimal is the exchange rate again: a bit spent on any coordinate above water cuts that coordinate's distortion by the same factor of four, so the bit should go wherever the distortion is currently largest, and the process ends when all the surviving distortions are equal. Equal distortion, not equal bits. The one allocation that is certainly wrong is the fair-looking one.

{{< fig src="img/rate-and-distortion/waterfill.png" alt="Left: eight coordinate variances from 64 down to 0.5 as bars on a log scale, a dashed water level at θ = 2, the bits each coordinate receives printed above, and three coordinates drowned; right: per-coordinate distortion of water-filling against equal allocation at the same 7.5 bits" caption="Eight coordinates, water level θ = 2: bits 2.5, 2, 1.5, 1, 0.5 for the five above water and none for the three below, 7.5 bits in all, total distortion 13.5. The same 7.5 bits spread evenly leave a distortion of 34.8, four decibels worse, almost all of it on the coordinate that was starved." >}}

{{< lab id="waterline" >}}

A quiet consequence is worth stating because it reorganises the rest of the map. Water-filling only has anything to allocate if the variances differ. Eight pixels of a photograph have nearly equal variances; nothing to do. The same eight pixels after a DCT have one huge coefficient and seven small ones, and now the water level drowns most of them for free. That is what a transform is for: it does not remove information, it *skews the variances* so that water-filling can harvest the skew. The harvest has a name, **transform coding gain**, and the [block transforms article](../block-transforms/) shows it equals the ratio of the arithmetic to the geometric mean of the coefficient variances. JPEG's quantization table is a water-filling allocation written down by hand, one $\Delta$ per coefficient, which is why the SCULPTOR fighter sits in the Butcher's chair rather than the Lens's. NVIDIA's 2026 KV-cache codec, KVTC, does the same allocation with a dynamic program over PCA coordinates, which is the WATERLINE fighter.

## Who decides what "error" means?

Everything above measured error as mean squared difference, and the honest thing to say about that choice is that it was made for the mathematician's convenience, not the listener's. Squared error has a closed-form $R(D)$, it adds across coordinates, and it survives orthonormal transforms unchanged, which is exactly why water-filling works in the transform domain. But Ruth's customers do not hear mean squared error. They hear hiss on quiet passages and nothing at all on loud ones; they are deaf to a tone next to a louder tone; their eyes cannot resolve colour at the resolution they resolve brightness. Each of those is a statement that two reconstructions with the same squared error are not equally bad.

So the third and last thing to take from this article is a definition. **Distortion is defined by the consumer.** $R(D)$ is a theorem about whatever $D$ you choose, and the choice is a model of who is listening. MSE is the stand-in you use before you have a better one; the whole job of the Con Artist's chair, the [perceptual models](../perceptual-models/) and, for transformers, [attention as the audience](../attention-as-audience/), is to replace it with a measure under which the water can be made deeper where nobody is looking. The psychoacoustic model does not change a single formula here. It changes the $\sigma_i^2$ that water-filling sees, by dividing each coordinate's error by how audible it is, and then lets the mathematics run unchanged. The exchange rate is universal; the currency is the consumer's.

One more bound ties this back to the floor. For any source with differential entropy $h(X)$, under squared error,

$$
R(D) \ \ge\ h(X) - \tfrac12\log_2(2\pi e D),
$$

the **Shannon lower bound**, with equality for the Gaussian. The Gaussian is the hardest source of its variance to compress; every real signal, being more structured, has a lower floor. Most of the Oracle's and Lens's chairs are ways of exposing that structure so that the quantizer behind them can get nearer to it.

## Connections

The **JPEG quality slider** is one number that scales one table. At quality 50 the standard luma table has a $\Delta$ of 16 for the DC term and roughly 100 for the highest frequencies; the slider multiplies the whole table by a factor, so moving it walks every coefficient's operating point along its own 6 dB per bit line in lockstep, a one-parameter family of points in the R(D) plane. Video's **CRF** is the same idea with a quantizer parameter instead of a table: in H.264 the step size doubles every six QP units, which is the 6 dB per bit law written into the standard's integer arithmetic.

The **Shannon lower bound** is also why "Gaussian" keeps appearing as the worst case in papers on quantizing neural network weights and activations: a layer whose values are bell-shaped offers no structure for a predictor to find, and its R(D) is the floor's floor.

In a **variational autoencoder** the training loss is a reconstruction term plus a KL divergence $D_{\mathrm{KL}}(q(z|x)\,\|\,p(z))$, and the KL term is literally a rate: it is the number of bits needed to transmit the latent $z$ under the prior $p(z)$. A VAE is a point on an R(D) plane chosen by its $\beta$, and the learned image codecs of the [learned transforms article](../learned-transforms/) make this explicit by replacing the KL with the bits an arithmetic coder actually spends.

## In the game

The Butcher's chair is this article's chair, and [GRID](/compressionfight/#fighter=uniform) is Ruth's step size made into a fighter. [WATERLINE](/compressionfight/#fighter=waterline) is reverse water-filling as a bit allocator in THE COLD STORAGE, and [SCULPTOR](/compressionfight/#fighter=qmatrix) is the same allocation hand-written as JPEG's table in THE GALLERY. Every heist from level 2 upward, THE GALLERY, WALKMAN, PRIME TIME, THE WEIGHTS VAULT and all of THE VAULT, is a position on the R(D) plane with a consumer attached.
