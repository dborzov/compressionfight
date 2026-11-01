---
title: "Learned transforms: the neuron"
slug: learned-transforms
chair: transform
family: "Learned transforms"
summary: "A DCT assumes signals are sums of cosines; a trained encoder, quantizer and decoder learn what speech or faces are, so the wire carries only which one, the decoder holds the prior, and the fidelity contract quietly changes from accurate to plausible."
fighters: [neural_enc, decoder_net, gan, mla, rvq]
related: [vector-quantization, perceptual-models, side-information, kv-cache-vaults]
reading_time: 14
---

## Three kilobits per second

Imagine, because this is a story, a researcher in the spring of 2020 whose problem is a phone line that is not a phone line. Her users talk over a mobile link that drops to a few kilobits per second whenever the signal is poor, and her job is to make a voice call survive it. The best general-purpose codec she has, Opus, is excellent down to about 6 kbps and then runs out of things to throw away; she needs intelligible, natural speech at 3 kbps, half of that, with less than a frame of delay. Every number she has learned about audio coding says no. A CD carries 1411 kbps. MP3 is transparent at around 128. She is being asked for a forty-to-one improvement on a codec that was already forty-to-one better than MP3's ancestors.

So she goes back to first principles and asks why the existing codecs stop where they do. MP3 and Opus's music mode run an MDCT, which is a fixed transform: it assumes the signal is locally a sum of cosines, which is true of any sound, and it tells the quantizer nothing about *which* sounds. Opus's speech mode runs linear prediction, which assumes the signal is a resonance driven by an excitation, which is true of a vocal tract, and it is the reason Opus beats MP3 on speech at low rates by a wide margin. Each step down in bitrate came from a stronger assumption about what the signal is. Her question, and the question of this article, is the next step: *a fixed DCT assumes signals are sums of cosines; what if the transform could learn what speech is?*

## A frame, a latent, a budget

Here is the arithmetic she starts from. One frame of 20 ms of 16 kHz speech is 320 samples of 16 bits.

```python
frame, bits_per_sample = 320, 16                 # 20 ms at 16 kHz
stages, entries = 8, 1024                         # residual VQ: 8 codebooks of 1024
bits_in  = frame * bits_per_sample
bits_out = stages * entries.bit_length() - stages # log2(1024) = 10 per stage
print("bits in :", bits_in, "  bits out:", bits_out, "  ratio", bits_in // bits_out, ": 1")
frames_per_s = 16000 / frame
print("wire    :", frames_per_s * bits_out / 1000, "kbit/s  =", frames_per_s * bits_out / 8, "bytes/s")
print("SoundStream at 75 frames/s:", 75 * bits_out / 1000, "kbit/s  =", 75 * bits_out / 8, "bytes/s")
decoder_bytes = 20e6
print("decoder weights / wire rate :", round(decoder_bytes / (75 * bits_out / 8) / 3600, 1), "hours of speech")
print("MP3 128 kbit/s vs 6 kbit/s  :", round(128 / 6, 1), "x")
```
```
bits in : 5120   bits out: 80   ratio 64 : 1
wire    : 4.0 kbit/s  = 500.0 bytes/s
SoundStream at 75 frames/s: 6.0 kbit/s  = 750.0 bytes/s
decoder weights / wire rate : 7.4 hours of speech
MP3 128 kbit/s vs 6 kbit/s  : 21.3 x
```

The plan has three boxes. An **encoder** network squeezes the 320 samples into a short vector, say 8 floats, a *latent*. A **quantizer** turns the 8 floats into a handful of integers; the one she picks is residual vector quantization, 8 stages each with a 1024-entry codebook, so 8 indices of 10 bits, 80 bits per frame. A **decoder** network takes the 80 bits, looks up the 8 codewords, adds them, and synthesises 320 samples. Five thousand one hundred and twenty bits in, eighty out: 64:1 before any entropy coding, with 4 kbps on the wire at her frame rate, or 6 kbps at the 75 frames per second SoundStream actually uses on 24 kHz audio. Drop to four stages and it is 3 kbps; the stages are ordered coarse to fine, so a congested link can simply stop sending the last ones and the decoder still produces speech, rougher. That bitrate scalability is a gift of the [residual VQ](../vector-quantization/) structure and it costs nothing.

The last two lines are the napkin math that tells you what kind of object this is. The decoder weights are about 20 MB. At 750 bytes per second on the wire, the decoder is as big as seven and a half hours of transmitted speech. And against MP3 at 128 kbps, 6 kbps is a factor of 21. Hold those two numbers together: the ratio is enormous, and the thing that makes it possible is 20 MB that never crosses the wire.

{{< fig src="img/learned-transforms/pipeline.png" alt="Block diagram of a neural audio codec: waveform frame, encoder, residual VQ, decoder, with the training loss attached to all three and the codebooks and decoder weights shipped to both ends" caption="The pipeline, with its bit budget. One 320-sample frame becomes 80 bits. The loss at the bottom is back-propagated through the encoder, the quantizer and the decoder at once, which is what makes the three boxes one object." >}}

## What a trained transform is, in the smallest case

Before the big networks, the question deserves a small honest example. What does an encoder-decoder pair learn when you train it, and how does that relate to the transforms of the previous two articles? Make the encoder and the decoder as simple as possible: linear maps, two dimensions to one and back, with the same weight vector $w$ in both, trained by gradient descent on the reconstruction error.

```python
import numpy as np
rng = np.random.default_rng(1)
X = rng.standard_normal((500, 2)) @ np.array([[2.0, 1.6], [0, 0.5]]); X -= X.mean(0)
w = np.array([0.1, -0.3])                       # encoder z = X w, decoder x_hat = z w^T (tied)
for step in range(3000):
    z = X @ w; E = X - np.outer(z, w)           # reconstruction error
    grad = -2 * (X.T @ (E @ w) + E.T @ z) / len(X)
    w -= 0.02 * grad
lam, V = np.linalg.eigh(np.cov(X.T, bias=True))
print("learned direction :", (w / np.linalg.norm(w)).round(4), " |w| =", np.linalg.norm(w).round(4))
print("PCA eigenvector   :", (V[:, 1] * np.sign(V[0, 1])).round(4))
print("autoencoder MSE   :", ((X - np.outer(X @ w, w))**2).sum(1).mean().round(5), " lambda_2 =", lam[0].round(5))
```
```
learned direction : [-0.7734 -0.6339]  |w| = 1.0
PCA eigenvector   : [0.7734 0.6339]
autoencoder MSE   : 0.15226  lambda_2 = 0.15226
```

The network started from a random direction and, with nothing but a squared-error loss, found the first principal component of the data, to four decimals, with unit norm, and its reconstruction error is exactly the second eigenvalue. A linear autoencoder trained with MSE *is* PCA. That is reassuring and deflating at once. Reassuring, because it says the learned transform is the same thing the [rotations article](../rotations-and-rank/) called the KLT: the optimal linear decorrelator, now found by gradient descent instead of an eigensolver. Deflating, because PCA was not what she needed. Everything a neural codec does beyond PCA must therefore come from the two things this toy lacks: nonlinearity, and a loss that is not squared error.

{{< fig src="img/learned-transforms/linear_ae.png" alt="A 2-to-1-to-2 linear autoencoder's weight vector converging onto the PCA eigenvector, and its loss curve landing on the second eigenvalue" caption="A linear autoencoder trained with MSE finds PCA. The learned direction rotates onto the first eigenvector and the loss stops at λ₂, the variance of the axis it dropped." >}}

Nonlinearity is what lets the encoder say things like "this frame is a voiced vowel with pitch 140 Hz and this formant pattern" instead of "this frame has these eight cosine amplitudes". A DCT basis is fixed; a convolutional encoder's basis is whatever the training data made it, and for speech that is phonetic. The quantizer then does not quantize cosine amplitudes, which are nearly Gaussian and need many bits, but a latent that the encoder has been shaped to make quantizable.

{{< lab id="neural_enc" >}}

## Aha one: the pipeline is one object

Here is the move that separates a learned codec from a DCT with a neural coat of paint, and it is a question of what is held fixed. In JPEG the transform was designed in 1974, the quantization table in 1990, and the Huffman tables separately again; each stage was optimised on its own under assumptions about its neighbours. In the 2020 researcher's codec all three boxes have parameters, and the loss is computed on the *output of the decoder* and back-propagated through the decoder, through the quantizer, and into the encoder, in one step.

Think about what that means for each box. The encoder is not rewarded for producing a faithful latent; it is rewarded for producing a latent that, *after* being snapped to the nearest codewords and *after* being decoded, sounds right. So it learns to put its outputs where the codebooks are and to spend its 8 dimensions on what the decoder can use. The codebooks are not fitted to a fixed latent distribution, as a Lloyd codebook would be; they move toward wherever the encoder is putting latents, which is moving toward them. And the decoder is not a generic inverse transform; it is trained to reconstruct from *these* codewords, which gives it the freedom to fill in everything the 80 bits left out with whatever is typical of the training data. The latents are shaped for the codebooks, the codebooks for the decoder, the decoder for the loss. The pipeline is one object, and no stage makes sense without the others.

The small gradient problem in the middle, that rounding to a codeword has zero derivative almost everywhere, is handled with the straight-through estimator: in the forward pass use the quantized codeword, in the backward pass pretend the quantizer was the identity. A commitment term keeps the encoder's outputs from drifting away from the codebooks that the straight-through trick is pretending do not exist.

```python
import numpy as np
plausible = np.array([-1.0, 1.0])                  # two equally likely true outputs
for guess in [-1.0, -0.5, 0.0, 0.5, 1.0]:
    print(f"guess {guess:+.1f}: expected squared error {((plausible - guess)**2).mean():.2f}")

v = np.array([0.73, -0.41])                        # one latent to quantize with 3 stages x 4 entries
books = [np.array([[.5, -.5], [-.5, -.5], [.5, .5], [-.5, .5]]),
         np.array([[.25, 0], [-.25, 0], [0, .25], [0, -.25]]),
         np.array([[.1, .1], [-.1, .1], [.1, -.1], [-.1, -.1]])]
rec, codes = np.zeros(2), []
for cb in books:
    i = np.argmin(((cb - (v - rec))**2).sum(1)); codes.append(int(i)); rec = rec + cb[i]
    print("code", i, " reconstruction", rec, " residual norm", np.linalg.norm(v - rec).round(3))
print("sent:", codes, "=", 3 * 2, "bits for a 2-D vector")
```
```
guess -1.0: expected squared error 2.00
guess -0.5: expected squared error 1.25
guess +0.0: expected squared error 1.00
guess +0.5: expected squared error 1.25
guess +1.0: expected squared error 2.00
code 0  reconstruction [ 0.5 -0.5]  residual norm 0.247
code 0  reconstruction [ 0.75 -0.5 ]  residual norm 0.092
code 1  reconstruction [ 0.65 -0.4 ]  residual norm 0.081
sent: [0, 0, 1] = 6 bits for a 2-D vector
```

The second half of that script is residual VQ in miniature: the first codebook approximates the latent, the second approximates the error of the first, the third the error of that, and the three indices together name a point among $4^3 = 64$ effective centroids although only 12 were stored. SoundStream's 8 stages of 1024 name one of $1024^8 \approx 10^{24}$ points with 8,192 stored vectors. Note the stage that helps least (the third, 0.092 to 0.081): each stage is greedy, which is the price of the scheme. And note the first half of the script, which is the next aha.

{{< lab id="rvq" >}}

## Aha two: MSE is the average, and the average is blurry

The researcher's first trained model works, in the sense that the loss goes down, and it sounds terrible: muffled, hollow, as if the speaker were under a blanket. Every pixel-level or sample-level learned codec of that period had the same first result, and the reason is in the top five lines of the output above.

Suppose there are two equally plausible true outputs for what the decoder has been told, $-1$ and $+1$. The 80 bits said "a vowel, roughly like this", and the fine structure of the waveform, the exact phase of every harmonic, is one of the things they did not say. A decoder trained with squared error must produce one output, and the output that minimises expected squared error against two equally likely targets is $0$: error 1.00, against 2.00 for guessing either target. The MSE-optimal decoder emits the *mean of the plausible outputs*. In a waveform the mean of two sine waves with different phases is a smaller sine wave; the mean over many is near silence at that frequency. In an image, the mean of all plausible textures is a smooth blur. MSE training does not produce bad output. It produces the mathematically correct answer to the wrong question: "what single waveform is closest on average to all the waveforms this code could have meant?"

Her fix, and the one SoundStream and EnCodec and every neural codec since have used, is to change the question. Add a second network, a **discriminator**, trained to tell decoded audio from real audio, and train the decoder to fool it. The discriminator does not care about squared error; it cares whether the output has the statistics of real speech: real harmonics with a definite phase, real noise in the fricatives, real texture. The only way for the decoder to fool it is to commit to *one* plausible output rather than averaging them all. The reconstruction loss keeps that output close to the original on coarse structure (SoundStream and EnCodec measure it on multi-scale spectrograms, which are phase-blind on purpose), a feature-matching loss keeps the discriminator's internal activations similar, and the adversarial term supplies the detail. At 3 kbps the result is speech that is clearly the speaker, clearly the words, and in waveform terms not the original signal at all.

That last clause is the deep part. The fidelity contract changed. A DCT codec at any quality is *accurate*: its error is bounded and measurable, and you can turn the quality up until the error is below any threshold you name. A GAN-trained decoder at 3 kbps is *plausible*: the output is a sample from the set of sounds consistent with the 80 bits, chosen to be realistic rather than close. The [perceptual models](../perceptual-models/) article treats the discriminator as what it is, a perceptual model nobody had to derive, and the fighter card calls its failure mode by name: confidently wrong. A phoneme can come out as a different plausible phoneme. A face in a learned image codec can come out with a plausible but different texture of skin. Nothing in the output warns you. For a voice call this is a fine trade; for a court transcript or a medical scan it is not a codec at all.

{{< lab id="decoder_net" >}}

## Where the knowledge lives

Now the 20 MB. In the taxonomy of [where the prior lives](../side-information/), every codec has a model of its data somewhere. A DCT's model is "smooth": it is in the standard, costs nothing, applies to everything, and helps a little. An LPC speech codec's model is "a resonant tube", in the standard, a few coefficients per frame on the wire. A neural codec's model of speech is the decoder weights: tens of megabytes, trained once, installed at both ends, never transmitted. The wire carries only the index into that knowledge: *which* speech, not *what* speech is. The 21× against MP3 is almost entirely the prior.

The same shape appears wherever a learned transform is used, with the loss and the side information rearranged for the job. Learned image codecs in the line of Ballé's work keep the encoder-quantizer-decoder skeleton but replace the fixed-rate VQ with a scalar quantizer and an *entropy coder whose probability model is itself learned*: a small second network, the hyperprior, predicts the distribution of each latent from a few transmitted side bits, and the rate term in the training loss is literally $-\log_2 p(\hat y)$, the bits the arithmetic coder will spend. Training then minimises distortion plus $\lambda$ times rate, and $\lambda$ is the quality dial; a different $\lambda$ is a different model. EnCodec takes a half step the same way: an optional small language model over the RVQ tokens feeds an arithmetic coder and buys another 25–40% over the fixed 80 bits per frame, at the cost of a Packer that must run at both ends. The VAE inside a latent diffusion model is the same object again under a different name: a learned encoder shrinks a 512×512×3 image to a 64×64×4 latent, 48 times fewer numbers, and the diffusion model never sees pixels. It is a codec whose consumer is another network, which is as pure an example of "distortion is defined by the consumer" as the [rate-distortion](../rate-and-distortion/) article could want.

And the failure modes follow from the same place. The prior is the training distribution. A speech codec trained on read English handed a cello, or a tonal language, or a speaker with a condition the corpus lacked, has no codeword for it and will produce the nearest plausible thing it does know. The fixed DCT degrades gracefully on everything because it assumes almost nothing; the learned transform is spectacular inside its distribution and strange outside it.

{{< fig src="img/learned-transforms/rd_sketch.png" alt="Qualitative sketch of perceived quality against bitrate on a log axis for an MP3-like, an Opus-like and a neural speech codec" caption="A qualitative sketch, not measured data: each stronger assumption about what the signal is moves the whole curve to the left. The neural codec's curve starts at a few kbps because the decoder already holds the prior; it also flattens early, because what it reproduces is plausibility, not the waveform." >}}

## Aha three: the same move, baked into the architecture

The researcher's codec learned a transform after the fact, for a signal that already existed. There is a version of the same idea in which the transform is decided *before* the data exists, and it has done more to shrink the KV cache of language models than every codec applied afterwards.

A standard attention layer with 128 heads of dimension 128 caches, per token per layer, a key and a value for every head: $2 \times 128 \times 128 = 32{,}768$ numbers. Multi-head Latent Attention, in DeepSeek-V2 and its successors, instead caches one 512-dimensional latent $c$ per token, from which the keys and values of all 128 heads are *defined* to be linear functions, $k_h = W_{UK,h}\, c$ and $v_h = W_{UV,h}\, c$. Plus one 64-dimensional key for RoPE, which cannot pass through a linear bottleneck and is kept separately. That is 576 numbers per token per layer instead of 32,768, 57 times fewer. The up-projections are absorbed into the query and output weights at inference, so the cache is never expanded; attention runs on the latent directly.

Compare this with the [rank-160 SVD](../rotations-and-rank/) that ShadowKV fits to the keys of a prompt after prefill. ShadowKV discovered that keys lie near a subspace and exploited it; MLA *decreed* that keys and values would lie in a subspace of dimension 512, and trained the model from scratch under that constraint. Because the constraint was present during pretraining, the model learned to make the most of it: DeepSeek reports that MLA is stronger than full multi-head attention, not weaker, at a cache the size of grouped-query attention with 2.25 groups. GQA, sharing one key-value pair among several query heads, is the simpler ancestor of the same move. DeepSeek-V4 extends it along the token axis, caching one entry per 4 or per 128 tokens in most layers, which is the learned-transform version of chunking.

The trade is as sharp as it is in the audio codec, only moved in time. The cost at inference is zero: no encoder, no decoder, no side information beyond the model itself. The choice after training is also zero: you cannot turn the bottleneck off, change its rank, or apply it to a model that was not trained with it. Everything downstream, every quantizer and selector and offload scheme in the [KV cache vaults](../kv-cache-vaults/), works on what the architects left.

{{< lab id="mla" >}}

## Connections

**Autoencoders and PCA.** The linear case was computed above: a linear autoencoder trained with squared error converges to the span of the top principal components, a result published in 1989, before the deep versions. Every deeper autoencoder can be read as "PCA, plus whatever nonlinearity and loss buy you", and for codecs they buy most of the ratio.

**Tokens as the interface.** SoundStream's 8 indices per frame were built as a compression format and became a representation: AudioLM and MusicGen generate audio by predicting those indices with a language model and handing them to the same decoder. A codec whose quantizer outputs discrete symbols is, by construction, a tokenizer, and the decoder is the vocoder.

**The VAE's rate term.** The evidence lower bound of a variational autoencoder is a reconstruction term minus $D_{KL}\big(q(z \mid x)\,\|\,p(z)\big)$. Read through the [bits-are-surprise](../modeling-and-coding/) lens, that KL term is the number of bits needed to transmit the latent under the prior $p(z)$, so the ELBO is a rate-distortion objective with $\lambda = 1$. Ballé's hyperprior codecs and the diffusion VAE are both this equation with the rate term made explicit.

**In the game:** NEURON (learned encoder), MATRYOSHKA (residual VQ), FORGER (adversarial loss), BRAINJAR (the decoder as prior) and BOTTLENECK (MLA) appear in THE PHONE CALL (SoundStream / EnCodec), THE SHOWROOM (learned image codecs as an accepted squad) and THE ARCHITECT'S CUT (GQA / MLA / DeepSeek-V4), where the Lens is the only chair with a budget and it is spent before pretraining.
