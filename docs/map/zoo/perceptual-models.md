---
title: "Perceptual models: the con artist"
slug: perceptual-models
chair: perceptual
family: "Human perceptual models"
summary: "If the error metric is wrong, every optimal codec is optimal for the wrong thing: what the ear and the eye actually fail to notice, how masking and chroma subsampling cash that in, and why a trained discriminator is a perceptual model nobody had to derive."
fighters: [psychoacoustic, chroma, gan]
related: [rate-and-distortion, block-transforms, learned-transforms, attention-as-audience]
reading_time: 14
---

# Perceptual models: the con artist

Imagine Erlangen, 1989. An engineer at the Fraunhofer institute, call her Katrin, is wearing headphones in a listening booth, and the piece of music she is hearing has been crushed to one twelfth of its size by the group's experimental coder. Her job is to find the damage. She plays the original, then the coded version, then the original again. She cannot tell them apart. Down the corridor, the measurement rig says the coded signal has a signal-to-noise ratio of about 15 dB, which in a telephone line would be unusable hiss. The rig is not wrong. Her ears are not wrong. The two instruments are measuring different things, and the whole of this article is the gap between them.

Then someone plays castanets. Each click, in the coded version, is preceded by a faint smear of noise, a ghost arriving a few milliseconds before the sound that caused it. The rig barely registers it. Katrin hears it on the first pass. (The story is invented; the problem it describes, the pre-echo on transient-heavy test tracks, is the real one the MP3 developers had to solve, and the broad lineage from Fraunhofer's work to MPEG-1 Layer III in 1993 is real too.)

The question this chair exists to answer: *if the error metric is wrong, every "optimal" codec is optimal for the wrong thing, so what does the consumer actually not notice?* The [rate-distortion](../rate-and-distortion/) article ended by saying that mean squared error is a stand-in and that this chair's job is to replace it. Here is the replacement, for the ear, the eye, and finally for a consumer nobody can interview.

## Why squared error lies

The Butcher's trade is bits for distortion, and so far distortion has meant $\sum (x - \hat{x})^2$. That number treats every sample alike: a unit of noise in a silent passage costs the same as a unit of noise under a trumpet. The ear does not. Two of its properties break the metric at once. Loudness is logarithmic: the same absolute noise power is deafening next to silence and nothing next to a fortissimo, which is why audio is measured in decibels at all. And the ear is a bank of overlapping band-pass filters, roughly two dozen **critical bands** across the audible range, and what happens in one band is nearly independent of what happens three bands away. A metric that does not know about bands and does not know about levels will spend bits where nobody is listening.

So the naive fix, "weight the squared error by the ear's sensitivity curve," is the right instinct and not nearly enough, because the ear's sensitivity is not a fixed curve. It depends on what else is playing. That dependence is the heart of the matter.

## Masking: the loud tone hides its neighbours

Play a pure tone at 80 dB and, while it plays, try to hear a quieter tone nearby in frequency. You cannot, unless it is loud enough; and "loud enough" is much louder than it would need to be in silence. The loud tone has raised the **threshold of audibility** around itself, steeply on the low-frequency side and gently on the high-frequency side. The quiet tone is **masked**. This is **simultaneous masking**, and it is the single most valuable fact in audio coding, because quantization noise is a quiet signal nearby in frequency, and an encoder gets to choose how much of it to make in each band.

Here is the toy. Twenty-four bands, one loud tone at 80 dB in band 8, a spreading function that falls 25 dB per band below the masker and 10 dB per band above it, and a masked threshold that sits 15 dB under the spread curve. The rest of the signal is a floor of quiet texture at 30 dB with three modest components.

```python
import numpy as np
bands = np.arange(24)
tone_band, tone_level = 8, 80.0                                  # one loud tone, 80 dB in band 8
spread = np.where(bands < tone_band, -25.0 * (tone_band - bands), -10.0 * (bands - tone_band))
mask = tone_level + spread - 15.0        # masked threshold: the tone's level, spread, 15 dB down
threshold = np.maximum(mask, 20.0)       # never below the threshold in quiet (20 dB in this toy)
signal = np.full(24, 30.0); signal[tone_band] = 80; signal[[3, 14, 20]] = [50, 48, 42]
free = signal < threshold
print("threshold per band:", threshold.astype(int))
print("free bands (signal already under the curve):", np.flatnonzero(free), "->", free.sum(), "of 24")
bits_masked = np.maximum(signal - threshold, 0) / 6.02          # 6.02 dB of SNR per bit
bits_quiet = np.maximum(signal - 20.0, 0) / 6.02
print("bits for the frame: masking", bits_masked.sum().round(1), " vs no masking", bits_quiet.sum().round(1))
print("the tone itself: ", bits_quiet[tone_band].round(1), "->", bits_masked[tone_band].round(1), "bits")
```
```
threshold per band: [20 20 20 20 20 20 20 40 65 55 45 35 25 20 20 20 20 20 20 20 20 20 20 20]
free bands (signal already under the curve): [ 7  9 10 11] -> 4 of 24
bits for the frame: masking 41.5  vs no masking 56.5
the tone itself:  10.0 -> 2.5 bits
```

Read the two savings separately, because they are different mechanisms. Four bands, the ones adjacent to the tone, hold content that is *already below* the masked threshold: whatever is there cannot be heard while the tone plays, so the encoder sends nothing for them. They are free. That is the obvious saving and it is the smaller one. The larger one is in the tone's own band. Without a masking model the encoder would need the noise 60 dB under the 80 dB tone to be sure it was inaudible, ten bits; with the model it knows the noise only needs to sit 15 dB under the tone, two and a half bits. The same holds in every band the curve touches. The masking model does not merely find silence to skip; it rewrites the required signal-to-noise ratio of every band from "below the threshold in quiet" to "below the masked threshold," and since the [rate-distortion](../rate-and-distortion/) article's rule is 6.02 dB per bit, every 6 dB the threshold rises is one bit per band the encoder no longer owes.

{{< fig src="img/perceptual-models/masking-curve.png" alt="Top: 24 band levels as bars with one 80 dB masker in band 8; the masked threshold is a peaked white curve falling 25 dB per band to the left and 10 dB per band to the right; four bars next to the masker sit under it and are labelled free. Bottom: bits per band with and without the masking model." caption="The toy frame. The masked threshold is where quantization noise becomes audible; the encoder's job is to put noise just under the curve in every band. Four bands are free outright, and the masker's own band drops from 10 bits to 2.5." >}}

Now the definitions the toy was standing in for. The **masking threshold** $T(b)$ in band $b$ is the level below which an added signal is inaudible given everything else in the frame. It is built by taking each significant component (tonal or noise-like; tones mask less than noise of the same level), spreading its level across neighbouring bands with a **spreading function** (the asymmetric slopes above are a crude one; real models use curves fitted to listening tests on the Bark scale), subtracting a masking offset, and combining the contributions, floored by the **threshold in quiet**, the ear's sensitivity curve in silence. The quantity the encoder actually uses is the **signal-to-mask ratio** $\mathrm{SMR}(b) = L(b) - T(b)$, the number of decibels of noise suppression band $b$ needs, and the bit allocation follows from

$$
R(b) \approx \max\!\left(0,\ \frac{\mathrm{SMR}(b)}{6.02}\right) \ \text{bits per sample},
$$

iterated against the real quantizer until the frame's bit budget is met. In MP3 the encoder runs an FFT alongside the MDCT purely to compute this; the result steers per-band scale factors that set the quantizer step. That is the first aha, and it is the arithmetic behind the number everyone knows: MP3 at 128 kbit/s against 1,411 kbit/s for CD audio, eleven to one, is mostly noise placed under a curve that the listener cannot see over.

Notice what is *not* in the bitstream. The threshold, the spreading function, the FFT: none of it is transmitted. The decoder receives scale factors and quantized coefficients and does arithmetic; it never runs the model and never knows why the encoder chose what it chose. The perceptual model is **encoder-only**. That is why two MP3 encoders at the same bitrate can sound different, why the format could be frozen in 1993 while encoders kept improving for twenty years, and why this chair never adds a byte of decoder complexity.

{{< lab id="psychoacoustic" >}}

## Pre-echo, and time

Back to the castanets. Masking also works in time: a loud sound masks quieter sounds for some tens of milliseconds *after* it stops (**post-masking**), and, strangely, for a few milliseconds *before* it starts (**pre-masking**), because the auditory system takes time to register a sound. A few milliseconds is the key number. The MDCT in MP3 works on frames of 576 coefficients per granule, and the window it analyses spans about 26 ms at 44.1 kHz; the [block transforms](../block-transforms/) article explains why long windows are good (frequency resolution, and latency is free for a music file). But quantization noise in the transform domain spreads over the whole window in time. If a click happens at the end of the window, the noise it causes smears across the 20 ms of near-silence *before* it, far outside the few milliseconds pre-masking can hide. That is the ghost Katrin heard.

The cure is **block switching**: a transient detector in the encoder (again, encoder-only) spots the click and switches that frame to three short windows of 192 coefficients each, each spanning roughly 9 ms instead of 26, confining the noise to a few milliseconds around the event, at the cost of frequency resolution for that frame. AAC, Opus's CELT mode and every later codec have some version of it. The lesson generalises: a perceptual model is a model of *when* as much as *what*, and the codec's structural dials, window length among them, are set by what the consumer can and cannot resolve.

## The eye: a different con, same trick

The eye's version of the sensitivity curve is the **contrast sensitivity function**: how faint a sinusoidal grating can be and still be seen, as a function of its spatial frequency in cycles per degree of visual angle. It is band-pass. Sensitivity peaks near 4 to 8 cycles per degree, falls off gently for coarse gradients, and collapses for fine texture: on the standard fit below, sensitivity at 30 cycles per degree is about a fifth of the peak, and by 60 it is a few hundredths of it.

{{< fig src="img/perceptual-models/csf.png" alt="A band-pass curve of relative contrast sensitivity against spatial frequency on a log axis, peaking near 8 cycles per degree and falling steeply above 16; the range an 8-by-8 DCT block covers at ordinary viewing distances is shaded." caption="The luminance contrast sensitivity function (the Mannos-Sakrison fit, normalised). An 8×8 DCT's high-frequency coefficients land on the falling right side, which is why JPEG's quantization table gives them steps six times coarser than the DC term." >}}

Lay this over the [DCT](../block-transforms/). Each of the 64 coefficients of an 8×8 block corresponds to a spatial frequency, and the high-order ones sit on the right-hand slope of the curve, where a given amount of error is invisible. JPEG's standard luminance quantization table is that curve compiled into step sizes: 16 for the DC term, rising to 99 in the bottom-right corner. Scaling the whole table by one factor is the quality slider, as the [scalar quantization](../scalar-quantization/) article showed; the *shape* of the table is this chair's contribution, and it is as encoder-chosen as a masking threshold, except that JPEG, unusually, ships the table in the file's header.

The second thing the eye does badly is colour. The retina samples brightness at full resolution and colour at a fraction of it: the opponent-colour channels (red-versus-green, blue-versus-yellow) have their own contrast sensitivity functions that peak lower and fall off at far lower spatial frequencies than the luminance curve. So after the [channel transform](../channel-and-layout/) has split the image into a luma plane Y and two chroma planes Cb and Cr, the chroma planes can simply be sampled more coarsely.

```python
import numpy as np
def to_ycc(p):     # JPEG's RGB -> YCbCr
    R, G, B = p[..., 0], p[..., 1], p[..., 2]
    return np.stack([0.299*R + 0.587*G + 0.114*B, 128 - 0.168736*R - 0.331264*G + 0.5*B,
                     128 + 0.5*R - 0.418688*G - 0.081312*B], -1)
def to_rgb(c):
    Y, Cb, Cr = c[..., 0], c[..., 1] - 128, c[..., 2] - 128
    return np.stack([Y + 1.402*Cr, Y - 0.344136*Cb - 0.714136*Cr, Y + 1.772*Cb], -1)
def subsample_420(block):             # keep 4 Y, average Cb and Cr over the 2×2
    c = to_ycc(block); c[..., 1] = c[..., 1].mean(); c[..., 2] = c[..., 2].mean(); return to_rgb(c)
skin = np.array([[[200, 180, 170], [196, 176, 168]], [[190, 172, 160], [185, 168, 158]]], float)
edge = np.array([[[220, 30, 30], [220, 30, 30]], [[30, 30, 220], [30, 30, 220]]], float)
for name, blk in [("skin", skin), ("red/blue edge", edge)]:
    err = subsample_420(blk) - blk
    print(f"{name:14s} 12 numbers -> 6;  max |error| {np.abs(err).max():6.2f};  per-pixel RMS",
          np.sqrt((err ** 2).mean(-1)).round(2).tolist())
```
```
skin           12 numbers -> 6;  max |error|   1.55;  per-pixel RMS [[0.59, 0.96], [0.95, 0.83]]
red/blue edge  12 numbers -> 6;  max |error| 112.58;  per-pixel RMS [[79.53, 79.53], [79.53, 79.53]]
```

A 2×2 block of skin tones is twelve numbers. Keep the four Y values, replace the four Cb values by their average and the four Cr values by theirs, and it is six numbers, with a worst-case error of 1.55 on a 0 to 255 scale, under one just-noticeable step. Half the data is gone before any transform or quantizer has run, which is why the game's card calls it the cheapest 2:1 in all of image coding, and why 4:2:0 is the default of JPEG and of essentially every consumer video format. The second aha is that the retina did the compression first: a codec that stores colour at full resolution is storing detail the eye threw away on the way in.

{{< fig src="img/perceptual-models/chroma-420.png" alt="Three 8-by-4 pixel grids. 4:4:4 has a colour sample at every pixel; 4:2:2 has one per horizontal pair; 4:2:0 has one per 2-by-2 block. The counts per 2-by-2 are 12, 8 and 6 numbers." caption="Chroma subsampling. The notation counts samples in a 4-wide reference row: 4:2:0 keeps all luma and one Cb and one Cr per 2×2 block, 6 numbers where 4:4:4 needs 12." >}}

The same snippet also shows where the con fails. A hard edge between saturated red and saturated blue, two red pixels above two blue ones, averages to a purple chroma for the whole block, and the per-pixel error is 80 on every pixel. The error is large *and* visible, because an edge is exactly where luminance and chrominance stop being independent: that pattern is red text on a blue background, a subtitle, a line in a chart. This is why screen content and pixel art ask for 4:4:4, and why the card's weakness line reads the way it does. A perceptual model is a model of typical content; the content that breaks it is the content the model did not expect.

{{< lab id="chroma" >}}

## Measuring "nobody notices"

Every curve in this article was measured on people, and the unit of measurement is worth a paragraph because it is the fidelity contract's actual definition. A **just-noticeable difference** is the smallest change in a stimulus that a listener or viewer detects reliably, conventionally at some fixed rate like 70 or 75 percent of trials. It is found with a **staircase**: present a difference, make it smaller after each correct detection (or, in the common 2-down-1-up variant, after two in a row), larger after each miss, and the stimulus level oscillates around the threshold; averaging the reversal points estimates it.

{{< fig src="img/perceptual-models/jnd-staircase.png" alt="A step plot of stimulus level against trial for a simulated listener: the level starts high, falls as differences are detected, then oscillates between about 2 and 4 with hits and misses marked and reversal points circled; a dashed line marks the listener's hidden detection level and a dotted line the estimate from the reversals." caption="A 2-down-1-up staircase on a simulated listener (labelled as such: there is no human behind this plot). The procedure converges on the level detected about 70.7% of the time; the estimate from the later reversals, 3.19, sits near the hidden value of 3.53." >}}

Weber's law, that the JND is roughly a constant *fraction* of the stimulus ($\Delta I / I \approx k$), is why both the ear and the eye are logarithmic, why decibels exist, and why a companded quantizer with relative rather than absolute steps matches them. "Perceptually lossless" means every quantization error in the file is below the relevant JND for the intended viewing or listening conditions. It is a real engineering statement with a measurement behind it, and it is also conditional: the conditions include the screen size, the viewing distance and the loudness, which is why a transparent 128 kbit/s file can stop being transparent on better headphones.

The same conditionality explains why the field has never settled on a number. **PSNR**, $10 \log_{10}(255^2 / \mathrm{MSE})$, is squared error in decibels and inherits every blindness above; a slightly blurred image often scores higher than a slightly noisy one and looks worse. **SSIM** compares local means, variances and covariances of small windows and tracks structure rather than pixels, which is closer to what the eye reports. Learned metrics (LPIPS for images, VMAF for video, trained on human ratings) do better again. None of them is the consumer; all of them are models of the consumer, and a codec tuned hard against any one of them will eventually learn to fool the metric rather than the person.

## A perceptual model nobody derived

Which brings the story to its end, and to a consumer that was never interviewed. Neural codecs replace the hand-built transform with a learned encoder and the hand-built decoder with a generator network, as the [learned transforms](../learned-transforms/) article describes. Train that pair on squared error and the output is blurry and muffled, and the reason is a theorem rather than a bug: the reconstruction that minimises expected squared error given the bitstream $z$ is the conditional mean,

$$
\hat{x}^\star = \arg\min_{\hat{x}} \mathbb{E}\big[\|x - \hat{x}\|^2 \,\big|\, z\big] = \mathbb{E}[x \mid z],
$$

and at 3 kbit/s the set of waveforms consistent with $z$ is large. The average of many plausible sharp outputs is one smooth output that resembles none of them. The ear and eye both object: a blurred face and a muffled voice are *obviously* wrong even though every pixel and sample is, on average, as close as it can be.

The adversarial fix is to stop asking for closeness and ask for plausibility. A second network, the **discriminator**, is trained to tell the decoder's outputs from real data; the decoder is trained to fool it. At equilibrium the decoder must produce outputs the discriminator cannot distinguish from real speech or real photographs, which forces it to commit to one sharp, plausible reconstruction rather than the mean of many. **Feature losses** make the same move with less drama: compare the activations of a pretrained network on $x$ and $\hat{x}$ instead of the raw samples, so that error is measured in a space that already discards what the network, like the eye, does not care about. SoundStream and EnCodec train against multi-scale discriminators, on the waveform and on the STFT; HiFiC does it for images and produces photographs at a fraction of a bit per pixel that look like photographs.

Here is the third aha. The discriminator *is a perceptual model*: a learned definition of what passes as real, which is what a masking curve or a contrast sensitivity function was all along, except that nobody sat in a booth to measure it. It never runs at inference, exactly like the psychoacoustic model; it shapes the decoder's weights during training and then goes away. And it changes the fidelity contract. MP3's contract was "errors you cannot hear"; the adversarially trained codec's contract is "an output you would believe," and that is a different promise. Its failure mode is not a hiss or a smear but a confident substitution: a slightly different phoneme, a texture that was not there, detail hallucinated from the decoder's prior rather than the signal. The card's warning about medical and forensic data is this exactly; a con artist who is very good is a different kind of risk from one who is clumsy.

{{< lab id="gan" >}}

There is one more consumer on the map, and it does not have ears. When the reader of the data is a transformer attending over a KV cache, "what it will not notice" becomes "what the softmax weights at nearly zero," and importance replaces masking as the model of irrelevance. That is the [next article](../attention-as-audience/), and it is this one with the booth replaced by an attention map.

## Connections

The **Weber-Fechner law** (perceived intensity grows with the logarithm of physical intensity) is the oldest result in psychophysics and the reason every curve here is drawn in decibels and log frequency. μ-law telephony, MP3's power-law quantizer and the JND staircase all descend from it.

The **JPEG quality slider** is one scalar multiplying one table whose shape is the contrast sensitivity function; the slider is rate control, the table is this chair.

The **loudness war** is masking's evil twin in mastering: dynamic-range compression pushes everything toward the loudest level because louder sounds "better" on first hearing, and in doing so it masks the quiet detail that a codec would otherwise have had to pay for. Heavily mastered music compresses more easily; it also has less in it.

**Perceptual hashing** (the pHash family) keeps only the lowest-frequency DCT coefficients of a downscaled greyscale image and thresholds them: it survives recompression, resizing and small edits because it keeps exactly the part of the image the contrast sensitivity function says a viewer sees, and throws away the rest. A fingerprint built from a perceptual model.

**In the game:** the HYPNOTIST (psychoacoustic) sits in the Con Artist's chair for WALKMAN (MP3), where the MASKED QUANT combo pairs it with the Bender; the ILLUSIONIST (chroma) for THE GALLERY (JPEG) and PRIME TIME (H.264/AV1), and the COLOR CON combo pairs it with the Prism; the FORGER (gan) for THE PHONE CALL (SoundStream / EnCodec) and THE SHOWROOM's learned codecs, where the PLAUSIBLE FORGERY combo needs a Brainjar to paint on and FORGER WITHOUT A CANVAS is the anti-combo when there is none.
