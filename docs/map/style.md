# Writing Guide

**When to read this:** You are writing articles, polishing an existing one, or making substantive content changes. This is the authoritative source for voice, structure, and quality standards. It supersedes any briefer writing notes elsewhere.

## The Ultimate Goal

Every article exists to give the reader a **deeper, intuitive understanding of first-principles concepts** — the mathematical and logical essence of what is being explained. Not a survey of facts. Not a cheat sheet. A shift in how the reader *thinks* about the topic.

Specifically, the reader should finish each article able to:
- Grok deep, highly technical results and experiments and see the big-picture meaning behind the data points.
- Hold a grounded, concrete understanding — grounded in plots and stripped-to-the-essence Python, not vague hand-waving chains of fancy terms.
- Notice surprising connections between algorithms, math methods, and phenomena that initially look unrelated.
- See the topic not as a collection of recipes and facts, but as part of one massive interconnected web of human thought.

Keep this goal in mind at all times. It overrides every stylistic suggestion below.

---

## Audience

**Target level:** TowardsDataScience blog readers.

**Assume:** Foundations in linear algebra, calculus, statistics, probability, discrete math. Mechanical understanding of ML basics and LLM architecture. Familiarity with Python, NumPy, and PyTorch.

**Rule:** Leverage their existing knowledge. Use appropriate terminology without over-explaining basics.

---

## Narrative Techniques (Use Judgment — Not a Checklist)

These are tools, not rules. Think about the ultimate goal and choose the tools that serve it for this specific topic.

### The Detective Mystery Arc

Frame the article as a detective story, not a list of lemmas and theorems:

1. Open with a **mystery** — a technical problem or question, framed through simplified historical context. Make the reader feel the challenge. Give it human stakes. 
2. 
3. Walk through **naive approaches and why they fail**. This is not wasted space — it's where the reader internalizes why the problem is hard.
4. Let the narrative **drift through related themes and subplots**, circling back to the framing story as understanding accumulates.
5. Deliver the **dramatic culmination**: the solution arrives as a natural endpoint of its segment, not a stranger. By the time the algorithm/concept lands, its elements should feel like breadcrumbs the reader already picked up.

Example: Stuart Lloyd at Bell Labs, puzzling over how to quantize analog phone sound into a discrete set of values. Show what's hard about it. Walk through why obvious approaches fail. Let the reader follow Lloyd to the insight. By the time Lloyd-Max appears, it isn't surprising — it's inevitable.

### An answer lands only if you ask yourself the question 
Think of the most important conceptual aha moments at the heart of the material you teach. Those mathematical observations from which comes everything else.

It could be a key math observation or an argument that powers an algorithm or a mathematical technique you are explaning. It could be some previously unnoticed characteristics of the problem we have at hard, key observations or arguments. It could be something else.

Making the reader grasp and fully appreciate these conceptual aha moments is your goal in your narrative. Yes, after you introduce the conceptual aha moments, you will need to go over the rest: showing how the rest of the material you teach flows from these key moments, going over how they connect to other things the reader knows about. But this stuff isn't what's difficult here. 

The hardest part is to make these moments land well with the reader. To make the reader see why this matters. If you do that, the reader will be able to connect the dots. The reader will get the intuitive big picture view of the whole material. 

How do you make these moments land? Think of each key moments as an answer to a question. A difficult, a challenging technical question that puzzled many people over history across the planet. And what makes these moments matter is that they are the answer to this question. 

Whether the material that you teach lands with your reader or not depends on whether you explained made the reader fully grasp and feel the underlying question first. Both why this matters and what makes this question is difficult to answer right.   
The key about explaining to the reader the material is to first make the reader  see and appreciate exactly what this technical question is. Use the best of your story-telling, have this narrative where you go over your examples, your explanations and your other narrative tools we describe here so that the reader sees that first.      

### Frame it as a story

Don't throw the key math insights at the reader as if they fell from the sky. Frame the material as a timeline of how humans — engineers, scientists, programmers — slowly shaped their understanding: early hunches, naive attempts, dead-ends, and eventual insight. 
   
The reader is on a journey through time, feeling the human stakes, appreciating what the material is really about.

Here we do this because good story-telling makes the abstract material we teach  land better. Yes, the story would frame this as a moment in history, but we neither claim nor seek historical accuracy. And our stories, and our heroes do not have to feature specific historical people, we could invent some historical moments. 

An example of what that means: let's say, we are teaching the engineering of cavity magnetrons. What can we use to show why they matter? "Invention of radars was game-changing for WW2 warfare." That's what a textbook would say.  We would  convey the same fact with a step-into-my-shoes story: it's Britain, 1940, the middle of the cloudy night on the white cliff shores... Our hero is a RAF soldier, we invent his name, we give broad strokes tp paint the scene, we have few words about the fear and  emotions, and how he  tries not to make it visible...  Engineering, yes, technical arguments yes. But also human stakes, and insights on the meaning and the impact behind those.   
### Hero's Journey

Technical questions are often best framed through the struggles and breakthroughs of a specific person. Include:
- A little background on the person, the place, the year.
- The relevant zeitgeist — what were the related fields grappling with?
- The human consequences of the problem: was this used in WW2 weaponry? Then say it helped win WW2. We are learning math, not testifying in court.

### The "Aha" Moment

Every article should have at least one genuine epiphany — a moment where the reader sees a core technical issue in a new light or notices a non-obvious connection. Requirements for a good epiphany:

- **Non-obvious to a technical audience.** Avoid eat-pray-love trivialities.
- **Mathematically deep.** Should result from careful empirical analysis, a surprising application of mathematical logic, or a new perspective that reframes earlier material.
- **Core to the topic.** Not a tangent.

### Theory of Mind While Writing

Be conscious of what the reader knows at each point in the article. Do not suddenly introduce terminology you have not yet used. Do not assume the reader (or the historical character facing the mystery) already knows the answer you are about to reveal. Maintain suspense.

---

## Concrete Anchoring

### Illustrative Python Examples for Intuitive Understanding First

Anchor the material that you introduce to Python toy examples.

Example of how: when explaining Vector Quantization, don't jump into the general case right away. Let's think up a smallest example that would be an illustrative and representative case here. And we lead with this toy example first: e.g. we make up a specific 3 point-array of 2D float vectors and show the actual values as a listing. We event invent a story around our example to make it even less abstract: maybe these are the coordinates of the houses of 3 little piggies on a map? 

And now as we go through our  mathematical arguments, we show them on our toy examples first. For our 3 piggy house coordinates,  we do "let's plot them" next, and notice how these points cluster together. Because the piggies are brothers they prefer to live next to each other etc. Notice that we give some story reasons for representative properties of our example data that matter to what we want to introduce.  We have story reasons for why we want to find a smaller approximate representation of those coordinates, and as by now it feels natural in our toy example case,  we naturally arrive at the idea of trying the average of those coordinates, and introduce the definition of a centroid.   

The point is that with these concrete specific example values, the reader can follow and reproduce how the mathematical operations apply to them in their head.  Then we go through trying the   invent what these values are and show manipulations on *those* values first. Make sure the reader can actually follow what happens.

That doesn't mean we don't do proper general case definitions, LaTex equations and serious math arguments. These  have a place, and are important. In particular we want to be very deliberate about explicitly introducing terminology, notations with clear proper definitions, not "toy" example definitions, because this would let us avoid  ambiguity and potential misunderstanding. Doing this matters and we still need to do that, it's just that these must come after we introduce them with a specific concrete case with the help of our toy examples. Because this way the reader gets an intuitive understanding of things first. 

As we go in our narration, we would introduce more examples, driven by our plan, and, when possible, it is better to have the next example be a buildup or a variant on the earlier examples before. For our VQ case, we can move on to the next example, where a family of chickens moves in and builds their houses somewhere else on the island, and now we have 2 clusters of points and so on.   

Now the role of these toy examples is not to baby-talk. The examples may be "toy", the mathematical arguments and explanations are everything but.  The point of this is clarity of language.  Mathematical arguments introduced with the examples  have less risk of being too ambiguous to be too misunderstood.    


### Napkin Math

For key quantities — parameters, bytes, FLOPs, tokens — include Fermi estimates and order-of-magnitude reasoning. Make quantitative arguments concrete.

### Metaphors and Analogies

For long stretches of abstraction, introduce a grounding metaphor early and return to it. Example: vector quantization → shoe sizes. Feet grow continuously (analog), society forces them into discrete boxes (sizes 9, 9.5, 10). Lloyd's job was finding where to place those sizes to minimize total discomfort (error) for the whole population.

---

## Ruthless Simplification

Get to the beating heart of the material as fast as possible. This means:

- **Sacrifice edge cases and the general case** — put them in an appendix ("You might have seen this as…") if needed.
- **Delay or skip scope entirely** if it doesn't serve the core insight.

---

## Connections and Cross-Links

An important part of learning new things is learning how these new things are connected to what the reader already knows. This could be truly revealing. If the new material are new "dots" in reader's mental picture, connecting it to the reader's prior knowledge is like connecting these dots, and see the big picture. The reader can internalize the material at much deeper level.  

It could be revealing to recognize that this new thing you just learned about has been playing a role in what the reader already knew. seen and known examples of what you are teaching, just didn't realize them yet.

So, in your article, try to find good examples of  such connections and introduce and explain those to the reader. 

What are some of those connections? Could be literally anything. It could be examples of how these things play a role or have some explanatory power in nature, in engineering, in history. For algorithms, it's often examples of how these algorithms power some digital products, or how some businesses use them to do things. Or it can often be seeing how these things connect to other areas of mathematics and science(that the reader actually might have familiarity with, see more about target audience background above), and maybe play out under different names and terms.

So mentioning and explaining such connections as you go in your narration could be great learning value.  

Or if you have enough material for  this, you can also even have  a dedicated article section about this that we can call "Connections". Of course that section would come at the end, only after we  introduce the core of our material, so that we can use right language in there as we explain those connections.
