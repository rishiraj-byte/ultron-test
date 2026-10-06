# Learning v2.B: weights, encoding, vectors, memory

Every idea here has a lesson you can run. It also has a dashboard tab that shows the same thing live.

```
python -m v2b.lessons 1          # ...through 7
python -m v2b.dashboard          # WEIGHTS and ACTIVITY tabs
```

---

## 1. A network is numbers in named boxes

`lessons 1` · dashboard **WEIGHTS**

A layer is a few **weight tensors**, which are grids of numbers, plus a rule: multiply the input by them, add, then bend the result (GELU). A network is a stack of layers. Every weight has a name:

```python
brain.vision.blocks[0].self_attn.q_proj.weight      # 768 x 768 numbers
```

In the dashboard, pick a region in **FOCUS**, open **WEIGHTS** and click any tensor:
- **The heatmap:** each output neuron is a row and each input is a column. Amber is positive, blue negative.
- **The histogram** of its values, and its statistics.
- **x live input:** each weight multiplied by how active its input was on your last percept. This shows which connections actually carried signal.

v2.B holds 1.23 billion frozen pretrained numbers, and 3.30 million it trained itself.

## 2. Encoding turns an input into a vector

`lessons 2` · dashboard **ACTIVITY**

Encoding means running an input through a network and keeping what comes out. That output is a **vector**, a list of numbers, and similar inputs give similar vectors.

| input | where in v2.B | what comes out |
|---|---|---|
| image | occipital: blocks 1-13 of SigLIP 2 So400m | a 7x7 grid of 1,152 numbers (V4-level features) |
| image | ventral stream, ending in IT: blocks 14-27 + head | a 7x7 grid, plus **one identity vector** |
| image | parietal: Depth Anything V2 Small | a depth map, and 388 numbers per place |
| sound | temporal (its auditory cortex): LAION CLAP | 16 slices of time, 1,024 numbers each, plus **one vector in CLAP's sound-word space** |
| phrase | frontal: SigLIP 2 text network | one vector, **in the same space as IT's identity** |
| sound name | temporal's lexicon: CLAP text network | one vector, **in the same space as the sound** |

The phrase rows are the trick. SigLIP 2 was trained so that a photo of a dog and the words "a photo of a dog" land close together, so a picture and a name can be compared directly. CLAP does the same for a bark and "the sound of a dog barking".

That's why v2.B can name about 1,600 things, most of which it never trained on. The temporal lobe's sound scores reach frontal directly, over the **arcuate fasciculus**. A single map from CLAP's space into SigLIP's was tried first. Fitted on words alone, it named test sounds only 17.5% right (CLAP alone: 89.8%), because even inside CLAP sounds and words don't sit in quite the same place.

**ACTIVITY** shows the encoding happening, block by block:
- how strongly each patch is represented;
- where attention went;
- a raster of which MLP neurons fired (amber) or were pushed below zero (blue), for every block.

## 3. Vectors can be compared, ranked and added

`lessons 3`

- **Cosine similarity:** normalise both vectors to length 1, then take a dot product. 1 means the same direction.
- **Ranking:** score one vector against many and sort. This is frontal's whole job: `scores = words @ embedding`.
- **Arithmetic:** add concept vectors, and whatever is nearest the sum is a blend of them.

## 4. Changing weights

`lessons 4` · dashboard **WEIGHTS**: "zero it" and "restore"

```python
w = brain.cortex.thalamus.MGN.gain.bias
saved = w.detach().clone()
with torch.no_grad():
    w.fill_(-20.0)            # or w[:, :384] = 0, w.mul_(0.5), ...
thought = brain.perceive(audio=sound)
with torch.no_grad():
    w.copy_(saved)
```

In the dashboard, select a tensor, press **zero it**, PERCEIVE again and watch the answer (and the amygdala's value) change, then press **restore**.

## 5. Training is loss, gradient, step

`lessons 5`

1. **Loss:** one number for how wrong the answers were (cross-entropy).
2. **Gradient:** `loss.backward()` works out which way each weight should move.
3. **Step:** `optimiser.step()` moves each weight a little that way.

**Frozen** means `requires_grad = False`, so those weights never move. To **fine-tune** part of a pretrained network, unfreeze just that part and use a small learning rate (1e-5). A 6 GB card can't hold gradients for all of it at once.

## 6. Saving, loading and swapping

`lessons 6`

`torch.save(module.state_dict(), path)` saves the weights, and `module.load_state_dict(torch.load(path))` loads them back by name.

To swap a pretrained network, change one line in `config_v2b.py`, then run `python -m v2b.train --fresh`. Widths are read from the loaded model. The checkpoint remembers which networks it was trained with and won't load into different ones.

## 7. Memory is retrieval-augmented recognition (RAG)

`lessons 7` · dashboard **RAG** (double-click the hippocampus) and **MEMORY**

The steps follow the anatomy. The RAG tab draws each one live, left to right, with its real numbers:

1. **Entorhinal cortex:** a trained projection turns POT's vector into a memory key.
2. **The store:** 237,001 examples, searched with faiss: every picture and sound the cortex trained on (CIFAR-100, Tiny-ImageNet, ImageNet photos, ESC-50, AudioSet, FSD50K).
3. **CA1:** the 32 nearest memories vote. Closer matches count more; that "sharpness" is trained. CA1 also judges **familiarity**, meaning how close the nearest memory really is. For something unlike anything stored, the vote fades.
   - It doesn't always work. In the build before ImageNet photos, a church (never trained) had Tiny-ImageNet altars as its nearest memories at similarity 0.76; the gate stayed open (0.89), memory voted 79% "altar", and the answer flipped from cortex's "church" to "altar". Memory cost never-trained classes 4.4 points then.
   - Now the stronger correction makes cortex right far more often, the fitted trust is 0.075, and memory barely moves an answer (never-trained: cortex 77.4%, final 77.4%). Watch any of this in the RAG tab.

4. **Trust:** CA1's vote is added to cortex's scores with a trained weight, fitted on the validation set.
5. **Amygdala:** hears cortex, and the hippocampus's recall directly: how sure each is and whether they agree. It predicts whether the answer is right and sends motor a verdict, GO or HOLD, with that confidence. HOLD means "not sure".

**The store is one layer of units.** Each memory's stored key works like one neuron's weights, and its activation is the cosine with this percept's key. Searching the store means asking which 32 of those 237,001 units fire hardest. The RAG tab draws the whole layer, one cell per memory.

The entorhinal projection was trained **leave-one-out**: each training example had to be recognised by its neighbours with itself removed. That's exactly the position a new picture is in.

**Teaching it** adds a memory; no weights change. In the dashboard, type the right name next to "no, it's a:". Your episodes take their own, direct path: an episode that matches (nearly) this same input votes for its label, up to 40 logits, fading with similarity. "Nearly" is measured per trained model: 0.975 similarity now, above what two different pictures of one class reach 99.5% of the time (a fixed 0.80 once renamed every bear after you taught one). They can't go through CA1, because the trust CA1 learns is small (0.075 now) and a correction would never win. It is tested to relabel the picture you taught, but not every other picture of that class.

---

## Recipes

- **Find out if a part matters.** Retrain without it: `python -m v2b.train --cut "frontal->thalamus"`. Cutting a part out of a trained system overstates its importance, because it also measures the shock of inputs the network never saw.
- **Add data.** Add a loader in `data.py`, add the names to `vocabulary()`, then run `train --fresh`. Only the small parts retrain.
