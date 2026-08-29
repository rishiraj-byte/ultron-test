# Brain-Inspired AI — Full Project Document

> Single handoff document. Supersedes all previous recaps. Carries every design
> decision, every correction, every flagged assumption, and every open question.

**Status as of this document: BUILT AND EXECUTED.** All seven build-order steps
are implemented and verified by running them. See §9 and §14.

---

## 0. What changed since the previous revision

The previous revision ended with *"NOTHING IN THIS SCAFFOLD HAS BEEN EXECUTED"*.
That is no longer true, and several things it flagged as open are now closed by
evidence rather than by argument. The delta, so nothing is silently rewritten:

| Section | Was | Now |
|---|---|---|
| §5 | Text and audio encoders were stubs raising `NotImplementedError` | Both are **real**. All three backbones load and produce exactly the dims §5 predicted (512 / 384 / 384) |
| §5 | "*UNVERIFIED ESTIMATE:* 30–60 forward passes/sec" | **Measured: 39.1/sec** for the full path (§14) |
| §9 | Scaffold written, never run | Every file executes; `verify.py` passes on every modality |
| §10 | 7 steps planned | All 7 implemented |
| §11 | Label set undecided | **Closed:** ImageNet-1k, inherited from the frozen encoder |
| §11 | "The lobes cannot train without a task and a labelled set" | **Closed:** distillation from the encoder's own head. No hand-labelling |
| §11 | Basal ganglia "named but unspecified" | **Closed:** direct/indirect pathway gate, `modules/basal_ganglia.py` |
| §11 | GPU telemetry not wired | **Closed:** `torch.cuda` + `nvidia-smi`, `modules/telemetry.py` |
| §4 | Frontal fusion fix was reasoned, not tested | **Validated by execution** (§14) |

Everything else below is the original design, preserved.

---

## 1. Goal

A modular AI system mapping four cortical lobes to four real, separate PyTorch
`nn.Module` networks, routed by a central thalamus, capable of navigating both
physical and digital environments.

Runs as a native desktop app. No browser.

**First use case, deliberately small:** see an image → extract features → "think"
→ give an identification.

---

## 2. Architecture

### 2.1 Signal path

```
  Image        Audio         Text
    |            |             |
    +------------+-------------+
                 |
            [ THALAMUS ]        <- modality-keyed relay, NOT a fixed pipeline
             /         \
   image ->  /           \  <- audio, text
            /             \
    [ OCCIPITAL ]      [ TEMPORAL ]
      features           meaning, language
        |    \               |
        |     \              |
   [ PARIETAL ] \            |
     space       \           |
        \         \          |
         +---------+---------+
                   |
              [ FRONTAL ]          <- convergence, fusion, decision
                   |
           [ BASAL GANGLIA ]       <- gate: allow or suppress
                   |
         [ OUTPUT GENERATOR ]      <- classifier head initially
```

Note occipital delivers to frontal **directly as well as** via parietal. That
is why frontal receives two streams on the image path — see §4, the shape
conflict.

### 2.2 Component responsibilities

| Component | Role | Status |
|---|---|---|
| Thalamus | Routes/gates input by modality tag | **Built, verified** |
| Occipital | Feature extraction (vision) | **Built, verified** |
| Parietal | Spatial relations; fed by occipital | **Built, verified** |
| Temporal | Meaning, language, memory match | **Built, verified** |
| Frontal | Convergence, fusion, decision | **Built, verified** |
| Output generator | Produces the final answer | **Built** (classifier + honest verbalisation) |
| Basal ganglia | Gates frontal's decision | **Built, verified** |
| Subcortical/limbic | Hippocampus, hypothalamus, amygdala, cerebellum, personality | Not built; modulators only |

### 2.3 Correction: the thalamus is a switchboard, not a funnel

**Original plan:** thalamus → occipital → temporal/parietal → frontal, for
everything.

**Problem:** in the real brain the thalamus does not send everything to occipital
first. Each modality has its own relay nucleus:

- LGN → occipital (vision)
- MGN → temporal auditory cortex (sound)
- VPL/VPM → parietal (touch/body)

**Corrected design:** the thalamus is a **modality-keyed switchboard**.

```
image  -> occipital
audio  -> temporal
text   -> temporal
```

Parietal is fed by **occipital**, never by the thalamus directly. This mirrors
the real dorsal "where" stream (V1 → parietal). Frontal remains the single
convergence point.

*As built:* the routing table is `config.THALAMIC_ROUTES`, and `Thalamus`
**raises at construction** if any route names parietal as a destination. The
rule is enforced, not just documented.

Each nucleus is a gated relay (`relay(x) * sigmoid(gate(x))`) rather than a
plain pass-through, because §2.2 says the thalamus *routes and gates*. Real
relay cells gain-control what passes; the mean gate opening is reported to the
dashboard per pass.

### 2.4 Deliberate deviations from biology

Flagged, not accidental:

- **Text has no thalamic relay in a real brain.** Reading enters through vision.
  Routing text straight to temporal is an engineering simplification. *As built:
  it gets a pulvinar-analogue nucleus, with the deviation flagged in the config
  beside the route.*
- **Language in humans spans temporal *and* frontal.** Ours is temporal-only.
- **Output does not route back through the thalamus.** An earlier design had
  frontal → thalamus → output. Real motor output leaves via motor cortex and
  brainstem, not a thalamic return trip. Current path is frontal → basal
  ganglia → output generator.
- **Temporal ↔ parietal.** Real brains have several connections (arcuate
  fasciculus, superior longitudinal fasciculus). Ours has none in v1; both
  lobes simply converge at frontal.

---

## 3. `LATENT_DIM = 256` — the decision that gates everything

Every lobe takes 256 numbers in and produces 256 numbers out.

**Why it matters:** each lobe is a separate network with a fixed input size. If
the encoder outputs 512 and the occipital lobe expects 256, the code crashes on
the first forward pass. Shape mismatches are the single most common failure in
this kind of build.

**Rule:** `LATENT_DIM = 256` lives in `config.py` and nowhere else. A bare `256`
in any other file is a bug.

**Consequence:** fixing the latent dimension *is* the encoder specification.
Everything downstream is plumbing.

*As built:* every derived shape is computed from it — `LOBE_HIDDEN_DIM`,
`FRONTAL_FUSION_IN_DIM`, and `LATENT_GRID_SIDE` (the 16×16 heatmap), which
raises if `LATENT_DIM` stops being a perfect square. `verify.py` asserts that
**every** inter-module tensor is `1 × LATENT_DIM`.

---

## 4. Shape conflict found in the previous recap — resolved

**The conflict.** §3 says every lobe is 256 in, 256 out. §2.1 has frontal
receiving from parietal *and* occipital simultaneously on the image path. Two
256-wide streams arriving at a 256-wide input is a mismatch. It would have
crashed at build step 2, which is exactly the step meant to catch it.

**Resolution as implemented in `modules/lobes.py`:** `FrontalLobe` fuses first,
then behaves like every other lobe.

```
n streams x 256  ->  [ fusion Linear ]  ->  256  ->  [ standard lobe MLP ]  ->  256
```

It accepts a variable number of streams, because the thalamus routes a different
count per modality — image sends two, text sends one. Unused slots are
zero-padded up to `max_streams`.

**THIS IS A DESIGN CHOICE, NOT SOMETHING THE ORIGINAL SPEC DICTATED.** The
alternative is summing the streams: zero extra parameters, but it discards which
stream contributed what. Change `FRONTAL_FUSION_MODE` in `config.py` if you
prefer that tradeoff — both modes are implemented.

**One thing the original spec left implicit, now made explicit.** Zero-padding
is only safe if slot order is *fixed*. Streams arrive as a dict keyed by
`config.FRONTAL_STREAM_SLOTS = ("occipital", "parietal", "temporal")`, and a
missing stream is padded into **its own** position. Otherwise the fusion layer
would see "parietal" in slot 1 for an image and "temporal" in slot 1 for text,
and the same weights would mean two different things. `verify.py` tests this
directly: reordering the dict changes nothing, swapping which tensor goes in
which slot changes the result.

**Verdict after execution: the fix HOLDS.** See §14.

---

## 5. Pretrained models — position changed

**Originally:** no pretrained models at all, "naked" networks only.

**Now:** hybrid.

```
[ frozen pretrained encoder ] -> [ projection layer ] -> [ your lobes ] -> [ output ]
       not trained                trainable, ->256        trained by you
```

**Reason for the change:** a randomly-initialised network produces heatmaps that
are pure noise. There is nothing to look at until it is trained. Pretrained
weights give structured, readable activations on day one.

**What is kept:** the lobes are still yours, still trained by you. Only the
encoder is borrowed.

| Modality | Model | Raw dim | Status |
|---|---|---|---|
| Vision | ResNet-18 (torchvision) | 512 | **Real, verified** |
| Text | all-MiniLM-L6-v2 | 384 | **Real, verified** |
| Audio | whisper-tiny (encoder half only) | 384 | **Real, verified** |

ResNet-18 over MobileNetV3: convolutional with wide, clean feature maps, so the
layer taps produce genuine spatial heatmaps rather than noise.

**The stubs are gone.** The previous revision specified text and audio as stubs
that raise, on the assumption the backbones could not be obtained. All three
were fetched and each produces *exactly* the dimension this table predicted, so
the stubs were replaced with real implementations. Each encoder asserts its
backbone's width against `config` at load and raises if they ever disagree.

Notes on how they are wired:

- MiniLM is called through plain `transformers`, not `sentence-transformers`, so
  the pooling is visible in our code: mask-weighted mean over tokens, then L2
  normalise.
- Only whisper's **encoder** is constructed. This system wants an acoustic
  embedding, not a transcript; the decoder is never built.
- Backbone loading is **lazy**. Constructing an encoder is free; weights arrive
  on first use, so the image path never pays for whisper.

*Measured, replacing the old unverified estimate:* **39.1 full passes/sec** on a
20-core CPU — see §14.

---

## 6. Real-time activation heatmaps

### 6.1 Mechanism — forward hooks

PyTorch hooks are callbacks that fire whenever data passes through a layer,
handing you the intermediate tensor. The model itself is never modified.

```python
activations = {}
def hook(name):
    def fn(module, inp, out):
        activations[name] = out.detach()
    return fn
model.layer2.register_forward_hook(hook("layer2"))
```

After one forward pass, `activations` holds every tapped layer's output.

*As built:* `ActivationRecorder` taps every relay nucleus, every lobe, the
striatum, and the four conv stages inside the frozen ResNet-18 — 11 layers on a
multimodal pass. A hook only fires if its module actually ran, so a silent lobe
correctly produces **no** heatmap. `verify.py` asserts that the set of captured
lobes equals the set that ran.

### 6.2 Layer diagrams -- what replaced the square heatmap

**The square reshape was wrong and has been removed for the lobes.**

Reshaping a 256-vector into a 16x16 grid puts neuron 0 next to neuron 16 for no
reason but the grid width. It shows *that* something changed without showing
*what the network is*. It was a picture of a vector, not a picture of a network.

Each network is now drawn the way a neural network is actually drawn: columns
of neurons, left to right, input layer -> hidden layers -> output layer, with
edges fanning between them and every neuron at its real index. The layer chain
is read off the live `nn.Module` tree by `modules/topology.py`, so it cannot
drift from what actually ran, and it follows `LATENT_DIM` automatically.

The real structure, which the heatmap was hiding:

| Network | Layer chain (units) |
|---|---|
| relay nucleus | 256 -> relay 256 -> gate 256 (sigmoid) -> 256 |
| occipital / parietal / temporal | 256 -> norm 256 -> **512** -> GELU **512** -> 256 -> 256 (+residual) |
| frontal | **768** streams -> fused 256 -> norm 256 -> norm 256 -> 512 -> GELU 512 -> 256 -> 256 |
| basal ganglia | 256 -> striatum 256 -> GELU 256 -> GO 1 / NoGo 1 |
| output head | 256 -> norm 256 -> **1000** logits |

**"Active" means different things in different layers, and the drawing must not
blur them together.**

* **GELU and sigmoid layers** -- a unit can genuinely be OFF. GELU drives
  negative pre-activations to ~0; a sigmoid gate saturates shut. These are
  scaled from zero, so a dark cell means *not firing*, and the count under the
  column (`167/512`) is a real count of firing units.
* **linear, norm and input layers** -- signed and centred on zero. "Off" is not
  a meaningful word for them, so they are scaled by *magnitude* and report
  `peak |x|` instead of an invented count.

**How it is drawn.** Circles in vertical columns, every circle in a column
connected to every circle in the next -- the standard textbook rendering. Fill
is the live activation; the ring stays visible at zero so an idle unit still
reads as a unit rather than vanishing.

**How many circles.** A 512-unit layer cannot be 512 circles at any readable
size -- sub-pixel circles are just a solid bar again. As many circles as fit
are drawn, each standing for a contiguous block, and the panel states the ratio
("11 circles, 47 units each") rather than implying one circle is one neuron.
Layers small enough to draw exactly (GO, NoGo) get one circle per unit.

**Blocks carry the MEAN, not the max.** A max over a 47-unit block lights the
circle whenever any single unit fires; at a typical 28% firing rate that lights
nearly every circle and the picture carries no information. The mean says how
active the block is, which is what visibly varies pass to pass. Binned values
are then rescaled by the largest block -- not min-max, which would drag a
genuinely dead block up off zero and invent activity that is not there.

**Connections carry the source node's activation**, over a floor bright enough
that the topology stays readable when a layer is quiet. The wiring exists
whether or not signal is flowing through it, and the fan visibly thins where a
layer goes silent.

What this makes visible that the heatmap could not: frontal's 768-wide input
column shows the zero-padded slot as **hollow circles** whenever a modality is
missing. The section 4 fusion fix is now something you can *see*.

Cost, measured: 3.4 ms for a full-width eight-column diagram, 0.23 ms for a node
thumbnail. A worst-case UI tick (one big diagram plus eight thumbnails) is
5.3 ms against a 100 ms refresh budget -- 95% headroom.

Convolutional taps inside the frozen encoder keep their spatial heatmaps -- a
conv feature map genuinely has a layout, so averaging channels into an image is
meaningful there. That is the one place the old approach was right.

### 6.3 Turning conv tensors into heatmaps

| Layer type | Tensor shape | How to render |
|---|---|---|
| Convolutional | `(channels, H, W)` | Average across channels → 2D grid → colour. The classic "model is looking here" image. |
| Fully-connected | flat list | Reshape to a square (256 → 16×16), colour each cell. This is what the lobes produce. |
| Transformer | tokens × dim | Use attention weights instead of raw activations. |

Brightness is normalised **per layer** — brightest value in each layer near
white, dimmest near black, regardless of the underlying numeric scale. Standard
heatmap practice, and what makes contrast visible even on small-magnitude
weights.

*Consequence worth stating, because the picture actively hides it:* per-layer
normalisation means **two equally bright heatmaps are not equally active**. Raw
min/max/mean/L2 are captured alongside every grid and shown in the expanded
view's detail pane, so scale is recoverable.

### 6.4 Threading rule

**Do not render heatmaps inside the inference loop.** Push activations to a
thread-safe queue; let the UI drain it on its own timer. Otherwise the
dashboard's redraw speed becomes the model's speed.

Queue depth is deliberately small (`ACTIVATION_QUEUE_MAXSIZE = 32`). A deep
queue lets the display drift behind reality.

*As built:* `ActivationBus` **drops the oldest frame** when full and never
blocks the producer — a stalled UI must not become a stalled model. `verify.py`
tests this by overflowing a 4-deep queue with 20 frames and asserting the
4 newest survive and 16 were dropped.

---

## 7. Dashboard spec

Native desktop window, no browser. Simple black and white. Fully offline, real
time.

### 7.1 Minimised view — one screen, no scrolling

```
+----------------+---------------------------+----------------+
|                |                           |                |
|  INPUT (20%)   |     LOBES (centre)        |  SYSTEM (right)|
|                |                           |                |
|  [ text  ]     |   thalamus + 4 lobes,     |  live state of |
|  [ image ]     |   connecting streams      |  each network  |
|  [ video ]     |   light up on data flow   |  CPU / GPU     |
|  [ audio ]     |                           |  active / idle |
|                |   [ expand view ]         |                |
+----------------+---------------------------+----------------+
|              OUTPUT — chatbot style, text only               |
+--------------------------------------------------------------+
```

- **Left, ~20%:** four input tabs — text, image, video, audio. Clicking a tab
  opens a panel: type directly for text, file picker for image/video/audio. On
  enter, the panel closes and the data goes to the thalamus.
- **Centre:** thalamus and the four lobes, with connecting streams that light up
  on data flow. Minimised heatmaps.
- **Right:** live system panel — per-network active state, CPU and GPU load,
  whether each module is running correctly.
- **Below, small strip:** output, chatbot style. **Text only in v1.** Image,
  audio, and video output are deferred.

*Built as specified.* Streams light on data flow and **decay** back to idle over
1.2 s, so you can see the pulse rather than a permanently-lit graph. Two build
notes worth keeping:

- **Tk pack order is load-bearing.** Fixed-size panes must be packed *before*
  the expanding centre, or Tk gives the leftover space to the centre and the
  right panel and output strip silently collapse to zero width. This was a real
  bug, caught by screenshotting the window rather than trusting that it built.
- **DPI awareness is set before the first window exists.** Without it, Tk renders
  at 96 DPI and Windows bitmap-stretches the result on any scaled display —
  which blurs precisely the nearest-neighbour heatmap grids the app exists to
  show.

### 7.2 Expanded view

A button at the top of the centre panel opens a second window containing the
full architecture live: thalamus, all four lobes, basal ganglia, and every data
stream, all lighting up in real time with full-size activation heatmaps, in the
exact routing order defined in §2.1.

Clicking any single network in the minimised view opens that network's detail in
the same expanded window.

*Built as specified.* The detail pane shows the selected network's full-size
heatmap, its raw pre-normalisation statistics, its trainable parameter count,
and per-node specifics: the thalamus shows its routing decisions and relay gate
values, the basal ganglia shows allow/suppress with the gate value against the
threshold, and the output shows the lobes' top-k beside the frozen encoder's
reference reading.

### 7.3 Entry points

- `verify.py` — terminal proof that tensors flow correctly. No GUI.
- `desktop_app.py` — the visual window. `--selftest` builds it, pumps the event
  loop, runs a real inference and exits, with no interaction needed.

---

## 8. Data handling — designed, and now built

- **Two formats:** physical (future real-world sensors) and digital (digital
  environment navigation). Each needs its own conversion/embedding mechanism.
  *Physical remains deferred.*
- **Decomposition sits *before* the thalamus** — `modules/decompose.py`:
  - video → frames + audio track
  - image → pixels + detected text regions
  - text found inside an image branches to **temporal** for detection and
    identification, then to **frontal** for processing.

*Built, with two limits stated rather than faked:*

- **Text detection works; text identification does not.** MSER finds where the
  glyph-like regions are, offline and with no model download. No OCR engine is
  installed, so the string handed to temporal is worded as a *layout
  description* — how many regions, where, what fraction of the frame — and says
  so inline, so nothing downstream can mistake it for a transcript.
- **Video yields frames but not audio.** OpenCV decodes video only. The
  decomposition reports this and prints the exact `ffmpeg` command to extract
  the track.

This is also the path that most exercises §4: one file becomes two modalities,
which the thalamus relays to two different lobes, which frontal then fuses.

---

## 9. Current file state

```
brain-ai/
├── config.py                     LATENT_DIM and every derived shape
├── verify.py                     step 1-2: proof that tensors flow
├── train.py                      step 7: distillation training loop
├── desktop_app.py                step 5: minimised + expanded views
├── requirements.txt
├── README.md                     how to run it
├── PROJECT.md                    this document
├── assets/                       dashboard screenshots
├── checkpoints/                  brain.pt lands here
└── modules/
    ├── __init__.py
    ├── encoders.py               frozen ResNet-18 / MiniLM / whisper + projections
    ├── thalamus.py               modality-keyed switchboard, gated relays
    ├── lobes.py                  four lobes; frontal fusion fix
    ├── basal_ganglia.py          direct/indirect go-no-go gate
    ├── output_generator.py       classifier head + honest verbalisation
    ├── activations.py            hooks, heatmaps, drop-oldest queue
    ├── pipeline.py               end-to-end orchestration -> BrainTrace
    ├── decompose.py              pre-thalamic input decomposition
    └── telemetry.py              CPU / GPU sampling
```

**Everything here has been executed.** `verify.py` passes on every modality;
`train.py` trains and the checkpoint round-trips bit-for-bit; `desktop_app.py`
builds, renders and runs real inference. See §14 for the numbers.

Not built, and still deferred: all subcortical modules beyond basal ganglia
(hippocampus, amygdala, hypothalamus, cerebellum, personality), and physical
sensor input.

---

## 10. Build order — status

1. **`verify.py`, image path end to end.** — **done**, passes.
2. **Confirm every shape is 256 where it should be.** — **done**; the frontal
   fusion fix in §4 is validated, not thrown out.
3. **Classifier head → first real answer out.** — **done**; ImageNet-1k labels,
   with an explicit untrained/trained distinction.
4. **Forward hooks + activation queue.** — **done**, including drop-oldest.
5. **`desktop_app.py`** — **done**, minimised then expanded.
6. **Audio and text paths.** — **done**, both real, both verified.
7. **Training loop for the lobes.** — **done**, distillation objective.

**Do not build the UI until the tensors flow correctly.** This order was
followed: `verify.py` was written and passing before a single widget existed.
It caught a real bug at step 4 immediately (an assertion that all four lobes
must produce activations — wrong, because temporal correctly stays silent on
the image path).

---

## 11. Open questions — most now closed

**Closed by this build:**

- ~~**Label set.**~~ **ImageNet-1k (1000 classes)**, inherited from the frozen
  ResNet-18. Chosen because it makes the label vocabulary and the training
  signal the same decision: the encoder already contains a 1000-way head, so it
  can act as teacher. `NUM_CLASSES` is derived in `config.py`.
- ~~**Training data and objective.**~~ **Distillation from the frozen encoder's
  own head**, `KL(student ‖ teacher)` at temperature 4. Any folder of images is
  a training set; the teacher supplies the labels, so nothing needs
  hand-annotating. The backbone runs once per image and is cached, which is what
  makes it CPU-viable. *Limit worth stating: a distilled student's ceiling is
  its teacher. This makes the lobes' answer meaningful, not better than
  ResNet-18.*
- ~~**Subcortical systems** (basal ganglia part).~~ Direct ("Go") and indirect
  ("NoGo") pathways read a shared striatal representation; the gate is
  `sigmoid(go − nogo)`, applied multiplicatively, with suppression below
  `BG_GATE_THRESHOLD`. Initialised open, so an untrained system reads as
  untrained rather than as broken.
- ~~**GPU telemetry.**~~ `nvidia-smi` first, `torch.cuda` as fallback, and an
  explicit "unavailable" when neither answers. It also detects the case where a
  CUDA card is present but torch is a CPU-only build, and says so instead of
  implying the GPU is working. `nvidia-smi` costs ~1.7 s per call on this
  machine, so it runs on its own 5 s clock while CPU/RAM sample at 1 Hz.

**Still open:**

- **Does parietal earn its place in v1?** Unchanged, and now testable: with a
  single image-identification task, occipital → frontal may be sufficient.
  `FRONTAL_STREAM_SLOTS` makes the ablation a one-line change, and the
  distillation loop gives you a metric to judge it by.
- **Output generator beyond classification.** A classifier head handles "what is
  this image." It becomes the bottleneck the moment you want language out. The
  verbalisation in `output_generator.py` is a template and is labelled as one —
  it is not generation, and it is not a design for generation.
- **The remaining subcortical systems.** Hippocampus, amygdala, hypothalamus,
  cerebellum, personality: still named but unspecified. Basal ganglia now has a
  concrete shape to imitate — a small module that reads frontal's state and
  returns a modulation — but no decision has been made for the others.
- **Physical sensor format.** Deferred entirely.
- **Multi-frame video.** `decompose.py` samples frames evenly and keeps them
  all, but only the middle frame drives a pass. There is no temporal model
  across frames, and no decision on whether that belongs in temporal lobe,
  hippocampus, or a recurrent state on frontal.

---

## 12. Glossary

- **Vector** — a list of numbers. The only thing a network actually reads.
- **Embedding** — a vector that carries meaning. Similar inputs land close
  together in the number space.
- **Latent dimension** — how long that list is. Ours is 256. "Latent" = internal,
  not human-readable.
- **Encoder** — converts raw input (pixels, waveform, characters) into an
  embedding. One per input type.
- **Projection layer** — a single small layer reshaping any encoder's output to
  your latent dimension. This is what lets off-the-shelf models plug in.
- **Frozen** — weights that do not change during training; used as a fixed
  feature extractor.
- **`nn.Module`** — PyTorch's unit of network structure: layers plus a
  `forward()` method.
- **Forward pass** — running data through once, input to output. "Thinking."
- **Head** — small final layer turning an embedding into an answer. A
  *classifier head* outputs a score per category.
- **Weights / parameters** — the adjustable numbers inside the network. Random
  until trained, which is why fresh networks output nonsense.
- **Loss** — the measured error between the network's answer and the correct one.
- **Backpropagation** — nudging every weight to reduce the loss.
- **Training loop** — show example → measure loss → backpropagate → repeat.
- **Forward hook** — a callback that fires when data passes through a layer,
  handing you the intermediate tensor without modifying the model.
- **Distillation** — training a "student" network to reproduce a "teacher"
  network's full output distribution rather than a hard label. Transfers more
  than a label does, because the teacher's ranking over *all* classes carries
  information about how the classes relate.
- **Temperature** — a divisor applied to logits before softmax. Above 1 it
  softens the distribution, exposing the teacher's ranking below the top class.

---

## 13. Running it

```bash
pip install -r requirements.txt

python verify.py                 # prove the wiring, image path
python verify.py --all --bench   # every modality, plus timings
python desktop_app.py            # the dashboard
python train.py --synthetic 400 --epochs 15
```

Key constraints, unchanged and now enforced in code:

- `LATENT_DIM` lives in `config.py`. Never hardcode 256 anywhere else.
- Parietal is fed by occipital, never by the thalamus directly — `Thalamus`
  raises at construction if a route violates this.
- Frontal receives multiple 256-dim streams and fuses them. It is not a plain
  256-input lobe.

---

## 14. Verification results

Windows 11, Python 3.13, torch 2.13.0+cpu, 20 cores, no GPU in use.

**`python verify.py --all --bench` — all checks pass.**

Every inter-module tensor is `1 × 256` on every path:

| Path | Streams into frontal | Zero-padded slots |
|---|---|---|
| image | occipital, parietal | temporal |
| text | temporal | occipital, parietal |
| audio | temporal | occipital, parietal |
| image + text | occipital, parietal, temporal | none |

**§4 verdict: the frontal fusion fix HOLDS.** Frontal fuses n × 256 → 256 before
behaving like a normal lobe, so occipital's direct delivery and parietal's
arrive together without a shape conflict. Slot identity was tested explicitly:
reordering the input dict changes nothing; swapping which tensor occupies which
slot changes the output.

**Performance**

| | |
|---|---|
| Full image pass (encoder + lobes + head) | 25.6 ms — **39.1 passes/sec** |
| Frozen backbone alone | 61 images/sec |
| Cold start, all backbones | ~2.7 s |

§5's unverified estimate was 30–60 forward passes/sec. The measured value sits
comfortably inside that range, for the *whole* path rather than the encoder
alone.

That figure is 1.5x better than the first measurement (30.4/sec), and the gain
came from a bug the layer diagrams exposed: `perceive()` was running the frozen
ResNet-18 **twice** per image pass — once for the lobes, once for the reference
readout — at 23 ms each. The backbone features are now computed once and shared.
Instrumenting the system for the diagrams is what made that visible.

**Trainable parameters — 2,301,674** (the borrowed backbones are excluded):

| Module | Parameters |
|---|---|
| thalamus (3 gated relay nuclei) | 396,288 |
| occipital | 263,424 |
| parietal | 263,424 |
| temporal | 263,424 |
| frontal (incl. 768→256 fusion) | 460,800 |
| basal ganglia | 66,818 |
| output head | 257,512 |
| encoder projections (×3) | 329,984 |

**Training** — 120 synthetic images, 40 epochs, ~1.7 s:

| | epoch 1 | epoch 40 |
|---|---|---|
| distillation loss | 2.4685 | 0.3519 |
| student/teacher top-1 agreement | 1.1% | 48.9% |

Gradients reach every trainable module. Synthetic shapes are a smoke test, not
a dataset — the teacher's logits on procedural images are confident nonsense.
Use real photographs for a model that means anything.

**Also verified:** checkpoint save/load reproduces logits bit-for-bit and
refuses a checkpoint whose `LATENT_DIM` or fusion mode disagrees with config;
wrong shapes and unroutable modalities raise rather than failing silently; the
activation queue drops oldest under overflow instead of blocking.
