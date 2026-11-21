# Brain-Inspired AI

An AI built the way a brain is wired instead of as one big network. Each brain
region is its own real network, sitting where the anatomy puts it, and every
connection between regions is a named, cuttable tract with its own delay. So
you can watch a picture or a sound travel through it, see which region did
what, and cut any connection to measure what it was worth.

## The original idea

Map the brain's lobes onto separate PyTorch networks and route between them
the way the brain does:

- a **thalamus** as the relay every sense passes through, not a fixed pipeline;
- **occipital** for sight, **temporal** for sound and meaning, **parietal**
  for space, fed by occipital and never by the thalamus (the "where" stream);
- **frontal** to fuse what the others send and decide;
- a **basal ganglia** to allow or suppress that decision, sending its output
  back to the thalamus, never straight to the answer.

It runs as a native desktop app (no browser) that shows every region live.
The first use case was deliberately small: see an image, extract features,
"think", give an identification. `PROJECT.md` is the original design document
and the authority on *why* anything is shaped the way it is.

## Four builds

| | v1 | v2 | v3 | **v2.B** |
|---|---|---|---|---|
| Folder | top level | top level | `v3/` | `v2b/` |
| Run it | `python desktop_app.py` | `python desktop_app_v2.py` | `python -m v3.dashboard` | `python -m v2b.dashboard` |
| Signals | float vectors | float vectors | binary spikes | float vectors |
| Learning | backprop | backprop | local plasticity only, no gradient anywhere | frozen open-weight networks + a 3.3M trained circuit |
| Senses | image, text, audio, grid navigation | text, image, audio, video, sensor | image, as Poisson spike trains | image, sound, video |
| Status | superseded | trained, working | runs, **does not learn** | **trained, deployed (round 9)** |
| Checks | 82 | 115 | 27 | 42 |

- **v1** is the original idea as first built: four lobes, thalamus, basal
  ganglia, frozen ResNet / MiniLM / Whisper encoders. Everything below this
  section documents v1.
- **v2** adds corticothalamic feedback, the thalamic reticular nucleus and
  un-pooled encoder taps. Image 59.3% (chance 1%), text 66.0% (chance 5%).
- **v3** asks whether the same wiring can learn with spikes and local rules
  alone, with no backprop. It runs, but it does not learn: a negative result,
  reported as one.
- **v2.B** places large frozen networks by anatomy: SigLIP 2 as occipital and
  the ventral stream, Depth Anything V2 as parietal, LAION CLAP as temporal
  with an arcuate path to frontal, and SigLIP 2's text side as frontal with
  1,629 names. Between them sits a small trained circuit (thalamic gates,
  dorsal stream, STS, POT, a 3-pass frontal loop), a **hippocampus** that does
  retrieval-augmented recognition over 339,364 stored memories with a
  familiarity gate, and an **amygdala** that decides GO (answer) or HOLD ("not
  sure -- maybe X"). There is no language model. Round 9 test results:

  | Test | v2.B |
  |---|---|
  | CIFAR-100 | 91.4% |
  | Tiny-ImageNet | 88.4% |
  | ImageNet photos (782 classes) | 86.4% |
  | ImageNet-R (art, cartoons, toys) | 94.7% |
  | ImageNet-Sketch | 75.5% |
  | ESC-50 sounds | 89.5% |
  | AudioSet eval | 35.9% |
  | FSD50K eval | 65.9% |
  | Never-trained classes | 73.3% (the untrained networks: 79.5%) |

  Full detail, every finding including the unflattering ones, and the data:
  [`v2b/README.md`](v2b/README.md). Round-by-round history: `STATE.md`.
  `test_pics/` holds 24 freely licensed pictures to try it on (credits in
  `test_pics/CREDITS.md`).

Each build lives on its own terms: v3 and v2.B import nothing from the others
and change nothing outside their own folders. The trained weights and the
encoded-feature caches are not in git; each build's training command rebuilds
them.

---

# v1

## Quick start

```bash
pip install -r requirements.txt
```

See what it does — a scripted walkthrough, every number computed live:

```bash
python demo.py
```

Prove the wiring:

```bash
python verify.py
```

Every modality, plus a benchmark:

```bash
python verify.py --all --bench
```

The dashboard:

```bash
python desktop_app.py
```

Retrain the lobes (all four pathways, ~4 min on CPU):

```bash
python train.py --epochs 30
```

That needs the cached features. Build them once:

```bash
python -c "from benchmark import data; [data.get(t) for t in ('cifar100','news20','gridnav','audiotone')]"
```

---

## What runs where

| File | Does |
|---|---|
| `config.py` | `LATENT_DIM` and every shape derived from it. Change shapes **here only**. |
| `demo.py` | Scripted walkthrough. Run this in front of people. |
| `verify.py` | Terminal proof that tensors flow. No GUI. |
| `explain.py` | 18 checks that each stage genuinely works, not just runs. |
| `modules/language.py` | Language production. Speaks what the lobes perceived. |
| `desktop_app.py` | The dashboard: minimised view + expanded view. |
| `train.py` | Multi-task pretraining: vision, language, navigation, acoustic. |
| `benchmark/` | GridNav task, baselines, ablation harness, LaTeX tables. |
| `paper/` | IEEE draft, auto-generated result tables. |
| `modules/encoders.py` | Frozen ResNet-18 / MiniLM / whisper-tiny + trainable projections. |
| `modules/thalamus.py` | Modality-keyed switchboard. Not a funnel. |
| `modules/lobes.py` | Occipital, parietal, temporal, frontal. |
| `modules/basal_ganglia.py` | Go / no-go gate on frontal's decision. |
| `modules/output_generator.py` | Classifier head + honest verbalisation. |
| `modules/activations.py` | Forward hooks, per-neuron capture, the hand-off queue. |
| `modules/topology.py` | Reads each network's real layer chain off the modules. |
| `modules/diagram.py` | Draws a network as input/hidden/output neuron columns. |
| `modules/pipeline.py` | End-to-end orchestration; produces a `BrainTrace`. |
| `modules/decompose.py` | Splits video/image inputs *before* the thalamus. |
| `modules/telemetry.py` | CPU / GPU sampling for the system panel. |

---

## The signal path

```
   image        audio         text
     |            |             |
     +------------+-------------+
                  |
             [ THALAMUS ]         modality-keyed relay
              /         \
     image ->/           \<- audio, text
            /             \
    [ OCCIPITAL ]     [ TEMPORAL ]
        |     \             |
        |      \            |
   [ PARIETAL ] \           |
         \       \          |
          +-------+---------+
                  |
             [ FRONTAL ]          fuses n x 256 -> 256, then decides
                  |
         [ BASAL GANGLIA ]        allow or suppress
                  |
        [ OUTPUT GENERATOR ]
```

Three rules that are easy to break by accident:

1. **`LATENT_DIM` lives in `config.py` and nowhere else.** A bare `256` in any
   other file is a bug.
2. **Parietal is fed by occipital, never by the thalamus.** The thalamus
   raises if you try; that is the dorsal "where" stream.
3. **Frontal is not a plain 256-input lobe.** It fuses a variable number of
   256-wide streams first. An image sends two, text sends one, and missing
   slots are zero-padded *positionally*.

---

## Measured on this machine

Windows 11, Python 3.13, RTX 3050 6GB (torch 2.13.0+cu126).

| | |
|---|---|
| Full perception pass (encoders + lobes + head) | **8.6 ms — 116.7 passes/sec** |
| Chat reply, full sentence | **1.25 s** |
| VRAM, everything loaded | 2.16 / 6.4 GB |
| Frozen backbone alone (training precompute) | 61 images/sec |
| Trainable parameters (yours, not borrowed) | 2,301,674 |
| Cold start, all backbones loaded | ~2.7 s |

`PROJECT.md` §5 carried an unverified estimate of 30–60 forward passes/sec.
The measured figure sits inside that range, for the *whole* path rather than
the encoder alone.

Running on the GPU. The `cu126` channel was chosen because it is the only one
carrying torch **2.13.0** — the exact version already installed — so the swap
changed the backend without moving the framework underneath everything else.
The language model runs in fp16 on CUDA (fp32 on CPU, where half precision is
slower rather than faster).

---

## Seeing it think

Every network is drawn as a real neural network — circles in vertical columns,
input layer to hidden layers to output layer, every circle wired to every circle
in the next column. Fill is the live activation; the ring stays visible at zero
so an idle unit still reads as a unit. The layer chain is read off the
`nn.Module` tree at runtime, so the picture cannot drift from what actually ran.

| Network | Layer chain |
|---|---|
| relay nucleus | 256 → relay 256 → gate 256 → 256 |
| each lobe | 256 → norm 256 → **512** → GELU **512** → 256 → 256 |
| frontal | **768** → fused 256 → norm → norm → 512 → GELU 512 → 256 → 256 |
| basal ganglia | 256 → striatum 256 → GELU 256 → GO 1 / NoGo 1 |
| output head | 256 → norm 256 → **1000** |

Under each column is a live firing count (`167/512`) — but **only** for GELU and
sigmoid layers, where a unit can genuinely be OFF. Linear and norm layers are
signed and centred on zero, so "off" is meaningless for them and they report
`peak |x|` instead of an invented number.

Click any node to open that network full-width in the expanded view. A lobe that
did not run this pass draws as empty outlines, not as stale data — on the image
path, temporal is correctly blank.

A 512-unit layer cannot be 512 readable circles, so each circle stands for a
contiguous block and the panel states the ratio ("11 circles, 47 units each").
Blocks carry the block *mean* — a max would light almost every circle at a
typical 28% firing rate and tell you nothing.

The clearest thing this shows: frontal's 768-wide input column has **hollow
circles** at the bottom whenever a modality is missing. That is the §4
zero-padding, visible.

---

## What the answer means before you train

On a fresh checkout the lobes are randomly initialised, so **their answer is
noise**. The system says this out loud rather than dressing it up:

```
lobes (UNTRAINED, this answer is noise): Arabian camel 0.6%
frozen encoder reference: lampshade 16.7%
run  python train.py  to make the lobes' answer mean something
```

The reference line is what the *borrowed* ResNet-18 head believes. It is shown
for contrast and is never presented as the lobes' own output.

`train.py` closes that gap by distilling the frozen encoder's 1000-way head
into the whole lobe stack. Any folder of images works as a training set — the
teacher supplies the labels, so nothing needs hand-annotating. On 120
synthetic images over 40 epochs, student/teacher top-1 agreement goes
1.1% → 48.9%, which is enough to show the plumbing carries information; real
photographs are what make it mean something.

---

## After an architecture change, retrain BOTH builds

v1 and v2 keep separate checkpoints and share `modules/lobes.py`, so a change
to a lobe invalidates both. Neither will load stale weights silently:

* v1 refuses outright -- `architecture_signature()` hashes the tensor names and
  shapes, and a mismatch raises rather than loading.
* v2 loads what still matches, prints which parameters the checkpoint predates,
  and reports itself UNTRAINED until it has been retrained.

```
python train.py        # v1, minutes
python train_v2.py     # v2, hours on a laptop GPU
```

## Honest limits

- **Text output only.** Image, audio and video *output* are deferred (§7.1).
- **No OCR.** `decompose.py` detects *where* text is in an image (MSER, offline,
  no model) but not *what it says* — no OCR engine is installed. The string it
  hands to temporal is worded as a layout description so nothing downstream can
  mistake it for a transcript.
- **Audio and video decoding depend on what is installed.** The v2 app reports
  what it found at startup on the input pane (`AUD:` and `VID:` lines) rather
  than failing on the file you picked. With `soundfile` and `imageio-ffmpeg`
  present it reads essentially any audio format and pulls the soundtrack out of
  a video; with only `scipy` it is limited to uncompressed WAV, which is what
  the line says. Install both with
  `pip install soundfile imageio-ffmpeg opencv-python`.
- **A clip is eight frames, not a film.** Video is sampled to
  `config.VIDEO_FRAMES` evenly spaced frames. The frames are read in order, so
  motion is visible to the recurrent lobe, but nothing here models duration or
  tracks an object between frames.
- **A distilled student cannot beat its teacher.** Training makes the lobes'
  answer meaningful, not better than ResNet-18.

---

## Deliberate deviations from biology

Flagged, not accidental — see `PROJECT.md` §2.4.

- Text has no thalamic relay in a real brain; reading enters through vision.
  Here it gets a pulvinar-analogue nucleus straight to temporal.
- Language in humans spans temporal *and* frontal. Ours is temporal-only.
- Real basal-ganglia output returns through the thalamus before reaching motor
  cortex. Here the gate applies directly on the way out.
- Temporal and parietal have no direct connection; they converge at frontal.
