# Brain-Inspired AI

Four cortical lobes as four real, separate PyTorch networks, routed by a
central thalamus, gated by a basal ganglia. Runs as a native desktop app.
No browser.

`PROJECT.md` is the design document and the authority on *why* anything is
shaped the way it is. This file is how to run it.

---

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
