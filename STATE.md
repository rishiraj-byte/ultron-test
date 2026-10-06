# STATE.md — read this first

Handoff for a fresh session. Everything here was measured by running it, not
recalled. Where a number could have drifted, the command to re-check it is given.

Last verified: **2026-09-24**

---

## What this repository contains

Three systems, each a complete build. **v2 is the one that works.**

| | v1 | v2 | v3 |
|---|---|---|---|
| Entry point | `desktop_app.py` | `desktop_app_v2.py` | `python -m v3.dashboard` |
| Signals | float vectors | float vectors | binary spikes |
| Time | single pass | 12 timesteps | 300 ms at 1 ms |
| Learning | backprop | backprop | local plasticity only |
| Status | superseded | **trained, working** | runs, does not learn |
| Checks | 82 | 115 | 27 |

v1 and v2 share `modules/`. **v3 is a clean fork in `v3/` and touches nothing else.**

---

## v2.B -- a fourth build (first 2026-09-24, rebuilt 2026-09-27)

Self-contained in `v2b/`, like v3. Frozen open-weight networks placed by anatomy:
SigLIP 2 So400m vision split into OCCIPITAL (blocks 1-13) and the VENTRAL STREAM
(14-27, ending in IT); Depth Anything V2 Small as PARIETAL; LAION CLAP as TEMPORAL
(auditory cortex), whose sound lexicon reaches frontal over the ARCUATE
FASCICULUS; SigLIP 2 text as FRONTAL, holding 1,629 names (CIFAR-100 + ImageNet,
ESC-50 + AudioSet). A 3.30M trained circuit: thalamus (+TRN), dorsal stream, STS,
POT in a 3-pass loop with frontal; a hippocampus with a 237,001-entry RAG database,
a CA1 vote gated by familiarity, and directly matched taught episodes; an
amygdala that hears cortex and hippocampus and gates motor (GO / HOLD +
confidence). No language model. Write-up: `v2b/README.md`; teaching: `v2b/LEARN.md`.

```
v2.B test accuracy        images 91.2%  (CIFAR-100, 10,000)
                          images 87.9%  (Tiny-ImageNet, 10,000, 200 classes)
                          images 84.4%  (ImageNet photos, 7,820, 782 classes; untrained 69.7%)
                          sounds 87.5%  (ESC-50 fold 5, 400; was 95.2% before AudioSet/FSD50K)
                          sounds 35.2%  (AudioSet eval, 4,000, 429 names; untrained 20.1%)
                          sounds 65.8%  (FSD50K eval, 4,000, 167 names; untrained 39.2%)
                          never-trained 77.4%  (Imagenette -- a development set now)
training data             178,792 pictures (CIFAR, Tiny, 782 ImageNet classes as photos)
                          + 58,209 sounds (ESC-50, AudioSet bal.+unbal., FSD50K);
                          16 epochs with label smoothing 0.1 + 10% feature dropout
                          (round 6: pooled validation 81.8% -> 82.3%; more data alone
                          had stopped paying). Best epoch 16.
expertise                 cautious keep 0.25 answers; a stored (familiar) answer is re-read
                          at 0.60, chosen on validation. Memory trust 0.075.
taught episodes           fire at >= 0.975 similarity (measured per model: q99.5 of
                          same-class, different-item similarity); 0.80 renamed a whole class
trade-off, chosen on val  ESC-50 -7 points for +20 / +27 on the broad sound tests;
                          previous model kept at checkpoints/v2b_expertise.pt
CIFAR-only on Tiny-ImageNet  33.0%, BELOW zero-shot 56.8% (see v2b/README.md finding 5)
amygdala                  answers 97% of images at 93.1%; holds back 3% that are 37.8% right
v2.B checks               42 pass, 0 fail
```

```bash
python -m v2b.dashboard         # 7 tabs incl. WEIGHTS (every tensor) and ACTIVITY (every block)
python -m v2b.verify_v2b        # 42 checks
python -m v2b.train --watch     # ~2.5 h train (cache 58 GB), live window
python -m v2b.train --refit     # reuse the cortex; redo memory/expertise/amygdala (~20 min)
python -m v2b.diagnose          # error breakdown per test set
python -m v2b.monitor           # the live window on its own
python -m v2b.lessons 1         # ...7
```

`checkpoints/v2b.pt` and `checkpoints/v2b_memory.npz` are what the dashboard loads;
the checkpoint refuses to load into different networks. `data/v2b_cache/` (58 GB)
is rebuildable. `data/ESC-50-master/` is CC BY-NC. `v2b/memory/` holds what the
user taught it and is gitignored.

**v2.B traps.** Hugging Face was measured at ~54 KB/s on 2026-09-27 and its
default transfer backend stalled at 0 bytes; `HF_HUB_DISABLE_XET=1` fixed it.
New downloads then failed with WinError 1314 (no symlink privilege): patch
`huggingface_hub.file_download.are_symlinks_supported` to return False before
downloading. CLAP ships only `pytorch_model.bin` -- a safetensors-only filter
silently skips its weights. fast.ai's Imagenette bucket ran at 12 KB/s; the
Hugging Face copy `johnowhitaker/imagenette2-320` works. Models load local-first.

---

## Verified numbers

```
v2 held-out accuracy      image 59.3%  (chance 1%)
                          text  66.0%  (chance 5%)
                          audio 100%   (chance 16.7%)
                          sensor 54.5% (chance 25%)
v2 training               25 epochs, 868,800 examples, 200.6 min
v2 gradient coverage      334 / 334 trainable tensors
v2 gates                  115 pass, 0 fail
v1 gates                   82 pass, 0 fail

v3 neurons                12,416 across 39 populations
v3 synapses               2,057,362
v3 gates                   27 pass, 0 fail
v3 development            cortex 6.1-6.4 Hz (target 5), GPi 60.98 Hz (target 60)
v3 held-out accuracy      5.0%  (chance 20%)  <- does not learn
```

Re-check with:

```bash
python status.py            # is v2 trained, what can this machine decode
python verify_v2.py         # 115 checks
python -m v3.verify_v3      # 27 checks
python contribution.py      # per-lobe contribution to accuracy
```

---

## Checkpoints

| File | What | Keep? |
|---|---|---|
| `checkpoints/cortex_v2.pt` | **the working v2 model** | yes — this is the product |
| `checkpoints/cortex_v2.prev.pt` | previous v2, 0.3 pts better, trained under a dopamine bug | backup |
| `checkpoints/v3_developed.pt` | v3 after development, before any task | the one the dashboard loads |
| `checkpoints/brain_v3.pt` | v3 after the failed training run | **pathological — do not use** |
| `checkpoints/brain.pt` | v1 | stale; v1 will refuse it (architecture hash) |

`brain_v3.pt` has GPi at 1.1 Hz instead of its tonic 60, and the gate's effect is
inverted. `v3/console.py` therefore takes `prefer="developed"` and the dashboard
uses it.

---

## The two findings worth publishing

**1. Displacement is not contribution.** Frontal cortex, ablated two ways:

```
state displacement when removed   142.1%   <- largest of any module in the system
accuracy cost when removed         +0.4    <- removing it HELPS
```

Cutting the arcuate fasciculus is the same story: state moves 134%, accuracy costs
0.0 points. Every module here passes a gradient-coverage check and an ablation sweep
on displacement. Frontal passes both and contributes nothing to the answer.

A likely cause is structural: the readout concatenates all four lobes, so frontal's
job of integrating them is done twice and the readout's copy wins.

**2. A biologically honest gate prevents learning.** In v3, two requirements taken
straight from the biology are in direct conflict:

- The basal ganglia gate must be authoritative — tonically active, everything vetoed
  by default. This is the entire reason v3 exists rather than v2.
- An untrained brain must be able to act, or reward learning has nothing to reinforce.

At full gate strength motor cortex reaches 19 Hz against a 40 Hz threshold: **0
decisions in 8 trials**, and 0% decided for 18 of 30 epochs. Weaken the gate enough
to permit action and it stops controlling the output (open 27.9 Hz vs shut 29.6 Hz)
— which is v2's problem returned.

The class signal is not the issue. Traced stage by stage, between-class spread over
within-class spread:

```
input rates 0.5447 -> LGN 0.3767 -> occipital 0.3208 -> parietal 0.3399
-> temporal 0.2963 -> frontal 0.2958 -> motor 0.2732
```

It reaches motor cortex. The failure is credit assignment from a single scalar
through a deep spiking network — an open research problem, not a defect.

---

## What is NOT true, despite appearances

- **v2 is not image-only.** It takes text, image, audio, video and sensor, and all
  9 input combinations are verified. Text is its *best* modality (66.0%).
- **v3 is not multimodal in practice.** All four thalamic nuclei are wired
  (LGN/MGN/pulvinar/VPL) with their radiations, but every script only ever feeds
  LGN. Text and audio are untested in v3.
- **v3 is not a chatbot and cannot be made into one at this scale.** Qwen was removed
  deliberately, both the external language module and the decoder layer inside
  frontal. Nothing replaced it.
- **The monolith comparison is stale.** The last matched-scale run predates the
  current v2 and has not been rerun. No claim is made about it.

---

## Traps that have already cost time

**`ps aux` in Git Bash does not see native Windows processes.** It returns nothing
for a running `python.exe` and a liveness check built on it always reports zero.
Acting on that produced three concurrent training runs competing for a machine with
1.25 GB free. Use PowerShell instead:

```powershell
Get-CimInstance Win32_Process -Filter "Name like 'python%'" | Select ProcessId,CommandLine
```

**Do not run GPU work while training.** A 6 GB card plus the user's own dashboard
(~3.7 GB) leaves little headroom; a concurrent check killed a v2 run at epoch 4.

**Training can make a checkpoint worse.** A run that spends most epochs unable to act
leaves homeostasis pushing on a network that never responds. Check the tonic GPi rate
before trusting a v3 checkpoint.

**Freezing parameters does not freeze BatchNorm buffers.** `requires_grad = False`
stops gradients and does nothing to running statistics. In v2 this silently
overwrote a frozen ResNet's ImageNet statistics with cortical state: image accuracy
**1.8%**, fixed to **48.1%** by forcing teachers to stay in eval mode. Nothing in the
loss curve indicated a problem.

---

## Standing constraints (from the original build spec, still in force)

- `LATENT_DIM` lives in `config.py` only. Never hardcode it.
- Every white-matter connection is a named `Tract` object with its own conduction
  delay — never an anonymous tensor add. Lesion experiments depend on this.
- Basal ganglia output goes to the thalamus, **not** to the output.
- Do not zero-init the BG direct/indirect matrices. Small random init, biases
  +1.5 / −1.5.
- Encoders stay un-pooled: ResNet layer4 keeps its 7×7 grid, MiniLM stays
  token-level, Whisper stays frame-level.
- All pretrained backbones frozen.
- Report gradient coverage **and** ablation delta per module at every gate. Gradient
  coverage alone is not enough — see finding 1.

---

## Open questions

1. **Does frontal earn its place?** Read out from frontal alone and compare. If it
   still contributes nothing, that is a publishable negative result about fusion
   modules in this class of architecture.
2. **Re-run the monolith benchmark** at matched scale so the paper's comparison is
   current. `benchmark_v2.py`, several hours.
3. **Wire text and audio into v3.** The nuclei exist; the encoders and testing do not.
4. **Train an abstain class** on out-of-domain text. v2 currently answers `1+1` with
   a topic label at 98% confidence; margin-based abstention helps but does not fix it.

---

## Commands

```bash
python desktop_app_v2.py          # v2 dashboard — routing map, click any module
python desktop_app_v2.py --selftest
python -m v3.dashboard            # v3 — spike rasters, membrane traces, the gate
python -m v3.console              # v3 without graphics

python status.py                  # trained? decoders present?
python verify_v2.py               # 115 checks
python -m v3.verify_v3            # 27 checks
python contribution.py            # does each lobe do any work
python ablate.py                  # cut each part, measure displacement
python prove.py                   # flow timing, loop closure, robustness

python train_v2.py                # retrain v2, ~3.5 h on this GPU
python -m v3.develop              # v3 finds its operating point, ~15 min
python -m v3.learn                # v3 reward learning, hours, does not converge
```

---

## File map

**v2** — `config.py` holds every constant. `cortex.py` is the 12-step settling loop.
`modules/lobes.py` the four operators, `modules/tracts.py` the 26 wires,
`modules/laminar.py` the L4/L2-3/L5-6 shell, `modules/thalamus_v2.py`,
`modules/basal_ganglia_v2.py`, `modules/limbic.py`, `modules/cerebellum.py`.
`brain_v2.py` assembles it; `perceive()` is the entry point. `train_v2.py`,
`verify_v2.py`, `desktop_app_v2.py`.

**v3** — `v3/config_v3.py` (millivolts and milliseconds), `v3/neurons.py` (the
spiking cell and the PV/SST/VIP microcircuit), `v3/synapses.py` (sparse masks, delay
lines, STDP), `v3/regions.py`, `v3/modulators.py`, `v3/brain_v3.py` (`step()` is one
millisecond), `v3/transduce.py`, `v3/develop.py`, `v3/learn.py`, `v3/verify_v3.py`,
`v3/dashboard.py`, `v3/console.py`.

**Paper** — `paper/main_v2.tex`, `paper/collect_v2.py`.

Every non-obvious decision in the code carries a comment stating the measurement that
forced it. Those comments are the real record; this file is the index.

---

## Reference documents

Published write-ups of this project, in increasing depth:

- *From Units to Spikes* — AI from first principles, then both systems specified
- *One Percept, Twice* — one input followed through v2 and then v3, step by step
- *Spikes and Vectors* — where the architecture matches real neuroanatomy and where
  it does not
- *Four Lobes, Twelve Steps* — v2 explained end to end

Ask the user for the links; they are private artifacts on claude.ai.
