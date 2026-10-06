# brain-ai

A brain-inspired AI in four builds. **Read `STATE.md` before doing anything** — it
carries the verified numbers, which checkpoint to trust, and the traps that have
already cost hours. This file is the short version that loads automatically.

## What is here

| | v1 | v2 | v3 |
|---|---|---|---|
| Run it | `python desktop_app.py` | `python desktop_app_v2.py` | `python -m v3.dashboard` |
| Signals | float vectors | float vectors | binary spikes |
| Learning | backprop | backprop | local plasticity only |
| Status | superseded | **trained, working** | runs, does not learn |
| Checks | 82 | 115 | 27 |

**v2 is the deliverable.** Image 59.3% (chance 1%), text 66.0% (chance 5%). It takes
text, image, audio, video and sensor; all 9 input combinations are verified.

**v3 is a clean fork under `v3/`** that imports nothing from v1 or v2 and modifies
nothing outside its own directory. Keep it that way.

**v2.B is a fourth build under `v2b/`**, self-contained on the same terms. Frozen
open-weight networks placed by anatomy (SigLIP 2 So400m vision split occipital /
ventral stream, Depth Anything V2 Small as parietal, LAION CLAP as temporal with
an arcuate path to frontal, SigLIP 2 text as frontal with 1,629 names), a 3.30M
trained circuit, a hippocampus with a RAG database and a familiarity gate, an
amygdala that gates motor (GO / HOLD + confidence), no language model. Image,
audio and video only -- no text input. Trained on 178,792 pictures (CIFAR-100,
Tiny-ImageNet, 782 ImageNet classes as photos) and 58,209 sounds (ESC-50,
AudioSet, FSD50K), with label smoothing + feature dropout. Test: CIFAR-100 91.2%,
Tiny-ImageNet 87.9%, ImageNet photos 84.4%, ESC-50 87.5% (was 95.2% before the
broad sound data), AudioSet 35.2%, FSD50K 65.8%. Train with `--candidate` so a
run never overwrites the deployed model. `python -m v2b.train --watch` shows training live; `--refit`
redoes everything after the cortex. Checks: `python -m v2b.verify_v2b` (42).
See `v2b/README.md`.

## Before you change anything

Run the checks. They take a few minutes and they are the whole point of the project:

```bash
python status.py            # is v2 trained; what can this machine decode
python verify_v2.py         # 115 checks
python -m v3.verify_v3      # 27 checks
```

## Standing rules

- `LATENT_DIM` lives in `config.py` only. Never hardcode it. Same for every other
  constant — if a number matters, it belongs in a config file.
- Every white-matter connection is a named `Tract` object with its own conduction
  delay. Never an anonymous tensor add. Lesion experiments depend on this.
- Basal ganglia output goes to the **thalamus**, never to the output.
- Encoders stay un-pooled: ResNet keeps its 7x7 grid, MiniLM stays token-level,
  Whisper stays frame-level.
- All pretrained backbones frozen. Note that `requires_grad = False` does **not**
  freeze BatchNorm buffers — that bug cost 46 accuracy points and was invisible in
  the loss curve.
- v3 contains no gradient anywhere. No loss, no optimiser, no `.backward()`. A gate
  scans the source for those tokens.
- Report gradient coverage **and** ablation delta per module. Coverage alone is not
  enough: frontal cortex receives gradient on every batch, displaces cortical state
  more than any other module when cut (142.1%), and contributes **nothing** to
  accuracy. See `STATE.md`.

## Working on this machine

- **`ps aux` in Git Bash cannot see Windows processes.** It reports nothing for a
  running `python.exe`. Use PowerShell:
  `Get-CimInstance Win32_Process -Filter "Name like 'python%'"`
- **Do not run GPU work while training.** 6 GB card, and the user's own dashboard
  takes ~3.7 GB of it.
- Training runs take hours. Launch them in the background and do not poll.

## Style

Comments explain **why**, and where a decision was forced by a measurement, they
quote the measurement. That convention is the project's actual documentation —
follow it rather than describing what the code already says.

Report findings straight, including the unflattering ones. Two of this project's
three real results are negative.
