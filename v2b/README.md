# v2.B

A brain-inspired perception system with no language model. It looks at a picture, listens to a sound, or watches a video, and says what it is, or says "not sure". Pretrained, open-weight networks sit where the anatomy puts them. Between them is a small circuit v2.B trained itself, a hippocampus that does retrieval-augmented recognition, and an amygdala that decides whether to answer. It lives entirely in `v2b/`, imports nothing from v1, v2 or v3, and changes nothing outside this folder.

```
image / sound / video (-> 8 frames + soundtrack)
  THALAMUS  [ pulvinar . LGN . MGN ]  tiny gates; frontal can turn each up or down (TRN)
  OCCIPITAL      SigLIP 2 So400m vision, blocks 1-13    V1 -> V4
  VENTRAL STREAM SigLIP 2 So400m vision, blocks 14-27   V4 -> IT: WHAT (temporal lobe)
  PARIETAL       Depth Anything V2 Small                depth, surface slope
  DORSAL STREAM  V4 grid + parietal, per place          WHERE                       trained
  TEMPORAL       LAION CLAP (auditory cortex)           hears; its sound LEXICON
                   names the sound -> ARCUATE FASCICULUS -> frontal
  STS            sight (ventral) meets sound (temporal)                              trained
  POT            fuses dorsal + STS  <-- 3 passes -->  FRONTAL (SigLIP 2 text,       trained
                                                        1,629 names)
  HIPPOCAMPUS    entorhinal keys -> RAG DATABASE (237,001) -> CA1 vote,              trained
                 gated by FAMILIARITY; + episodes you taught it
  AMYGDALA       hears cortex AND the hippocampus -> GO / HOLD + confidence          trained
  MOTOR          GO: the answer.  HOLD: "not sure -- maybe X".
```

| Network | Where | Weights | Licence |
|---|---|---|---|
| `google/siglip2-so400m-patch14-224`, vision | occipital (blocks 1-13) + ventral stream (14-27) | 428M | Apache-2.0 |
| `google/siglip2-so400m-patch14-224`, text | frontal (names every picture and sound) | 708M | Apache-2.0 |
| `depth-anything/Depth-Anything-V2-Small-hf` | parietal | 25M | Apache-2.0 |
| `laion/larger_clap_general`, audio | temporal (auditory cortex) | 69M | Apache-2.0 |
| `laion/larger_clap_general`, text | temporal's sound lexicon (built once) | not counted | Apache-2.0 |

That's 1.23 billion frozen weights. v2.B trains 3.30 million of its own:
- cortex: 3.00M
- memory readout: 0.30M
- amygdala: 289

**Vocabulary: 1,629 names** that frontal can answer with:
- CIFAR-100's 100 classes plus ImageNet's 1,000 (including "web site" and "monitor"): 1,075 picture names;
- ESC-50's 50 sounds plus AudioSet's 527: 554 sound names.

v2.B trains on about 1,560 of these names: ~1,070 picture names (CIFAR-100, Tiny-ImageNet, and 782 ImageNet classes as real photos) and ~490 sound names (ESC-50, AudioSet, FSD50K). It recognises the rest zero-shot.

## Commands

```
python -m v2b.dashboard                  # the console
python -m v2b.verify_v2b                 # 42 checks
python -m v2b.diagnose                   # what it gets wrong on every test, and why
python -m v2b.train --watch              # encode (cached; hours the first time) + train (~2.5 h), in a live window
python -m v2b.train --refit              # keep the trained cortex; redo memory, expertise, amygdala (~20 min)
python -m v2b.monitor                    # the live window on its own; it follows any run in progress
python -m v2b.train --cut "temporal->frontal"   # retrain WITHOUT a part, to measure it
python -m v2b.lessons 1..7               # learn it by poking it -- see LEARN.md
```

It needs `pyarrow` for Tiny-ImageNet, ImageNet, AudioSet, FSD50K and Imagenette (`pip install pyarrow`). The ImageNet photos are fetched once with `python -c "from v2b import data; data.fetch_imagenet()"`.

## Watching it train

`--watch` opens a window that follows the run as it happens. The trainer writes each event to `results/v2b_live.jsonl` and the window reads that file, so it can be closed and reopened (`python -m v2b.monitor`) without touching the run.

It shows:
- the stage, time elapsed and time left;
- progress bars while the frozen networks encode the data;
- the loss, every 5 steps;
- validation accuracy after every epoch: pictures, sounds, and never-trained classes;
- 8 pictures from the batch it is learning on, with its guess (green if right, red if wrong);
- speed, learning rate and GPU memory.

## The dashboard

One window with seven tabs. **FOCUS** (or a click on the graph) picks the region the other tabs look at. **Double-click** a region on the graph to open it up: its ACTIVITY (blocks, attention, which neurons fired), or MEMORY for the hippocampus and amygdala.

| Tab | What it shows |
|---|---|
| ROUTING | The graph, each arrow lit by what it carried on each of the 3 passes. It includes the THALAMUS box around its relays, the Ventral and Dorsal stream boxes, the **arcuate** line from temporal to frontal, the RAG DATABASE under the hippocampus, its familiarity, and the amygdala's GO/HOLD verdict to motor |
| NETWORKS | every region as columns of circles |
| ACTIVITY | per-block activation maps, attention maps, and which neurons fired in every block |
| WEIGHTS | every weight tensor, as a heatmap, histogram and statistics. **x live input** shows which connections carried signal; **zero it** / **restore** edit weights live |
| MEMORY | cortex vs memory vs final answer; CA1's familiarity; the amygdala's verdict; the RAG database; your episodes; the pictures recalled |
| RAG | the hippocampus as a network, every state and gate live: POT vector → entorhinal key → the store's 32 most active units → CA1's vote → the familiarity and trust gates → frontal's sum, name by name (cortex + memory + yours = final) → the amygdala's 7 inputs, 32 hidden units and value → the GO/HOLD gate → motor. Below it, **the whole store as one layer**: all 237,001 memories, one cell each, lit by how strongly this percept activates them, with the 32 recalled ringed. Hover any cell to see which memory it is, its picture, its rank and its share of the vote. Double-click the hippocampus, amygdala or database on the graph to open it |
| OVERLAY | where POT looked, and the depth map |
| GUIDE | for each region: what it does in a brain, what v2.B uses, where the two differ, and its live numbers |

## Measured (test sets no stage ever saw)

| | untrained | cortex | memory alone | **final** |
|---|---|---|---|---|
| images, CIFAR-100 (10,000; chance 1%) | 37.4% | 91.2% | 90.9% | **91.2%** |
| images, Tiny-ImageNet (10,000; 200 classes) | 56.8% | 87.8% | 87.8% | **87.9%** |
| images, **ImageNet photos** (7,820; 782 classes) | 69.7% | 84.3% | 84.0% | **84.4%** |
| sounds, ESC-50 fold 5 (400; 50 classes) | 47.2% | 87.0% | 83.5% | **87.5%** |
| sounds, **AudioSet eval** (4,000; 429 names) | 20.1% | 35.2% | 32.3% | **35.2%** |
| sounds, **FSD50K eval** (4,000; 167 names) | 39.2% | 65.5% | 60.8% | **65.8%** |
| never-trained classes, Imagenette (2,000; 10 classes) -- a development set now | 79.5% | 77.4% | | **77.4%** |

Every row is scored against all 1,629 names, by the exact name.
- "cortex" is the cortex as it answered, with expertise (below). At the cautious strength alone (0.25), CIFAR's cortex scores 89.0%.
- The untrained CIFAR figure is low because an exact CIFAR name must beat ImageNet near-synonyms ("bear" vs "brown bear").
- AudioSet's eval is hard by nature: 429 names, each clip given ONE of its several labels, and many labels are broad ("music", "vehicle").
- Imagenette has guided design choices (memory's familiarity gate, expertise), and its classes now have trained neighbours, so it is no longer a clean test.
- Per pass of the frontal loop, CIFAR (cautious strength): 88.3%, 88.9%, 89.0%.

**How it got here** (each column measured on the same tests):

| | 2026-09-29: CIFAR + Tiny | + ImageNet photos, 12 epochs, expertise | + 45k sounds | + 8 photos/class, 12k rare sounds, 16 epochs | **+ label smoothing, feature dropout (now)** |
|---|---|---|---|---|---|
| pooled validation | -- | 77.2% | 81.6% | 81.8% | **82.3%** |
| CIFAR-100 | 90.7% | 90.2% | 90.8% | 90.5% | **91.2%** |
| Tiny-ImageNet | 88.3% | 87.3% | 87.5% | 87.8% | **87.9%** |
| ImageNet photos, 782 classes | not trained on (untrained 69.7%) | 83.5% | 83.5% | 84.1% | **84.4%** |
| ESC-50 sounds | 96.2% | 95.2% | 88.2% | 87.0% | **87.5%** |
| AudioSet eval | -- | 14.7% | 34.8% | 33.4% | **35.2%** |
| FSD50K eval | -- | 36.9% | 63.4% | 65.6% | **65.8%** |
| Never-trained (dev) | 74.9% | 81.9% | 81.1% | 79.0% | 77.4% |

**The amygdala.**

| | images | sounds (ESC-50) |
|---|---|---|
| lets through (GO) | 97%, of which 93.1% are right | 83%, of which 93.7% are right |
| holds back (HOLD) | 3%, of which only 37.8% would have been right | 17%, of which 58.0% would have been right |

With no thalamic drive (for example, the ears' gate forced shut), it says HOLD instead of naming something.

**Memory (RAG).** The database holds every training example: 237,001 (178,792 pictures, 58,209 sounds). The fitted trust is **0.075**: cortex does most of the work; memory adds 0.1 on Tiny-ImageNet and ImageNet photos and 0.5 on ESC-50 sounds. Its jobs now are the familiarity signal that decides expertise, what you teach it (episodes), and the amygdala's inputs. See finding 7.

## Findings, including the unflattering ones

1. **Training on 150 names wrecked the other 1,479.** At full strength, POT's learned correction pulled never-trained pictures toward trained names, and accuracy on them fell from 79.5% to 33.4% (29.5% when trained on CIFAR alone).
   - Scaling the correction showed it was the right direction but far too strong.
   - v2.B keeps **0.40** of it. That was chosen on the *validation* sets alone: the weakest strength within 3 points of full strength on CIFAR and ESC-50 validation.
   - On validation pictures, 0.40 scores 87.5% against 88.6% at full strength.
   - Result: never-trained classes 79.3% from cortex, against 79.5% zero-shot.
   - This is WiSE-FT-style blending with the zero-shot model (Wortsman et al. 2022).
2. **Memory dragged novel things toward what it knew.** It stores only the classes it trained on, so for a church it recalled castles and voted for them just as confidently as for a familiar input. In the CIFAR-only build, never-trained accuracy fell 76.5% → 67.8%.
   - Fix: CA1 now gates its vote by **familiarity**, meaning how close the nearest memory really is.
   - The gate half-opens at 0.65 similarity. That threshold comes from validation data.
   - With the gate, the fall was 3.1 points (76.5% → 73.4%) in the CIFAR-only build. With 136,200 memories it is 4.4 (79.3% → 74.9%). A bigger store costs the never-trained classes more.
   - The remaining fall was deliberately left, because tuning it away on the test set would make the test meaningless.
3. **Sound needs its own lexicon.** A linear "bridge" from CLAP's space into frontal's, fitted on text pairs, named ESC-50 test sounds 17.5% right; CLAP in its own space got 89.8%. Even inside CLAP, sounds and words don't sit in quite the same place (the "modality gap").
   - So the temporal lobe keeps its own sound vocabulary (Wernicke's area) and sends its scores to frontal over the arcuate fasciculus.
   - Cutting that path after training costs 53.3 points on sounds.
4. **The pretrained networks still do most of the work,** and the feedback loops earn little here.
   - Cutting frontal→POT after training costs 1.1 points. Cutting frontal→thalamus *raises* accuracy by 1.5.
   - When measured on the previous networks by retraining without each, they were within noise.
   - Depth earns more: cutting pulvinar→parietal or parietal→dorsal costs 6.3 points, and dorsal→POT 3.9.
5. **Trained on CIFAR alone, v2.B was worse than no training at all on 200 ImageNet classes, and no test showed it.**
   - This was measured by retraining the old recipe (25,000 CIFAR pictures) on the same networks. It reproduced the old build: CIFAR 90.5%, sounds 95.0%, never-trained classes 73.8%.
   - On Tiny-ImageNet's test it scored **33.0%**. Zero-shot scores 56.8%.
   - 83% of its wrong answers were one of CIFAR's 100 names: monarch → "butterfly", school bus → "bus", bullet train → "train", black widow → "spider". It still recognised the kind of thing, but lost ImageNet's finer name. Zero-shot, only 18% of wrong answers are CIFAR names.
   - Memory made it worse (38.6% → 33.0%). Its store held only CIFAR classes, and the familiarity gate, set on CIFAR, did not see these pictures as unfamiliar. Their nearest memory sat at 0.648 similarity, just past the gate's 0.635, so memory voted on 55% of them.
   - The 10-class never-trained test only moved 79.5% → 76.5%. Its classes (church, parachute, golf ball) have no CIFAR near-name to collapse onto.
   - **What the extra data bought** (135,000 pictures against 25,000):

     | | CIFAR-only | now |
     |---|---|---|
     | Tiny-ImageNet | 33.0% | **88.3%** |
     | CIFAR-100 | 90.5% | 90.7% |
     | sounds | 95.0% | 96.2% (5 more clips of 400) |
     | never-trained classes | 73.8% | 74.9% |

   - Almost all of the gain is in the 200 classes it now trains on. On CIFAR it bought 0.2 points.
   - Retrain-without numbers for the current networks weren't rerun. The older ones are kept in `results/v2b_train_without_*.json`.

6. **More classes cost a little on the old benchmarks.** Training on 782 more ImageNet classes (real 256 px photos, 48 per class) took Tiny-ImageNet from 88.3% to 86.9% and CIFAR from 90.7% to 90.3%, while the new classes went from untrained 69.7% to 83.5%. More trained neighbours make fine distinctions harder.
7. **The cautious correction became too cautious, so expertise was added.** The 0.40/0.30 rule protects names never trained on; with ~1,070 of 1,075 picture names trained it cost ~3 validation points on all of them (0.30 -> 85.0%, 0.60 -> 88.0%).
   - Now the cautious strength answers first; an answer the hippocampus holds memories of is re-read at a stronger strength (0.75), chosen on validation. Cut `hippocampus->POT` and it is off.
   - A strong correction for EVERY answer scored higher on validation but cost never-trained classes 2.8 points; expertise cost them nothing.
   - Side effect: memory's fitted trust fell from 0.21 to 0.03. The cortex now does nearly all the work, and RAG's contribution to accuracy is about zero.
8. **Sounds: 10 times the names, 7 points lost on ESC-50.** 45,000 clips from AudioSet and FSD50K (one name per clip, AudioSet words mapped onto ESC-50's where they are the same sound) took AudioSet eval from 14.7% to 34.8% and FSD50K eval from 36.9% to 63.4%, and pictures gained 0.5 on validation. ESC-50 fell from 95.2% to 88.2%.
   - The errors are overlapping names, not deafness: crickets -> "insects", pouring water -> "liquid", chirping birds -> "bird vocalization", pig -> "livestock", and washing machine -> "engine" (no new washing-machine clips; hundreds of engines).
   - Logit adjustment for class frequency (Menon et al. 2021) did not fix it: on validation it traded ESC-50 (+2) for the new sounds (-3).
   - Chosen on validation anyway: pooled 81.6% vs 77.2%, and with each sound set weighted equally 68.4% vs 61.5%. The previous model is kept (`checkpoints/v2b_expertise.pt`) for anyone who only needs ESC-50's 50 sounds.
9. **Leaks found and closed before they counted.** 229 FSD50K clips are cut from the same freesound recordings as ESC-50's validation/test clips (removed by ID), 29 more were near-copies by CLAP similarity (removed), and FSD50K's val/test held recordings ESC-50's training clips came from (removed). ImageNet "twins" of held-out classes (tape player for cassette player, tiger cat for tabby, ...) and ImageNet's computer "mouse" (which the vocabulary filed under the rodent) are not trained on.
10. **More data has stopped paying much.** Round 5 added the next 8 photos of each ImageNet class (48 -> 56), 12,122 rare-sound clips from AudioSet's unbalanced set, and 16 epochs. Pooled validation rose only 81.6% -> 81.8% (pictures tied, sounds slightly up), so it was deployed by the rule, but the tests go both ways: ImageNet photos +0.6, FSD50K +2.2, Tiny +0.3, against CIFAR -0.3, ESC-50 -1.2, AudioSet -1.4, never-trained -2.1. The best epoch was 5 of 16: training past it only memorised. The frozen networks set the ceiling; disk (D: has ~1.8 GB free) limits further data.
11. **Regularisation beat more data.** Round 5 kept epoch 5 of 16: past it the model memorised the cached features, which are identical every epoch. Round 6 added label smoothing (0.1) and dropped 10% of each input grid during training. It kept improving to epoch 16, and pooled validation rose 81.8% -> 82.3%, the largest gain of the last three rounds. Every test improved except the never-trained development set (79.0% -> 77.4%).
12. **A taught memory fired for the whole class, and the checks caught it.** A taught episode fired when an input was 0.80 similar to it. Label smoothing pulls a class together: two different same-class validation pictures cleared 0.80 21% of the time with round 5's cortex and 72% with round 6's (median 0.70 -> 0.88). Teaching one bear "garden gnome" renamed another bear. The bar is now measured per trained model as the 99.5th percentile of same-class, different-item similarity on validation (0.975 for this model), so only the same input or a near-copy matches.

## Data

| Data | Use | Licence / notes |
|---|---|---|
| CIFAR-100 | 45,000 train, 5,000 validation, 10,000 test | |
| Tiny-ImageNet | 90,000 train (450 per class), 10,000 validation. Its own 10,000-picture validation set is the **test** | `zh-plus/tiny-imagenet`. ImageNet terms: non-commercial research |
| ImageNet photos | 782 classes not in Tiny or Imagenette: 56 train (48 + 8 more, a split of their own) + 6 validation per class (fetched by row group), and 10 per class from ImageNet's own validation photos as the **test**. No photo appears twice (checked by bytes) | `benjamin-paine/imagenet-1k-256x256`. ImageNet terms |
| ESC-50 | folds 1-3 train, 4 validation, 5 test | CC BY-NC 3.0: non-commercial |
| AudioSet (balanced) | ~16,000 training clips; 2,000 validation and 4,000 test from its eval segments | `danjacobellis/audioset_opus_24kbps_balanced`; labels CC-BY 4.0, audio from YouTube (research use) |
| AudioSet (unbalanced, 2 of 96 files) | 12,122 rare-name training clips (names with < 300 clips; no catch-alls; no YouTube id shared with the balanced/eval lists). The files were deleted after encoding | `danjacobellis/audioset_opus_24kbps` |
| FSD50K | ~29,000 training clips (capped at 600 per name with AudioSet); 2,000 validation; 4,000 of its eval set as a **test** | `philgzl/fsd50k`, CC-BY 4.0 (per-clip CC licences). Every clip from an ESC-50 validation/test recording is removed |
| Imagenette | 2,000 pictures, never trained on; a development set now | `johnowhitaker/imagenette2-320` |

## Files

| File | What it holds |
|---|---|
| `config_v2b.py` | every constant and model name |
| `senses.py` | decoding |
| `lobes.py` | the frozen networks, live recording, the sound lexicon |
| `cortex.py` | thalamus, TRN, dorsal, STS, POT, the loop, the arcuate |
| `memory.py` | entorhinal, CA1 (vote, familiarity), store, episodes, amygdala |
| `wiring.py` | the 22 named, cuttable connections |
| `brain.py` | the assembly and `perceive()` |
| `data.py` | datasets and the 1,629-name vocabulary |
| `train.py` | the training stages |
| `monitor.py` | the live training window |
| `diagnose.py` | error breakdown per test set |
| `ragview.py` | the RAG tab: the memory path and the whole store, drawn |
| `verify_v2b.py` | the checks |
| `dashboard.py`, `drawing.py`, `guide.py` | the console |
| `lessons.py`, `LEARN.md` | the teaching material |

Outside `v2b/`:
- `checkpoints/v2b.pt`: the trained parts, 13 MB.
- `checkpoints/v2b_memory.npz`: the RAG database, 525 MB, rebuilt by `train`.
- `checkpoints/v2b_cortex_best.pt`: the trained cortex with what it was trained on, for `--refit`.
- `checkpoints/v2b_round5.pt` (+ `v2b_memory_round5.npz`, `v2b_cortex_best_round5.pt`): the previous model. Copy them over `v2b.pt` / `v2b_memory.npz` / `v2b_cortex_best.pt` to roll back.
- `python -m v2b.train --candidate` trains into `*_candidate` files and never touches the deployed model; promote by renaming.
- `checkpoints/v2b_expertise.pt` (+ its memory): the model before broad sound training (ESC-50 95.2%, but only 50 sounds).
- `data/v2b_cache/`: 58 GB, rebuildable. `data/imagenet_subset/`: 0.66 GB of fetched photos.
- `results/v2b_live.jsonl`: the last run, event by event, for the live window.
- `v2b/memory/`: what you taught it, gitignored.
