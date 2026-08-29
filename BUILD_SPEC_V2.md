# Brain-Inspired AI — v2 Build Spec

> Full handoff. Supersedes PROJECT.md. Adds the subcortical/limbic systems and
> the white-matter tracts that v1 was missing, on top of a parallel recurrent
> settling loop.

---

## 0. Read this before building

**Scope warning, stated once.** v1 has four modules and has not yet demonstrated
that its modularity pays. This spec adds roughly fifteen more. That is a real
risk: more moving parts, more places for silent failures like the zero-init
basal ganglia bug, and a harder attribution problem when something works.

Mitigation is built into the build order in §9 — every stage has a gate that
must pass before the next begins. Do not skip the gates. A module that cannot
be shown to receive gradient AND change the output when ablated is not built,
it is only present.

---

## 1. What this system is

One AI. Four cortical lobes as four mathematically distinct networks running
**in parallel** every timestep, wrapped in a recurrent settling loop, modulated
by a full subcortical stack, wired together by named white-matter tracts.
Frozen pretrained backbones do perception. Everything above them is ours.

---

## 2. The settling loop — the core of the system

Real cortex does not run a pipeline. All areas are active at once, exchanging
signals across conduction delays, settling over roughly 100–300 ms.

```python
for t in range(T):                       # T = 8
    ach, ne, da, ht = neuromod.state(t)

    thal = thalamus(sensory, cortical_fb=fro[t-1], bg=bg[t-1],
                    trn=trn[t-1], attn=attn[t-1], ach=ach)

    # ---- FOUR LOBES COMPUTE SIMULTANEOUSLY ----
    # each reads state[t-1] only, so none waits on another within a step
    occ[t] = Occipital(thal.lgn,  fb=fro[t-1], lat=par[t-1], ne=ne)
    par[t] = Parietal (thal.vpl,  ff=occ[t-1], fb=fro[t-1],  ne=ne)
    tem[t] = Temporal (thal.mgn,  ff=occ[t-1], fb=fro[t-1],  ne=ne)
    fro[t] = Frontal  (occ[t-1], par[t-1], tem[t-1],
                       rec=fro[t-1], amy=amy[t-1], hpc=hpc[t-1],
                       cbm=cbm[t-1], da=da)

    amy[t] = Amygdala(thal.raw, tem[t-1], fro[t-1])      # low + high road
    hpc[t] = Hippocampus(ec[t-1], amy[t], ht=ht)
    hyp[t] = Hypothalamus(amy[t], fro[t-1])
    bg[t]  = BasalGanglia(fro[t], da=da)
    cbm[t] = Cerebellum(fro[t], sensory_pred_error)
    trn[t] = TRN(thal, fro[t-1])
```

Reading `t-1` for every cross-module input is what makes it genuinely parallel.
That delay is conduction latency — it is a feature, not a workaround.

**This one change delivers:** parallelism, top-down feedback, working memory
(persistent frontal state), oscillatory phase, and a settling dynamic.

---

## 3. Full module inventory

### 3.1 Cortex — the four lobes

| Lobe | Operator | Sits in |
|---|---|---|
| Occipital | grouped Conv1d, `groups=8` | L2/3 |
| Parietal | single-head self-attention over **real** spatial positions | L2/3 |
| Temporal | GRU over **real** token/frame sequences | L2/3 |
| Frontal | positional-slot concat fusion + recurrence | L2/3 |

**Laminar microcircuit** — every lobe becomes three sublayers:

```
  thalamic input ──► L4      (input layer, thalamorecipient)
                      │
  lateral input ────► L2/3   (lateral processing; the lobe's operator lives here)
                      │      └──► feedforward output to next area
  feedback input ───► L5/6   (deep layers)
                             ├──► feedback output to lower areas
                             └──► corticothalamic output to thalamus
```

L6 is the source of the corticothalamic projection. Cortex→thalamus synapses
outnumber thalamus→cortex by roughly an order of magnitude.

**Frontal subdivisions** (submodules inside frontal, not separate lobes):

| Subdivision | Role |
|---|---|
| dlPFC | working memory, the recurrent state |
| vmPFC / OFC | value, receives amygdala via uncinate |
| ACC | conflict monitoring, error detection, drives effort |
| Broca | language output interface, receives arcuate from temporal |

### 3.2 Thalamus — nuclei, not one block

| Nucleus | Carries | To |
|---|---|---|
| LGN | vision | occipital L4 |
| MGN | audio | temporal L4 |
| VPL/VPM | somatosensory, sensor | parietal L4 |
| Pulvinar | attention-modulated relay | parietal, temporal |
| MD (mediodorsal) | limbic/BG return | frontal |
| VA/VL | BG + cerebellar output | frontal, motor |
| **TRN** | **inhibitory shell** | gates all other nuclei |

**TRN is the piece v1 missed entirely.** A GABAergic sheet wrapping the
thalamus; every thalamocortical and corticothalamic axon passes through it. It
receives cortical L6 feedback and inhibits specific nuclei — the physical
substrate of selective attention.

`thal_out = relay(x) * sigmoid(gate(x) + W_fb @ fro[t-1] - trn_inhibition)`

### 3.3 Limbic system

**Amygdala** — salience and valence. Two input routes, both required:

```
  LOW ROAD:  thalamus ──────────────► amygdala     fast, crude, ~12ms
  HIGH ROAD: thalamus ► temporal ───► amygdala     slow, detailed
```

Low road takes raw thalamic output at `t`; high road takes temporal at `t-1`.
Outputs to hypothalamus, vmPFC, hippocampus, and modulates cortical gain.

**Hippocampus** — episodic memory, the real circuit:

```
  association cortex ──► ENTORHINAL CORTEX ──► Dentate Gyrus
                                │                    │  pattern separation
                                │                    ▼
                                │                  CA3  ◄─┐ recurrent collaterals
                                │                    │    │ pattern completion
                                │                    ▼    │
                                └──────────────────► CA1 ─┘
                                                     │
                                                 subiculum ──► fornix ──► hypothalamus
                                                     └──────────────────► frontal
```

- **EC** is the gateway; grid-cell-like coding lives here.
- **DG** does pattern separation — k-WTA at ~2%.
- **CA3** dense recurrent collaterals; autoassociative, pattern completion.
- **CA1** compares CA3 completion against direct EC input; mismatch = novelty.
- **Replay**: between batches, sample CA3 episodes back into cortex.

**Hypothalamus** — homeostasis. Persistent scalar state (energy, temperature,
threat, satiety). Slow timescale — across episodes, not timesteps.

**Mammillary bodies + anterior thalamus** — closes the Papez circuit.

### 3.4 Basal ganglia — full circuit, loop closed

v1 had BG as a terminal output gate. **That is anatomically wrong.** BG output
goes to thalamus, which returns to cortex. It gates *what cortex does next*.

```
  cortex ──► striatum ──┬──► DIRECT   (D1, Go)   ──► GPi ──┐
                        └──► INDIRECT (D2, NoGo) ──► GPe ──► STN ──► GPi
  cortex ──────────────────► STN (HYPERDIRECT, fast global brake)
                                                            │
                                              GPi ──► thalamus VA/VL ──► frontal
  SNc ──► dopamine ──► striatum (reward prediction error, sets learning rate)
```

- **Hyperdirect** (cortex→STN) is the emergency stop. v1 missed it.
- **SNc dopamine** modulates striatal learning rate, not just gain.
- **Init warning:** do NOT zero-init direct/indirect. Opposite gradients cancel
  and the striatum receives ~1e-6 gradient forever. Small random init, biases
  +1.5/−1.5.

### 3.5 Cerebellum — forward model

Predicts the sensory consequence of frontal's decision; prediction error trains it.

```
  frontal ──► pontine nuclei ──► cerebellar cortex ──► dentate ──► thalamus VL ──► frontal
```

Climbing fibers from the inferior olive carry the error.

### 3.6 Neuromodulators — four global scalars

| System | Source | Modulates |
|---|---|---|
| Acetylcholine | nucleus basalis | feedforward vs feedback balance |
| Norepinephrine | locus coeruleus | global gain / arousal |
| Dopamine | VTA / SNc | reward prediction error, striatal learning rate |
| Serotonin | raphe nuclei | time horizon, patience |

Scalars, not vectors. Driven by hypothalamus state and amygdala salience.

### 3.7 Oscillations

| Band | Frequency | Role |
|---|---|---|
| Theta | ~4–8 Hz | hippocampal encoding phase |
| Alpha | ~8–12 Hz | inhibitory gating |
| Gamma | ~30–80 Hz | binding, local processing |

`gain = 1 + A·sin(2πft + φ)` indexed by `t`, distinct `φ` per lobe.

### 3.8 Other

- **Superior colliculus** — fast reflexive orienting, bypasses cortex. Optional.
- **Claustrum** — **SPECULATIVE.** Include only to explore; do not claim
  function in the paper.
- **Insula** — interoception; bridges hypothalamus to frontal.

---

## 4. White matter — named tracts

Every one a named object, not an anonymous tensor add. Naming makes lesion
experiments trivial: cut a tract, measure.

| Tract | Connects | Function |
|---|---|---|
| **Optic radiation** | LGN → occipital | vision in |
| **Inferior longitudinal (ILF)** | occipital ↔ temporal | ventral "what" |
| **Superior longitudinal (SLF)** | parietal ↔ frontal | dorsal "where/how" |
| **Inferior fronto-occipital (IFOF)** | occipital ↔ frontal | visual-to-decision |
| **Arcuate fasciculus** | temporal ↔ frontal (Broca) | language |
| **Uncinate fasciculus** | temporal pole ↔ orbitofrontal | emotion-to-value |
| **Cingulum bundle** | cingulate ↔ hippocampus ↔ frontal | memory + conflict |
| **Fornix** | hippocampus → mammillary/hypothalamus | memory output |
| **Internal capsule** | cortex ↔ thalamus | thalamocortical loop |
| **Corticopontine** | frontal → pons → cerebellum | forward model in |
| **Dentatothalamic** | cerebellum → thalamus VL → frontal | forward model out |
| **Stria terminalis** | amygdala → hypothalamus | threat to autonomic |
| **Medial forebrain bundle** | VTA → striatum, cortex | dopamine delivery |

Each a small `Tract` module: learned linear map plus optional delay in
timesteps. Conduction delay differs by tract length — a principled reason for
different `t-k` offsets.

---

## 5. Encoders — stop pooling

**The single highest-value change in the spec.** v1 pooled every encoder to one
vector before the lobes. Pooling destroys exactly the structure parietal and
temporal were built to exploit — which is why they showed no measurable benefit.

| Modality | Backbone | Tap | Gives |
|---|---|---|---|
| Vision | ResNet-18 | **`layer4`, 7×7×512** | 49 real spatial positions |
| Text | all-MiniLM-L6-v2 | **token-level, pre-pooling** | real sequence |
| Audio | Whisper-tiny encoder | **frame-level, pre-pooling** | real sequence |
| Sensor | none — trainable MLP | raw 8×(dist, hit) | already the feature |

All backbones frozen. Every encoder ends in `Linear(raw → LATENT_DIM) +
LayerNorm` — load-bearing, or the largest-magnitude backbone dominates fusion.

**Alternative worth evaluating: CORnet-S** (brain-matched, Brain-Score).
**UNVERIFIED: confirm the package installs before committing.**

---

## 6. Training

### 6.1 Predictive coding — the primary objective

Each lobe predicts its own input from its current state. Only the **prediction
error** propagates upward.

```
  error = actual_input − predicted_input
  loss  = Σ_lobes ‖error‖²  (precision-weighted)
```

- **No labels needed** — the input is its own target.
- **No teacher sharing your features** — kills the circularity that made v1's
  distillation result uninterpretable.
- **Feedback becomes structurally necessary** rather than decorative.

### 6.2 Auxiliary objectives, round-robin

| Objective | Data | Trains |
|---|---|---|
| Vision task | CIFAR-100 | occipital, parietal, frontal |
| Language task | 20 Newsgroups | temporal, frontal |
| Navigation | GridNav | sensor path, parietal, BG |
| Acoustic | synthetic tones | temporal, MGN |
| Hippocampal replay | stored episodes | consolidation, all cortex |

### 6.3 The checks that matter

- **Gradient coverage** must be 100%.
- **Gradient coverage is not contribution.** A module can receive gradient
  forever and still be functionally bypassed. **Ablation delta**: zero each
  module at inference, measure output change. Near-zero delta = decorative.
- **Settling check**: state change per timestep must decrease across `T`.

---

## 7. Dashboard

Native desktop window, no browser, fully offline, black and white.

```
+----------------+---------------------------+----------------+
|  INPUT (20%)   |     CORTEX (centre)       |  SYSTEM (right)|
|  [ text  ]     |  thalamus + 4 lobes,      |  per-module    |
|  [ image ]     |  tracts light on flow,    |  active state  |
|  [ video ]     |  timestep counter t/T     |  CPU / GPU     |
|  [ audio ]     |                           |  neuromod bars |
|                |  [ expand ]               |  osc phase     |
+----------------+---------------------------+----------------+
|         OUTPUT — chatbot style, text only in v1              |
+--------------------------------------------------------------+
```

**Expanded view** — thalamic nuclei, four laminar lobes, amygdala, hippocampal
subfields, BG pathways, cerebellum, every named tract lighting up in routing
order, full-size heatmaps, and a **timestep scrubber**.

**Threading rule:** never render inside the inference loop. Bounded thread-safe
queue; UI drains on its own timer.

---

## 8. File structure

```
brain-ai/
├── config.py                LATENT_DIM, T_STEPS, sparsity k, osc freqs, delays
├── cortex.py                THE SETTLING LOOP — parallel 4-lobe update
├── modules/
│   ├── encoders.py          frozen backbones, spatial/token/frame taps
│   ├── tracts.py            named white-matter Tract class + delays
│   ├── laminar.py           L4 / L2-3 / L5-6 base class
│   ├── lobes.py             4 lobes = laminar shell + own operator
│   ├── thalamus.py          LGN/MGN/VPL/pulvinar/MD/VA-VL + TRN
│   ├── basal_ganglia.py     direct/indirect/hyperdirect, SNc dopamine
│   ├── amygdala.py          low road + high road
│   ├── hippocampus.py       EC → DG → CA3 → CA1 → subiculum, replay
│   ├── hypothalamus.py      homeostatic drives, slow timescale
│   ├── cerebellum.py        forward model + climbing fiber error
│   ├── neuromod.py          ACh, NE, DA, 5-HT
│   ├── oscillations.py      theta / alpha / gamma phase gains
│   └── predictive.py        prediction-error objective
├── train.py                 predictive coding + round-robin tasks
├── verify.py                shapes, parallelism, settling, gradient coverage
├── ablate.py                lesion any module or tract, measure delta
└── desktop_app.py           dashboard
```

---

## 9. Build order — gated, do not skip

| Stage | Build | Gate that must pass |
|---|---|---|
| 1 | `cortex.py` settling loop, 4 lobes, no subcortex | `verify.py`: shapes flow, lobes provably parallel, state change decreases over T |
| 2 | Un-pooled encoder taps | parietal receives 49 real positions; temporal receives a real sequence |
| 3 | Feedback + corticothalamic + TRN | ablating feedback changes output measurably |
| 4 | Laminar shells, k-WTA, E/I split | gradient coverage 100%; sparsity actually ~5% |
| 5 | BG full circuit, loop closed to thalamus | striatal gradient > 1e-4, gate value distribution moves across epochs |
| 6 | Amygdala, hippocampus, hypothalamus | replay improves a held-out task; low road fires before high road |
| 7 | Cerebellum forward model | prediction error decreases over training |
| 8 | Neuromodulators, oscillations | changing ACh measurably shifts FF/FB balance |
| 9 | `ablate.py` full sweep | every module has non-trivial ablation delta |
| 10 | `desktop_app.py` | — |

**Do not build the UI before stage 9.**

---

## 10. Known open questions

- Label set for the output heads is still placeholder.
- Physical sensor format deferred.
- Whether the added subcortical modules pay is unknown — that is what
  `ablate.py` exists to answer, and the honest answer may be no for some.
- v1's finding stands until re-measured: routing made the system more robust to
  sensor failure and less robust to internal damage. That dissociation is the
  most publishable result so far and should be re-run after this upgrade.
