"""
config.py -- the single source of truth for every shape in the system.

PROJECT.md section 3: LATENT_DIM lives here and nowhere else. A bare `256`
in any other file is a bug. Every derived shape is computed from these
constants, so changing LATENT_DIM changes the whole system consistently.
"""

from __future__ import annotations

import os
from pathlib import Path

# ---------------------------------------------------------------------------
# The decision that gates everything (PROJECT.md section 3)
# ---------------------------------------------------------------------------

LATENT_DIM = 256

# Every lobe is LATENT_DIM in, LATENT_DIM out. The internal width of a lobe's
# MLP is derived, never hardcoded.
LOBE_HIDDEN_DIM = LATENT_DIM * 2

# ---------------------------------------------------------------------------
# Encoder raw dimensions (PROJECT.md section 5)
# These are properties of the borrowed pretrained models, not choices.
# Verified against the actual downloaded checkpoints.
# ---------------------------------------------------------------------------

VISION_RAW_DIM = 512   # ResNet-18 penultimate (avgpool) features
TEXT_RAW_DIM = 384     # all-MiniLM-L6-v2 sentence embedding
AUDIO_RAW_DIM = 384    # whisper-tiny encoder d_model

# Embodied proximity sensing: a ring of rangefinder beams, as on a mobile
# robot. Unlike the three above there is no pretrained backbone to borrow and
# none is wanted -- the reading IS the feature, so this encoder is a small
# trainable MLP rather than a frozen model plus a projection.
SENSOR_BEAMS = 8
SENSOR_RAW_DIM = SENSOR_BEAMS * 2      # (normalised distance, hit flag) per beam

VISION_MODEL = "resnet18"
TEXT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
AUDIO_MODEL = "openai/whisper-tiny"

# ---------------------------------------------------------------------------
# Input preprocessing
# ---------------------------------------------------------------------------

IMAGE_SIZE = 224
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

AUDIO_SAMPLE_RATE = 16_000   # whisper expects 16 kHz mono
AUDIO_MAX_SECONDS = 30       # whisper's fixed receptive window

TEXT_MAX_TOKENS = 256

# Whisper emits 1500 frames for 30 s. Running all of them through a GRU on
# every one of T settling timesteps is 12000 recurrent steps per percept, for a
# signal that is highly redundant frame to frame. Strided mean-pooling down to
# this many keeps order, onsets and duration -- it is downsampling, not the
# pooling-to-a-point that v1 did.
AUDIO_FRAMES = 64

# ---------------------------------------------------------------------------
# Thalamic routing table (PROJECT.md section 2.3)
#
# The thalamus is a modality-keyed switchboard, NOT a funnel. Each modality
# has its own relay nucleus, mirroring LGN / MGN / VPL in a real brain.
#
# Parietal is deliberately absent here: it is fed by OCCIPITAL, never by the
# thalamus directly. That is the dorsal "where" stream (V1 -> parietal).
# ---------------------------------------------------------------------------

#   modality -> (relay nucleus name, destination lobe)
THALAMIC_ROUTES: dict[str, tuple[str, str]] = {
    "image": ("LGN", "occipital"),
    "audio": ("MGN", "temporal"),
    # DELIBERATE DEVIATION (PROJECT.md section 2.4): text has no thalamic relay
    # in a real brain -- reading enters through vision. Routing it straight to
    # temporal via a pulvinar-analogue nucleus is an engineering simplification.
    "text": ("PULVINAR", "temporal"),
    # Proprioceptive / rangefinder input for embodied navigation. VPL/VPM is
    # the somatosensory relay named in section 2.3, and it is the ONE route in
    # that list that was never built -- parietal previously had no thalamic
    # input at all. This completes the switchboard, and it is what gives
    # parietal a "where is my body in the world" signal to fuse with
    # occipital's "what am I looking at".
    "sensor": ("VPL", "parietal"),
}

MODALITIES = tuple(THALAMIC_ROUTES.keys())
RELAY_NUCLEI = tuple(dict.fromkeys(n for n, _ in THALAMIC_ROUTES.values()))

# Intracortical wiring the thalamus does not touch.
#   occipital -> parietal   (dorsal "where" stream)
#   occipital -> frontal    (direct, bypassing parietal)
CORTICAL_EDGES: tuple[tuple[str, str], ...] = (
    ("occipital", "parietal"),
    ("occipital", "frontal"),
    ("parietal", "frontal"),
    ("temporal", "frontal"),
)

LOBES = ("occipital", "parietal", "temporal", "frontal")

# ---------------------------------------------------------------------------
# Frontal convergence (PROJECT.md section 4 -- the resolved shape conflict)
#
# Frontal is NOT a plain LATENT_DIM-input lobe. It receives a variable number
# of LATENT_DIM streams and fuses them down to LATENT_DIM before behaving like
# any other lobe. Slot order is fixed so that a missing stream is always
# zero-padded into the SAME position -- otherwise the fusion layer would see
# a different meaning per slot depending on modality.
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Per-lobe inductive biases (modules/lobes.py)
#
# The central claim of this architecture is that specialised sub-networks beat
# one monolithic network. That claim is only testable if the lobes are actually
# different, so each carries a structural prior matched to its signal:
#
#   occipital  grouped/local connectivity  -> receptive fields
#   parietal   self-attention over slots   -> relations between positions
#   temporal   recurrence over steps       -> order in language and audio
#
# All must divide LATENT_DIM exactly; the lobes raise at construction if not.
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# v2 stage 4 -- laminar microcircuit, sparsity, E/I balance
#
# Cortex runs sparse: a few percent of neurons active at once, not half. This
# is enforced as a hard k-winners-take-all mask rather than an L1 penalty,
# because a penalty is something the optimiser trades against the task loss and
# usually loses. verify_v2.py MEASURES the achieved rate rather than trusting
# the setting.
CORTICAL_SPARSITY = 0.05
# How much of a head start a currently-active unit gets in the competition,
# in units of the score's own standard deviation. Set from the sweep in
# modules/laminar.k_winners; 0.0 reproduces plain memoryless k-WTA.
SPARSITY_HYSTERESIS = 4.0     # 5% of units survive L2/3 at the FINAL timestep

# Competition sharpens across the settling window rather than clamping from
# t=0. Fraction active goes from SPARSITY_START down to CORTICAL_SPARSITY
# linearly across T. Required for the loop to converge at all -- see the note
# in modules/laminar.py:l23.
SPARSITY_START = 0.60

# Fraction of the settling window spent tightening competition. After this the
# mask is HELD at the target, so the final timesteps converge with sparsity
# fixed -- otherwise the annealing itself perturbs every step and the loop can
# never be shown to settle rather than merely be dragged.
SPARSITY_ANNEAL_FRACTION = 0.5

# ~80/20 excitatory/inhibitory, and inhibitory neurons cannot excite. A plain
# Linear violates this; see modules/laminar.py.
INHIBITORY_FRACTION = 0.20

OCCIPITAL_GROUPS = 8       # local bands; LATENT_DIM/8 = 32 units per field
PARIETAL_POSITIONS = 16    # latent read as 16 places of 16 dims each
TEMPORAL_STEPS = 8         # latent read as 8 time steps of 32 dims each

FRONTAL_STREAM_SLOTS: tuple[str, ...] = ("occipital", "parietal", "temporal")
MAX_STREAMS = len(FRONTAL_STREAM_SLOTS)
FRONTAL_FUSION_IN_DIM = MAX_STREAMS * LATENT_DIM

# "concat" keeps which stream contributed what, at the cost of parameters.
# "sum" is free but discards provenance. See PROJECT.md section 4.
FRONTAL_FUSION_MODE = "concat"

# ---------------------------------------------------------------------------
# v2 -- THE SETTLING LOOP (BUILD_SPEC_V2 section 2)
#
# Cortex is not a pipeline. Every area is active on every timestep, exchanging
# signals across conduction delays, settling over what would be ~100-300 ms.
#
# T_STEPS is how many exchange rounds one percept gets. Every cross-module read
# is from state[t - delay], never state[t] -- that is what makes the four lobes
# genuinely parallel rather than a pipeline in disguise, and the delay IS the
# conduction latency rather than a workaround for it.
# ---------------------------------------------------------------------------

# Measured on REAL percepts, which are the harder test.
#
# An earlier revision raised this to 16 to work around dynamics that could
# not converge (sparsity was being applied inside L2/3 only, so the committed
# state was 100% dense). Once sparsity moved onto the committed state and
# k-WTA gained hysteresis, T=8 passed on 8 synthetic seeds and the number
# went back down. Encoded percepts then showed why 8 is too tight: settling
# is measured over the tail AFTER annealing, which at T=8 is only four
# steps, and a single late mask flip -- two units trading places when the
# state is already at 7% of its opening movement -- is enough to break it
# on some initialisations. Four real percepts through the frozen encoders:
#
#              green/rocket  red/question  blue/list  grey/story
#     T=8         0.151         0.164        0.112      0.087
#     T=12        0.064         0.074        0.070      0.089
#
# T=12 roughly halves every ratio and leaves a six-step tail, so one flip no
# longer decides the verdict. The cost is four extra timesteps.
T_STEPS = 12

# Conduction delay per tract, in timesteps. Longer tract, longer delay -- a
# principled reason for different t-k offsets rather than an arbitrary one.
# Every cortical tract is at least 1: a lobe may never read another lobe's
# CURRENT state, or the "parallel" claim is false.
TRACT_DELAYS: dict[str, int] = {
    "optic_radiation": 1,   # LGN -> occipital, short
    "acoustic_radiation": 1,  # MGN -> temporal
    "somato_radiation": 1,  # VPL -> parietal
    "ilf": 1,               # occipital <-> temporal, ventral "what"
    "slf": 1,               # parietal <-> frontal, dorsal "where/how"
    "ifof": 2,              # occipital <-> frontal, long association fibre
    "arcuate": 2,           # temporal <-> frontal, long
    "recurrent": 1,         # frontal -> frontal, working memory
}

# MEMBRANE TIME CONSTANT. A cortical population does not instantaneously adopt
# whatever its inputs imply; it integrates toward it. Discretised rate model:
#
#     state[t] = (1 - alpha) * state[t-1] + alpha * f(drive)
#
# Without this term the loop has no contraction: each timestep recomputes the
# state from scratch, so it wanders instead of converging. Measured on the
# first build with no leak, state change per step went 98 -> 120 -> 104 across
# T -- moving MORE at the end than the start, which fails the stage-1 gate.
# Alpha is the fraction of the new drive admitted per step; lower is slower and
# smoother. 1.0 reproduces the non-settling behaviour exactly.
CORTICAL_LEAK = 0.25

# A settled state should stop moving. Measured as ||state[t] - state[t-1]||.
#
# The gate the spec actually names is "state change per timestep DECREASES
# across T" -- monotonicity, which is a property rather than a magic number.
# The ratio below is a secondary sanity bound, and it is set from measurement
# rather than taste: at T=8 the convergence ratio plateaus near 0.38 for any
# leak that settles at all (0.48 at leak 0.2, 0.38 at 0.45 and 0.6), so a
# halving is comfortably achievable and an earlier guess of 0.35 was
# unreachable by construction. Both conditions must hold.
SETTLING_TOLERANCE = 0.25   # final delta <= a QUARTER of the first

# Timesteps to exclude from the settling measurement at the start.
#
# At t=0 the loop has not closed: every tract carries delay >= 1, so every
# cross-module read returns resting state and the lobes respond to sensory
# drive alone. The jump from rest to first response is ONSET, not settling, and
# once the thalamus joined the loop (stage 3) it showed up as a delta that
# rises before it falls -- 25 -> 43 -> 37 -> 31 -> ... -> 21. From t=1 that
# series is strictly decreasing.
#
# This is a structural fact about the delays, not a threshold chosen to make a
# test pass: with max tract delay 2, the loop is only fully closed from t=2,
# and one warm-up step is the conservative choice.
def _settling_warmup() -> int:
    """Timesteps excluded from the settling measurement.

    Two reasons stack, and both are structural rather than chosen:

    1. At t=0 the loop has not closed -- every tract delay is >= 1, so nothing
       recurrent has arrived and the lobes respond to sensory drive alone.
    2. While sparsity is annealing (stage 4) the DYNAMICS THEMSELVES are still
       changing. Convergence of a system whose rules are still moving is not a
       meaningful thing to measure, so settling is assessed once competition
       has reached its target and been held.

    The second subsumes the first, so this is simply the end of annealing.
    """
    return max(int(T_STEPS * SPARSITY_ANNEAL_FRACTION), 1)


SETTLING_WARMUP = _settling_warmup()

# Measured: leak >= 0.8 does not settle monotonically, leak <= 0.6 always does.
# 0.45 sits mid-range with margin on both sides.

# ---------------------------------------------------------------------------
# Basal ganglia gate (PROJECT.md section 2.1)
# ---------------------------------------------------------------------------

# Below this, the action is suppressed and the output generator reports a
# withheld decision rather than a confident wrong one.
BG_GATE_THRESHOLD = 0.5

# ---------------------------------------------------------------------------
# Output (PROJECT.md section 11 -- the label-set open question, now closed)
#
# The label set is ImageNet-1k, inherited from the frozen ResNet-18 encoder.
# This is what makes step 3 produce a MEANINGFUL answer: the lobes are trained
# by distillation against the encoder's own 1000-way head (see train.py), so
# there is a real objective and a real vocabulary without hand-labelling data.
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Language production (modules/language.py) -- the Broca's-area analogue
#
# The lobes perceive; this speaks. Frozen and borrowed, exactly like the
# sensory encoders: it supplies grammar and fluency, never perception. Small
# enough to generate at a usable rate on CPU (~10 tok/s measured).
# ---------------------------------------------------------------------------

#  Upgraded from 0.5B. The small model ignored its instructions -- told not to
#  invent detail, it still reported that the system had "recognized 'red' as an
#  object" when the report said no such thing. 1.5B in fp16 is 3.0 GB against
#  5.4 GB free once the encoders are resident, which fits; 3B does not.
LANGUAGE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
LANGUAGE_MAX_TOKENS = 160
LANGUAGE_HISTORY_TURNS = 6      # how much conversation to carry forward

LABEL_SET = "imagenet1k"
NUM_CLASSES = 1000
TOP_K = 5

# ---------------------------------------------------------------------------
# Activation capture (PROJECT.md section 6)
# ---------------------------------------------------------------------------

# Deliberately shallow. A deep queue lets the display drift behind reality.
ACTIVATION_QUEUE_MAXSIZE = 32


def _square_side(n: int) -> int:
    """256 -> 16. Derived, so the heatmap layout follows LATENT_DIM."""
    side = int(round(n ** 0.5))
    if side * side != n:
        raise ValueError(
            f"LATENT_DIM={n} is not a perfect square; flat activations cannot "
            f"be reshaped to a square heatmap. Pick a square latent dim or "
            f"change the heatmap layout."
        )
    return side


LATENT_GRID_SIDE = _square_side(LATENT_DIM)

# Convolutional taps inside the frozen vision encoder, in depth order.
# These produce the classic "the model is looking here" spatial heatmaps.
VISION_TAPS = ("layer1", "layer2", "layer3", "layer4")

# ---------------------------------------------------------------------------
# Runtime
# ---------------------------------------------------------------------------


def resolve_device() -> str:
    """Honour BRAINAI_DEVICE, else CUDA when present, else CPU."""
    forced = os.environ.get("BRAINAI_DEVICE")
    if forced:
        return forced
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    return "cpu"


SEED = 0

PROJECT_ROOT = Path(__file__).resolve().parent
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
ASSET_DIR = PROJECT_ROOT / "assets"
DEFAULT_CHECKPOINT = CHECKPOINT_DIR / "brain.pt"

# ---------------------------------------------------------------------------
# Dashboard (PROJECT.md section 7) -- simple black and white, native, offline.
# ---------------------------------------------------------------------------

# Tactical HUD palette. Near-black ground, amber for anything live, steel for
# anything structural. One accent hue only -- a readout that uses five colours
# stops being readable at a glance, which is the entire point of a HUD.
UI_BG = "#05070a"            # ground
UI_PANEL = "#080b10"         # panel fill, barely lifted off the ground
UI_GRID = "#0e141c"          # background rule lines
UI_EDGE_IDLE = "#1b2530"     # structure at rest
UI_DIM = "#3d4a5a"           # de-emphasised text
UI_MID = "#7e93a8"           # body text
UI_FG = "#d8e4f0"            # primary text
UI_ACCENT = "#ff9020"        # amber -- live signal, active modules
UI_ACCENT_DIM = "#6b4413"    # amber at rest
UI_ALERT = "#ff4a28"         # fault / suppressed
UI_COOL = "#4fd8ff"          # secondary data, wireframes
UI_EDGE_ACTIVE = UI_ACCENT

#  TYPOGRAPHY. One family, four sizes, and weight used to mean something.
#  The previous set ran from 7pt to 10pt, which on a high-DPI panel is a
#  smear -- labels inside the routing diagram were unreadable and the
#  hierarchy between a title and a data label had collapsed. Sizes now step
#  clearly and every structural label is bold, so the eye can find the
#  heading, the module name and the number without reading any of them.
UI_FONT = ("Consolas", 11)
UI_FONT_SMALL = ("Consolas", 10)
UI_FONT_TINY = ("Consolas", 9)
UI_FONT_TITLE = ("Consolas", 13, "bold")
UI_FONT_LABEL = ("Consolas", 10, "bold")      # module names in the diagram
UI_FONT_DATA = ("Consolas", 9, "bold")        # firing counts, gate values

UI_REFRESH_MS = 100          # UI drains the activation queue on its own timer
UI_TELEMETRY_MS = 1000       # CPU/GPU sampling interval
UI_MIN_WIDTH = 1280
UI_MIN_HEIGHT = 760
UI_INPUT_PANE_FRACTION = 0.20   # left column, PROJECT.md section 7.1
UI_BRACKET = 11              # corner-bracket arm length on framed panels


# ---------------------------------------------------------------------------
# Basal ganglia (BUILD_SPEC_V2 3.4, stage 5)
# ---------------------------------------------------------------------------
#  The direct and indirect populations must NOT start identical. Zero-init
#  gives them equal and opposite gradients that cancel in the sum, and the
#  striatum then receives ~1e-6 of gradient forever: a dead module that looks
#  alive because the loss still falls elsewhere. Small random weights, and
#  biases far enough apart that the two pathways begin in different regimes.
STRIATUM_INIT_STD = 0.02
STRIATUM_D1_BIAS = 1.5
STRIATUM_D2_BIAS = -1.5

#  GPi fires tonically and inhibits the thalamus, so the circuit rests CLOSED
#  and the direct pathway has to open it. A gate that rested open would make
#  selection something the circuit does by exception rather than by default.
GPI_TONIC = 1.0

#  Dopamine. The tonic level is what SNc emits with no reward to report; the
#  sensitivity is how far a prediction error moves striatal learning rate.
DOPAMINE_TONIC = 0.5
DOPAMINE_SENSITIVITY = 1.0
#  The smallest learning rate either striatal arm may be scaled to.
#
#  Without a floor the scaling is `1 -/+ sensitivity x phasic`, clamped at
#  zero, and a large enough prediction error multiplies one arm's gradient by
#  EXACTLY zero -- not "learns slowly", but "does not learn at all, this
#  batch". Measured before the floor existed, D2's learning rate was exactly
#  zero on 11 of 12 timesteps and its gradient came out at 0.000000.
#
#  Real striatal plasticity is biased by dopamine, not abolished by it: D2
#  cells still undergo LTP and LTD across the physiological dopamine range.
#  A floor says the same thing, and it keeps the pathway trainable while
#  leaving the asymmetry -- at this value D1 can still learn twenty times
#  faster than D2 under strong positive error.
DOPAMINE_LR_FLOOR = 0.05

#  Delays for the subcortical tracts stage 5 adds.
TRACT_DELAYS.update({
    "corticostriatal": 1,   # frontal -> striatum, the selection request
    "hyperdirect": 1,       # frontal -> STN, the fast global brake
    "bg_thalamic": 1,       # GPi -> VA/VL and MD, the return path
    "md_frontal": 1,        # MD -> frontal L4
    "vavl_frontal": 1,      # VA/VL -> frontal L4
})


# ---------------------------------------------------------------------------
# Limbic system (BUILD_SPEC_V2 3.3, stage 6)
# ---------------------------------------------------------------------------
#  Dentate gyrus. Sparser than cortex by design -- pattern SEPARATION needs a
#  wide layer with very few winners, so that two similar inputs land on nearly
#  disjoint sets rather than overlapping ones.
DG_EXPANSION = 4
DG_SPARSITY = 0.02

#  CA3 recurrent settling steps for pattern completion. More steps complete
#  more aggressively and also hallucinate more confidently.
CA3_STEPS = 3

#  How far CA3 moves toward its recall each step, and how loudly the learned
#  collaterals speak next to the Hebbian trace. The gain starts small so an
#  untrained CA3 completes correctly out of the box; see CA3.forward for the
#  measurement that forced this.
CA3_COMPLETION_RATE = 0.5
CA3_COLLATERAL_GAIN = 0.1

#  How many episode batches the hippocampus keeps available for replay.
REPLAY_CAPACITY = 64

#  Hypothalamic drives move across EPISODES, not timesteps, so this is small.
HOMEOSTATIC_RATE = 0.1

#  The resting value of every homeostatic drive. Not zero: see Hypothalamus.
HOMEOSTATIC_REST = 0.1

#  How much valence sharpens the amygdala's gain on top of salience. Strongly
#  positive or strongly negative episodes both warrant more processing, so it
#  is the MAGNITUDE that counts here, not the sign.
VALENCE_WEIGHT = 0.5

#  Limbic tract delays. The low road must arrive strictly before the high one;
#  the spec asks for thalamus at t and temporal at t-1, but a delay of 0 is
#  rejected by the parallelism gate, so both shift by one and the GAP -- which
#  is what carries the meaning -- is preserved.
TRACT_DELAYS.update({
    "low_road": 1,          # thalamus -> amygdala, fast and crude
    "high_road": 2,         # temporal -> amygdala, slow and detailed
    "cortico_hippocampal": 1,   # association cortex -> entorhinal
    "fornix": 1,            # subiculum -> hypothalamus and frontal
    "amygdala_cortex": 1,   # salience modulates cortical gain
    "hypothalamic": 1,      # homeostatic condition -> frontal deep layers
})


# ---------------------------------------------------------------------------
# Pretrained lobe cores
# ---------------------------------------------------------------------------
#  Each lobe runs a frozen pretrained network at its core instead of a randomly
#  initialised operator. Which layer is taken from each teacher matters: early
#  layers are closer to the input statistics, late layers more abstract.
PRETRAINED_LOBES = True
PRETRAINED_PARIETAL_LAYER = 3    # mid-depth: relations, not yet task-specific
PRETRAINED_TEMPORAL_LAYER = 2    # whisper encoder, sequence structure
PRETRAINED_FRONTAL_LAYER = 5     # MiniLM's last layer, the most abstract


# ---------------------------------------------------------------------------
# Cerebellum (BUILD_SPEC_V2 3.5, stage 7)
# ---------------------------------------------------------------------------
#  Granule-cell expansion. The cerebellum has more neurons than the rest of the
#  brain combined, almost all of them granule cells, and the width is the
#  mechanism: a very wide sparse layer makes nearly any two inputs linearly
#  separable, so a shallow readout can learn arbitrary mappings fast.
CEREBELLAR_EXPANSION = 4

TRACT_DELAYS.update({
    "pontine": 1,        # frontal -> cerebellum, the efference copy
    "dentate_vl": 1,     # cerebellum -> thalamus VL, the correction
})


# ---------------------------------------------------------------------------
# Neuromodulators and oscillations (BUILD_SPEC_V2 3.6 / 3.7, stage 8)
# ---------------------------------------------------------------------------
#  The names of the homeostatic drives, shared with the hypothalamus so the two
#  cannot disagree about how many there are.
HYPOTHALAMIC_DRIVES = ("energy", "temperature", "threat", "satiety")

#  Resting levels. Dopamine is NOT here: it is produced by the substantia nigra
#  built at stage 5, and DOPAMINE_TONIC above is its single definition.
ACH_BASELINE = 1.0
NE_BASELINE = 1.0
SEROTONIN_BASELINE = 1.0

#  The narrowest the feedforward share may get at zero acetylcholine. Not 0:
#  cortex that ignored its senses entirely would be hallucinating, not
#  remembering, and could never be corrected by evidence.
ACH_MIN_SHARE = 0.2

#  How far an untrained modulator readout may stray from baseline. Small, so a
#  fresh system starts with its dials centred instead of somewhere arbitrary.
MODULATOR_INIT_STD = 0.02

#  One settling timestep, in seconds. Without this an oscillation frequency is
#  a bare number multiplied by a loop index and "8 Hz" means nothing.
TIMESTEP_SECONDS = 0.010          # 10 ms per step

OSCILLATION_HZ = {"theta": 6.0, "alpha": 10.0, "gamma": 40.0}
OSCILLATION_AMPLITUDE = {"theta": 0.10, "alpha": 0.15, "gamma": 0.05}


#  Below this, removing a part changes nothing measurable and the part is
#  DECORATIVE. Stated once here so the sweep and its gate cannot disagree
#  about where the line is.
ABLATION_THRESHOLD = 0.01


# ---------------------------------------------------------------------------
# Training the v2 cortex (BUILD_SPEC_V2 section 7)
# ---------------------------------------------------------------------------
#  Five objectives, weighted. One loss cannot reach every pathway -- v1 shipped
#  a checkpoint where 34% of parameters had never seen a gradient because a
#  single classification loss trains a narrow path and nothing else.
OBJECTIVE_WEIGHTS = {
    "task": 1.0,          # classify the percept
    "semantic": 1.0,      # frontal must predict the frozen encoder's meaning
    "predictive": 0.3,    # state at t predicts state at t+1
    "forward": 0.1,       # the cerebellum's own error
    "value": 0.3,         # the SNc value head; nothing else can reach it
}

#  Tokens kept per text example when caching training streams. Long enough for
#  real word order to matter, short enough that the cache fits in memory.
TEXT_TOKENS_TRAIN = 32

TRAIN_DROPOUT = 0.1
GRAD_CLIP = 1.0

#  How often training samples gradient coverage. Every batch is too expensive
#  -- reducing 300 tensors on the GPU costs more than the step being measured.
COVERAGE_SAMPLE_EVERY = 7

#  Head sizes used before a checkpoint is loaded, so an untrained system can
#  still run and report itself as untrained instead of failing to construct.
DEFAULT_CLASSES = {
    "image": 100,     # CIFAR-100
    "text": 20,       # 20 newsgroups
    "audio": 6,       # synthetic acoustic classes
    "sensor": 4,      # grid navigation moves
}


# ---------------------------------------------------------------------------
# Telemetry
# ---------------------------------------------------------------------------
#  Whether the settling loop pulls its instrumentation back to the CPU as it
#  runs -- gate openings, achieved sparsity, novelty, per-step deltas.
#
#  Each of those is a float(tensor) call, and on a GPU that is a full
#  synchronisation: the CPU stops until every queued kernel has finished. The
#  loop does about fifteen of them per timestep, which on the RTX 3050 made a
#  single settle take 3.6 SECONDS against roughly 0.2 on the CPU, and turned
#  one training epoch into twenty minutes. The numbers themselves cost
#  nothing; waiting for them costs everything.
#
#  So the dashboard leaves this on and training turns it off. Nothing about
#  the computation changes either way -- only whether the answers are read
#  back while the loop is still running.
COLLECT_TELEMETRY = True


class no_telemetry:
    """Turn instrumentation off for a block. Training uses this."""

    def __enter__(self):
        global COLLECT_TELEMETRY
        self._was = COLLECT_TELEMETRY
        COLLECT_TELEMETRY = False
        return self

    def __exit__(self, *exc):
        global COLLECT_TELEMETRY
        COLLECT_TELEMETRY = self._was
        return False

#  The magnitude every lobe's committed state is normalised to. Cortex has no
#  other bound on its own activity and its loops have gain above one; see the
#  divisive-normalisation note in cortex.settle for the measurement that
#  forced this. The value is arbitrary in the way a unit is arbitrary -- what
#  matters is that it is constant.
CORTICAL_STATE_NORM = 4.0

#  The width each modality arrives at, before the trained projection into
#  LATENT_DIM. Stated once so the trainer and the assembled system cannot
#  disagree about a shape.
RAW_DIMS = {
    "image": VISION_RAW_DIM,     # ResNet layer4 channels
    "text": TEXT_RAW_DIM,        # MiniLM hidden
    "audio": AUDIO_RAW_DIM,      # Whisper encoder d_model
    "sensor": 2,                 # (distance, hit) per beam
}

#  Frames sampled from a video clip. The settling loop integrates ONE percept
#  and has no notion of a clip, so these are averaged into a single visual
#  input -- a summary of what the clip looked like, not a claim to have
#  watched it unfold.
VIDEO_FRAMES = 8

#  How far the camera sits from the brain in the 3D view, in the same unit
#  cube the regions live in. Larger is a longer lens: less perspective
#  distortion, less depth cue. Small enough to read as solid, large enough
#  that the frontal pole does not balloon when it swings toward you.
UI_BRAIN_CAMERA = 2.2
UI_BRAIN_SPIN = 0.006        # radians per UI frame when rotating


# ---------------------------------------------------------------------------
# Recognition and abstention
# ---------------------------------------------------------------------------
#  A classifier with a softmax always produces an answer, including for input
#  containing nothing it knows. Measured on this model:
#
#      "a red bicycle leaning on a wall"   similarity 0.493   p 0.968
#      "the doctor said the treatment..."  similarity 0.441   p 0.986
#      "1+1"                               similarity 0.214   p 0.632
#      "6y5ws46w4y6jruk5utdkegsbdn ..."    similarity 0.370   p 0.532
#      "what are you"                      similarity 0.213   p 0.743
#
#  Both signals separate cleanly, so a percept is treated as RECOGNISED only
#  when the semantic space has something genuinely close AND the classifier is
#  not merely picking the least-bad of its labels. Failing either, the system
#  reports that it does not recognise the input -- which is a true statement
#  about a 130-label world, and far better than confidently answering
#  "misc.forsale" to arithmetic.
RECOGNITION_SIMILARITY = 0.40
RECOGNITION_CONFIDENCE = 0.85


# ---------------------------------------------------------------------------
# How the language module is allowed to see the input
# ---------------------------------------------------------------------------
#  FALSE is the architecturally pure setting: the language model is blindfolded
#  and speaks ONLY from the four lobes' report. That is what makes "the lobes
#  are not decoration" provable from the output alone -- if they were bypassed,
#  the answers would be empty.
#
#  It also makes the system unable to answer anything outside its 130-concept
#  world. Asked "1+1" it cannot reply, because nothing that could reply ever
#  sees the question.
#
#  TRUE lets the language model read the user's message ALONGSIDE the
#  perceptual report. The system then answers like an assistant and the cortex
#  grounds rather than replaces that answer. The cost is real and worth stating
#  plainly: the lobes' contribution is no longer visible in the reply text, so
#  the claim that they matter now rests on the ablation sweep and the live
#  dashboard instead of on the conversation. Nothing about the architecture
#  changes -- only what the language module is permitted to read.
#  WHOSE VOICE THE REPLY IS IN.
#
#  The cortex's own answer -- Perception.as_answer(), built purely from what
#  the four lobes computed and involving no language model whatsoever -- is
#  ALWAYS produced and always shown. This setting only decides what the
#  spoken second line adds on top of it:
#
#    "grounded"  Qwen reads the user's text as well as the cortical report,
#                and holds a normal conversation. Most useful, least purely
#                cortical: on a chat turn most of what you read is Qwen.
#    "narrated"  Qwen is blindfolded -- it sees ONLY the cortical report and
#                phrases it. Every fact in the reply then originates in the
#                lobes; Qwen supplies grammar and nothing else.
#    "silent"    No language model runs at all. The cortex's own answer is
#                the entire output.
OUTPUT_MODE = "grounded"
GROUNDED_CHAT = OUTPUT_MODE == "grounded"

#  Gap required between the best and second-best label. Measured on this
#  model: genuine inputs 0.95-1.00, random keystrokes 0.41-0.47, so 0.80
#  separates them with room on both sides. It does NOT separate input that
#  maps confidently onto a wrong class -- see Thought.recognised for why no
#  confidence measure can.
RECOGNITION_MARGIN = 0.80

#  Put a real Qwen2.5 decoder layer inside the frontal lobe.
#
#  The other three cores contribute PERCEPTUAL structure -- what an image
#  looks like, what a sentence is about. A decoder layer from a model trained
#  to continue text carries something different: structure about how one thing
#  follows from another. Frontal is the lobe that decides what follows, so it
#  is the one place that prior belongs.
#
#  One layer is 46.8M parameters, 94 MB in bfloat16, against the 13.7M already
#  frozen across the other three cores -- so this roughly quadruples the frozen
#  knowledge in the cortex for 94 MB. Layer 20 of 28 is late enough to be
#  abstract and early enough not to be specialised for next-token prediction.
FRONTAL_USES_QWEN = True
PRETRAINED_QWEN_LAYER = 20

# ---------------------------------------------------------------------------
# Which pretrained network sits inside each lobe
# ---------------------------------------------------------------------------
#  Each is chosen for what that lobe DOES, not for size. The parietal choice
#  is a correction: it was running a MiniLM layer, which is attention over
#  words, on a lobe whose job is attention over 49 image positions. DINOv2
#  attends over image patches -- the same operation, learned on the same kind
#  of data.
OCCIPITAL_RESNET = 50                       # convolution, local fields
PARIETAL_MODEL = "facebook/dinov2-base"     # attention between positions
TEMPORAL_MODEL = "openai/whisper-small"     # sequence unfolding in time
PARIETAL_LAYER = 6
TEMPORAL_LAYER = 6

#  The scale the striatum sees, whatever the cortex upstream is made of. Every
#  stage of the basal ganglia is a softplus, so the circuit's gradients depend
#  on WHERE on that curve it operates -- and that moved when the lobe cores
#  were swapped, killing the indirect pathway outright.
STRIATUM_INPUT_NORM = 4.0

#  Initial strength of the D2 -> GPe inhibition, before softplus. D1 reaches
#  GPi in one step and D2 in three, so without a head start the indirect
#  pathway starts with a gradient near 1e-4 and never catches up.
#  Swept rather than guessed. D1 reaches GPi in one step, D2 in three, and
#  the gap between their gradients is enormous at any setting -- the question
#  is only whether D2 clears the threshold at which a module counts as alive:
#
#      strength   D1 gradient   D2 gradient   ratio
#          2.0        2.07        2.8e-05     74000      D2 dead
#          4.0        3.06        7.5e-03       410      chosen
#          8.0       67.5         2.2e-01       302      D1 exploding
#         16.0        2.28        3.7e-03       623
NOGO_STRENGTH_INIT = 4.0
