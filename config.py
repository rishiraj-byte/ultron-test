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

LANGUAGE_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
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

UI_FONT = ("Consolas", 10)
UI_FONT_SMALL = ("Consolas", 8)
UI_FONT_TINY = ("Consolas", 7)
UI_FONT_TITLE = ("Consolas", 10, "bold")

UI_REFRESH_MS = 100          # UI drains the activation queue on its own timer
UI_TELEMETRY_MS = 1000       # CPU/GPU sampling interval
UI_MIN_WIDTH = 1280
UI_MIN_HEIGHT = 760
UI_INPUT_PANE_FRACTION = 0.20   # left column, PROJECT.md section 7.1
UI_BRACKET = 11              # corner-bracket arm length on framed panels
