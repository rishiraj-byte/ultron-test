"""Brain-Inspired AI -- modular cortical lobes routed by a thalamus.

Import order matters only in that config must be importable standalone; every
module below reads its shapes from it and never hardcodes a latent size.
"""

from modules.activations import (
    ActivationBus,
    ActivationFrame,
    ActivationRecorder,
    build_frame,
    heatmap_stats,
    to_heatmap,
)
from modules.basal_ganglia import BasalGanglia, GateDecision
from modules.encoders import (
    AudioEncoder,
    BaseEncoder,
    EncoderBank,
    Projection,
    TextEncoder,
    VisionEncoder,
)
from modules.lobes import (
    Cortex,
    FrontalLobe,
    Lobe,
    OccipitalLobe,
    ParietalLobe,
    TemporalLobe,
)
from modules.output_generator import Answer, OutputGenerator, Prediction
from modules.pipeline import BrainAI, BrainTrace, Stage, build_brain
from modules.thalamus import RelayNucleus, Routing, Thalamus

__all__ = [
    "ActivationBus",
    "ActivationFrame",
    "ActivationRecorder",
    "build_frame",
    "heatmap_stats",
    "to_heatmap",
    "BasalGanglia",
    "GateDecision",
    "AudioEncoder",
    "BaseEncoder",
    "EncoderBank",
    "Projection",
    "TextEncoder",
    "VisionEncoder",
    "Cortex",
    "FrontalLobe",
    "Lobe",
    "OccipitalLobe",
    "ParietalLobe",
    "TemporalLobe",
    "Answer",
    "OutputGenerator",
    "Prediction",
    "BrainAI",
    "BrainTrace",
    "Stage",
    "build_brain",
    "RelayNucleus",
    "Routing",
    "Thalamus",
]
