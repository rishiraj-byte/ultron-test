"""
modulators.py -- the four diffuse systems, as actual nuclei.

WHAT THESE ARE AND WHY THEY ARE DIFFERENT FROM EVERYTHING ELSE.

Every other projection in v3 is point-to-point: this population contacts that
one, through a specific sparse mask, and the message means something specific.
These four are not like that. Each is a tiny cluster of cells -- the locus
coeruleus is about 15,000 neurons in a human, out of 86 billion -- whose axons
branch so widely that a single cell can contact most of the cortex. They do
not carry a message. They set a STATE.

That is why they are modelled as small populations whose output is a scalar
concentration rather than a spike pattern delivered somewhere in particular.
Getting that right matters: v2 represented neuromodulators as sixteen learned
weights producing three numbers, with no cells behind them at all.

The four:

  Locus coeruleus      noradrenaline   arousal, and the gain on everything
  Raphe nuclei         serotonin       patience, and the cost of waiting
  Basal forebrain      acetylcholine   attention, and the balance between
                                       believing your senses and your priors
  VTA / SNc            dopamine        reward prediction error

The acetylcholine one is the most functionally interesting. High ACh tilts
cortex toward its feedforward input -- trust what you are seeing. Low ACh
tilts it toward feedback -- trust what you expect. That single dial is the
difference between examining something novel and recognising something
familiar, and it is thought to be part of why hallucination and dreaming look
the way they do: feedback running with no sensory drive to check it.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from v3 import config_v3 as cfg
from v3.neurons import LIFPopulation


class DiffuseNucleus(nn.Module):
    """A small cluster whose output is a concentration, not a message.

    The firing rate of the population IS the signal. There is no projection
    matrix because in the real thing the axons go essentially everywhere, and
    a mask that is all ones is not carrying information.
    """

    def __init__(self, size: int, name: str, baseline: float,
                 tau_ms: float):
        super().__init__()
        self.name = name
        self.baseline = baseline
        self.cells = LIFPopulation(size, name=name,
                                   target_hz=cfg.MODULATOR_TARGET_HZ)
        self.register_buffer("level", torch.tensor(baseline))
        self.register_buffer("tau_ms", torch.tensor(tau_ms))

    def reset(self, batch: int, device) -> None:
        #  The LEVEL is not reset. A neuromodulator concentration does not
        #  return to baseline because a new stimulus arrived -- these are the
        #  slow variables, and their persistence across percepts is the point.
        self.cells.reset(batch, device)

    def step(self, drive: torch.Tensor) -> float:
        self.cells.receive(drive, "exc")
        self.cells.step()
        rate = float(self.cells.spikes.mean()) * (1000.0 / cfg.DT_MS)
        #  Concentration follows firing with its own slow kinetics -- a
        #  transmitter diffusing through tissue does not track a spike train.
        alpha = cfg.DT_MS / float(self.tau_ms)
        self.level += alpha * (rate / cfg.MODULATOR_TARGET_HZ - self.level)
        return float(self.level)


class Modulators(nn.Module):
    """All four, and the state they jointly define."""

    def __init__(self, cortex_size: int):
        super().__init__()
        self.locus_coeruleus = DiffuseNucleus(
            64, "locus_coeruleus", 1.0, cfg.NORADRENALINE_TAU_MS)
        self.raphe = DiffuseNucleus(
            64, "raphe", 1.0, cfg.SEROTONIN_TAU_MS)
        self.basal_forebrain = DiffuseNucleus(
            64, "basal_forebrain", 1.0, cfg.ACETYLCHOLINE_TAU_MS)
        self.vta = DiffuseNucleus(
            64, "vta", 0.0, cfg.DOPAMINE_TAU_MS)

    def reset(self, batch: int, device) -> None:
        for nucleus in (self.locus_coeruleus, self.raphe,
                        self.basal_forebrain, self.vta):
            nucleus.reset(batch, device)

    def step(self, salience: torch.Tensor, novelty: torch.Tensor) -> dict:
        """Advance all four.

        `salience` comes from the amygdala and `novelty` from CA1 -- the two
        signals that actually drive these nuclei in a brain. Noradrenaline
        tracks salience; acetylcholine tracks novelty, which is why an
        unfamiliar scene makes you attend to what is actually there rather
        than to what you assumed would be.
        """
        noradrenaline = self.locus_coeruleus.step(salience * 2.0)
        acetylcholine = self.basal_forebrain.step(novelty * 2.0)
        serotonin = self.raphe.step(torch.zeros_like(salience))
        return {"noradrenaline": noradrenaline,
                "acetylcholine": acetylcholine,
                "serotonin": serotonin}

    def sensory_weight(self, acetylcholine: float) -> float:
        """How much cortex should trust its senses right now, 0 to 1.

        High ACh means attend to the input; low ACh means lean on what you
        already expect. The same dial in reverse is why a system running on
        feedback with no sensory check produces confident nonsense.
        """
        tilt = cfg.ACH_TILT * (acetylcholine - 1.0)
        return float(min(max(0.5 + tilt, 0.05), 0.95))


__all__ = ["Modulators", "DiffuseNucleus"]
