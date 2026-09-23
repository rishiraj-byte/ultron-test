"""
neuromodulators.py -- four global scalars and three rhythms (spec 3.6/3.7, stage 8).

SCALARS, NOT VECTORS, and that restriction is the whole design.

Everything else in this system carries information: 256 numbers, each meaning
something specific about this percept. A neuromodulator carries almost none --
one number, broadcast everywhere at once. That is not a simplification, it is
what makes it a different kind of signal. A vector can tell one lobe something
another lobe does not hear. A scalar cannot say anything specific to anyone; it
can only change how everything already there behaves.

    acetylcholine   nucleus basalis    feedforward vs feedback balance
    norepinephrine  locus coeruleus    global gain, arousal
    dopamine        VTA / SNc          reward prediction error (stage 5 owns it)
    serotonin       raphe nuclei       time horizon, patience

ACETYLCHOLINE IS THE INTERESTING ONE, and it is stage 8's gate. High ACh tips
cortex toward its feedforward inputs -- toward the evidence actually arriving.
Low ACh tips it toward feedback -- toward what it already expected. That single
dial is the difference between perceiving and remembering, and it is why the
same cortex can do both. The gate measures it: turn ACh up and the feedforward
share of cortical drive must measurably rise.

DOPAMINE already exists, in the substantia nigra built at stage 5, and is NOT
reimplemented here. Two structures both claiming to produce dopamine would be
two structures disagreeing about it. This module reads that value and reports
it alongside the others, so there is one place to look and still only one place
it comes from.

OSCILLATIONS: gain = 1 + A*sin(2*pi*f*t + phase), a distinct phase per lobe.

The phase offset is the point of the whole mechanism. With every lobe in phase
a rhythm is just a global volume knob wobbling. Offset, the lobes peak at
different moments within each cycle, so which areas are most excitable depends
on WHEN in the cycle you look -- which is how a rhythm can bind some areas
together and separate others without any wiring change at all.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch
import torch.nn as nn

import config


@dataclass
class ModulatorState:
    """The four scalars at one moment. One number each, deliberately."""

    acetylcholine: float = config.ACH_BASELINE
    norepinephrine: float = config.NE_BASELINE
    dopamine: float = config.DOPAMINE_TONIC
    serotonin: float = config.SEROTONIN_BASELINE

    def report(self) -> str:
        return ("ACh %.3f   NE %.3f   DA %.3f   5-HT %.3f"
                % (self.acetylcholine, self.norepinephrine,
                   self.dopamine, self.serotonin))

    def as_dict(self) -> dict[str, float]:
        return {"acetylcholine": self.acetylcholine,
                "norepinephrine": self.norepinephrine,
                "dopamine": self.dopamine,
                "serotonin": self.serotonin}


class Neuromodulators(nn.Module):
    """Four brainstem and basal-forebrain systems, as four numbers.

    Driven by the hypothalamus (how am I doing) and the amygdala (does this
    matter), which is the right way round: neuromodulator levels are a
    consequence of the organism's condition, not an input to it.
    """

    def __init__(self):
        super().__init__()
        #  One weight per drive per modulator. Tiny on purpose -- a
        #  neuromodulator that could be computed from the percept in detail
        #  would be an information channel wearing a scalar's clothes.
        self.from_drives = nn.Linear(len(config.HYPOTHALAMIC_DRIVES), 3)

        #  INITIALISED SMALL, so an untrained system rests at baseline. With
        #  PyTorch's default init the readout put acetylcholine at 0.56 on a
        #  fresh model -- a permanent tilt toward feedback that nobody chose,
        #  which measurably slowed the settling loop on real percepts (ratio
        #  0.26 against a 0.25 bound) before this was tracked down. A dial
        #  should start in the middle.
        nn.init.normal_(self.from_drives.weight, std=config.MODULATOR_INIT_STD)
        nn.init.zeros_(self.from_drives.bias)

        #  NOT TRAINABLE, and marked rather than left to look dead. A
        #  modulator level leaves this module as a Python float, so there is
        #  no gradient path back into these weights and never can be -- which
        #  is correct: a global gain trained by the task loss would be
        #  optimised to whatever made the current batch easiest, and a
        #  neuromodulator is supposed to reflect the organism's condition
        #  rather than the convenience of the objective.
        self.from_drives.requires_grad_(False)
        self.salience_gain = nn.Parameter(torch.tensor(1.0),
                                          requires_grad=False)

    def forward(self, homeostasis: torch.Tensor | None = None,
                salience: torch.Tensor | None = None,
                dopamine: torch.Tensor | None = None) -> ModulatorState:
        state = ModulatorState()
        #  Every line below reads a tensor back to Python, which on a GPU is a
        #  full synchronisation. Training does not look at these numbers, and
        #  waiting for them cost more than the whole rest of the loop.
        if not config.COLLECT_TELEMETRY:
            return state
        if dopamine is not None:
            #  Read from the SNc, never recomputed here. See module docstring.
            state.dopamine = float(dopamine.mean().detach())

        if homeostasis is not None:
            levels = torch.tanh(self.from_drives(homeostasis)).reshape(-1)
            state.acetylcholine = config.ACH_BASELINE + float(levels[0].detach())
            state.norepinephrine = config.NE_BASELINE + float(levels[1].detach())
            state.serotonin = config.SEROTONIN_BASELINE + float(levels[2].detach())

        if salience is not None:
            #  Arousal follows salience: something that matters wakes the
            #  system up. This is locus coeruleus in one line.
            boost = float((self.salience_gain * salience.mean()).detach())
            state.norepinephrine = state.norepinephrine + boost

        #  Clamped, because a modulator is a level, not a signal that can run
        #  away. A negative or unbounded gain would silently invert the
        #  balance it is supposed to be tilting.
        state.acetylcholine = min(max(state.acetylcholine, 0.0), 2.0)
        state.norepinephrine = min(max(state.norepinephrine, 0.0), 2.0)
        state.serotonin = min(max(state.serotonin, 0.0), 2.0)
        return state

    @staticmethod
    def feedforward_share(acetylcholine: float) -> tuple[float, float]:
        """(feedforward weight, feedback weight) at this ACh level.

        The two are complementary by construction rather than by two separate
        knobs that could drift apart. High ACh weights the evidence arriving,
        low ACh weights what cortex already expected -- perceiving versus
        remembering, on one dial.
        """
        tilt = acetylcholine / 2.0                    # 0..1
        forward = config.ACH_MIN_SHARE + (1.0 - 2.0 * config.ACH_MIN_SHARE) * tilt
        return forward, 1.0 - forward


class Oscillations(nn.Module):
    """Theta, alpha and gamma as a per-lobe, per-timestep gain.

    gain = 1 + A*sin(2*pi*f*t + phase)

    Each lobe gets its own phase, so the bands do not merely turn the whole
    cortex up and down together -- which is the difference between a rhythm
    that can bind and separate areas and one that is just noise with a period.
    """

    BANDS = ("theta", "alpha", "gamma")

    def __init__(self, lobes: tuple[str, ...]):
        super().__init__()
        self.lobes = tuple(lobes)
        #  Evenly spread around the circle: no two lobes peak together.
        self.phases = {name: 2.0 * math.pi * i / len(self.lobes)
                       for i, name in enumerate(self.lobes)}

    def gain(self, lobe: str, t: int, band: str = "alpha") -> float:
        frequency = config.OSCILLATION_HZ[band]
        amplitude = config.OSCILLATION_AMPLITUDE[band]
        #  `t` is a settling step, not a second. Timesteps are given a nominal
        #  duration so that "8 Hz" means something rather than being a bare
        #  number multiplied by an index.
        seconds = t * config.TIMESTEP_SECONDS
        angle = 2.0 * math.pi * frequency * seconds + self.phases[lobe]
        return 1.0 + amplitude * math.sin(angle)

    def all_gains(self, t: int, band: str = "alpha") -> dict[str, float]:
        return {lobe: self.gain(lobe, t, band) for lobe in self.lobes}

    def report(self, t: int) -> str:
        rows = []
        for band in self.BANDS:
            values = self.all_gains(t, band)
            rows.append("%-6s %s" % (band, "  ".join(
                "%s %.3f" % (n[:4], v) for n, v in values.items())))
        return "\n".join(rows)


__all__ = ["Neuromodulators", "ModulatorState", "Oscillations"]
