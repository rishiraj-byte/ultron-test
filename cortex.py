"""
cortex.py -- the settling loop (BUILD_SPEC_V2 sections 2 and 9, stage 1).

WHAT REPLACED WHAT.

v1 ran a pipeline: occipital finished, then parietal ran on its output, then
frontal ran on both. One pass, one direction, and the call order WAS the
architecture. Nothing could feed back, nothing persisted between inputs, and
"parallel" was not even a meaningful word for it.

This runs the four lobes as four simultaneous branches for T timesteps. On
every timestep each lobe reads only `state[t - delay]` -- never any lobe's
current value -- and writes its own. That single rule is what makes them
genuinely parallel:

    no lobe can observe another's current state
      => no lobe waits on another within a timestep
      => the order they are written in cannot change the result

That last line is a falsifiable claim, and `verify.py` tests it by shuffling
the update order and demanding bit-identical output. A pipeline fails that test
immediately; this does not.

WHAT THE LOOP BUYS, beyond parallelism:

  * top-down feedback  -- frontal at t-1 reaches occipital at t, so priors
                          shape perception instead of only reading it
  * working memory     -- frontal reads its own previous state, so information
                          persists across timesteps rather than only forward
  * settling           -- the state stops moving as evidence is integrated,
                          which is measurable and is the stage-1 gate
  * lateral influence  -- parietal at t-1 reaches occipital at t

STAGE 1 SCOPE. Four lobes and their tracts only. No subcortex, no laminar
shells, no neuromodulators, no oscillations -- those are stages 3 to 8, and
adding them before the loop itself is proven would make attribution impossible
when something breaks.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import torch
import torch.nn as nn

import config
from modules.lobes import FrontalLobe, OccipitalLobe, ParietalLobe, TemporalLobe
from modules.thalamus_v2 import ThalamusV2
from modules.tracts import StateHistory, Tract, TractBundle

#  Cortical wiring for stage 1. Read as: (tract name, source, target).
#
#  Sources named "thal.*" are sensory drive arriving from a thalamic nucleus;
#  the rest are cortico-cortical association fibres. Delays come from
#  config.TRACT_DELAYS, so lengthening a pathway is a config change.
CORTICAL_TRACTS: tuple[tuple[str, str, str], ...] = (
    # sensory drive in
    ("optic_radiation",    "thal.LGN",      "occipital"),
    ("acoustic_radiation", "thal.MGN",      "temporal"),
    ("somato_radiation",   "thal.VPL",      "parietal"),
    ("pulvinar_temporal",  "thal.PULVINAR", "temporal"),
    # feedforward association
    ("ilf",                "occipital",     "temporal"),    # ventral "what"
    ("occ_par",            "occipital",     "parietal"),    # dorsal "where"
    ("ifof",               "occipital",     "frontal"),     # direct to decision
    ("slf",                "parietal",      "frontal"),     # dorsal to frontal
    ("arcuate",            "temporal",      "frontal"),     # language
    # feedback, frontal down onto sensory cortex
    ("fb_frontal_occ",     "frontal",       "occipital"),
    ("fb_frontal_par",     "frontal",       "parietal"),
    ("fb_frontal_tem",     "frontal",       "temporal"),
    # lateral
    ("lat_par_occ",        "parietal",      "occipital"),
    # working memory
    ("recurrent",          "frontal",       "frontal"),
)

#  Delays for the tracts stage 1 adds that config does not name yet. Kept here
#  rather than silently defaulting, so every delay in the system is a stated
#  number somewhere.
_EXTRA_DELAYS = {
    "pulvinar_temporal": 1,
    "occ_par": 1,
    "fb_frontal_occ": 2,     # feedback travels further than feedforward
    "fb_frontal_par": 1,
    "fb_frontal_tem": 2,
    "lat_par_occ": 1,
}


@dataclass
class SettleTrace:
    """Everything one settling run produced, for verification and the UI."""

    steps: int
    history: StateHistory
    #  timestep -> ||state[t] - state[t-1]|| summed over lobes
    deltas: list[float] = field(default_factory=list)
    #  lobe -> its state at the final timestep
    final: dict[str, torch.Tensor] = field(default_factory=dict)
    #  per timestep: nucleus -> gate opening, and nucleus -> TRN inhibition
    gates: list[dict[str, float]] = field(default_factory=list)
    inhibition: list[dict[str, float]] = field(default_factory=list)
    elapsed_ms: float = 0.0

    def measured(self) -> list[float]:
        """Deltas excluding the onset transient. See config.SETTLING_WARMUP."""
        return self.deltas[config.SETTLING_WARMUP:]

    def monotonic(self, slack: float = 1.02) -> bool:
        """Does state change DECREASE every timestep? The spec's actual words.

        Measured from the first CLOSED-LOOP step: at t=0 every tract delay is
        >= 1, so nothing recurrent has arrived and the lobes see sensory drive
        alone. That first jump is onset, not settling.

        `slack` allows a 2% wobble so floating-point noise on a nearly-flat
        tail does not fail an otherwise converging run.
        """
        d = self.measured()
        return len(d) >= 2 and all(d[i + 1] <= d[i] * slack for i in range(len(d) - 1))

    def settled(self, tolerance: float = config.SETTLING_TOLERANCE) -> bool:
        """Did the state stop moving? The stage-1 gate.

        Two conditions, and both matter. Monotonic decrease is the property the
        spec names. The ratio is a magnitude check on top -- a run could
        decrease by a hair each step and still not have converged to anything.
        A loop that fails either makes T meaningless: you would be running an
        RNN that never settles and stopping at an arbitrary point.
        """
        d = self.measured()
        if len(d) < 2 or d[0] <= 0:
            return False
        return self.monotonic() and (d[-1] / d[0]) <= tolerance

    def settling_ratio(self) -> float:
        d = self.measured()
        if len(d) < 2 or d[0] <= 0:
            return float("nan")
        return d[-1] / d[0]


class Cortex(nn.Module):
    """Four lobes, wired by named tracts, settling over T timesteps."""

    LOBES = ("occipital", "parietal", "temporal", "frontal")

    def __init__(self, dim: int = config.LATENT_DIM, dropout: float = 0.0,
                 leak: float = config.CORTICAL_LEAK):
        super().__init__()
        self.dim = dim
        self.leak = leak

        self.lobes = nn.ModuleDict({
            "occipital": OccipitalLobe(dim=dim, dropout=dropout),
            "parietal": ParietalLobe(dim=dim, dropout=dropout),
            "temporal": TemporalLobe(dim=dim, dropout=dropout),
            "frontal": FrontalLobe(dim=dim, dropout=dropout),
        })

        # Stage 3: the thalamus is INSIDE the loop, re-gated every timestep
        # under cortical feedback and TRN inhibition. In v1 it ran once, before
        # cortex, and could not be influenced by what cortex made of the input.
        self.thalamus = ThalamusV2(dim=dim)

        self.tracts = TractBundle()
        for name, source, target in CORTICAL_TRACTS:
            delay = config.TRACT_DELAYS.get(name, _EXTRA_DELAYS.get(name))
            if delay is None:
                raise ValueError(
                    f"tract {name!r} has no declared delay; add it to "
                    f"config.TRACT_DELAYS or _EXTRA_DELAYS rather than "
                    f"letting it default silently"
                )
            self.tracts.add(Tract(name, source, target, dim=dim, delay=delay))

    # ------------------------------------------------------------------
    def __getitem__(self, name: str) -> nn.Module:
        return self.lobes[name]

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    # ------------------------------------------------------------------
    def _drive(self, lobe: str, history: StateHistory, t: int) -> torch.Tensor:
        """Everything arriving at one lobe on timestep t.

        Every term is a named tract read at its own delay. There is no
        anonymous addition anywhere in this function -- that is the constraint
        that makes `ablate.py` able to cut a single pathway.
        """
        return self.tracts.deliver(lobe, history, t)

    def _step_lobe(self, lobe: str, history: StateHistory, t: int,
                   sequences: dict[str, torch.Tensor] | None = None) -> torch.Tensor:
        """Compute ONE lobe for timestep t. Reads only t-1 and earlier.

        Frontal is the exception in form but not in principle: it takes its
        three afferents as named positional slots rather than a single summed
        drive, because slot identity is what its fusion layer was built around
        (v1 section 4). Those slots are still tract outputs read at t-delay.
        """
        if lobe == "frontal":
            streams = {}
            for tract in self.tracts.into("frontal"):
                if tract.source in ("occipital", "parietal", "temporal"):
                    streams[tract.source] = tract(
                        history.read(tract.source, t - tract.delay)
                    )
            recurrent = self.tracts["recurrent"]
            carried = recurrent(history.read("frontal", t - recurrent.delay))
            if not streams:
                return self.lobes["frontal"](carried)
            fused = self.lobes["frontal"](streams)
            return fused + carried

        # Stage 2: the un-pooled sensory sequence reaches the lobe whose
        # operator was built to exploit it -- 49 real image positions to
        # parietal's attention, a real token or frame sequence to temporal's
        # GRU. Pooling this away was what made both operators inert in v1.
        drive = self._drive(lobe, history, t)
        sequence = (sequences or {}).get(lobe)
        return self.lobes[lobe](drive, sequence)

    # ------------------------------------------------------------------
    def settle(
        self,
        sensory: dict[str, torch.Tensor],
        steps: int = config.T_STEPS,
        order: tuple[str, ...] | None = None,
        sequences: dict[str, torch.Tensor] | None = None,
        use_thalamus: bool = True,
        use_trn: bool = True,
        use_feedback: bool = True,
    ) -> SettleTrace:
        """Run the loop. `sensory` maps "thal.<NUCLEUS>" -> (B, dim).

        `sequences` maps a LOBE name to its un-pooled sensory sequence
        (B, N, dim) -- 49 image positions for occipital and parietal, tokens or
        frames for temporal. Constant across the settling window: one percept
        is being integrated.

        `order` exists ONLY so verify.py can shuffle it and prove the result
        does not depend on it. Nothing in normal operation should pass it.
        """
        if not sensory:
            raise ValueError("settle() received no sensory input")

        probe = next(iter(sensory.values()))
        batch = probe.shape[0] if probe.dim() > 1 else 1
        history = StateHistory(dim=self.dim, batch=batch, device=probe.device)

        # Raw sensory, keyed by nucleus. Constant across the window: one
        # percept is being integrated, not a stream.
        raw = {name.replace("thal.", ""): value for name, value in sensory.items()}

        # Feedback ablation for the stage-3 gate: sever the descending tracts
        # rather than special-casing the loop, so the thing being measured is
        # the same object the rest of the system uses.
        feedback_tracts = ("fb_frontal_occ", "fb_frontal_par", "fb_frontal_tem")
        if not use_feedback:
            for name in feedback_tracts:
                self.tracts[name].lesion(True)

        lobes = tuple(order) if order else self.LOBES
        if set(lobes) != set(self.LOBES):
            raise ValueError(f"order must be a permutation of {self.LOBES}")

        trace = SettleTrace(steps=steps, history=history)
        started = time.perf_counter()

        for t in range(steps):
            # ---- THALAMUS FIRST, under cortical control from t-1 ----
            # Gated by what frontal made of the last timestep. This is the
            # corticothalamic loop: cortex is not merely reading the thalamus,
            # it is continuously re-tuning what the thalamus lets through.
            feedback = history.read("frontal", t - 1) if use_thalamus else None
            if use_thalamus:
                relayed = self.thalamus(raw, feedback, use_trn=use_trn)
                trace.gates.append(dict(relayed.gates))
                trace.inhibition.append(dict(relayed.inhibition))
                for nucleus, value in relayed.nuclei.items():
                    # Relay cells have membrane time constants too. Without
                    # this the corticothalamic loop oscillates instead of
                    # converging: measured with an instantaneous gate, the
                    # settling ratio went from 0.38 to 0.84 and the gate values
                    # swung back and forth across T rather than resolving.
                    # Same physics as the cortical leak, applied to the thalamus.
                    key = f"thal.{nucleus}"
                    previous = history.read(key, t - 1) if t > 0 else value
                    history.write(
                        key, t, (1.0 - self.leak) * previous + self.leak * value
                    )
            else:
                for nucleus, value in raw.items():
                    history.write(f"thal.{nucleus}", t, value)

            # ---- ALL FOUR LOBES, SAME TIMESTEP ----
            # Computed into a staging dict and only then committed. Nothing
            # written here is visible to anything else in this loop body, so
            # `lobes` may be in any order and the result is identical.
            fresh = {lobe: self._step_lobe(lobe, history, t, sequences)
                     for lobe in lobes}

            # Leaky integration toward the new drive, not replacement by it.
            # This is the membrane time constant, and it is what makes the loop
            # contract: without it the state is a fresh function of the inputs
            # every step and never converges. See config.CORTICAL_LEAK.
            computed = {}
            for lobe in self.LOBES:
                previous = history.read(lobe, t - 1)
                computed[lobe] = (1.0 - self.leak) * previous + self.leak * fresh[lobe]
            for lobe, value in computed.items():
                history.write(lobe, t, value)

            # How far the whole cortical state moved this step.
            delta = 0.0
            for lobe in self.LOBES:
                previous = history.read(lobe, t - 1)
                delta += float((computed[lobe] - previous).detach().norm())
            trace.deltas.append(delta)

        if not use_feedback:
            for name in feedback_tracts:
                self.tracts[name].lesion(False)

        trace.final = {lobe: history.read(lobe, steps - 1) for lobe in self.LOBES}
        trace.elapsed_ms = (time.perf_counter() - started) * 1000.0
        return trace

    # ------------------------------------------------------------------
    def parameter_count(self) -> dict[str, int]:
        counts = {name: sum(p.numel() for p in lobe.parameters() if p.requires_grad)
                  for name, lobe in self.lobes.items()}
        counts["tracts"] = sum(p.numel() for p in self.tracts.parameters()
                               if p.requires_grad)
        return counts


__all__ = ["Cortex", "SettleTrace", "CORTICAL_TRACTS"]
