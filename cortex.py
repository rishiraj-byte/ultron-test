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
from modules.basal_ganglia_v2 import BasalGangliaV2
from modules.laminar import k_winners
from modules.cerebellum import Cerebellum
from modules.neuromodulators import Neuromodulators, Oscillations
from modules.limbic import Amygdala, Hippocampus, Hypothalamus
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
    # subcortical, stage 5. The BG loop: frontal asks, GPi decides, the
    # thalamus delivers the answer back to frontal's thalamorecipient layer.
    ("corticostriatal",    "frontal",       "striatum"),
    ("hyperdirect",        "frontal",       "stn"),
    ("md_frontal",         "thal.MD",       "frontal"),
    ("vavl_frontal",       "thal.VA_VL",    "frontal"),
    # limbic, stage 6. The two amygdala roads differ ONLY in delay and in the
    # depth of what processes them, which is the whole distinction.
    ("low_road",           "thal.PULVINAR", "amygdala"),
    ("high_road",          "temporal",      "amygdala"),
    ("cortico_hippocampal", "temporal",     "hippocampus"),
    ("fornix",             "hippocampus",   "frontal"),
    ("hypothalamic",       "hypothalamus",  "frontal"),
    ("amygdala_cortex",    "amygdala",      "cortex"),
    # cerebellum, stage 7. An efference copy of frontal's decision goes out,
    # a correction comes back through VL -- the same nucleus the basal
    # ganglia address, which is how it is in a real thalamus.
    ("pontine",            "frontal",       "cerebellum"),
    ("dentate_vl",         "cerebellum",    "thal.VA_VL"),
)

#  Tracts whose target is a subcortical structure rather than a lobe. They are
#  real Tract objects with real delays and real lesion switches -- they simply
#  are not delivered by `_afferents`, which walks cortical targets.
SUBCORTICAL_TARGETS = ("striatum", "stn", "amygdala", "hippocampus",
                       "cortex", "cerebellum", "thal.VA_VL")

#  WHERE EACH TRACT LANDS (stage 4). Cortex does not sum its afferents into
#  one pile: the layer a signal ARRIVES at determines what happens to it, and
#  that is what lets a prediction be compared against evidence instead of
#  merely added to it. Declared per-tract and checked for completeness in
#  Cortex.__init__ for the same reason the delays are -- so no pathway can
#  acquire a destination by defaulting silently.
#
#      l4    thalamorecipient. Sensory drive, and nothing else.
#      l23   lateral and feedforward cortico-cortical input.
#      l56   feedback from frontal, and sustained working-memory activity.
TRACT_LAYER: dict[str, str] = {
    "optic_radiation":    "l4",
    "acoustic_radiation": "l4",
    "somato_radiation":   "l4",
    "pulvinar_temporal":  "l4",
    "ilf":                "l23",
    "occ_par":            "l23",
    "ifof":               "l23",
    "slf":                "l23",
    "arcuate":            "l23",
    "lat_par_occ":        "l23",
    "fb_frontal_occ":     "l56",
    "fb_frontal_par":     "l56",
    "fb_frontal_tem":     "l56",
    #  Frontal's own recurrence is sustained deep-layer activity, not a new
    #  afferent arriving from outside -- it lands where feedback lands.
    "recurrent":          "l56",
    #  The BG return path is thalamic drive, so it lands where thalamic drive
    #  lands. This is what finally gives frontal an L4 with something in it.
    "md_frontal":         "l4",
    "vavl_frontal":       "l4",
    #  These two leave cortex rather than arriving at it; named so the
    #  completeness check still covers them.
    "corticostriatal":    "efferent",
    "hyperdirect":        "efferent",
    #  Limbic. The subiculum's projection to frontal is association input, so
    #  it lands in L2/3; the hypothalamic drive is a slow contextual signal
    #  and lands deep, with feedback.
    "fornix":             "l23",
    "hypothalamic":       "l56",
    "low_road":           "efferent",
    "high_road":          "efferent",
    "cortico_hippocampal": "efferent",
    "amygdala_cortex":    "efferent",
    "pontine":            "efferent",
    "dentate_vl":         "efferent",
}

#  WHICH CORTICAL AREA ADDRESSES WHICH NUCLEUS (stage 4). The inverse of the
#  l4 tracts above: a nucleus that drives a lobe is written to by that same
#  lobe's layer 6. Corticothalamic synapses outnumber thalamocortical ones
#  by roughly an order of magnitude, so this is the larger half of the loop,
#  and routing all of it through frontal alone left three of the four L6
#  projections with no gradient at all.
CORTICOTHALAMIC: dict[str, str] = {
    "LGN":      "occipital",
    "MGN":      "temporal",
    "VPL":      "parietal",
    "PULVINAR": "temporal",
    #  MD and VA/VL are the basal-ganglia return path. Frontal addresses them,
    #  but nothing drives them until stage 5 builds the BG circuit.
    "MD":       "frontal",
    "VA_VL":    "frontal",
}

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
    #  fraction of L2/3 units allowed active at each timestep
    sparsity: list[float] = field(default_factory=list)
    #  per timestep: the basal ganglia decision that gated it (stage 5)
    bg: list = field(default_factory=list)
    #  per timestep: the amygdala evaluation, and hippocampal novelty (stage 6)
    amygdala: list = field(default_factory=list)
    novelty: list[float] = field(default_factory=list)
    #  per timestep: the cerebellum's prediction and how wrong the last was
    cerebellum: list = field(default_factory=list)
    #  per timestep: the four neuromodulator levels (stage 8)
    modulators: list = field(default_factory=list)
    elapsed_ms: float = 0.0

    def measured(self) -> list[float]:
        """Deltas excluding the onset transient. See config.SETTLING_WARMUP."""
        return self.deltas[config.SETTLING_WARMUP:]

    def converging(self, jitter: float = 1.0) -> bool:
        """Is the state change TRENDING DOWN, with jitter that stays bounded?

        This deliberately does not test step-to-step monotonicity, and the
        reason is structural rather than convenient. The cortical code is a
        hard k-winners mask, which is discrete: as the state drifts, two units
        near a tie swap places and the delta jumps by about one unit. So the
        loop converges to a NOISY fixed point -- a small neighbourhood it
        stays inside -- not to an exact one. Measured on real percepts the
        tail sits at 6-11% of the opening movement and wobbles by one to three
        units within that. Hysteresis was added to damp exactly this and does
        help, but it saturates: sweeping it over 4, 8, 16, 32 left the same
        single-step bumps, because the discreteness is the cause and no
        amount of incumbency advantage removes it.

        Insisting on strict monotonicity would therefore be measuring the
        quantisation noise, and passing would depend on the seed. What is
        actually claimed -- that the loop settles rather than merely stopping
        when T runs out -- is tested here as two conditions that a system
        which does not settle cannot satisfy:

            1. nothing after the first measured step ever climbs back above
               it, so the movement never returns to its opening scale;
            2. the second half of the tail is quieter than the first half,
               so the trend is down rather than flat or rising.

        `settled()` adds the third and strongest: the final step must be
        under SETTLING_TOLERANCE of the first. The leak=1.0 control in gate 3
        fails these, which is what keeps them honest.
        """
        d = self.measured()
        if len(d) < 4:
            return False
        if max(d[1:]) > d[0] * jitter:
            return False
        half = len(d) // 2
        early = sum(d[:half]) / half
        late = sum(d[half:]) / len(d[half:])
        return late < early

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
        return self.converging() and (d[-1] / d[0]) <= tolerance

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

        # Stage 5: the basal ganglia, with output going to the THALAMUS.
        # v1 hung this off the end as a terminal output gate, which made the
        # loop it exists to close invisible.
        self.basal_ganglia = BasalGangliaV2(dim=dim)

        # Stage 6: the limbic system. Three structures on three timescales --
        # milliseconds, one episode, and many episodes.
        self.amygdala = Amygdala(dim=dim)
        self.hippocampus = Hippocampus(dim=dim)
        self.hypothalamus = Hypothalamus(dim=dim)

        # Stage 7: the forward model. The first thing in this system that
        # ANTICIPATES rather than reacts.
        self.cerebellum = Cerebellum(dim=dim)

        # Stage 8: four global scalars and three rhythms. Neither carries
        # information -- they change how everything already present behaves.
        self.neuromodulators = Neuromodulators()
        self.oscillations = Oscillations(self.LOBES)

        self.tracts = TractBundle()
        for name, source, target in CORTICAL_TRACTS:
            delay = config.TRACT_DELAYS.get(name, _EXTRA_DELAYS.get(name))
            if delay is None:
                raise ValueError(
                    f"tract {name!r} has no declared delay; add it to "
                    f"config.TRACT_DELAYS or _EXTRA_DELAYS rather than "
                    f"letting it default silently"
                )
            if name not in TRACT_LAYER:
                raise ValueError(
                    f"tract {name!r} has no declared cortical layer; add it "
                    f"to TRACT_LAYER rather than letting it land nowhere"
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

    def _afferents(self, lobe: str, history: StateHistory, t: int,
                   acetylcholine: float | None = None
                   ) -> dict[str, torch.Tensor]:
        """Everything arriving at one lobe, GROUPED BY THE LAYER IT LANDS IN.

        Grouping is not anonymisation: every term is still a named tract
        evaluated at its own delay, and `lesion()` still cuts exactly one.
        What changes is that thalamic drive, lateral input and feedback no
        longer arrive as one indistinguishable sum.
        """
        groups: dict[str, torch.Tensor] = {}
        for tract in self.tracts.into(lobe):          # sorted by name
            signal = tract(history.read(tract.source, t - tract.delay))
            layer = TRACT_LAYER[tract.name]
            groups[layer] = signal if layer not in groups else groups[layer] + signal
        return self._balance(groups, acetylcholine)

    @staticmethod
    def _balance(groups: dict[str, torch.Tensor],
                 acetylcholine: float | None) -> dict[str, torch.Tensor]:
        """Tilt between arriving evidence and standing expectation (stage 8).

        This is what acetylcholine DOES, and it is stage 8's gate. Thalamic
        drive and lateral input are the evidence; L5/6 feedback is what cortex
        already expected. High ACh weights the first, low ACh the second --
        perceiving versus remembering, on one dial.

        The x2 keeps the total gain at 1.0 when the two shares are equal, so
        ACh at its baseline changes nothing at all. Without it, turning the
        dial would also quietly turn the whole cortex down.
        """
        thalamic, lateral, feedback = (groups.get("l4"), groups.get("l23"),
                                       groups.get("l56"))
        if acetylcholine is not None:
            forward, backward = Neuromodulators.feedforward_share(acetylcholine)
            forward, backward = forward * 2.0, backward * 2.0
            if thalamic is not None:
                thalamic = thalamic * forward
            if lateral is not None:
                lateral = lateral * forward
            if feedback is not None:
                feedback = feedback * backward
        return {"thalamic": thalamic, "lateral": lateral, "feedback": feedback}

    def _step_lobe(self, lobe: str, history: StateHistory, t: int,
                   sequences: dict[str, torch.Tensor] | None = None,
                   acetylcholine: float | None = None) -> torch.Tensor:
        """Compute ONE lobe for timestep t. Reads only t-1 and earlier.

        Frontal is the exception in form but not in principle: it takes its
        three afferents as named positional slots rather than a single summed
        drive, because slot identity is what its fusion layer was built around
        (v1 section 4). Those slots are still tract outputs read at t-delay.
        """
        if lobe == "frontal":
            # Frontal differs in ONE respect: its three association afferents
            # are kept as named slots rather than summed, because slot
            # identity is what its fusion layer was built around. Everything
            # else it receives is routed by layer exactly like any other lobe.
            #
            # This used to be a hand-written list of the sources frontal knew
            # about, and it silently dropped every tract added afterwards --
            # the fornix and the hypothalamic projection were both declared,
            # both lesionable, and both delivered nowhere. Deriving the split
            # from the tract table instead means a new tract cannot be
            # forgotten here; it can only be misrouted, which the layer
            # declaration makes visible.
            streams: dict[str, torch.Tensor] = {}
            other: dict[str, torch.Tensor] = {}
            for tract in self.tracts.into("frontal"):
                signal = tract(history.read(tract.source, t - tract.delay))
                if tract.source in ("occipital", "parietal", "temporal"):
                    streams[tract.source] = signal
                    continue
                layer = TRACT_LAYER[tract.name]
                other[layer] = (signal if layer not in other
                                else other[layer] + signal)

            shell = self.lobes["frontal"].laminar
            other = self._balance(other, acetylcholine)
            other = {"l4": other["thalamic"], "l23": other["lateral"],
                     "l56": other["feedback"]}
            #  Frontal reads the sensory sequence too, arriving at L4 like any
            #  other thalamic drive. It did not before, and the measurement
            #  that prompted this showed three of four lobes idle on a text
            #  input; see FrontalLobe.select.
            own = (sequences or {}).get("frontal")
            if own is not None:
                other["l4"] = (own if other.get("l4") is None
                               else other["l4"] + own)
            lateral = other.get("l23")
            if streams:
                fused = self.lobes["frontal"](streams)
                lateral = fused if lateral is None else fused + lateral
            return shell.receive(thalamic=other.get("l4"), lateral=lateral,
                                 feedback=other.get("l56"))

        afferents = self._afferents(lobe, history, t, acetylcholine)
        drive = self.lobes[lobe].laminar.receive(**afferents)
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
        use_bg: bool = True,
        use_limbic: bool = True,
        use_cerebellum: bool = True,
        use_modulators: bool = True,
        on_step=None,
        use_oscillations: bool = True,
        acetylcholine_override: float | None = None,
        reward: torch.Tensor | None = None,
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

        # A new percept is a new episode: the prediction left over from the
        # last one was about a world that has since been replaced.
        self.cerebellum.reset()

        # THE SENSORY SEQUENCE DOES NOT CHANGE WHILE CORTEX SETTLES, so each
        # lobe's reading of it is computed once here rather than re-derived
        # identically at all T timesteps. With pretrained cores in the lobes
        # that redundancy dominated the forward pass -- twelve ResNet and
        # transformer passes per lobe per percept instead of one. The value
        # is bit-identical either way, so nothing about the result changes.
        if sequences:
            #  GRADIENT CHECKPOINTING, and not as a micro-optimisation.
            #
            #  Every sense now reaches every lobe, so a percept runs FOUR
            #  pretrained sequence operators instead of one -- a ResNet stage,
            #  a DINOv2 block, a Whisper block and a Qwen block, each over the
            #  same 49 or 64 positions. Storing all four sets of activations
            #  for the backward pass quadrupled peak memory, and on a 6 GB
            #  card that turned a batch of 128 into CUBLAS_STATUS_EXECUTION_
            #  FAILED twenty batches in.
            #
            #  Checkpointing stores the inputs and recomputes the forward pass
            #  during backward instead. It costs roughly a third more time and
            #  gives back most of the memory, which is the right trade here:
            #  the alternative was a four-times-smaller batch, which costs far
            #  more than a third.
            #
            #  Only while training. At inference there is no backward pass, so
            #  there is nothing to trade and recomputing would be pure loss.
            def read(lobe: str, tokens):
                if tokens is None or tokens.dim() != 3:
                    return tokens
                operator = self.lobes[lobe].sequence_contribution
                if not (self.training and tokens.requires_grad):
                    return operator(tokens)
                from torch.utils.checkpoint import checkpoint
                return checkpoint(operator, tokens, use_reentrant=False)

            sequences = {lobe: read(lobe, tokens)
                         for lobe, tokens in sequences.items()}

        trace = SettleTrace(steps=steps, history=history)
        pending: list = []
        started = time.perf_counter()

        for t in range(steps):
            # ---- THALAMUS FIRST, under cortical control from t-1 ----
            # Gated by what frontal made of the last timestep. This is the
            # corticothalamic loop: cortex is not merely reading the thalamus,
            # it is continuously re-tuning what the thalamus lets through.
            # L6 IS the corticothalamic source. Feeding the thalamus raw
            # frontal state instead would make the loop an echo of cortical
            # activity rather than a deliberate control signal -- and it left
            # l6_to_thalamus without a gradient in every lobe, which is how
            # this was caught.
            # ---- LIMBIC, on three different clocks ----
            if use_limbic:
                lr, hr = self.tracts["low_road"], self.tracts["high_road"]
                low = lr(history.read("thal.PULVINAR", t - lr.delay))
                # Passed as None rather than as zeros when it has not arrived.
                # The high road CANNOT contribute before t = its delay, and
                # saying so explicitly is what makes the ordering claim
                # checkable instead of merely probable.
                high = (hr(history.read("temporal", t - hr.delay))
                        if t >= hr.delay else None)
                evaluation = self.amygdala(low, high)
                trace.amygdala.append(evaluation)
                # The modulation is written to history and collected by its
                # own named tract below, like every other projection. Applying
                # `evaluation.gain` directly here was simpler and wrong: the
                # amygdala_cortex tract existed, was lesionable, and carried
                # nothing, so cutting it would have changed precisely nothing
                # while appearing in the ablation table as a real pathway.
                history.write("amygdala", t, evaluation.gain - 1.0)

                ch = self.tracts["cortico_hippocampal"]
                recalled, novelty, _ = self.hippocampus(
                    ch(history.read("temporal", t - ch.delay))
                )
                history.write("hippocampus", t, recalled)
                if config.COLLECT_TELEMETRY:
                    trace.novelty.append(float(novelty.mean().detach()))

                # The homeostatic state is written into the loop every step,
                # but it CHANGES only between episodes -- that is what makes it
                # a condition rather than another observation.
                history.write("hypothalamus", t,
                              self.hypothalamus(batch=batch))

            # ---- BASAL GANGLIA, reading frontal at t-1 ----
            # Its output is thalamic drive for MD and VA/VL, NOT an output
            # gate. What it decides is what frontal receives on the next
            # timestep, which is the entire point of the circuit.
            if use_bg:
                cs = self.tracts["corticostriatal"]
                hd = self.tracts["hyperdirect"]
                asked = cs(history.read("frontal", t - cs.delay))
                fast = hd(history.read("frontal", t - hd.delay))
                decision = self.basal_ganglia(asked, reward=reward,
                                              hyperdirect_input=fast)
                trace.bg.append(decision)
                for nucleus, drive in decision.thalamic_drive.items():
                    raw[nucleus] = drive

            # ---- NEUROMODULATORS: four scalars, broadcast everywhere ----
            # Computed from the organism's CONDITION -- homeostatic state and
            # amygdala salience -- rather than from the percept. A modulator
            # driven by the content of what is being perceived would be an
            # information channel with a scalar's name.
            acetylcholine = None
            if use_modulators:
                salience = (trace.amygdala[-1].salience
                            if trace.amygdala else None)
                dopamine = trace.bg[-1].dopamine if trace.bg else None
                modulators = self.neuromodulators(
                    homeostasis=self.hypothalamus.state,
                    salience=salience, dopamine=dopamine,
                )
                trace.modulators.append(modulators)
                # An explicitly supplied level wins, so the gate can turn the
                # dial directly instead of trying to induce a level by
                # manipulating the hypothalamus into cooperating.
                acetylcholine = (modulators.acetylcholine
                                 if acetylcholine_override is None
                                 else acetylcholine_override)

            # ---- CEREBELLUM: predict what frontal is about to become ----
            if use_cerebellum:
                pon = self.tracts["pontine"]
                efference = pon(history.read("frontal", t - pon.delay))
                # The state that ACTUALLY arrived, against which the previous
                # prediction is scored. Never the state the prediction was
                # made from -- an identity function would score perfectly.
                arrived = history.read("frontal", t - 1) if t > 0 else None
                predicted = self.cerebellum(efference, arrived)
                trace.cerebellum.append(predicted)
                history.write("cerebellum", t, predicted.correction)

                dv = self.tracts["dentate_vl"]
                correction = dv(history.read("cerebellum", t - dv.delay))
                # VA/VL receives BOTH basal ganglia and cerebellar input, so
                # the two arrive summed rather than one overwriting the other.
                raw["VA_VL"] = (correction if "VA_VL" not in raw
                                else raw["VA_VL"] + correction)

            feedback = None
            trn_drive = None
            if use_thalamus:
                l6 = {lobe: self.lobes[lobe].laminar.corticothalamic(
                          history.read(lobe, t - 1))
                      for lobe in self.LOBES}
                feedback = {nucleus: l6[source]
                            for nucleus, source in CORTICOTHALAMIC.items()}
                # The TRN arbitrates BETWEEN nuclei, so it is driven by the
                # area that does the arbitrating.
                trn_drive = l6["frontal"]
            if use_thalamus:
                relayed = self.thalamus(raw, feedback, use_trn=use_trn,
                                        trn_drive=trn_drive)
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

            # Sharpen competition as the percept resolves. See laminar.l23.
            anneal_until = max(int(steps * config.SPARSITY_ANNEAL_FRACTION), 1)
            progress = min(t / anneal_until, 1.0)
            now = config.SPARSITY_START + (
                config.CORTICAL_SPARSITY - config.SPARSITY_START) * progress
            for lobe in self.LOBES:
                self.lobes[lobe].sparsity_now = now
            trace.sparsity.append(now)

            # ---- ALL FOUR LOBES, SAME TIMESTEP ----
            # Computed into a staging dict and only then committed. Nothing
            # written here is visible to anything else in this loop body, so
            # `lobes` may be in any order and the result is identical.
            fresh = {lobe: self._step_lobe(lobe, history, t, sequences,
                                           acetylcholine)
                     for lobe in lobes}

            # Leaky integration toward the new drive, not replacement by it.
            # This is the membrane time constant, and it is what makes the loop
            # contract: without it the state is a fresh function of the inputs
            # every step and never converges. See config.CORTICAL_LEAK.
            #
            # SPARSITY IS APPLIED HERE, ON THE COMMITTED STATE, and that
            # placement was a bug fix rather than a preference. Enforcing
            # k-WTA only inside L2/3 measured 5.1% at the layer and 100%
            # dense in the state -- because leaky integration mixes two
            # sparse vectors whose masks disagree, and the union fills in.
            # The tracts read the STATE, so the state is what the axons
            # carry, so the state is what has to be sparse. Measuring the
            # layer and reporting it as cortical sparsity would have been
            # true of a signal that is smeared one line later.
            # Amygdala gain, arriving over its own tract at its own delay.
            gain = None
            if use_limbic:
                ac = self.tracts["amygdala_cortex"]
                gain = 1.0 + ac(history.read("amygdala", t - ac.delay))

            computed = {}
            for lobe in self.LOBES:
                previous = history.read(lobe, t - 1)
                drive = fresh[lobe] if gain is None else fresh[lobe] * gain
                # Normalised to the SAME scale as the state it is about to be
                # mixed with. A membrane time constant averages a cell's
                # current activity with its incoming drive, which only means
                # anything if the two are commensurate; when the drive arrives
                # orders of magnitude larger, `(1-a)*prev + a*fresh` is just
                # `fresh` with extra steps and the leak stops smoothing
                # anything. That showed up as a settled state whose support
                # had locked while its values still swung by most of their
                # own magnitude every step.
                over = (drive.norm(dim=-1, keepdim=True)
                        / config.CORTICAL_STATE_NORM).clamp(min=1.0)
                drive = drive / over
                if use_oscillations:
                    # A per-lobe, per-timestep gain. The PHASE differs by lobe,
                    # so this is not one volume knob wobbling -- which areas are
                    # most excitable depends on when in the cycle you look.
                    drive = drive * self.oscillations.gain(lobe, t)

                integrated = (1.0 - self.leak) * previous + self.leak * drive
                # The incumbent winners get a head start, so the active set can
                # lock instead of trading places forever. See k_winners.
                winners = k_winners(integrated, now, incumbent=previous != 0)

                # DIVISIVE NORMALISATION. Nothing else in this loop bounds the
                # magnitude of the state, and the cortico-cortical loops have
                # gain above one: occipital and parietal are reciprocally
                # connected, so a percept driving BOTH of them (an image does)
                # amplifies about 2.4x per timestep. Measured over a 12-step
                # window the frontal state reached a norm of 567,699 against a
                # sensory drive of 13 -- and a hundred-way read-out off that
                # sat at chance while the identical features gave 56.8% under
                # a plain linear probe. The information was not lost in the
                # architecture; it was buried under four orders of magnitude.
                #
                # Real cortex has the same problem and the same answer: firing
                # rates saturate, and divisive normalisation is one of the
                # best-established cortical computations there is. Scaling to
                # a fixed norm leaves the PATTERN -- which units won, in what
                # proportion -- exactly as it was, and only fixes the volume.
                # A CEILING, not a fixed norm. Three versions of this were
                # measured and the first two each broke something:
                #
                #   no bound          the image path reached norm 567,699 and
                #                     a 100-way read-out sat at chance
                #   divide by norm    bounded, and learned well -- but a live
                #                     divisor projects away the gradient with
                #                     respect to MAGNITUDE, and the basal
                #                     ganglia act on magnitude. Striatal
                #                     gradient fell from 63 to 3.6e-8: dead.
                #   detached divisor  striatum alive, but the gradient then
                #                     disagrees with the forward pass about
                #                     what scaling does, and accuracy after
                #                     one epoch fell from 55% to 10%.
                #
                # Clamping only the excess keeps all three properties. Below
                # the ceiling this is the identity, so gradients are exact and
                # nothing is projected away; above it the state is rescaled
                # and the explosion is bounded. Real cortex saturates the same
                # way -- firing rates have a ceiling, not a fixed value.
                magnitude = winners.norm(dim=-1, keepdim=True)
                excess = (magnitude / config.CORTICAL_STATE_NORM).clamp(min=1.0)
                computed[lobe] = winners / excess
            for lobe, value in computed.items():
                history.write(lobe, t, value)

            # How far the whole cortical state moved this step.
            # Accumulated ON THE DEVICE and read back once at the end, not
            # four times per timestep. Same numbers; one synchronisation
            # instead of forty-eight.
            step_delta = sum((computed[lobe] - history.read(lobe, t - 1))
                             .detach().norm() for lobe in self.LOBES)
            pending.append(step_delta)

            # PER-TIMESTEP OBSERVATION POINT. Called after the state for this
            # step is committed, so anything watching sees a consistent
            # cortex rather than one lobe updated and three not. This is what
            # lets a display show the settling actually happening instead of
            # a single snapshot of where it ended up -- the loop runs twelve
            # times and every module fires twelve times, but a forward hook
            # only keeps the last one unless something reads it in between.
            if on_step is not None:
                on_step(t, history)

        if not use_feedback:
            for name in feedback_tracts:
                self.tracts[name].lesion(False)

        # One read-back for the whole window.
        trace.deltas = [float(d) for d in pending]

        trace.final = {lobe: history.read(lobe, steps - 1) for lobe in self.LOBES}

        # ---- END OF EPISODE ----
        # The hippocampus commits the episode and the hypothalamus integrates
        # it. Both happen ONCE per settle, not once per timestep, because both
        # are slower clocks than the settling loop.
        #
        # Gated on `self.training`, following the same convention as running
        # batch statistics: these are persistent state, and mutating them
        # during evaluation would make a measurement depend on how many times
        # it had already been taken. That is how a gate quietly stops being
        # reproducible.
        if use_limbic and self.training:
            self.hippocampus.remember(trace.final["temporal"])
            valence = (trace.amygdala[-1].valence if trace.amygdala else None)
            self.hypothalamus.update(trace.final["frontal"], valence)

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
