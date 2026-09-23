"""
ablate.py -- cut each part out and measure what changes (spec 9, stage 9).

WHY THIS SCRIPT IS THE ONE THAT MATTERS.

Every other check in this project asks whether something is WIRED. Shapes flow,
gradients arrive, tracts deliver. None of that establishes that a module does
anything: a component can receive input, produce output, train happily, and be
functionally bypassed -- the rest of the system routing around it while it
contributes nothing to the answer. Gradient coverage cannot see that. Only
removing the part and watching what happens can.

So this sweeps the whole system, one lesion at a time:

    every named tract      severed individually (gain 0.0)
    every subsystem        switched off at the settle() call
    every lobe             all of its efferent tracts cut at once

and reports how far the settled cortical state moves in each case. A part whose
removal changes nothing is DECORATIVE, and the report says so in those words.
That is a finding worth printing loudly rather than a failure worth hiding: v1
shipped with several such parts, and nobody noticed because everything looked
connected.

WHAT THE NUMBERS MEAN. The delta is the distance between the settled state with
the part and without it, as a percentage of the intact state's norm. It is a
measure of INFLUENCE, not of usefulness -- a large delta means the system would
behave differently, not that it would behave worse. Establishing better or
worse needs a task and a metric, which is what benchmark/ is for.

    python ablate.py                the full sweep
    python ablate.py --tracts       only the white matter
    python ablate.py --modules      only the subsystems and lobes
    python ablate.py --threshold 5  flag anything moving less than 5%
"""

from __future__ import annotations

import argparse

import torch

import config
from cortex import Cortex

#  Subsystems that settle() can switch off, and what cutting each one means.
SUBSYSTEMS = {
    "thalamus": ("use_thalamus", "all gating; sensory passes straight through"),
    "TRN": ("use_trn", "thalamic inhibition, so nuclei stop competing"),
    "cortical feedback": ("use_feedback", "frontal's descending projections"),
    "basal ganglia": ("use_bg", "action selection; MD and VA/VL go silent"),
    "limbic": ("use_limbic", "amygdala, hippocampus, hypothalamus"),
    "cerebellum": ("use_cerebellum", "the forward model and its correction"),
    "neuromodulators": ("use_modulators", "ACh, NE, 5-HT; the FF/FB dial"),
    "oscillations": ("use_oscillations", "theta, alpha and gamma rhythms"),
}


#  Parts whose influence AT REST is genuinely small, each with the reason and
#  the gate that proves it works when asked to do something. Declared here so
#  the tool and the stage gate cannot disagree about what counts as expected.
#
#  A part not on this list that moves nothing is a finding, and both the sweep
#  and gate 13 fail on it.
EXPECTED_QUIET = {
    "tract: hyperdirect":
        "an emergency brake, measured while nothing needs stopping. Gate 9 "
        "shows raising it demonstrably closes the gate; at rest the direct "
        "pathway starts biased to +1.5 and drowns it out.",
    "system: neuromodulators":
        "a dial resting at its centre. The readout is initialised small so an "
        "untrained system sits at baseline, and baseline ACh is EXACTLY "
        "neutral by construction (gate 12 measures that drift at 0.00e+00). "
        "Gate 12 moves it instead: the feedforward share goes 35% -> 89%.",
}


def sensory_batch(seed: int = 0, batch: int = 4) -> dict[str, torch.Tensor]:
    torch.manual_seed(seed)
    return {f"thal.{n}": torch.randn(batch, config.LATENT_DIM)
            for n in ("LGN", "MGN", "VPL", "PULVINAR")}


class AblationSweep:
    """Runs every lesion and collects the deltas."""

    def __init__(self, cortex: Cortex | None = None, seeds: tuple[int, ...] = (0, 1, 2)):
        torch.manual_seed(config.SEED)
        self.cortex = cortex if cortex is not None else Cortex()
        self.cortex.eval()
        self.seeds = seeds
        self.results: dict[str, tuple[float, str]] = {}

    # -- helpers ---------------------------------------------------------
    def _settle(self, seed: int, **kwargs):
        with torch.no_grad():
            return self.cortex.settle(sensory_batch(seed), **kwargs)

    def _delta(self, lesioned, intact) -> float:
        """How far the settled state moved, relative to the intact state."""
        gap = sum(float((intact.final[l] - lesioned.final[l]).norm())
                  for l in self.cortex.LOBES)
        scale = sum(float(intact.final[l].norm()) for l in self.cortex.LOBES)
        return gap / scale if scale else 0.0

    def _mean_delta(self, note: str, run) -> float:
        """Average over seeds, because one percept is not evidence."""
        deltas = []
        for seed in self.seeds:
            intact = self._settle(seed)
            deltas.append(self._delta(run(seed), intact))
        return sum(deltas) / len(deltas)

    # -- the three sweeps -------------------------------------------------
    def sweep_tracts(self) -> None:
        for name in sorted(self.cortex.tracts.names()):
            tract = self.cortex.tracts[name]

            def run(seed, tract=tract):
                tract.lesion(True)
                try:
                    return self._settle(seed)
                finally:
                    tract.lesion(False)

            note = "%s -> %s, delay %d" % (tract.source, tract.target, tract.delay)
            self.results["tract: " + name] = (self._mean_delta(name, run), note)

    def sweep_subsystems(self) -> None:
        for label, (flag, note) in SUBSYSTEMS.items():
            def run(seed, flag=flag):
                return self._settle(seed, **{flag: False})

            self.results["system: " + label] = (self._mean_delta(label, run), note)

    def sweep_lobes(self) -> None:
        """Silence a whole lobe by cutting everything leaving it."""
        for lobe in self.cortex.LOBES:
            leaving = [self.cortex.tracts[n]
                       for n in self.cortex.tracts.names()
                       if self.cortex.tracts[n].source == lobe]

            def run(seed, leaving=leaving):
                for tract in leaving:
                    tract.lesion(True)
                try:
                    return self._settle(seed)
                finally:
                    for tract in leaving:
                        tract.lesion(False)

            note = "%d efferent tract(s) cut" % len(leaving)
            self.results["lobe: " + lobe] = (self._mean_delta(lobe, run), note)

    def run(self, tracts: bool = True, modules: bool = True) -> dict:
        if tracts:
            self.sweep_tracts()
        if modules:
            self.sweep_subsystems()
            self.sweep_lobes()
        return self.results

    # -- reporting --------------------------------------------------------
    def report(self, threshold: float = 0.01) -> list[str]:
        """Ranked table. Returns the names of anything below threshold."""
        print("%-34s %10s   %s" % ("what was removed", "delta", "what it is"))
        print("-" * 92)
        decorative = []
        for name, (delta, note) in sorted(self.results.items(),
                                          key=lambda kv: -kv[1][0]):
            if delta >= threshold:
                mark = ""
            elif name in EXPECTED_QUIET:
                mark = "   <-- quiet at rest, declared"
            else:
                mark = "   <-- DECORATIVE"
                decorative.append(name)
            print("%-34s %9.1f%%   %s%s" % (name, 100 * delta, note, mark))
        print("-" * 92)
        for name in EXPECTED_QUIET:
            if name in self.results and self.results[name][0] < threshold:
                print()
                print("%s is quiet at rest, and that is expected:" % name)
                print("   %s" % EXPECTED_QUIET[name])
        return decorative


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tracts", action="store_true", help="white matter only")
    parser.add_argument("--modules", action="store_true", help="subsystems only")
    parser.add_argument("--threshold", type=float, default=1.0,
                        help="percent below which a part is called decorative")
    parser.add_argument("--seeds", type=int, default=3)
    args = parser.parse_args()

    both = not (args.tracts or args.modules)
    print("=" * 92)
    print("ABLATION SWEEP -- what actually matters")
    print("=" * 92)
    print("Each row: that part removed, everything else intact, and how far the")
    print("settled cortical state moved. Averaged over %d percepts, because"
          % args.seeds)
    print("one is not evidence. A part that changes nothing is named decorative.")
    print()

    sweep = AblationSweep(seeds=tuple(range(args.seeds)))
    sweep.run(tracts=both or args.tracts, modules=both or args.modules)
    decorative = sweep.report(threshold=args.threshold / 100.0)

    if decorative:
        print()
        print("%d PART(S) CHANGE NOTHING when removed:" % len(decorative))
        for name in decorative:
            print("   - %s" % name)
        print()
        print("That is a real finding, not a crash. Either the part is")
        print("genuinely bypassed, or it is wired somewhere it cannot be felt.")
        return 1

    print()
    print("Every part changes the output when removed. Nothing is decorative.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
