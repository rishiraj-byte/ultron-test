"""
develop.py -- let the network find its own operating point.

WHAT THIS IS INSTEAD OF. v2 set its operating point by construction: a hard
top-5% mask selected the winners and zeroed everything else, and a divisive
ceiling clamped the magnitude. Both are mechanisms imposed from outside the
tissue, and neither has a biological implementation.

v3 has neither. Nothing sorts anything, nothing normalises anything. Cells
compete through inhibition, and two homeostatic mechanisms -- intrinsic
excitability and synaptic scaling -- move each population toward its own
target firing rate. This script is that process, run until it settles.

It is called development rather than training because no task is involved and
no reward is given. The network is only learning how excitable to be.
"""

from __future__ import annotations

import argparse
import time

import torch

from v3 import config_v3 as cfg
from v3.brain_v3 import BrainV3
from v3.regions import Thalamus
from v3.transduce import poisson_spikes


def healthy(record: dict, tolerance: float = 0.6) -> bool:
    """Is this epoch's state one worth keeping?

    WHY THIS CHECK EXISTS. Development converges in the MEAN but fluctuates
    hard epoch to epoch -- a spread of eight to eleven Hz around a five Hz
    average, because each epoch sees a different random stimulus. Saving
    whatever state the loop happened to stop on is therefore a lottery, and
    it was lost: a 160-epoch run whose last twenty epochs averaged 6.1 to 6.4
    Hz across all four lobes ended on an epoch reading 9.89, 0.01, 0.02, 0.00
    -- three lobes silent -- and THAT is the checkpoint that got written.

    Loading it, the console showed three of four lobes dead. The network was
    fine; the snapshot was not.
    """
    for lobe in ("occipital", "parietal", "temporal", "frontal"):
        rate = record.get(lobe, 0.0)
        if not (cfg.TARGET_RATE_HZ * tolerance <= rate
                <= cfg.TARGET_RATE_HZ / tolerance):
            return False
    return abs(record.get("GPi", 0.0) - cfg.GPI_TONIC_HZ) < cfg.GPI_TONIC_HZ * 0.5


def develop(brain: BrainV3, epochs: int, speed: float = 1.0,
            steps: int = 120, report_every: int = 10) -> list[dict]:
    history = []
    for epoch in range(epochs):
        rates = torch.rand(1, Thalamus.RELAY_SIZE) * cfg.INPUT_RATE_MAX_HZ
        train = poisson_spikes(rates, steps)
        brain.reset(1, "cpu")

        acc = {n: 0.0 for n in brain.LOBES}
        gpi = 0.0
        for t in range(steps):
            out = brain.step({"LGN": train[t]}, learn=False)
            for n in brain.LOBES:
                acc[n] += float(out["cortex"][n].mean())
            gpi += float(out["gpi"].mean())
        brain.homeostasis(speed=speed)

        k = 1000.0 / steps
        record = {n: acc[n] * k for n in brain.LOBES}
        record["GPi"] = gpi * k
        history.append(record)
        if epoch % report_every == 0 or epoch == epochs - 1:
            print("%-6d %7.2f %7.2f %7.2f %7.2f %8.2f" % (
                epoch, record["occipital"], record["parietal"],
                record["temporal"], record["frontal"], record["GPi"]),
                flush=True)
    return history


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument("--settle", type=int, default=60,
                        help="extra epochs allowed to reach a saveable state")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    torch.manual_seed(cfg.SEED)
    brain = BrainV3(n_actions=20)
    path = cfg.PROJECT_ROOT / "checkpoints" / "v3_developed.pt"
    if args.resume and path.exists():
        brain.load_state_dict(torch.load(path, map_location="cpu"))
        print("resumed from %s" % path)

    print("target: cortex %.0f Hz, GPi %.0f Hz" % (cfg.TARGET_RATE_HZ,
                                                   cfg.GPI_TONIC_HZ))
    print("%-6s %7s %7s %7s %7s %8s" % ("epoch", "occip", "pariet", "tempor",
                                        "frontal", "GPi"))
    started = time.perf_counter()
    history = develop(brain, args.epochs, speed=args.speed)

    #  KEEP DEVELOPING UNTIL THE NETWORK IS IN A STATE WORTH SAVING.
    #
    #  The mean being right is not enough -- the snapshot has to be right too.
    #  See `healthy` for the run that converged beautifully and then saved a
    #  moment where three lobes were silent.
    extra = 0
    while not healthy(history[-1]) and extra < args.settle:
        history += develop(brain, 1, speed=args.speed, report_every=999)
        extra += 1
    if healthy(history[-1]):
        print()
        print("settled into a healthy state after %d extra epoch(s)" % extra)
    else:
        print()
        print("WARNING: could not reach a healthy state in %d extra epochs."
              % args.settle)
        print("Saving anyway, but the checkpoint is a bad moment, not a bad"
              " network -- rerun development.")

    tail = history[-20:]
    print()
    print("LAST 20 EPOCHS, mean +/- spread:")
    for name in list(brain.LOBES) + ["GPi"]:
        values = [h[name] for h in tail]
        mean = sum(values) / len(values)
        spread = max(values) - min(values)
        target = cfg.GPI_TONIC_HZ if name == "GPi" else cfg.TARGET_RATE_HZ
        print("   %-10s %6.2f Hz  spread %5.2f  (target %.0f)"
              % (name, mean, spread, target))
    print()
    print("%.0f s wall clock" % (time.perf_counter() - started))
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(brain.state_dict(), path)
    print("saved -> %s" % path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
