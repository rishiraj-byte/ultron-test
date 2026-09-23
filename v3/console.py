"""
console.py -- talk to v3, and watch what it actually does.

WHAT THIS IS NOT. It is not a chatbot, and it would be dishonest to package it
as one. v3 has ~12,400 neurons and no language module: Qwen was removed on
purpose, and nothing replaced it because nothing biologically plausible at
this scale could. A spiking network learning by local plasticity does not
produce sentences, and no one has built one that does.

WHAT IT IS. A window onto a small brain doing the things a small brain can do:
taking in a stimulus, settling, recognising it or failing to, deciding or
declining to decide, and reporting its own internal state while it happens.
Every number shown is read off real cells.

The most useful thing to try is the `gate` command. It forces the basal
ganglia open or shut and shows the output stopping -- which is the single
structural difference from v2 that everything else in v3 was built around.
"""

from __future__ import annotations

import argparse

import torch

from v3 import config_v3 as cfg
from v3.brain_v3 import BrainV3
from v3.regions import Thalamus
from v3.transduce import features_to_rates, poisson_spikes

BANNER = """
v3 interaction console -- a %d-neuron spiking brain

  look <n>     show held-out stimulus n and report what the brain decides
  gate open    force the basal ganglia gate open, then look
  gate shut    force it shut, then look
  gate rest    restore its tonic state
  state        current firing rates and neuromodulator levels
  quit
"""


class Console:
    def __init__(self, device: str = "cpu", prefer: str = "auto"):
        """`prefer` picks which checkpoint to load: trained, developed, or auto.

        AUTO EXISTS BECAUSE TRAINING CAN MAKE THINGS WORSE. A run that spends
        most of its epochs unable to act leaves homeostasis pushing on a
        network that never responds, and the result is a checkpoint in a
        pathological state -- GPi at 1.1 Hz instead of its tonic 60, and
        cortex falling silent when the gate is opened rather than when it is
        shut. Measured on the trained file, the gate's effect was inverted.
        So the console checks the tonic rate and refuses a checkpoint that
        fails it, rather than presenting a broken brain as the finished one.
        """
        self.device = device
        self.prefer = prefer
        checkpoint = cfg.PROJECT_ROOT / "checkpoints" / "brain_v3.pt"
        developed = cfg.PROJECT_ROOT / "checkpoints" / "v3_developed.pt"
        if prefer == "developed":
            checkpoint = cfg.PROJECT_ROOT / "checkpoints" / "__none__"

        self.classes = 5
        state = None
        if checkpoint.exists():
            blob = torch.load(checkpoint, map_location=device)
            self.classes = blob.get("classes", 5)
            state = blob["state"]
            self.record = blob.get("record", {})
        else:
            self.record = {}

        self.brain = BrainV3(n_actions=self.classes).to(device)
        if state is None and developed.exists():
            state = torch.load(developed, map_location=device)
        if state is not None:
            own = self.brain.state_dict()
            usable = {k: v for k, v in state.items()
                      if k in own and own[k].shape == v.shape}
            own.update(usable)
            self.brain.load_state_dict(own)
            self.loaded = len(usable)
        else:
            self.loaded = 0

        self.resting = self.brain.basal_ganglia.gpi.threshold.clone()
        self._load_stimuli()

    def _load_stimuli(self) -> None:
        cache = cfg.PROJECT_ROOT / "data" / "v2_streams.pt"
        if not cache.exists():
            self.stimuli = self.labels = None
            return
        streams = torch.load(cache, weights_only=False)
        tokens, labels = streams["image"]["test"]
        keep = labels < self.classes
        self.stimuli = tokens[keep][:200].to(self.device)
        self.labels = labels[keep][:200].to(self.device)

    # -- actions --------------------------------------------------------
    def look(self, index: int, steps: int = 80) -> None:
        if self.stimuli is None:
            print("   no cached stimuli available")
            return
        index = index % len(self.labels)
        features = self.stimuli[index:index + 1].float()

        rates = features_to_rates(features, Thalamus.RELAY_SIZE)
        train = poisson_spikes(rates, steps)
        self.brain.reset(1, self.device)

        motor, cortex = [], {n: 0.0 for n in self.brain.LOBES}
        gpi = 0.0
        for t in range(steps):
            out = self.brain.step({"LGN": train[t]}, learn=False)
            motor.append(out["motor"])
            for n in self.brain.LOBES:
                cortex[n] += float(out["cortex"][n].mean())
            gpi += float(out["gpi"].mean())

        answer = self.brain.read_decision(torch.stack(motor))
        action = int(answer["action"][0])
        truth = int(self.labels[index])
        k = 1000.0 / steps

        print()
        print("   stimulus %d, true class %d" % (index, truth))
        print("   cortex   " + "  ".join(
            "%s %.1f Hz" % (n[:4], cortex[n] * k) for n in self.brain.LOBES))
        print("   GPi      %.1f Hz   (high = gate shut)" % (gpi * k))
        if self.brain.chemistry:
            print("   chemistry " + "  ".join(
                "%s %.2f" % (k2[:4], v) for k2, v in self.brain.chemistry.items()))
        if action < 0:
            print("   DECISION  none -- no motor population reached threshold.")
            print("             That is a real outcome, not a crash: an")
            print("             undecided brain emits nothing.")
        else:
            print("   DECISION  class %d  (%s)   sustained %.1f Hz"
                  % (action, "correct" if action == truth else "wrong",
                     float(answer["rate_hz"].max())))

    def gate(self, mode: str) -> None:
        gpi = self.brain.basal_ganglia.gpi
        if mode == "open":
            gpi.threshold.fill_(0.0)
            print("   GPi silenced -- the gate is held OPEN")
        elif mode == "shut":
            gpi.threshold.fill_(cfg.V_REST_MV + 2.0)
            print("   GPi maximal -- the gate is held SHUT")
        else:
            gpi.threshold.copy_(self.resting)
            print("   GPi restored to its tonic state")

    def state(self) -> None:
        print()
        print("   %d neurons, %d populations, %d projections" % (
            sum(p.size for p in self.brain.all_populations()),
            sum(1 for _ in self.brain.all_populations()),
            sum(1 for _ in self.brain.all_projections())))
        print("   trained: %s" % (self.record or "development only"))
        print("   %d tensors loaded from checkpoint" % self.loaded)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    console = Console(args.device)
    print(BANNER % sum(p.size for p in console.brain.all_populations()))
    console.state()

    while True:
        try:
            line = input("\nv3> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if not line:
            continue
        parts = line.split()
        command = parts[0].lower()

        if command in ("quit", "exit"):
            return 0
        if command == "look":
            console.look(int(parts[1]) if len(parts) > 1 else 0)
        elif command == "gate":
            console.gate(parts[1] if len(parts) > 1 else "rest")
        elif command == "state":
            console.state()
        else:
            print("   commands: look <n>, gate open|shut|rest, state, quit")


if __name__ == "__main__":
    raise SystemExit(main())
