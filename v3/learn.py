"""
learn.py -- teaching v3 to recognise things, without backpropagation.

HOW THIS DIFFERS FROM train_v2.py, which it replaces.

v2 computed a cross-entropy loss at the readout and transported gradients
backward through every tract and all twelve timesteps. Nothing here does that.
The entire learning signal is one scalar -- dopamine -- released after the
brain has already acted, reaching every synapse in the system at once and
carrying no information whatsoever about which synapse was responsible.

The eligibility traces are what make that work. During the percept, every
synapse quietly records whether it participated: pre fired shortly before
post, or it did not. When the dopamine arrives it multiplies those tags. A
synapse that helped cause the action that got rewarded is strengthened; one
that was silent is untouched, because zero times anything is zero.

That is the three-factor rule, and it is the best credit assignment biology
has. It is much worse than backpropagation, and the accuracy will show that.

WHAT COUNTS AS AN ANSWER. Not an argmax over logits -- there are no logits.
The brain acts when a motor population sustains its firing rate above
threshold for long enough, and if none does, it has not decided and emits
nothing. "No decision" is a real outcome here and it is reported separately
from a wrong one, because they are different failures.
"""

from __future__ import annotations

import argparse
import time

import torch

from v3 import config_v3 as cfg
from v3.brain_v3 import BrainV3
from v3.regions import Thalamus
from v3.transduce import features_to_rates, poisson_spikes


class Nigra:
    """Reward prediction error, kept outside the synapses that use it.

    Dopamine reports SURPRISE, not reward: something better than expected.
    A reward you fully predicted releases nothing, which is why an animal
    that has mastered a task stops learning it. Tracking a running
    expectation is the whole mechanism.
    """

    def __init__(self, rate: float = 0.05):
        self.expectation = 0.0
        self.rate = rate

    def dopamine(self, reward: float) -> float:
        error = reward - self.expectation
        self.expectation += self.rate * error
        return error


def present(brain: BrainV3, features: torch.Tensor, steps: int,
            learn: bool, teach: torch.Tensor | None = None) -> dict:
    """Show one stimulus for `steps` ms and see what the brain does.

    When `teach` is given the correct response is driven for the FIRST part of
    the presentation and then released, so the last part measures what the
    brain does on its own. Teaching all the way through would only ever
    measure the teacher.
    """
    rates = features_to_rates(features, Thalamus.RELAY_SIZE)
    train = poisson_spikes(rates, steps)
    brain.reset(features.shape[0], features.device)

    #  The teacher must let go BEFORE the decision window opens, or the
    #  measurement is of the teacher and not of the brain. Measured with an
    #  overlapping window: 100% training accuracy and 0% held out, which is
    #  what reading your own answer key looks like.
    window = int(cfg.MOTOR_WINDOW_MS / cfg.DT_MS)
    latest = max(steps - window - cfg.TEACHER_GAP_STEPS, 0)
    guided = min(int(steps * cfg.TEACHER_FRACTION), latest) if teach is not None else 0
    motor = []
    for t in range(steps):
        brain.teach(teach if t < guided else None)
        out = brain.step({"LGN": train[t]}, learn=learn)
        motor.append(out["motor"])
    brain.teach(None)
    return brain.read_decision(torch.stack(motor))


def run(brain: BrainV3, tokens: torch.Tensor, labels: torch.Tensor,
        nigra: Nigra, steps: int, learn: bool,
        batch: int = 1) -> tuple[float, float]:
    """One pass. Returns (accuracy, fraction that produced any decision)."""
    correct = decided = total = 0
    for index in range(0, len(labels) - batch + 1, batch):
        chunk = tokens[index:index + batch].float()
        target = labels[index:index + batch]

        answer = present(brain, chunk, steps, learn=learn,
                         teach=target if learn else None)
        hit = (answer["action"] == target)
        made = answer["decided"]

        if learn:
            #  The taught action is correct by construction, so the
            #  association is confirmed rather than gambled on. Reward still
            #  varies with whether the brain got there itself once the
            #  teacher let go, which is what keeps it from simply memorising
            #  the teacher's drive.
            reward = torch.where(hit, 1.0, 0.5)
            brain.consolidate(nigra.dopamine(float(reward.mean())))

        correct += int((hit & made).sum())
        decided += int(made.sum())
        total += len(target)
    return correct / max(total, 1), decided / max(total, 1)


def load_task(classes: int, per_class: int, device: str):
    """Cached encoder features from v2, cut down to a manageable task.

    Reading exactly the same cached features v2 trained on means a comparison
    between them is about the brain and not about the eyes.
    """
    cache = cfg.PROJECT_ROOT / "data" / "v2_streams.pt"
    if not cache.exists():
        raise SystemExit("no cached streams at %s -- run v2's trainer once "
                         "to build them" % cache)
    streams = torch.load(cache, weights_only=False)
    tokens, labels = streams["image"]["train"]
    test_tokens, test_labels = streams["image"]["test"]

    def cut(t, l, n):
        keep = l < classes
        t, l = t[keep], l[keep]
        picked = []
        for c in range(classes):
            idx = (l == c).nonzero(as_tuple=True)[0][:n]
            picked.append(idx)
        order = torch.cat(picked)
        order = order[torch.randperm(len(order))]
        return t[order].to(device), l[order].to(device)

    return cut(tokens, labels, per_class), cut(test_tokens, test_labels, 50)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--classes", type=int, default=10)
    parser.add_argument("--per-class", type=int, default=40)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--steps", type=int, default=120,
                        help="milliseconds per stimulus")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--test-size", type=int, default=60,
                        help="held-out examples per evaluation; spiking "
                             "simulation is slow and this dominates runtime")
    parser.add_argument("--eval-every", type=int, default=5)
    parser.add_argument("--resume", action="store_true", default=True)
    args = parser.parse_args()

    torch.manual_seed(cfg.SEED)
    print("=" * 72)
    print("v3 LEARNING -- three-factor plasticity, no gradient")
    print("=" * 72)

    (train_x, train_y), (test_x, test_y) = load_task(
        args.classes, args.per_class, args.device)
    print("%d classes, %d training examples, %d held out"
          % (args.classes, len(train_y), len(test_y)))
    print("%d ms per stimulus, chance is %.1f%%"
          % (args.steps, 100.0 / args.classes))

    brain = BrainV3(n_actions=args.classes).to(args.device)
    developed = cfg.PROJECT_ROOT / "checkpoints" / "v3_developed.pt"
    if args.resume and developed.exists():
        state = torch.load(developed, map_location=args.device)
        #  The developed file was written with a different action count, so
        #  motor-sized tensors will not match. Load what does.
        own = brain.state_dict()
        usable = {k: v for k, v in state.items()
                  if k in own and own[k].shape == v.shape}
        own.update(usable)
        brain.load_state_dict(own)
        print("resumed development: %d of %d tensors carried over"
              % (len(usable), len(own)))

    nigra = Nigra()
    print()
    print("%-6s %10s %10s %10s %10s" % ("epoch", "train acc", "decided",
                                        "test acc", "test dec"))
    started = time.perf_counter()
    best = 0.0
    for epoch in range(args.epochs):
        acc, dec = run(brain, train_x, train_y, nigra, args.steps, learn=True)
        brain.homeostasis(speed=0.5)
        #  Exploration fades as competence grows.
        brain.exploration = max(brain.exploration * cfg.EXPLORATION_DECAY,
                                cfg.EXPLORATION_FLOOR)
        #  Evaluating every epoch costs more wall clock than training does at
        #  these sizes, so it is periodic.
        if epoch % args.eval_every == 0 or epoch == args.epochs - 1:
            #  Evaluated WITHOUT exploration: the test measures what the
            #  brain has learned, not what it stumbles into by chance.
            held = brain.exploration
            brain.exploration = 0.0
            t_acc, t_dec = run(brain, test_x[:args.test_size],
                               test_y[:args.test_size], nigra,
                               args.steps, learn=False)
            brain.exploration = held
            best = max(best, t_acc)
            print("%-6d %9.1f%% %9.1f%% %9.1f%% %9.1f%%"
                  % (epoch, 100 * acc, 100 * dec, 100 * t_acc, 100 * t_dec),
                  flush=True)
        else:
            print("%-6d %9.1f%% %9.1f%% %10s %10s"
                  % (epoch, 100 * acc, 100 * dec, "-", "-"), flush=True)
    print()
    print("best held-out: %.1f%%   chance: %.1f%%"
          % (100 * best, 100.0 / args.classes))

    print()
    print("%.0f s wall clock" % (time.perf_counter() - started))
    out = cfg.PROJECT_ROOT / "checkpoints" / "brain_v3.pt"
    torch.save({"state": brain.state_dict(), "classes": args.classes,
                "record": {"epochs": args.epochs, "classes": args.classes,
                           "examples": len(train_y)}}, out)
    print("saved -> %s" % out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
