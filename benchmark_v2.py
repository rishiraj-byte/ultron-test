"""
benchmark_v2.py -- is the brain architecture actually better than one MLP?

THE QUESTION THIS EXISTS TO ANSWER, AND WHY IT IS UNCOMFORTABLE.

v1 was benchmarked honestly and the answer was no: the modular architecture was
statistically indistinguishable from its own ablations, and slightly WORSE than
a parameter-matched monolithic MLP on single-modality tasks (CIFAR-100 66.5%
against 68.0%, p=0.049). That finding was never retracted and never retested.

v2 rebuilt the architecture so that every module is measurably load-bearing --
26 tracts, none decorative, every one changing the output when cut. That is a
stronger claim than v1 could make. It is NOT the same claim as being better at
anything, and the distinction matters: a system can be full of parts that all
demonstrably matter and still lose to a simpler one.

FAIRNESS, and the two ways v1 got it wrong before it was fixed:

  1. IDENTICAL INPUT. Both models get the same frozen encoder features. v1
     originally handed the monolith clean symbolic features while the modular
     side got lossy ResNet output, which flattered the monolith.
  2. PARAMETER-MATCHED ON THE COMPLETE MODEL. The monolith's hidden width is
     searched against the modular system's FULL trainable count -- projections,
     tracts, subcortex, heads, everything. v1 originally matched against a bare
     stack and handed the monolith an 18% capacity deficit.

BOTH ARE TRAINED TO CONVERGENCE, not for the same number of epochs, and that
distinction decides the result. Equal epochs favours whichever architecture
converges faster, and these differ enormously: the monolith reaches its best
held-out accuracy in about two seconds, the modular system takes minutes and
needs roughly twenty epochs before its image accuracy leaves the floor. Run for
six epochs each, the comparison read 41.2% against 2.9% -- which measures
convergence speed and says nothing about capacity. Both are therefore given a
budget long enough to plateau, and each is scored on its BEST held-out epoch.

The wall-clock difference is reported separately, because it is a real cost and
hiding it inside a fairness adjustment would be its own dishonesty.

Both get the same optimiser, the same schedule, the same budget, The monolith sees the pooled feature vector, which is the
input a normal network would be given; the modular system sees the un-pooled
sequence, which is the input its architecture exists to exploit. That is a real
asymmetry and it is the architecture's whole point -- reporting it as anything
else would be dishonest in the other direction.

    python benchmark_v2.py                 the full comparison
    python benchmark_v2.py --seeds 5       more repeats
    python benchmark_v2.py --epochs 4      shorter
"""

from __future__ import annotations

import argparse
import math
import statistics
import time

import torch
import torch.nn as nn

import config
from train_v2 import BrainV2, STREAM_CACHE, readout


class Monolith(nn.Module):
    """One MLP on the pooled features. The thing to beat.

    Deliberately given every advantage that does not amount to cheating: it
    sees the same frozen features, it is parameter-matched, and it has a
    per-task head exactly as the modular system does, so no part of the gap
    can be blamed on the modular side having separate read-outs.
    """

    def __init__(self, classes: dict[str, int], hidden: int):
        super().__init__()
        self.trunk = nn.ModuleDict({
            name: nn.Sequential(nn.Linear(raw, hidden), nn.LayerNorm(hidden),
                                nn.GELU())
            for name, raw in config.RAW_DIMS.items()
        })
        self.body = nn.Sequential(
            nn.Linear(hidden, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(),
            nn.LayerNorm(hidden))
        self.heads = nn.ModuleDict({
            name: nn.Linear(hidden, count) for name, count in classes.items()
        })

    def forward(self, modality: str, tokens: torch.Tensor) -> torch.Tensor:
        #  Pooled, because that is the input a normal network is given.
        pooled = self.trunk[modality](tokens.mean(dim=1))
        return self.heads[modality](self.body(pooled))


def match_parameters(classes: dict[str, int], target: int) -> int:
    """Widest hidden size whose COMPLETE trainable count stays under target."""
    best = 16
    for hidden in range(16, 4096, 16):
        count = sum(p.numel() for p in Monolith(classes, hidden).parameters()
                    if p.requires_grad)
        if count > target:
            break
        best = hidden
    return best


def load_streams():
    if not STREAM_CACHE.exists():
        raise SystemExit("no stream cache. run:  python train_v2.py --prepare-only")
    return torch.load(STREAM_CACHE, weights_only=False)


def epochs_of(model, streams, modalities, device, epochs, batch, lr,
              modular: bool):
    """Train to a fixed budget; return best held-out per task and when it came.

    `best_epoch` is what tells you whether the budget was long enough. If a
    model's best epoch is its last, it had not finished improving and the
    number understates it -- which is exactly the trap the first version of
    this benchmark fell into.
    """
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimiser = torch.optim.AdamW(trainable, lr=lr, weight_decay=0.01)
    best = {m: 0.0 for m in modalities}
    best_epoch = {m: 0 for m in modalities}


    order = []
    for modality in modalities:
        _, labels = streams[modality]["train"]
        for start in range(0, len(labels) - batch, batch):
            order.append((modality, start))

    #  BOTH sides get the cosine schedule. It was added for the modular
    #  system, where a fixed rate left accuracy oscillating instead of
    #  settling -- so giving it to only one architecture would be handing that
    #  one a tuning advantage and calling the difference architectural.
    steps_per_epoch = max(len(order), 1)
    total_steps = max(steps_per_epoch * epochs, 1)
    warmup = min(int(0.03 * total_steps), 300)

    def rate(step: int) -> float:
        if step < warmup:
            return (step + 1) / max(warmup, 1)
        progress = (step - warmup) / max(total_steps - warmup, 1)
        return 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))

    schedule = torch.optim.lr_scheduler.LambdaLR(optimiser, rate)

    for epoch in range(epochs):
        model.train()
        torch.manual_seed(epoch)
        shuffled = list(order)
        torch.randperm(len(shuffled))
        for modality, start in shuffled:
            tokens, labels = streams[modality]["train"]
            chunk = tokens[start:start + batch].float().to(device)
            target = labels[start:start + batch].to(device)
            if modular:
                trace, _ = model.settle(modality, chunk)
                logits = model.heads[modality](readout(trace))
            else:
                logits = model(modality, chunk)
            optimiser.zero_grad(set_to_none=True)
            nn.functional.cross_entropy(logits, target).backward()
            torch.nn.utils.clip_grad_norm_(trainable, config.GRAD_CLIP)
            optimiser.step()
            schedule.step()

        model.eval()
        for modality in modalities:
            tokens, labels = streams[modality]["test"]
            hits = total = 0
            with torch.no_grad():
                for start in range(0, len(labels) - batch, batch):
                    chunk = tokens[start:start + batch].float().to(device)
                    target = labels[start:start + batch].to(device)
                    if modular:
                        trace, _ = model.settle(modality, chunk)
                        logits = model.heads[modality](readout(trace))
                    else:
                        logits = model(modality, chunk)
                    hits += int((logits.argmax(dim=-1) == target).sum())
                    total += len(target)
            if total and hits / total > best[modality]:
                best[modality] = hits / total
                best_epoch[modality] = epoch
    return best, best_epoch


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=30,
                        help="a budget both sides can plateau in")
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--limit", type=int, default=2000,
                        help="examples per task, to keep the sweep affordable")
    #  Audio and sensor are PATHWAY TRAINING and are excluded by default.
    #  Audio is synthetic and trivially separable (a linear probe scores 100%),
    #  and the sensor task is not determined by its own input -- an 8-beam
    #  rangefinder reports walls, not the goal, and a linear probe reaches 27%
    #  against 25% chance. Putting either in a comparison would be reporting
    #  noise as a result for both architectures at once.
    parser.add_argument("--tasks", default="image,text",
                        help="comma-separated; audio and sensor are excluded "
                             "by default because neither is a benchmark")
    args = parser.parse_args()

    device = config.resolve_device()
    streams = load_streams()
    wanted = tuple(t.strip() for t in args.tasks.split(",") if t.strip())
    modalities = tuple(m for m in wanted
                       if m in streams and "train" in streams[m])
    classes = {m: int(streams[m]["train"][1].max()) + 1
               for m in ("image", "text", "audio", "sensor")
               if m in streams and "train" in streams[m]}

    #  Trim, so both sides see exactly the same examples.
    for modality in modalities:
        for split, cap in (("train", args.limit), ("test", args.limit // 2)):
            tokens, labels = streams[modality][split]
            streams[modality][split] = (tokens[:cap], labels[:cap])

    print("=" * 78)
    print("BENCHMARK -- brain architecture against one MLP")
    print("=" * 78)
    print("Same frozen features, same optimiser, same schedule, same seeds.")
    print("%d seeds x %d epochs, %d examples per task." % (
        args.seeds, args.epochs, args.limit))
    print()

    reference = BrainV2(classes)
    modular_count = sum(p.numel() for p in reference.parameters()
                        if p.requires_grad)
    hidden = match_parameters(classes, modular_count)
    monolith_count = sum(p.numel() for p in Monolith(classes, hidden).parameters()
                         if p.requires_grad)
    print("modular   %s trainable parameters" % format(modular_count, ","))
    print("monolith  %s trainable parameters (hidden %d) -- %.1f%% of modular"
          % (format(monolith_count, ","), hidden,
             100 * monolith_count / modular_count))
    print()

    results = {"modular": {m: [] for m in modalities},
               "monolith": {m: [] for m in modalities}}
    timing: dict[str, list] = {}

    for seed in range(args.seeds):
        for name, build, modular in (
                ("monolith", lambda: Monolith(classes, hidden), False),
                ("modular", lambda: BrainV2(classes), True)):
            torch.manual_seed(config.SEED + seed)
            model = build().to(device)
            started = time.perf_counter()
            with config.no_telemetry():
                best, best_epoch = epochs_of(
                    model, streams, modalities, device,
                    args.epochs, args.batch, args.lr, modular)
            elapsed = time.perf_counter() - started
            for modality in modalities:
                results[name][modality].append(best[modality])
            timing.setdefault(name, []).append(elapsed)
            print("  seed %d  %-9s %s   (%.0f s)"
                  % (seed, name,
                     "  ".join("%s %.1f%% @e%d" % (m, 100 * best[m],
                                                   best_epoch[m])
                               for m in modalities), elapsed))
            for modality in modalities:
                if best_epoch[modality] >= args.epochs - 1:
                    print("      WARNING: %s best was the LAST epoch -- %s had "
                          "not converged, so this understates it"
                          % (modality, name))
            del model
            if device == "cuda":
                torch.cuda.empty_cache()

    print()
    print("%-8s %10s %10s %10s   %s" % ("task", "chance", "monolith",
                                        "modular", "verdict"))
    print("-" * 78)
    for modality in modalities:
        mono = results["monolith"][modality]
        modu = results["modular"][modality]
        chance = 100.0 / classes[modality]
        gap = statistics.mean(modu) - statistics.mean(mono)
        spread = max(statistics.pstdev(mono), statistics.pstdev(modu), 1e-9)
        #  A difference smaller than the run-to-run spread is not a result.
        if abs(gap) < spread:
            verdict = "no difference (gap < spread)"
        elif gap > 0:
            verdict = "MODULAR better by %.1f pts" % (100 * gap)
        else:
            verdict = "monolith better by %.1f pts" % (-100 * gap)
        print("%-8s %9.1f%% %9.1f%% %9.1f%%   %s"
              % (modality, chance, 100 * statistics.mean(mono),
                 100 * statistics.mean(modu), verdict))
    print("-" * 78)
    print()
    print("wall clock per seed:  %s"
          % "   ".join("%s %.0f s" % (k, statistics.mean(v))
                       for k, v in sorted(timing.items())))
    print()
    print("Read this carefully. The modular system sees the UN-POOLED")
    print("sequence and the monolith sees the pooled vector, because that is")
    print("the input each architecture is built for. Where the modular side")
    print("wins, part of the credit belongs to having structure to exploit;")
    print("where it loses, no excuse applies -- it had more to work with.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
