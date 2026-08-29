"""
run.py -- the experiment driver. Produces every number in the paper.

    python -m benchmark.run --quick          smoke test, 2 seeds, few epochs
    python -m benchmark.run                  the full protocol
    python -m benchmark.run --tasks gridnav  one task only

WHAT IS COMPARED, and why each row exists:

    modular      routing + specialised lobes            the proposal
    no_routing   modalities broadcast to every lobe     isolates ROUTING
    homogeneous  four lobes, all the plain MLP block    isolates the per-lobe
                                                        INDUCTIVE BIASES
    shared_lobe  one lobe reused for every stream       isolates MODULARITY
    monolithic   one MLP over concatenated input        the conventional model

Every variant is parameter-matched to `modular` and trained under an identical
protocol. Seeds are paired, so `modular` vs each ablation is a paired t-test
rather than a comparison of two independent means.

LESIONS. Two different failures are measured and must not be confused:
    lobe lesions      damage to the network        ("occipital", ...)
    modality dropout  a sensor failing             ("modality:image", ...)
The first is the brain analogy; the second is the engineering case for an
embodied system whose camera can be occluded or whose rangefinder can fail.
"""

from __future__ import annotations

import argparse
import sys
import time

import numpy as np

from benchmark import data
from benchmark.baselines import ModularStack, count_params
from benchmark.harness import (
    RESULTS_DIR,
    RunResult,
    paired_test,
    run_once,
    save_results,
    summarise,
)

VARIANTS = ("modular", "no_routing", "homogeneous", "shared_lobe", "monolithic")

#  Which failures to probe, per task. Only lobes that a task actually drives
#  are worth lesioning -- lesioning temporal on a vision-only task measures
#  nothing.
LESIONS = {
    "gridnav": (
        "occipital", "parietal", "temporal",
        "modality:image", "modality:sensor", "modality:text",
    ),
    "cifar100": ("occipital", "parietal"),
    "news20": ("temporal",),
}


def reference_params(task) -> int:
    """The modular system's trainable count. Every baseline is matched to it."""
    from benchmark.harness import ModularModel

    return count_params(ModularModel(task, "modular"))


def banner(text: str) -> None:
    print(f"\n{'=' * 78}\n{text}\n{'=' * 78}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", nargs="+", default=["gridnav", "cifar100", "news20"])
    parser.add_argument("--variants", nargs="+", default=list(VARIANTS))
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--quick", action="store_true",
                        help="2 seeds, 6 epochs -- for checking the plumbing")
    parser.add_argument("--out", default="results")
    args = parser.parse_args()

    if args.quick:
        args.seeds, args.epochs = 2, 6

    print("=" * 78)
    print("MODULAR vs MONOLITHIC -- experimental protocol")
    print("=" * 78)
    print(f"tasks    {args.tasks}")
    print(f"variants {args.variants}")
    print(f"seeds    {args.seeds}   epochs {args.epochs}")

    results: list[RunResult] = []
    started = time.perf_counter()

    for task_name in args.tasks:
        cached = data.load(task_name)
        if cached is None:
            print(f"\n!! no feature cache for {task_name}; run:")
            print(f"     python -c \"from benchmark import data; data.get('{task_name}')\"")
            continue

        banner(f"TASK: {task_name}   "
               f"({cached.size('train')} train / {cached.size('test')} test, "
               f"{cached.num_classes} classes, modalities {cached.modalities})")

        target = reference_params(cached)
        print(f"parameter budget (from modular): {target:,}")

        lesions = tuple(
            l for l in LESIONS.get(task_name, ())
            if not l.startswith("modality:") or l.split(":", 1)[1] in cached.modalities
        )

        for variant in args.variants:
            accs = []
            for seed in range(args.seeds):
                result = run_once(
                    cached, variant, seed,
                    reference_params=target,
                    epochs=args.epochs,
                    # The monolithic model has no lobes to lesion, but a
                    # sensor can still fail on it -- and without that row there
                    # is nothing to compare graceful degradation against.
                    lesions=tuple(l for l in lesions if l.startswith("modality:"))
                    if variant == "monolithic" else lesions,
                )
                results.append(result)
                accs.append(result.test_accuracy)
                print(f"  {variant:<12} seed {seed}  "
                      f"test {result.test_accuracy:.4f}  "
                      f"val {result.val_accuracy:.4f}  "
                      f"{result.params:>9,} params  "
                      f"{result.train_seconds:5.1f}s")
            print(f"  {variant:<12} MEAN {np.mean(accs):.4f} "
                  f"+/- {np.std(accs, ddof=1) if len(accs) > 1 else 0:.4f}")

    if not results:
        print("\nno results -- build the feature caches first")
        return 1

    # ---- comparisons ------------------------------------------------
    banner("PAIRED COMPARISONS (modular vs each alternative)")
    summary = summarise(results)
    by_key: dict[tuple[str, str], list[float]] = {}
    for r in results:
        by_key.setdefault((r.task, r.variant), []).append(r.test_accuracy)

    print(f"{'task':<10} {'vs':<13} {'delta':>8} {'p':>9} {'d_z':>8}  verdict")
    print("-" * 78)
    for task_name in args.tasks:
        base = by_key.get((task_name, "modular"))
        if not base:
            continue
        for variant in args.variants:
            if variant == "modular":
                continue
            other = by_key.get((task_name, variant))
            if not other:
                continue
            stat = paired_test(base, other)
            verdict = "--"
            if stat["p"] == stat["p"]:      # not NaN
                if stat["p"] < 0.05:
                    verdict = "modular better" if stat["mean_diff"] > 0 else "WORSE"
                else:
                    verdict = "no difference"
            print(f"{task_name:<10} {variant:<13} {stat['mean_diff']:>+8.4f} "
                  f"{stat['p']:>9.4f} {stat['d']:>8.2f}  {verdict}")

    # ---- damage tolerance -------------------------------------------
    banner("DAMAGE TOLERANCE (accuracy retained under failure)")
    print(f"{'task':<10} {'variant':<13} {'failure':<20} {'acc':>8} {'retained':>10}")
    print("-" * 78)
    for key, entry in summary.items():
        if not entry["lesion"]:
            continue
        task_name, variant = key.split("/")
        intact = entry["test_mean"]
        for failure, stats in entry["lesion"].items():
            retained = stats["mean"] / intact if intact > 0 else float("nan")
            print(f"{task_name:<10} {variant:<13} {failure:<20} "
                  f"{stats['mean']:>8.4f} {retained:>9.1%}")

    # ---- compute ------------------------------------------------------
    banner("INFERENCE COST")
    print(f"{'task':<10} {'variant':<13} {'params':>11} {'MMACs/example':>15}")
    print("-" * 78)
    for key, entry in summary.items():
        task_name, variant = key.split("/")
        print(f"{task_name:<10} {variant:<13} {entry['params']:>11,} "
              f"{entry['macs'] / 1e6:>15.2f}")

    path = save_results(results, args.out)
    print(f"\ntotal wall time {time.perf_counter() - started:.1f}s")
    print(f"results -> {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
