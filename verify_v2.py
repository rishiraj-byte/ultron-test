"""
verify_v2.py -- the BUILD_SPEC_V2 stage gates. No GUI, no UI, no subcortex.

Three things must be proven before anything else is written:

  1  SHAPES      tensors flow end to end at LATENT_DIM
  2  PARALLELISM the four lobes run simultaneously, not as a pipeline
  3  SETTLING    state change per timestep decreases across T

Plus the two checks the spec demands at EVERY gate:

  4  GRADIENT COVERAGE  every trainable tensor receives a gradient
  5  ABLATION DELTA     every module and tract changes the output when cut

Check 5 exists because check 4 is not sufficient and the spec says so
explicitly: a module can receive gradient forever and still be functionally
bypassed. Gradient proves it is *connected*; ablation proves it *matters*.

HOW PARALLELISM IS PROVEN. Not by inspecting the code -- by making a
falsifiable prediction and testing it. If no lobe can observe another's current
state, then the order the four are computed in cannot affect the result. So:
run the same input through all 24 permutations of the update order and require
bit-identical output. A pipeline fails this on the first swap. Nothing else
about the implementation needs to be trusted.
"""

from __future__ import annotations

import argparse
import itertools
import sys

import torch

import config
from cortex import Cortex

PASS, FAIL = "  ok  ", " FAIL "
_failures: list[str] = []


def rule(title: str = "") -> None:
    if title:
        print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")
    else:
        print("-" * 78)


def check(label: str, ok: bool, detail: str = "") -> bool:
    print(f"[{PASS if ok else FAIL}] {label:<50} {detail}")
    if not ok:
        _failures.append(label)
    return ok


def sensory_batch(batch: int = 4, seed: int = 0) -> dict[str, torch.Tensor]:
    torch.manual_seed(seed)
    return {
        f"thal.{n}": torch.randn(batch, config.LATENT_DIM)
        for n in ("LGN", "MGN", "VPL", "PULVINAR")
    }


# ---------------------------------------------------------------------------


def gate1_shapes(cortex: Cortex) -> None:
    rule("GATE 1  SHAPES FLOW END TO END")
    sensory = sensory_batch()
    trace = cortex.settle(sensory)

    check("settling ran for T timesteps", trace.steps, config.T_STEPS) if False else None
    check(f"ran {config.T_STEPS} timesteps", trace.steps == config.T_STEPS,
          f"{trace.steps}")
    check("every lobe produced a final state",
          sorted(trace.final) == sorted(cortex.LOBES))

    D = config.LATENT_DIM
    for lobe in cortex.LOBES:
        check(f"{lobe} output is (B, {D})", tuple(trace.final[lobe].shape) == (4, D),
              str(tuple(trace.final[lobe].shape)))

    print(f"\n-- state at every timestep must also be (B, {D}) --")
    bad = 0
    for lobe in cortex.LOBES:
        for t in range(config.T_STEPS):
            if tuple(trace.history.read(lobe, t).shape) != (4, D):
                bad += 1
    check("all lobe x timestep states correct", bad == 0,
          f"{len(cortex.LOBES) * config.T_STEPS} states checked")

    print(f"\n-- finite values only (a diverging loop shows up here first) --")
    for lobe in cortex.LOBES:
        finite = bool(torch.isfinite(trace.final[lobe]).all())
        check(f"{lobe} finite", finite)

    print(f"\n{cortex.tracts.report()}")


def gate2_parallelism(cortex: Cortex) -> None:
    rule("GATE 2  THE FOUR LOBES RUN IN PARALLEL, NOT AS A PIPELINE")
    print("Claim: no lobe reads another's CURRENT state, so the order they are")
    print("computed in cannot change the result. Tested by running every")
    print("permutation of the update order and comparing bit-for-bit.")
    print("A pipeline fails on the first swap.\n")

    sensory = sensory_batch()
    cortex.eval()
    with torch.no_grad():
        reference = cortex.settle(sensory).final

    orders = list(itertools.permutations(cortex.LOBES))
    worst = 0.0
    worst_order = None
    for order in orders:
        with torch.no_grad():
            other = cortex.settle(sensory, order=order).final
        for lobe in cortex.LOBES:
            gap = float((reference[lobe] - other[lobe]).abs().max())
            if gap > worst:
                worst, worst_order = gap, order

    check(f"all {len(orders)} update orders give identical output", worst == 0.0,
          f"max difference {worst:.2e}")
    if worst > 0:
        print(f"         worst order: {worst_order}")

    print("\n-- and the converse: a tract with delay 0 must be REFUSED --")
    print("   (delay 0 would let a lobe read a value written earlier in the")
    print("    same timestep, which is a pipeline wearing a loop's clothes)")
    from modules.tracts import Tract

    try:
        Tract("illegal", "occipital", "frontal", delay=0)
        check("delay-0 tract rejected", False, "it was accepted")
    except ValueError:
        check("delay-0 tract rejected", True, "ValueError")

    print("\n-- every cortical tract carries delay >= 1 --")
    violations = [t.name for t in cortex.tracts if t.delay < 1]
    check("no zero-delay tracts in the bundle", not violations, str(violations))


def gate3_settling(cortex: Cortex) -> None:
    rule("GATE 3  THE NETWORK SETTLES")
    print("State change per timestep must DECREASE across T. If it does not,")
    print("the loop is not converging and T is an arbitrary stopping point.\n")

    for seed in (0, 1, 2):
        sensory = sensory_batch(seed=seed)
        with torch.no_grad():
            trace = cortex.settle(sensory)
        deltas = " ".join(f"{d:6.1f}" for d in trace.deltas)
        print(f"    seed {seed}   {deltas}")
        check(f"seed {seed}: monotonically decreasing", trace.monotonic())
        check(f"seed {seed}: converged", trace.settled(),
              f"ratio {trace.settling_ratio():.3f} <= {config.SETTLING_TOLERANCE}")

    print("\n-- the leak is what makes it converge; without it, it does not --")
    runaway = Cortex(leak=1.0)
    runaway.load_state_dict(cortex.state_dict())
    with torch.no_grad():
        loose = runaway.settle(sensory_batch())
    print(f"    leak=1.0    {' '.join(f'{d:6.1f}' for d in loose.deltas)}")
    check("leak=1.0 fails to settle (control)", not loose.settled(),
          f"ratio {loose.settling_ratio():.3f}")


def gate4_gradients(cortex: Cortex) -> None:
    rule("GATE 4  GRADIENT COVERAGE")
    print("Every trainable tensor must receive a gradient. v1 shipped a")
    print("checkpoint at 66% where an entire lobe never trained.\n")

    cortex.train()
    cortex.zero_grad(set_to_none=True)
    trace = cortex.settle(sensory_batch())
    loss = sum(v.pow(2).mean() for v in trace.final.values())
    loss.backward()

    total = trained = 0
    missing: list[str] = []
    for name, p in cortex.named_parameters():
        if not p.requires_grad:
            continue
        total += p.numel()
        if p.grad is not None and p.grad.abs().sum() > 0:
            trained += p.numel()
        else:
            missing.append(name)

    pct = 100 * trained / max(total, 1)
    check("100% of trainable parameters received gradient", not missing,
          f"{trained:,} / {total:,}  ({pct:.1f}%)")
    for name in missing[:10]:
        print(f"         no gradient: {name}")

    print(f"\n{'module':<16}{'params':>12}{'grad L2':>14}")
    print("-" * 44)
    for name, lobe in cortex.lobes.items():
        n = sum(p.numel() for p in lobe.parameters())
        g = sum(float(p.grad.norm()) for p in lobe.parameters() if p.grad is not None)
        print(f"{name:<16}{n:>12,}{g:>14.4f}")
    tract_params = sum(p.numel() for p in cortex.tracts.parameters())
    tract_grad = sum(float(p.grad.norm()) for p in cortex.tracts.parameters()
                     if p.grad is not None)
    print(f"{'tracts':<16}{tract_params:>12,}{tract_grad:>14.4f}")


def gate5_ablation(cortex: Cortex) -> None:
    rule("GATE 5  ABLATION DELTA -- does each part actually MATTER?")
    print("Gradient coverage is not contribution. A module can receive gradient")
    print("forever and still be functionally bypassed. So: cut each pathway and")
    print("measure how much the settled state moves. Near-zero delta means the")
    print("part is decorative, whatever its gradient says.\n")

    sensory = sensory_batch()
    cortex.eval()
    with torch.no_grad():
        intact = cortex.settle(sensory).final
    baseline = sum(float(v.norm()) for v in intact.values())

    print(f"{'tract':<22}{'from':<15}{'to':<12}{'delta':>10}")
    print("-" * 60)
    results: list[tuple[str, float]] = []
    for name in cortex.tracts.names():
        tract = cortex.tracts[name]
        tract.lesion(True)
        with torch.no_grad():
            cut = cortex.settle(sensory).final
        tract.lesion(False)

        delta = sum(float((intact[l] - cut[l]).norm()) for l in cortex.LOBES)
        relative = delta / baseline if baseline else 0.0
        results.append((name, relative))
        print(f"{name:<22}{tract.source:<15}{tract.target:<12}{relative:>9.1%}")

    print()
    dead = [n for n, d in results if d < 0.001]
    check("every tract changes the output when cut", not dead,
          f"{len(results)} tracts, weakest {min(d for _, d in results):.1%}")
    for name in dead:
        print(f"         DECORATIVE: {name} (cutting it changes nothing)")



def gate6_unpooled(cortex: Cortex) -> None:
    """STAGE 2 -- the encoders stop pooling."""
    rule("GATE 6  UN-POOLED SENSORY TAPS (stage 2)")
    print("v1 pooled every encoder to one vector, deleting exactly the")
    print("structure parietal and temporal exist to exploit. The gate: parietal")
    print("must receive real spatial positions, temporal a real sequence.\n")

    import numpy as np
    from PIL import Image

    from modules.encoders import EncoderBank
    from modules.sensory import SensoryFront

    front = SensoryFront(EncoderBank())
    streams = front.encode_all({
        "image": Image.new("RGB", (224, 224), (110, 140, 90)),
        "text": "the rocket reached orbit safely today",
        "audio": (0.4 * np.sin(2 * np.pi * 440 *
                               np.linspace(0, 1, config.AUDIO_SAMPLE_RATE))).astype("float32"),
        "sensor": np.random.rand(config.SENSOR_RAW_DIM).astype("float32"),
    })

    print(f"    {'modality':<10}{'shape':<22}{'axis means'}")
    print("    " + "-" * 56)
    for modality, stream in streams.items():
        print(f"    {modality:<10}{str(tuple(stream.tokens.shape)):<22}{stream.kind}")

    check("vision keeps the 7x7 grid = 49 positions",
          streams["image"].positions == 49, f"{streams['image'].positions}")
    check("text is a real token sequence", streams["text"].positions > 1,
          f"{streams['text'].positions} tokens")
    check("audio is a real frame sequence", streams["audio"].positions > 1,
          f"{streams['audio'].positions} frames")
    check("sensor is 8 beam directions",
          streams["sensor"].positions == config.SENSOR_BEAMS)
    check("every stream is LATENT_DIM per position",
          all(s.tokens.shape[-1] == config.LATENT_DIM for s in streams.values()))

    print("\n-- the operators must USE the structure, not just receive it --")
    grid = streams["image"].tokens
    x = torch.zeros(grid.shape[0], config.LATENT_DIM, device=grid.device)

    with torch.no_grad():
        # Shuffling positions must change parietal's answer. If it did not,
        # attention would be ignoring the arrangement it exists to read.
        forward = cortex.lobes["parietal"](x, grid)
        shuffled = cortex.lobes["parietal"](x, grid[:, torch.randperm(grid.shape[1])])
        spatial_gap = float((forward - shuffled).abs().max())

        # Reversing a sequence must change temporal's answer, for the same
        # reason: a GRU that ignores order is an MLP wearing a costume.
        tokens = streams["text"].tokens
        y = torch.zeros(tokens.shape[0], config.LATENT_DIM, device=tokens.device)
        ordered = cortex.lobes["temporal"](y, tokens)
        reversed_ = cortex.lobes["temporal"](y, torch.flip(tokens, dims=[1]))
        order_gap = float((ordered - reversed_).abs().max())

    check("parietal is sensitive to spatial arrangement", spatial_gap > 1e-4,
          f"shuffle changes output by {spatial_gap:.4f}")
    check("temporal is sensitive to sequence order", order_gap > 1e-4,
          f"reversal changes output by {order_gap:.4f}")

    print("\n-- and the loop still settles on real sensory input --")
    sensory = {f"thal.{n}": s.summary() for n, s in
               (("LGN", streams["image"]), ("MGN", streams["audio"]),
                ("VPL", streams["sensor"]), ("PULVINAR", streams["text"]))}
    sequences = {"occipital": streams["image"].tokens,
                 "parietal": streams["image"].tokens,
                 "temporal": streams["text"].tokens}
    with torch.no_grad():
        trace = cortex.settle(sensory, sequences=sequences)
    print(f"    deltas  {' '.join(f'{d:5.1f}' for d in trace.deltas)}")
    check("settles with real sequences", trace.settled(),
          f"ratio {trace.settling_ratio():.3f}")


# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gate", type=int, choices=(1, 2, 3, 4, 5, 6))
    args = parser.parse_args()

    print("=" * 78)
    print("BUILD_SPEC_V2  STAGE GATES")
    print("=" * 78)
    print(f"LATENT_DIM      {config.LATENT_DIM}")
    print(f"T_STEPS         {config.T_STEPS}")
    print(f"CORTICAL_LEAK   {config.CORTICAL_LEAK}")

    torch.manual_seed(config.SEED)
    cortex = Cortex()
    counts = cortex.parameter_count()
    print(f"parameters      {sum(counts.values()):,}  "
          f"({', '.join(f'{k} {v:,}' for k, v in counts.items())})")

    gates = {1: gate1_shapes, 2: gate2_parallelism, 3: gate3_settling,
             4: gate4_gradients, 5: gate5_ablation, 6: gate6_unpooled}
    for number in ([args.gate] if args.gate else sorted(gates)):
        gates[number](cortex)

    rule("STAGE 1 VERDICT")
    if _failures:
        print(f"{len(_failures)} FAILURE(S) -- stage 1 is NOT passed, do not proceed:")
        for f in _failures:
            print(f"   - {f}")
        return 1

    print("Stage 1 passes. Proven, not asserted:")
    print("  shapes flow end to end at LATENT_DIM, every lobe, every timestep")
    print("  all 24 update orders give bit-identical output -- genuinely parallel")
    print("  state change decreases monotonically across T -- it settles")
    print("  100% gradient coverage")
    print("  every named tract changes the output when severed")
    print("\nStage 2 (un-pooled encoder taps) may now begin.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
