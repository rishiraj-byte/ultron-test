"""
verify.py -- terminal proof that tensors flow correctly. No GUI.

PROJECT.md build order step 1 and step 2:

    1. verify.py, image path only. One image end to end, print tensor shapes
       at every stage. Prove the wiring.
    2. Confirm every shape is LATENT_DIM where it should be. This is where the
       frontal fusion fix gets validated or thrown out.

Usage
    python verify.py                     image path, synthetic test image
    python verify.py --image cat.jpg     image path, your image
    python verify.py --all               every modality, plus multimodal
    python verify.py --bench             time the forward pass
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

import config
from modules import diagram
from modules.activations import build_frame
from modules.pipeline import BrainAI, BrainTrace, build_brain

D = config.LATENT_DIM

PASS = "  ok  "
FAIL = " FAIL "

_failures: list[str] = []


def rule(title: str = "") -> None:
    if title:
        print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")
    else:
        print("-" * 78)


def check(label: str, actual, expected) -> bool:
    ok = actual == expected
    mark = PASS if ok else FAIL
    print(f"[{mark}] {label:<46} {str(actual):>12}  expect {expected}")
    if not ok:
        _failures.append(f"{label}: got {actual}, expected {expected}")
    return ok


def synthetic_image() -> "Image.Image":
    """A deterministic image with real structure -- a bright disc on a gradient.

    Uniform noise makes every heatmap look the same, which hides exactly the
    wiring bugs this script exists to catch.
    """
    from PIL import Image

    size = config.IMAGE_SIZE
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
    gradient = (xx / size) * 120.0
    disc = ((xx - size * 0.6) ** 2 + (yy - size * 0.4) ** 2) < (size * 0.22) ** 2
    canvas = np.stack(
        [
            gradient + disc * 130.0,
            gradient * 0.5 + disc * 90.0,
            (yy / size) * 90.0 + disc * 40.0,
        ],
        axis=-1,
    )
    return Image.fromarray(np.clip(canvas, 0, 255).astype(np.uint8))


# ---------------------------------------------------------------------------
# Stage-by-stage report
# ---------------------------------------------------------------------------


def print_trace(trace: BrainTrace) -> None:
    print(f"\n{'stage':<34} {'shape':>12}   note")
    rule()
    for stage in trace.stages:
        print(stage)
    rule()
    print(f"forward pass: {trace.elapsed_ms:.1f} ms")


def assert_latent_stages(trace: BrainTrace, label: str) -> None:
    """PROJECT.md step 2: confirm every shape is LATENT_DIM where it should be."""
    print(f"\n-- {label}: every inter-module tensor must be 1 x {D} --")
    for stage in trace.stages:
        if stage.name.startswith("output."):
            continue
        check(stage.name, tuple(stage.shape), (1, D))


# ---------------------------------------------------------------------------
# Structural checks that do not need a forward pass
# ---------------------------------------------------------------------------


def verify_structure(brain: BrainAI) -> None:
    rule("STRUCTURE")

    print("-- routing table (PROJECT.md 2.3): thalamus is a switchboard --")
    for modality, (nucleus, destination) in config.THALAMIC_ROUTES.items():
        print(f"[{PASS}] {modality:<10} -> {nucleus:<9} -> {destination}")
    # Section 2.3's rule, stated precisely: it is the VISUAL path that must not
    # shortcut into parietal. Somatosensory input reaches parietal directly via
    # VPL in a real brain, and now does here too.
    vision_dest = {d for m, (_, d) in config.THALAMIC_ROUTES.items() if m == "image"}
    check("image never routes straight to parietal", "parietal" in vision_dest, False)
    check(
        "parietal still fed by occipital",
        ("occipital", "parietal") in config.CORTICAL_EDGES,
        True,
    )
    somato = [m for m, (_, d) in config.THALAMIC_ROUTES.items() if d == "parietal"]
    print(f"         somatosensory routes into parietal: {somato or 'none'}")
    check(
        "occipital ALSO feeds frontal directly",
        ("occipital", "frontal") in config.CORTICAL_EDGES,
        True,
    )

    print("\n-- frozen backbones must not carry gradients (PROJECT.md 5) --")
    vision = brain.encoders.image
    frozen = [p.requires_grad for p in vision.backbone.parameters()]
    check("resnet18 backbone params trainable", any(frozen), False)
    check(
        "vision projection trainable",
        all(p.requires_grad for p in vision.projection.parameters()),
        True,
    )

    print("\n-- frontal fusion geometry (PROJECT.md 4) --")
    frontal = brain.cortex.frontal
    check("fusion slots", frontal.slots, config.FRONTAL_STREAM_SLOTS)
    check("fusion mode", frontal.mode, config.FRONTAL_FUSION_MODE)
    if frontal.mode == "concat":
        check("fusion in_features", frontal.fuse.in_features, config.MAX_STREAMS * D)
        check("fusion out_features", frontal.fuse.out_features, D)

    print("\n-- trainable parameters (yours, not borrowed) --")
    counts = brain.parameter_count()
    for name, n in counts.items():
        print(f"        {name:<20} {n:>10,}")
    print(f"        {'TOTAL':<20} {sum(counts.values()):>10,}")


def verify_frontal_padding(brain: BrainAI) -> None:
    """The fusion fix, exercised directly: 1, 2, and 3 streams must all work
    and must all produce exactly LATENT_DIM out."""
    rule("FRONTAL FUSION -- variable stream count (PROJECT.md 4)")
    frontal = brain.cortex.frontal
    one = torch.randn(1, D, device=brain.device_name)

    cases = {
        "temporal only (text path)": {"temporal": one},
        "occipital + parietal (image path)": {"occipital": one, "parietal": one},
        "all three (image + text)": {
            "occipital": one, "parietal": one, "temporal": one
        },
    }
    for label, streams in cases.items():
        out = frontal(streams)
        check(f"{label}", tuple(out.shape), (1, D))
        filled = frontal.last_active_slots
        padded = [s for s, on in filled.items() if not on]
        print(f"         zero-padded slots: {padded or 'none'}")

    print("\n-- slot order must be positional, not arrival order --")
    a, b = torch.randn(1, D), torch.randn(1, D)
    forward = frontal.fuse_streams({"occipital": a, "temporal": b})
    reverse = frontal.fuse_streams({"temporal": b, "occipital": a})
    check("dict order changes nothing", torch.allclose(forward, reverse), True)

    swapped = frontal.fuse_streams({"occipital": b, "temporal": a})
    check("slot identity actually matters", torch.allclose(forward, swapped), False)


def verify_bad_input(brain: BrainAI) -> None:
    """Wrong shapes and wrong modalities must fail loudly, not silently."""
    rule("FAILURE MODES -- wrong input must raise")

    def raises(label: str, fn, expected) -> None:
        try:
            fn()
        except expected as exc:
            print(f"[{PASS}] {label:<46} {type(exc).__name__}")
            return
        except Exception as exc:  # noqa: BLE001
            print(f"[{FAIL}] {label:<46} raised {type(exc).__name__}")
            _failures.append(f"{label}: wrong exception {type(exc).__name__}")
            return
        print(f"[{FAIL}] {label:<46} did not raise")
        _failures.append(f"{label}: did not raise")

    raises(
        "unroutable modality",
        lambda: brain.thalamus("smell", torch.randn(1, D)),
        KeyError,
    )
    raises(
        "wrong latent width into thalamus",
        lambda: brain.thalamus("image", torch.randn(1, D + 1)),
        ValueError,
    )
    raises(
        "unknown frontal slot",
        lambda: brain.cortex.frontal({"cerebellum": torch.randn(1, D)}),
        KeyError,
    )
    raises(
        "wrong width into a lobe",
        lambda: brain.cortex.occipital(torch.randn(1, D * 2)),
        ValueError,
    )
    raises(
        "empty input to perceive",
        lambda: brain.perceive({}),
        ValueError,
    )


# ---------------------------------------------------------------------------
# The modality paths
# ---------------------------------------------------------------------------


def verify_image(brain: BrainAI, image_path: str | None) -> BrainTrace:
    rule("IMAGE PATH -- thalamus(LGN) -> occipital -> parietal -> frontal")

    if image_path:
        source = Path(image_path)
        if not source.exists():
            sys.exit(f"no such image: {source}")
        image = source
        print(f"input: {source}")
    else:
        image = synthetic_image()
        print("input: synthetic 224x224 test image (bright disc on a gradient)")

    trace = brain.perceive({"image": image}, publish=True)
    print_trace(trace)
    assert_latent_stages(trace, "image path")

    check("output logits", tuple(trace.logits.shape), (1, config.NUM_CLASSES))
    check(
        "frontal received two streams",
        sorted(trace.streams),
        ["occipital", "parietal"],
    )
    check(
        "occipital->frontal direct edge lit",
        ("occipital", "frontal") in trace.active_edges,
        True,
    )
    check("temporal slot zero-padded", "temporal" in trace.streams, False)

    print("\n-- answer --")
    for line in trace.answer.verbalise().splitlines():
        print(f"        {line}")

    if trace.answer.reference:
        print("\n-- frozen encoder reference (NOT the lobes' answer) --")
        for p in trace.answer.reference[:3]:
            print(f"        {p.label:<34} {p.probability:>7.2%}")
    return trace


def verify_text(brain: BrainAI) -> BrainTrace:
    rule("TEXT PATH -- thalamus(PULVINAR) -> temporal -> frontal")
    print("note: DELIBERATE DEVIATION, PROJECT.md 2.4 -- text has no thalamic")
    print("      relay in a real brain; reading enters through vision.")
    sentence = "a small striped cat sitting on a wooden table"
    print(f"\ninput: {sentence!r}")

    trace = brain.perceive({"text": sentence}, publish=True)
    print_trace(trace)
    assert_latent_stages(trace, "text path")
    check("frontal received one stream", sorted(trace.streams), ["temporal"])
    check("occipital slot zero-padded", "occipital" in trace.streams, False)
    return trace


def verify_audio(brain: BrainAI) -> BrainTrace | None:
    rule("AUDIO PATH -- thalamus(MGN) -> temporal -> frontal")
    duration = 2.0
    t = np.linspace(0, duration, int(config.AUDIO_SAMPLE_RATE * duration), endpoint=False)
    tone = (0.4 * np.sin(2 * np.pi * 440 * t) + 0.2 * np.sin(2 * np.pi * 880 * t))
    print(f"input: synthetic {duration:g}s 440 Hz + 880 Hz tone @ "
          f"{config.AUDIO_SAMPLE_RATE} Hz")

    try:
        trace = brain.perceive({"audio": tone.astype(np.float32)}, publish=True)
    except Exception as exc:  # noqa: BLE001
        print(f"[{FAIL}] audio path raised {type(exc).__name__}: {exc}")
        _failures.append(f"audio path: {exc}")
        return None

    print_trace(trace)
    assert_latent_stages(trace, "audio path")
    check("frontal received one stream", sorted(trace.streams), ["temporal"])
    return trace


def verify_multimodal(brain: BrainAI) -> BrainTrace:
    rule("MULTIMODAL -- image + text fill all three frontal slots")
    trace = brain.perceive(
        {"image": synthetic_image(), "text": "a bright circle on a gradient"},
        publish=True,
    )
    print_trace(trace)
    assert_latent_stages(trace, "multimodal")
    check(
        "all three slots filled",
        sorted(trace.streams),
        ["occipital", "parietal", "temporal"],
    )
    check("no zero padding", all(brain.cortex.frontal.last_active_slots.values()), True)
    return trace


# ---------------------------------------------------------------------------
# Activation capture
# ---------------------------------------------------------------------------


def verify_activations(brain: BrainAI, last: BrainTrace) -> None:
    rule("ACTIVATION CAPTURE (PROJECT.md 6)")

    frames = brain.bus.drain(limit=64)
    check("frames reached the queue", len(frames) > 0, True)
    if not frames:
        return

    frame = frames[-1]
    print(f"\nlast frame: seq={frame.sequence} modality={frame.modality} "
          f"layers={len(frame.heatmaps)}")
    print(f"\n{'layer':<28} {'grid':>11}   {'range 0..1':>12}   raw l2")
    rule()
    for name, grid in frame.heatmaps.items():
        stats = frame.stats[name]
        print(
            f"{name:<28} {grid.shape[0]:>4} x {grid.shape[1]:<4} "
            f"  {grid.min():>5.2f}..{grid.max():<5.2f}   {stats['l2']:.3f}"
        )

    conv = [n for n in frame.heatmaps if n.startswith("vision.")]
    check("conv taps captured", len(conv) > 0, True)
    check(
        "only conv taps stay spatial",
        all(n.startswith("vision.") for n in frame.heatmaps),
        True,
    )

    # A hook only fires if its lobe actually ran. On the image path temporal is
    # silent by design, so the captured set must equal the ACTIVE lobes -- not
    # all four. A mismatch here means a lobe ran that should not have, or one
    # that should have did not.
    fired = sorted({
        n.split(".")[1] for n in frame.vectors
        if n.startswith("lobe.") and n.split(".")[1] in config.LOBES
    })
    expected = sorted(l for l in config.LOBES if l in last.active_nodes)
    check("captured lobes == lobes that ran", fired, expected)
    silent = [l for l in config.LOBES if l not in expected]
    print(f"         silent this pass (correctly nothing drawn): {silent or 'none'}")

    for name, grid in frame.heatmaps.items():
        ok = 0.0 <= float(grid.min()) and float(grid.max()) <= 1.0
        if not ok:
            check(f"{name} normalised 0..1", False, True)

    print("\n-- queue must drop, never block (PROJECT.md 6.3) --")
    from modules.activations import ActivationBus, ActivationFrame

    bus = ActivationBus(maxsize=4)
    for i in range(20):
        bus.publish(ActivationFrame(sequence=i, modality="test", timestamp=0.0))
    check("queue capped at maxsize", len(bus), 4)
    check("overflow dropped, not blocked", bus.dropped, 16)
    remaining = [f.sequence for f in bus.drain(limit=16)]
    check("oldest dropped, newest kept", remaining, [16, 17, 18, 19])


def verify_layers(brain: BrainAI, last: BrainTrace) -> None:
    """The layer topology must match the real modules, and the diagram must
    draw every layer that actually ran."""
    rule("LAYER TOPOLOGY -- the real network structure")

    # Run a fresh pass and judge against THAT trace. verify_activations has
    # already drained the bus, and `last` may describe a different modality --
    # comparing this frame against that trace would be comparing two passes.
    last = brain.perceive({"image": synthetic_image()}, publish=True)
    frame = brain.bus.latest()
    if frame is None:
        check("an activation frame is available", False, True)
        return

    print(f"{'network':<16} {'layer chain (units)':<58}")
    rule()
    for name, layers in brain.topology.items():
        chain = " -> ".join(f"{l.size}" for l in layers)
        print(f"{name:<16} {chain:<58}")

    print("\n-- widths must match the live modules, not a hardcoded list --")
    check("lobe input width", brain.topology["occipital"][0].size, D)
    #  With a pretrained core the lobe has no hidden layer of its own: its
    #  operator IS a frozen network, and the width that matters is that
    #  network's, read off the live module rather than a constant. Asserting
    #  LOBE_HIDDEN_DIM here would be checking a config value against itself.
    occipital = brain.cortex["occipital"]
    core = getattr(occipital, "core", None)
    if core is None:
        check("lobe hidden width", brain.topology["occipital"][3].size,
              config.LOBE_HIDDEN_DIM)
    else:
        #  Read from the live core, not hardcoded. Naming the expected model
        #  here means this check has to be edited every time the core changes
        #  -- and it failed for exactly that reason when occipital moved from
        #  ResNet-18 to ResNet-50, reporting a successful upgrade as a defect.
        #  What matters is that a pretrained convolution IS there and its
        #  width comes from the model rather than from a constant.
        check("occipital operator is a pretrained convolution",
              "ResNet" in core.teacher_name and "layer4" in core.teacher_name,
              True)
        check("its width is the teacher's, not a constant",
              core.teacher_dim > config.LATENT_DIM, True)
        check("its weights are frozen",
              all(not p.requires_grad for p in core.teacher.parameters()), True)
    check("lobe output width", brain.topology["occipital"][-1].size, D)
    check("frontal input is the stream block",
          brain.topology["frontal"][0].size, config.FRONTAL_FUSION_IN_DIM)
    check("output layer is the label set",
          brain.topology["output"][-1].size, config.NUM_CLASSES)

    print("\n-- per-neuron vectors, not square reshapes --")
    for net in ("occipital", "frontal"):
        for spec in brain.topology[net]:
            values = frame.vectors.get(spec.key)
            if values is None:
                check(f"{net}: {spec.label} captured", False, True)
                continue
            check(f"{net}: {spec.label}", values.shape, (spec.size,))

    print("\n-- what the diagram reports (live activation counts) --")
    for net in ("occipital", "parietal", "frontal", "basal_ganglia", "output"):
        _, columns = diagram.render(brain.topology[net], frame.vectors, 400, 200)
        print(f"\n{net.upper()}")
        for line in diagram.summarise(columns):
            print("  " + line)

    print("\n-- a silent lobe must draw as EMPTY, not as stale data --")
    temporal_ran = "temporal" in last.active_nodes
    hidden = brain.topology["temporal"][3]
    captured = hidden.key in frame.vectors
    check("temporal captured only if it ran", captured, temporal_ran)

    print("\n-- firing counts only where a unit can genuinely be OFF --")
    for net in ("occipital", "frontal"):
        _, columns = diagram.render(brain.topology[net], frame.vectors, 400, 200)
        for column in columns:
            expect_count = column.spec.kind in ("gelu", "sigmoid")
            has_count = column.active is not None and column.total > 0
            if column.total == 0:
                continue
            check(f"{net}: {column.spec.label} counted", has_count, expect_count)



def verify_persistence(brain: BrainAI) -> None:
    rule("CHECKPOINT ROUND-TRIP")
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "roundtrip.pt"
        before = brain.perceive({"image": synthetic_image()}).logits.clone()
        brain.save(path)
        check("checkpoint written", path.exists(), True)

        fresh = BrainAI(device=brain.device_name)
        fresh.load_labels()
        check("loads into a fresh brain", fresh.load(path), True)
        after = fresh.perceive({"image": synthetic_image()}).logits
        check("identical logits after reload", torch.allclose(before, after, atol=1e-5), True)
        check("trained flag set by load", fresh.trained, True)


def bench(brain: BrainAI, n: int = 20) -> None:
    rule("BENCHMARK")
    image = synthetic_image()
    brain.perceive({"image": image})           # warm up
    start = time.perf_counter()
    for _ in range(n):
        brain.perceive({"image": image})
    elapsed = time.perf_counter() - start
    print(f"device: {brain.device_name}")
    print(f"{n} image passes in {elapsed:.2f}s "
          f"-> {elapsed / n * 1000:.1f} ms each, {n / elapsed:.1f} passes/sec")
    print("\nPROJECT.md 5 carried an UNVERIFIED ESTIMATE of 30-60 fwd/sec on a")
    print("modern laptop CPU. The number above is the measured value on this")
    print("machine, for the FULL path (encoder + lobes), not the encoder alone.")


# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", help="path to an image; omit for a synthetic one")
    parser.add_argument("--all", action="store_true", help="every modality")
    parser.add_argument("--bench", action="store_true", help="time the forward pass")
    parser.add_argument("--no-checkpoint", action="store_true",
                        help="ignore checkpoints/brain.pt")
    args = parser.parse_args()

    print("=" * 78)
    print("BRAIN-INSPIRED AI -- WIRING VERIFICATION")
    print("=" * 78)
    print(f"LATENT_DIM       {config.LATENT_DIM}")
    print(f"heatmap grid     {config.LATENT_GRID_SIDE} x {config.LATENT_GRID_SIDE}")
    print(f"frontal fusion   {config.MAX_STREAMS} x {config.LATENT_DIM} "
          f"-> {config.LATENT_DIM}  ({config.FRONTAL_FUSION_MODE})")
    print(f"label set        {config.LABEL_SET} ({config.NUM_CLASSES} classes)")
    print(f"device           {config.resolve_device()}")

    print("\nloading frozen backbones...")
    t0 = time.perf_counter()
    brain = build_brain(load_checkpoint=not args.no_checkpoint)
    brain.attach_recorder()
    brain.encoders.image.backbone  # force the vision load so timing is honest
    print(f"ready in {time.perf_counter() - t0:.1f}s  "
          f"(lobes {'TRAINED' if brain.trained else 'UNTRAINED'})")

    verify_structure(brain)
    verify_frontal_padding(brain)
    last = verify_image(brain, args.image)

    if args.all:
        verify_text(brain)
        verify_audio(brain)
        # multimodal runs last, so it is the frame verify_activations inspects
        last = verify_multimodal(brain)

    verify_activations(brain, last)
    verify_layers(brain, last)
    verify_bad_input(brain)
    verify_persistence(brain)

    if args.bench:
        bench(brain)

    rule("RESULT")
    if _failures:
        print(f"{len(_failures)} FAILURE(S):\n")
        for f in _failures:
            print(f"  - {f}")
        return 1

    print("All checks passed. Every inter-module tensor is 1 x "
          f"{config.LATENT_DIM}.")
    print("\nPROJECT.md step 2 verdict: the frontal fusion fix (section 4) HOLDS.")
    print("Frontal fuses n x LATENT_DIM -> LATENT_DIM before behaving like a")
    print("normal lobe, so occipital's direct delivery and parietal's arrive")
    print("together without a shape conflict.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
