"""
explain.py -- evidence that each stage does what it claims. No GUI.

    python explain.py              run every check
    python explain.py --stage 2    just one

The dashboard shows activity, but activity is not proof. A system with random
weights also lights up. Each check below is written so that a BROKEN system
FAILS it -- if the encoders were returning noise, or a lobe were an identity
function, or training had not changed anything, the numbers here would say so.

  1  FEATURE EXTRACTION  do the frozen encoders carry meaning?
  2  ROUTING             does each modality reach the right lobe?
  3  THE LOBES           does each one transform its input, and differently?
  4  TRAINING            did learning actually change the networks?
  5  FUSION              does combining modalities change the answer?
  6  END TO END          the full trace for one input
"""

from __future__ import annotations

import argparse
import sys

import numpy as np
import torch

import config
from modules.pipeline import build_brain

OK, BAD = "  ok  ", " FAIL "
_fails: list[str] = []


def rule(title: str) -> None:
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


def check(label: str, passed: bool, detail: str = "") -> None:
    print(f"[{OK if passed else BAD}] {label:<48} {detail}")
    if not passed:
        _fails.append(label)


def cos(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(torch.nn.functional.cosine_similarity(
        a.flatten().unsqueeze(0), b.flatten().unsqueeze(0)
    ))


# ---------------------------------------------------------------------------


def stage1_features(brain) -> None:
    rule("1  FEATURE EXTRACTION -- do the frozen encoders carry meaning?")
    print("If an encoder were broken it would still return numbers. What it")
    print("could NOT do is place related things near each other and unrelated")
    print("things apart. That is what is measured.\n")

    text = brain.encoders.text
    groups = {
        "space": ["the rocket launched into orbit",
                  "the satellite orbits the earth"],
        "medicine": ["the doctor prescribed antibiotics",
                     "the patient recovered in hospital"],
    }
    vecs = {k: [text.encode_raw(s) for s in v] for k, v in groups.items()}

    within = (cos(vecs["space"][0], vecs["space"][1])
              + cos(vecs["medicine"][0], vecs["medicine"][1])) / 2
    between = (cos(vecs["space"][0], vecs["medicine"][0])
               + cos(vecs["space"][1], vecs["medicine"][1])) / 2
    print(f"    same topic, different words   cosine {within:+.3f}")
    print(f"    different topics              cosine {between:+.3f}")
    check("text encoder separates meaning", within > between + 0.2,
          f"gap {within - between:+.3f}")

    from PIL import Image
    solid = Image.new("RGB", (224, 224), (200, 40, 40))
    noise = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
    fa, fb = brain.encoders.image.encode_raw(solid), brain.encoders.image.encode_raw(noise)
    print(f"\n    flat colour vs random noise   cosine {cos(fa, fb):+.3f}")
    check("vision encoder distinguishes inputs", cos(fa, fb) < 0.9,
          "different images -> different features")
    check("vision features are not constant", float(fa.std()) > 1e-3,
          f"std {float(fa.std()):.4f}")


def stage2_routing(brain) -> None:
    rule("2  ROUTING -- does each modality reach the right lobe?")
    print("The thalamus is meant to be a switchboard. If it were a funnel, or")
    print("if routing were ignored, every modality would light the same lobes.\n")

    from PIL import Image
    probes = {
        "image": Image.new("RGB", (224, 224), (120, 160, 90)),
        "text": "a sentence about astronomy",
        "audio": (0.4 * np.sin(2 * np.pi * 440 *
                               np.linspace(0, 1, config.AUDIO_SAMPLE_RATE))).astype("float32"),
        "sensor": np.random.rand(config.SENSOR_RAW_DIM).astype("float32"),
    }
    print(f"    {'modality':<10} {'nucleus':<10} {'lobes that ran':<34} expected")
    print("    " + "-" * 70)
    expected = {"image": {"occipital", "parietal"}, "text": {"temporal"},
                "audio": {"temporal"}, "sensor": {"parietal"}}
    for modality, value in probes.items():
        trace = brain.perceive({modality: value})
        ran = {n for n in trace.active_nodes if n in config.LOBES} - {"frontal"}
        nucleus = trace.routings[0].nucleus
        good = ran == expected[modality]
        print(f"    {modality:<10} {nucleus:<10} {str(sorted(ran)):<34} "
              f"{sorted(expected[modality])}")
        check(f"{modality} routed correctly", good)


def stage3_lobes(brain) -> None:
    rule("3  THE LOBES -- does each one transform, and differently?")
    print("A lobe that returned its input unchanged, or that behaved the same")
    print("as its neighbours, would make the architecture decorative.\n")

    # Built on the brain's own device: these calls go straight into the lobes,
    # bypassing the pipeline that would normally move inputs for us.
    device = next(brain.cortex.parameters()).device
    x = torch.randn(8, config.LATENT_DIM, device=device)
    outs = {}
    for name in ("occipital", "parietal", "temporal"):
        lobe = brain.cortex[name]
        lobe.eval()
        with torch.no_grad():
            outs[name] = lobe(x)
        similarity = cos(outs[name], x)
        print(f"    {name:<10} cosine(out, in) {similarity:+.3f}   "
              f"{type(lobe).__name__}")
        check(f"{name} is not an identity function", similarity < 0.999)

    print()
    pairs = [("occipital", "parietal"), ("occipital", "temporal"),
             ("parietal", "temporal")]
    for a, b in pairs:
        similarity = cos(outs[a], outs[b])
        print(f"    {a:<10} vs {b:<10} cosine {similarity:+.3f}")
        check(f"{a} differs from {b}", similarity < 0.99)


def stage4_training(brain) -> None:
    rule("4  TRAINING -- did learning actually change the networks?")
    print("Compared against a freshly initialised system. If training had not")
    print("worked, the weights would be identical and behaviour unchanged.\n")

    fresh = build_brain(load_checkpoint=False)
    print(f"    {'module':<16} {'mean |trained - random|':>24}")
    print("    " + "-" * 42)
    total = 0.0
    for name in config.LOBES:
        a = dict(brain.cortex[name].named_parameters())
        b = dict(fresh.cortex[name].named_parameters())
        delta = float(np.mean([float((a[k] - b[k]).detach().abs().mean()) for k in a]))
        total += delta
        print(f"    {name:<16} {delta:>24.5f}")
    check("lobes moved away from initialisation", total > 0.01,
          f"summed change {total:.4f}")

    print("\n    behaviour, on the same input:")
    text = "the space shuttle reached orbit safely"
    trained = brain.perceive({"text": text}).answer.predictions[0]
    fresh.load_labels()
    random_out = fresh.perceive({"text": text}).answer.predictions[0]
    print(f"      trained  -> {trained.label} ({trained.probability:.1%})")
    print(f"      random   -> {random_out.label} ({random_out.probability:.1%})")
    check("trained answer differs from random", trained.label != random_out.label)


def stage5_fusion(brain) -> None:
    rule("5  FUSION -- does combining modalities change the answer?")
    print("If frontal ignored its extra streams, adding a modality would leave")
    print("the decision untouched. The vector is compared directly.\n")

    from PIL import Image
    image = Image.new("RGB", (224, 224), (150, 120, 80))
    sensor = np.random.rand(config.SENSOR_RAW_DIM).astype("float32")

    only_image = brain.perceive({"image": image}).decision
    with_text = brain.perceive({"image": image, "text": "a wall to the north"}).decision
    with_both = brain.perceive(
        {"image": image, "text": "a wall to the north", "sensor": sensor}
    ).decision

    print(f"    image alone      vs  image+text     cosine {cos(only_image, with_text):+.3f}")
    print(f"    image alone      vs  image+text+sensor cosine {cos(only_image, with_both):+.3f}")
    check("adding text changes the decision", cos(only_image, with_text) < 0.999)
    check("adding sensor changes it further", cos(with_text, with_both) < 0.999)

    slots = brain.cortex.frontal.last_active_slots
    print(f"\n    frontal slots filled on the 3-stream pass: "
          f"{[k for k, v in slots.items() if v]}")
    check("all three fusion slots used", all(slots.values()))


def stage6_endtoend(brain) -> None:
    rule("6  END TO END -- the full trace for one input")
    from PIL import Image

    image = Image.new("RGB", (224, 224), (90, 130, 170))
    trace = brain.perceive({"image": image, "text": "the goal lies north"})
    print(f"{'stage':<34} {'shape':>12}   note")
    print("-" * 78)
    for st in trace.stages:
        print(st)
    print("-" * 78)
    print(f"answer domain : {trace.domain}")
    print(f"gate          : {trace.gate.describe()}")
    print(f"elapsed       : {trace.elapsed_ms:.1f} ms")
    print(f"\n{trace.answer.verbalise()}")


STAGES = {
    1: stage1_features, 2: stage2_routing, 3: stage3_lobes,
    4: stage4_training, 5: stage5_fusion, 6: stage6_endtoend,
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", type=int, choices=sorted(STAGES))
    args = parser.parse_args()

    print("=" * 78)
    print("DOES IT ACTUALLY WORK?  evidence, not activity")
    print("=" * 78)
    brain = build_brain(load_checkpoint=True)
    print(f"checkpoint loaded: {brain.trained}   device: {brain.device_name}")

    # Where every part of the system actually lives. A backbone left behind on
    # the CPU while the lobes run on the GPU does not merely go slowly -- it
    # raises a device mismatch on the first matmul, so this is worth stating.
    print(f"torch {torch.__version__}  cuda build {torch.version.cuda}  "
          f"available {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        free, total = torch.cuda.mem_get_info()
        print(f"gpu: {torch.cuda.get_device_name(0)}  "
              f"{(total - free) / 1e9:.2f}/{total / 1e9:.1f} GB used")
    placements = {}
    for name in config.MODALITIES:
        placements[name] = str(brain.encoders[name].device)
    placements["lobes"] = str(next(brain.cortex.parameters()).device)
    print("placement: " + "  ".join(f"{k}={v}" for k, v in placements.items()))

    for number in ([args.stage] if args.stage else sorted(STAGES)):
        STAGES[number](brain)

    rule("VERDICT")
    if _fails:
        print(f"{len(_fails)} check(s) failed:")
        for f in _fails:
            print(f"   - {f}")
        return 1
    print("Every stage does what it claims:")
    print("  the frozen encoders carry meaning, not noise")
    print("  each modality reaches its own lobe")
    print("  each lobe transforms its input, and differently from the others")
    print("  training moved the weights and changed the behaviour")
    print("  fusion genuinely combines streams rather than ignoring them")
    return 0


if __name__ == "__main__":
    sys.exit(main())
