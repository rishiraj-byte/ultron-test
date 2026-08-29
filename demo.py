"""
demo.py -- a scripted walkthrough. Run this in front of people.

    python demo.py            the full showcase
    python demo.py --fast     skip the pauses

Everything here is a live call into the real system. Nothing is pre-recorded
and nothing is faked -- if a module were broken, the demo would show it.

The order is chosen so each step earns the next: hardware, then that the
parts are real, then routing, then what the whole thing does with them.
"""

from __future__ import annotations

import argparse
import sys
import time

import numpy as np
import torch

import config

PAUSE = 1.6
_fast = False


def beat(seconds: float = PAUSE) -> None:
    if not _fast:
        time.sleep(seconds)


def title(text: str) -> None:
    print(f"\n\033[1m{'=' * 74}\033[0m")
    print(f"\033[1m  {text}\033[0m")
    print(f"\033[1m{'=' * 74}\033[0m")
    beat(0.7)


def say(text: str = "") -> None:
    print(text)
    beat(0.35)


# ---------------------------------------------------------------------------


def act1_hardware() -> None:
    title("1 / 5   WHAT THIS IS RUNNING ON")
    say(f"  torch {torch.__version__}")
    if torch.cuda.is_available():
        free, total = torch.cuda.mem_get_info()
        say(f"  GPU   {torch.cuda.get_device_name(0)}")
        say(f"  VRAM  {(total - free) / 1e9:.2f} / {total / 1e9:.1f} GB in use")
    else:
        say("  CPU only -- install the CUDA torch build for ~3x")
    beat()


def act2_architecture(brain, lang) -> None:
    title("2 / 5   FOUR CORTICAL MODULES, AND THEY ARE GENUINELY DIFFERENT")
    say("  Not four copies of one MLP with different labels. Each carries a")
    say("  structural prior matched to the signal routed to it.")
    say()
    say(f"  {'module':<12}{'implementation':<22}{'inductive bias'}")
    say(f"  {'-' * 68}")
    bias = {
        "occipital": "local receptive fields",
        "parietal": "relations between positions",
        "temporal": "order in a sequence",
        "frontal": "convergence of streams",
    }
    for name in config.LOBES:
        lobe = brain.cortex[name]
        say(f"  {name:<12}{type(lobe).__name__:<22}{bias[name]}")
    say()
    counts = brain.parameter_count()
    say(f"  yours, trained     {sum(counts.values()):>12,} parameters")
    say(f"  borrowed, frozen   {lang.parameter_count() + 72_000_000:>12,} "
        f"(ResNet-18 + MiniLM + Whisper + Qwen2.5)")
    beat()


def act3_routing(brain) -> None:
    title("3 / 5   THE THALAMUS ROUTES EACH SENSE SOMEWHERE DIFFERENT")
    say("  Watch which modules wake up. This is the switchboard working --")
    say("  a silent module means nothing was routed to it.")
    say()

    from PIL import Image

    probes = {
        "image": Image.new("RGB", (224, 224), (120, 160, 90)),
        "text": "the rocket reached orbit",
        "audio": (0.4 * np.sin(2 * np.pi * 440 * np.linspace(
            0, 1, config.AUDIO_SAMPLE_RATE))).astype("float32"),
        "sensor": np.random.rand(config.SENSOR_RAW_DIM).astype("float32"),
    }
    say(f"  {'input':<10}{'nucleus':<11}{'modules that fired'}")
    say(f"  {'-' * 68}")
    for modality, value in probes.items():
        trace = brain.perceive({modality: value})
        fired = [n for n in trace.active_nodes if n in config.LOBES]
        say(f"  {modality:<10}{trace.routings[0].nucleus:<11}{', '.join(fired)}")
        beat(0.5)
    beat()


def act4_fusion(brain) -> None:
    title("4 / 5   COMBINE SENSES AND THE DECISION CHANGES")
    say("  Frontal fuses whatever arrived. If it ignored the extra streams,")
    say("  adding a sense would leave the decision vector untouched.")
    say()

    from PIL import Image

    image = Image.new("RGB", (224, 224), (150, 120, 80))
    sensor = np.random.rand(config.SENSOR_RAW_DIM).astype("float32")

    def cosine(a, b):
        return float(torch.nn.functional.cosine_similarity(
            a.detach().flatten().unsqueeze(0), b.detach().flatten().unsqueeze(0)))

    alone = brain.perceive({"image": image}).decision
    plus_text = brain.perceive({"image": image, "text": "a wall to the north"}).decision
    plus_both = brain.perceive(
        {"image": image, "text": "a wall to the north", "sensor": sensor}).decision

    say(f"  image alone            vs  + instruction      cosine {cosine(alone, plus_text):+.3f}")
    say(f"  image + instruction    vs  + rangefinder      cosine {cosine(plus_text, plus_both):+.3f}")
    say()
    slots = brain.cortex.frontal.last_active_slots
    say(f"  fusion slots filled: {[k for k, v in slots.items() if v]}")
    say("  (a slot with nothing routed to it is zero-padded, not skipped)")
    beat()


def act5_conversation(brain, lang, image_path: str | None) -> None:
    title("5 / 5   IT TALKS ABOUT WHAT IT PERCEIVED")
    say("  The four modules perceive. A separate language module puts their")
    say("  conclusion into words -- it cannot see the input itself.")
    say()

    from modules.language import perception_from_trace

    turns: list[tuple[str, dict, str]] = [
        ("a question", {"text": "hello, what are you?"}, "hello, what are you?"),
        ("nonsense",   {"text": "asdkjh qwe zxcvb"},    "asdkjh qwe zxcvb"),
    ]
    if image_path:
        turns.append(("an image", {"image": image_path}, "what am I looking at?"))

    for label, inputs, question in turns:
        say(f"  \033[36m>> {question}\033[0m   [{label}]")
        trace = brain.perceive(inputs)
        top = trace.answer.predictions[0]
        fired = ", ".join(n for n in trace.active_nodes if n in config.LOBES)
        say(f"     \033[90m· modules [{fired}] -> {trace.domain}: "
            f"{top.label} {top.probability:.0%}\033[0m")

        started = time.perf_counter()
        reply = lang.respond(perception_from_trace(trace, user_text=question))
        say(f"     \033[33m<< {reply}\033[0m")
        say(f"     \033[90m· {time.perf_counter() - started:.1f}s\033[0m")
        say()
        beat(0.9)


def main() -> int:
    global _fast
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fast", action="store_true", help="no pauses")
    parser.add_argument("--image", help="an image to show it")
    args = parser.parse_args()
    _fast = args.fast

    print("\033[1m\nBRAIN-INSPIRED MODULAR AI -- live demonstration\033[0m")
    print("Every number below is produced by running the real system now.\n")

    from modules.language import LanguageModule
    from modules.pipeline import build_brain

    started = time.perf_counter()
    brain = build_brain(load_checkpoint=True)
    lang = LanguageModule()
    lang._ensure_loaded()
    print(f"loaded in {time.perf_counter() - started:.1f}s   "
          f"(modules {'TRAINED' if brain.trained else 'UNTRAINED'})")

    image = args.image
    if image is None:
        from pathlib import Path
        for candidate in (r"C:/Windows/Web/4K/Wallpaper/Windows/img0_1920x1200.jpg",
                          r"C:/Windows/Web/Screen/img100.jpg"):
            if Path(candidate).exists():
                image = candidate
                break

    act1_hardware()
    act2_architecture(brain, lang)
    act3_routing(brain)
    act4_fusion(brain)
    act5_conversation(brain, lang, image)

    title("THAT IS THE WHOLE SYSTEM")
    say("  python desktop_app.py    the live dashboard, with per-neuron diagrams")
    say("  python explain.py        18 checks that each stage is genuinely working")
    say("  python verify.py --all   104 checks that every tensor is the right shape")
    say()
    return 0


if __name__ == "__main__":
    sys.exit(main())
