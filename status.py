"""
status.py -- what is actually built, read from the code rather than claimed.

Run this any time. It does not trust a changelog, a comment or anything anyone
said in a chat window: it imports the real modules, looks for the real classes,
counts the real parameters, and reports what it finds. If a stage is listed as
BUILT here, the file and the class exist and import cleanly. If it says the
system is untrained, the checkpoint genuinely did not load.

    python status.py            what exists, what does not
    python status.py --gates    the above, then run every gate for real
"""

from __future__ import annotations

import argparse
import importlib
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent

#  stage -> (what it is, module path, the class that proves it, gate number)
STAGES = [
    (1, "recurrent settling loop, named tracts", "cortex", "Cortex", 1),
    (2, "un-pooled encoder taps", "modules.sensory", "SensoryFront", 6),
    (3, "corticothalamic feedback + TRN", "modules.thalamus_v2", "ThalamusV2", 7),
    (4, "laminar L4/L2-3/L5-6, k-WTA, Dale", "modules.laminar", "LaminarShell", 8),
    (5, "basal ganglia, loop closed", "modules.basal_ganglia_v2", "BasalGangliaV2", 9),
    (6, "amygdala, hippocampus, hypothalamus", "modules.limbic", "Hippocampus", 10),
    (7, "cerebellar forward model", "modules.cerebellum", "Cerebellum", 11),
    (8, "neuromodulators + oscillations", "modules.neuromodulators", "Neuromodulators", 12),
    (9, "ablate.py full sweep", "ablate", "AblationSweep", 13),
    (10, "desktop UI rebuilt for v2", "desktop_app_v2", "Console", None),
]

#  Beyond the ten build stages: the pieces that make it a usable product
#  rather than a verified architecture.
DELIVERABLES = [
    ("training", "train_v2", "BrainV2", "python train_v2.py"),
    ("assembled system", "brain_v2", "BrainV2System", "used by the console"),
    ("live brain view", "brainview", "BrainRenderer", "shown in the console"),
    ("ablation sweep", "ablate", "AblationSweep", "python ablate.py"),
    ("benchmark", "benchmark_v2", "Monolith", "python benchmark_v2.py"),
]

#  Paper sources, checked as files rather than imports.
PAPER_FILES = [
    ("paper draft", "paper/main_v2.tex", "pdflatex main_v2 (in paper/)"),
    ("number collector", "paper/collect_v2.py", "python paper/collect_v2.py"),
]


def probe(module_path: str, class_name: str) -> tuple[bool, str]:
    """Does this module import, and does it contain that class?"""
    try:
        module = importlib.import_module(module_path)
    except ModuleNotFoundError:
        return False, "no module %s" % module_path
    except Exception as exc:                                  # noqa: BLE001
        return False, "%s: %s" % (type(exc).__name__, str(exc)[:44])
    if not hasattr(module, class_name):
        return False, "%s has no %s" % (module_path, class_name)
    return True, "%s.%s" % (module_path, class_name)


def gate_numbers() -> set[int]:
    """Which gates verify_v2.py actually defines, by reading its source."""
    source = (ROOT / "verify_v2.py").read_text(encoding="utf-8")
    return {int(line.split("gate")[1].split("_")[0])
            for line in source.splitlines()
            if line.startswith("def gate")}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gates", action="store_true",
                        help="also RUN every gate (slow, but it is the proof)")
    args = parser.parse_args()

    print("=" * 74)
    print("BRAIN-AI  --  BUILD STATUS")
    print("=" * 74)

    gates = gate_numbers()
    print("%-5s %-38s %-9s %s" % ("stage", "what it is", "code", "gate"))
    print("-" * 74)

    built = 0
    for number, what, module_path, class_name, gate in STAGES:
        ok, detail = probe(module_path, class_name)
        built += ok
        has_gate = "gate %d" % gate if gate in gates else (
            "-" if gate is None else "MISSING")
        print("%-5d %-38s %-9s %s"
              % (number, what, "BUILT" if ok else "not yet", has_gate))
        if not ok and number <= 6:
            print("      ^ expected this to exist: %s" % detail)

    print("-" * 74)
    print("%d of %d stages built." % (built, len(STAGES)))

    print()
    print("%-20s %-9s %s" % ("deliverable", "code", "how to run it"))
    print("-" * 74)
    for label, module_path, class_name, how in DELIVERABLES:
        ok, _ = probe(module_path, class_name)
        print("%-20s %-9s %s" % (label, "BUILT" if ok else "not yet", how))
    for label, path, how in PAPER_FILES:
        ok = (ROOT / path).exists()
        print("%-20s %-9s %s" % (label, "BUILT" if ok else "not yet", how))

    # ---- is it trained? ----
    print()
    print("TRAINED?")
    checkpoint = ROOT / "checkpoints" / "cortex_v2.pt"
    if not checkpoint.exists():
        print("   no v2 checkpoint at %s" % checkpoint)
        print("   the lobes are randomly initialised -- answers are noise")
        print("   Fix:  python train_v2.py")
    else:
        try:
            import torch
            state = torch.load(checkpoint, map_location="cpu",
                               weights_only=False)
            record = state.get("training_record") or {}
            if record:
                print("   TRAINED. %s"
                      % "  ".join("%s=%s" % kv for kv in sorted(record.items())))
            else:
                #  "It loads" is not "it is trained". A checkpoint written by
                #  a save/reload TEST loads perfectly and contains random
                #  weights; this line used to call that trained.
                print("   a checkpoint exists and loads, but carries NO")
                print("   training record -- so nothing trained these weights.")
                print("   Most likely a test artefact. The lobes are random,")
                print("   and any answer they give is noise.")
                print("   Fix:  python train_v2.py")
        except Exception as exc:                              # noqa: BLE001
            print("   a checkpoint exists but DOES NOT LOAD:")
            print("   %s" % str(exc).splitlines()[0])
            print("   -> the lobes are randomly initialised, answers are noise")

    #  ---- can this machine actually read the files you have? ----
    #
    #  Worth a section of its own, because the failure it catches does not
    #  look like a missing library from the outside. With only scipy present
    #  every .mp3 fell through the whole loader chain and reported "no audio
    #  loader available", which reads as "nothing is installed" when in fact
    #  the install was fine and the FORMAT was not supported.
    print()
    print("WHAT THIS MACHINE CAN DECODE")
    try:
        import desktop_app_v2
        for kind, how in desktop_app_v2.media_support().items():
            healthy = how != "UNAVAILABLE" and "uncompressed only" not in how
            print("   %-6s %-34s %s" % (kind.upper(), how,
                                        "ok" if healthy else "LIMITED"))
        print("   Fix any LIMITED line with:")
        print("     pip install soundfile imageio-ffmpeg opencv-python")
    except Exception as exc:                                  # noqa: BLE001
        print("   could not check: %s" % str(exc).splitlines()[0])

    # ---- what is in the lobes ----
    print()
    print("WHAT IS INSIDE THE FOUR LOBES")
    try:
        import config
        from cortex import Cortex
        import torch
        torch.manual_seed(config.SEED)
        cortex = Cortex()
        for name in cortex.LOBES:
            lobe = cortex.lobes[name]
            core = getattr(lobe, "core", None)
            if core is None:
                print("   %-10s randomly initialised MLP (not pretrained)" % name)
            else:
                frozen = sum(p.numel() for p in core.teacher.parameters())
                print("   %-10s %-20s %s params, frozen"
                      % (name, core.teacher_name, format(frozen, ",")))
        trainable = sum(p.numel() for p in cortex.parameters() if p.requires_grad)
        frozen = sum(p.numel() for p in cortex.parameters() if not p.requires_grad)
        print("   %-10s trainable %s | frozen pretrained %s"
              % ("TOTAL", format(trainable, ","), format(frozen, ",")))
    except Exception as exc:                                  # noqa: BLE001
        print("   could not inspect: %s: %s" % (type(exc).__name__, exc))

    print()
    print("COMMANDS")
    print("   python status.py --gates   run every gate and see it pass or fail")
    print("   python verify_v2.py        the v2 stage gates, in detail")
    print("   python verify.py           the v1 shape and wiring checks")
    print("   python explain.py          what each part does, demonstrated")
    print("   python train_v2.py         train the four lobes")
    print("   python desktop_app_v2.py   talk to it and watch it think")
    print("   python ablate.py           cut each part out, see what matters")
    print("   python benchmark_v2.py     modular vs one MLP, parameter-matched")
    print("   python paper/collect_v2.py measure everything the paper claims")
    print("   python desktop_app_v2.py   the v2 dashboard (window)")
    print("   python desktop_app_v2.py --selftest   the same, as text")

    if args.gates:
        print()
        print("=" * 74)
        print("RUNNING THE GATES FOR REAL -- this takes a few minutes")
        print("=" * 74)
        for script in ("verify_v2.py", "verify.py"):
            print()
            print("--- %s ---" % script)
            result = subprocess.run([sys.executable, script],
                                    capture_output=True, text=True)
            lines = result.stdout.splitlines()
            passed = sum(1 for l in lines if "[  ok  ]" in l)
            failed = [l for l in lines if "[ FAIL ]" in l]
            print("   %d checks passed, %d failed" % (passed, len(failed)))
            for line in failed[:10]:
                print("   %s" % line.strip())
            if result.returncode != 0 and not failed:
                print("   script exited %d" % result.returncode)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
