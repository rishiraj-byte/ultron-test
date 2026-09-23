"""
collect_v2.py -- measure everything the paper claims, and emit it as macros.

NOTHING IN THE PAPER'S RESULTS IS TYPED BY HAND. This script runs the real
gates, the real ablation sweep and the real checkpoint, writes the numbers to
paper/generated/v2.tex as LaTeX macros, and the paper \\input{}s them. If a
number in the paper is wrong, this script is where it came from, and rerunning
it is how it gets fixed.

That discipline matters more than usual here, because the paper's central
claim is about VERIFICATION. A paper arguing that brain-inspired architectures
should be machine-checked, whose own numbers were transcribed by hand from a
terminal three days ago, would be making its own point against itself.

    python paper/collect_v2.py
    python paper/collect_v2.py --with-benchmark    also run the comparison
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
#  This script lives in paper/ but measures the project in the directory
#  above it, so the project has to be importable from here.
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
GENERATED = ROOT / "paper" / "generated"
RESULTS = ROOT / "paper" / "results_v2.json"


def run(script: str, *args) -> str:
    """Run one of the project's own tools and hand back its output."""
    print("  running %s %s" % (script, " ".join(args)))
    proc = subprocess.run([sys.executable, script, *args],
                          cwd=str(ROOT), capture_output=True, text=True)
    return proc.stdout


def collect_gates() -> dict:
    text = run("verify_v2.py")
    passed = len(re.findall(r"\[  ok  \]", text))
    failed = len(re.findall(r"\[ FAIL \]", text))

    found = {}
    patterns = {
        "settling_ratio": r"seed 0: converged\s+ratio ([0-9.]+)",
        "permutation_gap": r"all 24 update orders give identical output\s+max difference ([0-9.e+-]+)",
        "gradient_coverage": r"no UNEXPECTED parameter is without gradient\s+[\d,]+ / [\d,]+\s+\(([0-9.]+)%\)",
        "sparsity_achieved": r"occipital reaches target sparsity\s+([0-9.]+)%",
        "dg_sparsity": r"DG reaches its target sparsity\s+([0-9.]+)%",
        "ach_low": r"raising ACh raises the feedforward share\s+([0-9.]+)% ->",
        "ach_high": r"raising ACh raises the feedforward share\s+[0-9.]+% -> ([0-9.]+)%",
        "replay_gain": r"replay improves retention of the earlier task\s+[0-9.]+% vs [0-9.]+% \(\+([0-9.]+) pts\)",
        "ca3_before": r"CA3 moves the cue toward the stored pattern\s+([0-9.]+) ->",
        "ca3_after": r"CA3 moves the cue toward the stored pattern\s+[0-9.]+ -> ([0-9.]+)",
        "weakest_ablation": r"every other part changes the output\s+weakest ([0-9.]+)%",
    }
    for key, pattern in patterns.items():
        match = re.search(pattern, text)
        if match:
            found[key] = match.group(1)
    found["gates_passed"] = passed
    found["gates_failed"] = failed
    return found


def collect_checkpoint() -> dict:
    import torch

    import config
    path = config.PROJECT_ROOT / "checkpoints" / "cortex_v2.pt"
    if not path.exists():
        return {}
    state = torch.load(path, map_location="cpu", weights_only=False)
    record = state.get("training_record") or {}
    out = {"epochs": record.get("epochs"),
           "examples": record.get("examples_seen"),
           "minutes": record.get("minutes")}
    out["classes"] = state.get("classes", {})
    return out


def collect_parameters() -> dict:
    import torch

    import config
    from cortex import Cortex
    torch.manual_seed(config.SEED)
    cortex = Cortex()
    return {
        "trainable": sum(p.numel() for p in cortex.parameters() if p.requires_grad),
        "frozen": sum(p.numel() for p in cortex.parameters() if not p.requires_grad),
        "tracts": len(cortex.tracts.names()),
        "timesteps": config.T_STEPS,
        "latent": config.LATENT_DIM,
        "sparsity": config.CORTICAL_SPARSITY,
    }


def macro(name: str, value) -> str:
    """One LaTeX macro. Percent signs escaped, because LaTeX eats them."""
    text = str(value).replace("%", r"\%").replace("_", r"\_")
    return r"\newcommand{\%s}{%s}" % (name, text)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--with-benchmark", action="store_true")
    args = parser.parse_args()

    print("collecting measurements for the paper...")
    results = {
        "gates": collect_gates(),
        "checkpoint": collect_checkpoint(),
        "architecture": collect_parameters(),
    }

    if args.with_benchmark:
        text = run("benchmark_v2.py")
        rows = re.findall(
            r"^(image|text|audio|sensor)\s+([0-9.]+)%\s+([0-9.]+)%\s+([0-9.]+)%\s+(.+)$",
            text, re.MULTILINE)
        results["benchmark"] = [
            {"task": r[0], "chance": r[1], "monolith": r[2],
             "modular": r[3], "verdict": r[4].strip()} for r in rows]

    RESULTS.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print("wrote %s" % RESULTS)

    GENERATED.mkdir(parents=True, exist_ok=True)
    lines = ["%% generated by paper/collect_v2.py -- do not edit by hand", ""]
    for group, values in results.items():
        if not isinstance(values, dict):
            continue
        for key, value in values.items():
            if isinstance(value, (str, int, float)):
                name = "".join(part.capitalize()
                               for part in (group + "_" + key).split("_"))
                lines.append(macro(name, value))
    (GENERATED / "v2.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("wrote %s" % (GENERATED / "v2.tex"))

    print()
    for group, values in results.items():
        print("%s:" % group)
        if isinstance(values, dict):
            for key, value in values.items():
                print("   %-22s %s" % (key, value))
        else:
            for row in values:
                print("   %s" % row)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
