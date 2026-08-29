"""
tables.py -- results.json -> LaTeX tables and figures for the paper.

    python -m benchmark.tables

Generated files are written under paper/generated/ and \\input{} by main.tex,
so the paper never contains a hand-typed number. Anything typed by hand is a
number that can silently go stale after a rerun; anything generated here is
guaranteed to match results.json.

Every table reports mean +/- std over seeds, and the comparison table reports
a paired p-value and Cohen's d_z rather than accuracy alone, because with five
seeds a bare mean difference is not evidence.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import config
from benchmark.harness import RESULTS_DIR, paired_test

OUT_DIR = config.PROJECT_ROOT / "paper" / "generated"

VARIANT_LABEL = {
    "modular": r"\textbf{Modular (ours)}",
    "no_routing": "-- routing (broadcast)",
    "homogeneous": "-- specialised lobes",
    "shared_lobe": "-- modularity (shared)",
    "monolithic": "Monolithic MLP",
}

TASK_LABEL = {
    "gridnav": "GridNav",
    "cifar100": "CIFAR-100",
    "news20": "20 Newsgroups",
}

FAILURE_LABEL = {
    "occipital": "occipital lesion",
    "parietal": "parietal lesion",
    "temporal": "temporal lesion",
    "modality:image": "camera fails",
    "modality:sensor": "rangefinder fails",
    "modality:text": "instruction lost",
}


def _load(name: str = "results") -> dict:
    path = RESULTS_DIR / f"{name}.json"
    if not path.exists():
        raise SystemExit(
            f"no results at {path}. Run:  python -m benchmark.run"
        )
    return json.loads(path.read_text(encoding="utf-8"))


def _esc(text: str) -> str:
    return text.replace("_", r"\_").replace("%", r"\%")


# ---------------------------------------------------------------------------


def table_main(payload: dict) -> str:
    """Accuracy per task per variant, mean +/- std over seeds."""
    summary = payload["summary"]
    tasks, variants = [], []
    for key in summary:
        task, variant = key.split("/")
        if task not in tasks:
            tasks.append(task)
        if variant not in variants:
            variants.append(variant)

    lines = [
        r"\begin{table}[t]",
        r"\caption{Test accuracy, mean $\pm$ s.d. over seeds. All variants are "
        r"parameter-matched to the modular model and trained under an identical "
        r"protocol. GridNav is scored as set accuracy, since several actions can "
        r"be simultaneously optimal.}",
        r"\label{tab:main}",
        r"\centering",
        r"\begin{tabular}{l" + "c" * len(tasks) + "}",
        r"\hline",
        "Model & " + " & ".join(TASK_LABEL.get(t, t) for t in tasks) + r" \\",
        r"\hline",
    ]

    for variant in variants:
        cells = []
        for task in tasks:
            entry = summary.get(f"{task}/{variant}")
            cells.append(
                "--" if entry is None
                else f"{entry['test_mean']:.3f} $\\pm$ {entry['test_std']:.3f}"
            )
        lines.append(VARIANT_LABEL.get(variant, variant) + " & " + " & ".join(cells) + r" \\")

    lines += [r"\hline", r"\end{tabular}", r"\end{table}", ""]
    return "\n".join(lines)


def table_significance(payload: dict) -> str:
    """Paired modular-vs-alternative comparisons."""
    runs = payload["runs"]
    by_key: dict[tuple[str, str], dict[int, float]] = {}
    for r in runs:
        by_key.setdefault((r["task"], r["variant"]), {})[r["seed"]] = r["test_accuracy"]

    tasks = sorted({t for t, _ in by_key})
    lines = [
        r"\begin{table}[t]",
        r"\caption{Paired comparison of the modular model against each "
        r"alternative. Seeds are shared, so each row is a paired $t$-test over "
        r"matched runs; $d_z$ is the paired effect size. Positive $\Delta$ "
        r"favours the modular model.}",
        r"\label{tab:significance}",
        r"\centering",
        r"\begin{tabular}{llrrr}",
        r"\hline",
        r"Task & Compared with & $\Delta$acc & $p$ & $d_z$ \\",
        r"\hline",
    ]

    for task in tasks:
        base = by_key.get((task, "modular"))
        if not base:
            continue
        for variant in ("no_routing", "homogeneous", "shared_lobe", "monolithic"):
            other = by_key.get((task, variant))
            if not other:
                continue
            seeds = sorted(set(base) & set(other))
            stat = paired_test([base[s] for s in seeds], [other[s] for s in seeds])
            mark = "$^{*}$" if stat["p"] == stat["p"] and stat["p"] < 0.05 else ""
            lines.append(
                f"{TASK_LABEL.get(task, task)} & {_esc(VARIANT_LABEL.get(variant, variant))} & "
                f"{stat['mean_diff']:+.3f}{mark} & {stat['p']:.3f} & {stat['d']:.2f} \\\\"
            )

    lines += [
        r"\hline",
        r"\multicolumn{5}{l}{\footnotesize $^{*}$ $p<0.05$}",
        r"\end{tabular}",
        r"\end{table}",
        "",
    ]
    return "\n".join(lines)


def table_damage(payload: dict) -> str:
    """Accuracy retained under lobe lesion and sensor failure."""
    summary = payload["summary"]
    rows: dict[str, dict[str, float]] = {}
    failures: list[str] = []

    for key, entry in summary.items():
        task, variant = key.split("/")
        if task != "gridnav" or not entry.get("lesion"):
            continue
        intact = entry["test_mean"]
        rows[variant] = {"intact": intact}
        for failure, stats in entry["lesion"].items():
            if failure not in failures:
                failures.append(failure)
            rows[variant][failure] = stats["mean"] / intact if intact else float("nan")

    if not rows:
        return "% no damage-tolerance data\n"

    lines = [
        r"\begin{table}[t]",
        r"\caption{Fraction of intact accuracy retained on GridNav under module "
        r"lesion and sensor failure. A lesioned lobe contributes nothing and its "
        r"fusion slot is zero-padded; a failed sensor is removed at the input.}",
        r"\label{tab:damage}",
        r"\centering",
        r"\begin{tabular}{l" + "c" * len(failures) + "}",
        r"\hline",
        "Model & " + " & ".join(_esc(FAILURE_LABEL.get(f, f)) for f in failures) + r" \\",
        r"\hline",
    ]
    for variant, values in rows.items():
        cells = [f"{values.get(f, float('nan')):.1%}".replace("%", r"\%")
                 for f in failures]
        lines.append(VARIANT_LABEL.get(variant, variant) + " & " + " & ".join(cells) + r" \\")

    lines += [r"\hline", r"\end{tabular}", r"\end{table}", ""]
    return "\n".join(lines)


def table_cost(payload: dict) -> str:
    """Parameters and inference MACs."""
    summary = payload["summary"]
    lines = [
        r"\begin{table}[t]",
        r"\caption{Trainable parameters and inference cost per example. "
        r"Multiply--accumulates are counted from an executed forward pass, so a "
        r"module that routing skips contributes nothing.}",
        r"\label{tab:cost}",
        r"\centering",
        r"\begin{tabular}{llrr}",
        r"\hline",
        r"Task & Model & Params & MMACs \\",
        r"\hline",
    ]
    for key, entry in summary.items():
        task, variant = key.split("/")
        lines.append(
            f"{TASK_LABEL.get(task, task)} & {VARIANT_LABEL.get(variant, variant)} & "
            f"{entry['params']:,} & {entry['macs'] / 1e6:.2f} \\\\"
        )
    lines += [r"\hline", r"\end{tabular}", r"\end{table}", ""]
    return "\n".join(lines)


def figure_damage(payload: dict) -> Path | None:
    """Bar chart of retained accuracy under failure."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return None

    summary = payload["summary"]
    data: dict[str, dict[str, float]] = {}
    for key, entry in summary.items():
        task, variant = key.split("/")
        if task != "gridnav" or not entry.get("lesion"):
            continue
        intact = entry["test_mean"]
        data[variant] = {
            f: (s["mean"] / intact if intact else 0.0)
            for f, s in entry["lesion"].items()
        }
    if not data:
        return None

    failures = list(next(iter(data.values())).keys())
    x = np.arange(len(failures))
    width = 0.8 / max(len(data), 1)

    fig, ax = plt.subplots(figsize=(7.0, 3.0), dpi=200)
    for i, (variant, values) in enumerate(data.items()):
        ax.bar(x + i * width, [values[f] for f in failures], width,
               label=variant.replace("_", " "))
    ax.axhline(1.0, color="0.4", linewidth=0.8, linestyle="--")
    ax.set_xticks(x + width * (len(data) - 1) / 2)
    ax.set_xticklabels([FAILURE_LABEL.get(f, f) for f in failures],
                       rotation=20, ha="right", fontsize=8)
    ax.set_ylabel("accuracy retained")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / "damage.pdf"
    fig.savefig(path)
    plt.close(fig)
    return path


def main() -> int:
    payload = _load()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    generated = {
        "table_main.tex": table_main(payload),
        "table_significance.tex": table_significance(payload),
        "table_damage.tex": table_damage(payload),
        "table_cost.tex": table_cost(payload),
    }
    for name, body in generated.items():
        (OUT_DIR / name).write_text(body, encoding="utf-8")
        print(f"  wrote {OUT_DIR / name}")

    figure = figure_damage(payload)
    if figure:
        print(f"  wrote {figure}")
    else:
        print("  (no damage figure: matplotlib missing or no lesion data)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
