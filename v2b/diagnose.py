"""What v2.B gets wrong, and why -- measured on the cached test sets.

    python -m v2b.diagnose

Runs every test set through the trained cortex, the RAG memory and the
amygdala exactly as training measured them, then breaks the errors down:
which sense's names it answered with, whether it fell back on a coarser
CIFAR name, what memory fixed and what it broke, how familiar the input
looked to CA1. Writes results/v2b_diagnose.json. ~2 minutes of GPU.

The "what if" rows change nothing in the checkpoint; they show what a
change WOULD do, so it can be justified before anything is retrained.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from collections import Counter

import numpy as np
import torch

from v2b import config_v2b as C
from v2b import train as T
from v2b.cortex import Cortex, Dims
from v2b.memory import CA1, Amygdala, Entorhinal
from v2b.wiring import Wiring

#  (label, kind, cached split). Imagenette's ten classes are in no training set.
SETS = (("CIFAR-100", "img", "test"),
        ("Tiny-ImageNet", "img", "tiny_test"),
        ("ImageNet photos", "img", "inet_test"),
        ("ESC-50 sounds", "aud", "test"),
        ("AudioSet eval", "aud", "snd_atest"),
        ("FSD50K eval", "aud", "snd_ftest"),
        ("Imagenette, never trained", "img", "open"))
CIFAR_NAMES = 100        # CIFAR-100 comes first in the vocabulary (data.image_vocabulary)
OUT = C.ROOT / "results" / "v2b_diagnose.json"


def wilson(right: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% interval for an accuracy. ESC-50's test is 400 clips: one clip is
    0.25 points, and a 1-point change is inside the noise."""
    if n == 0:
        return 0.0, 0.0
    p = right / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return 100 * (centre - half), 100 * (centre + half)


def training_running() -> bool:
    """The live log's last event, if a run is still writing it."""
    if not T.LIVE.exists() or time.time() - T.LIVE.stat().st_mtime > 300:
        return False
    lines = T.LIVE.read_text(encoding="utf-8").strip().splitlines()
    return bool(lines) and json.loads(lines[-1]).get("stage") != "done"


def load(device):
    manifest = json.loads(T._path("manifest.json").read_text())
    dims = Dims(**manifest["dims"])
    saved = torch.load(C.CHECKPOINT, map_location=device, weights_only=False)
    cortex = Cortex(dims, manifest["scale"], Wiring(),
                    sound_scale=manifest.get("sound_scale", 1.0)).to(device)
    cortex.load_state_dict(saved["cortex"])
    cortex.lexicon = torch.from_numpy(np.load(T._path("lexicon.npy"))).to(device)
    entorhinal = Entorhinal(dims.embed).to(device)
    entorhinal.load_state_dict(saved["entorhinal"])
    ca1 = CA1().to(device)
    ca1.load_state_dict(saved["ca1"])
    amygdala = Amygdala().to(device)
    amygdala.load_state_dict(saved["amygdala"])
    for m in (cortex, entorhinal, ca1, amygdala):
        m.eval()
    stored = np.load(C.MEMORY_BANK)
    bank = torch.from_numpy(stored["vectors"].astype(np.float32)).to(device)
    bank_labels = torch.from_numpy(stored["labels"]).to(device)
    words = torch.from_numpy(np.load(T._path("words.npy"))).to(device)
    return (dims, cortex, entorhinal, ca1, amygdala, bank, bank_labels, words,
            saved["names"], np.asarray(saved["kinds"]), saved.get("record", {}),
            saved.get("keep_expert"))


def masked(scores: torch.Tensor, rows: torch.Tensor) -> torch.Tensor:
    out = scores.clone()
    out[:, rows] = -1e4
    return out


@torch.no_grad()
def diagnose(device) -> dict:
    (dims, cortex, entorhinal, ca1, amygdala, bank, bank_labels, words,
     names, kinds, record, keep_expert) = load(device)
    vocab = len(names)
    known = torch.zeros(vocab, dtype=torch.bool, device=device)
    known[bank_labels.unique()] = True
    cautious = float(cortex.pot.keep)
    sound_rows = torch.from_numpy(kinds == "sound").to(device)
    trained = set(bank_labels.unique().tolist())
    report = {"checkpoint_finished": record.get("finished"), "memories": len(bank),
              "trust": float(ca1.trust), "gate_half_open": float(ca1.familiar_at),
              "sets": {}}
    for label, kind, part in SETS:
        if not T._path("%s_%s.labels.npy" % (kind, part)).exists():
            continue
        cache = T.Cache(kind, part, dims)
        logits, vectors, per_pass, y = T.run(cortex, cache, words, device)
        logits, y = logits.to(device), y.to(device)
        memory, closest = [], []
        for start in range(0, len(vectors), 1024):
            lp, cl = T._recall(entorhinal, ca1, bank, bank_labels,
                               vectors[start:start + 1024].to(device), vocab)
            memory.append(lp)
            closest.append(cl)
        memory, closest = torch.cat(memory), torch.cat(closest)
        gate = ca1.familiarity(closest)
        #  EXPERTISE, exactly as train.decide(): an answer the store holds
        #  memories of is re-read at keep_expert; the memory term is unchanged.
        cautious_logits = logits
        fused = logits + ca1.trust * gate[:, None] * memory
        if keep_expert is not None and keep_expert > cautious:
            cortex.pot.keep.fill_(keep_expert)
            hi = T.run(cortex, cache, words, device)[0].to(device)
            cortex.pot.keep.fill_(cautious)
            expert = known[fused.argmax(-1)]
            logits = torch.where(expert[:, None], hi, logits)
            fused = logits + ca1.trust * gate[:, None] * memory
        value = amygdala(Amygdala.features(logits, memory, fused, closest))
        go = value >= C.ABSTAIN_BELOW

        n = len(y)
        guess = fused.argmax(-1)
        right = guess == y
        cortex_right = logits.argmax(-1) == y
        wrong = ~right
        other_sense = (sound_rows[guess] != (kind == "aud")) & wrong
        coarser = (guess < CIFAR_NAMES) & (y >= CIFAR_NAMES) & wrong & ~other_sense
        guess_trained = torch.tensor([int(g) in trained for g in guess.tolist()], device=device)
        truth_trained = torch.tensor([int(t) in trained for t in y.tolist()], device=device)
        another_trained = guess_trained & wrong & ~other_sense & ~coarser
        untrained_name = wrong & ~other_sense & ~coarser & ~another_trained

        #  What if a picture could only be named with names of things seen,
        #  and a sound only with names of things heard?
        other = sound_rows if kind == "img" else ~sound_rows
        m_cortex = masked(logits, other).argmax(-1) == y
        m_fused = masked(fused, other).argmax(-1) == y

        per_class = {}
        for t in sorted(set(y.tolist())):
            sel = y == t
            per_class[names[t]] = round(float(right[sel].float().mean() * 100), 1)
        confusions = Counter((names[int(t)], names[int(g)])
                             for t, g in zip(y[wrong].tolist(), guess[wrong].tolist()))
        r = int(right.sum())
        lo, hi = wilson(r, n)
        report["sets"][label] = {
            "n": n,
            "cortex": round(float(cortex_right.float().mean() * 100), 2),
            "cortex_cautious": round(float((cautious_logits.argmax(-1) == y).float().mean() * 100), 2),
            "memory": round(float((memory.argmax(-1) == y).float().mean() * 100), 2),
            "final": round(100 * r / n, 2), "final_ci95": [round(lo, 1), round(hi, 1)],
            "final_top5": round(float((fused.topk(5, -1).indices == y[:, None]).any(-1)
                                      .float().mean() * 100), 2),
            "per_pass": [round(p, 2) for p in per_pass],
            "memory_fixed": int((~cortex_right & right).sum()),
            "memory_broke": int((cortex_right & ~right).sum()),
            "closest_median": round(float(closest.median()), 3),
            "gate_open_share": round(float((gate >= 0.5).float().mean() * 100), 1),
            "truth_is_trained_class": round(float(truth_trained.float().mean() * 100), 1),
            "go_share": round(float(go.float().mean() * 100), 1),
            "right_when_go": round(float(right[go].float().mean() * 100), 1) if go.any() else None,
            "right_when_hold": round(float(right[~go].float().mean() * 100), 1) if (~go).any() else None,
            "errors": {
                "total": int(wrong.sum()),
                "named_with_other_sense": int(other_sense.sum()),
                "coarser_cifar_name": int(coarser.sum()),
                "another_trained_name": int(another_trained.sum()),
                "an_untrained_name": int(untrained_name.sum())},
            "what_if_names_of_this_sense_only": {
                "cortex": round(float(m_cortex.float().mean() * 100), 2),
                "final": round(float(m_fused.float().mean() * 100), 2)},
            "worst_classes": dict(sorted(per_class.items(), key=lambda kv: kv[1])[:10]),
            "top_confusions": [[a, b, c] for (a, b), c in confusions.most_common(10)],
        }
    return report


def show(report: dict) -> None:
    print("\nv2.B DIAGNOSIS   memories %d   CA1 trust %.3f   gate half-open at %.3f"
          % (report["memories"], report["trust"], report["gate_half_open"]))
    print("%-27s %6s %7s %7s %7s %13s %6s %5s %8s"
          % ("test", "n", "cortex", "memory", "final", "95% interval", "top-5", "GO", "right|GO"))
    for label, s in report["sets"].items():
        print("%-27s %6d %6.1f%% %6.1f%% %6.1f%%  [%4.1f, %4.1f] %5.1f%% %4.0f%% %7.1f%%"
              % (label, s["n"], s["cortex"], s["memory"], s["final"], *s["final_ci95"],
                 s["final_top5"], s["go_share"], s["right_when_go"] or 0))
    for label, s in report["sets"].items():
        e = s["errors"]
        w = s["what_if_names_of_this_sense_only"]
        print("\n%s -- %d wrong" % (label, e["total"]))
        print("  named with the other sense's names %5d   coarser CIFAR name %5d"
              % (e["named_with_other_sense"], e["coarser_cifar_name"]))
        print("  another trained name               %5d   an untrained name  %5d"
              % (e["another_trained_name"], e["an_untrained_name"]))
        print("  memory fixed %d, broke %d   closest memory median %.3f, gate open on %.0f%%"
              % (s["memory_fixed"], s["memory_broke"], s["closest_median"], s["gate_open_share"]))
        print("  what if only this sense's names could answer: cortex %.1f%%  final %.1f%%"
              % (w["cortex"], w["final"]))
        print("  top confusions: " + "; ".join("%s -> %s %d" % tuple(c)
                                               for c in s["top_confusions"][:6]))
        print("  worst classes: " + ", ".join("%s %.0f%%" % kv
                                             for kv in list(s["worst_classes"].items())[:6]))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--anyway", action="store_true",
                        help="run even though a training run looks active")
    args = parser.parse_args()
    #  A concurrent GPU job killed a v2 training run at epoch 4 (STATE.md).
    if training_running() and not args.anyway:
        raise SystemExit("a training run is writing %s -- wait for it (or --anyway)" % T.LIVE)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    report = diagnose(device)
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2))
    show(report)
    print("\nwritten to %s" % OUT)


if __name__ == "__main__":
    main()
