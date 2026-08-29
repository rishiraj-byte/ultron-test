"""
harness.py -- train, evaluate and compare, with the rigour a paper needs.

Design rules, each of which exists because a reviewer would otherwise ask:

  * EVERY variant sees the same cached frozen features, the same splits, the
    same optimiser, the same schedule and the same epoch budget. The only
    difference is the architecture.
  * EVERY configuration runs over several seeds. A single run is an anecdote;
    the tables report mean +/- std.
  * Seeds are PAIRED across variants -- seed k means the same data order and
    the same initialisation draw for every model -- so the comparison can use a
    paired test, which is far more sensitive than comparing two independent
    means.
  * A validation split is carved out of TRAIN for model selection. The test
    split is touched exactly once per run, at the end. Selecting on test is the
    single most common way this kind of result turns out to be nothing.
  * Accuracy on gridnav is SET accuracy: several actions can be equally
    optimal, so scoring against one arbitrary label would punish a correct
    choice. See gridworld.Observation.optimal.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import config
from benchmark.baselines import (
    ModalityProjections,
    ModularStack,
    MonolithicMLP,
    count_params,
    matched_hidden,
)
from benchmark.data import TaskData

RESULTS_DIR = config.PROJECT_ROOT / "results"
DEVICE = config.resolve_device()


@dataclass
class RunResult:
    task: str
    variant: str
    seed: int
    params: int
    val_accuracy: float
    test_accuracy: float
    epochs_ran: int
    train_seconds: float
    #  lesion label -> test accuracy with that module disabled
    lesion: dict[str, float] = field(default_factory=dict)
    #  mean multiply-accumulates per example at inference
    macs: float = 0.0


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


class ModularModel(nn.Module):
    """Projections + the routed lobe stack. One of the variants under test."""

    def __init__(self, task: TaskData, variant: str, lobe_hidden: int | None = None):
        super().__init__()
        self.projections = ModalityProjections(task.raw_dims())
        self.stack = ModularStack(
            num_classes=task.num_classes, variant=variant, lobe_hidden=lobe_hidden
        )
        self.modalities = task.modalities

    def forward(self, batch: dict[str, torch.Tensor],
                lesion: set[str] | None = None,
                drop: set[str] | None = None) -> torch.Tensor:
        """`drop` removes whole MODALITIES (a sensor failing); `lesion` removes
        LOBES (damage to the network itself). They are different experiments
        and must not be conflated."""
        raw = {m: v for m, v in batch.items() if not drop or m not in drop}
        if not raw:                       # every sensor gone
            raw = {self.modalities[0]: torch.zeros_like(batch[self.modalities[0]])}
        return self.stack(self.projections(raw), lesion=lesion)


class MonolithicModel(nn.Module):
    """The flat baseline, wrapped to the same call signature.

    Receives the concatenation of exactly the same encoded features the modular
    model receives, so the comparison isolates structure.
    """

    def __init__(self, task: TaskData, target_params: int):
        super().__init__()
        self.layout = flat_layout(task)
        in_dim = sum(hi - lo for lo, hi in self.layout.values())
        self.net = MonolithicMLP(in_dim, task.num_classes, target_params)

    def forward(self, batch: dict[str, torch.Tensor],
                lesion: set[str] | None = None,
                drop: set[str] | None = None) -> torch.Tensor:
        x = batch["__flat__"]
        if drop:
            # A failed sensor zeroes its slice. There is no lobe to lesion, so
            # `lesion` is ignored -- that asymmetry is the point of the
            # comparison, not an oversight.
            x = x.clone()
            for modality in drop:
                if modality in self.layout:
                    lo, hi = self.layout[modality]
                    x[:, lo:hi] = 0.0
        return self.net(x)


def build_model(task: TaskData, variant: str, reference_params: int | None = None):
    if variant == "monolithic":
        return MonolithicModel(task, reference_params or 1_000_000)
    hidden = None
    if variant in ("shared_lobe", "homogeneous") and reference_params:
        # Search over the COMPLETE model, projections included, so the matched
        # count is the count that actually gets trained.
        hidden = matched_hidden(
            variant, task.num_classes, reference_params,
            build=lambda h: ModularModel(task, variant, lobe_hidden=h),
        )
    return ModularModel(task, variant, lobe_hidden=hidden)


# ---------------------------------------------------------------------------
# Data plumbing
# ---------------------------------------------------------------------------


def flat_layout(task: TaskData) -> dict[str, tuple[int, int]]:
    """Where each modality sits inside the monolithic model's input vector.

    Needed so that "the camera failed" can be applied to the monolithic
    baseline too, by zeroing its slice. Without this the baseline could not
    appear in the damage table at all, and a degradation claim with no
    comparison point is not a claim.
    """
    layout, cursor = {}, 0
    for modality in task.modalities:
        width = int(task.features["train"][modality].shape[1])
        layout[modality] = (cursor, cursor + width)
        cursor += width
    return layout


def _tensors(task: TaskData, split: str, index: np.ndarray | None = None) -> dict:
    """Per-modality tensors, plus the monolithic model's concatenated input.

    THE FLAT VECTOR IS BUILT FROM THE SAME ENCODED FEATURES, not from a
    separate representation. An earlier version handed the baseline clean
    symbolic grid flags while the modular model got lossy ResNet features of a
    rendered image -- so the baseline won on GridNav by holding better
    information, not better structure. Deriving the flat vector here, by
    concatenation, makes the two provably identical in content and different
    only in how that content is organised.
    """
    out: dict[str, torch.Tensor] = {}
    ordered: list[torch.Tensor] = []
    for modality in task.modalities:
        values = task.features[split][modality]
        tensor = torch.from_numpy(
            values if index is None else values[index]
        ).to(DEVICE)
        out[modality] = tensor
        ordered.append(tensor)
    out["__flat__"] = torch.cat(ordered, dim=1)
    return out


def _accepts(task: TaskData, split: str, index: np.ndarray | None = None):
    if task.accepts is None:
        return None
    values = task.accepts[split]
    return torch.from_numpy(values if index is None else values[index]).to(DEVICE)


def accuracy(logits: torch.Tensor, labels: torch.Tensor, accepts=None) -> float:
    """Set accuracy when several answers are optimal, plain accuracy otherwise."""
    picked = logits.argmax(dim=-1)
    if accepts is None:
        return float((picked == labels).float().mean())
    return float(accepts[torch.arange(len(picked)), picked].float().mean())


# ---------------------------------------------------------------------------
# Inference cost
# ---------------------------------------------------------------------------


def estimate_macs(model: nn.Module, batch: dict[str, torch.Tensor]) -> float:
    """Multiply-accumulates for ONE example, counted by hooking real layers.

    Counted rather than derived, because the whole point of routing is that a
    module which does not run costs nothing -- and only an actual forward pass
    knows which modules ran.
    """
    total = 0.0
    handles = []

    def hook(module, inputs, output):
        nonlocal total
        if isinstance(module, nn.Linear):
            total += module.in_features * module.out_features
        elif isinstance(module, nn.Conv1d):
            total += (module.in_channels // module.groups) * module.out_channels * \
                     module.kernel_size[0]
        elif isinstance(module, nn.GRU):
            total += 3 * module.input_size * module.hidden_size * 2
        elif isinstance(module, nn.MultiheadAttention):
            dim = module.embed_dim
            total += 4 * dim * dim

    for m in model.modules():
        if isinstance(m, (nn.Linear, nn.Conv1d, nn.GRU, nn.MultiheadAttention)):
            handles.append(m.register_forward_hook(hook))

    one = {k: v[:1] for k, v in batch.items()}
    with torch.no_grad():
        model(one)
    for h in handles:
        h.remove()
    return total


# ---------------------------------------------------------------------------
# Train / evaluate one configuration
# ---------------------------------------------------------------------------


def run_once(
    task: TaskData,
    variant: str,
    seed: int,
    reference_params: int | None = None,
    epochs: int = 40,
    batch_size: int = 256,
    lr: float = 3e-4,
    weight_decay: float = 1e-4,
    val_fraction: float = 0.15,
    patience: int = 8,
    lesions: tuple[str, ...] = (),
    verbose: bool = False,
) -> RunResult:
    torch.manual_seed(seed)
    np.random.seed(seed)

    n = task.size("train")
    order = np.random.default_rng(seed).permutation(n)
    n_val = int(n * val_fraction)
    val_idx, train_idx = order[:n_val], order[n_val:]

    train_x = _tensors(task, "train", train_idx)
    train_y = torch.from_numpy(task.labels["train"][train_idx]).to(DEVICE)
    train_a = _accepts(task, "train", train_idx)

    val_x = _tensors(task, "train", val_idx)
    val_y = torch.from_numpy(task.labels["train"][val_idx]).to(DEVICE)
    val_a = _accepts(task, "train", val_idx)

    test_x = _tensors(task, "test")
    test_y = torch.from_numpy(task.labels["test"]).to(DEVICE)
    test_a = _accepts(task, "test")

    model = build_model(task, variant, reference_params).to(DEVICE)
    optimiser = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=epochs)

    best_val, best_state, best_epoch, stale = -1.0, None, 0, 0
    started = time.perf_counter()

    for epoch in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(len(train_y))
        for i in range(0, len(perm), batch_size):
            sel = perm[i : i + batch_size]
            batch = {k: v[sel] for k, v in train_x.items()}
            loss = F.cross_entropy(model(batch), train_y[sel])
            optimiser.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimiser.step()
        scheduler.step()

        model.eval()
        with torch.no_grad():
            val_acc = accuracy(model(val_x), val_y, val_a)

        if val_acc > best_val:
            best_val, best_epoch, stale = val_acc, epoch, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
            if stale >= patience:
                break
        if verbose:
            print(f"    epoch {epoch:>3}  val {val_acc:.4f}")

    # Model selection used VALIDATION only. Test is read once, now.
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()

    with torch.no_grad():
        test_acc = accuracy(model(test_x), test_y, test_a)
        lesion_scores: dict[str, float] = {}
        for target in lesions:
            if target.startswith("modality:"):
                name = target.split(":", 1)[1]
                out = model(test_x, drop={name})
            else:
                out = model(test_x, lesion={target})
            lesion_scores[target] = accuracy(out, test_y, test_a)

    return RunResult(
        task=task.name,
        variant=variant,
        seed=seed,
        params=count_params(model),
        val_accuracy=best_val,
        test_accuracy=test_acc,
        epochs_ran=best_epoch,
        train_seconds=time.perf_counter() - started,
        lesion=lesion_scores,
        macs=estimate_macs(model, test_x),
    )


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------


def paired_test(a: list[float], b: list[float]) -> dict[str, float]:
    """Paired comparison of two variants across shared seeds.

    Paired because seed k means the same split and the same initialisation for
    both models, so the seed-to-seed variance cancels. Reports Cohen's d_z
    alongside p, because with five seeds a p-value alone says very little.
    """
    if len(a) != len(b) or len(a) < 2:
        return {"n": len(a), "mean_diff": float("nan"), "p": float("nan"), "d": float("nan")}

    diff = np.array(a, dtype=float) - np.array(b, dtype=float)
    mean = float(diff.mean())
    sd = float(diff.std(ddof=1))
    if sd < 1e-12:
        return {"n": len(diff), "mean_diff": mean, "p": 0.0 if mean else 1.0,
                "d": float("inf") if mean else 0.0}

    t = mean / (sd / math.sqrt(len(diff)))
    try:
        from scipy import stats

        p = float(2 * stats.t.sf(abs(t), df=len(diff) - 1))
    except Exception:
        p = float("nan")
    return {"n": len(diff), "mean_diff": mean, "p": p, "d": mean / sd}


def summarise(results: list[RunResult]) -> dict:
    """Group runs by (task, variant) and reduce to mean/std."""
    table: dict[tuple[str, str], list[RunResult]] = {}
    for r in results:
        table.setdefault((r.task, r.variant), []).append(r)

    out = {}
    for (task, variant), runs in table.items():
        accs = [r.test_accuracy for r in runs]
        entry = {
            "n_seeds": len(runs),
            "params": runs[0].params,
            "macs": runs[0].macs,
            "test_mean": float(np.mean(accs)),
            "test_std": float(np.std(accs, ddof=1)) if len(accs) > 1 else 0.0,
            "val_mean": float(np.mean([r.val_accuracy for r in runs])),
            "seconds_mean": float(np.mean([r.train_seconds for r in runs])),
            "accuracies": accs,
        }
        lesion_keys = runs[0].lesion.keys()
        entry["lesion"] = {
            k: {
                "mean": float(np.mean([r.lesion[k] for r in runs])),
                "std": float(np.std([r.lesion[k] for r in runs], ddof=1))
                if len(runs) > 1 else 0.0,
            }
            for k in lesion_keys
        }
        out[f"{task}/{variant}"] = entry
    return out


def save_results(results: list[RunResult], name: str = "results") -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / f"{name}.json"
    payload = {
        "runs": [asdict(r) for r in results],
        "summary": summarise(results),
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


__all__ = [
    "RunResult",
    "run_once",
    "summarise",
    "paired_test",
    "save_results",
    "build_model",
    "estimate_macs",
    "flat_layout",
    "RESULTS_DIR",
]
