"""
baselines.py -- the models the modular architecture has to beat.

A claim that "specialised modules beat one monolithic network" is only worth
printing if the comparison is fair, so every baseline here is built to remove
exactly ONE property of the full system while holding everything else equal.

    Monolithic      no modules at all. One MLP over the concatenated input.
                    Removes: modularity, routing, specialisation.

    NoRouting       the same four lobes, but every modality is broadcast to
                    every lobe instead of being relayed to one.
                    Removes: thalamic routing only.

    SharedLobe      one lobe instance reused for all streams.
                    Removes: specialisation only. Same depth, same routing.

    Homogeneous     four separate lobes, but all four are the plain MLP block.
                    Removes: the per-lobe inductive biases only.

The last three are the ones that matter. Monolithic-vs-modular confounds three
changes at once, so on its own it could never tell you WHICH property helped.
Reviewers ask exactly that question, so the ablations are the real experiment
and the monolithic number is context.

PARAMETER MATCHING. Every baseline is sized to the modular system's trainable
parameter count with `match_width`, because a capacity difference would
explain any result on its own. Matched counts are reported in the results
table so the reader can check.
"""

from __future__ import annotations

import torch
import torch.nn as nn

import config
from modules.lobes import FrontalLobe, Lobe


def match_width(in_dim: int, out_dim: int, target_params: int, depth: int = 3) -> int:
    """Hidden width that gives an MLP roughly `target_params` parameters.

    Solved rather than guessed: for a depth-d MLP the count is
        in*h + h + (d-2)*(h*h + h) + h*out + out
    which is quadratic in h. Binary search is simpler than the quadratic
    formula and avoids sign mistakes.
    """
    def count(h: int) -> int:
        n = in_dim * h + h                       # first layer
        n += (depth - 2) * (h * h + h)           # middle layers
        n += h * out_dim + out_dim               # head
        return n

    lo, hi = 1, 8192
    while lo < hi:
        mid = (lo + hi) // 2
        if count(mid) < target_params:
            lo = mid + 1
        else:
            hi = mid
    return max(lo, 1)


class MonolithicMLP(nn.Module):
    """One network, no modules. The thing the architecture claims to beat.

    Receives the concatenation of every modality, so it is not handicapped on
    information -- only on structure.
    """

    name = "monolithic"

    def __init__(
        self,
        in_dim: int,
        num_classes: int,
        target_params: int,
        depth: int = 3,
        dropout: float = 0.0,
    ):
        super().__init__()
        hidden = match_width(in_dim, num_classes, target_params, depth)
        layers: list[nn.Module] = [nn.LayerNorm(in_dim), nn.Linear(in_dim, hidden), nn.GELU()]
        for _ in range(depth - 2):
            layers += [nn.Linear(hidden, hidden), nn.GELU(), nn.Dropout(dropout)]
        layers += [nn.Linear(hidden, num_classes)]
        self.net = nn.Sequential(*layers)
        self.hidden = hidden

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ModularStack(nn.Module):
    """The modular system, reduced to what a benchmark needs.

    Deliberately NOT the full BrainAI: no dashboard, no activation capture, no
    encoders. It takes already-projected LATENT_DIM streams per modality and
    reproduces the routing, the lobes, the frontal fusion and the gate. Keeping
    it small is what makes ablations honest -- each variant differs from this
    class by one flag and nothing else.

    variant:
        "modular"      routing + specialised lobes            (the proposal)
        "no_routing"   every modality broadcast to every lobe (routing removed)
        "shared_lobe"  one lobe instance for all streams      (modules removed)
        "homogeneous"  four plain MLP lobes                   (biases removed)
    """

    def __init__(
        self,
        num_classes: int,
        variant: str = "modular",
        dim: int = config.LATENT_DIM,
        dropout: float = 0.0,
        use_gate: bool = True,
        lobe_hidden: int | None = None,
    ):
        """`lobe_hidden` widens or narrows the ablated lobes to hold capacity
        constant. Without it the ablations are confounded: sharing one lobe
        removes two thirds of the lobe parameters, so a drop in accuracy could
        be capacity rather than lost specialisation, and a reviewer would be
        right to say so. `matched_hidden()` computes the value."""
        super().__init__()
        if variant not in ("modular", "no_routing", "shared_lobe", "homogeneous"):
            raise ValueError(f"unknown variant {variant!r}")
        self.variant = variant
        self.dim = dim
        self.use_gate = use_gate
        hidden = lobe_hidden or config.LOBE_HIDDEN_DIM

        if variant == "shared_lobe":
            shared = Lobe(dim=dim, hidden=hidden, dropout=dropout)
            self.lobes = nn.ModuleDict(
                {name: shared for name in ("occipital", "parietal", "temporal")}
            )
        elif variant == "homogeneous":
            self.lobes = nn.ModuleDict(
                {name: Lobe(dim=dim, hidden=hidden, dropout=dropout)
                 for name in ("occipital", "parietal", "temporal")}
            )
        else:
            from modules.lobes import OccipitalLobe, ParietalLobe, TemporalLobe

            self.lobes = nn.ModuleDict({
                "occipital": OccipitalLobe(dim=dim, dropout=dropout),
                "parietal": ParietalLobe(dim=dim, dropout=dropout),
                "temporal": TemporalLobe(dim=dim, dropout=dropout),
            })

        self.frontal = FrontalLobe(dim=dim, dropout=dropout)
        self.gate = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, 1))
        self.head = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, num_classes))

        #  modality -> lobe, mirroring config.THALAMIC_ROUTES
        self.routes = {m: d for m, (_, d) in config.THALAMIC_ROUTES.items()}

    def forward(
        self,
        streams: dict[str, torch.Tensor],
        lesion: set[str] | None = None,
    ) -> torch.Tensor:
        """`streams` maps modality -> (B, dim). `lesion` names lobes to disable.

        A lesioned lobe contributes nothing -- its frontal slot is zero-padded,
        exactly as a missing modality already is. That is what makes damage
        tolerance measurable instead of hypothetical.
        """
        lesion = lesion or set()
        any_stream = next(iter(streams.values()))
        batch = any_stream.shape[0]

        # -- route inputs to lobes -------------------------------------
        inbox: dict[str, list[torch.Tensor]] = {}
        for modality, value in streams.items():
            if self.variant == "no_routing":
                targets = list(self.lobes.keys())      # broadcast: no switchboard
            else:
                target = self.routes.get(modality)
                targets = [target] if target in self.lobes else []
            for t in targets:
                inbox.setdefault(t, []).append(value)

        # -- run each lobe that received something ---------------------
        outputs: dict[str, torch.Tensor] = {}
        for name in ("occipital", "parietal", "temporal"):
            if name in lesion:
                continue
            sources = list(inbox.get(name, []))
            # Dorsal stream: parietal also receives occipital's output.
            if name == "parietal" and "occipital" in outputs:
                sources.append(outputs["occipital"])
            if not sources:
                continue
            driven = sources[0] if len(sources) == 1 else torch.stack(sources).mean(0)
            outputs[name] = self.lobes[name](driven)

        if not outputs:
            # Every route lesioned. Return zeros rather than crashing, so a
            # full-lesion row can appear in the table.
            fused = torch.zeros(batch, self.dim, device=any_stream.device)
        else:
            fused = self.frontal(outputs)

        if self.use_gate:
            fused = fused * torch.sigmoid(self.gate(fused))
        return self.head(fused)

    def lobe_params(self, name: str) -> int:
        return sum(p.numel() for p in self.lobes[name].parameters())


class ModalityProjections(nn.Module):
    """Per-modality raw -> LATENT_DIM projections, shared by every variant.

    Held outside the stack on purpose: all models get the same projection
    machinery, so the comparison isolates what happens AFTER the input is in a
    common space. Otherwise a win could just be a better input adapter.
    """

    def __init__(self, raw_dims: dict[str, int], dim: int = config.LATENT_DIM):
        super().__init__()
        self.proj = nn.ModuleDict({
            name: nn.Sequential(nn.Linear(d, dim), nn.LayerNorm(dim))
            for name, d in raw_dims.items()
        })

    def forward(self, raw: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return {name: self.proj[name](value) for name, value in raw.items()
                if name in self.proj}



def matched_hidden(
    variant: str,
    num_classes: int,
    target: int,
    dim: int = config.LATENT_DIM,
    build=None,
) -> int:
    """Lobe hidden width that brings `variant` to `target` trainable params.

    Searched rather than derived, because the lobe blocks differ structurally
    (grouped conv, attention, GRU) and there is no single closed form.

    `build` must construct the SAME object that will actually be trained. An
    earlier version searched over a bare ModularStack while the target came
    from a full model including per-modality projections, so the two counts
    were measuring different things and the "matched" ablations ended up 18%
    larger than the model they were meant to match -- which would have handed
    them an unearned advantage in every table.
    """
    def make(hidden: int):
        if build is not None:
            return build(hidden)
        return ModularStack(num_classes, variant=variant, dim=dim, lobe_hidden=hidden)

    best, best_gap = config.LOBE_HIDDEN_DIM, None
    for hidden in range(64, 2049, 16):
        try:
            model = make(hidden)
        except ValueError:
            continue
        gap = abs(count_params(model) - target)
        if best_gap is None or gap < best_gap:
            best, best_gap = hidden, gap
    return best


def count_params(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


__all__ = [
    "MonolithicMLP",
    "ModularStack",
    "ModalityProjections",
    "match_width",
    "matched_hidden",
    "count_params",
]
