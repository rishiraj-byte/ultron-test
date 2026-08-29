"""
lobes.py -- the four cortical lobes as four real, separate nn.Modules.

Every lobe is LATENT_DIM in, LATENT_DIM out (PROJECT.md section 3), with one
documented exception: FrontalLobe, which fuses first. See section 4.

None of these are pretrained. The lobes are yours; only the encoders are
borrowed.
"""

from __future__ import annotations

import torch
import torch.nn as nn

import config


class Lobe(nn.Module):
    """LATENT_DIM -> LOBE_HIDDEN_DIM -> LATENT_DIM, residual.

    Pre-norm + residual is not decoration: four of these run in series on the
    image path (occipital -> parietal -> frontal, plus fusion), and a plain
    stacked MLP that deep drifts in scale and trains badly.
    """

    role = "generic"

    def __init__(
        self,
        dim: int = config.LATENT_DIM,
        hidden: int = config.LOBE_HIDDEN_DIM,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.dim = dim
        self.norm = nn.LayerNorm(dim)
        self.net = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, dim),
        )

    @property
    def device(self) -> torch.device:
        return self.norm.weight.device

    def _check(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 1:
            x = x.unsqueeze(0)
        if x.shape[-1] != self.dim:
            raise ValueError(
                f"{type(self).__name__} expected last dim {self.dim}, "
                f"got {tuple(x.shape)}"
            )
        # Accept input from wherever the caller had it. A module knows where it
        # lives; making every caller track that too is how a GPU migration
        # turns into a hunt through every test and script for stray CPU
        # tensors. Moving an already-correct tensor is a no-op.
        return x.to(self.device) if x.device != self.device else x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._check(x)
        return x + self.net(self.norm(x))


# ---------------------------------------------------------------------------
# The four lobes are NOT four copies of the same MLP.
#
# This is the load-bearing claim of the whole architecture, so it has to be
# true in the code and not only in the variable names. If occipital, parietal
# and temporal were the same class with different labels, the honest
# description of this system would be "a 3-layer MLP with a routing table",
# and every result would be attributable to capacity rather than structure.
#
# Each lobe therefore carries an inductive bias matched to the structure of
# the signal it receives:
#
#   occipital  local, retinotopic     -> GROUPED connectivity. A unit sees one
#                                        band of the input, not all of it, the
#                                        way a receptive field is local.
#   parietal   relational, "where"    -> SELF-ATTENTION over the latent read as
#                                        a grid of positions. Spatial relations
#                                        are comparisons BETWEEN positions,
#                                        which a per-unit MLP cannot express.
#   temporal   sequential, "what"     -> RECURRENT. Language and audio arrive as
#                                        sequences; a GRU carries order, an MLP
#                                        discards it.
#   frontal    convergence            -> FUSION over several streams (section 4).
#
# All four still honour the LATENT_DIM in / LATENT_DIM out contract, so they
# remain interchangeable at the wiring level and every ablation stays valid.
# ---------------------------------------------------------------------------


class OccipitalLobe(Lobe):
    """Vision. Fed by the thalamus (LGN route).

    LOCAL RECEPTIVE FIELDS. The hidden layer is grouped: the latent is split
    into `groups` bands and each band is transformed by its own weights, so a
    hidden unit reads one region of the representation rather than all of it.
    That is the defining property of early visual cortex, and it is what a
    plain dense layer throws away. A final mixing layer lets the bands talk,
    the way V1 feeds forward into larger receptive fields downstream.

    Delivers to parietal AND to frontal directly -- that is why frontal has to
    fuse two streams on the image path (PROJECT.md sections 2.1 and 4).
    """

    role = "local feature extraction (vision)"

    def __init__(
        self,
        dim: int = config.LATENT_DIM,
        hidden: int = config.LOBE_HIDDEN_DIM,
        dropout: float = 0.0,
        groups: int = config.OCCIPITAL_GROUPS,
    ):
        super().__init__(dim=dim, hidden=hidden, dropout=dropout)
        if dim % groups or hidden % groups:
            raise ValueError(
                f"OccipitalLobe: dim={dim} and hidden={hidden} must both divide "
                f"by groups={groups}"
            )
        self.groups = groups
        # Conv1d with groups=g over a length-1 sequence IS a grouped linear
        # layer, and is the cheapest way to express one in PyTorch.
        self.net = nn.Sequential(
            nn.Conv1d(dim, hidden, kernel_size=1, groups=groups),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(hidden, dim, kernel_size=1, groups=groups),
        )
        self.mix = nn.Linear(dim, dim)      # cross-band integration

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._check(x)
        h = self.net(self.norm(x).unsqueeze(-1)).squeeze(-1)
        return x + self.mix(h)


class ParietalLobe(Lobe):
    """Space. Fed by OCCIPITAL, never by the thalamus (PROJECT.md section 2.3).

    RELATIONAL. The latent is read as a grid of `positions` slots, and
    single-head self-attention lets every position attend to every other. A
    spatial relation ("left of", "above", "nearer than") is a comparison
    between two positions; an elementwise MLP has no operator for that, which
    is precisely why the original spec could not say what parietal was for.

    This is the lobe whose usefulness PROJECT.md section 11 left open. Giving
    it an operator that the other lobes lack is what makes the ablation
    meaningful: removing it now removes a capability, not just parameters.
    """

    role = "spatial relations (attention over positions)"

    def __init__(
        self,
        dim: int = config.LATENT_DIM,
        hidden: int = config.LOBE_HIDDEN_DIM,
        dropout: float = 0.0,
        positions: int = config.PARIETAL_POSITIONS,
    ):
        super().__init__(dim=dim, hidden=hidden, dropout=dropout)
        if dim % positions:
            raise ValueError(
                f"ParietalLobe: dim={dim} must divide by positions={positions}"
            )
        self.positions = positions
        self.slot = dim // positions
        self.attn = nn.MultiheadAttention(
            embed_dim=self.slot, num_heads=1, dropout=dropout, batch_first=True
        )
        self.attn_norm = nn.LayerNorm(self.slot)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._check(x)
        b = x.shape[0]
        # (B, dim) -> (B, positions, slot): the latent as a row of places
        grid = self.attn_norm(self.norm(x).view(b, self.positions, self.slot))
        attended, _ = self.attn(grid, grid, grid, need_weights=False)
        x = x + attended.reshape(b, self.dim)
        return x + self.net(self.norm(x))


class TemporalLobe(Lobe):
    """Meaning, language, memory match. Fed by the thalamus (MGN / text route).

    SEQUENTIAL. The latent is read as `steps` time steps and run through a GRU,
    so order is carried rather than discarded. Both signals routed here --
    language and audio -- are sequences; this is the only lobe that has any
    notion of "before" and "after".

    The GRU's hidden state is also what a memory-match role would need, and is
    where a hippocampal module would attach if one is ever built.
    """

    role = "meaning, language, memory (recurrent)"

    def __init__(
        self,
        dim: int = config.LATENT_DIM,
        hidden: int = config.LOBE_HIDDEN_DIM,
        dropout: float = 0.0,
        steps: int = config.TEMPORAL_STEPS,
    ):
        super().__init__(dim=dim, hidden=hidden, dropout=dropout)
        if dim % steps:
            raise ValueError(
                f"TemporalLobe: dim={dim} must divide by steps={steps}"
            )
        self.steps = steps
        self.step_dim = dim // steps
        self.rnn = nn.GRU(
            input_size=self.step_dim,
            hidden_size=self.step_dim,
            batch_first=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._check(x)
        b = x.shape[0]
        seq = self.norm(x).view(b, self.steps, self.step_dim)
        out, _ = self.rnn(seq)
        x = x + out.reshape(b, self.dim)
        return x + self.net(self.norm(x))


class FrontalLobe(Lobe):
    """Convergence, fusion, decision.

    THE RESOLVED SHAPE CONFLICT (PROJECT.md section 4).

    Section 3 says every lobe is LATENT_DIM in / out. Section 2.1 has frontal
    receiving from parietal AND occipital at once on the image path. Two
    LATENT_DIM streams into a LATENT_DIM input is a mismatch -- it would crash
    on the first forward pass.

    Resolution: fuse first, then behave like every other lobe.

        n streams x LATENT_DIM -> [ fusion ] -> LATENT_DIM -> [ lobe MLP ] -> LATENT_DIM

    Streams arrive as a dict keyed by config.FRONTAL_STREAM_SLOTS. Slot order
    is FIXED and missing streams are zero-padded into their own slot, so the
    fusion layer sees the same meaning in the same position regardless of
    modality. Text sends one stream; an image sends two.

    THIS IS A DESIGN CHOICE. "concat" keeps which stream contributed what at
    the cost of parameters; "sum" is free but discards provenance. Flip
    config.FRONTAL_FUSION_MODE to change the tradeoff.
    """

    role = "convergence, fusion, decision"

    def __init__(
        self,
        dim: int = config.LATENT_DIM,
        hidden: int = config.LOBE_HIDDEN_DIM,
        slots: tuple[str, ...] = config.FRONTAL_STREAM_SLOTS,
        mode: str = config.FRONTAL_FUSION_MODE,
        dropout: float = 0.0,
    ):
        super().__init__(dim=dim, hidden=hidden, dropout=dropout)
        if mode not in ("concat", "sum"):
            raise ValueError(f"unknown FRONTAL_FUSION_MODE {mode!r}")
        self.slots = tuple(slots)
        self.mode = mode
        self.fusion_norm = nn.LayerNorm(dim)
        self.fuse = (
            nn.Linear(len(self.slots) * dim, dim) if mode == "concat" else nn.Identity()
        )
        self._last_slots: dict[str, bool] = {s: False for s in self.slots}

    @property
    def last_active_slots(self) -> dict[str, bool]:
        """Which slots carried real signal last pass. For the dashboard."""
        return dict(self._last_slots)

    def fuse_streams(self, streams: dict[str, torch.Tensor]) -> torch.Tensor:
        unknown = set(streams) - set(self.slots)
        if unknown:
            raise KeyError(
                f"frontal has no slot for {sorted(unknown)}; slots are {self.slots}"
            )
        if not streams:
            raise ValueError("frontal received no streams")

        any_stream = next(iter(streams.values())).to(self.device)
        batch = any_stream.shape[0] if any_stream.dim() > 1 else 1
        ref = any_stream if any_stream.dim() > 1 else any_stream.unsqueeze(0)

        device = self.device
        ordered: list[torch.Tensor] = []
        for slot in self.slots:
            value = streams.get(slot)
            if value is not None and value.device != device:
                value = value.to(device)
            self._last_slots[slot] = value is not None
            if value is None:
                # Zero-pad the unused slot. Its position still means "parietal".
                ordered.append(torch.zeros(batch, self.dim, dtype=ref.dtype,
                                           device=ref.device))
                continue
            if value.dim() == 1:
                value = value.unsqueeze(0)
            if value.shape[-1] != self.dim:
                raise ValueError(
                    f"frontal slot {slot!r} expected last dim {self.dim}, "
                    f"got {tuple(value.shape)}"
                )
            ordered.append(value)

        if self.mode == "sum":
            fused = torch.stack(ordered, dim=0).sum(dim=0)
        else:
            fused = self.fuse(torch.cat(ordered, dim=-1))
        return self.fusion_norm(fused)

    def forward(self, streams: dict[str, torch.Tensor] | torch.Tensor) -> torch.Tensor:
        if isinstance(streams, torch.Tensor):
            # Tolerate a single pre-fused tensor so frontal can still be driven
            # like a plain lobe in isolation (verify.py leans on this).
            fused = self._check(streams)
        else:
            fused = self.fuse_streams(streams)
        return super().forward(fused)


class Cortex(nn.Module):
    """The four lobes as one addressable unit."""

    def __init__(self, dropout: float = 0.0):
        super().__init__()
        self.occipital = OccipitalLobe(dropout=dropout)
        self.parietal = ParietalLobe(dropout=dropout)
        self.temporal = TemporalLobe(dropout=dropout)
        self.frontal = FrontalLobe(dropout=dropout)

    def __getitem__(self, name: str) -> Lobe:
        try:
            return getattr(self, name)
        except AttributeError as exc:
            raise KeyError(f"no lobe named {name!r}; known: {config.LOBES}") from exc


__all__ = [
    "Lobe",
    "OccipitalLobe",
    "ParietalLobe",
    "TemporalLobe",
    "FrontalLobe",
    "Cortex",
]
