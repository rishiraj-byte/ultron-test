"""
lobes.py -- the four cortical lobes as four real, separate nn.Modules.

Every lobe is LATENT_DIM in, LATENT_DIM out (PROJECT.md section 3), with one
documented exception: FrontalLobe, which fuses first. See section 4.

None of these are pretrained. The lobes are yours; only the encoders are
borrowed.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

import config
from modules.laminar import LaminarShell



def sinusoidal_positions(n: int, dim: int, device) -> torch.Tensor:
    """Standard sinusoidal position codes, (1, n, dim).

    Needed because self-attention is permutation-EQUIVARIANT and a mean over
    positions afterwards makes the whole operation permutation-INVARIANT.
    Without position codes the 49 image patches are a BAG of features, not a
    grid, and "left of" remains inexpressible -- which is precisely the
    capability parietal exists to provide. Measured before this was added:
    shuffling all 49 positions changed parietal's output by exactly 0.0000.

    Sinusoidal rather than a learned table so any N works -- 49 image patches,
    8 sensor beams, 64 audio frames -- with no size cap to maintain.
    """
    position = torch.arange(n, device=device, dtype=torch.float32).unsqueeze(1)
    index = torch.arange(0, dim, 2, device=device, dtype=torch.float32)
    freq = torch.exp(-math.log(10000.0) * index / dim)
    code = torch.zeros(n, dim, device=device)
    code[:, 0::2] = torch.sin(position * freq)
    code[:, 1::2] = torch.cos(position * freq)[:, : code[:, 1::2].shape[1]]
    return code.unsqueeze(0)


class Lobe(nn.Module):
    #  Overridden by each real lobe; selects that lobe's pretrained teacher.
    LOBE_NAME = "generic"

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
        # THE OPERATOR. Where a matching pretrained network exists, that IS
        # the operator -- the random MLP below is not built at all, so it
        # cannot sit unused and dark in the gradient count. See
        # modules/pretrained_lobes.py for which teacher each lobe gets and why.
        self.core = None
        if config.PRETRAINED_LOBES and self.LOBE_NAME != "generic":
            from modules.pretrained_lobes import build_core
            self.core = build_core(self.LOBE_NAME, dim)
        self.net = None if self.core is not None else nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, dim),
        )
        # Stage 4: the shared cortical microcircuit every area has. The lobe's
        # own operator above is what makes it THIS area; the shell is what
        # makes it cortex.
        self.laminar = LaminarShell(dim=dim, dropout=dropout)
        #  set per timestep by the settling loop; None = resting sparsity
        self.sparsity_now: float | None = None

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

    def operate(self, x: torch.Tensor) -> torch.Tensor:
        """Run the lobe's operator. Accepts (B, dim) or (B, N, dim)."""
        return self.core(x) if self.core is not None else self.net(x)

    def sequence_contribution(self, sequence: torch.Tensor) -> torch.Tensor:
        """What this lobe makes of its sensory sequence, as (B, dim).

        Overridden by the lobes that have a sequence operator. It exists as
        one name so the settling loop can compute this ONCE per percept: the
        sensory sequence does not change while cortex settles, so running the
        operator again at every timestep recomputes an identical tensor. With
        pretrained cores that redundancy dominated the whole forward pass --
        twelve ResNet and transformer passes per lobe per percept instead of
        one. The cached value is bit-identical, so gradients are unaffected.
        """
        raise NotImplementedError(
            "%s has no sequence operator" % type(self).__name__
        )

    @property
    def reads_sequences(self) -> bool:
        return type(self).sequence_contribution is not Lobe.sequence_contribution

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._check(x)
        return x + self.laminar.l23(self.operate(self.norm(x)), self.sparsity_now)


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

    LOBE_NAME = "occipital"

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
        #  Only when there is no pretrained convolution to use instead. The
        #  ResNet core is itself a convolution, so the inductive bias this
        #  grouped stack expressed is preserved rather than traded away --
        #  it is the same operation, with ImageNet weights behind it.
        self.net = None if self.core is not None else nn.Sequential(
            nn.Conv1d(dim, hidden, kernel_size=1, groups=groups),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(hidden, dim, kernel_size=1, groups=groups),
        )
        self.mix = nn.Linear(dim, dim)      # cross-band integration

    def forward(self, x: torch.Tensor,
                sequence: torch.Tensor | None = None) -> torch.Tensor:
        x = self._check(x)
        h = self.operate(self.norm(x))
        out = x + self.laminar.l23(self.mix(h), self.sparsity_now)
        if sequence is not None:
            out = out + (sequence if sequence.dim() == 2
                         else self.receptive_fields(sequence))
        return out

    def sequence_contribution(self, sequence: torch.Tensor) -> torch.Tensor:
        return self.receptive_fields(sequence)

    def receptive_fields(self, sequence: torch.Tensor) -> torch.Tensor:
        """Grouped local filtering at EVERY real spatial position.

        (B, N, D) -> grouped conv independently per position -> pooled (B, D).

        This is what the grouped operator was always for. In v1 it ran on a
        single pooled vector, so "local receptive field" described bands of an
        abstract 256-vector rather than regions of an image. Now N is 49 actual
        patches of the ResNet grid and locality means what it says.
        """
        b, n, d = sequence.shape
        normed = self.norm(sequence.reshape(b * n, d)).reshape(b, n, d)
        filtered = self.operate(normed)
        return self.mix(filtered.mean(dim=1))


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

    LOBE_NAME = "parietal"

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
        # Attention over REAL sensory positions (stage 2), separate from the
        # slot attention above which operates on the recurrent state.
        #  Not built when a pretrained attention layer is available: MiniLM's
        #  encoder layer IS attention between positions, which is the operation
        #  this module existed to perform. Keeping both would leave one of them
        #  without a gradient and make the ablation table claim a pathway that
        #  does nothing.
        self.grid_attn = None if self.core is not None else nn.MultiheadAttention(
            embed_dim=dim, num_heads=4, dropout=dropout, batch_first=True
        )
        self.grid_norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor,
                sequence: torch.Tensor | None = None) -> torch.Tensor:
        x = self._check(x)
        b = x.shape[0]
        # (B, dim) -> (B, positions, slot): the latent as a row of places
        grid = self.attn_norm(self.norm(x).view(b, self.positions, self.slot))
        attended, _ = self.attn(grid, grid, grid, need_weights=False)
        x = x + attended.reshape(b, self.dim)
        if sequence is not None:
            x = x + (sequence if sequence.dim() == 2
                     else self.spatial_attention(sequence))
        return x + self.laminar.l23(self.operate(self.norm(x)), self.sparsity_now)

    def sequence_contribution(self, sequence: torch.Tensor) -> torch.Tensor:
        return self.spatial_attention(sequence)

    def spatial_attention(self, sequence: torch.Tensor) -> torch.Tensor:
        """Attend across REAL positions -- the 49 patches of the ResNet grid.

        This is the operator finally getting the data it was built for. In v1
        it attended over 16 arbitrary slices of a pooled vector, which carry no
        spatial relationship at all; "left of" was not expressible because
        there was no left. Here position j genuinely is a different place in
        the image from position k.
        """
        # Position codes FIRST, or the whole operation is permutation-invariant
        # and the arrangement it is supposed to read is invisible to it.
        placed = sequence + sinusoidal_positions(
            sequence.shape[1], sequence.shape[2], sequence.device
        )
        if self.core is not None:
            #  A pretrained transformer layer is self-attention plus its own
            #  feed-forward block; the position codes above are still required,
            #  because attention remains permutation-equivariant no matter how
            #  well trained it is.
            attended = self.core(placed)
        else:
            attended, _ = self.grid_attn(placed, placed, placed, need_weights=False)
        return self.grid_norm(attended).mean(dim=1)


class TemporalLobe(Lobe):
    """Meaning, language, memory match. Fed by the thalamus (MGN / text route).

    SEQUENTIAL. The latent is read as `steps` time steps and run through a GRU,
    so order is carried rather than discarded. Both signals routed here --
    language and audio -- are sequences; this is the only lobe that has any
    notion of "before" and "after".

    The GRU's hidden state is also what a memory-match role would need, and is
    where a hippocampal module would attach if one is ever built.
    """

    LOBE_NAME = "temporal"

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
        # GRU over the REAL sensory sequence (stage 2), separate from the
        # slot-recurrence above which operates on the recurrent state.
        self.seq_rnn = nn.GRU(input_size=dim, hidden_size=dim, batch_first=True)
        self.seq_norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor,
                sequence: torch.Tensor | None = None) -> torch.Tensor:
        x = self._check(x)
        b = x.shape[0]
        seq = self.norm(x).view(b, self.steps, self.step_dim)
        out, _ = self.rnn(seq)
        x = x + out.reshape(b, self.dim)
        if sequence is not None:
            x = x + (sequence if sequence.dim() == 2
                     else self.read_sequence(sequence))
        return x + self.laminar.l23(self.operate(self.norm(x)), self.sparsity_now)

    def sequence_contribution(self, sequence: torch.Tensor) -> torch.Tensor:
        return self.read_sequence(sequence)

    def read_sequence(self, sequence: torch.Tensor) -> torch.Tensor:
        """Run the REAL token or frame sequence through a GRU.

        v1 chopped one pooled vector into 8 pretend "timesteps" whose order was
        an artefact of how the dimensions happened to be laid out. Here the
        sequence is genuinely temporal: MiniLM tokens in reading order, or
        Whisper frames in time order. Reversing it now changes the answer,
        which is the whole point of using a recurrent operator.
        """
        #  The GRU SURVIVES the pretrained rewrite, and deliberately. There is
        #  no pretrained GRU to drop in here, and recurrence is temporal's
        #  inductive bias -- the thing that makes it a different network from
        #  parietal rather than a second attention stack with another name.
        #  So the pretrained Whisper layer, which was trained on exactly this
        #  kind of frame sequence, provides the features and the GRU still
        #  does the recurring over them.
        prepared = self.core(sequence) if self.core is not None else sequence
        out, hidden = self.seq_rnn(prepared)
        return self.seq_norm(hidden[-1])


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

    LOBE_NAME = "frontal"

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
        #  Frontal's own sequence operator -- see `select` for why it is a
        #  relevance gate and not a fourth copy of somebody else's operator.
        self.relevance = nn.Linear(dim, 1)
        self.seq_norm = nn.LayerNorm(dim)

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
        return fused + self.laminar.l23(self.operate(self.norm(fused)),
                                        self.sparsity_now)

    def sequence_contribution(self, sequence: torch.Tensor) -> torch.Tensor:
        return self.select(sequence)

    def select(self, sequence: torch.Tensor) -> torch.Tensor:
        """Score every position for relevance and integrate by that score.

        WHY FRONTAL READS THE INPUT AT ALL. It did not, until the routing
        table was measured: on a text input only temporal received the
        sequence, and occipital, parietal and frontal saw nothing but echoes
        arriving over the tracts. Three of four networks were not parsing the
        input. Prefrontal cortex is not sensory-blind in a real brain either
        -- the mediodorsal thalamus projects to it directly -- so the fix is
        anatomically ordinary as well as necessary.

        WHY THIS OPERATOR AND NOT ANOTHER. The claim that the lobes are four
        different networks only survives if each reads a sequence its own way,
        so a fourth attention stack or a second GRU here would quietly falsify
        the architecture. Occipital filters locally, parietal compares
        positions, temporal carries order. Frontal does the thing prefrontal
        cortex is characterised by: it decides which parts of the input are
        worth holding. A learned relevance score per position, softmaxed into
        weights, integrates the sequence by task salience rather than by
        proximity, arrangement or order. Selection, not comparison.

        The score is computed from the Qwen features, so the pretrained
        language model contributes here too -- but it contributes FEATURES.
        What gets kept is decided by a head this system trained itself.
        """
        placed = sequence + sinusoidal_positions(
            sequence.shape[1], sequence.shape[2], sequence.device
        )
        features = self.core(placed) if self.core is not None else placed
        features = self.seq_norm(features)
        weights = self.relevance(features).softmax(dim=1)
        return (features * weights).sum(dim=1)


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
