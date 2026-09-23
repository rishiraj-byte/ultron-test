"""
laminar.py -- cortical layers, sparsity and E/I balance (spec 3.1, stage 4).

WHY A LOBE BECOMES THREE SUBLAYERS.

Until now a lobe was one block that summed everything arriving and applied its
operator. Real cortex does not do that: where a signal ARRIVES determines what
happens to it.

    thalamic drive ──► L4      thalamorecipient. Sensory input lands here and
                       │       nowhere else.
    lateral      ────► L2/3    the lobe's own operator lives here, and this is
                       │       what projects FORWARD to the next area.
    feedback     ────► L5/6    deep layers. Feedback lands here, and L6 is the
                               source of the corticothalamic projection.

That separation is functional, not decorative. Feedforward and feedback signals
arriving at different layers is what lets a prediction be compared against
evidence rather than merely added to it -- and it is why the corticothalamic
projection can be driven by feedback state rather than by sensory drive.

K-WTA SPARSITY. Cortex is sparse: a few percent of neurons active at once, not
half of them. Enforced here as a hard k-winners-take-all mask -- keep the top
k%, zero the rest. This is not regularisation-by-hope like L1; it is a
guarantee, and `verify_v2.py` measures the achieved rate rather than trusting
the setting.

E/I SPLIT. Roughly 80% of cortical neurons are excitatory and 20% inhibitory,
and inhibitory ones cannot excite. A plain Linear violates this: its weights
take either sign freely, so one "neuron" both excites and inhibits depending on
the target. Here the two populations are separate, their weights are held
non-negative by construction, and inhibition is SUBTRACTED. That makes
divisive/subtractive normalisation a structural property rather than something
the optimiser might discover.
"""

from __future__ import annotations

import torch
import torch.nn as nn

import config


def k_winners(
    x: torch.Tensor,
    fraction: float,
    incumbent: torch.Tensor | None = None,
    hysteresis: float | None = None,
) -> torch.Tensor:
    """Keep the top `fraction` of units per example; zero the rest.

    A hard mask rather than a penalty. An L1 term makes sparsity something the
    optimiser trades against the task loss and usually loses; this makes it a
    property of the forward pass that cannot be traded away.

    Gradient flows only through the survivors, which is also how it works in a
    winner-take-all circuit: units that lost the competition did not
    participate, so they receive no credit.
    """
    if fraction >= 1.0:
        return x
    k = max(1, int(round(x.shape[-1] * fraction)))

    # HYSTERESIS -- why the incumbent gets a head start.
    #
    # Plain k-WTA has no memory, so two units within a hair of each other
    # trade places on any perturbation. Inside a settling loop that is fatal:
    # measured without this, the winner set churned at 0.37-0.80 overlap
    # between consecutive timesteps and never locked, so the loop plateaued
    # instead of converging no matter how the leak was tuned (ratio fell to
    # 0.087 at leak=0.15 and STILL was not monotonic -- the magnitudes had
    # converged and the mask was still flipping).
    #
    # Cortex does not have this problem, because a neuron already firing is
    # held up by recurrent self-excitation and is easier to keep firing than
    # a silent one is to recruit. The bonus is scaled by the spread of the
    # scores so it is a relative advantage, not a fixed offset that would
    # mean different things at different activation scales.
    # Resolved at call time, not bound as a default: a default argument is
    # evaluated once at import and would silently ignore any later change.
    if hysteresis is None:
        hysteresis = config.SPARSITY_HYSTERESIS

    score = x
    if incumbent is not None and hysteresis > 0.0:
        spread = x.std(dim=-1, keepdim=True)
        score = x + hysteresis * spread * incumbent.to(x.dtype)

    threshold = score.topk(k, dim=-1).values[..., -1:]
    return x * (score >= threshold).to(x.dtype)


class ExcitatoryInhibitory(nn.Module):
    """Separate E and I populations, with signs that cannot be violated.

    A standard Linear lets one unit excite some targets and inhibit others,
    which no neuron does. Here the excitatory weights are forced non-negative
    via softplus and ADDED; the inhibitory weights likewise non-negative and
    SUBTRACTED. Dale's principle becomes a fact about the parameterisation
    rather than a hope about what training finds.
    """

    def __init__(self, dim: int, inhibitory_fraction: float = config.INHIBITORY_FRACTION):
        super().__init__()
        self.dim = dim
        self.n_inhibitory = max(1, int(round(dim * inhibitory_fraction)))
        self.n_excitatory = dim - self.n_inhibitory

        self.excite = nn.Parameter(torch.randn(dim, self.n_excitatory) * 0.02)
        self.inhibit = nn.Parameter(torch.randn(dim, self.n_inhibitory) * 0.02)
        self.bias = nn.Parameter(torch.zeros(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e = x[..., : self.n_excitatory]
        i = x[..., self.n_excitatory:]
        # softplus keeps both weight sets non-negative; the SIGN of each
        # population's contribution is structural, not learned.
        drive = e @ nn.functional.softplus(self.excite).t()
        veto = i @ nn.functional.softplus(self.inhibit).t()
        return drive - veto + self.bias

    def balance(self) -> tuple[float, float]:
        """(mean excitatory weight, mean inhibitory weight). For inspection."""
        return (float(nn.functional.softplus(self.excite).mean()),
                float(nn.functional.softplus(self.inhibit).mean()))


class LaminarShell(nn.Module):
    """L4 / L2-3 / L5-6 around a lobe's operator.

    The operator itself is supplied by the lobe and lives in L2/3, which is
    where the lobe's identity is. Everything here is the shared cortical
    microcircuit that every area has regardless of what it computes.
    """

    def __init__(
        self,
        dim: int = config.LATENT_DIM,
        sparsity: float = config.CORTICAL_SPARSITY,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.dim = dim
        self.sparsity = sparsity

        # L4 -- thalamorecipient. Sensory drive lands here and nowhere else.
        self.l4 = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim), nn.GELU())

        # L2/3 -- where the lobe's own operator runs. Lateral input joins here.
        self.l23_norm = nn.LayerNorm(dim)
        self.l23_ei = ExcitatoryInhibitory(dim)

        # L5/6 -- deep layers. Feedback lands here; L6 projects to thalamus.
        self.l56 = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim), nn.GELU())
        self.l6_to_thalamus = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

        self._last_sparsity = 0.0

    def receive(
        self,
        thalamic: torch.Tensor | None = None,
        lateral: torch.Tensor | None = None,
        feedback: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Route each afferent to the layer that actually receives it."""
        parts = []
        if thalamic is not None:
            parts.append(self.l4(thalamic))
        if lateral is not None:
            parts.append(lateral)
        if feedback is not None:
            parts.append(self.l56(feedback))
        if not parts:
            raise ValueError("laminar shell received nothing")
        return sum(parts)

    def l23(self, x: torch.Tensor, sparsity: float | None = None) -> torch.Tensor:
        """Superficial layer: E/I balance, then k-WTA sparsity.

        Sparsity last, so the measured rate is the rate that leaves the layer.

        `sparsity` overrides the resting value so the caller can ANNEAL
        competition across the settling window. That is not a convenience:
        hard k-WTA inside a recurrent loop is discontinuous, and a small state
        change flips which units win, which produces a jump. Measured with a
        fixed 5% mask, the loop failed to settle at EVERY leak value tried
        (ratios 0.77 to 0.99, none monotonic) -- lowering the leak made it
        worse, not better.

        Annealing fixes it and is the better description anyway: cortical
        competition sharpens as a percept resolves, it is not clamped hard from
        the first millisecond. Early timesteps stay dense enough for the state
        to find its basin; the final timesteps carry the true target sparsity,
        so the rate that matters is still exactly the rate that is measured.
        """
        activated = self.l23_ei(self.l23_norm(x))
        sparse = k_winners(activated, self.sparsity if sparsity is None else sparsity)
        if config.COLLECT_TELEMETRY:
            self._last_sparsity = float((sparse != 0).float().mean().detach())
        return self.dropout(sparse)

    def corticothalamic(self, deep: torch.Tensor) -> torch.Tensor:
        """L6's projection back to the thalamus.

        Driven by the DEEP layers, which is why it carries feedback state
        rather than a copy of the sensory drive -- that is what makes it a
        control signal instead of an echo.
        """
        return self.l6_to_thalamus(self.l56(deep))

    @property
    def achieved_sparsity(self) -> float:
        """Fraction of units non-zero on the last pass. Measured, not assumed."""
        return self._last_sparsity


__all__ = ["LaminarShell", "ExcitatoryInhibitory", "k_winners"]
