"""
basal_ganglia.py -- the gate between frontal's decision and the output.

PROJECT.md section 2.1 places this between frontal and the output generator:
"gate: allow or suppress". Section 11 listed it as named but unspecified.
This is the specification.

Biology being modelled, minimally:

    cortex --> STRIATUM --+--> DIRECT pathway   ("Go")     --+
                          |                                  +--> THALAMUS -> action
                          +--> INDIRECT pathway ("NoGo")   --+

The direct pathway disinhibits the thalamus and lets an action through; the
indirect pathway suppresses it. The two compete, and what escapes is the
difference. That is exactly what is implemented: two heads read the striatal
representation, and the gate is sigmoid(go - nogo).

Note what is deliberately NOT modelled: the real loop returns through the
thalamus before reaching motor cortex. PROJECT.md section 2.4 explicitly drops
the thalamic return trip, so the gate applies directly on the way out.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

import config


@dataclass
class GateDecision:
    """The gate's verdict, in a form the UI and the output generator can read."""

    gate: float          # 0..1, how open the gate is
    go: float            # direct-pathway drive
    nogo: float          # indirect-pathway drive
    allowed: bool        # gate >= threshold
    threshold: float
    gated: torch.Tensor  # frontal's decision, scaled by the gate

    def describe(self) -> str:
        verdict = "ALLOW" if self.allowed else "SUPPRESS"
        return (
            f"{verdict}  gate={self.gate:.3f} "
            f"(go={self.go:.3f} nogo={self.nogo:.3f} thr={self.threshold:.2f})"
        )


class BasalGanglia(nn.Module):
    """LATENT_DIM in -> LATENT_DIM out, scaled by a learned go/no-go gate."""

    def __init__(
        self,
        dim: int = config.LATENT_DIM,
        threshold: float = config.BG_GATE_THRESHOLD,
    ):
        super().__init__()
        self.dim = dim
        self.threshold = threshold

        # Striatum: the shared representation both pathways read from.
        self.striatum = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
            nn.GELU(),
        )
        self.direct = nn.Linear(dim, 1)     # Go
        self.indirect = nn.Linear(dim, 1)   # NoGo

        # Start the gate open, via the BIASES only. A randomly-initialised gate
        # that happens to sit below threshold would suppress every output on an
        # untrained system, which reads as a broken build rather than an
        # untrained one.
        #
        # The weights must NOT be zeroed. Zeroing both makes the striatum a dead
        # pathway: the gradient reaching it is (direct.W - indirect.W)^T . g,
        # and the two heads receive exactly opposite gradients, so they stay
        # near-symmetric and their difference stays near zero. Measured on the
        # original zero-init: striatal gradient magnitude ~1e-6, i.e. 66k
        # parameters that never trained. Small opposite-signed init breaks the
        # symmetry while leaving the gate open at start.
        nn.init.normal_(self.direct.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.indirect.weight, mean=0.0, std=0.02)
        nn.init.constant_(self.direct.bias, 1.5)
        nn.init.constant_(self.indirect.bias, -1.5)

    def forward(self, decision: torch.Tensor) -> GateDecision:
        if decision.dim() == 1:
            decision = decision.unsqueeze(0)
        # Same tolerance as the lobes: the module knows its own device.
        _device = self.direct.weight.device
        if decision.device != _device:
            decision = decision.to(_device)
        if decision.shape[-1] != self.dim:
            raise ValueError(
                f"basal ganglia expected LATENT_DIM={self.dim}, "
                f"got {tuple(decision.shape)}"
            )

        striatal = self.striatum(decision)
        go = self.direct(striatal)
        nogo = self.indirect(striatal)
        gate = torch.sigmoid(go - nogo)              # (B, 1)

        gate_value = float(gate.mean().detach())
        return GateDecision(
            gate=gate_value,
            go=float(go.mean().detach()),
            nogo=float(nogo.mean().detach()),
            allowed=gate_value >= self.threshold,
            threshold=self.threshold,
            gated=decision * gate,
        )


__all__ = ["BasalGanglia", "GateDecision"]
