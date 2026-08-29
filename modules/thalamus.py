"""
thalamus.py -- the modality-keyed switchboard (PROJECT.md section 2.3).

The correction this file exists to encode: the thalamus is NOT a funnel that
sends everything to occipital first. In a real brain each modality has its own
relay nucleus --

    LGN     -> occipital   (vision)
    MGN     -> temporal    (sound)
    VPL/VPM -> parietal    (touch/body)

-- so this is a switch, keyed on the modality tag, reading its routing table
from config.THALAMIC_ROUTES.

Parietal is NOT reachable from here. It is fed by occipital, mirroring the
dorsal "where" stream (V1 -> parietal). Anything that routes the thalamus
straight into parietal has reintroduced the bug this file fixes.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn as nn

import config


@dataclass
class Routing:
    """What the thalamus decided, in a form the UI can draw."""

    modality: str
    nucleus: str
    destination: str
    gate_mean: float
    #  destination lobe -> relayed LATENT_DIM tensor
    payload: dict[str, torch.Tensor] = field(default_factory=dict)


class RelayNucleus(nn.Module):
    """One thalamic nucleus: LATENT_DIM in, LATENT_DIM out, multiplicatively gated.

    Real thalamic relay cells do not just forward -- they gain-control what
    passes, which is why PROJECT.md section 2.2 calls this "routes/gates".
    The gate is a learned per-feature sigmoid, so the nucleus can attenuate
    parts of a signal without the cortex being involved.
    """

    def __init__(self, name: str, dim: int = config.LATENT_DIM):
        super().__init__()
        self.name = name
        self.relay = nn.Linear(dim, dim)
        self.gate = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)
        self._last_gate: float = 0.0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = torch.sigmoid(self.gate(x))
        self._last_gate = float(gate.mean().detach())
        return self.norm(self.relay(x) * gate)

    @property
    def last_gate(self) -> float:
        """Mean gate opening from the most recent pass. For the dashboard."""
        return self._last_gate


class Thalamus(nn.Module):
    """Reads config.THALAMIC_ROUTES; dispatches one modality to one lobe."""

    def __init__(self, dim: int = config.LATENT_DIM):
        super().__init__()
        self.dim = dim
        self.nuclei = nn.ModuleDict(
            {name: RelayNucleus(name, dim) for name in config.RELAY_NUCLEI}
        )
        self.routes = dict(config.THALAMIC_ROUTES)

        # Fail loudly at construction rather than at some later forward pass.
        for modality, (nucleus, destination) in self.routes.items():
            if nucleus not in self.nuclei:
                raise ValueError(f"route {modality!r} names unknown nucleus {nucleus!r}")
            # The rule from PROJECT.md section 2.3, stated precisely.
            #
            # It was originally written as "the thalamus must never feed
            # parietal". That was too strong: it is the VISUAL path that must
            # not shortcut into parietal, because the dorsal "where" stream
            # runs V1 -> parietal, not LGN -> parietal. Somatosensory input
            # genuinely does reach parietal directly, through VPL/VPM, and
            # section 2.3 names that route itself.
            #
            # So the invariant is about vision specifically: image must arrive
            # at parietal only by way of occipital.
            if destination == "parietal" and modality == "image":
                raise ValueError(
                    "the thalamus must not feed vision into parietal directly "
                    "(PROJECT.md section 2.3); the dorsal stream is "
                    "occipital -> parietal, not LGN -> parietal"
                )
            if destination not in config.LOBES:
                raise ValueError(f"route {modality!r} names unknown lobe {destination!r}")

    def destination_for(self, modality: str) -> str:
        return self._route(modality)[1]

    def _route(self, modality: str) -> tuple[str, str]:
        try:
            return self.routes[modality]
        except KeyError as exc:
            raise KeyError(
                f"unroutable modality {modality!r}; the thalamus knows "
                f"{tuple(self.routes)}"
            ) from exc

    def forward(self, modality: str, embedding: torch.Tensor) -> Routing:
        if embedding.dim() == 1:
            embedding = embedding.unsqueeze(0)
        # Same tolerance as the lobes: the module knows its own device.
        _device = self.nuclei[next(iter(self.nuclei))].relay.weight.device
        if embedding.device != _device:
            embedding = embedding.to(_device)
        if embedding.shape[-1] != self.dim:
            raise ValueError(
                f"thalamus expected LATENT_DIM={self.dim}, got {tuple(embedding.shape)}"
            )

        nucleus_name, destination = self._route(modality)
        nucleus = self.nuclei[nucleus_name]
        relayed = nucleus(embedding)

        return Routing(
            modality=modality,
            nucleus=nucleus_name,
            destination=destination,
            gate_mean=nucleus.last_gate,
            payload={destination: relayed},
        )


__all__ = ["Thalamus", "RelayNucleus", "Routing"]
