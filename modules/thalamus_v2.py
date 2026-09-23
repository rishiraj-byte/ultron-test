"""
thalamus_v2.py -- nuclei plus the TRN (BUILD_SPEC_V2 section 3.2, stage 3).

WHAT v1 MISSED.

v1's thalamus was a switchboard: one gated relay per modality, and that was
all. Two things were absent, and both are load-bearing.

**Corticothalamic feedback.** In a real brain, cortex->thalamus synapses
outnumber thalamus->cortex by roughly an order of magnitude. The thalamus is
not an input stage that cortex reads; it is a structure cortex is constantly
writing to. Layer 6 of every cortical area projects back down. Without that, a
"relay" is just an input layer with extra steps.

**The TRN -- thalamic reticular nucleus.** A sheet of GABAergic neurons
wrapping the thalamus, which every thalamocortical and corticothalamic axon
passes through. It receives cortical L6 feedback and *inhibits specific
nuclei*. This is the physical substrate of selective attention: cortex decides
what to let in, one channel at a time, by telling the TRN what to suppress.

    thal_out = relay(x) * sigmoid(gate(x) + W_fb @ frontal[t-1] - trn_inhibition)

The TRN term is subtractive and the feedback term additive, inside one sigmoid.
That is not cosmetic: it means cortex can both open a channel it wants and,
through the TRN, close the ones competing with it. A purely multiplicative gain
could do neither selectively.

STAGE 3 SCOPE. Feedback and TRN only. The BG return path (MD, VA/VL) is
stubbed with the right shape but not driven until stage 5, when the basal
ganglia circuit exists to drive it -- wiring a nucleus to nothing and calling
it built is exactly what the gated build order is meant to prevent.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn as nn

import config


@dataclass
class ThalamicOutput:
    """One relay pass, with every gate value kept for inspection."""

    #  nucleus -> relayed signal (B, dim)
    nuclei: dict[str, torch.Tensor] = field(default_factory=dict)
    #  nucleus -> mean gate opening, 0..1
    gates: dict[str, float] = field(default_factory=dict)
    #  nucleus -> mean TRN inhibition applied
    inhibition: dict[str, float] = field(default_factory=dict)
    #  raw, ungated sensory -- the amygdala's low road taps this (stage 6)
    raw: dict[str, torch.Tensor] = field(default_factory=dict)

    def __getitem__(self, nucleus: str) -> torch.Tensor:
        return self.nuclei[nucleus]

    def report(self) -> str:
        lines = [f"{'nucleus':<12}{'gate':>8}{'inhibition':>12}"]
        for name in sorted(self.nuclei):
            lines.append(f"{name:<12}{self.gates.get(name, 0):>8.3f}"
                         f"{self.inhibition.get(name, 0):>12.3f}")
        return "\n".join(lines)


class ReticularNucleus(nn.Module):
    """TRN -- the inhibitory shell. Cortex's handle on what gets through.

    Produces one inhibition scalar per nucleus from cortical L6 feedback, and
    a competition term: nuclei compete, so attending to one channel actively
    suppresses the others. Softmax over nuclei is what makes it selective
    rather than a global volume knob.
    """

    def __init__(self, nuclei: tuple[str, ...], dim: int = config.LATENT_DIM):
        super().__init__()
        self.nuclei = tuple(nuclei)
        self.from_cortex = nn.Linear(dim, len(self.nuclei))
        # Positive bias = mildly inhibited at rest, so cortex has something to
        # release rather than only something to add to.
        nn.init.constant_(self.from_cortex.bias, 0.5)
        self.strength = nn.Parameter(torch.tensor(1.0))

    def forward(self, cortical_feedback: torch.Tensor) -> dict[str, torch.Tensor]:
        drive = self.from_cortex(cortical_feedback)          # (B, n_nuclei)
        # Competition: attending to one channel suppresses its rivals.
        share = torch.softmax(drive, dim=-1)
        inhibition = self.strength * share * len(self.nuclei)
        return {name: inhibition[:, i:i + 1]
                for i, name in enumerate(self.nuclei)}


class RelayNucleusV2(nn.Module):
    """One thalamic nucleus, now under cortical and TRN control."""

    def __init__(self, name: str, dim: int = config.LATENT_DIM):
        super().__init__()
        self.name = name
        self.relay = nn.Linear(dim, dim)
        self.gate = nn.Linear(dim, dim)
        # The corticothalamic projection: cortex L6 writing to the thalamus.
        self.from_cortex = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)
        self._last_gate = 0.0

    def forward(
        self,
        sensory: torch.Tensor,
        cortical_feedback: torch.Tensor | None = None,
        trn_inhibition: torch.Tensor | None = None,
    ) -> torch.Tensor:
        drive = self.gate(sensory)
        if cortical_feedback is not None:
            drive = drive + self.from_cortex(cortical_feedback)
        if trn_inhibition is not None:
            drive = drive - trn_inhibition
        opening = torch.sigmoid(drive)
        if config.COLLECT_TELEMETRY:
            self._last_gate = float(opening.mean().detach())
        return self.norm(self.relay(sensory) * opening)

    @property
    def last_gate(self) -> float:
        return self._last_gate


class ThalamusV2(nn.Module):
    """Nuclei + TRN + corticothalamic feedback.

    Nuclei present but not yet driven are declared here rather than added
    later, so the anatomy is visible from the start; `driven` records which
    ones a stage has actually wired up.
    """

    #  nucleus -> what it carries. MD and VA/VL wait for stage 5 (BG return).
    NUCLEI = ("LGN", "MGN", "VPL", "PULVINAR", "MD", "VA_VL")
    DRIVEN_AT_STAGE_3 = ("LGN", "MGN", "VPL", "PULVINAR")

    def __init__(self, dim: int = config.LATENT_DIM):
        super().__init__()
        self.dim = dim
        self.nuclei = nn.ModuleDict(
            {name: RelayNucleusV2(name, dim) for name in self.NUCLEI}
        )
        self.trn = ReticularNucleus(self.NUCLEI, dim)
        self.driven = set(self.DRIVEN_AT_STAGE_3)

    def forward(
        self,
        sensory: dict[str, torch.Tensor],
        cortical_feedback: torch.Tensor | dict[str, torch.Tensor] | None = None,
        use_trn: bool = True,
        trn_drive: torch.Tensor | None = None,
    ) -> ThalamicOutput:
        """`sensory` maps nucleus name -> (B, dim). Missing nuclei stay silent.

        `cortical_feedback` may be one tensor shared by every nucleus, or a
        dict nucleus -> feedback. The dict form is the real anatomy: LGN is
        addressed by visual cortex, MGN by auditory cortex, and so on. Every
        cortical area has an L6 corticothalamic projection, not just the one
        at the top -- a single shared tensor quietly makes three of those
        four projections nonexistent.

        `trn_drive` is separate because the TRN is an ATTENTIONAL controller:
        it is told what to suppress by frontal cortex, and its whole function
        is to arbitrate BETWEEN nuclei, which a per-nucleus signal cannot do.
        """
        out = ThalamicOutput()

        if trn_drive is None and isinstance(cortical_feedback, torch.Tensor):
            trn_drive = cortical_feedback

        inhibition = {}
        if use_trn and trn_drive is not None:
            inhibition = self.trn(trn_drive)

        for name, signal in sensory.items():
            if name not in self.nuclei:
                raise KeyError(
                    f"no thalamic nucleus {name!r}; known: {self.NUCLEI}"
                )
            nucleus = self.nuclei[name]
            trn_term = inhibition.get(name)
            if isinstance(cortical_feedback, dict):
                own_feedback = cortical_feedback.get(name)
            else:
                own_feedback = cortical_feedback
            relayed = nucleus(signal, own_feedback, trn_term)

            out.nuclei[name] = relayed
            out.raw[name] = signal
            out.gates[name] = nucleus.last_gate
            out.inhibition[name] = (
                float(trn_term.mean().detach())
                if trn_term is not None and config.COLLECT_TELEMETRY else 0.0
            )
        return out


__all__ = ["ThalamusV2", "ReticularNucleus", "RelayNucleusV2", "ThalamicOutput"]
