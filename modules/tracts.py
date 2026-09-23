"""
tracts.py -- white matter as named objects (BUILD_SPEC_V2 section 4).

WHY A CONNECTION IS AN OBJECT AND NOT A TENSOR ADD.

In v1 the wiring was implicit: `parietal(occipital_output)` and the connection
between them existed only as the shape of the call stack. Nothing could be
pointed at, named, delayed, or cut. Lesion experiments were therefore
impossible at the level that matters -- you could ablate a MODULE, but not the
pathway between two intact modules, which is what most real white-matter injury
actually is.

Every projection here is a `Tract`: a named, learned linear map with its own
conduction delay, registered in a `TractBundle` that can enumerate and lesion
them by name. `ablate.py` depends on this and nothing else.

CONDUCTION DELAY IS THE POINT, NOT A WORKAROUND.

A tract with delay k reads its source's state at `t - k`. That is what makes
the settling loop genuinely parallel: no module can observe another module's
CURRENT timestep, so within a step there is no ordering between them and none
waits on another. Real axons take milliseconds to conduct, and longer tracts
take longer -- so IFOF and arcuate (long association fibres) carry delay 2
while short radiations carry 1. The delay is anatomy, not scheduling.

Delay 0 is REFUSED for any tract between two modules. Allowing it would let a
lobe read a value computed earlier in the same timestep, which silently turns
the parallel loop back into a pipeline whose behaviour depends on the order the
lobes happen to be written in. `verify.py` tests exactly that by shuffling the
update order and requiring identical output.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn as nn

import config


class Tract(nn.Module):
    """One named white-matter pathway: source -> (delay, linear map) -> target.

    `gain` is not a parameter but a lesion handle: set it to 0.0 and the tract
    is severed while everything either side stays intact. That is the whole
    reason this is a module rather than a matmul.
    """

    def __init__(
        self,
        name: str,
        source: str,
        target: str,
        dim: int = config.LATENT_DIM,
        delay: int | None = None,
        source_dim: int | None = None,
    ):
        super().__init__()
        if delay is None:
            delay = config.TRACT_DELAYS.get(name, 1)
        if delay < 1:
            raise ValueError(
                f"tract {name!r} has delay {delay}: a tract between modules must "
                f"delay at least one timestep, or the reader observes the "
                f"writer's current state and the settling loop stops being "
                f"parallel. See the module docstring."
            )
        self.name = name
        self.source = source
        self.target = target
        self.delay = delay
        self.dim = dim
        self.projection = nn.Linear(source_dim or dim, dim)
        #  1.0 = intact, 0.0 = severed. Not a parameter; a lesion switch.
        self.register_buffer("gain", torch.ones(()))
        #  1.0 = the signal is silenced but the tonic bias still flows.
        #
        #  NOT PERSISTED. This is an analysis handle, not something the model
        #  learned, and adding it to the state dict invalidated every existing
        #  checkpoint the moment it was introduced. A tract is muted to ask a
        #  question and unmuted afterwards; that state has no business
        #  surviving a save.
        self.register_buffer("muted", torch.zeros(()), persistent=False)

    def forward(self, activation: torch.Tensor) -> torch.Tensor:
        #  MUTING silences the SIGNAL and leaves the tonic bias; lesioning
        #  removes the whole projection. Both are real interventions and they
        #  answer different questions.
        #
        #  A lesion is the anatomical one: the axon is gone, tonic output and
        #  all. But a Linear with zero input still emits its bias, so cutting
        #  a tract also removes a CONSTANT -- and that constant, pushed
        #  through the k-winners nonlinearity, changes which units win one
        #  step later. Measured, that made two delay-2 feedback tracts look
        #  as though information had arrived a step early when nothing had:
        #  the knock-on from removing a constant is not the same as the
        #  signal itself, and only muting can tell them apart.
        if bool(self.muted):
            activation = torch.zeros_like(activation)
        return self.projection(activation) * self.gain

    def lesion(self, severed: bool = True) -> None:
        self.gain.fill_(0.0 if severed else 1.0)

    def mute(self, silent: bool = True) -> None:
        """Carry no signal, but keep emitting the tonic bias.

        The intervention that isolates INFORMATION transmission from the
        constant a projection contributes regardless of its input.
        """
        self.muted.fill_(1.0 if silent else 0.0)

    @property
    def severed(self) -> bool:
        return bool(self.gain.item() == 0.0)

    def extra_repr(self) -> str:
        return (f"{self.source} -> {self.target}, delay={self.delay}"
                f"{', SEVERED' if self.severed else ''}")


@dataclass
class StateHistory:
    """Per-module activation history, addressed by absolute timestep.

    A tract with delay k asks for `t - k`, and before the loop has run that far
    the answer is the resting state -- zeros. That is the correct answer, not a
    special case: at t=0 nothing has been computed yet, so every cross-module
    input genuinely is absent, and the first timestep is driven purely by
    sensory input. The picture builds up as the tracts start delivering.
    """

    dim: int
    batch: int
    device: torch.device
    steps: dict[str, dict[int, torch.Tensor]] = field(default_factory=dict)

    def write(self, module: str, t: int, value: torch.Tensor) -> None:
        self.steps.setdefault(module, {})[t] = value

    def read(self, module: str, t: int) -> torch.Tensor:
        """State of `module` at timestep `t`, or resting state if not yet computed."""
        stored = self.steps.get(module, {})
        if t in stored:
            return stored[t]
        return torch.zeros(self.batch, self.dim, device=self.device)

    def series(self, module: str) -> list[torch.Tensor]:
        stored = self.steps.get(module, {})
        return [stored[t] for t in sorted(stored)]

    def modules(self) -> list[str]:
        return sorted(self.steps)


class TractBundle(nn.Module):
    """Every tract in the system, addressable by name.

    Deliberately flat: `bundle["slf"]` reaches the superior longitudinal
    fasciculus wherever it sits in the architecture, so a lesion sweep is a
    loop over names rather than a walk of the module tree.
    """

    def __init__(self, tracts: list[Tract] | None = None):
        super().__init__()
        self.tracts = nn.ModuleDict()
        for tract in tracts or []:
            self.add(tract)

    def add(self, tract: Tract) -> Tract:
        if tract.name in self.tracts:
            raise ValueError(f"duplicate tract name {tract.name!r}")
        self.tracts[tract.name] = tract
        return tract

    def __getitem__(self, name: str) -> Tract:
        try:
            return self.tracts[name]
        except KeyError as exc:
            raise KeyError(
                f"no tract {name!r}; known: {sorted(self.tracts)}"
            ) from exc

    def __iter__(self):
        return iter(self.tracts.values())

    def __len__(self) -> int:
        return len(self.tracts)

    def names(self) -> list[str]:
        return sorted(self.tracts)

    def into(self, target: str) -> list[Tract]:
        """Every tract terminating on `target`, in a stable order.

        Stable because the order tracts are summed in must not depend on dict
        iteration -- a floating-point sum is not associative, and a comparison
        that shifts by 1e-7 between runs is a comparison that cannot be trusted.
        """
        return sorted((t for t in self.tracts.values() if t.target == target),
                      key=lambda t: t.name)

    def deliver(self, target: str, history: StateHistory, t: int) -> torch.Tensor:
        """Sum of everything arriving at `target` on timestep `t`.

        Each tract reads its own source at its own delay, so this is where
        conduction latency actually bites. The sum is dendritic integration:
        several afferent pathways converging on one population.
        """
        incoming = self.into(target)
        if not incoming:
            return torch.zeros(history.batch, history.dim, device=history.device)
        total = None
        for tract in incoming:
            arrived = tract(history.read(tract.source, t - tract.delay))
            total = arrived if total is None else total + arrived
        return total

    def lesion(self, *names: str) -> None:
        for name in names:
            self[name].lesion(True)

    def restore(self) -> None:
        for tract in self:
            tract.lesion(False)

    def report(self) -> str:
        lines = [f"{'tract':<22}{'from':<15}{'to':<12}{'delay':>6}  state"]
        lines.append("-" * 65)
        for name in self.names():
            t = self[name]
            lines.append(f"{name:<22}{t.source:<15}{t.target:<12}{t.delay:>6}  "
                         f"{'SEVERED' if t.severed else 'intact'}")
        return "\n".join(lines)


__all__ = ["Tract", "TractBundle", "StateHistory"]
