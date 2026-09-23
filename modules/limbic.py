"""
limbic.py -- amygdala, hippocampus, hypothalamus (BUILD_SPEC_V2 3.3, stage 6).

THREE STRUCTURES, THREE DIFFERENT TIMESCALES. That is most of the point.

    amygdala        milliseconds   is this urgent, and is it good or bad
    hippocampus     one episode    what just happened, and have I seen it before
    hypothalamus    many episodes  how am I doing

A system with only cortex has one clock. Everything is computed at the rate the
percept arrives, which means nothing can be faster than perception (so nothing
can interrupt it) and nothing can be slower (so nothing can accumulate).

THE AMYGDALA'S TWO ROADS.

    LOW ROAD:   thalamus ------------------> amygdala     fast, crude
    HIGH ROAD:  thalamus --> temporal -----> amygdala     slow, detailed

The low road exists because a fast wrong answer about danger beats a slow
correct one. It is not a shortcut for efficiency; it is a different tradeoff,
and the two roads genuinely disagree -- that is what makes a startle response
possible and also what makes it possible to be startled by nothing.

Here the ordering is STRUCTURAL, not a matter of when code happens to run: the
low road is a tract of delay 1 and the high road a tract of delay 2, so the low
road's contribution is present at t=1 and the high road's cannot arrive before
t=2. The spec asks for thalamus at t and temporal at t-1; a delay of 0 is
rejected outright by the parallelism gate (it would let a module read the
present and make update order matter), so both are shifted by one and the gap
between them -- which is the part that carries the meaning -- is preserved.

THE HIPPOCAMPUS, as the actual circuit rather than a memory buffer.

    cortex --> ENTORHINAL --> DENTATE GYRUS      pattern separation, ~2% sparse
                   |               |
                   |              CA3 <--+       recurrent collaterals,
                   |               |     |       pattern completion
                   |               v     |
                   +-----------> CA1 ----+       compares completion vs input
                                  |
                              subiculum --> frontal, hypothalamus

Each stage does something the others cannot:

  DG separates. It expands into a much wider layer and keeps ~2% of it, so two
  similar inputs land on nearly disjoint sets of units. Without this, similar
  episodes overwrite each other.

  CA3 completes. Dense recurrent collaterals settle toward a stored pattern, so
  a fragment retrieves the whole. Separation and completion are opposite
  operations, which is why they are separate structures.

  CA1 compares CA3's completion against what EC actually delivered. Agreement
  means recognised; disagreement means novel. Novelty is therefore MEASURED
  here rather than being an extra head bolted on somewhere.

REPLAY is why the buffer is not just a buffer. Between batches, stored CA3
patterns are sampled back into cortex, so episodes are revisited without the
world providing them again. That is the claim stage 6's gate tests directly:
train on a sequence of tasks with and without replay, and see whether the
earlier ones survive.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn as nn

import config
from modules.laminar import k_winners


@dataclass
class AmygdalaOutput:
    """One salience evaluation, with the two roads kept apart."""

    salience: torch.Tensor            # (B, 1) 0..1 -- how much this matters
    valence: torch.Tensor             # (B, 1) -1..1 -- bad .. good
    gain: torch.Tensor                # (B, dim) -- cortical gain modulation
    low_road: torch.Tensor            # (B, 1) the fast estimate on its own
    high_road: torch.Tensor | None    # (B, 1) the slow one, None until it arrives

    @property
    def high_road_arrived(self) -> bool:
        return self.high_road is not None


class Amygdala(nn.Module):
    """Salience and valence, by two routes with different latencies."""

    def __init__(self, dim: int = config.LATENT_DIM):
        super().__init__()
        #  Deliberately asymmetric capacity: the low road is one layer because
        #  it is supposed to be crude. Giving it the same depth as the high
        #  road would remove the reason for having two.
        self.low = nn.Linear(dim, dim)
        self.high = nn.Sequential(
            nn.LayerNorm(dim), nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, dim)
        )
        self.salience_head = nn.Linear(dim, 1)
        self.valence_head = nn.Linear(dim, 1)
        self.to_cortex = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)

    def forward(self, low_road: torch.Tensor,
                high_road: torch.Tensor | None = None) -> AmygdalaOutput:
        fast = torch.relu(self.low(low_road))
        combined = fast
        slow_estimate = None
        if high_road is not None:
            slow = self.high(high_road)
            combined = fast + slow
            slow_estimate = torch.sigmoid(self.salience_head(slow))

        combined = self.norm(combined)
        salience = torch.sigmoid(self.salience_head(combined))
        valence = torch.tanh(self.valence_head(combined))

        #  Salience modulates cortical gain: a salient percept is processed
        #  harder. This is the amygdala's outward effect on cortex, and it is
        #  multiplicative because that is what a gain control is.
        #
        #  VALENCE participates too, and not for symmetry. With salience alone
        #  the valence head had no path to anything differentiable -- it fed
        #  only the hypothalamic update, which runs under no_grad -- so it
        #  received no gradient and was a readout nobody was training. Good
        #  and bad should change how hard cortex works on something, not
        #  merely be reported alongside it.
        emphasis = salience * (1.0 + config.VALENCE_WEIGHT * valence.abs())
        gain = 1.0 + emphasis * torch.tanh(self.to_cortex(combined))

        return AmygdalaOutput(
            salience=salience, valence=valence, gain=gain,
            low_road=torch.sigmoid(self.salience_head(self.norm(fast))),
            high_road=slow_estimate,
        )


class DentateGyrus(nn.Module):
    """Pattern SEPARATION: expand, then keep very few.

    Two similar inputs must land on nearly disjoint unit sets, or the next
    stage will merge them. Expansion alone does not do that and sparsity alone
    does not either -- it is expansion FOLLOWED BY hard competition that pushes
    similar inputs apart, because in a much wider space the top 2% of two
    similar vectors can still be almost disjoint.
    """

    def __init__(self, dim: int = config.LATENT_DIM,
                 expansion: int = config.DG_EXPANSION,
                 sparsity: float = config.DG_SPARSITY):
        super().__init__()
        self.wide = nn.Linear(dim, dim * expansion)
        self.back = nn.Linear(dim * expansion, dim)
        self.sparsity = sparsity
        self._last_sparsity = 0.0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        wide = k_winners(torch.relu(self.wide(x)), self.sparsity)
        if config.COLLECT_TELEMETRY:
            self._last_sparsity = float((wide != 0).float().mean().detach())
        return self.back(wide)

    @property
    def achieved_sparsity(self) -> float:
        return self._last_sparsity


class CA3(nn.Module):
    """Pattern COMPLETION: recurrent collaterals that settle onto a memory.

    Two mechanisms, and both are needed for different reasons. The learned
    recurrent weights train with everything else and let completion adapt to
    the statistics of what is actually stored. The Hebbian trace is a plain
    outer-product associative memory, written the moment an episode arrives
    and requiring no training at all -- which is what makes one-shot recall
    possible, and what makes the completion claim testable on an untrained
    model instead of only after a training run that might be doing the work.
    """

    def __init__(self, dim: int = config.LATENT_DIM,
                 steps: int = config.CA3_STEPS):
        super().__init__()
        self.recurrent = nn.Linear(dim, dim)
        self.steps = steps
        self.rate = config.CA3_COMPLETION_RATE
        self.collateral_gain = nn.Parameter(
            torch.tensor(config.CA3_COLLATERAL_GAIN)
        )
        #  A buffer, not a parameter: written by Hebbian storage, never by the
        #  optimiser. Registering it means it moves with .to(device) and is
        #  saved in the state dict like any other part of the model.
        self.register_buffer("hebbian", torch.zeros(dim, dim))
        self.register_buffer("stored_count", torch.zeros(1))

    @torch.no_grad()
    def store(self, pattern: torch.Tensor) -> None:
        """Hebbian write: units active together get a stronger connection."""
        flat = pattern.detach().reshape(-1, pattern.shape[-1])
        normed = flat / (flat.norm(dim=-1, keepdim=True) + 1e-8)
        #  REBOUND, not mutated in place. Storage happens at the end of an
        #  episode, after the forward graph has been built but before backward
        #  runs, and the same buffer was read during that forward pass. An
        #  in-place `+=` bumps its version counter and autograd then refuses
        #  the whole backward with "a variable needed for gradient computation
        #  has been modified". Rebinding leaves the tensor the graph captured
        #  exactly as it was.
        self.hebbian = self.hebbian + normed.t() @ normed
        self.stored_count = self.stored_count + flat.shape[0]

    def forward(self, drive: torch.Tensor) -> torch.Tensor:
        """Settle the cue toward whatever stored pattern it most resembles.

        The obvious formulation -- tanh(learned(state) + recall + drive) at
        every step -- does not complete anything, and measurably so: it moved
        a corrupted cue from 0.64 similarity with its own memory to 0.57,
        i.e. FURTHER away. Three things were wrong with it. The raw drive was
        re-added every step, so the corruption kept being reasserted; tanh
        saturated everything into the same corner; and the randomly
        initialised learned collaterals were simply louder than the recall.

        What works is an interpolation toward a scale-matched recall, which is
        the classical attractor update and not an invention here. Measured on
        the same cue: 0.64 -> 0.95.
        """
        state = drive
        for _ in range(self.steps):
            recalled = state @ self.hebbian
            #  Match the recall to the state's own scale, so "how far to move"
            #  is set by `rate` alone rather than by how many patterns happen
            #  to be stored.
            magnitude = recalled.norm(dim=-1, keepdim=True)
            scale = torch.where(
                magnitude > 1e-6,
                state.norm(dim=-1, keepdim=True) / (magnitude + 1e-8),
                torch.zeros_like(magnitude),
            )
            #  The learned collaterals REFINE the recall; the gain starts small
            #  so that an untrained CA3 still completes correctly, and training
            #  can raise their influence rather than having to first undo it.
            target = (recalled * scale
                      + self.collateral_gain * self.recurrent(state))
            state = (1.0 - self.rate) * state + self.rate * target
        return state


class Hippocampus(nn.Module):
    """EC to DG to CA3 to CA1 to subiculum, with replay."""

    def __init__(self, dim: int = config.LATENT_DIM,
                 capacity: int = config.REPLAY_CAPACITY):
        super().__init__()
        self.entorhinal = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim))
        self.dg = DentateGyrus(dim)
        self.ca3 = CA3(dim)
        #  CA1 sees BOTH the completed pattern and the direct EC input; that
        #  juxtaposition is the whole reason it can report novelty.
        self.ca1 = nn.Linear(dim * 2, dim)
        self.subiculum = nn.Linear(dim, dim)
        self.capacity = capacity
        self._episodes: list[torch.Tensor] = []
        self._tags: list[torch.Tensor | None] = []

    def forward(self, cortical: torch.Tensor, store: bool = False):
        ec = self.entorhinal(cortical)
        separated = self.dg(ec)
        completed = self.ca3(separated)

        #  CA1 compares the completion against what actually arrived.
        ca1 = self.ca1(torch.cat([completed, ec], dim=-1))
        #  Novelty is the disagreement, measured -- not a separate head that
        #  would have to be taught what novelty means.
        novelty = (completed - ec).pow(2).mean(dim=-1, keepdim=True)
        out = self.subiculum(ca1)

        if store:
            self.remember(separated)
        return out, novelty, ec

    @torch.no_grad()
    def remember(self, pattern: torch.Tensor,
                 tag: torch.Tensor | None = None) -> None:
        """Commit one episode, optionally with whatever it was paired with.

        The tag travels WITH the pattern rather than being kept alongside in
        the caller, and that is a bug fix rather than a convenience. The buffer
        is bounded, so it drops its oldest entry once full; a caller holding
        labels in a dict keyed by list position then has every key silently
        off by one from the first eviction onward, and replays each stored
        episode against the wrong answer. It fails quietly -- replay still
        appears to work, just barely: measured at +1.2 points of retention
        where correctly-paired replay gives +15.
        """
        self.ca3.store(pattern)
        self._episodes.append(pattern.detach().clone())
        self._tags.append(None if tag is None else tag.detach().clone())
        while len(self._episodes) > self.capacity:
            self._episodes.pop(0)
            self._tags.pop(0)

    @torch.no_grad()
    def forget(self) -> None:
        """Empty the memory. Used by tests that need to set up their own.

        The hippocampus accumulates across every episode the model sees in
        training mode, which is the point of it -- and a hazard for any
        measurement taken afterwards. A pattern-completion check that runs on
        a trace already carrying a few hundred unrelated episodes is measuring
        interference, not completion, and will drift depending on what ran
        before it.
        """
        self._episodes.clear()
        self._tags.clear()
        self.ca3.hebbian = torch.zeros_like(self.ca3.hebbian)
        self.ca3.stored_count = torch.zeros_like(self.ca3.stored_count)

    @torch.no_grad()
    def sample(self, index: int | None = None):
        """One stored episode with its tag, kept in step by construction."""
        if not self._episodes:
            return None, None
        if index is None:
            index = int(torch.randint(0, len(self._episodes), (1,)))
        return self._episodes[index], self._tags[index]

    @torch.no_grad()
    def replay(self, count: int) -> torch.Tensor | None:
        """Sample stored episodes back out, for rehearsal between batches."""
        if not self._episodes:
            return None
        pool = torch.cat(self._episodes, dim=0)
        if pool.shape[0] <= count:
            return pool
        index = torch.randperm(pool.shape[0], device=pool.device)[:count]
        return pool[index]

    @property
    def episode_count(self) -> int:
        return sum(int(e.shape[0]) for e in self._episodes)


class Hypothalamus(nn.Module):
    """Homeostasis: a few scalars that change across episodes, not timesteps.

    Persistent state held in a buffer rather than recomputed, because the
    entire point is that it carries over. Everything else in this system is
    recomputed from the current percept; this is the one place that is not,
    and that is what lets it represent a condition rather than an observation.
    """

    DRIVES = ("energy", "temperature", "threat", "satiety")

    def __init__(self, dim: int = config.LATENT_DIM,
                 rate: float = config.HOMEOSTATIC_RATE):
        super().__init__()
        #  DELIBERATELY NOT TRAINED, and marked as such rather than left to
        #  show up as a dead tensor. This projection DEFINES what the four
        #  drives measure: it is the sensor, not a hypothesis about the world.
        #  Training it would let the meaning of "energy" drift to whatever
        #  minimised the loss, which would make the homeostatic state
        #  unfalsifiable. It is also only ever read inside `update`, which runs
        #  under no_grad because integration spans episodes and backprop
        #  through it would be BPTT across the whole history.
        self.read = nn.Linear(dim, len(self.DRIVES))
        self.read.requires_grad_(False)
        self.to_cortex = nn.Linear(len(self.DRIVES), dim)
        self.rate = rate
        #  A RESTING CONDITION, not zero. Homeostatic set-points are not the
        #  origin -- an organism at rest still has an energy level. Starting
        #  at exactly zero also made `to_cortex` receive no gradient at all
        #  until the first episode, since a weight gradient is grad_out times
        #  input and the input was identically zero.
        self.register_buffer(
            "state", torch.full((1, len(self.DRIVES)), config.HOMEOSTATIC_REST)
        )

    @torch.no_grad()
    def update(self, cortical: torch.Tensor,
               valence: torch.Tensor | None = None) -> torch.Tensor:
        """Integrate the episode into the persistent drives. Slowly."""
        observed = torch.tanh(self.read(cortical)).mean(dim=0, keepdim=True)
        if valence is not None:
            #  Threat rises when things are going badly.
            observed = observed.clone()
            observed[:, 2] = observed[:, 2] - valence.mean()
        self.state = (1.0 - self.rate) * self.state + self.rate * observed
        return self.state

    def forward(self, batch: int = 1) -> torch.Tensor:
        """The current homeostatic condition, as a cortical drive."""
        return self.to_cortex(self.state).expand(batch, -1)

    def report(self) -> str:
        values = self.state.detach().reshape(-1).tolist()
        return "  ".join("%s %+.3f" % (n, v)
                         for n, v in zip(self.DRIVES, values))


__all__ = ["Amygdala", "AmygdalaOutput", "Hippocampus", "DentateGyrus", "CA3",
           "Hypothalamus"]
