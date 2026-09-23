"""
synapses.py -- sparse connections that learn from what they can locally sense.

THE THING THIS FILE EXISTS TO NOT DO. There is no backward pass here. No
gradient, no loss, no optimiser, nothing transported from a distant output.
Every weight change below is computed from three quantities that are
physically present at the synapse: a trace of the cell before it firing, a
trace of the cell after it firing, and the concentration of a neuromodulator
in the surrounding tissue.

That constraint is the whole point of v3, and it is expensive. Backpropagation
is a far better credit-assignment algorithm than anything biology has, which
is why v2 reaches accuracies v3 will not. What v3 buys instead is that every
mechanism in it could, in principle, be implemented by tissue.

WHY A SEPARATE ELIGIBILITY TRACE. Spike-timing-dependent plasticity alone
cannot bridge time: by the time a decision is rewarded, the synapses that
caused it stopped signalling hundreds of milliseconds ago and nothing marks
them out. So a coincidence does not immediately change a weight -- it deposits
a slow-decaying eligibility tag, and the weight only moves when a
neuromodulator arrives to say whether that tag was worth keeping. This is the
three-factor rule, and it is biology's answer to the same problem backprop
solves by transporting error.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from v3 import config_v3 as cfg


class DelayLine:
    """Holds spikes in flight along an axon.

    An action potential takes real time to travel, and in v3 that time is
    milliseconds rather than an abstract tick. A 15 ms arcuate fasciculus
    genuinely delivers fifteen steps late, which means the fast amygdala route
    (8 ms) and the slow one (25 ms) are separated by seventeen milliseconds of
    real difference rather than by one unit of index.
    """

    def __init__(self, delay_ms: float, size: int):
        self.steps = max(int(round(delay_ms / cfg.DT_MS)), 1)
        self.size = size
        self._buffer: list[torch.Tensor] = []

    def reset(self, batch: int, device) -> None:
        self._buffer = [torch.zeros(batch, self.size, device=device)
                        for _ in range(self.steps)]

    def push_pop(self, spikes: torch.Tensor) -> torch.Tensor:
        """Insert this millisecond's spikes, return what arrives now."""
        self._buffer.append(spikes)
        return self._buffer.pop(0)


class SparseProjection(nn.Module):
    """A bundle of axons from one population to another.

    SPARSE AND FIXED. The connectivity mask is drawn once at construction and
    never changes, which is how anatomy works -- experience adjusts synaptic
    strength, it does not grow new long-range bundles on the fly. v2's dense
    matrices claimed every neuron contacts every other neuron, which is the
    least biological property of a standard weight matrix.

    NON-NEGATIVE, ALWAYS. The sign lives in the presenting population, not in
    the weight. An excitatory cell's synapse can weaken to zero but it cannot
    become inhibitory, because that would mean the cell changed which
    transmitter it releases.
    """

    def __init__(self, pre_size: int, post_size: int, p_connect: float,
                 sign: str, name: str = "projection", driver: bool = False,
                 dopamine_gated: bool = False):
        super().__init__()
        self.name = name
        self.sign = sign
        #  WHICH SYNAPSES NEED PERMISSION TO CHANGE.
        #
        #  Not all of them, and treating them alike was a real mistake. In a
        #  brain, dopamine-DEPENDENT plasticity is largely a striatal
        #  speciality: corticostriatal synapses genuinely will not change
        #  without it. Cortico-cortical and thalamocortical synapses undergo
        #  spike-timing-dependent plasticity on their own, with no reward
        #  involved -- which is how sensory cortex develops selectivity from
        #  experience alone, before anything is rewarding.
        #
        #  Built with every synapse gated on dopamine, nothing in the sensory
        #  pathway could ever develop, so frontal received a randomly mixed
        #  signal carrying almost no class information and the readout had
        #  nothing to learn from. Accuracy sat at chance regardless of how
        #  long it trained.
        self.dopamine_gated = dopamine_gated
        self.pre_size = pre_size
        self.post_size = post_size

        generator = torch.Generator().manual_seed(
            cfg.SEED + (hash(name) % 100_000))
        mask = (torch.rand(pre_size, post_size, generator=generator)
                < p_connect).float()
        self.register_buffer("mask", mask)

        #  Weights are in MILLIVOLTS of postsynaptic potential per presynaptic
        #  spike, not in arbitrary units. See cfg.EPSP_MV for why that
        #  distinction is load-bearing and what breaks silently without it.
        #  See cfg.DRIVER_MV: a driver is a different kind of synapse, not a
        #  louder one, and the system is silent without the distinction.
        amplitude = (cfg.DRIVER_MV if driver
                     else cfg.IPSP_MV if sign == "inh" else cfg.EPSP_MV)
        initial = 0.5 + torch.rand(pre_size, post_size, generator=generator)
        self.register_buffer("weight", initial * mask * amplitude)

        #  Plasticity state. Traces are per-neuron; eligibility is per-synapse
        #  because that is where the tag physically sits.
        self.register_buffer("trace_pre", torch.zeros(1, pre_size))
        self.register_buffer("trace_post", torch.zeros(1, post_size))
        self.register_buffer("eligibility", torch.zeros(pre_size, post_size))

        #  Astrocyte state: one slow local gain per postsynaptic cell. Glia
        #  were entirely absent from v2 and are not passive scaffolding --
        #  they sense local activity and modulate transmission over seconds.
        self.register_buffer("astrocyte", torch.zeros(post_size))

        #  Cut like a v2 Tract, so lesion experiments still work.
        self.register_buffer("gain", torch.tensor(1.0))

        self.delay: DelayLine | None = None

    # -- wiring ---------------------------------------------------------
    def with_delay(self, delay_ms: float) -> "SparseProjection":
        self.delay = DelayLine(delay_ms, self.pre_size)
        return self

    def reset(self, batch: int, device) -> None:
        if self.delay is not None:
            self.delay.reset(batch, device)
        self.trace_pre = torch.zeros(batch, self.pre_size, device=device)
        self.trace_post = torch.zeros(batch, self.post_size, device=device)
        #  ELIGIBILITY IS CLEARED BETWEEN TRIALS.
        #
        #  The tag decays with a 500 ms time constant and a percept lasts
        #  around 100 ms, so without this roughly 85% of the previous trial's
        #  eligibility is still present when the next one's dopamine arrives
        #  -- and that dopamine then reinforces synapses that participated in
        #  a DIFFERENT stimulus. Real trials are separated by enough time and
        #  enough intervening activity for the tag to wash out; discrete
        #  trials presented back to back are not, so it is cleared explicitly.
        self.eligibility.zero_()

    def lesion(self, cut: bool = True) -> None:
        self.gain.fill_(0.0 if cut else 1.0)

    # -- transmission ---------------------------------------------------
    def forward(self, pre_spikes: torch.Tensor) -> torch.Tensor:
        """Current delivered to the postsynaptic population this millisecond."""
        arriving = (self.delay.push_pop(pre_spikes) if self.delay is not None
                    else pre_spikes)
        #  Astrocytic gain: slow, local, activity-dependent. Bounded so it
        #  modulates rather than gates.
        modulated = self.weight * (1.0 + cfg.ASTROCYTE_STRENGTH
                                   * torch.tanh(self.astrocyte)).unsqueeze(0)
        return arriving @ modulated * self.gain

    # -- plasticity -----------------------------------------------------
    @torch.no_grad()
    def accumulate(self, pre_spikes: torch.Tensor,
                   post_spikes: torch.Tensor) -> None:
        """Update traces and deposit eligibility. No weight moves here.

        This is the second factor being recorded and held. Whether it becomes
        a lasting change depends on a neuromodulator that has not arrived yet
        and may never.
        """
        dt = cfg.DT_MS
        self.trace_pre = (self.trace_pre
                          * torch.exp(torch.tensor(-dt / cfg.STDP_TAU_PRE_MS))
                          + pre_spikes)
        self.trace_post = (self.trace_post
                           * torch.exp(torch.tensor(-dt / cfg.STDP_TAU_POST_MS))
                           + post_spikes)

        #  POTENTIATION: post fired, and pre had fired recently before it.
        #  Averaged over the batch because eligibility is per-synapse and the
        #  synapse is shared across the batch dimension.
        potentiate = self.trace_pre.transpose(0, 1) @ post_spikes / max(
            pre_spikes.shape[0], 1)
        #  DEPRESSION: pre fired, and post had fired recently before it -- so
        #  pre arrived too late to have helped cause anything.
        depress = pre_spikes.transpose(0, 1) @ self.trace_post / max(
            pre_spikes.shape[0], 1)

        delta = cfg.STDP_A_PLUS * potentiate - cfg.STDP_A_MINUS * depress

        if self.dopamine_gated:
            #  Held as a tag. Nothing moves until a neuromodulator arrives.
            decay = torch.exp(torch.tensor(-dt / cfg.TAU_ELIGIBILITY_MS))
            self.eligibility = self.eligibility * decay + delta * self.mask
        else:
            #  Plain STDP, applied immediately and unsupervised. This is what
            #  lets the sensory pathway become selective without anyone
            #  telling it what the right answer is.
            self.weight += cfg.UNSUPERVISED_RATE * delta * self.mask
            self.weight.clamp_(cfg.WEIGHT_MIN, cfg.WEIGHT_MAX)

        #  Astrocytes track postsynaptic activity on their own slow clock.
        a_decay = dt / cfg.ASTROCYTE_TAU_MS
        self.astrocyte += a_decay * (post_spikes.mean(dim=0)
                                     * (1000.0 / dt) / cfg.TARGET_RATE_HZ
                                     - 1.0 - self.astrocyte)

    @torch.no_grad()
    def consolidate(self, neuromodulator: float) -> None:
        """The third factor arrives. Eligible synapses change; the rest do not.

        A neuromodulator level of zero means nothing happens no matter how
        much eligibility was accumulated -- coincidence without consequence is
        not learning, and this is where v3 refuses to do what a Hebbian
        network would.
        """
        if neuromodulator == 0.0:
            return
        self.weight += (cfg.LEARNING_RATE * neuromodulator
                        * self.eligibility * self.mask)
        #  Dale's law and a physical ceiling. A synapse has a finite number of
        #  receptors; it cannot strengthen without bound.
        self.weight.clamp_(cfg.WEIGHT_MIN, cfg.WEIGHT_MAX)
        self.weight *= self.mask


__all__ = ["SparseProjection", "DelayLine"]
