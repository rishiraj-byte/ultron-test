"""
neurons.py -- leaky integrate-and-fire cells with two dendritic compartments.

WHAT REPLACED WHAT. In v2 a "neuron" was one number in a 256-vector, and its
value was its output. Here a neuron has a membrane potential that charges and
leaks, a threshold it must cross, a refractory period during which it cannot
fire however hard it is pushed, and an output that is a single bit: it spiked
this millisecond, or it did not.

That change is not cosmetic and it costs something real. A binary spike
carries less per event than a float, so information has to live in WHICH cells
fire and WHEN -- which is the point, because that is where the brain puts it.

TWO COMPARTMENTS, AND WHY IT MATTERS MORE THAN IT LOOKS.

A real pyramidal cell is not a point. Feedforward input arrives on basal
dendrites close to the soma and can fire the cell. Feedback from higher areas
arrives on the apical tuft, hundreds of microns away, and on its own it
essentially cannot -- what it does instead is change how the cell responds to
its basal input.

That asymmetry is how cortex keeps "what I am being told" separate from "what
I expect" inside a single cell. v2 summed feedforward and feedback into one
drive and had no way to express the difference; the laminar shell separated
them by layer on the way in and then added them anyway. Here the separation
survives all the way to the membrane.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from v3 import config_v3 as cfg


class LIFPopulation(nn.Module):
    """A population of leaky integrate-and-fire neurons, all one sign.

    Dale's principle is enforced by the population, not by the weights: this
    object is excitatory or inhibitory for its whole life, and every synapse
    leaving it carries that sign. A weight can go to zero here but it can
    never cross to the other side, because the sign is not stored in the
    weight at all.
    """

    def __init__(self, size: int, inhibitory: bool = False,
                 name: str = "population", target_hz: float | None = None):
        super().__init__()
        self.size = size
        self.inhibitory = inhibitory
        self.name = name
        #  Every population defends its OWN target rate. Most cortical cells
        #  want a few Hz; GPi wants sixty, because a tonically active output
        #  nucleus that drifts down to cortical rates has stopped being a
        #  brake. Making the target per-population rather than global is what
        #  lets one homeostatic mechanism serve both.
        self.target_hz = (cfg.TARGET_RATE_HZ if target_hz is None
                          else target_hz)

        tau_m = cfg.TAU_M_INH_MS if inhibitory else cfg.TAU_M_EXC_MS
        self.register_buffer("tau_m", torch.tensor(tau_m))

        #  Per-neuron excitability, adjusted by homeostasis over seconds. Real
        #  cells defend a target firing rate by moving their own threshold;
        #  this is that, and it is what stops a purely Hebbian network from
        #  either falling silent or seizing.
        self.threshold = nn.Parameter(
            torch.full((size,), cfg.V_THRESHOLD_MV), requires_grad=False)

        #  Running estimate of this neuron's own firing rate, which is the
        #  only thing homeostasis is allowed to see.
        #  STARTS AT THE TARGET, not at zero.
        #
        #  Initialised at zero, this reads "catastrophically underactive" on
        #  the very first percept of an untrained network that is in fact
        #  already firing at the right rate. Homeostasis then spends dozens of
        #  percepts correcting an error that does not exist, lowering
        #  thresholds and raising synaptic scale until the network seizes --
        #  measured: 4.05 Hz at epoch 0 driven to 77 Hz by epoch 21, with the
        #  estimate still lagging at a third of the true value the whole way.
        #
        #  A control loop whose sensor lags its actuator oscillates. The
        #  sensor here is the rate estimate, and starting it at the set point
        #  means an already-correct network is left alone.
        self.register_buffer("rate_estimate",
                             torch.full((size,), float(self.target_hz)))

        #  SYNAPTIC SCALING (Turrigiano). A neuron firing persistently above
        #  its set point scales ALL of its excitatory inputs down together,
        #  multiplicatively, preserving their relative strengths. Threshold
        #  adjustment alone cannot fix an input that is an order of magnitude
        #  too strong -- it just saturates against the clamp, which is exactly
        #  what happened here: cortex sat at 10-35 Hz against a 5 Hz target
        #  with thresholds pinned at their ceiling.
        #
        #  This is the second homeostatic mechanism and it is a real one, not
        #  a normalisation layer bolted on. It is slow, it is postsynaptic,
        #  and it cannot change which inputs matter -- only how loud they all
        #  are together.
        self.register_buffer("input_scale", torch.ones(size))

        #  Spikes accumulated across the current percept. Homeostasis reads
        #  THIS, once, at the end -- not the instantaneous spike vector.
        #
        #  Applying it every millisecond compounded the multiplicative scaling
        #  120 times per percept and overshot by a factor of forty, which is
        #  what produced the bistability: one epoch silent, the next seizing.
        #  Real synaptic scaling operates over hours. Sampling it per percept
        #  is still far faster than that, and it is the slowest this model can
        #  go and still converge in reasonable wall-clock time.
        self.register_buffer("spike_count", torch.zeros(size))
        self.register_buffer("step_count", torch.zeros(()))

        self._state: dict[str, torch.Tensor] = {}

    # -- state ----------------------------------------------------------
    def reset(self, batch: int, device) -> None:
        """Start a fresh percept. Membranes begin at rest, nothing pending."""
        zeros = lambda: torch.zeros(batch, self.size, device=device)
        self._state = {
            "v": torch.full((batch, self.size), cfg.V_REST_MV, device=device),
            "i_exc": zeros(),
            "i_inh": zeros(),
            "i_apical": zeros(),
            "refractory": zeros(),
            "spikes": zeros(),
        }

    @property
    def spikes(self) -> torch.Tensor:
        """Who fired on the last step. Binary, and the only public output."""
        return self._state["spikes"]

    @property
    def voltage(self) -> torch.Tensor:
        return self._state["v"]

    # -- input ----------------------------------------------------------
    def receive(self, current: torch.Tensor, kind: str = "exc") -> None:
        """Deliver synaptic current arriving this millisecond.

        `kind` is "exc", "inh" or "apical". The first two sum into the soma
        and can fire the cell. The third lands on the apical tuft and cannot
        -- see `step` for what it does instead.
        """
        #  "apical_inh" is SST inhibition landing ON the tuft: it subtracts
        #  from the feedback current rather than from the soma, so it gates
        #  what CONTEXT reaches the cell without changing the cell's own
        #  excitability. Somatic and dendritic inhibition are different
        #  operations, and collapsing them loses the very distinction that
        #  VIP disinhibition exists to exploit.
        if kind == "apical_inh":
            self._state["i_apical"] = self._state["i_apical"] - current
            return
        key = {"exc": "i_exc", "inh": "i_inh", "apical": "i_apical"}[kind]
        if kind == "exc":
            current = current * self.input_scale.unsqueeze(0)
        self._state[key] = self._state[key] + current

    # -- the millisecond ------------------------------------------------
    def step(self) -> torch.Tensor:
        """Advance one DT_MS. Returns this step's spikes."""
        s = self._state
        dt = cfg.DT_MS

        #  Synaptic currents decay with their own kinetics. Inhibition is
        #  slower than excitation, so it outlasts the excitation that
        #  recruited it -- which is what makes runaway hard.
        s["i_exc"] = s["i_exc"] * torch.exp(torch.tensor(-dt / cfg.TAU_SYN_EXC_MS))
        s["i_inh"] = s["i_inh"] * torch.exp(torch.tensor(-dt / cfg.TAU_SYN_INH_MS))
        s["i_apical"] = s["i_apical"] * torch.exp(
            torch.tensor(-dt / cfg.TAU_SYN_EXC_MS))

        #  APICAL MODULATION. The tuft multiplies the somatic drive rather
        #  than adding to it, and the multiplier is bounded below by
        #  APICAL_DRIVE_ALONE so that feedback with no feedforward input
        #  produces nothing. Expectation cannot manufacture a percept; it can
        #  only amplify or suppress one that the senses supplied.
        #  Ongoing background bombardment. See cfg.BACKGROUND_MEAN for why
        #  the network has no stable operating point without it.
        background = (cfg.BACKGROUND_MEAN
                      + cfg.BACKGROUND_SIGMA
                      * torch.randn_like(s["v"]))

        #  APICAL GAIN APPLIES TO SYNAPTIC DRIVE ONLY, never to background.
        #
        #  Written the other way round it multiplied the background barrage
        #  too, so a cell receiving nothing but feedback still fired 291 times
        #  in 100 ms -- feedback was manufacturing activity out of noise,
        #  which is the exact failure mode this compartment exists to prevent.
        #  Expectation modulates how you respond to what your senses deliver;
        #  it does not amplify the spontaneous hum of the tissue.
        somatic = s["i_exc"] - s["i_inh"]
        gain = 1.0 + cfg.APICAL_GAIN * torch.tanh(s["i_apical"])
        drive = somatic * gain + background

        #  Leak toward rest, plus whatever arrived.
        dv = (cfg.V_REST_MV - s["v"]) / self.tau_m * dt + drive

        #  A refractory cell integrates nothing. This is the rate ceiling, and
        #  it is a limitation being kept rather than removed: no matter how
        #  hard this cell is driven it cannot exceed 1000/REFRACTORY_MS Hz.
        awake = (s["refractory"] <= 0).float()
        s["v"] = s["v"] + dv * awake
        s["refractory"] = torch.clamp(s["refractory"] - dt, min=0.0)

        fired = (s["v"] >= self.threshold.unsqueeze(0)).float() * awake
        s["v"] = torch.where(fired > 0,
                             torch.full_like(s["v"], cfg.V_RESET_MV), s["v"])
        s["refractory"] = torch.where(
            fired > 0, torch.full_like(s["refractory"], cfg.REFRACTORY_MS),
            s["refractory"])
        s["spikes"] = fired
        self.spike_count += fired.mean(dim=0)
        self.step_count += 1
        return fired

    # -- homeostasis ----------------------------------------------------
    @torch.no_grad()
    def homeostatic_update(self, speed: float = 1.0) -> None:
        """Nudge each neuron's own threshold toward its target rate.

        Deliberately slow -- TAU_HOMEOSTATIC_MS is ten seconds, thousands of
        times slower than a synapse. A homeostatic mechanism fast enough to
        track a percept would erase the percept.
        """
        if float(self.step_count) < 1:
            return
        #  This percept's actual mean rate, in Hz.
        rate_hz = self.spike_count / self.step_count * (1000.0 / cfg.DT_MS)
        self.spike_count.zero_()
        self.step_count.zero_()

        #  Blended over a few percepts, not a few thousand. The per-percept
        #  rate is already averaged over hundreds of milliseconds, so it is
        #  not noisy enough to need heavy smoothing -- and heavy smoothing is
        #  precisely the lag that made this oscillate.
        blend = min(speed * cfg.RATE_ESTIMATE_BLEND, 1.0)
        self.rate_estimate += blend * (rate_hz - self.rate_estimate)

        #  Firing too much raises the bar; too little lowers it.
        error = self.rate_estimate - self.target_hz
        self.threshold += speed * cfg.HOMEOSTATIC_RATE * error
        self.threshold.clamp_(cfg.V_REST_MV + 2.0, cfg.V_REST_MV + 40.0)

        #  Multiplicative, and relative to the set point rather than absolute,
        #  so a cell firing at twice its target scales down by the same
        #  proportion whatever the absolute rates happen to be.
        #  Bounded per update so one bad percept cannot move the operating
        #  point far. Multiplicative and relative to the set point.
        ratio = (self.rate_estimate + 1.0) / (self.target_hz + 1.0)
        step = (speed * cfg.SCALING_RATE * (ratio - 1.0)).clamp(-0.1, 0.1)
        self.input_scale *= (1.0 - step)
        self.input_scale.clamp_(0.02, 20.0)


class CorticalArea(nn.Module):
    """One cortical area: an excitatory population and its inhibitory partner.

    The 80/20 split and the local wiring between them is the microcircuit
    every cortical area shares. Excitatory cells drive each other and drive
    the interneurons; the interneurons inhibit the excitatory cells and each
    other. That recurrent inhibition is what produces sparse activity here --
    unlike v2, where sparsity was imposed by selecting the top 5% and zeroing
    everything else by hand.

    This is one of the places v3 removes an artificial mechanism rather than
    adding one. Nothing sorts anything. Cells compete, and inhibition settles
    who wins.
    """

    def __init__(self, name: str):
        super().__init__()
        self.name = name
        self.exc = LIFPopulation(cfg.EXC_PER_AREA, inhibitory=False,
                                 name="%s.exc" % name)
        #  Three interneuron classes with three different targets. See
        #  cfg.PV_PER_AREA for what each does and why one type is not enough.
        self.pv = LIFPopulation(cfg.PV_PER_AREA, inhibitory=True,
                                name="%s.pv" % name)
        self.sst = LIFPopulation(cfg.SST_PER_AREA, inhibitory=True,
                                 name="%s.sst" % name)
        self.vip = LIFPopulation(cfg.VIP_PER_AREA, inhibitory=True,
                                 name="%s.vip" % name)

        from v3.synapses import SparseProjection
        #  Local microcircuit. Sparse and fixed, like real local connectivity:
        #  a pyramidal cell contacts about a tenth of its neighbours, not all
        #  of them.
        self.e_to_e = SparseProjection(cfg.EXC_PER_AREA, cfg.EXC_PER_AREA,
                                       cfg.P_CONNECT_LOCAL, "exc", "%s.ee" % name)
        self.e_to_pv = SparseProjection(cfg.EXC_PER_AREA, cfg.PV_PER_AREA,
                                        cfg.P_CONNECT_LOCAL * 2, "exc",
                                        "%s.e_pv" % name)
        self.e_to_sst = SparseProjection(cfg.EXC_PER_AREA, cfg.SST_PER_AREA,
                                         cfg.P_CONNECT_LOCAL, "exc",
                                         "%s.e_sst" % name)
        #  PV onto the soma: gain control, and the fast loop that keeps the
        #  network out of runaway.
        self.pv_to_e = SparseProjection(cfg.PV_PER_AREA, cfg.EXC_PER_AREA,
                                        cfg.P_CONNECT_INHIBITORY, "inh",
                                        "%s.pv_e" % name)
        self.pv_to_pv = SparseProjection(cfg.PV_PER_AREA, cfg.PV_PER_AREA,
                                         cfg.P_CONNECT_LOCAL, "inh",
                                         "%s.pv_pv" % name)
        #  SST onto the tuft: decides whether context reaches the cell.
        self.sst_to_e = SparseProjection(cfg.SST_PER_AREA, cfg.EXC_PER_AREA,
                                         cfg.P_CONNECT_SST, "inh",
                                         "%s.sst_e" % name)
        #  VIP onto SST: the disinhibition switch that opens that door.
        self.vip_to_sst = SparseProjection(cfg.VIP_PER_AREA, cfg.SST_PER_AREA,
                                           cfg.P_CONNECT_VIP, "inh",
                                           "%s.vip_sst" % name)

    @property
    def inh(self):
        """The PV population, for callers written before the three-type split."""
        return self.pv

    def interneurons(self):
        return (self.pv, self.sst, self.vip)

    def reset(self, batch: int, device) -> None:
        self.exc.reset(batch, device)
        for pop in self.interneurons():
            pop.reset(batch, device)

    def step_local(self) -> None:
        """Deliver this area's own recurrent traffic, using LAST step's spikes.

        Reading the previous step is not a convenience -- a synapse cannot
        transmit a spike that has not happened yet, and the delay is real. It
        is the same principle v2 enforced with integer tract delays, now
        falling out of the physics instead of being asserted.
        """
        e = self.exc.spikes
        self.exc.receive(self.e_to_e(e), "exc")
        self.pv.receive(self.e_to_pv(e), "exc")
        self.sst.receive(self.e_to_sst(e), "exc")

        #  PV inhibits the soma, SST inhibits the tuft, VIP inhibits SST.
        #  Three inhibitory populations doing three different jobs.
        self.exc.receive(self.pv_to_e(self.pv.spikes), "inh")
        self.pv.receive(self.pv_to_pv(self.pv.spikes), "inh")
        self.exc.receive(self.sst_to_e(self.sst.spikes), "apical_inh")
        self.sst.receive(self.vip_to_sst(self.vip.spikes), "inh")

    def drive_vip(self, current: torch.Tensor) -> None:
        """Top-down input onto VIP, which opens the apical door.

        This is the attention pathway, and note what it does NOT do: it does
        not excite pyramidal cells at all. It silences SST, which stops
        clamping the apical tuft, which lets feedback influence what the cell
        makes of what its senses are giving it. Attention as disinhibition
        rather than as amplification.
        """
        self.vip.receive(current, "exc")

    def step(self) -> torch.Tensor:
        self.exc.step()
        for pop in self.interneurons():
            pop.step()
        return self.exc.spikes

    def firing_rate_hz(self) -> torch.Tensor:
        return self.exc.spikes.mean() * (1000.0 / cfg.DT_MS)


__all__ = ["LIFPopulation", "CorticalArea"]
