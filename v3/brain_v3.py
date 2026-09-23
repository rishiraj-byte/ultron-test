"""
brain_v3.py -- the whole thing, assembled, running in milliseconds.

THE ONE CORRECTION THAT MATTERED MOST. In v2 the answer was read straight off
cortical state: `readout()` concatenated the four lobes and handed them to a
classifier head. That tap bypassed the basal ganglia completely, so the gate
could be slammed fully shut and the answer still came out. Measuring it showed
the gate contributing nothing to the output because it was not in the output
path at all.

Here there is no tap. A decision exists only when a motor population sustains
a high enough firing rate, and motor cortex is driven through the VA/VL
thalamic nucleus, which GPi holds shut by default. Close the gate and the
system emits nothing -- not a low-confidence answer, nothing. That is what a
gate is for, and it is why frontal and the basal ganglia can finally be
measured honestly.

WHAT THAT COSTS. The system can now decline to answer by simply never
crossing threshold, and on hard inputs it will. v2 always produced a label.
This is a drawback carried forward on purpose: a brain that has not decided
does not emit a guess.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from v3 import config_v3 as cfg
from v3.neurons import CorticalArea, LIFPopulation
from v3.regions import (Amygdala, BasalGanglia, Cerebellum, Hippocampus,
                        Hypothalamus, Thalamus)
from v3.synapses import SparseProjection


#  source -> target, by tract name. Same connectome as v2, now with real
#  conduction delays and one addition: v2 had five feedback tracts against
#  thirteen feedforward, and cortex has at least as much feedback as forward.
CORTICAL_TRACTS = {
    "occ_par":        ("occipital", "parietal", "basal"),
    "ilf":            ("occipital", "temporal", "basal"),
    "ifof":           ("occipital", "frontal",  "basal"),
    "slf":            ("parietal",  "frontal",  "basal"),
    "arcuate":        ("temporal",  "frontal",  "basal"),
    #  FEEDBACK lands on the APICAL compartment, which is the whole reason
    #  two compartments exist. It cannot fire a cell by itself; it can only
    #  change how that cell responds to what its senses are telling it.
    "fb_frontal_occ": ("frontal",   "occipital", "apical"),
    "fb_frontal_par": ("frontal",   "parietal",  "apical"),
    "fb_frontal_tem": ("frontal",   "temporal",  "apical"),
    "fb_par_occ":     ("parietal",  "occipital", "apical"),
    "fb_tem_occ":     ("temporal",  "occipital", "apical"),
    "lat_par_occ":    ("parietal",  "occipital", "apical"),
}

#  Which nucleus each sense enters by. Unlike v2, a sense does NOT reach every
#  lobe -- it enters its own nucleus and spreads through the connectome, which
#  is what actually happens. The all-to-all routing v2 adopted was a fix for a
#  measurement problem that does not exist here, because feedback now carries
#  information to areas the input never directly touched.
SENSE_NUCLEUS = {"image": "LGN", "audio": "MGN", "text": "PULVINAR",
                 "touch": "VPL"}
NUCLEUS_TARGET = {"LGN": "occipital", "MGN": "temporal",
                  "PULVINAR": "temporal", "VPL": "parietal"}
RADIATION = {"LGN": "optic_radiation", "MGN": "acoustic_radiation",
             "PULVINAR": "pulvinar_temporal", "VPL": "somato_radiation"}


class BrainV3(nn.Module):
    """Four cortical areas, six thalamic nuclei, five subcortical systems."""

    LOBES = ("occipital", "parietal", "temporal", "frontal")

    def __init__(self, n_actions: int = 20):
        super().__init__()
        torch.manual_seed(cfg.SEED)
        self.n_actions = n_actions

        self.areas = nn.ModuleDict({n: CorticalArea(n) for n in self.LOBES})
        self.thalamus = Thalamus()

        self.tracts = nn.ModuleDict()
        for name, (src, dst, compartment) in CORTICAL_TRACTS.items():
            self.tracts[name] = SparseProjection(
                cfg.EXC_PER_AREA, cfg.EXC_PER_AREA, cfg.P_CONNECT_TRACT,
                "exc", name).with_delay(cfg.DELAY_MS[name])

        #  Thalamus into cortex, and cortex layer 6 back down.
        self.radiations = nn.ModuleDict({
            #  Strong but FEW. A driver synapse is powerful; a cortical cell
            #  still only receives a handful of them, and thalamic input is a
            #  small minority of layer 4's total synapses. Connecting 10% of
            #  256 relay cells at driver strength gave every cortical cell 25
            #  powerful afferents and drove the whole sheet into saturation.
            nuc: SparseProjection(Thalamus.RELAY_SIZE, cfg.EXC_PER_AREA, 0.02,
                                  "exc", RADIATION[nuc], driver=True).with_delay(
                cfg.DELAY_MS[RADIATION[nuc]])
            for nuc in RADIATION})
        self.corticothalamic = nn.ModuleDict({
            n: SparseProjection(cfg.EXC_PER_AREA, Thalamus.RELAY_SIZE, 0.05,
                                "exc", "ct_%s" % n) for n in self.LOBES})

        size = cfg.EXC_PER_AREA
        self.basal_ganglia = BasalGanglia(size)
        self.amygdala = Amygdala(size, Thalamus.RELAY_SIZE)
        self.hippocampus = Hippocampus(size)
        self.cerebellum = Cerebellum(size)
        self.hypothalamus = Hypothalamus(size)
        #  The four diffuse nuclei. Small, slow, and projecting everywhere.
        from v3.modulators import Modulators
        self.modulators = Modulators(size)

        #  MOTOR CORTEX -- the only way out. Driven from VA/VL, which GPi
        #  holds shut. One population per candidate action; whichever sustains
        #  threshold first is the decision.
        self.motor = LIFPopulation(n_actions * cfg.MOTOR_POPULATION,
                                   name="motor")
        self.vavl_motor = SparseProjection(
            Thalamus.RELAY_SIZE, n_actions * cfg.MOTOR_POPULATION, 0.10,
            "exc", "vavl_motor", driver=True,
            dopamine_gated=True).with_delay(
            cfg.DELAY_MS["vavl_frontal"])
        #  Frontal reaches motor cortex too, but on the APICAL compartment --
        #  it MODULATES which action is favoured and cannot initiate one.
        #
        #  Put on the basal compartment, as it was first, this projection
        #  could fire motor cells by itself and simply substituted for the
        #  gated thalamic drive: forcing the gate shut then RAISED motor
        #  output to 38.8 Hz. Action initiation in a real brain goes through
        #  the basal-ganglia-thalamocortical loop; cortex biases which action
        #  is released, it does not release one on its own.
        #  The action-selection synapse: this one genuinely needs a reward
        #  signal, because which action is correct is not deducible from the
        #  statistics of the input alone.
        self.frontal_motor = SparseProjection(
            size, n_actions * cfg.MOTOR_POPULATION, 0.05, "exc",
            "frontal_motor", dopamine_gated=True).with_delay(6.0)
        self.md_frontal = SparseProjection(
            Thalamus.RELAY_SIZE, size, 0.02, "exc", "md_frontal",
            driver=True).with_delay(cfg.DELAY_MS["md_frontal"])
        self.gpi_thalamus = SparseProjection(
            BasalGanglia.PALLIDUM, Thalamus.RELAY_SIZE, 0.20, "inh",
            "gpi_thalamus").with_delay(cfg.DELAY_MS["gpi_thalamus"])

        self.dopamine = 0.0
        self.chemistry: dict = {}
        #  Spontaneous exploratory drive on motor cortex. See
        #  cfg.EXPLORATION_DRIVE: without it an untrained brain never acts and
        #  reward learning cannot start.
        self.exploration = cfg.EXPLORATION_DRIVE
        #  Which action is being taught this trial, if any. See `teach`.
        self.teacher: torch.Tensor | None = None

    # -- lifecycle ------------------------------------------------------
    def all_projections(self):
        for module in self.modules():
            if isinstance(module, SparseProjection):
                yield module

    def reset(self, batch: int, device) -> None:
        for area in self.areas.values():
            area.reset(batch, device)
        for module in (self.thalamus, self.basal_ganglia, self.amygdala,
                       self.hippocampus, self.cerebellum, self.hypothalamus):
            module.reset(batch, device)
        self.motor.reset(batch, device)
        self.modulators.reset(batch, device)
        for proj in self.all_projections():
            proj.reset(batch, device)

    # -- teaching -------------------------------------------------------
    def teach(self, action: torch.Tensor | None) -> None:
        """Drive the correct motor population, or stop driving one.

        SUPERVISED HEBBIAN ASSOCIATION, and it is not a gradient.

        Trial-and-error with a scalar reward is the hardest possible way to
        learn a mapping: the brain must stumble onto the right action by
        chance before anything can be reinforced, and with five options and a
        deep random network it mostly does not. Measured over 6,000 trials it
        reached 26% against 20% chance.

        Guided learning is what animals mostly get instead. A parent moves the
        child's hand; a teacher supplies the answer; the correct response is
        DRIVEN while the stimulus is present, and plain Hebbian plasticity
        associates the two. Everything about it is local -- a synapse sees its
        own pre and post cells firing together and strengthens. Nothing is
        transported backward and no error is computed anywhere.

        Passing None ends teaching, which is what evaluation does.
        """
        self.teacher = action

    # -- one millisecond ------------------------------------------------
    def step(self, sensory: dict[str, torch.Tensor], learn: bool = False) -> dict:
        """Advance the entire brain by DT_MS.

        `sensory` maps a nucleus name to the spikes arriving from the sense
        organs this millisecond. Every other signal in here comes from the
        previous millisecond, because that is how long a synapse takes.
        """
        area_spikes = {n: self.areas[n].exc.spikes for n in self.LOBES}

        # ---- thalamus, under cortical and pallidal control ----
        for nucleus, spikes in sensory.items():
            #  Retinal and cochlear afferents are DRIVERS -- a few of them
            #  fire a relay cell outright. See cfg.DRIVER_MV.
            self.thalamus.relay[nucleus].receive(spikes * cfg.DRIVER_MV, "exc")
        for lobe in self.LOBES:
            #  Layer 6 back to the thalamus -- more of it than there is
            #  feedforward, which is the real ratio.
            back = self.corticothalamic[lobe](area_spikes[lobe])
            #  VA/VL is included, and it has to be. The basal ganglia gates
            #  this nucleus, and a gate on a nucleus with no excitatory drive
            #  gates nothing: with VA/VL fed only by the cerebellum, forcing
            #  the gate open and forcing it shut produced the same 0.9 Hz at
            #  motor cortex, because there was nothing there to let through.
            #
            #  The loop is cortex -> thalamus -> cortex, with the basal
            #  ganglia deciding whether the return leg is permitted. Both
            #  legs have to exist for the decision to mean anything.
            for nucleus in ("PULVINAR", "MD", "VA_VL"):
                self.thalamus.relay[nucleus].receive(back, "exc")

        gpi = self.basal_ganglia.gpi.spikes
        #  Scaled so the gate has a working range. See cfg.GPI_INHIBITION_GAIN
        #  for what happens without it: a gate with only two positions, both
        #  useless.
        inhibition = self.gpi_thalamus(gpi) * cfg.GPI_INHIBITION_GAIN
        for nucleus in Thalamus.GATED:
            self.thalamus.relay[nucleus].receive(inhibition, "inh")

        self.cerebellum.step(area_spikes["frontal"])
        self.thalamus.relay["VA_VL"].receive(
            self.cerebellum.out(self.cerebellum.deep.spikes), "exc")

        relayed = self.thalamus.step()

        # ---- cortex ----
        for nucleus, lobe in NUCLEUS_TARGET.items():
            self.areas[lobe].exc.receive(
                self.radiations[nucleus](relayed[nucleus]), "exc")
        #  MD is the prefrontal return door and it needs its own projection.
        #  This previously reused the pulvinar radiation multiplied by zero --
        #  a gated nucleus wired to nothing, which made half the basal
        #  ganglia's influence on cortex silently inert.
        self.areas["frontal"].exc.receive(
            self.md_frontal(relayed["MD"]), "exc")

        for name, (src, dst, compartment) in CORTICAL_TRACTS.items():
            current = self.tracts[name](area_spikes[src])
            kind = "apical" if compartment == "apical" else "exc"
            self.areas[dst].exc.receive(current, kind)
            if compartment == "apical":
                #  A feedback axon does not only reach the tuft -- it also
                #  contacts VIP cells, which silence SST, which stops
                #  clamping that same tuft. So top-down input both arrives at
                #  the dendrite AND opens the door it arrives through.
                #
                #  Without this the VIP population has no afferent at all and
                #  the disinhibition motif is anatomy that never fires.
                self.areas[dst].drive_vip(
                    current.mean(dim=1, keepdim=True)
                    .expand(-1, cfg.VIP_PER_AREA) * cfg.VIP_FEEDBACK_GAIN)

        amy = self.amygdala.step(relayed["PULVINAR"], area_spikes["temporal"])
        self.areas["frontal"].exc.receive(self.amygdala.to_cortex(amy), "exc")

        hip = self.hippocampus.step(area_spikes["temporal"])
        self.areas["frontal"].exc.receive(self.hippocampus.out(hip), "exc")

        #  Neuromodulators read salience from the amygdala and novelty from
        #  CA1, which are the signals that actually drive them.
        chem = self.modulators.step(amy.mean(dim=1, keepdim=True)
                                    .expand(-1, self.modulators
                                            .locus_coeruleus.cells.size),
                                    hip.mean(dim=1, keepdim=True)
                                    .expand(-1, self.modulators
                                            .basal_forebrain.cells.size))
        self.chemistry = chem

        hyp = self.hypothalamus.step(amy)
        self.areas["frontal"].exc.receive(self.hypothalamus.to_cortex(hyp), "exc")

        for area in self.areas.values():
            area.step_local()
            area.step()

        # ---- basal ganglia, reading frontal ----
        if self.exploration > 0.0:
            #  And this is what actually gets an untrained brain moving: a
            #  restless direct pathway. Extra drive on D1 quiets GPi, the gate
            #  opens, and SOMETHING is released -- which is the only way a
            #  reward-driven system ever gets a first reward to learn from.
            self.basal_ganglia.d1.receive(
                torch.randn_like(self.basal_ganglia.d1.voltage)
                * self.exploration * cfg.EXPLORATION_STRIATAL, "exc")
        self.basal_ganglia.step(area_spikes["frontal"])

        # ---- motor: the ONLY exit, and it goes through the gate ----
        self.motor.receive(
            self.vavl_motor(relayed["VA_VL"]) * cfg.MOTOR_PERMIT_GAIN, "exc")
        if self.exploration > 0.0:
            #  Exploration BIASES which action wins; it does not release one.
            #  Driving motor cortex directly let noise fire actions with the
            #  gate held shut -- 73.8 Hz against a 40 Hz threshold -- which
            #  makes the gate decorative again. Spontaneous action in a real
            #  brain is still gated; what varies is which option the striatum
            #  releases, not whether the pallidum is bypassed.
            batch = self.motor.spikes.shape[0]
            per_action = torch.randn(batch, self.n_actions, 1,
                                     device=self.motor.spikes.device)
            spread = per_action.expand(-1, -1, cfg.MOTOR_POPULATION)
            self.motor.receive(
                spread.reshape(batch, -1) * self.exploration, "apical")
        #  COINCIDENCE, NOT SUBSTITUTION.
        #
        #  Cortex selects WHICH action; the basal ganglia decides WHETHER any
        #  action is released. Motor cells need both and can be fired by
        #  neither alone -- which is the real arrangement, and the only one
        #  that satisfies both requirements at once.
        #
        #  Put frontal on the apical compartment (the previous fix) the gate
        #  became authoritative but nothing class-specific could drive motor
        #  at all: VA/VL is permissive and carries no information about what
        #  the stimulus was, so reward had nothing to select between and
        #  accuracy sat at chance through 8,000 trials. Put frontal on basal
        #  at full strength and it fires motor by itself, and the gate stops
        #  meaning anything. Sub-threshold on both is the arrangement that
        #  makes each necessary.
        self.motor.receive(
            self.frontal_motor(area_spikes["frontal"]) * cfg.MOTOR_SELECT_GAIN,
            "exc")
        if self.teacher is not None:
            #  Drive the correct population hard enough to fire it, so that
            #  the synapses carrying the current cortical pattern are active
            #  at the same moment and Hebbian plasticity can associate them.
            batch = self.motor.spikes.shape[0]
            mask = torch.zeros(batch, self.n_actions, cfg.MOTOR_POPULATION,
                               device=self.motor.spikes.device)
            mask[torch.arange(batch), self.teacher] = 1.0
            self.motor.receive(
                mask.reshape(batch, -1) * cfg.TEACHER_DRIVE, "exc")

        self.motor.step()

        if learn:
            self._accumulate(area_spikes, relayed)

        return {"motor": self.motor.spikes, "gpi": gpi,
                "cortex": {n: self.areas[n].exc.spikes for n in self.LOBES}}

    def _accumulate(self, area_spikes, relayed) -> None:
        """Deposit eligibility everywhere. No weight moves until dopamine."""
        for name, (src, dst, _c) in CORTICAL_TRACTS.items():
            self.tracts[name].accumulate(area_spikes[src],
                                         self.areas[dst].exc.spikes)
        for nucleus, lobe in NUCLEUS_TARGET.items():
            self.radiations[nucleus].accumulate(relayed[nucleus],
                                                self.areas[lobe].exc.spikes)
        self.frontal_motor.accumulate(area_spikes["frontal"], self.motor.spikes)
        self.vavl_motor.accumulate(relayed["VA_VL"], self.motor.spikes)
        self.basal_ganglia.cortex_d1.accumulate(area_spikes["frontal"],
                                                self.basal_ganglia.d1.spikes)
        self.basal_ganglia.cortex_d2.accumulate(area_spikes["frontal"],
                                                self.basal_ganglia.d2.spikes)

    # -- a whole percept ------------------------------------------------
    def perceive(self, spike_trains: dict[str, torch.Tensor],
                 learn: bool = False) -> dict:
        """Run PERCEPT_MS of simulated time on one input.

        `spike_trains` maps nucleus -> (steps, batch, relay_size) binary.
        """
        any_train = next(iter(spike_trains.values()))
        steps, batch = any_train.shape[0], any_train.shape[1]
        device = any_train.device
        self.reset(batch, device)

        motor_history = []
        gpi_history = []
        rates = {n: [] for n in self.LOBES}

        for t in range(steps):
            sensory = {nuc: train[t] for nuc, train in spike_trains.items()}
            out = self.step(sensory, learn=learn)
            motor_history.append(out["motor"])
            gpi_history.append(out["gpi"].mean())
            for n in self.LOBES:
                rates[n].append(out["cortex"][n].mean())

        motor = torch.stack(motor_history)             # (T, B, actions*pop)
        return {
            "motor": motor,
            "decision": self.read_decision(motor),
            "gpi_hz": torch.stack(gpi_history).mean() * (1000.0 / cfg.DT_MS),
            "rates_hz": {n: torch.stack(v).mean() * (1000.0 / cfg.DT_MS)
                         for n, v in rates.items()},
        }

    def read_decision(self, motor: torch.Tensor) -> dict:
        """Which action population sustained threshold, if any.

        Returns index -1 when nothing crossed. That is a real outcome, not a
        failure: an undecided brain emits no action.
        """
        window = int(cfg.MOTOR_WINDOW_MS / cfg.DT_MS)
        recent = motor[-window:]                       # (W, B, actions*pop)
        batch = recent.shape[1]
        per_action = recent.reshape(recent.shape[0], batch, self.n_actions,
                                    cfg.MOTOR_POPULATION)
        rate_hz = per_action.mean(dim=(0, 3)) * (1000.0 / cfg.DT_MS)  # (B, A)
        best_rate, best = rate_hz.max(dim=-1)
        decided = best_rate >= cfg.MOTOR_THRESHOLD_HZ
        return {"action": torch.where(decided, best,
                                      torch.full_like(best, -1)),
                "rate_hz": rate_hz, "decided": decided}

    # -- learning -------------------------------------------------------
    @torch.no_grad()
    def consolidate(self, reward: float) -> None:
        """Release dopamine proportional to prediction error, everywhere.

        A single scalar reaching every synapse at once is biologically
        correct -- dopamine is sprayed over whole territories, not addressed
        to individual connections. It carries no information about WHICH
        synapse was responsible; only the eligibility trace knows that.
        """
        self.dopamine = cfg.DOPAMINE_GAIN * reward
        for proj in self.all_projections():
            proj.consolidate(self.dopamine)

    def all_populations(self):
        for area in self.areas.values():
            yield area.exc
            #  All three interneuron classes, or homeostasis silently leaves
            #  two thirds of the inhibition unregulated.
            yield from area.interneurons()
        for nucleus in self.thalamus.relay.values():
            yield nucleus
        yield self.thalamus.trn
        yield from self.basal_ganglia.populations()
        yield self.amygdala.cells
        for pop in (self.hippocampus.ec, self.hippocampus.dg,
                    self.hippocampus.dg_inh, self.hippocampus.ca3,
                    self.hippocampus.ca1):
            yield pop
        for pop in (self.cerebellum.granule, self.cerebellum.purkinje,
                    self.cerebellum.deep):
            yield pop
        yield self.hypothalamus.cells
        yield self.motor

    @torch.no_grad()
    def homeostasis(self, speed: float = 1.0) -> None:
        """Every population nudges its own excitability toward its target.

        This is the ONLY mechanism that sets operating point in v3. There is
        no hand-tuned weight scale and no normalisation layer: cells that fire
        too much make themselves harder to fire, and the network finds its own
        balance. It is also how the E/I balance is maintained, which in v2
        required a hard top-5% mask applied from outside.
        """
        for pop in self.all_populations():
            pop.homeostatic_update(speed=speed)


__all__ = ["BrainV3", "CORTICAL_TRACTS", "SENSE_NUCLEUS"]
