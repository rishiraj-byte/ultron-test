"""
regions.py -- the subcortical machinery, rebuilt out of spiking cells.

Every module here existed in v2 as a stack of linear layers computing a float.
None of them compute a float any more. A thalamic gate is not a number between
0 and 1 -- it is whether the relay cells are hyperpolarised enough by GPi to
stop passing spikes. "Go" is not a scalar -- it is GPi cells falling silent.

That change removes a whole class of cheating. In v2 a gate value of 0.3 still
let 30% of the signal through, so nothing was ever truly gated. Here a
sufficiently inhibited relay cell emits nothing at all, and the downstream
area receives silence.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from v3 import config_v3 as cfg
from v3.neurons import LIFPopulation
from v3.synapses import SparseProjection


class Thalamus(nn.Module):
    """Six relay nuclei, every one of them gateable.

    THE RETICULAR NUCLEUS is a thin sheet of purely inhibitory cells wrapped
    around the thalamus. Every relay axon passing out to cortex sends a
    collateral into it, and it inhibits the relay cells back -- including
    NEIGHBOURING nuclei. So one nucleus becoming active suppresses the others.

    That is attention, and here it is built out of actual inhibitory cells
    competing rather than out of a softmax. v2 implemented this as normalised
    competition between gate values; the mechanism is the same idea, but this
    version can be silenced by lesioning the TRN population and the other
    cannot.
    """

    NUCLEI = ("LGN", "MGN", "VPL", "PULVINAR", "MD", "VA_VL")
    #  MD and VA/VL are the two the basal ganglia holds shut. They are the
    #  return doors into frontal and motor cortex, and they are the reason a
    #  decision has to be released rather than produced.
    GATED = ("MD", "VA_VL")
    RELAY_SIZE = 256
    TRN_SIZE = 128

    def __init__(self):
        super().__init__()
        self.relay = nn.ModuleDict({
            n: LIFPopulation(self.RELAY_SIZE, name="thal.%s" % n)
            for n in self.NUCLEI})
        self.trn = LIFPopulation(self.TRN_SIZE, inhibitory=True, name="thal.TRN")

        self.to_trn = nn.ModuleDict({
            n: SparseProjection(self.RELAY_SIZE, self.TRN_SIZE, 0.15, "exc",
                                "trn_in_%s" % n) for n in self.NUCLEI})
        self.from_trn = nn.ModuleDict({
            n: SparseProjection(self.TRN_SIZE, self.RELAY_SIZE, 0.15, "inh",
                                "trn_out_%s" % n) for n in self.NUCLEI})

    def reset(self, batch: int, device) -> None:
        for n in self.NUCLEI:
            self.relay[n].reset(batch, device)
            self.to_trn[n].reset(batch, device)
            self.from_trn[n].reset(batch, device)
        self.trn.reset(batch, device)

    def step(self) -> dict[str, torch.Tensor]:
        """Relay cells fire, TRN hears them, TRN inhibits everyone back."""
        for n in self.NUCLEI:
            self.trn.receive(self.to_trn[n](self.relay[n].spikes), "exc")
        self.trn.step()
        for n in self.NUCLEI:
            self.relay[n].receive(self.from_trn[n](self.trn.spikes), "inh")
            self.relay[n].step()
        return {n: self.relay[n].spikes for n in self.NUCLEI}


class BasalGanglia(nn.Module):
    """Direct, indirect and hyperdirect pathways, output tonically active.

    THE GPi NEVER STOPS FIRING on its own. It is given a constant depolarising
    current so that, left alone, it holds MD and VA/VL shut permanently and
    nothing reaches frontal or motor cortex. Every action is vetoed by default.

    The direct pathway is the only thing that can release it, and it does so
    by INHIBITING GPi -- striatal D1 cells are GABAergic. So "go" is a fall in
    GPi firing, and the thalamus becomes free to relay. That double negative
    is not a quirk; it is what lets one action be released while everything
    else stays suppressed.

    v2 had this shape too, but as arithmetic on floats. Here the tonic
    activity is a real current into real cells, and you can watch GPi rate
    drop when a decision is made.
    """

    STRIATUM = 512
    PALLIDUM = 256
    STN_SIZE = 128

    def __init__(self, cortex_size: int):
        super().__init__()
        #  Striatal projection cells are INHIBITORY -- both pathways.
        self.d1 = LIFPopulation(self.STRIATUM, inhibitory=True, name="bg.D1")
        self.d2 = LIFPopulation(self.STRIATUM, inhibitory=True, name="bg.D2")
        self.gpe = LIFPopulation(self.PALLIDUM, inhibitory=True, name="bg.GPe")
        #  GPi's target is its TONIC rate, not a cortical one. It is
        #  supposed to be loud.
        self.gpi = LIFPopulation(self.PALLIDUM, inhibitory=True, name="bg.GPi",
                                 target_hz=cfg.GPI_TONIC_HZ)
        #  STN is the one excitatory nucleus in the loop.
        self.stn = LIFPopulation(self.STN_SIZE, inhibitory=False, name="bg.STN")

        p = cfg.P_CONNECT_TRACT
        self.cortex_d1 = SparseProjection(cortex_size, self.STRIATUM, p, "exc",
                                          "corticostriatal_d1",
                                          dopamine_gated=True).with_delay(
            cfg.DELAY_MS["corticostriatal"])
        self.cortex_d2 = SparseProjection(cortex_size, self.STRIATUM, p, "exc",
                                          "corticostriatal_d2",
                                          dopamine_gated=True).with_delay(
            cfg.DELAY_MS["corticostriatal"])
        self.cortex_stn = SparseProjection(cortex_size, self.STN_SIZE, p, "exc",
                                           "hyperdirect").with_delay(
            cfg.DELAY_MS["hyperdirect"])

        self.d1_gpi = SparseProjection(self.STRIATUM, self.PALLIDUM, 0.15,
                                       "inh", "striatum_gpi").with_delay(
            cfg.DELAY_MS["striatum_gpi"])
        self.d2_gpe = SparseProjection(self.STRIATUM, self.PALLIDUM, 0.15,
                                       "inh", "striatum_gpe").with_delay(
            cfg.DELAY_MS["striatum_gpe"])
        self.gpe_stn = SparseProjection(self.PALLIDUM, self.STN_SIZE, 0.15,
                                        "inh", "gpe_stn").with_delay(
            cfg.DELAY_MS["gpe_stn"])
        self.stn_gpi = SparseProjection(self.STN_SIZE, self.PALLIDUM, 0.20,
                                        "exc", "stn_gpi").with_delay(
            cfg.DELAY_MS["stn_gpi"])

    def reset(self, batch: int, device) -> None:
        for pop in (self.d1, self.d2, self.gpe, self.gpi, self.stn):
            pop.reset(batch, device)
        for proj in (self.cortex_d1, self.cortex_d2, self.cortex_stn,
                     self.d1_gpi, self.d2_gpe, self.gpe_stn, self.stn_gpi):
            proj.reset(batch, device)

    def populations(self):
        return (self.d1, self.d2, self.gpe, self.gpi, self.stn)

    def step(self, cortex_spikes: torch.Tensor) -> torch.Tensor:
        """One millisecond. Returns GPi spikes -- the inhibition on thalamus."""
        self.d1.receive(self.cortex_d1(cortex_spikes), "exc")
        self.d2.receive(self.cortex_d2(cortex_spikes), "exc")
        self.stn.receive(self.cortex_stn(cortex_spikes), "exc")

        #  Indirect: striatum inhibits GPe, which stops inhibiting STN, which
        #  drives GPi harder. Three synapses, and every one of them real.
        self.gpe.receive(self.d2_gpe(self.d2.spikes), "inh")
        self.stn.receive(self.gpe_stn(self.gpe.spikes), "inh")

        #  TONIC DRIVE. Without this GPi is silent and the gate is open by
        #  default, which inverts the entire logic of the circuit.
        tonic = torch.full_like(self.gpi.voltage, cfg.GPI_TONIC_DRIVE)
        self.gpi.receive(tonic, "exc")
        self.gpi.receive(self.stn_gpi(self.stn.spikes), "exc")
        #  Direct: the only thing that can quiet GPi.
        self.gpi.receive(self.d1_gpi(self.d1.spikes), "inh")

        for pop in (self.d1, self.d2, self.gpe, self.stn, self.gpi):
            pop.step()
        return self.gpi.spikes


class Amygdala(nn.Module):
    """Two roads to the same conclusion, seventeen milliseconds apart.

    In v2 the fast and slow routes differed by one abstract timestep, which
    made the whole point nearly invisible. Here the low road is 8 ms and the
    high road is 25 ms, so the crude thalamic answer genuinely arrives and
    acts before the accurate cortical one has left temporal cortex.
    """

    SIZE = 256

    def __init__(self, cortex_size: int, relay_size: int):
        super().__init__()
        self.cells = LIFPopulation(self.SIZE, name="amygdala")
        self.low = SparseProjection(relay_size, self.SIZE, 0.12, "exc",
                                    "low_road").with_delay(cfg.DELAY_MS["low_road"])
        self.high = SparseProjection(cortex_size, self.SIZE, 0.08, "exc",
                                     "high_road").with_delay(cfg.DELAY_MS["high_road"])
        self.to_cortex = SparseProjection(self.SIZE, cortex_size, 0.05, "exc",
                                          "amygdala_cortex").with_delay(
            cfg.DELAY_MS["amygdala_cortex"])

    def reset(self, batch: int, device) -> None:
        self.cells.reset(batch, device)
        for p in (self.low, self.high, self.to_cortex):
            p.reset(batch, device)

    def step(self, thalamic: torch.Tensor, cortical: torch.Tensor) -> torch.Tensor:
        self.cells.receive(self.low(thalamic), "exc")
        self.cells.receive(self.high(cortical), "exc")
        self.cells.step()
        return self.cells.spikes


class Hippocampus(nn.Module):
    """Separate, complete, compare -- as four spiking populations.

    THE DENTATE GYRUS is the reason this works. It expands into a much larger
    population held at very sparse activity, so two similar inputs land on
    almost non-overlapping cell sets. That is pattern separation, and here it
    is produced by strong feedback inhibition rather than by taking a top-k.

    CA3 is recurrently wired to itself, so a fragment of a stored pattern
    pulls the rest back -- pattern completion.

    CA1 receives BOTH the completed pattern from CA3 and the current input
    from entorhinal cortex. Where they disagree is novelty. Nothing computes a
    difference; the disagreement is simply how much CA1 fires.
    """

    EC = 256
    DG = 1024                     # expanded, and held very sparse
    CA3 = 256
    CA1 = 256

    def __init__(self, cortex_size: int):
        super().__init__()
        self.ec = LIFPopulation(self.EC, name="hip.EC")
        self.dg = LIFPopulation(self.DG, name="hip.DG")
        self.dg_inh = LIFPopulation(self.DG // 4, inhibitory=True, name="hip.DG_inh")
        self.ca3 = LIFPopulation(self.CA3, name="hip.CA3")
        self.ca1 = LIFPopulation(self.CA1, name="hip.CA1")

        self.cortex_ec = SparseProjection(cortex_size, self.EC, 0.06, "exc",
                                          "cortico_hippocampal").with_delay(
            cfg.DELAY_MS["cortico_hippocampal"])
        self.ec_dg = SparseProjection(self.EC, self.DG, 0.05, "exc", "ec_dg")
        self.dg_i = SparseProjection(self.DG, self.DG // 4, 0.10, "exc", "dg_i")
        self.i_dg = SparseProjection(self.DG // 4, self.DG, 0.30, "inh", "i_dg")
        self.dg_ca3 = SparseProjection(self.DG, self.CA3, 0.04, "exc", "mossy")
        self.ca3_ca3 = SparseProjection(self.CA3, self.CA3, 0.20, "exc", "ca3_rec")
        self.ca3_ca1 = SparseProjection(self.CA3, self.CA1, 0.15, "exc", "schaffer")
        #  The direct path that makes comparison possible: CA1 sees the raw
        #  input as well as the recalled one.
        self.ec_ca1 = SparseProjection(self.EC, self.CA1, 0.10, "exc", "perforant")
        self.out = SparseProjection(self.CA1, cortex_size, 0.05, "exc",
                                    "fornix").with_delay(cfg.DELAY_MS["fornix"])

    def reset(self, batch: int, device) -> None:
        for pop in (self.ec, self.dg, self.dg_inh, self.ca3, self.ca1):
            pop.reset(batch, device)
        for p in (self.cortex_ec, self.ec_dg, self.dg_i, self.i_dg,
                  self.dg_ca3, self.ca3_ca3, self.ca3_ca1, self.ec_ca1, self.out):
            p.reset(batch, device)

    def step(self, cortical: torch.Tensor) -> torch.Tensor:
        self.ec.receive(self.cortex_ec(cortical), "exc")
        self.ec.step()

        self.dg.receive(self.ec_dg(self.ec.spikes), "exc")
        self.dg.receive(self.i_dg(self.dg_inh.spikes), "inh")
        self.dg_inh.receive(self.dg_i(self.dg.spikes), "exc")
        self.dg.step()
        self.dg_inh.step()

        self.ca3.receive(self.dg_ca3(self.dg.spikes), "exc")
        self.ca3.receive(self.ca3_ca3(self.ca3.spikes), "exc")
        self.ca3.step()

        self.ca1.receive(self.ca3_ca1(self.ca3.spikes), "exc")
        self.ca1.receive(self.ec_ca1(self.ec.spikes), "exc")
        self.ca1.step()
        return self.ca1.spikes


class Cerebellum(nn.Module):
    """A forward model whose readout cells are inhibitory.

    PURKINJE CELLS ARE GABAergic, and that is not a detail. The cerebellar
    output nucleus is driven by mossy-fibre collaterals and SCULPTED by
    Purkinje inhibition -- the cerebellum shapes movement by selectively
    removing inhibition, the same trick the basal ganglia uses. v2 modelled
    Purkinje cells as a linear readout, which loses this entirely.

    The granule layer expands massively, which is what makes the mossy input
    separable enough for a single Purkinje layer to learn from.
    """

    GRANULE = 1024
    PURKINJE = 128
    DEEP = 128

    def __init__(self, cortex_size: int):
        super().__init__()
        self.granule = LIFPopulation(self.GRANULE, name="cb.granule")
        self.purkinje = LIFPopulation(self.PURKINJE, inhibitory=True,
                                      name="cb.purkinje")
        self.deep = LIFPopulation(self.DEEP, name="cb.deep")

        self.pontine = SparseProjection(cortex_size, self.GRANULE, 0.04, "exc",
                                        "pontine").with_delay(cfg.DELAY_MS["pontine"])
        self.parallel = SparseProjection(self.GRANULE, self.PURKINJE, 0.06,
                                         "exc", "parallel_fibre")
        self.mossy_deep = SparseProjection(cortex_size, self.DEEP, 0.04, "exc",
                                           "mossy_collateral")
        self.purkinje_deep = SparseProjection(self.PURKINJE, self.DEEP, 0.25,
                                              "inh", "purkinje_deep")
        self.out = SparseProjection(self.DEEP, Thalamus.RELAY_SIZE, 0.10, "exc",
                                    "dentate_vl").with_delay(cfg.DELAY_MS["dentate_vl"])

    def reset(self, batch: int, device) -> None:
        for pop in (self.granule, self.purkinje, self.deep):
            pop.reset(batch, device)
        for p in (self.pontine, self.parallel, self.mossy_deep,
                  self.purkinje_deep, self.out):
            p.reset(batch, device)

    def step(self, cortical: torch.Tensor) -> torch.Tensor:
        self.granule.receive(self.pontine(cortical), "exc")
        self.granule.step()
        self.purkinje.receive(self.parallel(self.granule.spikes), "exc")
        self.purkinje.step()
        self.deep.receive(self.mossy_deep(cortical), "exc")
        self.deep.receive(self.purkinje_deep(self.purkinje.spikes), "inh")
        self.deep.step()
        return self.deep.spikes


class Hypothalamus(nn.Module):
    """Four slow drives that survive between percepts.

    The only state in the whole system that does not reset. Everything else
    starts each percept at resting potential; these do not, because hunger
    does not restart when you look at something new.
    """

    DRIVES = ("energy", "temperature", "threat", "satiety")
    SIZE = 64

    def __init__(self, cortex_size: int):
        super().__init__()
        self.cells = LIFPopulation(self.SIZE, name="hypothalamus")
        self.register_buffer("drive", torch.zeros(len(self.DRIVES)))
        self.to_cortex = SparseProjection(self.SIZE, cortex_size, 0.03, "exc",
                                          "hypothalamic").with_delay(
            cfg.DELAY_MS["hypothalamic"])

    def reset(self, batch: int, device) -> None:
        #  Deliberately does NOT clear self.drive.
        self.cells.reset(batch, device)
        self.to_cortex.reset(batch, device)

    def step(self, amygdala_spikes: torch.Tensor) -> torch.Tensor:
        threat = amygdala_spikes.mean()
        self.drive[2] += 0.01 * (threat * 1000.0 / cfg.TARGET_RATE_HZ
                                 - self.drive[2])
        level = float(self.drive.abs().mean())
        self.cells.receive(torch.full_like(self.cells.voltage, level * 0.3), "exc")
        self.cells.step()
        return self.cells.spikes


__all__ = ["Thalamus", "BasalGanglia", "Amygdala", "Hippocampus",
           "Cerebellum", "Hypothalamus"]
