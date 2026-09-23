"""
layerview.py -- the four networks as layer diagrams, live.

WHY THIS EXISTS ALONGSIDE brainview.py.

The anatomical view answers "where is activity right now". This one answers
"what is each network actually doing" -- input column, hidden columns, output
column, circles standing for units, brightness standing for activation. They
are complementary and neither replaces the other: an anatomical map cannot show
you that a lobe's superficial layer is 5% active, and a layer diagram cannot
show you that four lobes are running at once.

WHAT A CIRCLE MEANS. A 1024-unit layer cannot be 1024 circles at any readable
size, so each circle stands for a contiguous block of units and the caller is
told how many. `modules/diagram.py` does the drawing and reports `shown` for
exactly this reason -- so the label can say "11 of 256" rather than implying
one circle is one neuron.

WHERE THE NUMBERS COME FROM. Real forward hooks on the real modules, captured
during a real settle. A layer whose module did not run draws as hollow
outlines, which is how a silent pathway is supposed to look.
"""

from __future__ import annotations

import numpy as np

import config
from modules.activations import ActivationRecorder
from modules.topology import IN_SUFFIX, LayerSpec

#  Which networks get a diagram, and in what order they are offered.
NETWORKS = ("occipital", "parietal", "temporal", "frontal",
            "thalamus", "basal ganglia", "amygdala", "hippocampus",
            "cerebellum", "hypothalamus")


class LayerView:
    """Builds layer specs for the v2 cortex and captures live activations."""

    def __init__(self, cortex):
        self.cortex = cortex
        self.recorder = ActivationRecorder()
        self.specs: dict[str, list[LayerSpec]] = {}
        #  One dict of activation vectors per settling timestep.
        self.frames: list[dict] = []
        self._build()

    # -- specs and hooks ---------------------------------------------------
    def _watch(self, key: str, module, capture_input: bool = False) -> None:
        try:
            self.recorder.watch(key, module, capture_input)
        except ValueError:
            pass                    # already watching; harmless

    def _build(self) -> None:
        dim = config.LATENT_DIM

        for name in self.cortex.LOBES:
            lobe = self.cortex.lobes[name]
            shell = lobe.laminar
            core = getattr(lobe, "core", None)
            specs: list[LayerSpec] = []

            #  L4 is thalamorecipient, so its INPUT is the drive arriving from
            #  the thalamus -- that is the diagram's input column.
            self._watch("%s.l4" % name, shell.l4, capture_input=True)
            #  The recorder stores a captured INPUT under the module's key
            #  plus IN_SUFFIX, so the spec has to ask for that exact key --
            #  inventing a suffix here just produces an empty column.
            specs.append(LayerSpec("%s.l4%s" % (name, IN_SUFFIX),
                                   "input (drive)", dim, "input", True))
            specs.append(LayerSpec("%s.l4" % name, "L4 thalamorecipient",
                                   dim, "gelu", True))

            if core is not None:
                self._watch("%s.core" % name, core)
                teacher = getattr(core, "teacher_name", "pretrained")
                specs.append(LayerSpec("%s.core" % name, teacher, dim,
                                       "gelu", True))

            #  L2/3 is where the lobe's operator runs and where sparsity is
            #  enforced, so this column is the one that should look mostly off.
            self._watch("%s.l23" % name, shell.l23_ei)
            specs.append(LayerSpec("%s.l23" % name, "L2/3 (E/I)", dim,
                                   "linear", True))

            self._watch("%s.l56" % name, shell.l56)
            specs.append(LayerSpec("%s.l56" % name, "L5/6 feedback", dim,
                                   "gelu"))
            specs.append(LayerSpec("%s.state" % name, "state (5% sparse)",
                                   dim, "output", True))
            self.specs[name] = specs

        #  Thalamus: one column per nucleus gate.
        thalamus = self.cortex.thalamus
        specs = []
        for nucleus in thalamus.NUCLEI:
            relay = thalamus.nuclei[nucleus]
            self._watch("thal.%s" % nucleus, relay.norm, capture_input=False)
            specs.append(LayerSpec("thal.%s" % nucleus, nucleus, dim,
                                   "sigmoid", True))
        self.specs["thalamus"] = specs

        #  Basal ganglia: the pathway that decides.
        bg = self.cortex.basal_ganglia
        for key, module, label, kind in (
                ("bg.direct", bg.direct, "D1 direct (Go)", "gelu"),
                ("bg.indirect", bg.indirect, "D2 indirect (NoGo)", "gelu"),
                ("bg.stn", bg.stn, "STN", "gelu"),
                ("bg.gpi", bg.gpi, "GPi (output)", "linear")):
            self._watch(key, module)
        #  LIMBIC, CEREBELLAR AND HOMEOSTATIC MODULES. Every structure the
        #  routing diagram names gets a real diagram, because a box a reader
        #  can click that then shows nothing teaches them the module is
        #  decorative -- which is the opposite of what the ablation sweep
        #  says about all of them.
        amygdala = self.cortex.amygdala
        for key, module in (("amy.low", amygdala.low),
                            ("amy.high", amygdala.high),
                            ("amy.salience", amygdala.salience_head),
                            ("amy.valence", amygdala.valence_head),
                            ("amy.gain", amygdala.to_cortex)):
            self._watch(key, module)
        self.specs["amygdala"] = [
            LayerSpec("amy.low", "low road (fast)", dim, "gelu", True),
            LayerSpec("amy.high", "high road (slow)", dim, "gelu", True),
            LayerSpec("amy.salience", "salience", 1, "sigmoid", True),
            LayerSpec("amy.valence", "valence", 1, "sigmoid", True),
            LayerSpec("amy.gain", "gain to cortex", dim, "output", True),
        ]

        hippocampus = self.cortex.hippocampus
        self._watch("hip.ec", hippocampus.entorhinal)
        self._watch("hip.dg", hippocampus.dg.wide)
        self._watch("hip.ca3", hippocampus.ca3.recurrent)
        self._watch("hip.ca1", hippocampus.ca1)
        self._watch("hip.sub", hippocampus.subiculum)
        wide = dim * config.DG_EXPANSION
        self.specs["hippocampus"] = [
            LayerSpec("hip.ec", "entorhinal", dim, "linear", True),
            LayerSpec("hip.dg", "dentate (2% sparse)", wide, "gelu", True),
            LayerSpec("hip.ca3", "CA3 completion", dim, "linear", True),
            LayerSpec("hip.ca1", "CA1 compare", dim, "linear", True),
            LayerSpec("hip.sub", "subiculum out", dim, "output", True),
        ]

        cerebellum = self.cortex.cerebellum
        self._watch("cb.pontine", cerebellum.pontine)
        self._watch("cb.granule", cerebellum.granule)
        self._watch("cb.purkinje", cerebellum.purkinje)
        self._watch("cb.dentate", cerebellum.dentate)
        self.specs["cerebellum"] = [
            LayerSpec("cb.pontine", "pontine (copy)", dim, "gelu", True),
            LayerSpec("cb.granule", "granule (expanded)",
                      dim * config.CEREBELLAR_EXPANSION, "gelu", True),
            LayerSpec("cb.purkinje", "Purkinje readout", dim, "linear", True),
            LayerSpec("cb.dentate", "dentate out", dim, "output", True),
        ]

        hypothalamus = self.cortex.hypothalamus
        self._watch("hyp.read", hypothalamus.read)
        self._watch("hyp.out", hypothalamus.to_cortex)
        self.specs["hypothalamus"] = [
            LayerSpec("hyp.read", "drive sensor", len(hypothalamus.DRIVES),
                      "linear", True),
            LayerSpec("hyp.out", "to cortex", dim, "output", True),
        ]

        self.specs["basal ganglia"] = [
            LayerSpec("bg.direct", "D1 direct (Go)", dim, "gelu", True),
            LayerSpec("bg.indirect", "D2 indirect (NoGo)", dim, "gelu", True),
            LayerSpec("bg.stn", "STN", dim, "gelu", True),
            LayerSpec("bg.gpi", "GPi (output)", dim, "linear", True),
        ]

    # -- capture -----------------------------------------------------------
    def vectors(self, trace=None) -> dict[str, np.ndarray]:
        """Activation vector per layer key, from the last pass through.

        The lobe STATE columns come from the settle trace rather than a hook,
        because the committed state is assembled in the loop and never passes
        through a module of its own.
        """
        snapshot = self.recorder.snapshot()
        out: dict[str, np.ndarray] = {}
        for key, tensor in snapshot.items():
            flat = tensor.detach().float().reshape(tensor.shape[0], -1)[0]
            out[key] = flat.cpu().numpy()

        if trace is not None:
            for name in self.cortex.LOBES:
                state = trace.history.read(name, config.T_STEPS - 1)
                out["%s.state" % name] = (
                    state.detach().float()[0].cpu().numpy())
        return out

    # -- per-timestep recording -------------------------------------------
    def start_recording(self) -> None:
        """Begin a new sequence of per-timestep frames."""
        self.frames: list[dict] = []

    def record(self, t: int, history) -> None:
        """Snapshot every watched layer as it stands at timestep `t`.

        Called from cortex.settle's on_step hook. Without this the diagram
        shows only whatever the hooks held after the LAST timestep, which is
        a still photograph of a process -- and the process is the thing worth
        looking at, since sparsity anneals from 60% down to 5% across the
        window and the active set changes completely on the way.
        """
        snapshot = self.recorder.snapshot()
        frame = {}
        for key, tensor in snapshot.items():
            flat = tensor.detach().float().reshape(tensor.shape[0], -1)[0]
            frame[key] = flat.cpu().numpy()
        for name in self.cortex.LOBES:
            state = history.read(name, t)
            frame["%s.state" % name] = state.detach().float()[0].cpu().numpy()
        self.frames.append(frame)

    def frame_count(self) -> int:
        return len(getattr(self, "frames", []))

    def render(self, network: str, width: int, height: int, trace=None,
               step: int | None = None):
        """(PIL image, [ColumnRender]) for one network."""
        from modules import diagram
        specs = self.specs.get(network, [])
        frames = getattr(self, "frames", [])
        if step is not None and frames:
            vectors = frames[min(step, len(frames) - 1)]
        else:
            vectors = self.vectors(trace)
        return diagram.render(specs, vectors, width, height)

    def remove(self) -> None:
        self.recorder.remove()


__all__ = ["LayerView", "NETWORKS"]
