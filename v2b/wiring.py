"""
wiring.py -- every arrow in the v2.B graph is a named connection.

Named, not an anonymous tensor hand-off, so that the dashboard can light each
one by what it carried, and so that any one can be CUT to find out what the
thing it feeds really contributes. Cutting is a switch, never a code edit.
"""

from __future__ import annotations

#  (name, source node, target node, what it is in the brain)
CONNECTIONS = (
    ("pulvinar->parietal", "pulvinar", "parietal", "pulvinar projection to posterior parietal"),
    ("LGN->occipital", "LGN", "occipital", "optic radiation to V1"),
    ("MGN->temporal", "MGN", "temporal", "acoustic radiation to A1 (temporal lobe)"),
    ("occipital->ventral", "occipital", "ventral", "ventral stream, V4 -> inferotemporal (IT)"),
    ("occipital->dorsal", "occipital", "dorsal", "dorsal stream, V4/MT -> parietal"),
    ("parietal->dorsal", "parietal", "dorsal", "within posterior parietal"),
    ("ventral->STS", "ventral", "STS", "inferotemporal (IT) -> superior temporal sulcus"),
    ("temporal->STS", "temporal", "STS", "auditory belt -> superior temporal sulcus"),
    ("temporal->frontal", "temporal", "frontal",
     "arcuate fasciculus: Wernicke's area (sound meaning) -> Broca's area (naming)"),
    ("dorsal->POT", "dorsal", "POT", "parietal -> TPO junction"),
    ("STS->POT", "STS", "POT", "temporal -> TPO junction"),
    ("POT->frontal", "POT", "frontal", "association fibres to prefrontal"),
    ("frontal->POT", "frontal", "POT", "top-down feedback, prefrontal -> association"),
    ("frontal->thalamus", "frontal", "thalamus", "layer 6 corticothalamic, via the TRN"),
    ("POT->hippocampus", "POT", "hippocampus", "association cortex -> entorhinal"),
    ("hippocampus->frontal", "hippocampus", "frontal", "subiculum / CA1 -> prefrontal"),
    ("hippocampus->POT", "hippocampus", "POT",
     "entorhinal feedback to association cortex: a familiar kind is read with expertise"),
    ("POT->amygdala", "POT", "amygdala", "temporal association cortex -> amygdala"),
    ("hippocampus->amygdala", "hippocampus", "amygdala", "CA1 / subiculum -> basolateral amygdala"),
    ("amygdala->frontal", "amygdala", "frontal", "amygdala -> orbitofrontal"),
    ("frontal->motor", "frontal", "motor", "prefrontal -> premotor -> motor"),
    ("amygdala->motor", "amygdala", "motor",
     "go / hold: central amygdala -> brainstem (hold = freeze), with orbitofrontal"),
)

NAMES = tuple(c[0] for c in CONNECTIONS)
ANATOMY = {c[0]: c[3] for c in CONNECTIONS}


class Wiring:
    """Which connections are intact. Everything starts connected."""

    names = NAMES

    def __init__(self):
        self.cut: set[str] = set()

    def open(self, name: str) -> bool:
        if name not in NAMES:
            raise KeyError("no connection called %r" % name)
        return name not in self.cut

    def lesion(self, name: str, cut: bool = True) -> None:
        if name not in NAMES:
            raise KeyError("no connection called %r" % name)
        (self.cut.add if cut else self.cut.discard)(name)

    def restore(self) -> None:
        self.cut.clear()
