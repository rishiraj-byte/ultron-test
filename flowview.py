"""
flowview.py -- the routing diagram: senses at the top, decision at the bottom.

WHAT THIS SHOWS THAT THE OTHER TWO VIEWS CANNOT.

    layerview   what one network is doing, layer by layer
    brainview   where activity is, anatomically
    flowview    WHERE THE DATA GOES -- which nucleus feeds which lobe, which
                lobe feeds which, and which paths run backwards

This is the original v1 dashboard picture, rebuilt for the architecture as it
now stands. v1 could draw it as a straight top-to-bottom pipeline because it
was one: input entered, passed through the thalamus, through the lobes, out.
That is no longer true, and the diagram has to be honest about it -- so
FEEDBACK AND RETURN PATHS ARE DRAWN AS SUCH, curving back up the left side,
rather than being omitted to keep the picture tidy. A routing diagram that
hides the loops would be showing a system this is not.

Rows are DEPTH FROM THE SENSES, not time. Nothing here happens in sequence:
every lobe updates simultaneously from delayed copies of the others. What the
rows encode is how many synapses a signal has crossed to arrive, which is a
real property and the one the layout is for.

Edge brightness is the signal that pathway is carrying at this timestep, read
from the settle trace. A severed tract is drawn as a broken red line rather
than removed, so a lesion is visible as damage instead of as absence.
"""

from __future__ import annotations

import math

import config

#  node -> (row, column position 0..1, label, kind)
#
#  Rows are SYNAPSES FROM THE SENSE, not time. Nothing here happens in
#  sequence -- every lobe updates simultaneously from delayed copies of the
#  others -- so the rows encode how many synapses a signal has crossed to
#  arrive, which is a real property and the one the layout is for.
#
#  Nodes marked "diagram" are drawn as their actual layer diagram: input,
#  hidden and output columns of circles, lit by live activation. A box with a
#  name in it tells you a module exists; the diagram tells you what it is
#  doing, and putting it INSIDE the routing graph is the only way to see both
#  at once.
NODES: dict[str, tuple[int, float, str, str]] = {
    #  row 0 -- the world
    "in.image":     (0, 0.07, "IMAGE", "input"),
    "in.text":      (0, 0.29, "TEXT", "input"),
    "in.audio":     (0, 0.51, "AUDIO", "input"),
    "in.sensor":    (0, 0.73, "SENSOR", "input"),

    #  row 1 -- thalamus, the switchboard every sense passes through
    "thalamus":     (1, 0.50, "THALAMUS", "diagram"),

    #  row 2 -- sensory cortex
    "occipital":    (2, 0.15, "OCCIPITAL", "diagram"),
    "temporal":     (2, 0.85, "TEMPORAL", "diagram"),

    #  row 3 -- association, and the memory that runs beside it
    "parietal":     (3, 0.15, "PARIETAL", "diagram"),
    "hippocampus":  (3, 0.85, "HIPPOCAMPUS", "diagram"),

    #  row 4 -- the decision
    "frontal":      (4, 0.50, "FRONTAL", "diagram"),

    #  row 5 -- subcortical, on their own clocks. Every one of these is a
    #  real network with a real diagram behind it, so every one is clickable.
    "basal ganglia": (5, 0.12, "BASAL GANGLIA", "diagram"),
    "amygdala":      (5, 0.38, "AMYGDALA", "diagram"),
    "cerebellum":    (5, 0.64, "CEREBELLUM", "diagram"),
    "hypothalamus":  (5, 0.88, "HYPOTHALAMUS", "diagram"),
}

#  Tracts name finer structures than the picture draws; these fold onto the
#  node that contains them, so no pathway is silently dropped.
#  Sense -> the nucleus it enters by. Every one of these lands in the
#  thalamus node above; the labels are kept so a dark sense is visible.
NUCLEUS_OF = {"in.image": "LGN", "in.text": "PULVINAR",
              "in.audio": "MGN", "in.sensor": "VPL"}

#  Tracts whose real endpoint is a thalamic nucleus are drawn to the single
#  thalamus node, since that is what the picture shows.
ALIAS = {"thal.LGN": "thalamus", "thal.MGN": "thalamus",
         "thal.VPL": "thalamus", "thal.PULVINAR": "thalamus",
         "thal.MD": "thalamus", "thal.VA_VL": "thalamus",
         "striatum": "basal ganglia", "stn": "basal ganglia",
         "cortex": "frontal"}

KIND_COLOUR = {
    "input": "#8fa6bd",
    "diagram": config.UI_ACCENT,
    "relay": config.UI_COOL,
    "return": "#7fd8c0",
    "lobe": config.UI_ACCENT,
    "bg": "#c86bff",
    "limbic": "#ff5f7e",
    "cerebellum": "#57e08a",
}

def delivered_lobes(present) -> set[str]:
    """Which lobes actually received a sensory sequence this percept.

    Read from the ONE routing table the system runs on, never a second copy
    kept in the drawing code: the previous constant here was a duplicate of
    NUCLEUS_OF that nothing referenced, and a diagram that maintains its own
    private idea of the wiring will eventually disagree with the wiring.

    This matters more than it used to. Every sense now reaches every lobe, and
    the whole reason that change was made is invisible unless the picture
    shows it -- a reader looking at four lit boxes has no way to tell whether
    all four parsed the input or three of them are just echoing the first.
    """
    from train_v2 import BrainV2

    #  "video" is a modality the user sees but not a route the thalamus has:
    #  a clip travels down the VISUAL route, as frames. Without this line the
    #  diagram drew no delivery at all for a video, which reads as "the clip
    #  reached nothing" -- the exact opposite of what happened.
    same_route_as = {"video": "image"}

    out: set[str] = set()
    for modality in (present or ()):
        route = BrainV2.ROUTES.get(same_route_as.get(modality, modality))
        if route:
            out.update(route[1])
    return out


def _mix(background: str, colour: str, amount: float) -> str:
    amount = max(0.0, min(1.0, amount))
    a = [int(background[i:i + 2], 16) for i in (1, 3, 5)]
    b = [int(colour[i:i + 2], 16) for i in (1, 3, 5)]
    return "#%02x%02x%02x" % tuple(
        int(round(x + (y - x) * amount)) for x, y in zip(a, b))


class FlowRenderer:
    """Draws the routing table as a graph, lit by the settle trace."""

    def __init__(self, cortex):
        self.cortex = cortex
        #  Endpoints are resolved through ALIAS, so a tract into thal.LGN is
        #  drawn into the thalamus node rather than dropped for having an
        #  endpoint the picture does not name.
        self.edges = []
        for name in cortex.tracts.names():
            tract = cortex.tracts[name]
            source = ALIAS.get(tract.source, tract.source)
            target = ALIAS.get(tract.target, tract.target)
            if source in NODES and target in NODES and source != target:
                self.edges.append((name, source, target))

    # -- geometry ----------------------------------------------------------
    def layout(self, width: int, height: int, top: int = 52, pad: int = 46):
        rows = max(row for row, _, _, _ in NODES.values()) + 1
        usable = height - top - pad
        return {
            name: (pad + column * (width - 2 * pad),
                   top + row * (usable / max(rows - 1, 1)))
            for name, (row, column, _, _) in NODES.items()
        }

    def signal(self, trace, t: int) -> dict[str, float]:
        """How much each tract is carrying into timestep t, scaled 0..1."""
        out = {}
        for name, _, _ in self.edges:
            tract = self.cortex.tracts[name]
            try:
                state = trace.history.read(tract.source, t - tract.delay)
            except Exception:                                  # noqa: BLE001
                continue
            out[name] = float(state.abs().mean()) * float(tract.gain)
        peak = max(out.values()) if out else 1.0
        return {k: min(v / (peak or 1.0), 1.0) for k, v in out.items()}

    def firing(self, trace, t: int) -> dict[str, tuple[int, int]]:
        """(units firing, units total) per node -- the number under each box.

        A count rather than a percentage, because the count says something a
        percentage hides: how WIDE the code is. At 5% sparsity the answer is
        thirteen, and seeing "13/256" is the point.
        """
        out: dict[str, tuple[int, int]] = {}
        for lobe in self.cortex.LOBES:
            state = trace.history.read(lobe, t)
            out[lobe] = (int((state != 0).sum(dim=-1).float().mean()),
                         int(state.shape[-1]))
        if trace.gates and t < len(trace.gates):
            open_now = sum(1 for g in trace.gates[t].values() if g > 0.5)
            out["thalamus"] = (open_now, len(trace.gates[t]))
        if trace.bg and t < len(trace.bg):
            go = trace.bg[t].go
            out["basal ganglia"] = (int((go > go.mean()).sum(dim=-1)
                                        .float().mean()), int(go.shape[-1]))
        if trace.amygdala and t < len(trace.amygdala):
            gain = trace.amygdala[t].gain
            out["amygdala"] = (int((gain > 1.0).sum(dim=-1).float().mean()),
                               int(gain.shape[-1]))
        return out

    def activity(self, trace, t: int) -> dict[str, float]:
        """Node brightness, scaled against the busiest node this timestep."""
        values: dict[str, float] = {}
        for lobe in self.cortex.LOBES:
            values[lobe] = float(trace.history.read(lobe, t).abs().mean())
        if trace.gates and t < len(trace.gates):
            values["thalamus"] = (sum(trace.gates[t].values())
                                  / max(len(trace.gates[t]), 1))
            for nucleus in ("MD", "VA_VL"):
                if nucleus in trace.gates[t]:
                    values["thal.%s" % nucleus] = trace.gates[t][nucleus]
        if trace.bg and t < len(trace.bg):
            values["striatum"] = float(trace.bg[t].go.mean().detach())
            values["stn"] = float(trace.bg[t].stn.mean().detach())
        if trace.amygdala and t < len(trace.amygdala):
            salience = float(trace.amygdala[t].salience.mean())
            values["amygdala"] = salience
            values["cortex"] = salience
        if trace.novelty and t < len(trace.novelty):
            values["hippocampus"] = trace.novelty[t]
        if trace.cerebellum and t < len(trace.cerebellum):
            values["cerebellum"] = trace.cerebellum[t].error_value
        values.setdefault("hypothalamus", 0.4)
        peak = max(values.values()) or 1.0
        return {k: min(v / peak, 1.0) for k, v in values.items()}

    # -- drawing -----------------------------------------------------------
    def draw(self, canvas, width: int, height: int, trace, t: int,
             lesioned=None, present=None, layers=None, photos=None,
             tile=(150, 78)) -> None:
        """One frame.

        `layers` is a LayerView. When supplied, every node marked "diagram"
        draws as its actual input/hidden/output columns rather than as a box
        with a name in it -- which is the whole point of merging the two
        views: a box tells you a module exists, the diagram tells you what it
        is doing, and the routing tells you where its output goes.
        """
        lesioned = lesioned or set()
        present = present or set()
        photos = photos if photos is not None else {}
        points = self.layout(width, height, top=64, pad=90)
        #  Screen boxes, recorded so the console can hit-test a click and open
        #  the module that was clicked.
        self.boxes: dict[str, tuple[float, float, float, float]] = {}

        canvas.create_text(16, 16, anchor="w", fill=config.UI_ACCENT,
                           font=config.UI_FONT_TITLE,
                           text="NETWORKS AND ROUTING")
        canvas.create_text(16, 34, anchor="w", fill=config.UI_MID,
                           font=config.UI_FONT_SMALL,
                           text="CLICK ANY MODULE TO OPEN IT     rows are "
                                "synapses from the sense, not time")

        if trace is None:
            canvas.create_text(width / 2, height / 2, fill=config.UI_DIM,
                               font=config.UI_FONT, text="idle -- no percept")
            return

        signal = self.signal(trace, t)
        activity = self.activity(trace, t)
        firing = self.firing(trace, t)
        half_w, half_h = tile[0] // 2, tile[1] // 2

        #  senses into the thalamus, dashed where that sense never arrived
        for source, nucleus in NUCLEUS_OF.items():
            live = source.split(".")[1] in present
            x0, y0 = points[source]
            x1, y1 = points["thalamus"]
            canvas.create_line(x0, y0 + 12, x1, y1 - half_h,
                               fill=_mix(config.UI_BG, config.UI_COOL,
                                         0.85 if live else 0.12),
                               width=2 if live else 1,
                               dash=() if live else (2, 4), arrow="last")

        #  THALAMUS INTO EVERY LOBE THAT ACTUALLY GOT THE SEQUENCE. These are
        #  not tracts and deliberately draw differently: a tract carries one
        #  lobe's state to another and is lesionable, whereas this is the
        #  sensory sequence itself being handed to a lobe's own operator.
        #  Drawing them the same way would invite the reader to try cutting
        #  one in the lesion panel and find it missing.
        for lobe in sorted(delivered_lobes(present)):
            if lobe not in points:
                continue
            x0, y0 = points["thalamus"]
            x1, y1 = points[lobe]
            canvas.create_line(x0, y0 + half_h, x1, y1 - half_h - 12,
                               fill=_mix(config.UI_BG, config.UI_COOL, 0.55),
                               width=2, dash=(6, 3), arrow="last")

        #  tracts
        for name, source, target in self.edges:
            x0, y0 = points[source]
            x1, y1 = points[target]
            level = signal.get(name, 0.0)
            if name in lesioned:
                canvas.create_line(x0, y0, x1, y1, fill=config.UI_ALERT,
                                   width=2, dash=(3, 4))
                continue
            backward = y1 < y0
            mx, my = (x0 + x1) / 2, (y0 + y1) / 2
            bow = -84 if backward else 20
            colour = _mix(config.UI_EDGE_IDLE,
                          "#7fd8c0" if backward else config.UI_ACCENT, level)
            canvas.create_line(x0, y0, mx + bow, my, x1, y1, fill=colour,
                               width=1 + 2.5 * level, smooth=True,
                               arrow="last")

        #  nodes
        for name, (row, column, label, kind) in NODES.items():
            x, y = points[name]
            colour = KIND_COLOUR[kind]
            level = activity.get(name, 0.0)

            if kind == "input":
                live = name.split(".")[1] in present
                canvas.create_rectangle(x - 36, y - 11, x + 36, y + 11,
                                        outline=_mix(config.UI_BG, colour,
                                                     1.0 if live else 0.3),
                                        fill=config.UI_PANEL)
                canvas.create_text(x, y, font=config.UI_FONT_LABEL, text=label,
                                   fill=_mix(config.UI_BG, colour,
                                             1.0 if live else 0.35))
                continue

            if kind == "diagram":
                self.boxes[name] = (x - half_w, y - half_h - 12,
                                    x + half_w, y + half_h + 14)
            if kind == "diagram" and layers is not None:
                canvas.create_text(x, y - half_h - 9, fill=colour,
                                   font=config.UI_FONT_LABEL, text=label)
                drawn = False
                try:
                    from PIL import ImageTk
                    image, _ = layers.render(name, tile[0], tile[1],
                                             trace, step=t)
                    photo = ImageTk.PhotoImage(image)
                    #  Tk keeps no reference of its own; a local one is
                    #  collected before it paints and the box comes out blank.
                    photos[name] = photo
                    canvas.create_image(x - half_w, y - half_h, image=photo,
                                        anchor="nw")
                    drawn = True
                except Exception:                              # noqa: BLE001
                    pass
                if not drawn:
                    canvas.create_rectangle(x - half_w, y - half_h,
                                            x + half_w, y + half_h,
                                            outline=colour,
                                            fill=config.UI_PANEL)
                canvas.create_rectangle(x - half_w, y - half_h,
                                        x + half_w, y + half_h,
                                        outline=_mix(config.UI_BG, colour,
                                                     0.4 + 0.6 * level))
                if name in firing:
                    active, total = firing[name]
                    canvas.create_text(x, y + half_h + 10, fill=config.UI_FG,
                                       font=config.UI_FONT_DATA,
                                       text="%d/%d firing" % (active, total))
                continue

            canvas.create_rectangle(
                x - 40, y - 12, x + 40, y + 12,
                fill=_mix(config.UI_PANEL, colour, 0.15 + 0.6 * level),
                outline=_mix(config.UI_BG, colour, 0.5 + 0.5 * level))
            canvas.create_text(x, y, fill=config.UI_FG,
                               font=config.UI_FONT_LABEL, text=label)

        canvas.create_text(16, height - 12, anchor="w", fill=config.UI_MID,
                           font=config.UI_FONT_DATA,
                           text="t = %d / %d      AMBER forward     "
                                "GREEN feedback and return     RED severed"
                                % (t, config.T_STEPS - 1))


__all__ = ["FlowRenderer", "NODES"]
