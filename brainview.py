"""
brainview.py -- the anatomical live view: a brain, thinking, in real time.

WHY A BRAIN SHAPE AND NOT A BLOCK DIAGRAM.

A block diagram of this system would be a lie of emphasis. It would put the
boxes in a row and imply an order, when the whole architectural claim is that
there is no order -- four lobes update simultaneously from delayed copies of
each other, and the subcortical structures run alongside on their own clocks.
Laid out anatomically, the thing you see is what is actually true: everything
lit at once, signal travelling along real tracts, the pattern changing every
timestep until it stops changing.

WHAT IS ANIMATED IS REAL. Every glow is a measured number from the settle
trace -- a lobe's activation energy at that timestep, a tract's signal, a
gate's opening. Nothing here is a decorative pulse on a timer. When the display
goes quiet it is because the network converged, and the moment it goes quiet is
the moment the deltas stopped shrinking.

    positions are anatomical (left = posterior, right = anterior)
    edge brightness  = |signal| carried by that tract at this timestep
    node glow        = that region's activation energy at this timestep
    node ring        = its gate or its sparsity, whichever it has
"""

from __future__ import annotations

import math

import config

#  ANATOMICAL LAYOUT IN THREE DIMENSIONS. x runs posterior (0) to anterior
#  (1), y dorsal (0) to ventral (1), and z is LATERAL DISPLACEMENT from the
#  midline: 0.5 is on the midline, values away from it sit out in a
#  hemisphere. That third axis is what a sagittal drawing cannot show, and it
#  is not decoration -- the four lobes are lateral structures and the
#  thalamus, hypothalamus and brainstem nuclei are midline ones, so a flat
#  picture necessarily draws them overlapping something they sit beside.
REGIONS: dict[str, tuple[float, float, float, str, str]] = {
    #  name:            (x,    y,    z,    label,        kind)
    "occipital":        (0.13, 0.46, 0.72, "OCCIPITAL",  "lobe"),
    "parietal":         (0.38, 0.20, 0.74, "PARIETAL",   "lobe"),
    "temporal":         (0.36, 0.74, 0.80, "TEMPORAL",   "lobe"),
    "frontal":          (0.80, 0.36, 0.70, "FRONTAL",    "lobe"),
    "thalamus":         (0.52, 0.46, 0.50, "THALAMUS",   "relay"),
    "striatum":         (0.63, 0.34, 0.62, "STRIATUM",   "bg"),
    "stn":              (0.58, 0.56, 0.57, "STN",        "bg"),
    "amygdala":         (0.50, 0.72, 0.66, "AMYGDALA",   "limbic"),
    "hippocampus":      (0.36, 0.86, 0.68, "HIPPOCAMPUS", "limbic"),
    "hypothalamus":     (0.62, 0.68, 0.50, "HYPOTHAL",   "limbic"),
    "cerebellum":       (0.14, 0.82, 0.55, "CEREBELLUM", "cerebellum"),
}

#  How each kind is drawn.
KIND_COLOUR = {
    "lobe": config.UI_ACCENT,
    "relay": config.UI_COOL,
    "bg": "#c86bff",
    "limbic": "#ff5f7e",
    "cerebellum": "#57e08a",
}

#  Which region a tract endpoint belongs to, where the tract names something
#  finer-grained than the picture shows.
ENDPOINT = {
    "thal.LGN": "thalamus", "thal.MGN": "thalamus", "thal.VPL": "thalamus",
    "thal.PULVINAR": "thalamus", "thal.MD": "thalamus", "thal.VA_VL": "thalamus",
    "cortex": "frontal",
}


def endpoint(name: str) -> str | None:
    if name in REGIONS:
        return name
    return ENDPOINT.get(name)


def brain_outline(width: int, height: int, pad: int = 30) -> list[tuple[float, float]]:
    """A sagittal brain silhouette, as a closed polygon.

    Hand-shaped rather than traced from an atlas: enough to read as a brain at
    a glance, and honest about being a schematic.
    """
    #  (x, y) in unit space, anticlockwise from the frontal pole
    unit = [
        (0.94, 0.34), (0.92, 0.20), (0.84, 0.10), (0.72, 0.04), (0.56, 0.02),
        (0.40, 0.04), (0.26, 0.10), (0.14, 0.20), (0.07, 0.34), (0.05, 0.48),
        (0.07, 0.60), (0.12, 0.68),
        #  cerebellar notch
        (0.10, 0.74), (0.06, 0.82), (0.10, 0.90), (0.20, 0.93), (0.28, 0.90),
        (0.30, 0.82),
        #  temporal pole sweeping forward and under
        (0.36, 0.88), (0.46, 0.90), (0.58, 0.86), (0.68, 0.78), (0.74, 0.68),
        (0.82, 0.62), (0.90, 0.52), (0.95, 0.44),
    ]
    w, h = width - 2 * pad, height - 2 * pad
    return [(pad + x * w, pad + y * h) for x, y in unit]


def project(x: float, y: float, z: float, angle: float,
            width: int, height: int, pad: int = 34):
    """One 3D point to screen, with perspective. Returns (px, py, depth).

    Rotation is about the vertical axis, which is the one that matters here:
    it swings the hemispheres out from behind the midline so a structure that
    sits beside the thalamus stops being drawn on top of it. `depth` comes
    back so the caller can paint far things before near ones and scale them --
    without that a 3D layout is just a skewed 2D one.
    """
    cx, cz = 0.5, 0.5
    dx, dz = x - cx, z - cz
    rx = dx * math.cos(angle) - dz * math.sin(angle)
    rz = dx * math.sin(angle) + dz * math.cos(angle)

    #  Weak perspective: near things larger. The camera is far enough away
    #  that the brain does not distort into a funnel.
    depth = rz + cz
    scale = config.UI_BRAIN_CAMERA / (config.UI_BRAIN_CAMERA + depth)
    w, h = width - 2 * pad, height - 2 * pad
    px = pad + w * (0.5 + rx * scale)
    py = pad + h * (0.5 + (y - 0.5) * scale)
    return px, py, depth


def place(width: int, height: int, angle: float = 0.0, pad: int = 34):
    """name -> (screen x, screen y, depth) at this rotation."""
    return {name: project(x, y, z, angle, width, height, pad)
            for name, (x, y, z, _, _) in REGIONS.items()}


def glow(canvas, x: float, y: float, radius: float, colour: str,
         intensity: float, layers: int = 4) -> None:
    """Concentric fading rings. Tk has no alpha, so this fakes it.

    Colours are interpolated toward the panel background rather than drawn
    transparent, which is why the ground colour has to be known here.
    """
    intensity = max(0.0, min(1.0, intensity))
    for i in range(layers, 0, -1):
        share = (i / layers)
        r = radius * (0.6 + 1.5 * share)
        blend = intensity * (1.0 - share * 0.8)
        canvas.create_oval(x - r, y - r, x + r, y + r,
                           outline=_mix(config.UI_BG, colour, blend), width=1)


def _mix(background: str, colour: str, amount: float) -> str:
    """Blend two #rrggbb colours. `amount` 0 = background, 1 = colour."""
    amount = max(0.0, min(1.0, amount))
    a = [int(background[i:i + 2], 16) for i in (1, 3, 5)]
    b = [int(colour[i:i + 2], 16) for i in (1, 3, 5)]
    return "#%02x%02x%02x" % tuple(
        int(round(x + (y - x) * amount)) for x, y in zip(a, b))


class BrainRenderer:
    """Draws one timestep of a settle trace onto a tk canvas."""

    def __init__(self, cortex):
        self.cortex = cortex
        #  Precompute which tracts are drawable: both ends must be on the map.
        self.edges = []
        for name in cortex.tracts.names():
            tract = cortex.tracts[name]
            a, b = endpoint(tract.source), endpoint(tract.target)
            if a and b and a != b:
                self.edges.append((name, a, b, tract.delay))

    # -- per-timestep signal, read from the trace -------------------------
    def energies(self, trace, t: int) -> dict[str, float]:
        """Activation energy per region at timestep t, normalised to 0..1."""
        values: dict[str, float] = {}
        for lobe in self.cortex.LOBES:
            state = trace.history.read(lobe, t)
            values[lobe] = float(state.abs().mean())

        if trace.gates and t < len(trace.gates):
            values["thalamus"] = sum(trace.gates[t].values()) / max(
                len(trace.gates[t]), 1)
        if trace.bg and t < len(trace.bg):
            decision = trace.bg[t]
            values["striatum"] = float(decision.go.mean().detach())
            values["stn"] = float(decision.stn.mean().detach())
        if trace.amygdala and t < len(trace.amygdala):
            values["amygdala"] = float(trace.amygdala[t].salience.mean())
        if trace.novelty and t < len(trace.novelty):
            values["hippocampus"] = trace.novelty[t]
        if trace.cerebellum and t < len(trace.cerebellum):
            values["cerebellum"] = trace.cerebellum[t].error_value
        values.setdefault("hypothalamus", 0.35)

        return values

    def normalised(self, trace) -> list[dict[str, float]]:
        """Every timestep, each region scaled against ITS OWN range.

        Scaling everything against one global peak looks reasonable and is
        useless: these are different quantities in different units. A lobe's
        activation energy sits near 0.1, a thalamic gate near 0.4, and the
        cerebellum reports a RELATIVE error that runs to 15 -- so a single
        peak makes the cerebellum the only thing ever lit and the four lobes
        permanently dark. What the picture is for is showing when each region
        is doing something, which is a question about that region's own range.
        """
        frames = [self.energies(trace, t) for t in range(len(trace.deltas))]
        names = {name for frame in frames for name in frame}
        scaled: list[dict[str, float]] = []
        span = {}
        for name in names:
            series = [f.get(name, 0.0) for f in frames]
            low, high = min(series), max(series)
            span[name] = (low, (high - low) or 1.0)
        for frame in frames:
            scaled.append({name: (frame.get(name, 0.0) - span[name][0])
                           / span[name][1] for name in names})
        return scaled

    def edge_signal(self, trace, t: int) -> dict[str, float]:
        """How much each tract carried into timestep t."""
        out = {}
        for name, a, b, delay in self.edges:
            tract = self.cortex.tracts[name]
            source = tract.source
            step = t - delay
            try:
                state = trace.history.read(source, step)
            except Exception:                                  # noqa: BLE001
                continue
            out[name] = float(state.abs().mean()) * float(tract.gain)
        peak = max(out.values()) if out else 1.0
        return {k: min(v / (peak or 1.0), 1.0) for k, v in out.items()}

    # -- drawing -----------------------------------------------------------
    def draw(self, canvas, width: int, height: int, trace, t: int,
             lesioned: set[str] | None = None, angle: float = 0.0) -> None:
        """Draw one timestep, viewed from `angle` radians about the vertical.

        Painted BACK TO FRONT. Without depth sorting a 3D layout is only a
        skewed 2D one: a structure behind the thalamus would be drawn over it
        half the time depending on dictionary order, which is worse than not
        rotating at all.
        """
        lesioned = lesioned or set()
        points = place(width, height, angle)

        canvas.create_oval(34, 34, width - 34, height - 34,
                           outline=config.UI_EDGE_IDLE, fill=config.UI_PANEL)

        if trace is None:
            canvas.create_text(width / 2, height / 2, fill=config.UI_DIM,
                               font=config.UI_FONT, text="idle -- no percept")
            return

        signal = self.edge_signal(trace, t)
        if getattr(self, "_cached_for", None) is not trace:
            self._cached_for = trace
            self._frames = self.normalised(trace)
        energy = self._frames[min(t, len(self._frames) - 1)]

        #  Tracts first, far ones first, so nearer ones lie over them.
        edges = []
        for name, a, b, delay in self.edges:
            x0, y0, d0 = points[a]
            x1, y1, d1 = points[b]
            edges.append(((d0 + d1) / 2, name, x0, y0, x1, y1))
        for depth, name, x0, y0, x1, y1 in sorted(edges, reverse=True):
            level = signal.get(name, 0.0)
            if name in lesioned:
                canvas.create_line(x0, y0, x1, y1, fill=config.UI_ALERT,
                                   width=1, dash=(2, 4))
                continue
            #  Far edges are dimmer, which is what makes the depth readable.
            fade = 1.0 - min(max(depth, 0.0), 1.0) * 0.45
            mx, my = (x0 + x1) / 2, (y0 + y1) / 2
            nx, ny = -(y1 - y0), (x1 - x0)
            length = math.hypot(nx, ny) or 1.0
            bow = 14 * (1 if name < "m" else -1)
            cx, cy = mx + nx / length * bow, my + ny / length * bow
            colour = _mix(config.UI_EDGE_IDLE, config.UI_ACCENT, level * fade)
            canvas.create_line(x0, y0, cx, cy, x1, y1, fill=colour,
                               width=1 + 2 * level * fade, smooth=True)

        #  Then the regions, also back to front, scaled by depth.
        order = sorted(REGIONS, key=lambda n: -points[n][2])
        for name in order:
            x, y, depth = points[name]
            _, _, _, label, kind = REGIONS[name]
            level = energy.get(name, 0.0)
            colour = KIND_COLOUR[kind]
            near = config.UI_BRAIN_CAMERA / (config.UI_BRAIN_CAMERA + depth)
            radius = (13 if kind == "lobe" else 9) * near * 1.1
            fade = 1.0 - min(max(depth, 0.0), 1.0) * 0.35

            glow(canvas, x, y, radius, colour, level * fade)
            canvas.create_oval(x - radius, y - radius, x + radius, y + radius,
                               fill=_mix(config.UI_BG, colour,
                                         (0.25 + 0.75 * level) * fade),
                               outline=_mix(config.UI_BG, colour, fade))
            canvas.create_text(x, y + radius + 9,
                               fill=_mix(config.UI_BG, config.UI_MID, fade),
                               font=config.UI_FONT_TINY, text=label)

        canvas.create_text(14, height - 26, anchor="w", fill=config.UI_DIM,
                           font=config.UI_FONT_TINY,
                           text="t = %d / %d" % (t, config.T_STEPS - 1))
        canvas.create_text(14, height - 12, anchor="w", fill=config.UI_DIM,
                           font=config.UI_FONT_TINY,
                           text="3D -- drag to rotate, or let it spin")


__all__ = ["BrainRenderer", "REGIONS", "place", "project"]
