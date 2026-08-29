"""
diagram.py -- draws a network the way a neural network is actually drawn.

Columns of neurons, left to right: input layer, hidden layers, output layer.
Edges fanning between them. Every neuron's brightness is its REAL activation
this pass, at its real index -- neuron i is at position i, always.

Why not the old heatmap: reshaping a 256-vector into a 16x16 square puts
neuron 0 next to neuron 16 for no reason but the grid width. It shows that
something changed without showing what the network is. This draws the actual
topology read from the model by `topology.py`.

WHAT "ACTIVE" MEANS, precisely -- it differs by layer type and the drawing
must not blur them together:

  * GELU / sigmoid layers  -- a unit can genuinely be OFF. GELU drives
    negative pre-activations to ~0; sigmoid gates saturate shut. These are
    scaled from zero, so a dark cell means "not firing" and the ACTIVE count
    under the column is a real count of firing units.

  * linear / norm / input layers -- signed and centred around zero. "Off" is
    not meaningful, so these are scaled by MAGNITUDE: bright = large |value|
    in either direction, dark = near zero. The count is suppressed rather
    than invented.

HOW IT IS DRAWN. Circles in vertical columns, every circle in one column
connected to every circle in the next -- the standard way a network is drawn.
Circle fill is the live activation, and the ring stays visible at zero so an
idle unit still reads as a unit rather than disappearing.

A 512-unit layer cannot be 512 circles at any readable size, so as many
circles as fit are drawn and each stands for a contiguous block of units,
carrying that block's MEAN. Mean, not max: a max over a 47-unit block lights
the circle whenever any single unit fires, and at a typical 28% firing rate
that lights nearly every circle, destroying the signal. `ColumnRender.shown`
records the circle count so callers can state the ratio instead of implying
one circle is one neuron.

Rendering goes through PIL into a single image rather than hundreds of canvas
items, because a few thousand circles and connections as individual Tk shapes
will not hold a frame rate.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

import config
from modules.topology import LayerSpec

#  A unit counts as firing above this fraction of the layer's own peak.
#  Only applied to layers where "off" is meaningful (see module docstring).
ACTIVE_THRESHOLD = 0.05

#  Display-only curve; see the note where it is applied.
DISPLAY_GAMMA = 0.55

#  Connection brightness. EDGE_FLOOR keeps the topology visible when a layer is
#  quiet -- the wiring exists whether or not signal is flowing through it --
#  and EDGE_SPAN is how much brighter an actively-driven connection gets.
EDGE_FLOOR = 30
EDGE_SPAN = 118

#  Outline brightness of a completely inactive node, so an idle neuron still
#  reads as a neuron rather than vanishing into the background.
NODE_RING_MIN = 105



def _rgb(hex_colour: str) -> tuple[int, int, int]:
    h = hex_colour.lstrip("#")
    return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))


def _blend(cold: str, hot: str, t: float, floor: float = 0.0) -> tuple[int, int, int]:
    """Interpolate cold -> hot. `floor` lifts the minimum so structure stays
    visible at zero activation: wiring exists whether or not signal flows."""
    t = max(floor, min(1.0, t))
    a, b = _rgb(cold), _rgb(hot)
    return tuple(int(x + (y - x) * t) for x, y in zip(a, b))


@dataclass
class ColumnRender:
    """Where a column landed, so callers can hit-test and label it."""

    spec: LayerSpec
    x: int
    top: int
    bottom: int
    width: int
    active: int | None      # units firing, or None where the idea is meaningless
    total: int
    shown: int              # neurons that got their own pixel row
    peak: float


def _scale(values: np.ndarray, spec: LayerSpec) -> tuple[np.ndarray, float]:
    """Activation vector -> 0..1 brightness, honouring what the layer is.

    Returns (brightness, peak) where peak is the raw scale that was divided
    out, so the caller can show that two equally bright columns are not
    equally energetic.
    """
    if values.size == 0:
        return values, 0.0

    if spec.is_activation:
        # Genuinely one-sided: zero means off. Scale from zero, do not re-centre.
        positive = np.maximum(values, 0.0)
        peak = float(positive.max())
        if peak < 1e-12:
            return np.zeros_like(positive), 0.0
        return positive / peak, peak

    # Signed and centred: brightness is magnitude in either direction.
    magnitude = np.abs(values)
    peak = float(magnitude.max())
    if peak < 1e-12:
        return np.zeros_like(magnitude), 0.0
    return magnitude / peak, peak


def _bin_nodes(brightness: np.ndarray, count: int) -> np.ndarray:
    """N activations -> `count` node values, one per drawn circle.

    When a layer has more units than the column has circles, each circle
    represents a contiguous block and takes that block's MEAN.

    This is the opposite choice to the old solid strip, which used max. A max
    over a 28-unit block lights the circle if any single unit fires, so at a
    typical 28% firing rate almost every circle would be lit and the picture
    would carry no information. The mean says how active the block is, which is
    what varies visibly from pass to pass.
    """
    n = brightness.size
    if n == 0:
        return np.zeros(count, dtype=np.float32)
    if n <= count:
        out = np.zeros(count, dtype=np.float32)
        out[:n] = brightness
        return out
    edges = np.linspace(0, n, count + 1).astype(int)
    binned = np.array(
        [brightness[edges[i] : max(edges[i + 1], edges[i] + 1)].mean()
         for i in range(count)],
        dtype=np.float32,
    )
    # Averaging 32 units pulls every block towards the layer mean, so the
    # circles come out near-identical mid-grey. Rescale by the largest block --
    # NOT min-max, which would drag a genuinely dead block up off zero and
    # invent activity that is not there.
    top = float(binned.max())
    return binned / top if top > 1e-12 else binned


def render(
    layers: list[LayerSpec],
    vectors: dict[str, np.ndarray],
    width: int,
    height: int,
    edges: bool = True,
    label_space: int = 26,
    min_node_d: int = 7,
    max_node_d: int = 19,
    max_nodes: int = 11,
):
    """Draw the network as circles in columns, fully connected between layers.

    Returns (PIL.Image 'L', [ColumnRender]).

    `vectors` maps layer key -> raw activation vector. A layer with no entry
    (its module did not run this pass) draws as hollow outlines, which is how
    a silent lobe is meant to look.

    HOW MANY CIRCLES. A 512-unit layer cannot be 512 circles: at any readable
    circle size they would not fit, and sub-pixel circles are just the solid bar
    again. As many circles as fit are drawn, each standing for a contiguous
    block of units, and `ColumnRender.shown` records how many so the caller can
    say so rather than implying one circle is one neuron. Small layers (GO,
    NoGo) are one circle per unit, exactly.
    """
    from PIL import Image, ImageDraw

    width = max(int(width), 60)
    height = max(int(height), 40)
    image = Image.new("RGB", (width, height), _rgb(config.UI_PANEL))
    draw = ImageDraw.Draw(image)

    if not layers:
        return image, []

    top = 4
    bottom = max(height - label_space, top + 10)
    column_h = max(bottom - top, 8)

    # Circle size and how many fit, from the height we actually have.
    pitch_ratio = 1.6                       # circle + the gap beneath it
    fits = max(1, int(column_h // (min_node_d * pitch_ratio)))
    fits = min(fits, max_nodes)
    diameter = int(min(max_node_d, max(min_node_d, column_h / (fits * pitch_ratio))))
    radius = max(diameter / 2.0, 1.0)

    n_cols = len(layers)
    col_w = diameter
    side = 0 if label_space == 0 else min(34, max(8, width // 24))
    usable = max(width - 2 * side, col_w + 2)
    gap = (usable - n_cols * col_w) / max(n_cols - 1, 1) if n_cols > 1 else 0
    left_pad = side if n_cols > 1 else (width - col_w) // 2

    rendered: list[ColumnRender] = []
    node_values: list[np.ndarray] = []      # 0..1 per drawn circle
    node_xy: list[list[tuple[float, float]]] = []

    for i, spec in enumerate(layers):
        x = int(left_pad + i * (col_w + gap)) if n_cols > 1 else int(left_pad)
        x = min(x, width - col_w - 1)

        values = vectors.get(spec.key)
        count = min(spec.size, fits) if spec.size else fits
        count = max(count, 1)

        spacing = column_h / count
        centres = [(x + radius, top + spacing * (j + 0.5)) for j in range(count)]
        node_xy.append(centres)

        if values is None or values.size == 0:
            node_values.append(np.zeros(count, dtype=np.float32))
            rendered.append(
                ColumnRender(spec, x, top, bottom, col_w, None, spec.size, 0, 0.0)
            )
            continue

        brightness, peak = _scale(values, spec)
        node_values.append(_bin_nodes(brightness, count))
        active = (
            int((brightness > ACTIVE_THRESHOLD).sum()) if spec.is_activation else None
        )
        rendered.append(
            ColumnRender(spec, x, top, bottom, col_w, active,
                         values.size, count, peak)
        )

    # --- connections first, so the circles sit on top --------------------
    if edges and n_cols > 1:
        for i in range(n_cols - 1):
            src_pts, dst_pts = node_xy[i], node_xy[i + 1]
            src_vals = node_values[i]
            x1 = rendered[i].x + col_w
            x2 = rendered[i + 1].x
            if x2 <= x1 or not src_pts or not dst_pts:
                continue

            for (sx, sy), strength in zip(src_pts, src_vals):
                # Edge brightness follows the SOURCE node, so the fan visibly
                # thins where a layer has gone quiet. The floor is high enough
                # that the topology stays readable even at zero activation --
                # the connection exists whether or not signal is flowing.
                lit = float(strength) ** DISPLAY_GAMMA
                colour = _blend(config.UI_EDGE_IDLE, config.UI_ACCENT, lit,
                                floor=EDGE_FLOOR / 255.0)
                for dx, dy in dst_pts:
                    draw.line([x1, sy, x2, dy], fill=colour, width=1)

    # --- neurons ---------------------------------------------------------
    for column, centres, values in zip(rendered, node_xy, node_values):
        for (cx, cy), value in zip(centres, values):
            # Gamma lift. Activations are skewed hard towards zero, so a linear
            # ramp paints almost every circle black and a merely quiet layer
            # looks dead. DISPLAY ONLY -- the counts and peaks reported
            # alongside come from the raw values.
            lit = float(np.clip(value, 0.0, 1.0)) ** DISPLAY_GAMMA
            fill = _blend(config.UI_PANEL, config.UI_ACCENT, lit)
            # The ring stays visible at zero so an idle node still reads as a
            # node, exactly like an unfilled circle in a textbook diagram.
            ring = _blend(config.UI_DIM, config.UI_FG, lit)
            draw.ellipse(
                [cx - radius, cy - radius, cx + radius, cy + radius],
                fill=fill,
                outline=ring,
                width=1 if radius < 5 else 2,
            )

    return image, rendered


def annotate(
    canvas,
    rendered: list[ColumnRender],
    offset_x: int,
    offset_y: int,
    bottom: int,
    font,
    fg: str,
    dim: str,
    show_counts: bool = True,
) -> None:
    """Write each column's label and live firing count under the diagram.

    Kept separate from `render` so the picture stays a plain image and the text
    stays crisp Tk text rather than bitmap glyphs.
    """
    if not rendered:
        return
    first_x = rendered[0].x
    last_x = rendered[-1].x + rendered[-1].width

    for column in rendered:
        cx = offset_x + column.x + column.width / 2
        # The outermost labels are wider than their columns, so anchor them
        # inwards rather than letting them run off the canvas.
        if column.x == first_x and len(rendered) > 1:
            anchor = "nw"
            tx = offset_x + column.x - 2
        elif column.x + column.width == last_x and len(rendered) > 1:
            anchor = "ne"
            tx = offset_x + column.x + column.width + 2
        else:
            anchor, tx = "n", cx

        canvas.create_text(
            tx, bottom + 6, text=column.spec.label, fill=dim, font=font,
            anchor=anchor,
        )
        if not show_counts:
            continue
        text = f"{column.total}" if column.active is None else (
            f"{column.active}/{column.total}"
        )
        canvas.create_text(
            tx, bottom + 18, text=text,
            fill=fg if column.active else dim, font=font, anchor=anchor,
        )


def summarise(rendered: list[ColumnRender]) -> list[str]:
    """Text form of the same information, for the terminal and the detail pane."""
    lines: list[str] = []
    for column in rendered:
        spec = column.spec
        head = f"  {spec.label:<14}{spec.size:>5}u"
        if column.total == 0:
            lines.append(head + "  (did not run)")
            continue
        if column.active is None:
            lines.append(head + f"  peak {column.peak:6.2f}")
        else:
            pct = column.active / max(column.total, 1)
            lines.append(
                head + f"  {column.active:>4} on {pct:4.0%} pk {column.peak:5.2f}"
            )
        if 0 < column.shown < column.total:
            per = column.total / column.shown
            lines.append(f"  {'':<14}     {column.shown} circles, {per:.0f} units each")
    return lines


__all__ = [
    "render",
    "annotate",
    "summarise",
    "ColumnRender",
    "ACTIVE_THRESHOLD",
    "DISPLAY_GAMMA",
    "EDGE_FLOOR",
    "EDGE_SPAN",
]
