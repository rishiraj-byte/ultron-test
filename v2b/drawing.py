"""
drawing.py -- a network drawn the way networks are drawn: columns of circles.

The same convention as the v2 console. Left to right: what came in, the
layers it went through, what came out. A circle's brightness is the real
activation this percept produced. A column that did not run this time (its
sense was absent, or its connection was cut) is drawn as hollow rings.

A 768-unit layer cannot be 768 readable circles, so each circle stands for a
contiguous block of units and carries that block's mean magnitude; the label
under the column says how many units there really are.

Rendered to one PIL image rather than thousands of Tk canvas items -- a
network redrawn ten times a second as Tk shapes is what makes a UI stutter.
"""

from __future__ import annotations

import numpy as np

from v2b import config_v2b as C


def _rgb(colour: str) -> tuple[int, int, int]:
    colour = colour.lstrip("#")
    return tuple(int(colour[i:i + 2], 16) for i in (0, 2, 4))


def _mix(a: str, b: str, t: float) -> tuple[int, int, int]:
    ca, cb = _rgb(a), _rgb(b)
    t = float(min(max(t, 0.0), 1.0))
    return tuple(int(x + (y - x) * t) for x, y in zip(ca, cb))


def _blocks(values: np.ndarray, count: int) -> np.ndarray:
    """Mean |value| per contiguous block, scaled so the column's max is 1."""
    magnitude = np.abs(values.astype(np.float32))
    if magnitude.size <= count:
        blocks = magnitude
    else:
        edges = np.linspace(0, magnitude.size, count + 1).astype(int)
        blocks = np.array([magnitude[a:b].mean() for a, b in zip(edges[:-1], edges[1:])])
    peak = float(blocks.max()) if blocks.size else 0.0
    return blocks / peak if peak > 1e-9 else blocks * 0


def render(columns: list, width: int, height: int, max_nodes: int = 12):
    """columns: [(label, vector or None)] -> PIL RGB image."""
    from PIL import Image, ImageDraw, ImageFont

    width, height = max(int(width), 80), max(int(height), 60)
    image = Image.new("RGB", (width, height), _rgb(C.UI_PANEL))
    draw = ImageDraw.Draw(image)
    if not columns:
        return image
    try:
        font = ImageFont.truetype("consola.ttf", 11)
    except Exception:                                         # noqa: BLE001
        font = ImageFont.load_default()

    label_space = 30
    top, bottom = 8, height - label_space
    usable = max(bottom - top, 20)
    count = len(columns)
    xs = [int(width * (i + 0.5) / count) for i in range(count)]
    diameter = int(max(6, min(18, usable / (max_nodes * 1.5), width / (count * 3))))
    fits = max(1, min(max_nodes, int(usable / (diameter * 1.5))))

    layout = []
    for (label, vector), x in zip(columns, xs):
        if vector is None or np.size(vector) == 0:
            size, shown, bright = 0, min(fits, 6), None
        else:
            vector = np.asarray(vector).reshape(-1)
            size = vector.size
            shown = min(fits, size)
            bright = _blocks(vector, shown)
        span = shown * diameter * 1.5
        ys = [int(top + (usable - span) / 2 + diameter * 1.5 * (j + 0.5))
              for j in range(shown)]
        layout.append((label, x, ys, bright, size))

    #  Edges first, so circles sit on top. Brightness follows the signal
    #  arriving at the right-hand column.
    for (_, x0, ys0, b0, _), (_, x1, ys1, b1, _) in zip(layout, layout[1:]):
        live = b0 is not None and b1 is not None
        for j0, y0 in enumerate(ys0):
            for j1, y1 in enumerate(ys1):
                if live:
                    level = 0.15 + 0.6 * float(b0[j0] * b1[j1])
                    colour = _mix(C.UI_EDGE_IDLE, C.UI_ACCENT_DIM, level * 1.4)
                else:
                    colour = _rgb(C.UI_EDGE_IDLE)
                draw.line([(x0, y0), (x1, y1)], fill=colour, width=1)

    r = diameter // 2
    for label, x, ys, bright, size in layout:
        for j, y in enumerate(ys):
            box = (x - r, y - r, x + r, y + r)
            if bright is None:
                draw.ellipse(box, outline=_rgb(C.UI_DIM), fill=_rgb(C.UI_PANEL))
            else:
                level = float(bright[j])
                draw.ellipse(box, fill=_mix(C.UI_PANEL, C.UI_ACCENT, 0.12 + 0.88 * level),
                             outline=_mix(C.UI_DIM, C.UI_ACCENT, level))
        #  Cut each label to the room its column has, so neighbours in a
        #  narrow tile never print over each other.
        room = width / count - 4
        sub = ("%d units" % size) if size else "silent"
        for line, dy, colour in ((label, 4, C.UI_MID), (sub, 16, C.UI_DIM)):
            while len(line) > 1 and draw.textlength(line, font=font) > room:
                line = line[:-2] + "."
            w = draw.textlength(line, font=font)
            draw.text((x - w / 2, bottom + dy), line, fill=_rgb(colour), font=font)
    return image


def heat_overlay(rgb: np.ndarray, grid: np.ndarray, alpha: float = 0.55):
    """Paint a 7x7 attention grid over the 224x224 retina image."""
    from PIL import Image

    base = Image.fromarray(rgb).convert("RGB")
    g = np.asarray(grid, np.float32).reshape(C.GRID, C.GRID)
    #  Min-max, not max alone: POT's attention is often spread thin, and
    #  dividing by the max painted the whole picture the same colour.
    g = (g - g.min()) / max(float(g.max() - g.min()), 1e-9)
    heat = Image.fromarray((g * 255).astype(np.uint8)).resize(
        base.size, Image.BICUBIC)
    heat = np.asarray(heat, np.float32)[..., None] / 255.0
    accent = np.array(_rgb(C.UI_ACCENT), np.float32)
    dimmed = np.asarray(base, np.float32) * (0.35 + 0.65 * heat)
    painted = dimmed * (1 - alpha * heat) + accent * alpha * heat
    return Image.fromarray(painted.clip(0, 255).astype(np.uint8))


def depth_image(depth: np.ndarray):
    """Parietal's depth map: near is bright amber, far is dark."""
    from PIL import Image

    d = np.asarray(depth, np.float32)
    lo, hi = _rgb(C.UI_PANEL), _rgb(C.UI_ACCENT)
    rgb = np.stack([lo[i] + (hi[i] - lo[i]) * d for i in range(3)], -1)
    return Image.fromarray(rgb.clip(0, 255).astype(np.uint8))


# ---------------------------------------------------------------------------
# Heatmaps, histograms, rasters -- the WEIGHTS and ACTIVITY views
# ---------------------------------------------------------------------------


def _fit(array: np.ndarray, rows: int, cols: int) -> np.ndarray:
    """Block-mean a 2-D array down to at most rows x cols (keeps sign)."""
    a = np.asarray(array, np.float32)
    r, c = a.shape
    if r > rows:
        edges = np.linspace(0, r, rows + 1).astype(int)
        a = np.stack([a[x:y].mean(0) for x, y in zip(edges[:-1], edges[1:])])
    if a.shape[1] > cols:
        edges = np.linspace(0, a.shape[1], cols + 1).astype(int)
        a = np.stack([a[:, x:y].mean(1) for x, y in zip(edges[:-1], edges[1:])], 1)
    return a


def heatmap(array, width: int, height: int, signed: bool = True, peak: float | None = None):
    """A 2-D array as an image. Signed: blue negative, dark zero, amber positive.
    Unsigned: dark to amber. `peak` fixes the scale; otherwise the 99th
    percentile of |value| is used, so one outlier cannot wash out the rest."""
    from PIL import Image

    width, height = max(int(width), 8), max(int(height), 8)
    a = np.asarray(array, np.float32)
    if a.ndim == 1:
        side = int(np.ceil(np.sqrt(a.size)))
        a = np.pad(a, (0, side * side - a.size)).reshape(side, side)
    a = _fit(a, height, width)
    if peak is None:
        peak = float(np.percentile(np.abs(a), 99)) if a.size else 1.0
    peak = max(peak, 1e-9)
    t = np.clip(a / peak, -1, 1)
    panel, hot, cold = (np.array(_rgb(c), np.float32)
                        for c in (C.UI_PANEL, C.UI_ACCENT, C.UI_NEG))
    if signed:
        rgb = np.where(t[..., None] >= 0, panel + (hot - panel) * t[..., None],
                       panel + (cold - panel) * (-t[..., None]))
    else:
        t = np.clip(np.abs(t), 0, 1)
        rgb = panel + (hot - panel) * t[..., None]
    image = Image.fromarray(rgb.clip(0, 255).astype(np.uint8))
    return image.resize((width, height), Image.NEAREST)


def histogram(values, width: int, height: int, bins: int = 64):
    """Distribution of values, with the zero line marked."""
    from PIL import Image, ImageDraw

    width, height = max(int(width), 40), max(int(height), 20)
    image = Image.new("RGB", (width, height), _rgb(C.UI_PANEL))
    v = np.asarray(values, np.float32).reshape(-1)
    if not v.size:
        return image
    low, high = np.percentile(v, [0.5, 99.5])
    if high <= low:
        high = low + 1e-6
    counts, edges = np.histogram(np.clip(v, low, high), bins=bins, range=(low, high))
    draw = ImageDraw.Draw(image)
    top = max(int(counts.max()), 1)
    bar = width / bins
    for i, count in enumerate(counts):
        h = (height - 4) * count / top
        colour = _rgb(C.UI_NEG) if edges[i + 1] <= 0 else _rgb(C.UI_ACCENT)
        draw.rectangle([i * bar, height - h, (i + 1) * bar - 1, height], fill=colour)
    if low < 0 < high:
        x = width * (0 - low) / (high - low)
        draw.line([(x, 0), (x, height)], fill=_rgb(C.UI_FG))
    return image


def raster(rows: list, width: int, height: int, bins: int = 192):
    """Neuron activity per block: one row per block, one column per group of
    neurons (mean activation), blue below zero, amber above."""
    usable = [np.asarray(r, np.float32) for r in rows if r is not None]
    if not usable:
        return heatmap(np.zeros((1, 1)), width, height)
    matrix = np.stack([_fit(r[None], 1, bins)[0] if r.size > bins else
                       np.pad(r, (0, bins - r.size)) for r in usable])
    return heatmap(matrix, width, height, signed=True)
