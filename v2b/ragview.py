"""
ragview.py -- the hippocampus drawn as the network it is (the dashboard's RAG tab).

THE PATH, left to right, every state and gate live for this percept:

    POT vector -> ENTORHINAL (trained, 1,152 x 256) -> memory key
      -> STORE: every memory is one unit whose weights are its stored key;
         its activation is the cosine with this percept's key
      -> the k most active -> CA1 vote (softmax of similarity x sharpness)
      -> FAMILIARITY gate -> TRUST gate -> + cortex = frontal's scores
      -> AMYGDALA (7 inputs -> 32 hidden -> value) -> GO / HOLD gate -> MOTOR
    plus the EPISODES you taught, on their own direct path.

THE WHOLE STORE as that one layer: each cell is one stored memory, grouped by
where it came from and sorted by label, lit by this percept. Before anything
is perceived it shows what is stored; hover any cell to see that memory.

Everything is drawn into one PIL image, as drawing.py does: 218,623 cells as
Tk rectangles would take seconds per frame.
"""

from __future__ import annotations

import numpy as np

from v2b import config_v2b as C
from v2b.drawing import _blocks, _mix, _rgb

GROUPS = (("cifar", "CIFAR-100 pictures"), ("tiny", "Tiny-ImageNet pictures"),
          ("inet", "ImageNet photos"), ("esc50", "ESC-50 sounds"))
#  Cosine -> colour. Low is near the background so the few memories this
#  percept really activates stand out of ~219,000.
RAMP = ((0.0, "#0b1016"), (0.40, "#123a5a"), (0.72, C.UI_COOL), (0.90, C.UI_FG),
        (1.0, C.UI_ACCENT))
STAGES = ("POT", "ENTORHINAL", "STORE", "CA1", "GATES", "FRONTAL", "AMYGDALA", "MOTOR")


def _font(size: int, bold: bool = False):
    from PIL import ImageFont
    try:
        return ImageFont.truetype("consolab.ttf" if bold else "consola.ttf", size)
    except Exception:                                         # noqa: BLE001
        return ImageFont.load_default()


def _lut() -> np.ndarray:
    xs = np.linspace(0, 1, 256)
    out = np.zeros((256, 3), np.uint8)
    for i, x in enumerate(xs):
        for (a, ca), (b, cb) in zip(RAMP[:-1], RAMP[1:]):
            if a <= x <= b:
                out[i] = _mix(ca, cb, (x - a) / max(b - a, 1e-9))
                break
    return out


class StoreLayout:
    """Where each stored memory sits in the store raster. Built once per store."""

    def __init__(self, labels: np.ndarray, sources: list[str], names: list[str]):
        self.labels = np.asarray(labels)
        self.sources = sources
        self.names = names
        prefix = np.array([str(s).split(":")[0].split("-")[0] for s in sources])
        self.groups = []
        for key, title in GROUPS + tuple((p, p) for p in sorted(set(prefix) - {g for g, _ in GROUPS})):
            members = np.flatnonzero(prefix == key)
            if len(members):
                #  Sorted by label, so a class is one contiguous run of cells.
                members = members[np.lexsort((members, self.labels[members]))]
                self.groups.append((title, members))
        self.size = None

    def fit(self, width: int, height: int) -> None:
        """Choose cells: 1 px wide, as tall as fits; several memories per cell
        only when even 1 px rows would not fit (each cell then shows its most
        active memory)."""
        if self.size == (width, height):
            return
        self.size = (width, height)
        #  Every group needs at least one row however many memories share a
        #  cell; below that the loop below would never end.
        height = max(int(height), len(self.groups), 1)
        cols = max(int(width), 1)
        per = 1
        while sum(-(-len(m) // (cols * per)) for _, m in self.groups) > max(height, 1):
            per += 1
        rows = sum(-(-len(m) // (cols * per)) for _, m in self.groups)
        self.cols, self.per = cols, per
        self.cell_h = max(1, height // max(rows, 1))
        self.rows = rows
        self.members = np.full((rows * cols, per), -1, np.int64)
        self.bands = []                                   # (title, first row, rows, count)
        row = 0
        for title, members in self.groups:
            span = -(-len(members) // (cols * per))
            padded = np.full(span * cols * per, -1, np.int64)
            padded[:len(members)] = members
            self.members[row * cols:(row + span) * cols] = padded.reshape(span * cols, per)
            self.bands.append((title, row, span, len(members)))
            row += span
        self.cell_of = np.full(len(self.labels), -1, np.int64)
        cells, slots = np.nonzero(self.members >= 0)
        self.cell_of[self.members[cells, slots]] = cells

    def raster(self, similarity: np.ndarray | None, lo: float, hi: float) -> np.ndarray:
        """-> uint8 (rows*cell_h, cols, 3)."""
        lut = _lut()
        filled = self.members >= 0
        if similarity is not None and len(similarity) == len(self.labels):
            value = np.full(self.members.shape, -np.inf, np.float32)
            value[filled] = similarity[self.members[filled]]
            best = value.max(1)
            level = np.clip((best - lo) / max(hi - lo, 1e-6), 0, 1)
            colour = lut[(level * 255).astype(int)]
        else:
            #  At rest: what is stored. Alternate shades mark class boundaries.
            first = np.where(filled[:, 0], self.members[:, 0], 0)
            rank = np.zeros(len(self.labels), np.int64)
            for _, members in self.groups:
                change = np.r_[0, np.cumsum(np.diff(self.labels[members]) != 0)]
                rank[members] = change
            tint = {0: C.UI_COOL, 1: C.UI_GOOD, 2: C.UI_ACCENT}
            colour = np.zeros((len(first), 3), np.uint8)
            for g, (_, members) in enumerate(self.groups):
                inside = np.isin(first, members) & filled[:, 0]
                for parity, level in ((0, 0.30), (1, 0.48)):
                    pick = inside & (rank[first] % 2 == parity)
                    colour[pick] = _mix(C.UI_PANEL, tint.get(g, C.UI_MID), level)
        colour[~filled[:, 0]] = _rgb(C.UI_PANEL)
        image = colour.reshape(self.rows, self.cols, 3)
        return np.repeat(image, self.cell_h, axis=0)

    def memory_at(self, x: int, y: int, similarity: np.ndarray | None) -> int:
        """Raster pixel -> the memory shown there (-1 for none)."""
        row, col = int(y) // self.cell_h, int(x)
        if not (0 <= row < self.rows and 0 <= col < self.cols):
            return -1
        members = self.members[row * self.cols + col]
        members = members[members >= 0]
        if not len(members):
            return -1
        if similarity is not None and len(similarity) == len(self.labels):
            return int(members[np.argmax(similarity[members])])
        return int(members[0])

    def position(self, index: int) -> tuple[int, int] | None:
        """A memory's pixel (x, y centre) in the raster."""
        if not 0 <= index < len(self.cell_of) or self.cell_of[index] < 0:
            return None
        cell = int(self.cell_of[index])
        return cell % self.cols, (cell // self.cols) * self.cell_h + self.cell_h // 2


# ---------------------------------------------------------------------------
# Drawing
# ---------------------------------------------------------------------------


class _Pen:
    def __init__(self, draw):
        self.d = draw
        self.fonts: dict = {}

    def font(self, size, bold=False):
        if (size, bold) not in self.fonts:
            self.fonts[size, bold] = _font(size, bold)
        return self.fonts[size, bold]

    def text(self, xy, s, colour=C.UI_FG, size=11, bold=False, anchor="la"):
        self.d.text(xy, s, fill=_rgb(colour), font=self.font(size, bold), anchor=anchor)

    def width(self, s, size=10, bold=False) -> float:
        return self.font(size, bold).getlength(s)

    def fit(self, s, room, size=10, bold=False) -> str:
        """s, cut to what fits in `room` pixels."""
        while s and self.width(s, size, bold) > room:
            s = s[:-1]
        return s

    def wrap(self, s, room, size=10) -> list:
        lines, line = [], ""
        for word in s.split():
            trial = (line + " " + word).strip()
            if line and self.width(trial, size) > room:
                lines.append(line)
                line = word
            else:
                line = trial
        return lines + ([line] if line else [])

    def elbow(self, x0, y0, x1, y1, level, lit, cut=False, label=""):
        """A connection routed around the boxes: down/up to `level`, across,
        then into the target, arrowhead last."""
        colour = C.UI_ALERT if cut else (C.UI_ACCENT if lit else C.UI_EDGE_IDLE)
        width = 2 if lit or cut else 1
        self.d.line([(x0, y0), (x0, level), (x1, level), (x1, y1)], fill=_rgb(colour),
                    width=width, joint="curve")
        tip = -6 if y1 < level else 6
        self.d.polygon([(x1, y1), (x1 - 5, y1 - tip), (x1 + 5, y1 - tip)], fill=_rgb(colour))
        if label:
            self.text(((x0 + x1) // 2, level - 2 if y1 > level else level + 2), label,
                      C.UI_ALERT if cut else (C.UI_MID if lit else C.UI_DIM), 10,
                      anchor="md" if y1 > level else "ma")
        if cut:
            mx = (x0 + x1) // 2
            self.d.line((mx - 5, level - 6, mx + 5, level + 6), fill=_rgb(C.UI_ALERT), width=3)
            self.d.line((mx - 5, level + 6, mx + 5, level - 6), fill=_rgb(C.UI_ALERT), width=3)

    def box(self, x0, y0, x1, y1, lit, title, colour=C.UI_COOL):
        self.d.rectangle((x0, y0, x1, y1), fill=_rgb(C.UI_PANEL),
                         outline=_rgb(colour if lit else C.UI_EDGE_IDLE), width=2 if lit else 1)
        self.text((x0 + 6, y0 + 5), self.fit(title, x1 - x0 - 10, 12, True),
                  colour if lit else C.UI_DIM, 12, True)

    def arrow(self, x0, y, x1, lit, label="", cut=False):
        colour = C.UI_ALERT if cut else (C.UI_ACCENT if lit else C.UI_EDGE_IDLE)
        self.d.line((x0, y, x1 - 6, y), fill=_rgb(colour), width=3 if lit else 1)
        self.d.polygon([(x1, y), (x1 - 8, y - 5), (x1 - 8, y + 5)], fill=_rgb(colour))
        if cut:
            mx = (x0 + x1) // 2
            self.d.line((mx - 5, y - 6, mx + 5, y + 6), fill=_rgb(C.UI_ALERT), width=3)
            self.d.line((mx - 5, y + 6, mx + 5, y - 6), fill=_rgb(C.UI_ALERT), width=3)
        if label:
            self.text(((x0 + x1) // 2, y - 7), label, C.UI_ALERT if cut else C.UI_MID, 10,
                      anchor="md")

    def column(self, x, y0, y1, values, lit, count=16, colour=C.UI_COOL):
        """A layer as a column of circles (block means of |activation|)."""
        levels = _blocks(np.asarray(values, np.float32), count) if values is not None and len(values) \
            else np.zeros(count)
        step = (y1 - y0) / count
        r = max(2, min(7, int(step / 2) - 1))
        for i, level in enumerate(levels):
            cy = int(y0 + step * (i + 0.5))
            if lit:
                self.d.ellipse((x - r, cy - r, x + r, cy + r),
                               fill=_mix(C.UI_PANEL, colour, 0.15 + 0.85 * float(level)))
            else:
                self.d.ellipse((x - r, cy - r, x + r, cy + r), outline=_rgb(C.UI_EDGE_IDLE))

    def gauge(self, x0, y, w, value, lit, colour, mark=None):
        """A gate drawn as a valve: how open it is, 0 .. 1."""
        self.d.rectangle((x0, y, x0 + w, y + 10), outline=_rgb(C.UI_EDGE_IDLE),
                         fill=_rgb(C.UI_BG))
        if lit:
            self.d.rectangle((x0, y, x0 + int(w * min(max(value, 0), 1)), y + 10),
                             fill=_rgb(colour))
        if mark is not None:
            mx = x0 + int(w * mark)
            self.d.line((mx, y - 3, mx, y + 13), fill=_rgb(C.UI_FG), width=2)


def render(trace: dict | None, layout: StoreLayout | None, width: int, height: int,
           reveal: float = 99.0, hover: int = -1):
    """-> (PIL image, store raster origin (x, y) or None)."""
    from PIL import Image, ImageDraw

    width, height = max(int(width), 640), max(int(height), 420)
    image = Image.new("RGB", (width, height), _rgb(C.UI_BG))
    pen = _Pen(ImageDraw.Draw(image))
    t = trace or {}
    path_h = max(250, min(330, int(height * 0.50)))
    _draw_path(pen, t, width, path_h, reveal,
               len(layout.labels) if layout is not None else t.get("bank", 0))
    origin = None
    if layout is not None:
        origin = _draw_store(pen, image, t, layout, 0, path_h + 8, width, height - path_h - 8,
                             reveal, hover)
    return image, origin


def _draw_path(pen: _Pen, t: dict, width: int, h: int, reveal: float, n: int) -> None:
    gates = t.get("gates", {})
    lit = [bool(t) and reveal > s for s in range(len(STAGES))]
    # columns: POT | ENT | STORE | CA1 | GATES | FRONTAL | AMYGDALA | MOTOR
    shares = (0.06, 0.07, 0.12, 0.13, 0.13, 0.23, 0.16, 0.10)
    gap = 18
    usable = width - gap * (len(shares) + 1)
    xs, x = [], gap
    for share in shares:
        xs.append((x, x + int(usable * share)))
        x += int(usable * share) + gap
    top, bottom = 22, h - 44
    mid = (top + bottom) // 2
    pen.text((gap, 4), "THE PATH -- every state and gate, live", C.UI_ACCENT, 12, True)

    def frame(i, title, colour=C.UI_COOL):
        x0, x1 = xs[i]
        pen.box(x0, top, x1, bottom, lit[i], title, colour)
        return x0, x1

    def link(i, label="", cut=False):
        pen.arrow(xs[i][1] + 1, mid, xs[i + 1][0] - 1, lit[i + 1], label, cut)

    # 1. POT vector
    x0, x1 = frame(0, "POT")
    q = t.get("query")
    pen.column((x0 + x1) // 2, top + 24, bottom - 26, q, lit[0], 18, C.UI_ACCENT)
    pen.text(((x0 + x1) // 2, bottom - 8),
             pen.fit("%d units" % (len(q) if q is not None else 1152), x1 - x0 - 4),
             C.UI_MID, 10, anchor="ms")
    link(0, "", not gates.get("POT->hippocampus", True))

    # 2. Entorhinal key
    x0, x1 = frame(1, "ENTORHINAL")
    pen.column((x0 + x1) // 2, top + 24, bottom - 26, t.get("key"), lit[1], 16)
    pen.text(((x0 + x1) // 2, bottom - 8), "key %d" % C.KEY_DIM, C.UI_MID, 10, anchor="ms")
    pen.text((x0, bottom + 6), "W %dx%d, trained" % (len(q) if q is not None else 1152, C.KEY_DIM),
             C.UI_DIM, 10)
    link(1)

    # 3. The store, as a layer: its k most active units
    x0, x1 = frame(2, "STORE  (DG/CA3)")
    sim = t.get("similarity")
    pen.text((x0 + 6, top + 22), pen.fit("%s units" % f"{n:,}", x1 - x0 - 10), C.UI_MID, 10)
    pen.text((x0 + 6, top + 35), pen.fit("act = cosine(key, its key)", x1 - x0 - 10),
             C.UI_DIM, 10)
    recalled = t.get("recalled", [])
    if lit[2] and recalled:
        sims = [r["similarity"] for r in recalled]
        lo = min(sims) - 0.05
        #  Float steps: integer bar widths of 1 px drew inverted rectangles
        #  (and crashed the tab) on any canvas under ~820 px.
        step = max(x1 - x0 - 12, 1) / max(len(sims), 1)
        base = bottom - 30
        for i, s in enumerate(sims):
            bh = int((base - top - 62) * (s - lo) / max(1 - lo, 1e-6))
            left = x0 + 6 + int(i * step)
            right = max(left, x0 + 6 + int((i + 1) * step) - 2)
            pen.d.rectangle((left, base - bh, right, base),
                            fill=_mix(C.UI_PANEL, C.UI_COOL, 0.35 + 0.65 * (i == 0)))
        pen.text((x0 + 6, base + 4), pen.fit("the %d most active" % len(sims), x1 - x0 - 10),
                 C.UI_FG, 10)
        pen.text((x0 + 6, base + 16), pen.fit("closest %.3f  k-th %.3f" % (sims[0], sims[-1]),
                                               x1 - x0 - 10), C.UI_MID, 10)
    elif t and lit[2]:
        why = ("POT->hippocampus cut" if not gates.get("POT->hippocampus", True)
               else "the store is empty")
        pen.text((x0 + 6, mid - 8), "not searched", C.UI_ALERT, 11, True)
        pen.text((x0 + 6, mid + 8), pen.fit(why, x1 - x0 - 10), C.UI_ALERT, 10)
    link(2)

    # 4. CA1 vote
    x0, x1 = frame(3, "CA1 vote")
    if lit[3]:
        pen.text((x0 + 6, top + 22), pen.fit("softmax(sim x %.1f)" % t.get("sharpness", 0),
                                             x1 - x0 - 10), C.UI_MID, 10)
        for i, (name, share) in enumerate(t.get("votes", [])[:7]):
            yy = top + 42 + i * 22
            pen.text((x0 + 6, yy), pen.fit(name, x1 - x0 - 46), C.UI_FG, 10)
            w = int((x1 - x0 - 12) * share)
            pen.d.rectangle((x0 + 6, yy + 12, x0 + 6 + max(w, 1), yy + 17),
                            fill=_rgb(C.UI_GOOD))
            pen.text((x1 - 6, yy), "%.0f%%" % (100 * share), C.UI_MID, 10, anchor="ra")
    link(3, "", not gates.get("hippocampus->frontal", True))

    # 5. The gates on memory's vote
    x0, x1 = frame(4, "GATES", C.UI_GOOD)
    room = x1 - x0 - 10
    if lit[4]:
        fam, at, wd = t.get("familiarity", 0), t.get("familiar_at", 0), t.get("familiar_width", 1)
        y = top + 26
        if t.get("recalled"):
            pen.text((x0 + 6, y), "FAMILIARITY  %.2f" % fam,
                     C.UI_GOOD if fam >= 0.5 else C.UI_ALERT, 11, True)
            pen.text((x0 + 6, y + 15), pen.fit("sigmoid((%.3f-%.3f)/%.3f)"
                                               % (t.get("closest", 0), at, wd), room), C.UI_MID, 10)
            pen.gauge(x0 + 6, y + 31, x1 - x0 - 12, fam, True,
                      C.UI_GOOD if fam >= 0.5 else C.UI_ALERT, mark=0.5)
            pen.text((x0 + 6, y + 46), "familiar" if fam >= 0.5 else "NOVEL: memory muted",
                     C.UI_MID, 10)
        else:
            pen.text((x0 + 6, y), "FAMILIARITY  --", C.UI_ALERT, 11, True)
            pen.text((x0 + 6, y + 15), pen.fit("nothing recalled to judge", room), C.UI_MID, 10)
            pen.gauge(x0 + 6, y + 31, x1 - x0 - 12, 0, False, C.UI_ALERT, mark=0.5)
        y += 72
        trust = t.get("trust", 0)
        pen.text((x0 + 6, y), "TRUST  %.3f" % trust, C.UI_COOL, 11, True)
        pen.text((x0 + 6, y + 15), pen.fit("fitted on validation", room), C.UI_MID, 10)
        pen.gauge(x0 + 6, y + 31, x1 - x0 - 12, trust, True, C.UI_COOL)
        y += 56
        open_ = t.get("use", False)
        pen.text((x0 + 6, y), pen.fit("to frontal: %s" % ("x %.3f" % (trust * fam) if open_
                                                          else "CUT"), room),
                 C.UI_FG if open_ else C.UI_ALERT, 10)
    link(4)

    # 6. Frontal's sum, name by name
    e = t.get("expertise") or {}
    x0, x1 = frame(5, "FRONTAL  scores%s" % (
        "  (expert re-read, keep %.2f)" % e["keep"] if e.get("used") else
        "  (keep %.2f)" % e["cautious_keep"] if e.get("cautious_keep") is not None else ""))
    if lit[5]:
        rows = t.get("ledger", [])[:8]
        yours = ["%+.1f" % r["episodes"] if abs(r["episodes"]) > 0.05 else "" for r in rows]
        narrow = x1 - x0 < 260
        cells = [("cx" if narrow else "cortex", ["%.1f" % r["cortex"] for r in rows]),
                 ("+mem" if narrow else "+memory", ["%+.1f" % r["memory"] for r in rows])]
        if any(yours):
            cells.append(("+you" if narrow else "+yours", yours))
        cells.append(("=" if narrow else "=final", ["%.1f" % r["final"] for r in rows]))
        #  Columns laid out right to left by their widest entry, so numbers
        #  never land on names; the name gets what is left.
        edges, right = [], x1 - 8
        for head, values in reversed(cells):
            edges.insert(0, right)
            right -= max(pen.width(v, 10, True) for v in values + [head]) + 8
        name_room = right - (x0 + 6) - 4
        y = top + 24
        for edge, (head, _) in zip(edges, cells):
            pen.text((edge, y), head, C.UI_MID, 10, anchor="ra")
        for i, row in enumerate(rows):
            yy = y + 16 + i * 17
            first = i == 0
            pen.text((x0 + 6, yy), pen.fit(row["name"], name_room, 10, first),
                     C.UI_ACCENT if first else C.UI_FG, 10, first)
            colours = [C.UI_FG, C.UI_GOOD if row["memory"] > -0.05 else C.UI_MID] + \
                [C.UI_GOOD] * (len(cells) - 3) + [C.UI_ACCENT if first else C.UI_FG]
            bolds = [False] * (len(cells) - 1) + [first]
            for edge, (_, values), colour, bold in zip(edges, cells, colours, bolds):
                pen.text((edge, yy), values[i], colour, 10, bold, anchor="ra")
    link(5, "", not gates.get("POT->amygdala", True))
    # hippocampus -> amygdala (below the boxes) and amygdala -> frontal (above)
    hears = t.get("amygdala", {}).get("hears_memory", True)
    pen.elbow((xs[3][0] + xs[3][1]) // 2, bottom, xs[6][0] + 26, bottom, bottom + 9, lit[6],
              bool(t) and not hears, "hippocampus->amygdala")
    pen.elbow(xs[6][0] + 26, top, (xs[5][0] + xs[5][1]) // 2, top, top - 6, lit[6],
              bool(t) and not gates.get("amygdala->frontal", True),
              "amygdala->frontal (bad episodes)")

    # 7. Amygdala: 7 inputs -> hidden -> value
    x0, x1 = frame(6, "AMYGDALA", C.UI_ACCENT)
    a = t.get("amygdala", {})
    inputs = list(a.get("inputs", {}).items())
    hidden = a.get("hidden")
    span_top, span_bottom = top + 26, bottom - 18
    ix, column_x, vx = x0 + 14, x1 - 46, x1 - 16
    label_room = column_x - 10 - (ix + 9)
    if inputs:
        step = (span_bottom - span_top) / len(inputs)
        for i, (name, value) in enumerate(inputs):
            cy = int(span_top + step * (i + 0.5))
            level = min(abs(value), 1.0)
            pen.d.ellipse((ix - 5, cy - 5, ix + 5, cy + 5),
                          fill=_mix(C.UI_PANEL, C.UI_ACCENT, 0.15 + 0.85 * level) if lit[6] else None,
                          outline=_rgb(C.UI_EDGE_IDLE))
            if lit[6]:
                pen.text((ix + 9, cy), pen.fit("%.2f %s" % (value, name), label_room),
                         C.UI_MID, 10, anchor="lm")
    pen.column(column_x, span_top, span_bottom, hidden, lit[6], 16, C.UI_ACCENT)
    if lit[6]:
        pen.text((column_x, span_bottom + 14), "%d hidden" % (len(hidden) if hidden is not None
                                                              else C.VALUE_HIDDEN),
                 C.UI_MID, 10, anchor="ms")
        v = a.get("value_net", 0)
        used = a.get("used", True)
        pen.d.ellipse((vx - 9, mid - 9, vx + 9, mid + 9),
                      fill=_mix(C.UI_PANEL, C.UI_ACCENT, v) if used else None,
                      outline=_rgb(C.UI_EDGE_IDLE if used else C.UI_ALERT))
        pen.text((vx, mid + 14), "%.2f" % v if used else "unused", C.UI_FG if used else C.UI_ALERT,
                 10, anchor="ma")
        if not a.get("hears_memory", True):
            pen.text((x0 + 6, top + 20), pen.fit("memory unheard: uniform, closest 0",
                                                  column_x - 10 - (x0 + 6)), C.UI_ALERT, 10)
    link(6, "", not gates.get("amygdala->motor", True))

    # 8. The GO/HOLD gate and motor
    x0, x1 = frame(7, "MOTOR", C.UI_GOOD if a.get("decision") == "GO" else C.UI_ALERT)
    if lit[7]:
        value, threshold = a.get("value", 0), a.get("threshold", C.ABSTAIN_BELOW)
        decision = a.get("decision", "-")
        go = decision == "GO"
        room = x1 - x0 - 10
        y = top + 26
        if decision == "-":
            pen.text((x0 + 6, y), "NO ANSWER", C.UI_ALERT, 13, True)
            y += 26
        else:
            pen.text((x0 + 6, y), decision, C.UI_GOOD if go else C.UI_ALERT, 13, True)
            pen.text((x0 + 6, y + 18), "value %.2f" % value, C.UI_FG, 10)
            pen.gauge(x0 + 6, y + 34, x1 - x0 - 12, value, True, C.UI_GOOD if go else C.UI_ALERT,
                      mark=threshold)
            pen.text((x0 + 6, y + 48), pen.fit("HOLD below %.2f" % threshold, room), C.UI_MID, 10)
            pen.text((x0 + 6, y + 70), pen.fit(a.get("answer") or "-", room, 12, True),
                     C.UI_ACCENT if go else C.UI_MID, 12, True)
            if not go:
                pen.text((x0 + 6, y + 86), "'not sure'", C.UI_MID, 10)
            y += 104
        for why in a.get("reasons", []):
            for line in pen.wrap(why, room):
                if y < bottom - 12:
                    pen.text((x0 + 6, y), line, C.UI_ALERT, 10)
                y += 13
            y += 3

    # The episodes' direct path, along the bottom (below the
    # hippocampus->amygdala connector at bottom + 9).
    y = bottom + 24
    episodes = t.get("episodes", [])
    applied = [e for e in episodes if e["applies"]]
    caption = ("YOUR EPISODES  %d taught: good ones vote for their name (up to +%.0f), bad ones "
               "push it away, when cosine >= %.2f" % (
                   len(episodes), C.EPISODE_VOTE, t.get("episode_threshold", C.EPISODE_SIMILARITY)))
    caption = pen.fit(caption, width - 2 * gap, 10, bool(applied))
    pen.text((gap, y), caption, C.UI_GOOD if applied else C.UI_DIM, 10, bool(applied))
    x = gap + pen.width(caption, 10, bool(applied)) + 16
    for e in sorted(episodes, key=lambda e: -e["cosine"])[:8]:
        tag = "%s%s %.2f %s" % ("+" if e["valence"] > 0 else "-", e["label"][:12], e["cosine"],
                                e.get("state", ""))
        w = pen.width(tag, 10, e["applies"]) + 14
        if x + w > width - gap:
            if y > bottom + 24:
                break
            x, y = gap, y + 13           # one more line, then stop
        pen.text((x, y), tag, C.UI_GOOD if e["applies"] else C.UI_DIM, 10, e["applies"])
        x += w


def _draw_store(pen: _Pen, image, t: dict, layout: StoreLayout, x0: int, y0: int,
                width: int, height: int, reveal: float, hover: int):
    from PIL import Image
    label_w = 190
    title_h = 22
    legend_h = 18
    layout.fit(width - label_w - 24, height - title_h - legend_h - 6)
    sim = t.get("similarity")
    active = (sim is not None and len(sim) > 0 and len(sim) == len(layout.labels)
              and reveal > 2)
    s = np.asarray(sim, np.float32) if active else None
    lo, hi = (float(np.percentile(s, 50)), float(s.max())) if active else (0.0, 1.0)
    if active:
        state = ", lit by this percept"
    elif t and not t.get("gates", {}).get("POT->hippocampus", True):
        state = " -- not searched: POT->hippocampus is cut"
    elif t and reveal > 2:
        state = " -- nothing to light"
    else:
        state = " (perceive something to light it up)" if not t else ""
    pen.text((x0 + 18, y0), pen.fit("THE WHOLE STORE as one layer -- %s memories, one cell each%s"
                                    % (f"{len(layout.labels):,}", state), width - 260, 12, True),
             C.UI_ACCENT, 12, True)
    pen.text((width - 18, y0 + 2), "hover a cell: which memory it is", C.UI_MID, 10, anchor="ra")
    rx, ry = x0 + label_w, y0 + title_h
    raster = layout.raster(s, lo, hi)
    image.paste(Image.fromarray(raster), (rx, ry))
    for title, row, span, count in layout.bands:
        by = ry + row * layout.cell_h
        pen.text((x0 + 18, by), title, C.UI_FG, 10, True)
        pen.text((x0 + 18, by + 13), "%s memories" % f"{count:,}", C.UI_MID, 10)
        pen.d.line((rx - 6, by, rx - 6, by + span * layout.cell_h - 1), fill=_rgb(C.UI_EDGE_IDLE))
    if active:
        for r in t.get("recalled", []):
            p = layout.position(r["index"])
            if p is not None:
                cx, cy = rx + p[0], ry + p[1]
                pen.d.ellipse((cx - 4, cy - 4, cx + 4, cy + 4), outline=_rgb(C.UI_ACCENT), width=2)
    if hover >= 0:
        p = layout.position(hover)
        if p is not None:
            cx, cy = rx + p[0], ry + p[1]
            pen.d.rectangle((cx - 6, cy - 6, cx + 6, cy + 6), outline=_rgb(C.UI_FG), width=1)
    ly = ry + layout.rows * layout.cell_h + 6
    if active:
        bar_x, bar_w = rx, 220
        lut = _lut()
        for i in range(bar_w):
            pen.d.line((bar_x + i, ly + 3, bar_x + i, ly + 11), fill=tuple(int(c) for c in lut[i * 255 // bar_w]))
        pen.text((bar_x + bar_w + 8, ly + 1), "cosine with this percept's key: %.2f (median) .. %.2f"
                 "   orange rings: the %d recalled" % (lo, hi, len(t.get("recalled", []))),
                 C.UI_MID, 10)
    else:
        pen.text((rx, ly + 1), "shades alternate at each class boundary; colour = where it came from",
                 C.UI_MID, 10)
    if layout.per > 1:
        pen.text((width - 18, ly + 1), "%d memories per cell (each shows its most active)"
                 % layout.per, C.UI_MID, 10, anchor="ra")
    return rx, ry
