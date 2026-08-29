"""
desktop_app.py -- the native dashboard. PROJECT.md section 7.

Native desktop window, no browser. Simple black and white. Fully offline,
real time.

MINIMISED VIEW -- one screen, no scrolling (section 7.1)

    +----------------+---------------------------+----------------+
    |  INPUT (20%)   |     LOBES (centre)        |  SYSTEM (right)|
    |  [ text  ]     |   thalamus + 4 lobes,     |  live state of |
    |  [ image ]     |   connecting streams      |  each network  |
    |  [ video ]     |   light up on data flow   |  CPU / GPU     |
    |  [ audio ]     |   [ expand view ]         |  active / idle |
    +----------------+---------------------------+----------------+
    |              OUTPUT -- chatbot style, text only              |
    +--------------------------------------------------------------+

EXPANDED VIEW (section 7.2) -- a second window with the full architecture
live: thalamus, all four lobes, basal ganglia, and every data stream, in the
exact routing order of section 2.1, with full-size heatmaps. Clicking any
single network in the minimised view opens that network's detail in the same
expanded window.

THE THREADING RULE (section 6.3): inference runs on a worker thread and pushes
activation frames to a queue. The UI drains that queue on its own timer and
does all rendering. Nothing renders inside the inference loop, so the
dashboard's redraw speed never becomes the model's speed.

Usage
    python desktop_app.py
    python desktop_app.py --selftest     build, tick, exit -- no interaction
"""

from __future__ import annotations

import argparse
import queue
import sys
import threading
import time
import tkinter as tk
from dataclasses import dataclass
from tkinter import filedialog

import numpy as np

import config
from modules import diagram
from modules.activations import ActivationFrame
from modules.telemetry import SystemSample, TelemetryMonitor

# ---------------------------------------------------------------------------
# Colour helpers -- everything on screen is a grey between UI_BG and UI_FG
# ---------------------------------------------------------------------------


def _hex_to_rgb(value: str) -> tuple[int, int, int]:
    value = value.lstrip("#")
    return tuple(int(value[i : i + 2], 16) for i in (0, 2, 4))  # type: ignore[return-value]


def mix(colour_a: str, colour_b: str, t: float) -> str:
    """Blend two hex colours. t=0 -> a, t=1 -> b."""
    t = max(0.0, min(1.0, t))
    a, b = _hex_to_rgb(colour_a), _hex_to_rgb(colour_b)
    return "#%02x%02x%02x" % tuple(int(x + (y - x) * t) for x, y in zip(a, b))


def _wrap(text: str, width: int, indent: str = "") -> list[str]:
    """Word-wrap for the fixed-width system panel. Empty text -> no lines."""
    if not text:
        return []
    import textwrap

    return [indent + line for line in textwrap.wrap(text, width)]



def bracket_frame(canvas, x1, y1, x2, y2, colour, arm=None, width=1, tag=None):
    """Draw only the CORNERS of a rectangle, HUD style.

    Corner brackets read as a targeting frame rather than a form field, and
    they leave the long edges empty so the content inside is what the eye
    lands on. Used for every panel and every node.
    """
    arm = arm or config.UI_BRACKET
    arm_x = min(arm, max((x2 - x1) / 2 - 1, 1))
    arm_y = min(arm, max((y2 - y1) / 2 - 1, 1))
    opts = {"fill": colour, "width": width}
    if tag:
        opts["tags"] = tag
    for cx, sx in ((x1, 1), (x2, -1)):
        for cy, sy in ((y1, 1), (y2, -1)):
            canvas.create_line(cx, cy, cx + sx * arm_x, cy, **opts)
            canvas.create_line(cx, cy, cx, cy + sy * arm_y, **opts)


def framed_panel(parent, title, width=None, height=None):
    """A titled panel with a rule under the heading. Returns (frame, body)."""
    outer = tk.Frame(parent, bg=config.UI_BG,
                     highlightthickness=1, highlightbackground=config.UI_EDGE_IDLE)
    if width:
        outer.configure(width=width)
    if height:
        outer.configure(height=height)

    head = tk.Frame(outer, bg=config.UI_BG)
    head.pack(fill=tk.X, padx=8, pady=(5, 0))
    tk.Label(head, text=title, bg=config.UI_BG, fg=config.UI_ACCENT,
             font=config.UI_FONT_TITLE, anchor="w").pack(side=tk.LEFT)
    tk.Frame(outer, bg=config.UI_EDGE_IDLE, height=1).pack(fill=tk.X, padx=8, pady=(3, 0))

    body = tk.Frame(outer, bg=config.UI_BG)
    body.pack(fill=tk.BOTH, expand=True, padx=8, pady=6)
    return outer, body


# ---------------------------------------------------------------------------
# The graph -- node positions in normalised 0..1 space
# ---------------------------------------------------------------------------


@dataclass
class Node:
    key: str
    label: str
    x: float
    y: float
    heatmap: str | None = None    # activation-frame key this node displays
    role: str = ""


#  y is a ROW INDEX expressed as a fraction: 0.0 is the top row, 1.0 the
#  bottom one, evenly divided. Nodes sharing a row share a y. Keeping these on
#  an even grid is what guarantees the boxes never overlap -- see _node_xy.

#  Minimised centre: thalamus + the four lobes (section 7.1). 4 rows.
_M = 1 / 3
MINI_NODES = [
    Node("thalamus", "THALAMUS", 0.50, 0 * _M, None, "modality-keyed relay"),
    Node("occipital", "OCCIPITAL", 0.18, 1 * _M, "lobe.occipital", "features (vision)"),
    Node("temporal", "TEMPORAL", 0.82, 1 * _M, "lobe.temporal", "meaning, language"),
    Node("parietal", "PARIETAL", 0.18, 2 * _M, "lobe.parietal", "spatial relations"),
    Node("frontal", "FRONTAL", 0.50, 3 * _M, "lobe.frontal", "convergence, decision"),
]

#  Expanded: the full architecture, in the exact routing order of section 2.1.
#  7 rows -- occipital and temporal share row 2.
_F = 1 / 6
FULL_NODES = [
    Node("input", "INPUT", 0.50, 0 * _F, None, "image / audio / text"),
    Node("thalamus", "THALAMUS", 0.50, 1 * _F, None, "modality-keyed relay"),
    Node("occipital", "OCCIPITAL", 0.17, 2 * _F, "lobe.occipital", "features (vision)"),
    Node("temporal", "TEMPORAL", 0.83, 2 * _F, "lobe.temporal", "meaning, language"),
    Node("parietal", "PARIETAL", 0.17, 3 * _F, "lobe.parietal", "spatial relations"),
    Node("frontal", "FRONTAL", 0.50, 4 * _F, "lobe.frontal", "convergence, decision"),
    Node("basal_ganglia", "BASAL GANGLIA", 0.50, 5 * _F, "basal_ganglia.striatum",
         "gate: allow or suppress"),
    Node("output", "OUTPUT", 0.50, 6 * _F, None, "the answer"),
]

MINI_EDGES = [
    ("thalamus", "occipital"),
    ("thalamus", "temporal"),
    ("occipital", "parietal"),
    ("occipital", "frontal"),     # the DIRECT delivery -- section 2.1
    ("parietal", "frontal"),
    ("temporal", "frontal"),
]

FULL_EDGES = MINI_EDGES + [
    ("input", "thalamus"),
    ("frontal", "basal_ganglia"),
    ("basal_ganglia", "output"),
]

EDGE_DECAY_SECONDS = 1.2   # how long a stream stays lit after data flows


def enable_dpi_awareness() -> None:
    """Tell Windows this process handles its own scaling.

    Without it, a Tk window on a display scaled above 100% is rendered at 96
    DPI and bitmap-stretched by the compositor -- every line and every glyph
    comes out soft. Heatmaps suffer worst, because a stretched nearest-neighbour
    grid stops being pixel-exact and the per-cell values blur into each other.

    No-op on anything that is not Windows.
    """
    if sys.platform != "win32":
        return
    try:
        import ctypes

        # PROCESS_PER_MONITOR_DPI_AWARE (Windows 8.1+)
        ctypes.windll.shcore.SetProcessDpiAwareness(2)
    except Exception:
        try:
            ctypes.windll.user32.SetProcessDPIAware()   # Vista+ fallback
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Inference worker -- owns the model, never touches a widget
# ---------------------------------------------------------------------------


@dataclass
class WorkerStatus:
    kind: str                 # "loading" | "ready" | "result" | "error"
    text: str = ""
    detail: str = ""
    elapsed_ms: float = 0.0


class BrainWorker(threading.Thread):
    """Loads the model and runs perceive() off the UI thread."""

    def __init__(self) -> None:
        super().__init__(daemon=True, name="brain-worker")
        self.jobs: queue.Queue = queue.Queue()
        self.status: queue.Queue[WorkerStatus] = queue.Queue()
        self.brain = None
        self.language = None
        #  (role, text) turns carried into the next reply
        self.history: list[tuple[str, str]] = []
        self.ready = threading.Event()
        self._stop = threading.Event()
        self.last_error: str = ""

    def submit(self, inputs: dict, label: str) -> None:
        self.jobs.put({"inputs": inputs, "label": label})

    def stop(self) -> None:
        self._stop.set()
        self.jobs.put(None)

    @staticmethod
    def _percept_line(trace) -> str:
        """One line naming what the lobes concluded, before the words start.

        Shown so the perceptual decision and its verbalisation stay visibly
        separate -- if the reply is wrong, this says whether the lobes were
        wrong or the language module embellished.
        """
        answer = trace.answer
        if not answer or not answer.predictions:
            return "lobes: no prediction"
        lobes = ", ".join(n for n in trace.active_nodes if n in config.LOBES)
        top = answer.predictions[0]
        return (f"lobes [{lobes}] -> {trace.domain}: "
                f"{top.label} {top.probability:.0%}")


    def run(self) -> None:
        self.status.put(WorkerStatus("loading", "loading frozen backbones..."))
        try:
            from modules.pipeline import build_brain

            started = time.perf_counter()
            brain = build_brain()
            brain.attach_recorder()
            brain.encoders.image.backbone       # force the vision load now

            # Language production. Loaded here, on the worker thread, because
            # it is half a billion parameters and would freeze the UI.
            from modules.language import LanguageModule

            self.status.put(WorkerStatus("loading", "loading language module..."))
            self.language = LanguageModule()
            self.language._ensure_loaded()
            self.brain = brain
            self.ready.set()
            state = "TRAINED" if brain.trained else "UNTRAINED"
            self.status.put(
                WorkerStatus(
                    "ready",
                    f"ready in {time.perf_counter() - started:.1f}s  (lobes {state})",
                    detail=state,
                )
            )
        except Exception as exc:  # noqa: BLE001
            self.last_error = f"{type(exc).__name__}: {exc}"
            self.status.put(WorkerStatus("error", "model failed to load",
                                         detail=self.last_error))
            return

        while not self._stop.is_set():
            job = self.jobs.get()
            if job is None:
                break
            try:
                started = time.perf_counter()
                trace = self.brain.perceive(job["inputs"], publish=True)
                self.last_error = ""

                # The lobes have finished perceiving. Report what they concluded,
                # then let the language module put it into words.
                self.status.put(
                    WorkerStatus("percept", self._percept_line(trace),
                                 detail=job["label"])
                )

                from modules.language import perception_from_trace

                spoken = job["inputs"].get("text", "")
                percept = perception_from_trace(
                    trace, user_text=spoken if isinstance(spoken, str) else ""
                )
                self.status.put(WorkerStatus("speaking"))
                reply = self.language.respond(
                    percept,
                    history=self.history,
                    on_token=lambda piece: self.status.put(
                        WorkerStatus("token", piece)
                    ),
                )
                self.history.append(("user", percept.user_text or job["label"]))
                self.history.append(("assistant", reply))

                self.status.put(
                    WorkerStatus(
                        "result", reply, detail=job["label"],
                        elapsed_ms=(time.perf_counter() - started) * 1000.0,
                    )
                )
            except Exception as exc:  # noqa: BLE001
                self.last_error = f"{type(exc).__name__}: {exc}"
                self.status.put(
                    WorkerStatus("error", self.last_error, detail=job["label"])
                )


# ---------------------------------------------------------------------------
# Heatmap rendering
# ---------------------------------------------------------------------------


class HeatmapRenderer:
    """numpy 0..1 grid -> greyscale PhotoImage, with a reference cache.

    Tk garbage-collects PhotoImages the instant nothing holds them, which
    shows up as blank rectangles. The cache is what keeps them alive.
    """

    def __init__(self) -> None:
        self._cache: dict[str, object] = {}

    def render(self, key: str, grid: np.ndarray, size: int):
        from PIL import Image

        pixels = np.clip(grid * 255.0, 0, 255).astype(np.uint8)
        image = Image.fromarray(pixels, mode="L").resize(
            (size, size), Image.NEAREST
        )
        return self.from_image(key, image)

    def from_image(self, key: str, image):
        """Cache and return a PhotoImage for an already-rendered PIL image."""
        from PIL import ImageTk

        photo = ImageTk.PhotoImage(image)
        self._cache[key] = photo          # hold the reference
        return photo

    def blank(self, key: str, size: int):
        return self.render(key, np.zeros((8, 8), dtype=np.float32), size)


# ---------------------------------------------------------------------------
# The graph canvas
# ---------------------------------------------------------------------------


class GraphCanvas(tk.Canvas):
    """Draws nodes and streams. Streams light up when data flows and decay."""

    def __init__(
        self,
        master,
        nodes: list[Node],
        edges: list[tuple[str, str]],
        node_w: int = 118,
        node_h: int = 96,
        on_click=None,
        **kwargs,
    ):
        super().__init__(master, bg=config.UI_BG, highlightthickness=0, **kwargs)
        self.nodes = {n.key: n for n in nodes}
        self.order = [n.key for n in nodes]
        self.edges = edges
        self.node_w = node_w
        self.node_h = node_h
        self.on_click = on_click

        self.renderer = HeatmapRenderer()
        self.frame: ActivationFrame | None = None
        #  network key -> [LayerSpec]; set once the worker has a model
        self.topology: dict = {}
        self.edge_lit: dict[tuple[str, str], float] = {}
        self.node_lit: dict[str, float] = {}
        self.selected: str | None = None

        self.bind("<Configure>", lambda _e: self.redraw())
        self.bind("<Button-1>", self._click)

    # -- state ----------------------------------------------------------
    def apply_frame(self, frame: ActivationFrame) -> None:
        self.frame = frame
        now = time.time()
        for edge in frame.active_edges:
            if edge in self.edges:
                self.edge_lit[edge] = now
        # The input->thalamus stream has no cortical edge; light it whenever
        # anything at all arrived.
        if ("input", "thalamus") in self.edges:
            self.edge_lit[("input", "thalamus")] = now
        for node in frame.active_nodes:
            if node in self.nodes:
                self.node_lit[node] = now
        if "output" in self.nodes:
            self.node_lit["output"] = now

    def _heat(self, timestamp: float | None) -> float:
        """1.0 just lit, fading to 0.0 over EDGE_DECAY_SECONDS."""
        if timestamp is None:
            return 0.0
        age = time.time() - timestamp
        return max(0.0, 1.0 - age / EDGE_DECAY_SECONDS)

    ROW_GAP = 9          # minimum clear space between two stacked boxes
    MIN_NODE_H = 26

    def _effective_node_h(self) -> float:
        """Box height that actually fits, given the canvas we ended up with.

        A fixed node height overlaps the moment the pane is shorter than
        rows * node_h -- which is what happens when the expanded window gives
        most of its height to the layer diagram. Shrinking to fit is the only
        thing that cannot collide at some window size.
        """
        rows = len({round(n.y, 4) for n in self.nodes.values()}) or 1
        available = max(self.winfo_height() - 2 * 10, self.MIN_NODE_H)
        fits = available / rows - self.ROW_GAP
        return max(min(self.node_h, fits), self.MIN_NODE_H)

    def _node_xy(self, node: Node) -> tuple[float, float]:
        """Normalised (x, y) -> pixel centre, with the box kept fully on canvas.

        Both axes reserve exactly half a node plus a margin at each end, so a
        node at 0.0 or 1.0 sits flush against the edge rather than half off it.
        """
        margin = 10
        node_h = self._effective_node_h()
        half_w, half_h = self.node_w / 2, node_h / 2
        w = max(self.winfo_width(), self.node_w + 2 * margin + 2)
        h = max(self.winfo_height(), node_h + 2 * margin + 2)
        x = half_w + margin + node.x * (w - self.node_w - 2 * margin)
        y = half_h + margin + node.y * (h - node_h - 2 * margin)
        return x, y

    # -- interaction ----------------------------------------------------
    def _click(self, event) -> None:
        if self.on_click is None:
            return
        for key in self.order:
            x, y = self._node_xy(self.nodes[key])
            if (
                abs(event.x - x) <= self.node_w / 2
                and abs(event.y - y) <= self.node_h / 2
            ):
                self.on_click(key)
                return

    # -- drawing --------------------------------------------------------
    def redraw(self) -> None:
        self.delete("all")
        w, h = self.winfo_width(), self.winfo_height()
        if w < 20 or h < 20:
            return

        for edge in self.edges:
            self._draw_edge(edge)
        for key in self.order:
            self._draw_node(self.nodes[key])

    def _draw_edge(self, edge: tuple[str, str]) -> None:
        src, dst = edge
        if src not in self.nodes or dst not in self.nodes:
            return
        x1, y1 = self._node_xy(self.nodes[src])
        x2, y2 = self._node_xy(self.nodes[dst])

        # Start and end at the box edges, not the centres.
        dx, dy = x2 - x1, y2 - y1
        length = max((dx * dx + dy * dy) ** 0.5, 1e-6)
        ox, oy = dx / length, dy / length
        inset = self._effective_node_h() / 2 + 2
        x1, y1 = x1 + ox * inset, y1 + oy * inset
        x2, y2 = x2 - ox * inset, y2 - oy * inset

        heat = self._heat(self.edge_lit.get(edge))
        colour = mix(config.UI_EDGE_IDLE, config.UI_ACCENT, heat)
        self.create_line(
            x1, y1, x2, y2,
            fill=colour,
            width=1 + 2 * heat,
            arrow=tk.LAST,
            arrowshape=(9, 11, 4),
        )

    def _draw_node(self, node: Node) -> None:
        x, y = self._node_xy(node)
        node_h = self._effective_node_h()
        half_w, half_h = self.node_w / 2, node_h / 2
        heat = self._heat(self.node_lit.get(node.key))

        live = node.key in (self.frame.active_nodes if self.frame else [])
        outline = mix(config.UI_EDGE_IDLE, config.UI_ACCENT, heat) if live             else config.UI_EDGE_IDLE
        width = 1 + int(heat > 0.15)
        if self.selected == node.key:
            outline, width = config.UI_ACCENT, 2

        self.create_rectangle(
            x - half_w, y - half_h, x + half_w, y + half_h,
            outline="", fill=config.UI_PANEL,
        )
        bracket_frame(self, x - half_w, y - half_h, x + half_w, y + half_h,
                      outline, arm=9, width=width)
        self.create_text(
            x, y - half_h + 10,
            text=node.label,
            fill=mix(config.UI_MID, config.UI_ACCENT, heat) if live else config.UI_DIM,
            font=config.UI_FONT_SMALL,
        )

        # A miniature of the network's REAL layer structure -- input column,
        # hidden column(s), output column -- with live per-neuron activation.
        # Edges are off at this size: at 90px wide a fan is noise, not signal.
        #
        # The diagram is anchored BELOW the label band rather than centred in
        # the box. Centring drew the columns straight through the node's own
        # name, because the label sits inside the box, not above it.
        layers = self._layers_for(node)
        label_band = 18
        caption_band = 12 if node_h >= 70 else 0
        top = y - half_h + label_band
        box_w = int(self.node_w - 16)
        box_h = int(node_h - label_band - caption_band - 9)
        if layers and self.frame is not None and box_h >= 10:
            image, columns = diagram.render(
                layers,
                self.frame.vectors,
                box_w,
                box_h,
                # Thumbnail-sized circles, and connections kept on: at three
                # columns the fan is sparse enough to read as wiring.
                edges=True,
                label_space=0,
                min_node_d=4,
                max_node_d=8,
                max_nodes=7,
            )
            photo = self.renderer.from_image(f"{id(self)}:{node.key}", image)
            self.create_image(x, top, image=photo, anchor="n")
            if columns and node_h >= 70:
                firing = [c for c in columns if c.active is not None]
                if firing:
                    total_on = sum(c.active for c in firing)
                    total = sum(c.total for c in firing)
                    self.create_text(
                        x, y + half_h - 7,
                        text=f"{total_on}/{total} firing",
                        fill=mix(config.UI_DIM, config.UI_FG, heat),
                        font=config.UI_FONT_SMALL,
                    )
        else:
            self.create_text(
                x, y + 6,
                text=self._placeholder(node),
                fill=config.UI_DIM,
                font=config.UI_FONT_SMALL,
                width=self.node_w - 12,
                justify=tk.CENTER,
            )

    def _layers_for(self, node: Node, full: bool = False) -> list:
        """The layer chain this node should draw.

        `full` gives every layer; otherwise only the primary input/hidden/output
        columns, which is all that fits in a node box.
        """
        if not self.topology:
            return []
        key = node.key
        if key == "thalamus":
            # Whichever relay nucleus actually fired this pass.
            for name in config.RELAY_NUCLEI:
                candidate = f"thalamus.{name}"
                if candidate in self.topology and self.frame is not None:
                    if any(
                        spec.key in self.frame.vectors
                        for spec in self.topology[candidate]
                    ):
                        key = candidate
                        break
            else:
                key = f"thalamus.{config.RELAY_NUCLEI[0]}"
        layers = self.topology.get(key, [])
        return layers if full else [spec for spec in layers if spec.primary]

    def _grid_for(self, node: Node) -> np.ndarray | None:
        """Spatial conv map, where one genuinely exists. Lobes have none."""
        if self.frame is None or node.heatmap is None:
            return None
        return self.frame.heatmaps.get(node.heatmap)

    def _placeholder(self, node: Node) -> str:
        if self.frame is None:
            return "idle"
        if node.key == "input":
            return self.frame.modality
        if node.key == "output":
            top = self.frame.meta.get("top") or []
            return top[0][0][:16] if top else "-"
        return "idle"


# ---------------------------------------------------------------------------
# Expanded view (section 7.2)
# ---------------------------------------------------------------------------


class ExpandedWindow(tk.Toplevel):
    """Full architecture live, plus a detail pane for one selected network."""

    def __init__(self, master, app: "Dashboard"):
        super().__init__(master)
        self.app = app
        self.title("Brain-Inspired AI -- expanded view")
        self.configure(bg=config.UI_BG)
        self.geometry("1340x900")
        self.minsize(1000, 760)

        header = tk.Frame(self, bg=config.UI_BG)
        header.pack(fill=tk.X, padx=12, pady=(10, 4))
        tk.Label(
            header, text="ARCHITECTURE ─ FULL SIGNAL PATH",
            bg=config.UI_BG, fg=config.UI_ACCENT, font=config.UI_FONT_TITLE,
        ).pack(side=tk.LEFT)
        self.seq_label = tk.Label(
            header, text="", bg=config.UI_BG, fg=config.UI_MID, font=config.UI_FONT_SMALL
        )
        self.seq_label.pack(side=tk.RIGHT)

        # --- the layer diagram gets the full window width -----------------
        # A network drawn as input -> hidden -> output columns is a WIDE
        # object. Putting it in a 360px side pane would squeeze eight columns
        # into 45px each, so it takes the top band and the architecture graph
        # moves below it.
        layer_pane = tk.Frame(self, bg=config.UI_BG)
        layer_pane.pack(fill=tk.X, padx=12, pady=(0, 6))

        title_row = tk.Frame(layer_pane, bg=config.UI_BG)
        title_row.pack(fill=tk.X)
        self.detail_title = tk.Label(
            title_row, text="SELECT A NETWORK", bg=config.UI_BG, fg=config.UI_ACCENT,
            font=config.UI_FONT_TITLE, anchor="w",
        )
        self.detail_title.pack(side=tk.LEFT)
        self.detail_role = tk.Label(
            title_row, text="click any node below", bg=config.UI_BG,
            fg=config.UI_MID, font=config.UI_FONT_SMALL, anchor="w",
        )
        self.detail_role.pack(side=tk.LEFT, padx=(10, 0))

        self.detail_canvas = tk.Canvas(
            layer_pane, bg=config.UI_BG, highlightthickness=1,
            highlightbackground=config.UI_DIM, height=330,
        )
        self.detail_canvas.pack(fill=tk.X, pady=(4, 0))
        self.detail_canvas.bind("<Configure>", lambda _e: self._draw_detail())
        self.detail_renderer = HeatmapRenderer()

        # --- architecture graph + numbers ---------------------------------
        body = tk.Frame(self, bg=config.UI_BG)
        body.pack(fill=tk.BOTH, expand=True, padx=12, pady=(0, 12))

        self.detail_text = tk.Text(
            body, bg=config.UI_BG, fg=config.UI_FG, font=config.UI_FONT_SMALL,
            width=52, relief=tk.FLAT, highlightthickness=1,
            highlightbackground=config.UI_DIM, wrap=tk.NONE, padx=8, pady=6,
        )
        self.detail_text.pack(side=tk.RIGHT, fill=tk.Y, padx=(12, 0))
        self.detail_text.configure(state=tk.DISABLED)

        self.graph = GraphCanvas(
            body, FULL_NODES, FULL_EDGES,
            # 7 rows share whatever height is left under the layer diagram, so
            # these boxes are deliberately short. The big diagram above is
            # where the layer detail lives; here the node is just a locator.
            node_w=124, node_h=52,
            on_click=self.select,
        )
        self.graph.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.selected: str | None = None
        self.protocol("WM_DELETE_WINDOW", self.hide)

    def hide(self) -> None:
        self.withdraw()
        self.app.expanded_visible = False

    def show(self, focus: str | None = None) -> None:
        self.deiconify()
        self.lift()
        self.app.expanded_visible = True
        if focus:
            self.select(focus)

    def select(self, key: str) -> None:
        self.selected = key
        self.graph.selected = key
        node = self.graph.nodes.get(key)
        if node is not None:
            self.detail_title.configure(text=node.label)
            self.detail_role.configure(text=node.role)
        self.refresh()

    def apply_frame(self, frame: ActivationFrame) -> None:
        self.graph.apply_frame(frame)
        self.seq_label.configure(
            text=f"frame {frame.sequence}  |  {frame.modality}  |  "
                 f"{frame.meta.get('elapsed_ms', 0):.0f} ms"
        )

    def refresh(self) -> None:
        self.graph.redraw()
        self._draw_detail()

    def _draw_detail(self) -> None:
        canvas = self.detail_canvas
        canvas.delete("all")
        frame = self.graph.frame
        if self.selected is None or frame is None:
            canvas.create_text(
                canvas.winfo_width() / 2 or 170, 160,
                text="no data yet", fill=config.UI_DIM, font=config.UI_FONT,
            )
            self._set_detail_text("")
            return

        node = self.graph.nodes[self.selected]
        layers = self.graph._layers_for(node, full=True)

        width = max(canvas.winfo_width(), 60)
        height = max(canvas.winfo_height(), 60)

        if not layers:
            canvas.create_text(
                width / 2, height / 2,
                text="this stage has no weights of its own",
                fill=config.UI_DIM, font=config.UI_FONT_SMALL,
            )
            self._set_detail_text(self._detail_body(node, frame))
            return

        margin = 12
        label_space = 34
        image, columns = diagram.render(
            layers,
            frame.vectors,
            width - 2 * margin,
            height - 2 * margin,
            edges=True,
            label_space=label_space,
        )
        photo = self.detail_renderer.from_image("detail", image)
        canvas.create_image(margin, margin, image=photo, anchor="nw")

        # Column labels and LIVE firing counts, as crisp Tk text rather than
        # glyphs baked into the bitmap.
        diagram.annotate(
            canvas,
            columns,
            offset_x=margin,
            offset_y=margin,
            bottom=margin + (height - 2 * margin - label_space),
            font=config.UI_FONT_SMALL,
            fg=config.UI_FG,
            dim=config.UI_MID,
        )
        self._set_detail_text(self._detail_body(node, frame, columns))

    def _detail_body(
        self, node: Node, frame: ActivationFrame, columns: list | None = None
    ) -> str:
        lines: list[str] = []

        if columns:
            lines.append("LAYERS THIS PASS")
            lines += diagram.summarise(columns)
            lines.append("")
            lines.append("firing counts are only for layers where a")
            lines.append("unit can genuinely be OFF (GELU, sigmoid).")
            lines.append("linear and norm layers are signed, so they")
            lines.append("report peak |x| instead of a fake count.")
            lines.append("")

        key = node.heatmap
        if node.key == "thalamus":
            lines.append("ROUTING THIS PASS")
            for modality, nucleus, destination, gate in frame.meta.get("routings", []):
                lines.append(f"  {modality:<7} -> {nucleus:<9} -> {destination}")
                lines.append(f"  {'':<7}    relay gate {gate:.3f}")
            key = next(
                (f"thalamus.{n}" for n in config.RELAY_NUCLEI
                 if f"thalamus.{n}" in frame.stats),
                None,
            )
        elif node.key == "basal_ganglia":
            allowed = frame.meta.get("allowed", True)
            lines.append("GATE")
            lines.append(f"  {'ALLOW' if allowed else 'SUPPRESS'}")
            lines.append(f"  value     {frame.meta.get('gate', 0):.3f}")
            lines.append(f"  threshold {config.BG_GATE_THRESHOLD:.2f}")
        elif node.key == "output":
            lines.append(f"LOBES ({config.LABEL_SET})")
            for label, prob in frame.meta.get("top", [])[: config.TOP_K]:
                lines.append(f"  {prob:6.1%}  {label}")
            reference = frame.meta.get("reference", [])
            if reference:
                lines.append("")
                lines.append("FROZEN ENCODER REFERENCE")
                lines.append("  (not the lobes' answer)")
                for label, prob in reference[:3]:
                    lines.append(f"  {prob:6.1%}  {label}")
        elif node.key == "input":
            lines.append(f"MODALITIES  {frame.modality}")

        if key and key in frame.stats:
            stats = frame.stats[key]
            lines.append("")
            lines.append("RAW ACTIVATION (pre-normalisation)")
            lines.append(f"  min   {stats['min']:+.4f}")
            lines.append(f"  max   {stats['max']:+.4f}")
            lines.append(f"  mean  {stats['mean']:+.4f}")
            lines.append(f"  l2    {stats['l2']:.4f}")
            lines.append("")
            lines.append("Two equally bright heatmaps are not equally")
            lines.append("active -- brightness is normalised per layer,")
            lines.append("so compare l2, not the picture.")

        counts = self.app.parameter_counts.get(node.key)
        if counts:
            lines.append("")
            lines.append(f"TRAINABLE PARAMETERS  {counts:,}")

        # The conv taps live inside the frozen encoder, upstream of occipital.
        if node.key == "occipital":
            taps = [n for n in frame.heatmaps if n.startswith("vision.")]
            if taps:
                lines.append("")
                lines.append("FROZEN ENCODER CONV TAPS (upstream)")
                lines.append("  spatial maps, not layer columns -- a")
                lines.append("  conv map genuinely has a layout")
                for tap in taps:
                    grid = frame.heatmaps[tap]
                    lines.append(f"  {tap:<16} {grid.shape[0]:>3} x {grid.shape[1]:<3}")
        return "\n".join(lines)

    def _set_detail_text(self, text: str) -> None:
        self.detail_text.configure(state=tk.NORMAL)
        self.detail_text.delete("1.0", tk.END)
        self.detail_text.insert("1.0", text)
        self.detail_text.configure(state=tk.DISABLED)


# ---------------------------------------------------------------------------
# Minimised view (section 7.1)
# ---------------------------------------------------------------------------


class Dashboard(tk.Tk):
    def __init__(self) -> None:
        enable_dpi_awareness()      # must happen before the first window exists
        super().__init__()
        self.title("Brain-Inspired AI")
        self.configure(bg=config.UI_BG)
        self.geometry(f"{config.UI_MIN_WIDTH}x{config.UI_MIN_HEIGHT}")
        self.minsize(config.UI_MIN_WIDTH, config.UI_MIN_HEIGHT)

        self.worker = BrainWorker()
        self.telemetry = TelemetryMonitor()
        self.parameter_counts: dict[str, int] = {}
        self.topology: dict = {}
        self.expanded: ExpandedWindow | None = None
        self.expanded_visible = False
        self.active_tab: str | None = None
        self.module_state: dict[str, str] = {k: "idle" for k in config.LOBES}
        self._replying = False
        self.last_frame: ActivationFrame | None = None
        self.frames_seen = 0

        self._build_layout()

        self.worker.start()
        self.telemetry.start()
        self.protocol("WM_DELETE_WINDOW", self.close)
        self.after(config.UI_REFRESH_MS, self._tick)
        self.after(config.UI_TELEMETRY_MS, self._tick_telemetry)

    # ------------------------------------------------------------------
    # Layout
    # ------------------------------------------------------------------
    def _build_layout(self) -> None:
        # PACK ORDER MATTERS. Tk gives leftover space to children packed with
        # expand=True, and a child packed BEFORE its siblings claims its slice
        # first. So every fixed-size pane must be packed before the expanding
        # one, or it gets squeezed to zero width and silently vanishes:
        #   output strip (bottom, fixed)  ->  then the top row
        #   input pane (left, fixed) -> system pane (right, fixed) -> centre
        self._build_output_strip()

        top = tk.Frame(self, bg=config.UI_BG)
        top.pack(fill=tk.BOTH, expand=True, padx=10, pady=(10, 4))

        left_w = int(config.UI_MIN_WIDTH * config.UI_INPUT_PANE_FRACTION)
        self._build_input_pane(top, left_w)
        self._build_system_pane(top)
        self._build_centre(top)

    # -- left: input ----------------------------------------------------
    def _build_input_pane(self, parent, width: int) -> None:
        pane = tk.Frame(parent, bg=config.UI_BG, width=width)
        pane.pack(side=tk.LEFT, fill=tk.Y)
        pane.pack_propagate(False)

        tk.Label(
            pane, text="INPUT ─ SOURCE SELECT", bg=config.UI_BG, fg=config.UI_ACCENT,
            font=config.UI_FONT_TITLE, anchor="w",
        ).pack(fill=tk.X, pady=(0, 6))

        self.tab_buttons: dict[str, tk.Button] = {}
        for name in ("text", "image", "video", "audio"):
            button = tk.Button(
                pane, text=f"  {name.upper():<7}[ ]", 
                command=lambda n=name: self.open_tab(n),
                bg=config.UI_PANEL, fg=config.UI_MID,
                activebackground=config.UI_ACCENT_DIM, activeforeground=config.UI_FG,
                relief=tk.FLAT, bd=0, font=config.UI_FONT, anchor="w",
                padx=10, pady=6, highlightthickness=1,
                highlightbackground=config.UI_EDGE_IDLE, cursor="hand2",
            )
            button.pack(fill=tk.X, pady=2)
            self.tab_buttons[name] = button

        # The panel a tab opens. Hidden until a tab is clicked; closes on enter.
        self.panel = tk.Frame(pane, bg=config.UI_BG, highlightthickness=1,
                              highlightbackground=config.UI_DIM)
        self.panel_label = tk.Label(
            self.panel, text="", bg=config.UI_BG, fg=config.UI_MID,
            font=config.UI_FONT_SMALL, anchor="w", justify=tk.LEFT, wraplength=width - 24,
        )
        self.panel_label.pack(fill=tk.X, padx=8, pady=(8, 4))

        self.text_entry = tk.Entry(
            self.panel, bg="#0d0d0d", fg=config.UI_FG, insertbackground=config.UI_FG,
            relief=tk.FLAT, font=config.UI_FONT,
        )
        self.text_entry.bind("<Return>", lambda _e: self._submit_text())

        self.pick_button = tk.Button(
            self.panel, text="choose file...", command=self._pick_file,
            bg=config.UI_BG, fg=config.UI_FG, activebackground=config.UI_FG,
            activeforeground=config.UI_BG, relief=tk.SOLID, bd=1,
            font=config.UI_FONT_SMALL, cursor="hand2",
        )
        self.close_button = tk.Button(
            self.panel, text="cancel", command=self.close_tab,
            bg=config.UI_BG, fg=config.UI_MID, activebackground=config.UI_BG,
            activeforeground=config.UI_FG, relief=tk.FLAT,
            font=config.UI_FONT_SMALL, cursor="hand2",
        )

        # ---- STAGING AREA -------------------------------------------
        # Inputs accumulate here instead of firing immediately, so any
        # combination can be sent in ONE pass: image alone, image+text,
        # text+audio+sensor, whatever. That is what the frontal fusion slots
        # and the thalamic routing table exist for -- before this, the UI could
        # only ever exercise one modality at a time and the multi-stream path
        # was unreachable by hand.
        tk.Frame(pane, bg=config.UI_EDGE_IDLE, height=1).pack(fill=tk.X, pady=(10, 6))
        tk.Label(
            pane, text="STAGED ─ SENT TOGETHER", bg=config.UI_BG,
            fg=config.UI_ACCENT, font=config.UI_FONT_TITLE, anchor="w",
        ).pack(fill=tk.X)

        self.staged: dict[str, tuple] = {}          # modality -> (value, caption)
        self.stage_list = tk.Text(
            pane, bg=config.UI_PANEL, fg=config.UI_MID, font=config.UI_FONT_SMALL,
            height=7, relief=tk.FLAT, wrap=tk.WORD, padx=6, pady=5,
            highlightthickness=1, highlightbackground=config.UI_EDGE_IDLE,
        )
        self.stage_list.pack(fill=tk.X, pady=(4, 6))
        self.stage_list.configure(state=tk.DISABLED)

        row = tk.Frame(pane, bg=config.UI_BG)
        row.pack(fill=tk.X)
        self.run_button = tk.Button(
            row, text="▸ RUN", command=self.run_staged,
            bg=config.UI_PANEL, fg=config.UI_ACCENT,
            activebackground=config.UI_ACCENT_DIM, activeforeground=config.UI_FG,
            relief=tk.FLAT, bd=0, font=config.UI_FONT_TITLE, cursor="hand2",
            padx=12, pady=5, highlightthickness=1,
            highlightbackground=config.UI_EDGE_IDLE,
        )
        self.run_button.pack(side=tk.LEFT, fill=tk.X, expand=True)
        tk.Button(
            row, text="CLEAR", command=self.clear_staged,
            bg=config.UI_PANEL, fg=config.UI_DIM,
            activebackground=config.UI_PANEL, activeforeground=config.UI_FG,
            relief=tk.FLAT, bd=0, font=config.UI_FONT_SMALL, cursor="hand2",
            padx=10, pady=5, highlightthickness=1,
            highlightbackground=config.UI_EDGE_IDLE,
        ).pack(side=tk.RIGHT, padx=(6, 0))

        self._render_staged()

        tk.Frame(pane, bg=config.UI_BG).pack(fill=tk.BOTH, expand=True)
        self.status_label = tk.Label(
            pane, text="starting...", bg=config.UI_BG, fg=config.UI_MID,
            font=config.UI_FONT_SMALL, anchor="w", justify=tk.LEFT,
            wraplength=width - 8,
        )
        self.status_label.pack(fill=tk.X, side=tk.BOTTOM)

    # -- centre: lobes --------------------------------------------------
    def _build_centre(self, parent) -> None:
        centre = tk.Frame(parent, bg=config.UI_BG)
        centre.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10)

        header = tk.Frame(centre, bg=config.UI_BG)
        header.pack(fill=tk.X)
        tk.Label(
            header, text="CORTEX ─ LIVE", bg=config.UI_BG, fg=config.UI_ACCENT,
            font=config.UI_FONT_TITLE,
        ).pack(side=tk.LEFT)
        tk.Button(
            header, text="EXPAND ▸", command=lambda: self.open_expanded(None),
            bg=config.UI_PANEL, fg=config.UI_ACCENT,
            activebackground=config.UI_ACCENT_DIM, activeforeground=config.UI_FG,
            relief=tk.FLAT, bd=0, font=config.UI_FONT_SMALL, cursor="hand2",
            padx=10, pady=2, highlightthickness=1,
            highlightbackground=config.UI_EDGE_IDLE,
        ).pack(side=tk.RIGHT)

        self.graph = GraphCanvas(
            centre, MINI_NODES, MINI_EDGES, on_click=self.open_expanded
        )
        self.graph.pack(fill=tk.BOTH, expand=True, pady=(6, 0))

    # -- right: system --------------------------------------------------
    def _build_system_pane(self, parent) -> None:
        pane = tk.Frame(parent, bg=config.UI_BG, width=252)
        pane.pack(side=tk.RIGHT, fill=tk.Y)
        pane.pack_propagate(False)

        tk.Label(
            pane, text="SYSTEM ─ TELEMETRY", bg=config.UI_BG, fg=config.UI_ACCENT,
            font=config.UI_FONT_TITLE, anchor="w",
        ).pack(fill=tk.X, pady=(0, 6))

        self.system_text = tk.Text(
            pane, bg=config.UI_PANEL, fg=config.UI_MID, font=config.UI_FONT_SMALL,
            relief=tk.FLAT, highlightthickness=1, highlightbackground=config.UI_EDGE_IDLE,
            wrap=tk.NONE, padx=8, pady=8,
        )
        self.system_text.pack(fill=tk.BOTH, expand=True)
        self.system_text.configure(state=tk.DISABLED)

    # -- bottom: output -------------------------------------------------
    def _build_output_strip(self) -> None:
        strip = tk.Frame(self, bg=config.UI_BG, height=168)
        strip.pack(side=tk.BOTTOM, fill=tk.X, padx=10, pady=(4, 10))
        strip.pack_propagate(False)

        tk.Label(
            strip, text="OUTPUT ─ STDOUT", bg=config.UI_BG, fg=config.UI_ACCENT,
            font=config.UI_FONT_TITLE, anchor="w",
        ).pack(fill=tk.X)

        wrap = tk.Frame(strip, bg=config.UI_PANEL, highlightthickness=1,
                        highlightbackground=config.UI_EDGE_IDLE)
        wrap.pack(fill=tk.BOTH, expand=True, pady=(4, 0))

        self.output_text = tk.Text(
            wrap, bg=config.UI_PANEL, fg=config.UI_FG, font=config.UI_FONT,
            relief=tk.FLAT, highlightthickness=0, wrap=tk.WORD, padx=10, pady=8,
        )
        scroll = tk.Scrollbar(wrap, command=self.output_text.yview,
                              bg=config.UI_BG, troughcolor=config.UI_BG)
        self.output_text.configure(yscrollcommand=scroll.set)
        scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.output_text.pack(fill=tk.BOTH, expand=True)

        self.output_text.tag_configure("you", foreground=config.UI_COOL)
        self.output_text.tag_configure("brain", foreground=config.UI_ACCENT)
        self.output_text.tag_configure("meta", foreground=config.UI_DIM)
        self.output_text.configure(state=tk.DISABLED)

        self.say("meta", "four cortical modules perceive; a language module "
                         "puts their conclusion into words. stage any mix of "
                         "sources on the left, then RUN.")

    # ------------------------------------------------------------------
    # Input tabs
    # ------------------------------------------------------------------
    def open_tab(self, name: str) -> None:
        if self.active_tab == name:
            self.close_tab()
            return
        self.active_tab = name
        for key, button in self.tab_buttons.items():
            live = key == name
            button.configure(
                bg=config.UI_ACCENT_DIM if live else config.UI_PANEL,
                fg=config.UI_FG if live else config.UI_MID,
                highlightbackground=config.UI_ACCENT if live else config.UI_EDGE_IDLE,
                text=f"  {key.upper():<7}[{'*' if live else ' '}]",
            )

        self.text_entry.pack_forget()
        self.pick_button.pack_forget()
        self.close_button.pack_forget()

        if name == "text":
            self.panel_label.configure(text="type, then press Enter")
            self.text_entry.pack(fill=tk.X, padx=8, pady=(0, 6))
            self.text_entry.focus_set()
        else:
            self.panel_label.configure(text=f"choose a {name} file")
            self.pick_button.pack(fill=tk.X, padx=8, pady=(0, 6))

        self.close_button.pack(fill=tk.X, padx=8, pady=(0, 6))
        self.panel.pack(fill=tk.X, pady=(8, 0))

    def close_tab(self) -> None:
        self.active_tab = None
        self.panel.pack_forget()
        for key, button in self.tab_buttons.items():
            button.configure(bg=config.UI_PANEL, fg=config.UI_MID,
                             highlightbackground=config.UI_EDGE_IDLE,
                             text=f"  {key.upper():<7}[ ]")

    def _submit_text(self) -> None:
        value = self.text_entry.get().strip()
        if not value:
            return
        self.text_entry.delete(0, tk.END)
        self.close_tab()                    # "on enter, the panel closes"
        self.stage("text", value, value)
        # Typing and pressing Enter is the chat gesture -- send it straight
        # away, together with anything else already staged.
        self.run_staged()

    def _pick_file(self) -> None:
        kind = self.active_tab or "image"
        filters = {
            "image": [("Images", "*.jpg *.jpeg *.png *.bmp *.gif *.webp *.tif *.tiff")],
            "video": [("Video", "*.mp4 *.avi *.mov *.mkv *.webm *.m4v")],
            "audio": [("Audio", "*.wav *.mp3 *.m4a *.flac *.ogg")],
        }[kind]
        path = filedialog.askopenfilename(
            title=f"choose {kind}", filetypes=filters + [("All files", "*.*")]
        )
        if not path:
            return
        self.close_tab()
        self._stage_file(kind, path)

    def _stage_file(self, kind: str, path: str) -> None:
        """Decompose a file and stage whatever streams it yields.

        A video legitimately produces more than one stream, so staging (rather
        than dispatching) is what lets it sit alongside a typed instruction or
        a second file before anything runs.
        """
        import os

        try:
            from modules.decompose import decompose

            result = decompose(path)
        except Exception as exc:  # noqa: BLE001
            self.say("meta", f"could not read that file -- {type(exc).__name__}: {exc}")
            return

        for note in result.notes:
            self.say("meta", note)
        for modality, value in result.inputs.items():
            self.stage(modality, value, os.path.basename(path))

    def _decompose_and_dispatch(self, path: str) -> None:
        """Decomposition sits BEFORE the thalamus (PROJECT.md 8)."""
        try:
            from modules.decompose import decompose

            result = decompose(path)
        except Exception as exc:  # noqa: BLE001
            self.say("meta", f"could not read that file -- {type(exc).__name__}: {exc}")
            return

        for note in result.notes:
            self.say("meta", note)
        if len(result.inputs) > 1:
            self.say(
                "meta",
                f"decomposed into {len(result.inputs)} streams "
                f"({', '.join(result.inputs)}) -- frontal will fuse them",
            )
        self.dispatch(result.inputs, label=f"{result.kind}: {result.source}")

    def dispatch(self, inputs: dict, label: str) -> None:
        if not self.worker.ready.is_set():
            self.say("meta", "still loading the backbones -- try again in a moment")
            return
        self.dispatch_raw(inputs, label)

    def dispatch_raw(self, inputs: dict, label: str) -> None:
        self.worker.submit(inputs, label)

    # ------------------------------------------------------------------
    # Staging: build any combination of modalities, then run once
    # ------------------------------------------------------------------
    def stage(self, modality: str, value, caption: str) -> None:
        """Add or replace one modality in the pending input."""
        self.staged[modality] = (value, caption)
        self._render_staged()
        self.say("meta", f"staged {modality}: {caption}")

    def clear_staged(self) -> None:
        self.staged.clear()
        self._render_staged()

    def _render_staged(self) -> None:
        self.stage_list.configure(state=tk.NORMAL)
        self.stage_list.delete("1.0", tk.END)
        if not self.staged:
            self.stage_list.insert(
                tk.END,
                "nothing staged\n\n"
                "pick any sources above, then RUN.\n"
                "they are sent together as one pass.",
            )
        else:
            for modality, (_, caption) in self.staged.items():
                route = config.THALAMIC_ROUTES.get(modality)
                where = f" -> {route[0]} -> {route[1]}" if route else ""
                self.stage_list.insert(tk.END, f"{modality.upper()}{where}\n")
                self.stage_list.insert(tk.END, f"   {caption[:46]}\n")
        self.stage_list.configure(state=tk.DISABLED)

        n = len(self.staged)
        self.run_button.configure(
            text=f"▸ RUN ({n})" if n else "▸ RUN",
            fg=config.UI_ACCENT if n else config.UI_DIM,
            highlightbackground=config.UI_ACCENT if n else config.UI_EDGE_IDLE,
        )

    def run_staged(self) -> None:
        if not self.staged:
            self.say("meta", "nothing staged -- pick a source first")
            return
        inputs = {m: v for m, (v, _) in self.staged.items()}
        caption = " + ".join(
            f"{m}({c[:22]})" for m, (_, c) in self.staged.items()
        )
        self.say("you", caption)
        if len(inputs) > 1:
            lobes = sorted({config.THALAMIC_ROUTES[m][1]
                            for m in inputs if m in config.THALAMIC_ROUTES})
            self.say("meta", f"{len(inputs)} streams -> {', '.join(lobes)} "
                             f"-> frontal fuses them")
        self.dispatch(inputs, label=caption)

    # ------------------------------------------------------------------
    # Output strip
    # ------------------------------------------------------------------
    def say(self, who: str, text: str) -> None:
        prefix = {"you": " >> ", "brain": " << ", "meta": "  · "}[who]
        self.output_text.configure(state=tk.NORMAL)
        for line in text.splitlines() or [""]:
            self.output_text.insert(tk.END, f"{prefix}{line}\n", who)
        self.output_text.see(tk.END)
        self.output_text.configure(state=tk.DISABLED)

    # ------------------------------------------------------------------
    # Streaming replies
    # ------------------------------------------------------------------
    def _open_reply(self) -> None:
        """Start a reply line that tokens will be appended to as they arrive."""
        self._replying = True
        self.output_text.configure(state=tk.NORMAL)
        self.output_text.insert(tk.END, " << ", "brain")
        self.output_text.configure(state=tk.DISABLED)
        self.output_text.see(tk.END)

    def _append_token(self, piece: str) -> None:
        if not getattr(self, "_replying", False):
            self._open_reply()
        self.output_text.configure(state=tk.NORMAL)
        # Keep the transcript readable: continuation lines line up under the
        # first, so a wrapped reply does not collide with the "<<" marker.
        self.output_text.insert(tk.END, piece.replace(chr(10), chr(10) + "    "),
                                "brain")
        self.output_text.configure(state=tk.DISABLED)
        self.output_text.see(tk.END)

    def _close_reply(self) -> None:
        if not getattr(self, "_replying", False):
            return
        self._replying = False
        self.output_text.configure(state=tk.NORMAL)
        self.output_text.insert(tk.END, chr(10))
        self.output_text.configure(state=tk.DISABLED)
        self.output_text.see(tk.END)

    # ------------------------------------------------------------------
    # Expanded view
    # ------------------------------------------------------------------
    def open_expanded(self, focus: str | None) -> None:
        if self.expanded is None:
            self.expanded = ExpandedWindow(self, self)
            self.expanded.graph.topology = self.topology
            if self.last_frame is not None:
                self.expanded.apply_frame(self.last_frame)
        self.expanded.show(focus)
        self.expanded.refresh()

    # ------------------------------------------------------------------
    # Timers -- the UI drains the queue on its OWN clock (section 6.3)
    # ------------------------------------------------------------------
    def _tick(self) -> None:
        try:
            self._drain_status()
            self._drain_frames()
            self.graph.redraw()
            if self.expanded is not None and self.expanded_visible:
                self.expanded.refresh()
        finally:
            self.after(config.UI_REFRESH_MS, self._tick)

    def _drain_status(self) -> None:
        while True:
            try:
                status = self.worker.status.get_nowait()
            except queue.Empty:
                break

            if status.kind == "loading":
                self.status_label.configure(text=status.text)
            elif status.kind == "ready":
                self.status_label.configure(text=status.text)
                self.say("meta", status.text)
                if self.worker.brain is not None:
                    self.parameter_counts = self._map_counts(
                        self.worker.brain.parameter_count()
                    )
                    # The layer chains are read off the live model, so the
                    # diagrams cannot drift from what actually ran.
                    self.topology = self.worker.brain.topology
                    self.graph.topology = self.topology
                    if self.expanded is not None:
                        self.expanded.graph.topology = self.topology
                if status.detail == "UNTRAINED":
                    self.say(
                        "meta",
                        "the lobes are untrained, so their answer is noise. "
                        "run  python train.py  to fix that. the frozen encoder's "
                        "own reading is shown alongside for contrast.",
                    )
            elif status.kind == "percept":
                # What the LOBES concluded, before any words are put to it.
                self.say("meta", status.text)
            elif status.kind == "speaking":
                self._open_reply()
                self.status_label.configure(text="speaking...")
            elif status.kind == "token":
                self._append_token(status.text)
            elif status.kind == "result":
                self._close_reply()
                self.status_label.configure(text=f"{status.elapsed_ms:.0f} ms")
            elif status.kind == "error":
                self.say("meta", f"ERROR -- {status.text}")
                self.status_label.configure(text="error")

    @staticmethod
    def _map_counts(counts: dict[str, int]) -> dict[str, int]:
        mapped = dict(counts)
        mapped["basal_ganglia"] = counts.get("basal_ganglia", 0)
        mapped["output"] = counts.get("output", 0)
        return mapped

    def _drain_frames(self) -> None:
        if self.worker.brain is None:
            return
        frames = self.worker.brain.bus.drain(limit=8)
        if not frames:
            return
        self.frames_seen += len(frames)
        frame = frames[-1]              # only the newest is worth drawing
        self.last_frame = frame
        self.graph.apply_frame(frame)
        for lobe in config.LOBES:
            self.module_state[lobe] = (
                "active" if lobe in frame.active_nodes else "idle"
            )
        if self.expanded is not None:
            self.expanded.apply_frame(frame)

    def _tick_telemetry(self) -> None:
        try:
            self._render_system(self.telemetry.latest())
        finally:
            self.after(config.UI_TELEMETRY_MS, self._tick_telemetry)

    def _render_system(self, sample: SystemSample) -> None:
        brain = self.worker.brain
        lines: list[str] = []

        lines.append("NETWORKS")
        for lobe in config.LOBES:
            state = self.module_state.get(lobe, "idle")
            marker = "#" if state == "active" else "."
            lines.append(f"  {marker} {lobe:<10} {state}")
        lines.append("")

        lines.append("MODULES")
        loaded = brain is not None
        health = "ERROR" if self.worker.last_error else ("ok" if loaded else "loading")
        for name in ("thalamus", "basal ganglia", "output gen"):
            lines.append(f"  {name:<14} {health}")
        lines.append(f"  {'encoders':<14} {'ok' if loaded else 'loading'}")
        if self.worker.last_error:
            lines.append(f"  last: {self.worker.last_error[:26]}")
        lines.append("")

        lines.append("LOAD")
        if not sample.valid:
            lines.append("  sampling...")
        else:
            lines.append(f"  CPU     {sample.cpu_percent:5.1f}%")
            lines.append(
                f"  RAM     {sample.ram_percent:5.1f}%  "
                f"{sample.ram_used_gb:.1f}/{sample.ram_total_gb:.1f}G"
            )
            lines.append(f"  {sample.gpu_line(compact=True)}")
            if sample.gpu_available:
                lines.append(f"  {sample.gpu_name[:24]}")
                lines.append(f"  via {sample.gpu_source}")
            lines += _wrap(sample.gpu_note(), 26, indent="  ")
            lines.append(f"  threads {sample.threads}")
        lines.append("")

        lines.append("PIPELINE")
        lines.append(f"  device      {brain.device_name if brain else '-'}")
        lines.append(f"  lobes       {'trained' if brain and brain.trained else 'untrained'}")
        lines.append(f"  latent dim  {config.LATENT_DIM}")
        lines.append(f"  frames      {self.frames_seen}")
        if brain is not None:
            lines.append(f"  queued      {len(brain.bus)}")
            lines.append(f"  dropped     {brain.bus.dropped}")

        self.system_text.configure(state=tk.NORMAL)
        self.system_text.delete("1.0", tk.END)
        self.system_text.insert("1.0", "\n".join(lines))
        self.system_text.configure(state=tk.DISABLED)

    # ------------------------------------------------------------------
    def close(self) -> None:
        self.worker.stop()
        self.telemetry.stop()
        self.destroy()


# ---------------------------------------------------------------------------


def selftest() -> int:
    """Build the whole UI, pump the event loop, exercise both views, exit.

    Proves the dashboard constructs and ticks without a human clicking it.
    """
    print("desktop selftest: building UI...")
    app = Dashboard()
    app.update()
    print(f"  minimised view    {len(app.graph.nodes)} nodes, "
          f"{len(app.graph.edges)} edges")

    print("  waiting for the worker to load the backbones...")
    deadline = time.time() + 180
    while not app.worker.ready.is_set() and time.time() < deadline:
        app.update()
        time.sleep(0.05)
    if not app.worker.ready.is_set():
        print(f"  FAIL: worker never became ready ({app.worker.last_error})")
        app.close()
        return 1
    print("  worker ready")

    app.open_expanded(None)
    app.update()
    print(f"  expanded view     {len(app.expanded.graph.nodes)} nodes, "
          f"{len(app.expanded.graph.edges)} edges")

    print("  submitting a text input through the real path...")
    app.dispatch_raw({"text": "a small striped cat on a table"}, label="selftest")

    deadline = time.time() + 60
    while app.frames_seen == 0 and time.time() < deadline:
        app.update()
        time.sleep(0.05)

    if app.frames_seen == 0:
        print("  FAIL: no activation frame reached the UI")
        app.close()
        return 1
    print(f"  frames rendered   {app.frames_seen}")

    for key in app.expanded.graph.nodes:
        app.expanded.select(key)
        app.update()
    print(f"  detail pane       exercised all {len(app.expanded.graph.nodes)} nodes")

    for _ in range(12):
        app.update()
        time.sleep(0.02)

    app.close()
    print("desktop selftest: PASS")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selftest", action="store_true",
                        help="build, tick and exit without interaction")
    args = parser.parse_args()

    if args.selftest:
        return selftest()

    app = Dashboard()
    app.mainloop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
