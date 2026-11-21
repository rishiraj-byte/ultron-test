"""
dashboard.py -- the v2.B console: one window to use it, watch it, and look
inside it down to single weights.

    python -m v2b.dashboard                    open the window
    python -m v2b.dashboard --selftest         the engine, printed, no window
    python -m v2b.dashboard --snapshot x.png   open, run, screenshot every tab

    +-------------+--------------------------------------------+-------------+
    |  INPUT      | ROUTING NETWORKS ACTIVITY WEIGHTS MEMORY   |  SYSTEM     |
    |  FOCUS      |            RAG OVERLAY GUIDE               |  ANSWER     |
    |  (regions)  |                                            |  cortex /   |
    |  lesions    |          the chosen view                   |  memory /   |
    |             |                                            |  final      |
    +-------------+--------------------------------------------+-------------+
    |  OUTPUT -- what motor cortex said; GOOD / BAD / "no, it's a ..."       |
    +------------------------------------------------------------------------+

THE VIEWS
  ROUTING   the graph, each arrow lit by what it carried on this pass
  NETWORKS  every region as columns of circles (the v2 layer diagrams)
  ACTIVITY  the focused region's real activations: per-block maps, attention
            maps, and which neurons fired in every block
  WEIGHTS   every tensor in the focused region, as numbers: heatmap,
            histogram, statistics -- and weights x live input for the last
            percept. Zero a tensor and perceive again to see what it did.
  MEMORY    what the hippocampus recalled, the pictures behind it, how the
            vote moved the answer, and what the amygdala made of it
  RAG       the hippocampus as a network, every state and gate from POT to
            motor (ragview.py), above the whole store drawn as one layer of
            ~219,000 units lit by this percept; hover any unit to see that
            memory. Double-click the hippocampus, amygdala or database to open it.
  OVERLAY   where POT looked on the image or in the sound, pass by pass
  GUIDE     what this region is in a brain, what v2.B uses, where they differ

THREADING. The brain runs on a worker thread and posts finished thoughts to a
queue; the UI thread does all drawing on its own timer. A lock keeps a weight
edit from landing in the middle of a percept.
"""

from __future__ import annotations

import argparse
import queue
import sys
import threading
import time
from pathlib import Path

import numpy as np

from v2b import config_v2b as C
from v2b import guide, wiring

#  Where each node sits in the routing view (fractions of the canvas) and its
#  kind. The three relays are drawn inside one THALAMUS box; the hippocampus
#  has its database (the RAG store) drawn as its own box beneath it.
NODES = {
    "input": (0.50, 0.05, "input"),
    "pulvinar": (0.17, 0.18, "trained"), "LGN": (0.50, 0.18, "trained"),
    "MGN": (0.83, 0.18, "trained"),
    "parietal": (0.17, 0.31, "frozen"), "occipital": (0.50, 0.31, "frozen"),
    "temporal": (0.83, 0.31, "frozen"),
    "dorsal": (0.23, 0.44, "trained"), "ventral": (0.50, 0.44, "frozen"),
    "STS": (0.77, 0.44, "trained"),
    "POT": (0.50, 0.57, "trained"),
    "hippocampus": (0.17, 0.70, "memory"), "frontal": (0.50, 0.70, "frozen"),
    "amygdala": (0.83, 0.70, "memory"),
    "store": (0.17, 0.86, "memory"), "motor": (0.50, 0.86, "input"),
}
DISPLAY = {"ventral": "Ventral stream", "dorsal": "Dorsal stream", "store": "RAG database"}
FOCUS = ("overview", "pulvinar", "LGN", "MGN", "thalamus", "occipital", "ventral",
         "parietal", "temporal", "dorsal", "STS", "POT", "frontal", "hippocampus",
         "amygdala")
TILES = ("pulvinar", "LGN", "MGN", "thalamus", "occipital", "ventral", "parietal",
         "temporal", "dorsal", "STS", "POT", "frontal", "hippocampus", "amygdala")
PER_PASS = ("dorsal", "STS", "POT", "frontal", "thalamus")
LIVE_LOBES = ("occipital", "ventral", "parietal", "temporal")
TABS = ("routing", "networks", "activity", "weights", "memory", "rag", "overlay", "guide")
KIND_COLOUR = {"frozen": C.UI_COOL, "trained": C.UI_ACCENT, "memory": C.UI_GOOD,
               "input": C.UI_MID}


# ---------------------------------------------------------------------------
# Engine -- no UI in it
# ---------------------------------------------------------------------------


#  --candidate: the model a training run wrote beside the deployed one, so the
#  two can be tried on the same pictures before one is promoted.
WHICH = {"checkpoint": C.CHECKPOINT, "bank": C.MEMORY_BANK}


class Engine:
    def __init__(self):
        from v2b.brain import BrainV2B
        self.brain = BrainV2B(checkpoint=WHICH["checkpoint"], bank=WHICH["bank"])
        self.lock = threading.Lock()
        self.watch = None            # (label, module) whose input/output to capture
        self.saved: dict = {}        # tensor name -> original values, while edited

    def roles(self) -> dict:
        b = self.brain
        vision = "SigLIP 2" if b.vision.family == "siglip" else "CLIP"
        return {"input": "image/sound/video", "pulvinar": "thalamus",
                "LGN": "thalamus", "MGN": "thalamus",
                "parietal": "DepthAnything V2-S",
                "occipital": "%s blk 1-%d" % (vision, b.vision.split),
                "temporal": "auditory cx, %s" % {"clap": "CLAP", "ast": "AST"}.get(
                    b.temporal.family, "Whisper"),
                "dorsal": "where: V4 + depth",
                "ventral": "V4 -> IT: what (blk %d-%d)" % (b.vision.split + 1, len(b.vision.blocks)),
                "STS": "sight + sound", "POT": "fuses streams",
                "hippocampus": "RAG memory", "frontal": "%s text" % vision,
                "amygdala": "learned value", "motor": "answer -> you"}

    def think(self, image=None, audio=None, video=None, source=""):
        with self.lock:
            captured: dict = {}
            handle = None
            if self.watch is not None:
                def hook(module, inputs, output):
                    x = inputs[0] if inputs else None
                    if hasattr(x, "detach"):
                        captured["input"] = x.detach().float()[0].reshape(-1, x.shape[-1]).abs().mean(0).cpu().numpy()
                    if hasattr(output, "detach"):
                        captured["output"] = output.detach().float()[0].reshape(-1, output.shape[-1]).mean(0).cpu().numpy()
                handle = self.watch[1].register_forward_hook(hook)
            try:
                thought = self.brain.perceive(image=image, audio=audio, video=video,
                                              source=source)
            finally:
                if handle is not None:
                    handle.remove()
            thought.watched = (self.watch[0], captured) if self.watch else None
            return thought

    # -- weights ---------------------------------------------------------------
    def tensors(self, region: str) -> list:
        """[(name, tensor, owning module, trainable)] for every weight in a region."""
        out = []
        for prefix, module in self.brain.networks().get(region, []):
            for name, param in module.named_parameters():
                owner = module.get_submodule(name.rsplit(".", 1)[0]) if "." in name else module
                out.append(("%s.%s" % (prefix, name), param, owner, param.requires_grad))
        return out

    def zero(self, name: str, tensor) -> None:
        import torch
        with self.lock, torch.no_grad():
            if name not in self.saved:
                self.saved[name] = (tensor, tensor.detach().clone())
            tensor.zero_()

    def restore(self, name: str | None = None) -> None:
        import torch
        with self.lock, torch.no_grad():
            for key in ([name] if name else list(self.saved)):
                if key in self.saved:
                    tensor, original = self.saved.pop(key)
                    tensor.copy_(original)


def telemetry() -> dict:
    out = {"cpu": 0.0, "ram": 0.0, "gpu": "-"}
    try:
        import psutil
        out["cpu"] = psutil.cpu_percent()
        out["ram"] = psutil.virtual_memory().percent
    except Exception:                                         # noqa: BLE001
        pass
    try:
        import torch
        if torch.cuda.is_available():
            free, total = torch.cuda.mem_get_info()
            out["gpu"] = "%.1f/%.1f GB" % ((total - free) / 1e9, total / 1e9)
    except Exception:                                         # noqa: BLE001
        pass
    return out


def tensor_view(tensor, max_elements: int = 4_000_000):
    """A 2-D float32 numpy view of a weight, sampled if it is huge.

    SigLIP's text vocabulary alone is 256,000 x 768 = 197M numbers; copying
    that to the CPU on every click would take seconds and ~800 MB, so large
    tensors are row-sampled and the statistics say so.
    """
    import torch
    t = tensor.detach()
    if t.dim() == 0:
        t = t.reshape(1, 1)
    elif t.dim() == 1:
        t = t[None]
    elif t.dim() > 2:
        t = t.reshape(t.shape[0], -1)
    sampled = t.numel() > max_elements
    if sampled:
        step = int(np.ceil(t.numel() / max_elements))
        t = t[::step]
    with torch.no_grad():
        return t.float().cpu().numpy(), sampled


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------


def selftest() -> int:
    from PIL import Image
    from v2b import data
    from v2b.drawing import heatmap, histogram, raster, render

    print("=" * 72)
    print("v2.B CONSOLE SELF-TEST -- engine and every view's data, no window")
    print("=" * 72)
    engine = Engine()
    brain = engine.brain
    print("networks: %s" % "  ".join("%s=%s" % kv for kv in brain.models().items()))
    print("trained : %s%s" % (brain.trained, ("  [%s]" % brain.mismatch) if brain.mismatch else ""))
    print("memory  : %d consolidated, %d yours" % brain.hippocampus.size)
    images, labels, _ = data.cifar("test", per_class=1)
    names = data.cifar_names()
    for index in (3, 42, 77):
        thought = engine.think(image=Image.fromarray(images[index]), source="cifar test")
        print("\n  a %-12s -> %s" % (names[labels[index]], thought.summary()))
    waves, tone_labels, _ = data.audio("test")
    thought_s = engine.think(audio=waves[0], source="sound")
    print("\n  sound of %-8s -> %s" % (brain.frontal.names[data.sound_offset() + tone_labels[0]],
                                        thought_s.summary()))
    problems = []
    for node in TILES:
        source = thought.pass_layers[-1] if node in PER_PASS else thought.layers
        cols = source.get(node, [])
        if not cols:
            problems.append("no columns for %s" % node)
        render(cols, 300, 150)
    for lobe in LIVE_LOBES:
        live = (thought_s if lobe == "temporal" else thought).live.get(lobe)
        if live is None or not live.blocks:
            problems.append("no live activity for %s" % lobe)
            continue
        raster(live.neurons, 300, 60)
        heatmap(live.blocks[0], 60, 60, signed=False)
    for region in FOCUS[1:]:
        tensors = engine.tensors(region)
        if region in ("hippocampus", "amygdala") or tensors:
            for name, t, _, _ in tensors[:1]:
                array, _ = tensor_view(t)
                heatmap(array, 200, 120)
                histogram(array, 200, 40)
        print("  %-12s %4d weight tensors, %12s numbers"
              % (region, len(tensors), f"{sum(t.numel() for _, t, _, _ in tensors):,}"))
    missing = [n for n in wiring.NAMES if n not in thought.carried[-1]]
    if missing:
        problems.append("connections with no value: %s" % missing)
    print("\n%s" % ("every view has data" if not problems else "PROBLEMS: %s" % problems))
    return 0 if not problems else 1


# ---------------------------------------------------------------------------
# The window
# ---------------------------------------------------------------------------


class Console:
    def __init__(self, root, snapshot: str | None = None):
        import tkinter as tk

        self.tk, self.root = tk, root
        self.queue: queue.Queue = queue.Queue()
        self.engine: Engine | None = None
        self.thought = None
        self.frame = C.FRONTAL_PASSES - 1
        self.playing = False
        self._since = 0
        self.tab = "routing"
        self.focus = "ventral"
        self.status = "loading five pretrained networks..."
        self.paths = {"image": None, "audio": None, "video": None}
        self.weight_list: list = []
        self.weight_index = 0
        self.weight_mode = "raw"
        self._photos: dict = {}
        self._boxes: dict = {}
        self._peaks: dict = {}
        self._tele = telemetry()
        self._tele_age = 0
        self._roles: dict = {}
        self.snapshot = snapshot
        self._dirty = True
        self._rag_since = 0.0            # when the shown thought arrived (RAG reveal)
        self._rag_hover = -1             # the stored memory under the mouse
        self._rag_mouse = (0, 0)
        self._rag_origin = None          # where the store raster sits on the canvas
        self._rag_layout = None
        self._rag_animating = False
        self._thumbs: dict = {}

        root.title("brain-ai  //  v2.B%s" % ("  //  CANDIDATE (%s)" % WHICH["checkpoint"].name
                                              if WHICH["checkpoint"] != C.CHECKPOINT else ""))
        root.configure(bg=C.UI_BG)
        #  Fit the screen: at 820 px tall the feedback row sat behind the
        #  Windows taskbar on a 768 px display.
        width = min(C.UI_MIN_WIDTH, root.winfo_screenwidth())
        height = min(C.UI_MIN_HEIGHT, root.winfo_screenheight() - 80)
        root.minsize(min(1200, width), min(640, height))
        root.geometry("%dx%d+0+0" % (width, height))
        try:
            root.state("zoomed")
        except Exception:                                     # noqa: BLE001
            pass

        bottom = tk.Frame(root, bg=C.UI_BG, height=170)
        bottom.pack(side="bottom", fill="x")
        bottom.pack_propagate(False)
        self._build_output(bottom)
        upper = tk.Frame(root, bg=C.UI_BG)
        upper.pack(side="top", fill="both", expand=True)
        left = tk.Frame(upper, bg=C.UI_BG, width=250)
        left.pack(side="left", fill="y")
        left.pack_propagate(False)
        self._build_input(left)
        self.side = tk.Canvas(upper, bg=C.UI_BG, highlightthickness=0, width=300)
        self.side.pack(side="right", fill="y")
        centre = tk.Frame(upper, bg=C.UI_BG)
        centre.pack(side="left", fill="both", expand=True)
        self._build_tabs(centre)
        self.weights_panel = tk.Frame(centre, bg=C.UI_BG, width=260)
        self._build_weights_panel(self.weights_panel)
        self.canvas = tk.Canvas(centre, bg=C.UI_BG, highlightthickness=0)
        self.canvas.pack(side="left", fill="both", expand=True)
        self.canvas.bind("<Button-1>", self._click)
        self.canvas.bind("<Double-Button-1>", self._open)
        self.canvas.bind("<Motion>", self._hover)
        self.canvas.bind("<Leave>", self._unhover)

        threading.Thread(target=self._boot, daemon=True).start()
        root.after(C.UI_REFRESH_MS, self._tick)

    # -- widgets ---------------------------------------------------------------
    def _button(self, parent, text, command, colour=None, side=None, width=None):
        tk = self.tk
        button = tk.Button(parent, text=text, command=command, bg=C.UI_PANEL,
                           fg=colour or C.UI_FG, activebackground=C.UI_ACCENT_DIM,
                           activeforeground=C.UI_FG, font=C.UI_FONT_SMALL, relief="flat",
                           highlightthickness=1, highlightbackground=C.UI_EDGE_IDLE,
                           width=width)
        if side:
            button.pack(side=side, padx=2, pady=2)
        else:
            button.pack(padx=12, pady=1, fill="x")
        return button

    def _title(self, parent, text):
        self.tk.Label(parent, text=text, bg=C.UI_BG, fg=C.UI_ACCENT,
                      font=C.UI_FONT_TITLE).pack(anchor="w", padx=12, pady=(10, 2))

    def _build_input(self, parent) -> None:
        tk = self.tk
        self._title(parent, "INPUT")
        for text, kind in (("IMAGE...", "image"), ("AUDIO...", "audio"), ("VIDEO...", "video")):
            self._button(parent, text, lambda k=kind: self._choose(k))
        self._button(parent, "clear", self._clear)
        self.media = tk.Label(parent, text="nothing loaded", bg=C.UI_BG, fg=C.UI_DIM,
                              font=C.UI_FONT_TINY, anchor="w", justify="left", wraplength=220)
        self.media.pack(anchor="w", padx=12, pady=(2, 4), fill="x")
        self._button(parent, "PERCEIVE", self._perceive, C.UI_ACCENT)
        self._button(parent, "replay the 3 passes", self._replay)
        self._title(parent, "FOCUS")
        self.focus_list = tk.Listbox(parent, bg=C.UI_PANEL, fg=C.UI_MID, font=C.UI_FONT_SMALL,
                                     selectbackground=C.UI_ACCENT_DIM, selectforeground=C.UI_FG,
                                     relief="flat", highlightthickness=1,
                                     highlightbackground=C.UI_EDGE_IDLE, height=12,
                                     exportselection=False, activestyle="none")
        for region in FOCUS:
            self.focus_list.insert("end", region)
        self.focus_list.selection_set(FOCUS.index(self.focus))
        self.focus_list.bind("<<ListboxSelect>>", self._pick_focus)
        self.focus_list.pack(padx=12, fill="x")
        self._button(parent, "lesions...", self._show_lesions, C.UI_MID)

    def _build_tabs(self, parent) -> None:
        tk = self.tk
        bar = tk.Frame(parent, bg=C.UI_BG)
        bar.pack(side="top", fill="x", pady=(6, 0))
        self.tab_buttons = {}
        for tab in TABS:
            self.tab_buttons[tab] = self._button(bar, tab.upper(), lambda t=tab: self._set_tab(t),
                                                 side="left")
        self._style_tabs()

    def _style_tabs(self) -> None:
        for tab, button in self.tab_buttons.items():
            button.configure(fg=C.UI_ACCENT if tab == self.tab else C.UI_MID,
                             bg=C.UI_ACCENT_DIM if tab == self.tab else C.UI_PANEL)

    def _build_weights_panel(self, parent) -> None:
        tk = self.tk
        tk.Label(parent, text="weight tensors in focus", bg=C.UI_BG, fg=C.UI_COOL,
                 font=C.UI_FONT_SMALL).pack(anchor="w", padx=4, pady=(4, 0))
        self.weight_box = tk.Listbox(parent, bg=C.UI_PANEL, fg=C.UI_MID, font=C.UI_FONT_TINY,
                                     selectbackground=C.UI_ACCENT_DIM, selectforeground=C.UI_FG,
                                     relief="flat", highlightthickness=1,
                                     highlightbackground=C.UI_EDGE_IDLE, exportselection=False,
                                     activestyle="none", width=38)
        self.weight_box.pack(fill="both", expand=True, padx=4, pady=2)
        self.weight_box.bind("<<ListboxSelect>>", self._pick_weight)
        row = tk.Frame(parent, bg=C.UI_BG)
        row.pack(fill="x", padx=2)
        self._button(row, "raw", lambda: self._weight_mode("raw"), side="left")
        self._button(row, "x live input", lambda: self._weight_mode("live"), side="left")
        row = tk.Frame(parent, bg=C.UI_BG)
        row.pack(fill="x", padx=2, pady=(0, 4))
        self._button(row, "zero it", self._zero_weight, C.UI_ALERT, side="left")
        self._button(row, "restore", self._restore_weight, C.UI_GOOD, side="left")
        self._button(row, "restore all", lambda: self._restore_weight(True), C.UI_GOOD, side="left")

    def _build_output(self, parent) -> None:
        tk = self.tk
        tk.Label(parent, text="OUTPUT", bg=C.UI_BG, fg=C.UI_ACCENT,
                 font=C.UI_FONT_TITLE).pack(anchor="w", padx=14, pady=(6, 0))
        #  Feedback row first, against the bottom edge, so a growing
        #  transcript can never push it out of the window.
        row = tk.Frame(parent, bg=C.UI_BG)
        row.pack(side="bottom", fill="x", padx=14, pady=(0, 8))
        self.transcript = tk.Text(parent, bg=C.UI_PANEL, fg=C.UI_FG, font=C.UI_FONT_SMALL,
                                  relief="flat", wrap="word", height=4, highlightthickness=1,
                                  highlightbackground=C.UI_EDGE_IDLE)
        self.transcript.pack(padx=14, pady=4, fill="both", expand=True)
        for tag, colour in (("you", C.UI_COOL), ("brain", C.UI_ACCENT), ("note", C.UI_DIM),
                            ("good", C.UI_GOOD), ("bad", C.UI_ALERT)):
            self.transcript.tag_configure(tag, foreground=colour)
        self.transcript.configure(state="disabled")
        tk.Label(row, text="was it right?", bg=C.UI_BG, fg=C.UI_MID,
                 font=C.UI_FONT_SMALL).pack(side="left", padx=(0, 6))
        for text, good, colour in (("GOOD", True, C.UI_GOOD), ("BAD", False, C.UI_ALERT)):
            self._button(row, text, lambda g=good: self._rate(g), colour, side="left", width=6)
        tk.Label(row, text="   no, it's a:", bg=C.UI_BG, fg=C.UI_MID,
                 font=C.UI_FONT_SMALL).pack(side="left", padx=(0, 6))
        self.entry = tk.Entry(row, bg=C.UI_PANEL, fg=C.UI_FG, insertbackground=C.UI_FG,
                              font=C.UI_FONT_SMALL, relief="flat", highlightthickness=1,
                              highlightbackground=C.UI_EDGE_IDLE)
        self.entry.pack(side="left", fill="x", expand=True)
        self.entry.bind("<Return>", lambda e: self._correct())
        self._button(row, "TEACH", self._correct, side="left")

    def _write(self, tag: str, text: str) -> None:
        self.transcript.configure(state="normal")
        self.transcript.insert("end", text + "\n", tag)
        self.transcript.see("end")
        self.transcript.configure(state="disabled")

    # -- actions -----------------------------------------------------------------
    def _choose(self, kind: str) -> None:
        from tkinter import filedialog
        types = {"image": [("images", "*.png *.jpg *.jpeg *.bmp *.webp")],
                 "audio": [("audio", "*.wav *.mp3 *.flac *.ogg *.m4a")],
                 "video": [("video", "*.mp4 *.avi *.mov *.mkv *.webm")]}[kind]
        path = filedialog.askopenfilename(filetypes=types + [("all", "*.*")])
        if path:
            if kind == "video":
                self.paths["image"] = self.paths["audio"] = None
            else:
                self.paths["video"] = None
            self.paths[kind] = path
            self._media_label()

    def _clear(self) -> None:
        self.paths = {"image": None, "audio": None, "video": None}
        self._media_label()

    def _media_label(self) -> None:
        parts = ["%s: %s" % (k, Path(p).name[:26]) for k, p in self.paths.items() if p]
        self.media.configure(text="\n".join(parts) or "nothing loaded")

    def _perceive(self, image=None, audio=None, source=None) -> None:
        if self.engine is None:
            return
        paths = dict(self.paths)
        if image is None and audio is None and not any(paths.values()):
            self._write("note", "[load an image, audio or video first]")
            return
        described = source or ", ".join(Path(p).name for p in paths.values() if p)
        self._write("you", "you     [%s]" % described)
        self.status = "thinking..."

        def work():
            try:
                given = image is not None or audio is not None
                thought = self.engine.think(
                    image=image if given else paths["image"],
                    audio=audio if given else paths["audio"],
                    video=None if given else paths["video"], source=described)
                self.queue.put(("thought", thought))
            except Exception as failure:                      # noqa: BLE001
                self.queue.put(("error", str(failure)))

        threading.Thread(target=work, daemon=True).start()

    def _rate(self, good: bool) -> None:
        if self.engine is None or self.thought is None:
            return
        try:
            self.engine.brain.rate(self.thought, good)
            self._write("good" if good else "bad",
                        "you     that was %s. the amygdala tagged it; the hippocampus keeps it."
                        % ("right" if good else "wrong"))
        except KeyError:
            self._write("note", "[already rated]")

    def _correct(self) -> None:
        label = self.entry.get().strip()
        if not label or self.engine is None or self.thought is None:
            return
        self.entry.delete(0, "end")
        new = label not in self.engine.brain.frontal.names
        self.engine.brain.correct(self.thought, label)
        self._write("good", "you     it's a %s.%s kept as a good memory%s."
                    % (label, " new word learned;" if new else "",
                       "; '%s' tagged bad" % self.thought.answer
                       if self.thought.answer and self.thought.answer != label else ""))

    def _replay(self) -> None:
        self.frame, self._since, self.playing = 0, 0, True

    def _set_tab(self, tab: str) -> None:
        self.tab = tab
        self._rag_hover = -1          # a tooltip must not outlive the cursor that raised it
        self._style_tabs()
        if tab == "weights":
            self.weights_panel.pack(side="left", fill="y", before=self.canvas)
            self._fill_weights()
        else:
            self.weights_panel.pack_forget()
        self._dirty = True

    def _set_focus(self, region: str) -> None:
        if region not in FOCUS:
            return
        self.focus = region
        self.focus_list.selection_clear(0, "end")
        self.focus_list.selection_set(FOCUS.index(region))
        if self.tab == "weights":
            self._fill_weights()
        self._dirty = True

    def _pick_focus(self, event=None) -> None:
        chosen = self.focus_list.curselection()
        if chosen:
            self._set_focus(FOCUS[chosen[0]])

    def _click(self, event) -> str | None:
        """-> the region clicked, or None for empty canvas."""
        if self.tab not in ("routing", "networks"):
            return None
        #  Smallest box first, so a relay wins over the THALAMUS box around it.
        for name, (x0, y0, x1, y1) in sorted(self._boxes.items(),
                                              key=lambda kv: (kv[1][2] - kv[1][0]) * (kv[1][3] - kv[1][1])):
            if x0 <= event.x <= x1 and y0 <= event.y <= y1:
                if name == "store":
                    self._set_focus("hippocampus")
                    self._set_tab("rag")
                    return "store"
                region = "thalamus" if name == "input" else name
                self._set_focus(region if region in FOCUS else "overview")
                return region
        return None

    def _open(self, event) -> None:
        """Double-click opens a region up. A single click only focused it, and
        a region's inside -- per-block activity, attention, which neurons
        fired -- is drawn on other tabs, so clicking a lobe looked dead."""
        if self.tab not in ("routing", "networks") or self._click(event) in (None, "store"):
            return
        #  Memory regions have no per-block activity; their inside is the RAG
        #  path, every state and gate from POT to motor.
        self._set_tab("rag" if self.focus in ("hippocampus", "amygdala") else "activity")

    # -- the weights panel -----------------------------------------------------
    def _fill_weights(self) -> None:
        self.weight_box.delete(0, "end")
        self.weight_list = [] if self.engine is None else self.engine.tensors(self.focus)
        for name, tensor, _, trainable in self.weight_list:
            self.weight_box.insert("end", "%s %s %s" % ("*" if trainable else " ",
                                                         name[:28], tuple(tensor.shape)))
        #  Open on the first real matrix: a 1-D norm vector says little.
        matrices = [i for i, (_, t, _, _) in enumerate(self.weight_list) if t.dim() >= 2]
        self.weight_index = matrices[0] if matrices else 0
        if self.weight_list:
            self.weight_box.selection_set(self.weight_index)
            self.weight_box.see(self.weight_index)
            self._watch_current()

    def _pick_weight(self, event=None) -> None:
        chosen = self.weight_box.curselection()
        if chosen:
            self.weight_index = chosen[0]
            self._watch_current()
            self._dirty = True

    def _watch_current(self) -> None:
        if self.engine is None or not self.weight_list:
            return
        name, tensor, owner, _ = self.weight_list[self.weight_index]
        self.engine.watch = (name, owner)

    def _weight_mode(self, mode: str) -> None:
        self.weight_mode = mode
        self._dirty = True

    def _zero_weight(self) -> None:
        if self.engine is None or not self.weight_list:
            return
        name, tensor, _, _ = self.weight_list[self.weight_index]
        self.engine.zero(name, tensor)
        self._write("bad", "[zeroed %s -- perceive again to see what it did]" % name)
        self._dirty = True

    def _restore_weight(self, everything: bool = False) -> None:
        if self.engine is None:
            return
        if everything:
            self.engine.restore()
            self._write("good", "[every edited weight restored]")
        elif self.weight_list:
            name = self.weight_list[self.weight_index][0]
            self.engine.restore(name)
            self._write("good", "[restored %s]" % name)
        self._dirty = True

    # -- lesions -------------------------------------------------------------------
    def _show_lesions(self) -> None:
        if getattr(self, "lesion_window", None) is not None:
            self.lesion_window.deiconify()
            self.lesion_window.lift()

    def _build_lesions(self) -> None:
        tk = self.tk
        window = tk.Toplevel(self.root)
        window.title("lesions")
        window.configure(bg=C.UI_BG)
        window.geometry("380x560")
        tk.Label(window, text="CUT A CONNECTION", bg=C.UI_BG, fg=C.UI_ACCENT,
                 font=C.UI_FONT_TITLE).pack(anchor="w", padx=10, pady=(10, 0))
        tk.Label(window, text="then perceive again and compare", bg=C.UI_BG,
                 fg=C.UI_DIM, font=C.UI_FONT_TINY).pack(anchor="w", padx=10)
        for name in wiring.NAMES:
            var = tk.IntVar(value=0)
            tk.Checkbutton(window, text="%-20s %s" % (name, wiring.ANATOMY[name]), variable=var,
                           command=lambda n=name, v=var: self._lesion(n, bool(v.get())),
                           bg=C.UI_BG, fg=C.UI_MID, selectcolor=C.UI_PANEL,
                           activebackground=C.UI_BG, font=C.UI_FONT_TINY,
                           anchor="w").pack(anchor="w", fill="x", padx=8)
        window.protocol("WM_DELETE_WINDOW", window.withdraw)
        window.withdraw()
        self.lesion_window = window

    def _lesion(self, name: str, cut: bool) -> None:
        with self.engine.lock:
            self.engine.brain.lesion(name, cut)
        self._dirty = True

    # -- plumbing ------------------------------------------------------------------
    def _boot(self) -> None:
        try:
            self.queue.put(("ready", Engine()))
        except Exception as failure:                          # noqa: BLE001
            self.queue.put(("error", "could not load: %s" % failure))

    def _tick(self) -> None:
        try:
            while True:
                kind, payload = self.queue.get_nowait()
                if kind == "ready":
                    self.engine = payload
                    self._roles = payload.roles()
                    self._build_lesions()
                    brain = payload.brain
                    record = brain.record
                    if brain.mismatch:
                        self._write("bad", "[%s -- run: python -m v2b.train]" % brain.mismatch)
                    if record:
                        t = record["test"]
                        self.status = "ready"
                        self._write("note", "[trained %s | held-out image %.1f%%, sound %.1f%% (%s) | "
                                    "memory: %d consolidated, %d yours]"
                                    % (record["finished"], t["image"]["fused"], t["audio"]["fused"],
                                       record["audio_data"], *brain.hippocampus.size))
                    else:
                        self.status = "UNTRAINED -- images work zero-shot; sound does not"
                    if self.tab == "weights":
                        self._fill_weights()
                    if self.snapshot:
                        self._snapshot_start()
                elif kind == "thought":
                    self.thought = payload
                    self._rag_since = time.monotonic()
                    self._rag_hover = -1
                    self.status = "%.0f ms" % payload.elapsed_ms
                    self._write("brain", "MOTOR   %s" % payload.summary())
                    self._replay()
                    if self.snapshot:
                        self.root.after(C.FRONTAL_PASSES * C.UI_PASS_MS + 300, self._snapshot_next)
                elif kind == "error":
                    self.status = "failed"
                    self._write("bad", "[failed: %s]" % payload)
                self._dirty = True
        except queue.Empty:
            pass
        if self.playing:
            self._since += C.UI_REFRESH_MS
            if self._since >= C.UI_PASS_MS:
                self._since = 0
                if self.frame < C.FRONTAL_PASSES - 1:
                    self.frame += 1
                else:
                    self.playing = False
                self._dirty = True
        if self.tab == "rag" and self._rag_animating:
            self._dirty = True
        self._tele_age += C.UI_REFRESH_MS
        if self._tele_age >= 1000:
            self._tele, self._tele_age, self._dirty = telemetry(), 0, True
        size = (self.canvas.winfo_width(), self.canvas.winfo_height())
        if self._dirty or size != getattr(self, "_size", None):
            self._size = size
            self._draw()
            self._dirty = False
        self.root.after(C.UI_REFRESH_MS, self._tick)

    # -- snapshot mode ---------------------------------------------------------------
    def _snapshot_start(self) -> None:
        from PIL import Image
        from v2b import data
        images, _, _ = data.cifar("test", per_class=1)
        self._shots = [("image", "routing", "ventral"), ("image", "networks", "ventral"),
                       ("image", "activity", "ventral"), ("image", "weights", "ventral"),
                       ("image", "memory", "hippocampus"), ("image", "overlay", "POT"),
                       ("image", "guide", "ventral"), ("sound", "activity", "temporal")]
        self._shot_images = [Image.fromarray(images[3])]
        self._perceive(image=self._shot_images[0], source="cifar test 3")

    def _snapshot_next(self) -> None:
        from PIL import ImageGrab
        if not self._shots:
            self.root.destroy()
            return
        kind, tab, focus = self._shots[0]
        current = "sound" if (self.thought and self.thought.kind == "sound") else "image"
        if kind != current:
            from v2b import data
            waves, _, _ = data.audio("test")
            self._perceive(audio=waves[0], source="test sound")
            return
        self._shots.pop(0)
        self._set_focus(focus)
        self._set_tab(tab)
        self.root.update()
        self._draw()
        self.root.update()
        x, y = self.root.winfo_rootx(), self.root.winfo_rooty()
        w, h = self.root.winfo_width(), self.root.winfo_height()
        base = Path(self.snapshot)
        ImageGrab.grab(bbox=(x, y, x + w, y + h)).save(
            base.with_name("%s_%s_%s%s" % (base.stem, tab, focus, base.suffix)))
        self.root.after(200, self._snapshot_next)

    # -- drawing -------------------------------------------------------------------------
    def _draw(self) -> None:
        canvas = self.canvas
        canvas.delete("all")
        width = max(canvas.winfo_width(), 300)
        height = max(canvas.winfo_height(), 300)
        #  MEMORY and RAG only ever show the final pass; a pass counter there
        #  read "PASS 1 / 3" over numbers from pass 3.
        passes = ("" if self.thought is None
                  else "   FINAL PASS" if self.tab in ("memory", "rag")
                  else "   PASS %d / %d" % (self.frame + 1, C.FRONTAL_PASSES))
        canvas.create_text(12, 14, anchor="w", fill=C.UI_ACCENT, font=C.UI_FONT_TITLE,
                           text="%s%s%s" % (self.tab.upper(),
                                            "" if self.tab in ("routing", "networks", "memory", "rag",
                                                               "overlay")
                                            else "  //  " + self.focus.upper(), passes))
        canvas.create_text(width - 12, 14, anchor="e", fill=C.UI_MID, font=C.UI_FONT_SMALL,
                           text=self.status.upper())
        try:
            getattr(self, "_draw_" + self.tab)(width, height)
        except Exception as failure:                          # noqa: BLE001
            canvas.create_text(width / 2, height / 2, fill=C.UI_ALERT, font=C.UI_FONT_SMALL,
                               text="could not draw %s: %s" % (self.tab, failure))
        self._draw_side()

    def _image(self, key, image, x, y, anchor="nw"):
        from PIL import ImageTk
        photo = ImageTk.PhotoImage(image)
        self._photos[key] = photo                 # Tk keeps no reference of its own
        self.canvas.create_image(x, y, image=photo, anchor=anchor)

    def _label(self, x, y, text, colour=C.UI_COOL, font=C.UI_FONT_SMALL, anchor="sw"):
        self.canvas.create_text(x, y, anchor=anchor, fill=colour, font=font, text=text)

    def _need_thought(self, width, height) -> bool:
        if self.thought is None:
            self.canvas.create_text(width / 2, height / 2, fill=C.UI_DIM, font=C.UI_FONT_SMALL,
                                    text="perceive something first")
            return False
        return True

    @staticmethod
    def _blend(a: str, b: str, level: float) -> str:
        from v2b.drawing import _mix
        return "#%02x%02x%02x" % _mix(a, b, level)

    def _level(self, name: str, value: float) -> float:
        peak = max(self._peaks.get(name, 0.0), value)
        self._peaks[name] = peak
        return value / peak if peak > 1e-9 else 0.0

    # ---- ROUTING
    def _node_note(self, name: str) -> str:
        t = self.thought
        if name == "store" and self.engine is not None:
            bank, mine = self.engine.brain.hippocampus.size
            return "%s + %d yours" % (f"{bank:,}", mine)
        if t is None:
            return self._roles.get(name, "")
        k = self.frame
        last = k == C.FRONTAL_PASSES - 1
        if name in ("pulvinar", "LGN", "MGN"):
            g = t.gains[k][name]
            return ("gain %.2f" % g) if g > 0 else "silent"
        if name == "frontal":
            return t.per_pass[k][0][0][:18]
        if name == "motor":
            if not last or not t.top:
                return "..."
            return "held back" if t.abstained else t.answer[:16]
        if name == "hippocampus":
            if not last:
                return "..."
            return "%s %.2f" % ("familiar" if t.familiarity >= 0.5 else "NOVEL", t.familiarity)
        if name == "amygdala":
            return "%s  %.2f" % (t.decision, t.value) if last else "..."
        if name in LIVE_LOBES and t.live.get(name) is not None:
            firing = t.live[name].firing
            return "%.0f%% neurons fired" % (100 * float(np.mean(firing)))
        return self._roles.get(name, "")

    def _node_live(self, name: str) -> float:
        t = self.thought
        if t is None:
            return 0.0
        g = t.gains[self.frame]
        last = self.frame == C.FRONTAL_PASSES - 1
        if name in g:
            return min(g[name], 1.0)
        if name in ("occipital", "ventral", "parietal"):
            return 1.0 if g["LGN"] > 0 else 0.0
        if name == "temporal":
            return 1.0 if g["MGN"] > 0 else 0.0
        if name in ("hippocampus", "amygdala", "motor", "store"):
            return 1.0 if last else 0.2
        return 1.0

    def _draw_routing(self, width: int, height: int) -> None:
        canvas = self.canvas
        top, bottom = 36, height - 24
        bw, bh = min(196, width * 0.2), 42
        at = {n: (x * width, top + y * (bottom - top)) for n, (x, y, _) in NODES.items()}
        carried = dict(self.thought.carried[self.frame]) if self.thought is not None else {}
        last = self.frame == C.FRONTAL_PASSES - 1
        if self.thought is not None and last and self.thought.recalled:
            carried["query"] = carried["recall"] = 1.0
        cut = self.engine.brain.wiring.cut if self.engine else set()

        def line(name, points, level, dashed=False, colour=None):
            if name in cut:
                colour, w, dashed = C.UI_ALERT, 1, True
            elif level > 0:
                colour = colour or self._blend(C.UI_EDGE_IDLE, C.UI_ACCENT, 0.3 + 0.7 * level)
                w = 1 + round(2 * level)
            else:
                colour, w = C.UI_EDGE_IDLE, 1
            canvas.create_line(*points, fill=colour, width=w, arrow="last",
                               dash=(4, 3) if dashed else None)

        def down(a, b, dx=0, dx2=None):
            (ax_, ay_), (bx_, by_) = at[a], at[b]
            return [ax_ + dx, ay_ + bh / 2, bx_ + (dx if dx2 is None else dx2), by_ - bh / 2 - 2]

        # -- the THALAMUS: one box around its three relays -----------------
        (ux, uy), (gx, _) = at["pulvinar"], at["MGN"]
        th = (ux - bw / 2 - 12, uy - bh / 2 - 20, gx + bw / 2 + 12, uy + bh / 2 + 8)
        canvas.create_rectangle(*th, outline=C.UI_ACCENT, dash=(3, 3),
                                width=2 if self.focus == "thalamus" else 1)
        canvas.create_text(th[0] + 8, th[1] + 9, anchor="w", text="THALAMUS",
                           fill=C.UI_ACCENT, font=C.UI_FONT_LABEL)

        for nucleus in ("pulvinar", "LGN", "MGN"):
            live = self.thought is not None and self.thought.gains[self.frame][nucleus] > 0
            line("input", down("input", nucleus), 1.0 if live else 0.0)
        (fx, fy), (px, py) = at["frontal"], at["POT"]
        (hx, hy), (ax, ay) = at["hippocampus"], at["amygdala"]
        (ix, iy), (sx, sy) = at["ventral"], at["STS"]
        (mx, my), (dbx, dby) = at["motor"], at["store"]
        between = (py + bh / 2 + hy - bh / 2) / 2          # the gap under POT
        below = max(my, dby) + bh / 2 + 16                 # under the bottom row
        paths = {
            "pulvinar->parietal": down("pulvinar", "parietal"),
            "LGN->occipital": down("LGN", "occipital"),
            "MGN->temporal": down("MGN", "temporal"),
            "occipital->ventral": down("occipital", "ventral"),
            "occipital->dorsal": down("occipital", "dorsal", -30, 20),
            "parietal->dorsal": down("parietal", "dorsal", 10, -20),
            "ventral->STS": [ix + bw / 2, iy, sx - bw / 2 - 2, sy],
            "temporal->STS": down("temporal", "STS"),
            "dorsal->POT": down("dorsal", "POT", 20, -30),
            "STS->POT": down("STS", "POT", -20, 30),
            "POT->frontal": [px - 18, py + bh / 2, fx - 18, fy - bh / 2 - 2],
            "frontal->POT": [fx + 18, fy - bh / 2, px + 18, py + bh / 2 + 2],
            "POT->hippocampus": [px - bw / 2, py, hx, py, hx, hy - bh / 2 - 2],
            "POT->amygdala": [px + bw / 2, py, ax, py, ax, ay - bh / 2 - 2],
            "query": [hx - 20, hy + bh / 2, hx - 20, dby - bh / 2 - 2],
            "recall": [hx + 20, dby - bh / 2, hx + 20, hy + bh / 2 + 2],
            "hippocampus->frontal": [hx + bw / 2, hy, fx - bw / 2 - 2, fy],
            #  Familiarity back to POT: a known kind is re-read with expertise.
            "hippocampus->POT": [hx + 14, hy - bh / 2, hx + 14, py + 10, px - bw / 2 - 2, py + 10],
            "hippocampus->amygdala": [hx + bw / 2, hy + 12, hx + bw / 2 + 16, hy + 12,
                                      hx + bw / 2 + 16, below, ax, below, ax, ay + bh / 2 + 2],
            "amygdala->frontal": [ax - bw / 2, ay, fx + bw / 2 + 2, fy],
            "frontal->motor": [fx, fy + bh / 2, mx, my - bh / 2 - 2],
            "amygdala->motor": [ax - 40, ay + bh / 2, ax - 40, my, mx + bw / 2 + 2, my],
            "frontal->thalamus": [fx - 60, fy - bh / 2, fx - 60, between, 8, between, 8, uy,
                                  th[0] - 2, uy],
            #  The arcuate fasciculus: temporal's sound lexicon straight to frontal.
            "temporal->frontal": [at["temporal"][0] + bw / 2, at["temporal"][1], width - 8,
                                  at["temporal"][1], width - 8, between + 8, fx + 60,
                                  between + 8, fx + 60, fy - bh / 2 - 2],
        }
        verdict = None
        if self.thought is not None and last and self.thought.top:
            verdict = C.UI_ALERT if self.thought.abstained else C.UI_GOOD
        for name, points in paths.items():
            level = self._level(name, float(carried.get(name, 0.0)))
            line(name, points, level, dashed=(name == "frontal->thalamus"),
                 colour=verdict if name == "amygdala->motor" else None)
        canvas.create_text(width - 12, (at["temporal"][1] + between) / 2, anchor="e",
                           fill=C.UI_DIM, font=C.UI_FONT_TINY, text="arcuate")
        if self.thought is not None and last and self.thought.top:
            canvas.create_text(ax - 46, (ay + bh / 2 + my) / 2, anchor="e", fill=verdict,
                               font=C.UI_FONT_DATA, text="%s %.2f" % (self.thought.decision,
                                                                      self.thought.value))

        self._boxes = {"thalamus": th}
        for name, (x, y) in at.items():
            kind = NODES[name][2]
            live = self._node_live(name)
            fill = self._blend(C.UI_PANEL, C.UI_ACCENT_DIM, 0.8 * live) if self.thought else C.UI_PANEL
            box = (x - bw / 2, y - bh / 2, x + bw / 2, y + bh / 2)
            focused = name == self.focus or (name == "store" and self.focus == "hippocampus")
            canvas.create_rectangle(*box, outline=KIND_COLOUR[kind], fill=fill,
                                    width=2 if focused else 1)
            canvas.create_text(x, y - 10, text=DISPLAY.get(name, name), fill=C.UI_FG,
                               font=C.UI_FONT_LABEL)
            canvas.create_text(x, y + 9, text=self._node_note(name), fill=C.UI_MID,
                               font=C.UI_FONT_TINY)
            self._boxes[name] = box
        lx = 12
        for text, colour in (("frozen, pretrained", C.UI_COOL), ("small, trained", C.UI_ACCENT),
                             ("memory / value", C.UI_GOOD), ("cut", C.UI_ALERT)):
            canvas.create_rectangle(lx, height - 15, lx + 10, height - 5, outline=colour)
            item = canvas.create_text(lx + 14, height - 10, anchor="w", text=text,
                                      fill=C.UI_MID, font=C.UI_FONT_TINY)
            lx = canvas.bbox(item)[2] + 16
        canvas.create_line(lx, height - 10, lx + 24, height - 10, fill=C.UI_MID, dash=(4, 3))
        canvas.create_text(lx + 30, height - 10, anchor="w", text="feedback",
                           fill=C.UI_MID, font=C.UI_FONT_TINY)
        canvas.create_text(width - 10, height - 10, anchor="e", fill=C.UI_DIM,
                           font=C.UI_FONT_TINY,
                           text="click a region to focus it -- double-click to open it up")

    # ---- NETWORKS
    def _columns(self, node: str):
        t = self.thought
        if t is None:
            return []
        if node in PER_PASS:
            return t.pass_layers[self.frame].get(node, [])
        return t.layers.get(node, [])

    def _draw_networks(self, width: int, height: int) -> None:
        from v2b.drawing import render
        columns, rows, pad, top = 5, 3, 8, 34
        tile_w = int((width - pad * (columns + 1)) / columns)
        tile_h = int((height - top - pad * (rows + 1)) / rows) - 14
        self._boxes = {}
        for index, node in enumerate(TILES):
            col, row = index % columns, index // columns
            x = pad + col * (tile_w + pad)
            y = top + 14 + row * (tile_h + pad + 14)
            kind = NODES.get(node, (0, 0, "trained"))[2]
            self._label(x + 2, y - 2, node.upper(), KIND_COLOUR[kind], C.UI_FONT_LABEL)
            self._image("tile_" + node, render(self._columns(node), tile_w, tile_h, max_nodes=10), x, y)
            if node == self.focus:
                self.canvas.create_rectangle(x, y, x + tile_w, y + tile_h, outline=C.UI_ACCENT)
            self._boxes[node] = (x, y, x + tile_w, y + tile_h)

    # ---- ACTIVITY
    def _draw_activity(self, width: int, height: int) -> None:
        if not self._need_thought(width, height):
            return
        t, focus = self.thought, self.focus
        if focus in LIVE_LOBES:
            live = t.live.get(focus)
            if live is None:
                self._label(20, 60, "%s was silent this percept -- nothing reached it" % focus,
                            C.UI_DIM, anchor="w")
                return
            return self._draw_lobe_activity(live, width, height)
        if focus == "overview":
            return self._draw_guide(width, height)
        self._draw_circuit_activity(width, height)

    def _draw_lobe_activity(self, live, width: int, height: int) -> None:
        from PIL import Image
        from v2b.drawing import heatmap, raster
        t, focus = self.thought, self.focus
        top, x0 = 40, 12
        blocks = len(live.blocks)
        first = {"occipital": 1, "ventral": self.engine.brain.vision.split + 1}.get(focus, 1)
        per_row = min(blocks, 6)
        rows = int(np.ceil(blocks / per_row))
        lead = 0
        if focus in ("occipital", "ventral", "parietal") and t.retina is not None:
            lead = min(150, height // 4)
            picture = Image.fromarray(t.retina) if focus != "parietal" else None
            if picture is None:
                from v2b.drawing import depth_image
                picture = depth_image(t.depth)
            self._image("act_lead", picture.resize((lead, lead)), x0, top + 16)
            self._label(x0, top + 14, "what the eyes saw" if focus != "parietal" else "depth map")
        elif focus == "temporal" and "spectrogram" in live.extra:
            lead = min(150, height // 4)
            self._image("act_lead", heatmap(live.extra["spectrogram"][::-1], lead, lead, signed=False),
                        x0, top + 16)
            self._label(x0, top + 14, "spectrogram")
        cell = int(min((width - lead - 40) / per_row - 8, (height * 0.42) / (rows * (2 if live.attention else 1)) - 18))
        cell = max(cell, 24)
        gx = x0 + lead + 16
        y = top + 16
        self._label(gx, top + 2, "per block: how strongly each %s is represented"
                    % ("patch" if focus != "temporal" else "frequency x time patch"), C.UI_MID,
                    C.UI_FONT_TINY)
        for i, grid in enumerate(live.blocks):
            cx = gx + (i % per_row) * (cell + 8)
            cy = y + (i // per_row) * (cell + 18)
            self._image("act_b%d" % i, heatmap(grid, cell, cell, signed=False), cx, cy)
            names = live.extra.get("map_labels") or []
            caption = (names[i] if i < len(names) else
                       "blk %d  %.0f%%" % (first + i, 100 * live.firing[i])
                       if i < len(live.firing) else "blk %d" % (first + i))
            self._label(cx, cy + cell + 12, caption, C.UI_MID, C.UI_FONT_TINY)
        y += rows * (cell + 18) + 6
        if live.attention:
            self._label(gx, y + 10, "per block: where attention went (heads averaged)",
                        C.UI_MID, C.UI_FONT_TINY)
            y += 14
            for i, grid in enumerate(live.attention):
                cx = gx + (i % per_row) * (cell + 8)
                cy = y + (i // per_row) * (cell + 18)
                self._image("act_a%d" % i, heatmap(grid, cell, cell, signed=False), cx, cy)
            y += rows * (cell + 18)
        rh = max(height - y - 30, 40)
        self._label(x0, y + 12, "NEURONS: one row per block, one column per group of MLP "
                                "neurons; amber = firing, blue = suppressed", C.UI_MID, C.UI_FONT_TINY)
        self._image("act_raster", raster(live.neurons, width - 150, rh), x0, y + 16)
        bx = width - 128
        for i, share in enumerate(live.firing):
            by = y + 16 + (i + 0.5) * rh / max(len(live.firing), 1)
            self.canvas.create_rectangle(bx, by - 3, bx + 90 * share / max(max(live.firing), 1e-6),
                                         by + 3, fill=C.UI_ACCENT, outline="")
            self.canvas.create_text(bx + 96, by, anchor="w", fill=C.UI_MID, font=C.UI_FONT_TINY,
                                    text="%.0f%%" % (100 * share))

    def _draw_circuit_activity(self, width: int, height: int) -> None:
        from v2b.drawing import heatmap
        t, focus = self.thought, self.focus
        top = 44
        if focus in PER_PASS:
            rows = [(k, t.pass_layers[k].get(focus, [])) for k in range(C.FRONTAL_PASSES)]
        else:
            rows = [(None, t.layers.get(focus, []))]
        labels = [label for label, _ in rows[0][1]]
        if not labels:
            self._label(20, 60, "no activity recorded for %s" % focus, C.UI_DIM, anchor="w")
            return
        col_w = (width - 110) / len(labels)
        row_h = min(60, (height - top - 200) / max(len(rows), 1))
        for j, label in enumerate(labels):
            self._label(100 + j * col_w, top + 12, label, C.UI_COOL, C.UI_FONT_TINY)
        for i, (k, cols) in enumerate(rows):
            y = top + 18 + i * (row_h + 8)
            self._label(12, y + row_h / 2 + 5, "pass %d" % (k + 1) if k is not None else "once",
                        C.UI_ACCENT if k == self.frame else C.UI_MID, C.UI_FONT_TINY)
            for j, (_, vector) in enumerate(cols):
                if vector is None:
                    self._label(100 + j * col_w, y + row_h / 2 + 5, "silent", C.UI_DIM, C.UI_FONT_TINY)
                    continue
                strip = np.asarray(vector, np.float32)[None]
                self._image("circ_%d_%d" % (i, j), heatmap(strip, int(col_w - 10), int(row_h)),
                            100 + j * col_w, y)
        y = top + 26 + len(rows) * (row_h + 8)
        self._draw_circuit_detail(focus, 12, y, width - 24, height - y - 10)

    def _draw_circuit_detail(self, focus, x, y, w, h) -> None:
        """Region-specific readouts under the heat strips."""
        from PIL import Image
        from v2b.drawing import heat_overlay
        t, canvas = self.thought, self.canvas

        def bars(items, bx, by, bw, title, colour=C.UI_ACCENT):
            self._label(bx, by, title, C.UI_COOL, C.UI_FONT_TINY)
            peak = max([abs(v) for _, v in items] + [1e-9])
            for i, (name, value) in enumerate(items):
                yy = by + 14 + i * 14
                canvas.create_text(bx, yy, anchor="w", fill=C.UI_FG, font=C.UI_FONT_TINY,
                                   text=str(name)[:18])
                canvas.create_rectangle(bx + 130, yy - 4, bx + 130 + (bw - 180) * abs(value) / peak,
                                        yy + 4, fill=colour, outline="")
                canvas.create_text(bx + bw - 44, yy, anchor="w", fill=C.UI_MID,
                                   font=C.UI_FONT_TINY, text="%.3f" % value)

        if focus == "frontal":
            for k in range(C.FRONTAL_PASSES):
                bars(t.per_pass[k], x + k * w / 3, y + 14, w / 3 - 10,
                     "pass %d: top guesses" % (k + 1),
                     C.UI_ACCENT if k == self.frame else C.UI_ACCENT_DIM)
        elif focus in ("pulvinar", "LGN", "MGN", "thalamus"):
            for k in range(C.FRONTAL_PASSES):
                bars(list(t.gains[k].items()), x + k * w / 3, y + 14, w / 3 - 10,
                     "pass %d: relay gains (x frontal's feedback)" % (k + 1))
        elif focus == "POT" and t.retina is not None:
            size = int(min(h - 30, w / 3 - 20))
            for k in range(C.FRONTAL_PASSES):
                where = t.attention[k][1:1 + C.GRID * C.GRID]
                self._image("pot_%d" % k, heat_overlay(t.retina, where).resize((size, size)),
                            x + k * (size + 20), y + 24)
                self._label(x + k * (size + 20), y + 22, "pass %d: where POT looked" % (k + 1))
        elif focus == "amygdala":
            bars(list(t.amygdala_inputs.items()) + [("VALUE", t.value)], x, y + 14, w * 0.6,
                 "what the amygdala reads -> P(answer is good); below %.2f it says 'not sure'"
                 % C.ABSTAIN_BELOW)
        elif focus == "hippocampus":
            bars([(r[1], r[0]) for r in t.recalled[:12]], x, y + 14, w * 0.6,
                 "recalled (similarity in entorhinal key space) -- MEMORY tab shows the pictures")

    # ---- WEIGHTS
    def _draw_weights(self, width: int, height: int) -> None:
        from v2b.drawing import heatmap, histogram
        canvas = self.canvas
        if self.engine is None or not self.weight_list:
            self._label(20, 60, "no weights in '%s' -- pick a region in FOCUS" % self.focus,
                        C.UI_DIM, anchor="w")
            return
        name, tensor, owner, trainable = self.weight_list[self.weight_index]
        array, sampled = tensor_view(tensor)
        edited = name in self.engine.saved
        top = 40
        self._label(12, top + 4, name, C.UI_ALERT if edited else C.UI_FG, C.UI_FONT_SMALL)
        self._label(12, top + 20, "shape %s   %s numbers   %s%s" % (
            tuple(tensor.shape), f"{tensor.numel():,}",
            "TRAINED by v2.B" if trainable else "frozen, pretrained",
            "   ZEROED -- restore to undo" if edited else ""),
            C.UI_ALERT if edited else C.UI_MID, C.UI_FONT_TINY)
        top += 12
        live = None
        if self.weight_mode == "live" and self.thought is not None and self.thought.watched:
            watched_name, captured = self.thought.watched
            if watched_name == name and "input" in captured and array.ndim == 2 \
                    and captured["input"].size == array.shape[1]:
                live = np.abs(array) * captured["input"][None]
        hm_h = int(height * 0.55)
        hm_w = width - 24
        shown = live if live is not None else array
        self._image("w_heat", heatmap(shown, hm_w, hm_h, signed=live is None), 12, top + 20)
        caption = ("|weight| x input activity, last percept: bright = carried signal"
                   if live is not None else
                   "rows = output neurons, columns = inputs; amber +, blue -"
                   + ("  (rows sampled)" if sampled else ""))
        if self.weight_mode == "live" and live is None:
            caption = "x live input: PERCEIVE once more -- this layer's input is recorded then"
        self._label(12, top + 36 + hm_h, caption, C.UI_MID, C.UI_FONT_TINY)
        y = top + 44 + hm_h
        self._image("w_hist", histogram(array, int(width * 0.55), height - y - 30), 12, y + 4)
        flat = array.reshape(-1)
        stats = [("mean", flat.mean()), ("std", flat.std()), ("min", flat.min()),
                 ("max", flat.max()), ("L2 norm", float(np.linalg.norm(flat))),
                 ("near zero", float((np.abs(flat) < 1e-3).mean()))]
        sx = int(width * 0.55) + 30
        for i, (label, value) in enumerate(stats):
            canvas.create_text(sx, y + 12 + i * 15, anchor="w", fill=C.UI_MID,
                               font=C.UI_FONT_TINY, text=label)
            canvas.create_text(sx + 110, y + 12 + i * 15, anchor="w", fill=C.UI_FG,
                               font=C.UI_FONT_DATA,
                               text=("%.1f%%" % (100 * value)) if label == "near zero" else "%.5f" % value)
        canvas.create_text(sx, y + 12 + len(stats) * 15 + 6, anchor="w", fill=C.UI_DIM,
                           font=C.UI_FONT_TINY, text="(%s)" % ("on a row sample" if sampled else "all values"))

    # ---- MEMORY
    def _draw_memory(self, width: int, height: int) -> None:
        if not self._need_thought(width, height):
            return
        from PIL import Image
        from v2b import data
        t, canvas = self.thought, self.canvas
        top = 40
        col = (width - 40) / 3
        for i, (title, items) in enumerate((("CORTEX alone", t.cortex_top),
                                            ("MEMORY alone (CA1 vote)", t.memory_top),
                                            ("FINAL = cortex + %.2f x %.2f x memory"
                                             % (t.trust, t.familiarity), t.top))):
            x = 12 + i * (col + 8)
            self._label(x, top + 14, title)
            for j, (name, p) in enumerate(items[:5]):
                yy = top + 30 + j * 15
                canvas.create_text(x, yy, anchor="w", fill=C.UI_FG, font=C.UI_FONT_TINY,
                                   text=name[:18])
                canvas.create_rectangle(x + 120, yy - 4, x + 120 + (col - 170) * p, yy + 4,
                                        fill=C.UI_ACCENT, outline="")
                canvas.create_text(x + col - 44, yy, anchor="w", fill=C.UI_MID,
                                   font=C.UI_FONT_TINY, text="%.0f%%" % (100 * p))
        y = top + 30 + 5 * 15 + 10
        a = t.amygdala_inputs
        self._label(12, y + 4, "AMYGDALA -> MOTOR:  %s   confidence %.2f  (HOLD below %.2f)%s"
                    % (t.decision, t.value, C.ABSTAIN_BELOW,
                       "  -- the answer is held back as 'not sure'" if t.abstained else ""),
                    C.UI_ALERT if t.abstained else C.UI_GOOD, C.UI_FONT_SMALL)
        self._label(12, y + 20, "it read, from the hippocampus: memory sure %.2f, closest memory "
                    "%.2f, memory agrees with cortex: %s;  from cortex: sure %.2f, margin %.2f"
                    % (a.get("memory sure", 0), a.get("closest memory", 0),
                       "yes" if a.get("cortex = memory", 0) > 0.5 else "no",
                       a.get("cortex sure", 0), a.get("cortex margin", 0)),
                    C.UI_MID, C.UI_FONT_TINY)
        y += 32
        hippo = self.engine.brain.hippocampus
        db = self._database_summary()
        self._label(12, y + 12, "RAG DATABASE -- the hippocampus's store", C.UI_GOOD, C.UI_FONT_LABEL)
        lines = [
            "%s entries:" % f"{db['total']:,}"] + ["   " + part for part in db["parts"]] + [
            "each: one %d-number POT vector + its label" % hippo.embed,
            "search: %d-number entorhinal keys, faiss exact inner product, %d nearest"
            % (C.KEY_DIM, C.MEMORY_K),
            "readout: CA1 sharpness %.1f, trust %.3f"
            % (float(self.engine.brain.ca1.sharpness.detach()), t.trust),
            "files: checkpoints/v2b_memory.npz (store) + v2b/memory/episodes.json (yours)"]
        for i, text in enumerate(lines):
            self._label(20, y + 28 + 14 * i, text, C.UI_FG, C.UI_FONT_TINY)
        #  Episodes to the right when there is room, otherwise underneath.
        ex = 12
        if width > 1100:
            ex = width * 0.55
        else:
            y += 28 + 14 * len(lines)
        self._label(ex, y + 12, "YOUR EPISODES (%d) -- direct path, up to %.0f logits" %
                    (len(hippo.kept), C.EPISODE_VOTE), C.UI_GOOD, C.UI_FONT_LABEL)
        matching = {e.id: s for e, s in t.tags}
        if not hippo.kept:
            self._label(ex + 8, y + 28, "none yet -- teach it with 'no, it's a:' below",
                        C.UI_DIM, C.UI_FONT_TINY)
        for i, e in enumerate(hippo.kept[-4:][::-1]):
            hit = e.id in matching
            self._label(ex + 8, y + 28 + 14 * i, "%s %-16s %-5s %s%s" % (
                "+" if e.valence > 0 else "-", e.label[:16], e.kind,
                time.strftime("%m-%d %H:%M", time.localtime(e.when)),
                "   <- matches this input (sim %.2f)" % matching[e.id] if hit else ""),
                C.UI_GOOD if hit else C.UI_MID, C.UI_FONT_TINY)
        y += 28 + 14 * (max(len(lines), 4) if ex > 12 else 4) + 6
        self._label(12, y + 12, "RECALLED FOR THIS INPUT -- the %d nearest entries.  CA1 familiarity "
                    "%.2f: %s" % (len(t.recalled), t.familiarity,
                                  "seen things like this, memory votes" if t.familiarity >= 0.5 else
                                  "NOVEL -- nothing stored is really like this, so memory barely votes"))
        size = int(min((width - 24) / 8 - 8, (height - y - 60) / 2 - 30))
        size = max(size, 40)
        for i, (sim, label, weight, what) in enumerate(t.recalled[:16]):
            cx = 12 + (i % 8) * (size + 8)
            cy = y + 20 + (i // 8) * (size + 34)
            kind, ref = what
            picture = None
            if kind == "bank" and str(ref).startswith(("cifar", "tiny-", "inet-")):
                pixels = data.picture(ref)
                if pixels is not None:
                    picture = Image.fromarray(pixels).resize((size, size), Image.NEAREST)
            if picture is not None:
                self._image("mem_%d" % i, picture, cx, cy)
            else:
                canvas.create_rectangle(cx, cy, cx + size, cy + size, outline=C.UI_EDGE_IDLE,
                                        fill=C.UI_PANEL)
                canvas.create_text(cx + size / 2, cy + size / 2, fill=C.UI_MID, font=C.UI_FONT_TINY,
                                   text="sound" if kind == "bank" else "yours")
            if kind == "episode":
                canvas.create_rectangle(cx, cy, cx + size, cy + size, outline=C.UI_GOOD, width=2)
            canvas.create_text(cx, cy + size + 8, anchor="w", fill=C.UI_FG, font=C.UI_FONT_TINY,
                               text=label[:12])
            canvas.create_text(cx, cy + size + 21, anchor="w", fill=C.UI_MID, font=C.UI_FONT_TINY,
                               text="sim %.2f%s" % (sim, "  x%.0f" % weight if weight != 1 else ""))

    def _database_summary(self) -> dict:
        """What the RAG database holds, counted once from its own records."""
        hippo = self.engine.brain.hippocampus
        key = (len(hippo.bank_labels), len(hippo.kept))
        if getattr(self, "_db_key", None) != key:
            sources = np.array([s.split(":")[0] for s in hippo.bank_sources])                 if hippo.bank_sources else np.array([])
            trained = hippo.bank_trained if len(hippo.bank_trained) == len(sources)                 else np.ones(len(sources), bool)
            images = ((np.char.startswith(sources.astype(str), "cifar")
                       | np.char.startswith(sources.astype(str), "tiny")
                       | np.char.startswith(sources.astype(str), "inet")) if len(sources) else sources)
            parts = []
            for label, mask in (("images cortex trained on", images & trained),
                                ("sounds cortex trained on", ~images & trained),
                                ("images it never trained on", images & ~trained),
                                ("sounds it never trained on", ~images & ~trained)):
                if len(sources) and mask.sum():
                    parts.append("%s %s" % (f"{int(mask.sum()):,}", label))
            self._db = {"total": len(sources), "parts": parts or ["empty -- run train"]}
            self._db_key = key
        return self._db

    # ---- RAG
    def _draw_rag(self, width: int, height: int) -> None:
        from v2b import ragview
        if self.engine is None:
            self._label(width / 2, height / 2, "loading the brain...", C.UI_DIM, anchor="center")
            return
        brain = self.engine.brain
        hippo = brain.hippocampus
        key = (len(hippo.bank_labels), id(hippo.bank_labels))
        if getattr(self, "_rag_key", None) != key:
            self._rag_layout = ragview.StoreLayout(hippo.bank_labels, hippo.bank_sources,
                                                   brain.frontal.names)
            self._rag_key = key
        trace = self.thought.rag if self.thought is not None and self.thought.rag else None
        reveal = ((time.monotonic() - self._rag_since) * 1000 / C.UI_RAG_STAGE_MS
                  if trace else 99.0)
        self._rag_animating = reveal < len(ragview.STAGES) + 1
        image, origin = ragview.render(trace, self._rag_layout, width - 24, height - 34,
                                       reveal, self._rag_hover)
        self._image("rag", image, 12, 30)
        self._rag_origin = (12 + origin[0], 30 + origin[1]) if origin else None
        self._draw_rag_tip(width, height, trace)

    def _rag_similarity(self):
        rag = self.thought.rag if self.thought is not None else None
        sim = rag.get("similarity") if rag else None
        return np.asarray(sim, np.float32) if sim is not None and len(sim) else None

    def _hover(self, event) -> None:
        if self.tab != "rag" or self._rag_origin is None or self._rag_layout is None:
            return
        ox, oy = self._rag_origin
        index = self._rag_layout.memory_at(event.x - ox, event.y - oy, self._rag_similarity())
        self._rag_mouse = (event.x, event.y)
        if index != self._rag_hover:
            self._rag_hover = index
            self._dirty = True                # redrawn on the next tick: 10 per second at most

    def _unhover(self, event=None) -> None:
        if self._rag_hover != -1:
            self._rag_hover = -1
            self._dirty = True

    def _draw_rag_tip(self, width: int, height: int, trace) -> None:
        """What the memory under the mouse is: its label, where it came from,
        how this percept activates it, and its picture."""
        i = self._rag_hover
        hippo = self.engine.brain.hippocampus
        if i < 0 or i >= len(hippo.bank_labels):
            return
        from v2b import data
        names = self.engine.brain.frontal.names
        label = int(hippo.bank_labels[i])
        source = hippo.bank_sources[i] if i < len(hippo.bank_sources) else ""
        lines = ["memory #%s   %s" % (f"{i:,}", names[label] if label < len(names) else "?"),
                 "from %s%s" % (source, "" if hippo.bank_trained[i]
                                else "  (stored, never trained on)")]
        sim = self._rag_similarity()
        if sim is not None and trace:
            rank = int((sim > sim[i]).sum()) + 1
            share = next((r["strength"] for r in trace.get("recalled", []) if r["index"] == i),
                         None)
            lines.append("cosine with this percept %.3f   rank %s of %s" % (
                float(sim[i]), f"{rank:,}", f"{len(sim):,}"))
            lines.append("RECALLED: carries %.0f%% of CA1's vote" % (100 * share)
                         if share is not None
                         else "not among the %d recalled" % len(trace.get("recalled", [])))
        if i not in self._thumbs:
            if len(self._thumbs) > 64:
                self._thumbs.clear()
            self._thumbs[i] = (data.picture(source)
                               if source.startswith(("cifar", "tiny-", "inet-"))
                               else None)
        pixels = self._thumbs[i]
        size = 72 if pixels is not None else 0
        box_w, box_h = 360 + size, max(18 + 15 * len(lines), size + 12)
        x, y = self._rag_mouse
        x = x + 18 if x + 18 + box_w < width else x - 18 - box_w
        y = y + 18 if y + 18 + box_h < height else y - 18 - box_h
        self.canvas.create_rectangle(x, y, x + box_w, y + box_h, fill=C.UI_PANEL,
                                     outline=C.UI_ACCENT)
        if pixels is not None:
            from PIL import Image
            self._image("rag_tip", Image.fromarray(pixels).resize((size, size), Image.NEAREST),
                        x + 6, y + 6)
        for j, text in enumerate(lines):
            self._label(x + size + 12, y + 20 + 15 * j, text,
                        C.UI_ACCENT if j == 0 else (C.UI_GOOD if text.startswith("RECALLED")
                                                    else C.UI_FG), C.UI_FONT_TINY)

    # ---- OVERLAY
    def _draw_overlay(self, width: int, height: int) -> None:
        if not self._need_thought(width, height):
            return
        from PIL import Image
        from v2b.drawing import depth_image, heat_overlay, heatmap
        t = self.thought
        size = int(min((width - 60) / 3, height - 220, 300))
        y = 64
        att = t.attention[self.frame]
        grid = C.GRID * C.GRID
        if t.retina is not None:
            it_live = t.live.get("ventral")
            panels = [("POT attention, pass %d" % (self.frame + 1), heat_overlay(t.retina, att[1:1 + grid])),
                      ("IT attention, block %d" % len(self.engine.brain.vision.blocks),
                       heat_overlay(t.retina, _pool7(it_live.attention[-1])) if it_live else Image.fromarray(t.retina)),
                      ("depth: near = bright", depth_image(t.depth))]
            for i, (label, image) in enumerate(panels):
                x = 20 + i * (size + 10)
                self._image("ov_%d" % i, image.resize((size, size)), x, y)
                self._label(x, y - 4, label)
        cursor = y + (size if t.retina is not None else 0) + 18
        split = {"nothing": float(att[0]), "dorsal (where)": float(att[1:1 + grid].sum()),
                 "STS (what)": float(att[1 + grid:].sum())}
        self._label(20, cursor + 4, "POT's attention, pass %d:  %s" % (self.frame + 1, "   ".join(
            "%s %.0f%%" % (k, 100 * v) for k, v in split.items())), C.UI_FG, C.UI_FONT_TINY)
        if t.retina is not None:
            self._label(20, cursor + 18, "bright corners in IT's map are attention sinks -- "
                        "tokens a vision transformer parks attention on; a known artifact, "
                        "not the object", C.UI_DIM, C.UI_FONT_TINY)
        cursor += 30
        if t.wave is not None:
            live = t.live.get("temporal")
            if live is not None and "spectrogram" in live.extra:
                self._image("ov_spec", heatmap(live.extra["spectrogram"][::-1], width - 60, 70, signed=False),
                            20, cursor)
                self._label(20, cursor - 4, "the sound (spectrogram, low pitch at the bottom)")
                cursor += 80
            sound = att[1 + grid + STS_SOUND_OFFSET:]
            self._label(20, cursor + 10, "POT attention over the sound's 16 time slices, pass %d"
                        % (self.frame + 1))
            cursor += 16
            peak = max(float(np.max(sound)) if len(sound) else 0, 1e-9)
            span = width - 60
            for i, value in enumerate(sound):
                x0 = 20 + i * span / len(sound)
                h = 50 * float(value) / peak
                self.canvas.create_rectangle(x0, cursor + 50 - h, x0 + span / len(sound) - 3,
                                             cursor + 50, fill=C.UI_ACCENT, outline="")
            cursor += 64
        self._label(20, cursor + 10, "frontal's top guesses, pass by pass")
        for k, guesses in enumerate(t.per_pass):
            self.canvas.create_text(20, cursor + 26 + 15 * k, anchor="w",
                                    fill=C.UI_ACCENT if k == self.frame else C.UI_MID,
                                    font=C.UI_FONT_DATA, text="pass %d   %s" % (k + 1, "   ".join(
                                        "%s %.0f%%" % (n, 100 * p) for n, p in guesses)))

    # ---- GUIDE
    def _draw_guide(self, width: int, height: int) -> None:
        item = self.canvas.create_text(20, 44, anchor="nw", fill=C.UI_FG, font=C.UI_FONT_SMALL,
                                       text=guide.text(self.focus), width=width - 40)
        live = self._live_facts()
        if live:
            y = self.canvas.bbox(item)[3] + 16
            self.canvas.create_text(20, y, anchor="nw", fill=C.UI_ACCENT, font=C.UI_FONT_SMALL,
                                    text="LIVE, THIS PERCEPT\n" + live, width=width - 40)

    def _live_facts(self) -> str:
        """The focused region's numbers from the last percept, in words."""
        t, f = self.thought, self.focus
        if t is None:
            return ""
        if f in ("pulvinar", "LGN", "MGN"):
            return "gain per pass: %s" % "  ".join("%.3f" % g[f] for g in t.gains)
        if f == "thalamus":
            return "\n".join("pass %d: %s" % (k + 1, "  ".join("%s %.3f" % kv for kv in g.items()))
                             for k, g in enumerate(t.gains))
        if f in LIVE_LOBES:
            live = t.live.get(f)
            if live is None:
                return "silent -- nothing reached it"
            return "share of MLP neurons firing, block by block: %s" % "  ".join(
                "%.0f%%" % (100 * x) for x in live.firing)
        if f == "frontal":
            return "\n".join("pass %d guesses: %s" % (k + 1, ", ".join(
                "%s %.0f%%" % (n, 100 * p) for n, p in g)) for k, g in enumerate(t.per_pass))
        if f == "hippocampus":
            return ("recalled %d memories; memory alone says %s; CA1 trust %.3f; "
                    "your episodes matching: %d" % (
                        len(t.recalled), t.memory_top[0][0] if t.memory_top else "-",
                        t.trust, len(t.tags)))
        if f == "amygdala":
            return "value %.2f (P answer is good) -> %s\nreads: %s" % (
                t.value, "not sure" if t.abstained else "answers",
                "  ".join("%s %.2f" % kv for kv in t.amygdala_inputs.items()))
        if f == "POT":
            a = t.attention[self.frame]
            g = C.GRID * C.GRID
            return "attention, pass %d: nothing %.0f%%, dorsal %.0f%%, STS %.0f%%" % (
                self.frame + 1, 100 * a[0], 100 * a[1:1 + g].sum(), 100 * a[1 + g:].sum())
        return "answer: %s (%.0f%%); gains %s" % (
            t.answer, 100 * t.confidence,
            "  ".join("%s %.2f" % kv for kv in t.gains[-1].items()))

    # ---- the side panel
    def _draw_side(self) -> None:
        side = self.side
        side.delete("all")
        width = max(side.winfo_width(), 160)
        cursor = 18

        def title(text, colour=C.UI_ACCENT):
            nonlocal cursor
            side.create_text(12, cursor, anchor="w", fill=colour,
                             font=C.UI_FONT_TITLE if colour == C.UI_ACCENT else C.UI_FONT_SMALL,
                             text=text)
            cursor += 20

        def row(label, value, colour=C.UI_FG):
            nonlocal cursor
            side.create_text(16, cursor, anchor="w", fill=C.UI_MID, font=C.UI_FONT_TINY, text=label)
            side.create_text(width - 12, cursor, anchor="e", fill=colour, font=C.UI_FONT_DATA,
                             text=str(value)[:26])
            cursor += 15

        def bar(label, value):
            nonlocal cursor
            side.create_text(16, cursor, anchor="w", fill=C.UI_FG, font=C.UI_FONT_TINY,
                             text=label[:20])
            side.create_rectangle(width - 84, cursor - 4, width - 14, cursor + 4,
                                  outline=C.UI_EDGE_IDLE)
            side.create_rectangle(width - 84, cursor - 4, width - 84 + 70 * value, cursor + 4,
                                  fill=C.UI_ACCENT, outline="")
            cursor += 15

        title("SYSTEM")
        row("cpu", "%.0f%%" % self._tele["cpu"])
        row("ram", "%.0f%%" % self._tele["ram"])
        row("gpu memory", self._tele["gpu"])
        if self.engine is not None:
            bank, mine = self.engine.brain.hippocampus.size
            row("memories", "%s + %d yours" % (f"{bank:,}", mine))
            if self.engine.saved:
                row("weights zeroed", len(self.engine.saved), C.UI_ALERT)
            if self.engine.brain.wiring.cut:
                row("connections cut", len(self.engine.brain.wiring.cut), C.UI_ALERT)
        cursor += 6
        t = self.thought
        if t is None:
            return
        title("ANSWER", C.UI_COOL)
        row("motor says", ("not sure: " if t.abstained else "") + (t.answer or "-"),
            C.UI_ALERT if t.abstained else C.UI_ACCENT)
        row("amygdala -> motor", "%s  confidence %.2f" % (t.decision, t.value),
            C.UI_ALERT if t.abstained else C.UI_GOOD)
        for name, p in t.top[:5]:
            bar(name, p)
        cursor += 6
        title("cortex / memory / final", C.UI_COOL)
        for label, items in (("cortex", t.cortex_top), ("memory", t.memory_top), ("final", t.top)):
            if items:
                row(label, "%s %.0f%%" % (items[0][0][:16], 100 * items[0][1]))
        row("CA1 trust", "%.2f" % t.trust)
        row("familiarity", "%.2f %s" % (t.familiarity, "familiar" if t.familiarity >= 0.5 else "novel"),
            C.UI_GOOD if t.familiarity >= 0.5 else C.UI_ALERT)
        cursor += 6
        title("thalamic gains, pass %d" % (self.frame + 1), C.UI_COOL)
        for name, gain in t.gains[self.frame].items():
            row(name, "%.3f" % gain if gain > 0 else "silent")
        cursor += 6
        title("this percept", C.UI_COOL)
        row("senses", "+".join(t.modalities))
        row("elapsed", "%.0f ms" % t.elapsed_ms)


STS_SOUND_OFFSET = 1 + C.GRID * C.GRID + 1   # STS tokens before the 16 time slices


def _pool7(grid) -> np.ndarray:
    """Any square attention map -> 7x7, to paint over the retina."""
    import torch
    import torch.nn.functional as F
    g = torch.as_tensor(np.asarray(grid, np.float32))[None, None]
    return F.adaptive_avg_pool2d(g, C.GRID)[0, 0].numpy()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--selftest", action="store_true")
    parser.add_argument("--snapshot", help="screenshot every view to <name>_<view>.png, then exit")
    parser.add_argument("--candidate", action="store_true",
                        help="load checkpoints/v2b_candidate.pt (+ its memory) instead of v2b.pt")
    args = parser.parse_args()
    if args.candidate:
        WHICH["checkpoint"] = C.CHECKPOINT.with_name("v2b_candidate.pt")
        WHICH["bank"] = C.MEMORY_BANK.with_name("v2b_memory_candidate.npz")
    if args.selftest:
        return selftest()
    import tkinter as tk
    try:
        import ctypes
        ctypes.windll.shcore.SetProcessDpiAwareness(1)        # crisp on Windows
    except Exception:                                         # noqa: BLE001
        pass
    root = tk.Tk()
    Console(root, snapshot=args.snapshot)
    root.mainloop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
