"""
monitor.py -- watch v2.B train, live.

    python -m v2b.train --watch      train and open this window
    python -m v2b.monitor            open it on a run already going

The trainer writes one line of JSON per event to results/v2b_live.jsonl; this
window follows that file and redraws. It never touches the GPU or the model,
so it cannot slow training down or crash it -- close it and training goes on.

WHAT YOU SEE
  top      the stage, how long it has run, and how long is left
  encode   the frozen networks turning every picture into features, per set
  loss     how wrong v2.B is on each training batch; it should fall
  accuracy after every epoch: held-out pictures and sounds it is not trained on
           (validation), and pictures of classes it has never trained on at all
           (shown only -- nothing is tuned on them)
  samples  real pictures from the batch being trained right now, what they
           are, and what v2.B currently says -- green right, red wrong
"""

from __future__ import annotations

import json
import queue
import subprocess
import sys
import threading
import time

from v2b import config_v2b as C

LIVE = C.ROOT / "results" / "v2b_live.jsonl"


class Monitor:
    def __init__(self, root):
        import tkinter as tk
        self.tk, self.root = tk, root
        root.title("brain-ai  //  v2.B training, live")
        root.configure(bg=C.UI_BG)
        root.geometry("1180x760")
        self.canvas = tk.Canvas(root, bg=C.UI_BG, highlightthickness=0)
        self.canvas.pack(fill="both", expand=True)
        self.reset()
        self._gpu = "-"
        self._photos: dict = {}
        self._thumbs: dict = {}
        self._wanted: queue.Queue = queue.Queue()
        threading.Thread(target=self._thumbnailer, daemon=True).start()
        threading.Thread(target=self._gpu_probe, daemon=True).start()
        root.after(200, self.tick)

    def reset(self):
        self.offset = 0
        self.started = None
        self.stage = "waiting for the trainer..."
        self.encode: dict = {}          # split -> (done, total)
        self.losses: list = []          # (global step, loss)
        self.epochs: list = []          # epoch events
        self.untrained: dict | None = None
        self.samples: list = []
        self.info: dict = {}
        self.last: dict = {}
        self.keep: dict | None = None
        self.done: dict | None = None
        self.memory_note = ""

    # -- reading the live log ---------------------------------------------------
    def poll(self) -> bool:
        if not LIVE.exists():
            return False
        size = LIVE.stat().st_size
        if size < self.offset:              # a new run truncated the file
            self.reset()
        if size == self.offset:
            return False
        with open(LIVE, "r", encoding="utf-8") as handle:
            handle.seek(self.offset)
            chunk = handle.read()
            self.offset = handle.tell()
        for line in chunk.splitlines():
            try:
                self.apply(json.loads(line))
            except json.JSONDecodeError:
                pass
        return True

    def apply(self, e: dict) -> None:
        stage = e.get("stage")
        self.started = self.started or e.get("t")
        self.last = e
        if stage == "start":
            self.info = e
            self.stage = "starting: %s pictures, %s sounds, %s epochs" % (
                f"{e['train_images']:,}", f"{e['train_sounds']:,}", e["epochs"])
        elif stage == "encode":
            self.encode[e["split"]] = (e["done"], e["total"])
            self.stage = "ENCODING with the frozen networks: %s" % e["split"]
        elif stage == "untrained":
            self.untrained = e
        elif stage == "cortex":
            steps, epoch, step = e["steps"], e["epoch"], e["step"]
            self.losses.append(((epoch - 1) * steps + step, e["loss"]))
            self.stage = "TRAINING the cortex: epoch %d of %d, step %d of %d" % (
                epoch, e["epochs"], step, steps)
            self.info.update(e)
        elif stage == "samples":
            self.samples = e["items"]
            for item in self.samples:
                if item["source"] not in self._thumbs:
                    self._wanted.put(item["source"])
        elif stage == "epoch":
            self.epochs.append(e)
        elif stage == "keep":
            self.keep = e
            self.stage = "choosing how much of the correction to keep (validation only)"
        elif stage == "memory":
            self.stage = "training the hippocampus and amygdala"
        elif stage == "done":
            self.done = e
            self.stage = ("DONE in %.1f min -- %s" % (e["minutes"],
                          "saved as v2b%s.pt; the dashboard still loads v2b.pt" % e["tag"]
                          if e.get("tag") else "the dashboard now loads this training"))

    # -- background helpers -------------------------------------------------------
    def _thumbnailer(self):
        from v2b import data
        while True:
            source = self._wanted.get()
            try:
                pixels = data.picture(source)
                if pixels is not None:
                    from PIL import Image
                    self._thumbs[source] = Image.fromarray(pixels).resize((84, 84), Image.NEAREST)
            except Exception:                                 # noqa: BLE001
                pass

    def _gpu_probe(self):
        while True:
            try:
                out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used,memory.total,utilization.gpu",
                                      "--format=csv,noheader,nounits"], capture_output=True,
                                     text=True, timeout=5,
                                     creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
                used, total, util = [x.strip() for x in out.stdout.split(",")]
                self._gpu = "%.1f / %.1f GB, %s%% busy" % (float(used) / 1024, float(total) / 1024, util)
            except Exception:                                 # noqa: BLE001
                self._gpu = "-"
            time.sleep(3)

    # -- drawing ----------------------------------------------------------------------
    def tick(self):
        self.poll()
        self.draw()
        self.root.after(400, self.tick)

    def text(self, x, y, s, colour=C.UI_FG, font=C.UI_FONT_SMALL, anchor="nw"):
        self.canvas.create_text(x, y, text=s, fill=colour, font=font, anchor=anchor)

    def draw(self):
        c = self.canvas
        c.delete("all")
        w = max(c.winfo_width(), 900)
        h = max(c.winfo_height(), 600)
        now = time.time()
        elapsed = (now - self.started) if self.started else 0
        self.text(16, 12, "v2.B TRAINING  //  LIVE", C.UI_ACCENT, C.UI_FONT_TITLE)
        self.text(16, 36, self.stage.upper(), C.UI_FG, C.UI_FONT_LABEL)
        eta = self.info.get("eta") if self.last.get("stage") == "cortex" else None
        self.text(w - 16, 12, "elapsed %s%s" % (_clock(elapsed),
                  "   cortex done in ~%s" % _clock(eta) if eta else ""),
                  C.UI_MID, C.UI_FONT_SMALL, "ne")
        self.text(w - 16, 30, "GPU %s" % self._gpu, C.UI_MID, C.UI_FONT_SMALL, "ne")
        top = 64
        # -- left column: encode bars or loss chart -----------------------------
        lw = int(w * 0.58)
        box = (16, top, lw, top + int(h * 0.42))
        if self.losses:
            self.loss_chart(box)
        else:
            self.encode_bars(box)
        # -- right column: accuracy per epoch --------------------------------------
        self.accuracy_chart((lw + 24, top, w - 16, top + int(h * 0.42)))
        # -- samples strip -----------------------------------------------------------
        y = top + int(h * 0.42) + 30
        self.samples_strip(16, y, w - 32)
        # -- stats / results -------------------------------------------------------------
        self.stats(16, y + 150, w - 32)

    def encode_bars(self, box):
        x0, y0, x1, y1 = box
        self.text(x0, y0, "ENCODING -- each picture through the frozen networks, once", C.UI_COOL)
        if not self.encode:
            self.text(x0, y0 + 24, "(waiting -- the first run after a data change encodes; later runs "
                      "reuse the cache)", C.UI_DIM, C.UI_FONT_TINY)
            return
        for i, (split, (done, total)) in enumerate(self.encode.items()):
            y = y0 + 26 + i * 26
            share = done / max(total, 1)
            self.text(x0, y, "%-12s" % split, C.UI_MID, C.UI_FONT_TINY)
            self.canvas.create_rectangle(x0 + 110, y + 2, x1 - 90, y + 14, outline=C.UI_EDGE_IDLE)
            self.canvas.create_rectangle(x0 + 110, y + 2, x0 + 110 + (x1 - 200 - x0) * share, y + 14,
                                         fill=C.UI_ACCENT, outline="")
            self.text(x1, y, "%s / %s" % (f"{done:,}", f"{total:,}"), C.UI_FG, C.UI_FONT_TINY, "ne")

    def loss_chart(self, box):
        x0, y0, x1, y1 = box
        c = self.canvas
        self.text(x0, y0, "LOSS -- how wrong it is on each training batch (lower is better)", C.UI_COOL)
        gx0, gy0, gx1, gy1 = x0 + 40, y0 + 22, x1, y1
        c.create_rectangle(gx0, gy0, gx1, gy1, outline=C.UI_EDGE_IDLE)
        steps = self.info.get("steps", 1) * self.info.get("epochs", 1)
        values = [v for _, v in self.losses]
        top = max(max(values), 1e-6)
        for frac in (0, 0.5, 1):
            y = gy1 - (gy1 - gy0) * frac
            self.text(gx0 - 6, y - 6, "%.2f" % (top * frac), C.UI_DIM, C.UI_FONT_TINY, "ne")
        pts, smooth, ema = [], [], None
        for step, v in self.losses:
            x = gx0 + (gx1 - gx0) * step / max(steps, 1)
            ema = v if ema is None else 0.9 * ema + 0.1 * v
            pts += [x, gy1 - (gy1 - gy0) * v / top]
            smooth += [x, gy1 - (gy1 - gy0) * ema / top]
        if len(pts) >= 4:
            c.create_line(*pts, fill=C.UI_ACCENT_DIM)
            c.create_line(*smooth, fill=C.UI_ACCENT, width=2)
        epochs = self.info.get("epochs", 1)
        for e in range(1, epochs):
            x = gx0 + (gx1 - gx0) * e / epochs
            c.create_line(x, gy0, x, gy1, fill=C.UI_EDGE_IDLE, dash=(2, 3))
        if values:
            self.text(gx1 - 4, gy0 + 4, "now %.3f" % ema, C.UI_ACCENT, C.UI_FONT_DATA, "ne")

    def accuracy_chart(self, box):
        x0, y0, x1, y1 = box
        c = self.canvas
        self.text(x0, y0, "ACCURACY after each epoch", C.UI_COOL)
        gx0, gy0, gx1, gy1 = x0 + 34, y0 + 22, x1, y1 - 34
        c.create_rectangle(gx0, gy0, gx1, gy1, outline=C.UI_EDGE_IDLE)
        for pct in (0, 50, 100):
            y = gy1 - (gy1 - gy0) * pct / 100
            self.text(gx0 - 6, y - 6, "%d%%" % pct, C.UI_DIM, C.UI_FONT_TINY, "ne")
        epochs = max(self.info.get("epochs", 6), 1)
        series = (("val_image", "held-out pictures", C.UI_ACCENT),
                  ("val_sound", "held-out sounds", C.UI_COOL),
                  ("open", "never-trained classes (shown only)", C.UI_GOOD))
        for i, (key, label, colour) in enumerate(series):
            pts = []
            if self.untrained and self.untrained.get(key) is not None:
                pts += [gx0, gy1 - (gy1 - gy0) * self.untrained[key] / 100]
            for e in self.epochs:
                if e.get(key) is None:
                    continue
                x = gx0 + (gx1 - gx0) * e["epoch"] / epochs
                y = gy1 - (gy1 - gy0) * e[key] / 100
                pts += [x, y]
                c.create_oval(x - 3, y - 3, x + 3, y + 3, fill=colour, outline="")
            if len(pts) >= 4:
                c.create_line(*pts, fill=colour, width=2)
            last = [e.get(key) for e in self.epochs if e.get(key) is not None]
            value = last[-1] if last else (self.untrained or {}).get(key)
            self.text(x0 + (i * (x1 - x0) / 3), gy1 + 8, "%s %s" % (
                label, "%.1f%%" % value if value is not None else "-"), colour, C.UI_FONT_TINY)

    def samples_strip(self, x, y, width):
        from PIL import ImageTk
        self.text(x, y - 20, "RIGHT NOW -- pictures from the batch being trained, and what v2.B says",
                  C.UI_COOL)
        if not self.samples:
            self.text(x, y + 10, "(appears once training starts)", C.UI_DIM, C.UI_FONT_TINY)
            return
        cell = min(int(width / 8) - 8, 128)
        for i, item in enumerate(self.samples[:8]):
            cx = x + i * (cell + 8)
            thumb = self._thumbs.get(item["source"])
            if thumb is not None:
                photo = ImageTk.PhotoImage(thumb.resize((min(cell, 84), min(cell, 84))))
                self._photos[i] = photo
                self.canvas.create_image(cx, y, image=photo, anchor="nw")
            else:
                self.canvas.create_rectangle(cx, y, cx + 84, y + 84, outline=C.UI_EDGE_IDLE)
            right = item["truth"] == item["guess"]
            self.text(cx, y + 88, item["truth"][:16], C.UI_MID, C.UI_FONT_TINY)
            self.text(cx, y + 102, ("%s %.0f%%" % (item["guess"][:13], 100 * item["p"])),
                      C.UI_GOOD if right else C.UI_ALERT, C.UI_FONT_TINY)

    def stats(self, x, y, width):
        i = self.info
        rows = []
        if "epoch" in i:
            rows += [("epoch", "%d / %d" % (i["epoch"], i["epochs"])),
                     ("step", "%d / %d" % (i["step"], i["steps"])),
                     ("examples seen", f"{i.get('seen', 0):,}"),
                     ("speed", "%.0f examples / s" % i.get("rate", 0)),
                     ("learning rate", "%.2e" % i.get("lr", 0))]
        if "train_images" in i:
            rows += [("training pictures", f"{i['train_images']:,}"),
                     ("training sounds", f"{i['train_sounds']:,}"),
                     ("names it can answer", f"{i.get('vocabulary', 0):,}")]
        for n, (label, value) in enumerate(rows):
            col, row = n % 4, n // 4
            self.text(x + col * (width / 4), y + row * 18, label, C.UI_MID, C.UI_FONT_TINY)
            self.text(x + col * (width / 4) + 150, y + row * 18, value, C.UI_FG, C.UI_FONT_DATA)
        if self.keep:
            self.text(x, y + 60, "kept %.2f of POT's learned correction (chosen on validation, so the "
                      "never-trained classes stay recognisable)" % self.keep["chosen"],
                      C.UI_COOL, C.UI_FONT_TINY)
        if self.done:
            t = self.done["test"]
            parts = ["CIFAR-100 test %.1f%%" % t["image"]["fused"],
                     "sounds %.1f%%" % t["audio"]["fused"]]
            if "tiny" in t:
                parts.append("Tiny-ImageNet test %.1f%%" % t["tiny"]["fused"])
            if self.done.get("open"):
                parts.append("never-trained classes %.1f%%" % self.done["open"]["fused"])
            self.text(x, y + 84, "FINAL (test sets):  " + "    ".join(parts), C.UI_GOOD, C.UI_FONT_LABEL)


def _clock(seconds) -> str:
    seconds = int(max(seconds or 0, 0))
    return "%d:%02d:%02d" % (seconds // 3600, seconds % 3600 // 60, seconds % 60)


def main() -> int:
    import tkinter as tk
    try:
        import ctypes
        ctypes.windll.shcore.SetProcessDpiAwareness(1)
    except Exception:                                         # noqa: BLE001
        pass
    root = tk.Tk()
    Monitor(root)
    root.mainloop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
