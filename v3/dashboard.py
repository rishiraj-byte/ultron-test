"""
dashboard.py -- watch v3 spike.

WHY v2's DASHBOARD COULD NOT BE REUSED. It draws rate-coded state: a bar per
unit, a brightness per lobe, a percentage per gate. v3 has none of those. Its
state is a set of binary events scattered in time, and the honest way to look
at that is the way electrophysiologists do -- a raster, one row per neuron,
one dot per spike.

WHAT EACH PANEL IS FOR.

  RASTER      The whole brain over 300 ms. Every dot is one cell firing once.
              Sparse cortex looks sparse here; a seizure looks like a wall.
              This is the panel that tells you whether the network is in the
              asynchronous irregular regime or has fallen out of it.

  MEMBRANE    Four individual cells, charging and discharging. Shows the leak,
              the threshold crossing, the reset, and the refractory gap -- the
              things that make this a neuron rather than a number.

  GATE        GPi firing rate against time, and what motor cortex does about
              it. Watch a decision be released rather than computed.

  RACE        The candidate actions competing. A decision is a population
              crossing threshold and staying there, not an argmax.

Nothing here is a simulation of the dashboard. Every dot is read from the same
tensors the model steps.
"""

from __future__ import annotations

import argparse
import queue
import threading

import torch

from v3 import config_v3 as cfg
from v3.brain_v3 import BrainV3
from v3.regions import Thalamus
from v3.transduce import features_to_rates, poisson_spikes

BG = "#05070a"
PANEL = "#080b10"
EDGE = "#1b2530"
DIM = "#3d4a5a"
MID = "#7e93a8"
FG = "#d8e4f0"
ACCENT = "#ff9020"
ALERT = "#ff4a28"
COOL = "#4fd8ff"
GREEN = "#57e08a"

FONT = ("Consolas", 10)
TINY = ("Consolas", 9)
TITLE = ("Consolas", 13, "bold")
LABEL = ("Consolas", 10, "bold")

#  Which populations get a raster row-block, in anatomical order, and how many
#  of their cells to draw. Drawing all 1024 of an area at this height would
#  put several neurons on the same pixel row and turn a sparse code into a
#  solid band, which is exactly the wrong impression.
RASTER_ROWS = [
    ("LGN relay",    "thal.LGN",      24, COOL),
    ("TRN",          "thal.TRN",      12, ALERT),
    ("occipital",    "occipital.exc", 40, ACCENT),
    ("occ PV",       "occipital.pv",  10, ALERT),
    ("occ SST",      "occipital.sst", 10, "#c86bff"),
    ("occ VIP",      "occipital.vip",  6, GREEN),
    ("parietal",     "parietal.exc",  40, ACCENT),
    ("temporal",     "temporal.exc",  40, ACCENT),
    ("frontal",      "frontal.exc",   40, ACCENT),
    ("striatum D1",  "bg.d1",         16, "#c86bff"),
    ("striatum D2",  "bg.d2",         16, "#c86bff"),
    ("GPi",          "bg.gpi",        16, ALERT),
    ("hippocampus",  "hip.ca1",       16, GREEN),
    ("amygdala",     "amygdala",      12, "#ff5f7e"),
    ("cerebellum",   "cb.deep",       12, GREEN),
    ("motor",        "motor",         32, FG),
]


class Recorder:
    """Collects spikes and voltages during a run, for drawing afterwards."""

    def __init__(self, brain: BrainV3):
        self.brain = brain
        self.sources = self._sources()
        self.reset()

    def _sources(self) -> dict:
        b = self.brain
        out = {"thal.TRN": b.thalamus.trn, "motor": b.motor,
               "amygdala": b.amygdala.cells, "hip.ca1": b.hippocampus.ca1,
               "cb.deep": b.cerebellum.deep,
               "bg.d1": b.basal_ganglia.d1, "bg.d2": b.basal_ganglia.d2,
               "bg.gpi": b.basal_ganglia.gpi}
        for nucleus, pop in b.thalamus.relay.items():
            out["thal.%s" % nucleus] = pop
        for name in b.LOBES:
            area = b.areas[name]
            out["%s.exc" % name] = area.exc
            out["%s.pv" % name] = area.pv
            out["%s.sst" % name] = area.sst
            out["%s.vip" % name] = area.vip
        return out

    def reset(self) -> None:
        self.spikes: dict[str, list] = {k: [] for k in self.sources}
        self.voltage: list = []
        self.gpi_hz: list = []
        self.motor_hz: list = []

    def capture(self) -> None:
        for key, pop in self.sources.items():
            self.spikes[key].append(pop.spikes[0].to(torch.bool).clone())
        #  Four occipital cells, followed individually.
        self.voltage.append(
            self.brain.areas["occipital"].exc.voltage[0, :4].clone())
        self.gpi_hz.append(float(self.brain.basal_ganglia.gpi.spikes.mean())
                           * 1000.0 / cfg.DT_MS)
        self.motor_hz.append(float(self.brain.motor.spikes.mean())
                             * 1000.0 / cfg.DT_MS)


class Dashboard:
    def __init__(self, root, device: str = "cpu"):
        import tkinter as tk

        self.tk = tk
        self.root = root
        self.device = device
        self.queue: queue.Queue = queue.Queue()
        self.brain = None
        self.recorder = None
        self.status = "loading the brain..."
        self.stimulus = 0
        self.forced = "rest"
        self.decision = None

        root.title("brain-ai v3  //  spiking cortex")
        root.configure(bg=BG)
        root.minsize(1180, 760)

        left = tk.Frame(root, bg=BG, width=230)
        left.pack(side="left", fill="y")
        left.pack_propagate(False)
        self._build_controls(left)

        right = tk.Frame(root, bg=BG)
        right.pack(side="left", fill="both", expand=True)

        self.raster = tk.Canvas(right, bg=PANEL, highlightthickness=1,
                                highlightbackground=EDGE)
        self.raster.pack(side="top", fill="both", expand=True, padx=8, pady=8)

        lower = tk.Frame(right, bg=BG, height=250)
        lower.pack(side="bottom", fill="x")
        lower.pack_propagate(False)
        self.trace = tk.Canvas(lower, bg=PANEL, highlightthickness=1,
                               highlightbackground=EDGE, width=430)
        self.trace.pack(side="left", fill="both", padx=(8, 4), pady=(0, 8))
        self.gate = tk.Canvas(lower, bg=PANEL, highlightthickness=1,
                              highlightbackground=EDGE, width=380)
        self.gate.pack(side="left", fill="both", padx=4, pady=(0, 8))
        self.race = tk.Canvas(lower, bg=PANEL, highlightthickness=1,
                              highlightbackground=EDGE)
        self.race.pack(side="left", fill="both", expand=True,
                       padx=(4, 8), pady=(0, 8))

        self._boot()
        root.after(120, self._tick)

    # -- panes ----------------------------------------------------------
    def _build_controls(self, parent) -> None:
        tk = self.tk

        tk.Label(parent, text="brain-ai v3", bg=BG, fg=ACCENT,
                 font=TITLE).pack(anchor="w", padx=12, pady=(14, 0))
        tk.Label(parent, text="spiking, no gradient", bg=BG, fg=DIM,
                 font=TINY).pack(anchor="w", padx=12, pady=(0, 12))

        self.status_label = tk.Label(parent, text=self.status, bg=BG, fg=MID,
                                     font=TINY, wraplength=205, justify="left")
        self.status_label.pack(anchor="w", padx=12, pady=(0, 12))

        tk.Label(parent, text="STIMULUS", bg=BG, fg=ACCENT,
                 font=LABEL).pack(anchor="w", padx=12, pady=(6, 2))
        row = tk.Frame(parent, bg=BG)
        row.pack(fill="x", padx=12)
        self.entry = tk.Entry(row, bg=PANEL, fg=FG, insertbackground=FG,
                              font=FONT, relief="flat", width=6,
                              highlightthickness=1, highlightbackground=EDGE)
        self.entry.insert(0, "0")
        self.entry.pack(side="left")
        self._button(row, "PERCEIVE", self._perceive).pack(side="left",
                                                           padx=(6, 0))

        tk.Label(parent, text="BASAL GANGLIA GATE", bg=BG, fg=ACCENT,
                 font=LABEL).pack(anchor="w", padx=12, pady=(18, 2))
        for label, mode in (("force OPEN", "open"), ("force SHUT", "shut"),
                            ("tonic (normal)", "rest")):
            self._button(parent, label,
                         lambda m=mode: self._force(m)).pack(fill="x", padx=12,
                                                             pady=2)

        self.readout = tk.Label(parent, text="", bg=BG, fg=MID, font=TINY,
                                justify="left", anchor="nw", wraplength=205)
        self.readout.pack(anchor="w", padx=12, pady=(18, 0), fill="both",
                          expand=True)

    def _button(self, parent, text, command):
        return self.tk.Button(parent, text=text, command=command, bg=PANEL,
                              fg=FG, activebackground="#241a10",
                              activeforeground=FG, font=FONT, relief="flat",
                              highlightthickness=1, highlightbackground=EDGE)

    # -- model ----------------------------------------------------------
    def _boot(self) -> None:
        def work():
            from v3.console import Console
            console = Console(self.device, prefer="developed")
            self.queue.put(("ready", console))
        threading.Thread(target=work, daemon=True).start()

    def _force(self, mode: str) -> None:
        if self.brain is None:
            return
        gpi = self.brain.basal_ganglia.gpi
        if mode == "open":
            gpi.threshold.fill_(0.0)
        elif mode == "shut":
            gpi.threshold.fill_(cfg.V_REST_MV + 2.0)
        else:
            gpi.threshold.copy_(self.resting)
        self.forced = mode
        self.status = "gate forced %s" % mode
        self._perceive()

    def _perceive(self) -> None:
        if self.brain is None:
            return
        try:
            self.stimulus = int(self.entry.get())
        except ValueError:
            self.stimulus = 0
        self.status = "settling 300 ms of simulated time..."
        self.status_label.configure(text=self.status)

        def work():
            try:
                self.queue.put(("done", self._run()))
            except Exception as failure:                      # noqa: BLE001
                self.queue.put(("error", str(failure)))
        threading.Thread(target=work, daemon=True).start()

    def _run(self):
        index = self.stimulus % len(self.labels)
        features = self.stimuli[index:index + 1].float()
        rates = features_to_rates(features, Thalamus.RELAY_SIZE)
        train = poisson_spikes(rates, cfg.STEPS)

        self.brain.reset(1, self.device)
        self.recorder.reset()
        for t in range(cfg.STEPS):
            self.brain.step({"LGN": train[t]}, learn=False)
            self.recorder.capture()

        motor = torch.stack([s.float().unsqueeze(0)
                             for s in self.recorder.spikes["motor"]])
        return {"index": index, "truth": int(self.labels[index]),
                "decision": self.brain.read_decision(motor)}

    # -- drawing --------------------------------------------------------
    def _tick(self) -> None:
        try:
            while True:
                kind, payload = self.queue.get_nowait()
                if kind == "ready":
                    console = payload
                    self.brain = console.brain
                    self.stimuli, self.labels = console.stimuli, console.labels
                    self.resting = self.brain.basal_ganglia.gpi.threshold.clone()
                    self.recorder = Recorder(self.brain)
                    self.status = "%d neurons ready" % sum(
                        p.size for p in self.brain.all_populations())
                    self._perceive()
                elif kind == "done":
                    self.decision = payload
                    self.status = "settled"
                    self._draw()
                elif kind == "error":
                    self.status = "failed: %s" % payload
        except queue.Empty:
            pass
        self.status_label.configure(text=self.status)
        self.root.after(150, self._tick)

    def _draw(self) -> None:
        self._draw_raster()
        self._draw_trace()
        self._draw_gate()
        self._draw_race()
        self._draw_readout()

    def _draw_raster(self) -> None:
        c = self.raster
        c.delete("all")
        w = max(c.winfo_width(), 100)
        h = max(c.winfo_height(), 100)
        left, top, bottom = 96, 30, h - 22

        c.create_text(10, 14, anchor="w", fill=ACCENT, font=LABEL,
                      text="SPIKE RASTER")
        c.create_text(w - 10, 14, anchor="e", fill=DIM, font=TINY,
                      text="one dot = one cell firing once   %d ms"
                           % int(cfg.PERCEPT_MS))

        total = sum(rows for _, _, rows, _ in RASTER_ROWS)
        span = max(bottom - top, 1)
        unit = span / max(total, 1)
        scale = (w - left - 16) / max(cfg.STEPS, 1)

        y = top
        for label, key, rows, colour in RASTER_ROWS:
            block = rows * unit
            c.create_text(left - 8, y + block / 2, anchor="e", fill=MID,
                          font=TINY, text=label)
            c.create_line(left, y + block, w - 16, y + block,
                          fill="#0e141c")
            history = self.recorder.spikes.get(key)
            if history:
                size = history[0].shape[0]
                step = max(size // rows, 1)
                for t, frame in enumerate(history):
                    fired = frame[::step][:rows].nonzero()
                    if fired.numel() == 0:
                        continue
                    x = left + t * scale
                    for r in fired.flatten().tolist():
                        yy = y + r * unit
                        c.create_line(x, yy, x + max(scale, 1.0), yy,
                                      fill=colour, width=max(unit * 0.7, 1.0))
            y += block

        for ms in range(0, int(cfg.PERCEPT_MS) + 1, 50):
            x = left + (ms / cfg.DT_MS) * scale
            c.create_line(x, top, x, bottom, fill="#0e141c")
            c.create_text(x, bottom + 10, fill=DIM, font=TINY, text="%d" % ms)

    def _draw_trace(self) -> None:
        c = self.trace
        c.delete("all")
        w, h = max(c.winfo_width(), 100), max(c.winfo_height(), 100)
        c.create_text(10, 14, anchor="w", fill=ACCENT, font=LABEL,
                      text="MEMBRANE POTENTIAL")
        c.create_text(10, 30, anchor="w", fill=DIM, font=TINY,
                      text="four occipital cells, mV")

        volts = torch.stack(self.recorder.voltage)       # (T, 4)
        top, bottom = 44, h - 26
        lo, hi = cfg.V_RESET_MV - 3, cfg.V_THRESHOLD_MV + 8
        scale_x = (w - 20) / max(cfg.STEPS, 1)

        def y_of(v):
            return bottom - (v - lo) / (hi - lo) * (bottom - top)

        c.create_line(10, y_of(cfg.V_THRESHOLD_MV), w - 10,
                      y_of(cfg.V_THRESHOLD_MV), fill=ALERT, dash=(3, 3))
        c.create_text(w - 12, y_of(cfg.V_THRESHOLD_MV) - 8, anchor="e",
                      fill=ALERT, font=TINY, text="threshold")
        c.create_line(10, y_of(cfg.V_REST_MV), w - 10, y_of(cfg.V_REST_MV),
                      fill="#16202c")

        for cell, colour in enumerate((ACCENT, COOL, GREEN, "#c86bff")):
            points = []
            for t in range(volts.shape[0]):
                points += [10 + t * scale_x, y_of(float(volts[t, cell]))]
            if len(points) >= 4:
                c.create_line(*points, fill=colour, width=1)

    def _draw_gate(self) -> None:
        c = self.gate
        c.delete("all")
        w, h = max(c.winfo_width(), 100), max(c.winfo_height(), 100)
        c.create_text(10, 14, anchor="w", fill=ACCENT, font=LABEL,
                      text="THE GATE")
        c.create_text(10, 30, anchor="w", fill=DIM, font=TINY,
                      text="GPi (red) holds the thalamus shut")

        top, bottom = 46, h - 26
        peak = max(max(self.recorder.gpi_hz + self.recorder.motor_hz), 1.0)
        scale_x = (w - 20) / max(cfg.STEPS, 1)

        for series, colour in ((self.recorder.gpi_hz, ALERT),
                               (self.recorder.motor_hz, FG)):
            points = []
            for t, v in enumerate(series):
                points += [10 + t * scale_x,
                           bottom - (v / peak) * (bottom - top)]
            if len(points) >= 4:
                c.create_line(*points, fill=colour, width=1)

        c.create_text(w - 10, bottom + 12, anchor="e", fill=DIM, font=TINY,
                      text="peak %.0f Hz   white = motor" % peak)

    def _draw_race(self) -> None:
        c = self.race
        c.delete("all")
        w, h = max(c.winfo_width(), 100), max(c.winfo_height(), 100)
        c.create_text(10, 14, anchor="w", fill=ACCENT, font=LABEL,
                      text="ACTION RACE")
        c.create_text(10, 30, anchor="w", fill=DIM, font=TINY,
                      text="a decision is a population holding threshold")

        if self.decision is None:
            return
        rates = self.decision["decision"]["rate_hz"][0]
        truth = self.decision["truth"]
        peak = max(float(rates.max()), cfg.MOTOR_THRESHOLD_HZ * 1.2)

        top = 48
        bar = min((h - top - 24) / max(len(rates), 1), 26)
        for i, value in enumerate(rates.tolist()):
            y = top + i * bar
            width = (w - 90) * (value / peak)
            colour = GREEN if i == truth else ACCENT
            c.create_rectangle(70, y + 2, 70 + max(width, 1), y + bar - 3,
                               fill=colour, outline="")
            c.create_text(62, y + bar / 2, anchor="e", fill=MID, font=TINY,
                          text="class %d" % i)
            c.create_text(74 + max(width, 1), y + bar / 2, anchor="w",
                          fill=DIM, font=TINY, text="%.0f" % value)

        x = 70 + (w - 90) * (cfg.MOTOR_THRESHOLD_HZ / peak)
        c.create_line(x, top, x, top + len(rates) * bar, fill=ALERT,
                      dash=(3, 3))
        c.create_text(x + 4, top - 8, anchor="w", fill=ALERT, font=TINY,
                      text="threshold")

    def _draw_readout(self) -> None:
        if self.decision is None:
            return
        d = self.decision["decision"]
        action = int(d["action"][0])
        truth = self.decision["truth"]
        lines = [
            "stimulus %d" % self.decision["index"],
            "true class %d" % truth,
            "",
            "gate: %s" % self.forced,
            "GPi mean %.1f Hz" % (sum(self.recorder.gpi_hz)
                                  / max(len(self.recorder.gpi_hz), 1)),
            "",
        ]
        if action < 0:
            lines += ["DECISION: none", "",
                      "No motor population held",
                      "threshold. An undecided",
                      "brain emits nothing --",
                      "that is an outcome, not",
                      "a crash."]
        else:
            lines += ["DECISION: class %d" % action,
                      "(%s)" % ("correct" if action == truth else "wrong"),
                      "peak %.1f Hz" % float(d["rate_hz"].max())]
        self.readout.configure(text="\n".join(lines))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    import tkinter as tk
    root = tk.Tk()
    Dashboard(root, args.device)
    root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
