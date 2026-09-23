"""
desktop_app_v2.py -- the v2 console: talk to it, and watch it think.

WHY THIS IS A NEW FILE RATHER THAN AN EDIT OF THE v1 DASHBOARD.

The v1 dashboard draws a PIPELINE: input enters, passes through the thalamus,
through four lobes, out. That was accurate then and is wrong now. There is no
longer a pass to follow -- cortex settles over T timesteps with every lobe
updating simultaneously from delayed copies of the others, subcortical
structures running alongside on their own clocks, and the whole thing
converging rather than finishing. A left-to-right diagram would have to lie
about the one property that took the most work to establish.

THE LAYOUT is the original dashboard: input on the left, every network drawn
at once in the centre, live system state on the right, and the conversation
across the bottom.

    +-------------+---------------------------+--------------+
    |  INPUT      |   NETWORKS (all at once)  |  SYSTEM      |
    |  text       |   thalamus + 4 lobes + BG |  live state  |
    |  image      |   live layer diagrams     |  CPU / GPU   |
    +-------------+---------------------------+--------------+
    |          OUTPUT -- chatbot style, text only            |
    +--------------------------------------------------------+

Each network is drawn by layerview.py as input, hidden and output columns of
circles, brightness standing for real activation captured by forward hooks
during the settle. Drawing all of them simultaneously is not a layout
preference: this architecture has no forward pass, and showing one network at
a time would imply a sequence that does not exist. Click a tile to expand it.

The ANATOMICAL VIEW (brainview.py) is still available from the left pane. It
answers a different question -- where activity is, rather than what each
network is doing -- and neither view substitutes for the other.

THE CHAT IS NOT A WRAPPER AROUND A LANGUAGE MODEL. What you type is encoded,
routed through the thalamus, settled on by four lobes, gated by the basal
ganglia, and read out through the semantic head. Qwen is handed that structured
report and never sees your text. That boundary is the entire point: if the
language model read the input directly it would answer well and the four lobes
would be decoration.

THE THREADING RULE, unchanged from v1: inference runs on a worker thread and
pushes finished traces to a queue; the UI drains that queue on its own timer
and does all drawing. Nothing renders inside the settling loop, so the
dashboard's redraw rate can never become the model's speed.

    python desktop_app_v2.py              open the console
    python desktop_app_v2.py --selftest   run the engine once and print
                                          everything as text, no window
"""

from __future__ import annotations

import argparse
import queue
import sys
import threading
from pathlib import Path

import config


def _video_thumbnails(path, count: int = config.VIDEO_FRAMES) -> list:
    """The evenly spaced frames a percept would sample, as PIL images.

    Deliberately mirrors BrainV2._video_frames rather than calling it: this
    runs on the UI thread while the models may still be loading, and a preview
    must never be the thing that needs a GPU.
    """
    try:
        import cv2
        import numpy as np
        from PIL import Image
    except Exception:                                         # noqa: BLE001
        return []

    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        return []
    total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
    frames = []
    for index in np.linspace(0, max(total - 1, 0), count, dtype=int):
        capture.set(cv2.CAP_PROP_POS_FRAMES, int(index))
        ok, frame = capture.read()
        if ok:
            frames.append(Image.fromarray(frame[:, :, ::-1]))
    capture.release()
    return frames


def media_support() -> dict[str, str]:
    """Which input types this machine can actually decode, and with what.

    Reported in the UI at startup rather than discovered by crashing. The
    audio path failed here for a while and the error read like a missing
    install when the install was fine and the FORMAT was unsupported --
    scipy handles uncompressed WAV and nothing else, so every .mp3 fell
    through the whole loader chain. A capability line makes the difference
    between "not installed" and "installed but limited" visible before the
    user picks a file rather than after.
    """
    import importlib

    def present(*names: str) -> list[str]:
        found = []
        for name in names:
            try:
                importlib.import_module(name)
                found.append(name.split(".")[0])
            except Exception:                                 # noqa: BLE001
                pass
        return found

    text = ["tokeniser"]
    image = present("PIL")
    audio = present("soundfile", "torchaudio", "imageio_ffmpeg", "scipy.io.wavfile")
    video = present("cv2", "imageio")

    def line(found: list[str], limited: str = "") -> str:
        if not found:
            return "UNAVAILABLE"
        if limited and found == [limited]:
            return "%s (uncompressed only)" % limited
        return "+".join(found[:3])

    return {"text": line(text), "image": line(image),
            "audio": line(audio, limited="scipy"), "video": line(video)}


def _resample(wave, rate: int, target: int):
    """Linear resample to `target` Hz. A no-op when the rates already match.

    Linear rather than a windowed-sinc: the waveform's next stop is a mel
    spectrogram with 25 ms windows, which discards the high-frequency detail a
    better kernel would have preserved. Correct rate matters enormously here;
    interpolation quality barely registers.
    """
    import numpy as np

    if rate == target or wave.size == 0:
        return wave
    count = max(int(round(wave.size * target / float(rate))), 1)
    source = np.linspace(0.0, wave.size - 1, wave.size, dtype="float64")
    wanted = np.linspace(0.0, wave.size - 1, count, dtype="float64")
    return np.interp(wanted, source, wave).astype("float32")


def _read_audio(path):
    """Load a waveform as float32 mono, using whatever is installed.

    Tried in order rather than depending on one package: soundfile is the
    obvious choice and is NOT a dependency of anything else here, so requiring
    it turned "choose an audio file" into an import error. torchaudio ships
    with torch, and scipy handles WAV without either.
    """
    import numpy as np

    #  THE RATE MATTERS AS MUCH AS THE SAMPLES. Every decoder returns audio at
    #  the file's OWN rate, and the auditory front end expects 16 kHz; a 44.1
    #  kHz file handed over unresampled is the same waveform played 2.76x too
    #  fast, which is not a subtle degradation -- it moves every formant. This
    #  went unnoticed while only scipy worked, because the test files were all
    #  16 kHz WAV. The moment real .mp3s decoded, they decoded wrong.
    rate = config.AUDIO_SAMPLE_RATE
    for loader in ("soundfile", "torchaudio", "scipy", "ffmpeg"):
        try:
            if loader == "soundfile":
                import soundfile
                wave, rate = soundfile.read(str(path), dtype="float32")
            elif loader == "torchaudio":
                import torchaudio
                tensor, rate = torchaudio.load(str(path))
                wave = tensor.mean(dim=0).numpy()
            elif loader == "scipy":
                from scipy.io import wavfile
                rate, raw = wavfile.read(str(path))
                wave = raw.astype("float32")
                if wave.size and abs(wave).max() > 1.0:
                    wave = wave / 32768.0
            else:
                #  LAST RESORT, and the one that actually covers the formats
                #  people have. scipy reads uncompressed WAV and nothing else,
                #  so before this an .mp3 or .m4a failed every loader in the
                #  chain and the system reported "no audio loader available"
                #  -- which read as a missing install when in fact the install
                #  was fine and the FORMAT was unsupported. ffmpeg decodes
                #  essentially anything, and imageio-ffmpeg ships the binary.
                import subprocess
                import imageio_ffmpeg
                done = subprocess.run(
                    [imageio_ffmpeg.get_ffmpeg_exe(), "-nostdin",
                     "-loglevel", "error", "-i", str(path), "-vn", "-ac", "1",
                     "-ar", str(config.AUDIO_SAMPLE_RATE), "-f", "f32le", "-"],
                    capture_output=True, timeout=120)
                if done.returncode != 0 or not done.stdout:
                    raise RuntimeError("ffmpeg could not decode this file")
                #  ffmpeg was told the target rate on the command line, so
                #  this branch is already correct.
                wave = np.frombuffer(done.stdout, dtype="<f4")
                rate = config.AUDIO_SAMPLE_RATE
        except Exception:                                     # noqa: BLE001
            continue
        wave = np.asarray(wave, dtype="float32")
        if wave.ndim > 1:
            wave = wave.mean(axis=1)
        return _resample(wave, int(rate), config.AUDIO_SAMPLE_RATE)
    raise RuntimeError(
        "could not decode %s -- no installed loader handled this format. "
        "Install one:  pip install soundfile imageio-ffmpeg"
        % Path(path).name)


# ---------------------------------------------------------------------------
# Engine -- no UI in it, so it can be tested with no display
# ---------------------------------------------------------------------------


class Engine:
    """Loads the trained system and runs one percept at a time."""

    def __init__(self, device: str | None = None):
        import brain_v2

        self.system = brain_v2.build(device=device)
        self.cortex = self.system.cortex
        self.history: list[tuple[str, str]] = []
        self.lesioned: set[str] = set()

    @property
    def trained(self) -> bool:
        return self.system.trained

    @property
    def training_record(self) -> dict:
        return self.system.training_record

    def tract_names(self) -> list[str]:
        return sorted(self.cortex.tracts.names())

    def lesion(self, name: str, cut: bool) -> None:
        self.cortex.tracts[name].lesion(cut)
        (self.lesioned.add if cut else self.lesioned.discard)(name)

    def think(self, text: str, image=None, audio=None, video=None,
              layers=None):
        """One percept. If a LayerView is supplied it records every timestep.

        The recording is what turns the diagrams from a still photograph into
        the process itself: the loop runs T times, every module fires T times,
        and a forward hook keeps only the last unless something reads it in
        between.
        """
        if layers is not None:
            layers.start_recording()
            on_step = layers.record
        else:
            on_step = None
        return self.system.perceive(text=text or None, image=image,
                                    audio=audio, video=video, on_step=on_step)

    def cortex_answer(self, thought) -> str:
        """What the four lobes concluded, in their own words. No Qwen.

        Shown before the spoken reply because it is the part of the output
        this system actually built. Everything in it is a formatting of a
        number the lobes produced.
        """
        if thought.trace is None:
            return "nothing perceived"
        return self.system.report(thought).as_answer()

    def speak(self, thought, on_token=None) -> str:
        #  "silent" means the cortex is the whole output and the language
        #  model never runs -- worth having as a real option, because it is
        #  the only configuration in which nothing borrowed touches the reply.
        if config.OUTPUT_MODE == "silent":
            return ""
        reply = self.system.say(thought, history=self.history,
                                on_token=on_token)
        self.history.append(("user", thought.text))
        self.history.append(("assistant", reply))
        return reply

    def readouts(self, thought) -> list[tuple[str, list[tuple[str, float]]]]:
        """Everything the right-hand column shows, as plain numbers."""
        trace = thought.trace
        groups: list[tuple[str, list[tuple[str, float]]]] = []
        if trace is None:
            return groups

        #  With the language model reading the user's text directly, the
        #  reply no longer proves the cortex was involved. This panel is
        #  where that claim now lives, so it goes first.
        contribution = self.system.contribution(thought)
        groups.append(("cortex contributed", [
            ("recognised", 1.0 if contribution["recognised"] else 0.0),
            ("reached the reply", 1.0 if contribution["in_report"] else 0.0),
            ("top similarity", contribution["top_similarity"]),
            ("confidence", contribution["confidence"]),
        ]))
        #  WHAT THE SENSES ACTUALLY DELIVERED. Worth its own group because
        #  "I loaded a video and nothing happened" and "I loaded a video, it
        #  decoded 8 frames and a soundtrack, and the answer is still wrong"
        #  are completely different problems, and the UI could not previously
        #  tell them apart.
        senses = [(name, 1.0 if name in thought.modalities else 0.0)
                  for name in ("text", "image", "audio", "video")]
        if thought.frames_read:
            senses.append(("video frames", float(thought.frames_read)))
        groups.append(("senses delivered", senses))

        groups.append(("convergence", [
            ("settling ratio", thought.settling_ratio),
            ("settled", 1.0 if thought.settled else 0.0),
            ("elapsed ms", thought.elapsed_ms),
        ]))
        if trace.gates:
            groups.append(("thalamic gates", sorted(trace.gates[-1].items())))
        if trace.bg:
            decision = trace.bg[-1]
            groups.append(("basal ganglia", [
                ("gate openness", decision.openness),
                ("go (D1)", float(decision.go.mean())),
                ("nogo (D2)", float(decision.nogo.mean())),
                ("dopamine", float(decision.dopamine.mean())),
            ]))
        if trace.amygdala:
            evaluation = trace.amygdala[-1]
            groups.append(("amygdala", [
                ("salience", float(evaluation.salience.mean())),
                ("valence", float(evaluation.valence.mean())),
            ]))
        if trace.novelty:
            groups.append(("hippocampus", [
                ("novelty", trace.novelty[-1]),
                ("DG sparsity", self.cortex.hippocampus.dg.achieved_sparsity),
            ]))
        if trace.cerebellum:
            errors = [p.error_value for p in trace.cerebellum]
            groups.append(("cerebellum", [
                ("prediction error", errors[-1]),
                ("fell from", max(errors)),
            ]))
        if trace.modulators:
            state = trace.modulators[-1]
            groups.append(("neuromodulators", [
                ("acetylcholine", state.acetylcholine),
                ("norepinephrine", state.norepinephrine),
                ("serotonin", state.serotonin),
            ]))
        return groups


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------


def selftest() -> int:
    """The whole compute path, printed. No window, no display needed."""
    print("=" * 74)
    print("CONSOLE SELF-TEST -- engine only, no window")
    print("=" * 74)

    engine = Engine()
    print("trained: %s" % engine.trained)
    if engine.training_record:
        print("record : %s" % "  ".join(
            "%s=%s" % kv for kv in sorted(engine.training_record.items())))
    else:
        print("record : NONE -- the lobes are random and answers are noise")
    print("tracts : %d" % len(engine.tract_names()))

    for prompt in ("what is a spacecraft made of",
                   "my computer keeps crashing when I open the browser",
                   "the doctor said the treatment was working"):
        thought = engine.think(prompt)
        print()
        print("-- %r" % prompt)
        print("   concepts   : %s" % ", ".join(
            "%s %.2f" % c for c in thought.concepts[:4]) or "none")
        print("   classifier : %s" % ", ".join(
            "%s %.0f%%" % (n, 100 * p) for n, p in thought.predictions[:3])
            or "none")
        print("   %s" % thought.summary())
        for group, rows in engine.readouts(thought)[:3]:
            print("   %-16s %s" % (group, "  ".join(
                "%s %.3f" % r for r in rows)))

    print()
    print("-- the LAYER DIAGRAM must render every network with live values --")
    print("   (input / hidden / output columns, circles for units. This is the")
    print("    default view; the anatomical map is the alternative, not the")
    print("    replacement.)")
    from layerview import LayerView, NETWORKS
    layers = LayerView(engine.cortex)
    probe = engine.think("a red bicycle leaning on a wall", layers=layers)

    print()
    print("   IS IT REALTIME? The loop runs %d times and every module fires"
          % config.T_STEPS)
    print("   %d times. A forward hook keeps only the last one unless"
          % config.T_STEPS)
    print("   something reads it in between, so this checks that each step")
    print("   was actually captured and that the values genuinely differ.")
    frames = layers.frames
    print("   frames recorded: %d of %d" % (len(frames), config.T_STEPS))
    if len(frames) >= 2:
        import numpy as np
        active = [int((f["frontal.state"] != 0).sum()) for f in frames]
        print("   frontal active units per timestep: %s" % active)
        moved = sum(1 for t in range(len(frames) - 1)
                    if not np.array_equal(frames[t]["frontal.state"],
                                          frames[t + 1]["frontal.state"]))
        print("   timesteps where the state changed: %d of %d"
              % (moved, len(frames) - 1))
    print()
    for network in NETWORKS:
        image, columns = layers.render(network, 620, 260, probe.trace,
                                       step=config.T_STEPS - 1)
        live = [c for c in columns if c.active is not None]
        print("   %-14s %d columns, %d with live activity"
              % (network, len(columns), len(live)))
        for column in columns[:2]:
            print("        %-24s %s of %d units, %d circles"
                  % (column.spec.label,
                     "-" if column.active is None else column.active,
                     column.total, column.shown))
    layers.remove()

    print()
    print("-- the ROUTING view must carry every tract, forward and back --")
    print("   (a routing diagram that quietly drops the return paths would")
    print("    be drawing a pipeline, which is the thing this architecture")
    print("    is not)")
    from flowview import FlowRenderer, NODES
    flow = FlowRenderer(engine.cortex)
    print("   nodes: %d   tracts drawn: %d of %d"
          % (len(NODES), len(flow.edges), len(engine.tract_names())))
    #  flow.edges holds (name, source node, target node) -- the endpoints are
    #  resolved through the node map, so a tract into thal.LGN is drawn into
    #  the single thalamus node. Treating an entry as a bare tract name is
    #  what crashed this on startup.
    backward = [name for name, source, target in flow.edges
                if NODES[target][0] < NODES[source][0]]
    print("   return / feedback paths drawn: %d  (%s)"
          % (len(backward), ", ".join(sorted(backward)[:4])))
    for step in (0, config.T_STEPS // 2, config.T_STEPS - 1):
        carrying = flow.signal(probe.trace, step)
        busiest = sorted(carrying.items(), key=lambda kv: -kv[1])[:3]
        print("   t=%-2d busiest paths: %s"
              % (step, ", ".join("%s %.2f" % b for b in busiest)))

    print()
    print("-- the brain view must have every region and real edges --")
    from brainview import REGIONS, BrainRenderer
    renderer = BrainRenderer(engine.cortex)
    print("   regions drawn : %d" % len(REGIONS))
    print("   tracts drawn  : %d of %d (both ends on the map)"
          % (len(renderer.edges), len(engine.tract_names())))
    thought = engine.think("a quick test of the live view")
    #  The NORMALISED frames, which is what actually gets drawn. Printing the
    #  raw energies here showed the cerebellum winning every frame purely
    #  because its number is in different units from a lobe's.
    frames = renderer.normalised(thought.trace)
    for t in (0, config.T_STEPS // 2, config.T_STEPS - 1):
        top = sorted(frames[t].items(), key=lambda kv: -kv[1])[:3]
        print("   t=%-2d brightest: %s" % (t, ", ".join(
            "%s %.2f" % e for e in top)))

    print()
    print("Engine OK.  python desktop_app_v2.py  opens the console.")
    return 0


# ---------------------------------------------------------------------------
# The window
# ---------------------------------------------------------------------------


def _bracket(canvas, x0, y0, x1, y1, colour, arm=config.UI_BRACKET):
    for (cx, cy, dx, dy) in ((x0, y0, 1, 1), (x1, y0, -1, 1),
                             (x0, y1, 1, -1), (x1, y1, -1, -1)):
        canvas.create_line(cx, cy, cx + arm * dx, cy, fill=colour)
        canvas.create_line(cx, cy, cx, cy + arm * dy, fill=colour)


class Console:
    """The v1 dashboard layout, driven by the v2 brain.

        +-------------+---------------------------+--------------+
        |  INPUT      |   NETWORKS (all at once)  |  SYSTEM      |
        |  text       |   thalamus + 4 lobes + BG |  live state  |
        |  image      |   live layer diagrams     |  CPU / GPU   |
        |  audio      |   click one to expand     |  active/idle |
        +-------------+---------------------------+--------------+
        |          OUTPUT -- chatbot style, text only            |
        +--------------------------------------------------------+

    Every network is drawn at once rather than one at a time, and that is not
    a layout preference. This architecture has no forward pass; showing a
    single network at a time would imply a sequence that does not exist.
    """

    def __init__(self, root):
        import tkinter as tk

        self.tk = tk
        self.root = root
        self.queue: queue.Queue = queue.Queue()
        self.engine = None
        self.layers = None
        self.renderer = None
        self.flow = None
        self.thought = None
        self.readouts: list = []
        self.frame = 0
        self.playing = False
        self.status = "loading the frozen backbones..."
        #  ROUTING is the view the dashboard opens on. It answers the
        #  question people actually have about this architecture -- which
        #  nucleus feeds which lobe, and which paths run backwards.
        self.expanded: str | None = "routing"
        self._photos: dict = {}
        self._tiles: dict = {}
        self.image_path: str | None = None
        self.audio_path: str | None = None
        self.video_path: str | None = None
        #  Rotation of the 3D anatomical view.
        self.angle = 0.0
        self.spinning = True
        self._drag_from: float | None = None

        root.title("brain-ai  //  v2")
        root.configure(bg=config.UI_BG)
        root.minsize(config.UI_MIN_WIDTH, config.UI_MIN_HEIGHT)

        #  Telemetry runs on its own thread and is sampled by the UI timer,
        #  never polled inside a draw.
        from modules.telemetry import TelemetryMonitor
        self.telemetry = TelemetryMonitor()
        self.telemetry.start()

        #  Bottom strip is packed FIRST so it keeps its height when the
        #  centre panel grows.
        bottom = tk.Frame(root, bg=config.UI_BG, height=200)
        bottom.pack(side="bottom", fill="x")
        bottom.pack_propagate(False)
        self._build_output(bottom)

        upper = tk.Frame(root, bg=config.UI_BG)
        upper.pack(side="top", fill="both", expand=True)

        left = tk.Frame(upper, bg=config.UI_BG,
                        width=int(config.UI_MIN_WIDTH
                                  * config.UI_INPUT_PANE_FRACTION))
        left.pack(side="left", fill="y")
        left.pack_propagate(False)
        self._build_input(left)

        self.side = tk.Canvas(upper, bg=config.UI_BG, highlightthickness=0,
                              width=290)
        self.side.pack(side="right", fill="y")

        self.canvas = tk.Canvas(upper, bg=config.UI_BG, highlightthickness=0)
        self.canvas.pack(side="left", fill="both", expand=True)
        self.canvas.bind("<Button-1>", self._click_network)
        self.canvas.bind("<B1-Motion>", self._drag_rotate)
        self.canvas.bind("<ButtonRelease-1>", self._drag_end)

        self._boot()
        root.after(config.UI_REFRESH_MS, self._tick)

    # -- panes -------------------------------------------------------------
    def _build_input(self, parent) -> None:
        tk = self.tk
        tk.Label(parent, text="INPUT", bg=config.UI_BG, fg=config.UI_ACCENT,
                 font=config.UI_FONT_TITLE).pack(anchor="w", padx=12,
                                                 pady=(14, 2))
        tk.Label(parent, text="any combination, or none",
                 bg=config.UI_BG, fg=config.UI_DIM,
                 font=config.UI_FONT_TINY).pack(anchor="w", padx=12)

        self.text_box = tk.Text(parent, height=6, bg=config.UI_PANEL,
                                fg=config.UI_FG, insertbackground=config.UI_FG,
                                font=config.UI_FONT_SMALL, relief="flat",
                                wrap="word", highlightthickness=1,
                                highlightbackground=config.UI_EDGE_IDLE)
        self.text_box.pack(padx=12, pady=8, fill="x")
        self.text_box.insert("1.0", "a red bicycle leaning on a wall")

        #  What this machine can decode, stated up front. See media_support.
        support = media_support()
        self._support_line = "  ".join(
            "%s:%s" % (kind.upper()[:3], how)
            for kind, how in support.items())

        self.image_label = tk.Label(parent, text=self._support_line,
                                    bg=config.UI_BG, fg=config.UI_DIM,
                                    font=config.UI_FONT_TINY,
                                    justify="left", anchor="w")
        for label, command in (("IMAGE...", self._choose_image),
                               ("AUDIO...", self._choose_audio),
                               ("VIDEO...", self._choose_video),
                               ("clear media", self._clear_media),
                               ("PERCEIVE", self._send),
                               ("replay settling", self._replay),
                               ("lesions...", self._show_lesions)):
            tk.Button(parent, text=label, command=command,
                      bg=config.UI_PANEL, fg=config.UI_FG,
                      activebackground=config.UI_ACCENT_DIM,
                      activeforeground=config.UI_FG,
                      font=config.UI_FONT_SMALL, relief="flat",
                      highlightthickness=1,
                      highlightbackground=config.UI_EDGE_IDLE).pack(
                padx=12, pady=3, fill="x")
        self.image_label.pack(anchor="w", padx=12, pady=(4, 4))

        #  WHAT THE SYSTEM ACTUALLY DECODED, shown before it is asked to
        #  perceive anything. A filename proves a file was chosen; it does not
        #  prove the file was readable, and the two failures look identical
        #  from the outside. For a clip this draws the real sampled frames in
        #  order, which is also the only way to see that a video is being
        #  read as a sequence rather than one picture.
        self.preview = tk.Canvas(parent, bg=config.UI_PANEL, height=52,
                                 highlightthickness=1,
                                 highlightbackground=config.UI_EDGE_IDLE)
        self.preview.pack(fill="x", padx=12, pady=(0, 10))
        self._preview_photos: list = []

        tk.Label(parent, text="VIEW", bg=config.UI_BG, fg=config.UI_ACCENT,
                 font=config.UI_FONT_TITLE).pack(anchor="w", padx=12,
                                                 pady=(10, 2))
        for label, target in (("routing", "routing"),
                              ("all networks", None),
                              ("anatomy 3D", "anatomy")):
            tk.Button(parent, text=label,
                      command=lambda t=target: self._expand(t),
                      bg=config.UI_PANEL, fg=config.UI_MID,
                      activebackground=config.UI_ACCENT_DIM,
                      activeforeground=config.UI_FG,
                      font=config.UI_FONT_SMALL, relief="flat",
                      highlightthickness=1,
                      highlightbackground=config.UI_EDGE_IDLE).pack(
                padx=12, pady=2, fill="x")

    def _build_output(self, parent) -> None:
        tk = self.tk
        tk.Label(parent, text="OUTPUT", bg=config.UI_BG, fg=config.UI_ACCENT,
                 font=config.UI_FONT_TITLE).pack(anchor="w", padx=14,
                                                 pady=(6, 0))
        self.transcript = tk.Text(parent, bg=config.UI_PANEL, fg=config.UI_FG,
                                  font=config.UI_FONT_SMALL, relief="flat",
                                  wrap="word", height=6, highlightthickness=1,
                                  highlightbackground=config.UI_EDGE_IDLE)
        self.transcript.pack(padx=14, pady=(4, 4), fill="both", expand=True)
        self.transcript.tag_configure("you", foreground=config.UI_COOL)
        self.transcript.tag_configure("brain", foreground=config.UI_FG)
        self.transcript.tag_configure("note", foreground=config.UI_DIM)
        #  The cortex's own answer is the system's own work, so it gets the
        #  accent colour and the bold face; the borrowed voice below it is
        #  set in plain foreground. The visual hierarchy should match where
        #  the computation actually happened.
        self.transcript.tag_configure("cortex", foreground=config.UI_ACCENT,
                                      font=config.UI_FONT_DATA)
        self.transcript.configure(state="disabled")

        row = tk.Frame(parent, bg=config.UI_BG)
        row.pack(fill="x", padx=14, pady=(0, 8))
        self.entry = tk.Entry(row, bg=config.UI_PANEL, fg=config.UI_FG,
                              insertbackground=config.UI_FG,
                              font=config.UI_FONT_SMALL, relief="flat",
                              highlightthickness=1,
                              highlightbackground=config.UI_EDGE_IDLE)
        self.entry.pack(side="left", fill="x", expand=True)
        self.entry.bind("<Return>", lambda e: self._send())
        tk.Button(row, text="SEND", command=self._send, bg=config.UI_PANEL,
                  fg=config.UI_FG, activebackground=config.UI_ACCENT_DIM,
                  activeforeground=config.UI_FG, font=config.UI_FONT_SMALL,
                  relief="flat", highlightthickness=1,
                  highlightbackground=config.UI_EDGE_IDLE).pack(
            side="left", padx=(6, 0))

    # -- actions -----------------------------------------------------------
    def _choose_image(self) -> None:
        from tkinter import filedialog
        path = filedialog.askopenfilename(
            filetypes=[("images", "*.png *.jpg *.jpeg *.bmp"), ("all", "*.*")])
        if path:
            self.image_path = path
            self._media_label()

    def _choose_audio(self) -> None:
        from tkinter import filedialog
        path = filedialog.askopenfilename(
            filetypes=[("audio", "*.wav *.mp3 *.flac *.ogg *.m4a"),
                       ("all", "*.*")])
        if path:
            self.audio_path = path
            self._media_label()

    def _choose_video(self) -> None:
        from tkinter import filedialog
        path = filedialog.askopenfilename(
            filetypes=[("video", "*.mp4 *.avi *.mov *.mkv *.webm"),
                       ("all", "*.*")])
        if path:
            self.video_path = path
            self._media_label()

    def _clear_media(self) -> None:
        self.image_path = self.audio_path = self.video_path = None
        self._media_label()

    def _media_label(self) -> None:
        parts = []
        for tag, path in (("img", self.image_path), ("aud", self.audio_path),
                          ("vid", self.video_path)):
            if path:
                parts.append("%s:%s" % (tag, Path(path).name[:12]))
        self.image_label.configure(text="  ".join(parts) or self._support_line)
        self._draw_preview()

    def _draw_preview(self) -> None:
        """Draw whatever media is currently loaded, decoded for real.

        Decoding here rather than trusting the extension means a file this
        machine cannot read says so NOW, on the pane, instead of at the end of
        a percept. Failures are drawn as a message rather than raised: this is
        a preview, and it must not be able to take the window down.
        """
        canvas = getattr(self, "preview", None)
        if canvas is None:
            return                      # called before the pane was built
        canvas.delete("all")
        self._preview_photos = []
        width = max(canvas.winfo_width(), 1)

        def say(message: str, colour=config.UI_DIM) -> None:
            canvas.create_text(8, 26, anchor="w", fill=colour,
                               font=config.UI_FONT_TINY, text=message)

        if not (self.image_path or self.audio_path or self.video_path):
            say("no media loaded -- text only")
            return

        from PIL import Image, ImageTk
        try:
            if self.video_path:
                #  Decoded here rather than through the engine, so the preview
                #  works while the models are still loading -- which is
                #  exactly when someone is picking their files.
                frames = _video_thumbnails(self.video_path)
                if not frames:
                    say("could not decode this video", config.UI_ALERT)
                    return
                #  Drawn left to right in time order, because that ordering is
                #  the thing the change to video processing was FOR.
                size = 44
                for index, frame in enumerate(frames):
                    thumb = frame.copy()
                    thumb.thumbnail((size, size))
                    photo = ImageTk.PhotoImage(thumb)
                    self._preview_photos.append(photo)
                    canvas.create_image(6 + index * (size + 4), 4,
                                        anchor="nw", image=photo)
                canvas.create_text(width - 6, 46, anchor="se",
                                   fill=config.UI_MID,
                                   font=config.UI_FONT_TINY,
                                   text="%d frames, in order" % len(frames))
                return

            if self.image_path:
                thumb = Image.open(self.image_path).convert("RGB")
                thumb.thumbnail((44, 44))
                photo = ImageTk.PhotoImage(thumb)
                self._preview_photos.append(photo)
                canvas.create_image(6, 4, anchor="nw", image=photo)

            if self.audio_path:
                wave = _read_audio(self.audio_path)
                seconds = len(wave) / float(config.AUDIO_SAMPLE_RATE)
                #  A waveform, drawn from the samples that will actually be
                #  handed to the auditory encoder -- so a file that decoded to
                #  silence is visibly flat rather than quietly wrong.
                left = 60 if self.image_path else 8
                span = max(width - left - 90, 20)
                peak = float(max(abs(wave).max(), 1e-6))
                columns = 120
                step = max(len(wave) // columns, 1)
                for index in range(min(columns, len(wave) // step)):
                    chunk = wave[index * step:(index + 1) * step]
                    height = float(abs(chunk).max()) / peak * 22.0
                    x = left + index * (span / columns)
                    canvas.create_line(x, 26 - height, x, 26 + height,
                                       fill=config.UI_COOL)
                canvas.create_text(width - 6, 46, anchor="se",
                                   fill=config.UI_MID,
                                   font=config.UI_FONT_TINY,
                                   text="%.1f s at %d Hz"
                                        % (seconds, config.AUDIO_SAMPLE_RATE))
        except Exception as failure:                          # noqa: BLE001
            say("could not decode: %s" % str(failure).splitlines()[0][:60],
                config.UI_ALERT)

    def _clear_image(self) -> None:
        self._clear_media()

    def _replay(self) -> None:
        self.frame = 0
        self.playing = True

    def _expand(self, network) -> None:
        self.expanded = network

    def _click_network(self, event) -> None:
        """Click a tile to expand it; click anywhere to come back."""
        if self.expanded == "anatomy":
            self._drag_from = event.x
            return
        if self.expanded == "routing":
            #  Clicking a module OPENS it. The routing view now records the
            #  screen box of every node it drew, so the diagram is navigable
            #  rather than only readable -- which is what a node being a live
            #  network rather than a label should mean.
            for name, (x0, y0, x1, y1) in getattr(self.flow, "boxes", {}).items():
                if x0 <= event.x <= x1 and y0 <= event.y <= y1:
                    self.expanded = name
                    return
            return
        if self.expanded is not None:
            self.expanded = None
            return
        for name, (x0, y0, x1, y1) in self._tiles.items():
            if x0 <= event.x <= x1 and y0 <= event.y <= y1:
                self.expanded = name
                return

    def _drag_rotate(self, event) -> None:
        if self.expanded != "anatomy":
            return
        if self._drag_from is not None:
            self.angle += (event.x - self._drag_from) * 0.01
            self.spinning = False
        self._drag_from = event.x

    def _drag_end(self, event) -> None:
        self._drag_from = None

    def _show_lesions(self) -> None:
        """Open the lesion window.

        It used to be built and immediately withdrawn with nothing able to
        reveal it, which left the whole lesion feature present in the code and
        absent from the product.
        """
        window = getattr(self, "lesion_window", None)
        if window is not None:
            window.deiconify()
            window.lift()

    def _write(self, tag: str, text: str) -> None:
        self.transcript.configure(state="normal")
        self.transcript.insert("end", text + "\n", tag)
        self.transcript.see("end")
        self.transcript.configure(state="disabled")

    def _send(self) -> None:
        if self.engine is None:
            return
        typed = self.entry.get().strip()
        text = typed or self.text_box.get("1.0", "end").strip()
        media = (self.image_path, self.audio_path, self.video_path)
        if not text and not any(media):
            return
        self.entry.delete(0, "end")
        described = [t for t, p in zip(("image", "audio", "video"), media) if p]
        self._write("you", "you    %s" % (text or "[%s]"
                                          % ", ".join(described)))
        self.status = "settling..."
        image_path, audio_path, video_path = media
        self._clear_media()
        layers = self.layers

        def work():
            try:
                image = None
                if image_path:
                    from PIL import Image
                    image = Image.open(image_path).convert("RGB")
                audio = None
                if audio_path:
                    audio = _read_audio(audio_path)
                thought = self.engine.think(text, image=image, audio=audio,
                                            video=video_path, layers=layers)
                self.queue.put(("thought", thought))
                #  The cortex's answer goes out FIRST, and goes out even if
                #  the language model is slow, disabled or broken -- it does
                #  not depend on it.
                self.queue.put(("cortex", self.engine.cortex_answer(thought)))
                reply = self.engine.speak(thought)
                if reply:
                    self.queue.put(("reply", reply))
            except Exception as failure:                      # noqa: BLE001
                #  Surfaced in the transcript rather than dying on a worker
                #  thread, where the traceback would go nowhere the user can
                #  see it and the UI would just stop responding.
                self.queue.put(("error", str(failure)))

        threading.Thread(target=work, daemon=True).start()

    # -- plumbing ----------------------------------------------------------
    def _boot(self) -> None:
        def work():
            self.queue.put(("ready", Engine()))
        threading.Thread(target=work, daemon=True).start()

    def _tick(self) -> None:
        try:
            while True:
                kind, payload = self.queue.get_nowait()
                if kind == "ready":
                    self.engine = payload
                    from brainview import BrainRenderer
                    from flowview import FlowRenderer
                    from layerview import LayerView, NETWORKS
                    self._network_names = set(NETWORKS)
                    self.renderer = BrainRenderer(payload.cortex)
                    self.flow = FlowRenderer(payload.cortex)
                    self.layers = LayerView(payload.cortex)
                    if payload.trained:
                        self.status = "ready"
                        self._write("note", "[trained: %s]" % "  ".join(
                            "%s=%s" % kv
                            for kv in sorted(payload.training_record.items())))
                    else:
                        self.status = "UNTRAINED -- answers are noise"
                        self._write("note", "[no trained checkpoint. run "
                                            "python train_v2.py -- until then "
                                            "the lobes are random]")
                    self._build_lesions()
                elif kind == "thought":
                    self.thought = payload
                    self.readouts = self.engine.readouts(payload)
                    self.frame = 0
                    self.playing = True
                    self.status = ("settled" if payload.settled
                                   else "did NOT settle")
                elif kind == "cortex":
                    self._write("cortex", "CORTEX  %s" % payload)
                elif kind == "reply":
                    self._write("brain", "SPEECH  %s" % payload)
                elif kind == "error":
                    self.status = "failed"
                    self._write("note", "[failed: %s]" % payload)
        except queue.Empty:
            pass

        if self.playing and self.thought is not None:
            self.frame += 1
            if self.frame >= config.T_STEPS:
                self.frame = config.T_STEPS - 1
                self.playing = False
        if self.spinning and self.expanded == "anatomy":
            self.angle += config.UI_BRAIN_SPIN

        self._draw()
        self.root.after(config.UI_REFRESH_MS, self._tick)

    def _build_lesions(self) -> None:
        tk = self.tk
        window = tk.Toplevel(self.root)
        window.title("lesions")
        window.configure(bg=config.UI_BG)
        window.geometry("240x540")
        tk.Label(window, text="CUT A TRACT", bg=config.UI_BG,
                 fg=config.UI_ACCENT, font=config.UI_FONT_TITLE).pack(
            anchor="w", padx=10, pady=(10, 0))
        tk.Label(window, text="then perceive again and watch it change",
                 bg=config.UI_BG, fg=config.UI_DIM,
                 font=config.UI_FONT_TINY).pack(anchor="w", padx=10)
        for name in self.engine.tract_names():
            var = tk.IntVar(value=0)

            def toggle(name=name, var=var):
                self.engine.lesion(name, bool(var.get()))

            tk.Checkbutton(window, text=name, variable=var, command=toggle,
                           bg=config.UI_BG, fg=config.UI_MID,
                           selectcolor=config.UI_PANEL,
                           activebackground=config.UI_BG,
                           activeforeground=config.UI_ACCENT,
                           font=config.UI_FONT_TINY, anchor="w").pack(
                anchor="w", fill="x", padx=8)
        window.protocol("WM_DELETE_WINDOW", window.withdraw)
        window.withdraw()
        self.lesion_window = window

    # -- drawing -----------------------------------------------------------
    def _draw(self) -> None:
        canvas = self.canvas
        canvas.delete("all")
        width = max(canvas.winfo_width(), 200)
        height = max(canvas.winfo_height(), 200)

        canvas.create_text(16, 14, anchor="w", fill=config.UI_ACCENT,
                           font=config.UI_FONT_TITLE,
                           text="NETWORKS   T = %d, ALL UPDATING TOGETHER"
                                % config.T_STEPS)
        canvas.create_text(16, 34, anchor="w", fill=config.UI_MID,
                           font=config.UI_FONT_SMALL, text=self.status.upper())

        if self.expanded == "routing":
            if self.flow is not None:
                trace = self.thought.trace if self.thought else None
                present = set(self.thought.modalities) if self.thought else set()
                self.flow.draw(canvas, width, height - 10, trace,
                               self.frame,
                               self.engine.lesioned if self.engine else set(),
                               present, layers=self.layers,
                               photos=self._photos)
        elif self.expanded == "anatomy":
            if self.renderer is not None:
                trace = self.thought.trace if self.thought else None
                self.renderer.draw(canvas, width, height - 40, trace,
                                   self.frame,
                                   self.engine.lesioned if self.engine
                                   else set(), self.angle)
        elif self.expanded in getattr(self, "_network_names", ()):
            self._draw_tiles(canvas, width, height)
        elif self.layers is not None:
            self._draw_tiles(canvas, width, height)

        self._draw_side()

    def _draw_tiles(self, canvas, width: int, height: int) -> None:
        from layerview import NETWORKS
        from PIL import ImageTk

        trace = self.thought.trace if self.thought else None
        names = [self.expanded] if self.expanded else list(NETWORKS)
        columns = 1 if self.expanded else 3
        rows = 1 if self.expanded else 2

        top, pad = 44, 8
        tile_w = int((width - pad * (columns + 1)) / columns)
        tile_h = int((height - top - pad * (rows + 1)) / rows)
        if tile_w < 60 or tile_h < 60:
            return
        self._tiles = {}

        for index, name in enumerate(names):
            column, row = index % columns, index // columns
            x = pad + column * (tile_w + pad)
            y = top + pad + row * (tile_h + pad)
            self._tiles[name] = (x, y, x + tile_w, y + tile_h)

            canvas.create_text(x + 4, y - 2, anchor="sw", fill=config.UI_COOL,
                               font=config.UI_FONT_LABEL, text=name.upper())
            try:
                #  `step` selects the recorded frame for THIS timestep, so
                #  the tiles animate with the settling rather than showing
                #  where it ended up.
                image, cols = self.layers.render(name, tile_w, tile_h, trace,
                                                 step=self.frame)
            except Exception as failure:                      # noqa: BLE001
                canvas.create_text(x + tile_w / 2, y + tile_h / 2,
                                   fill=config.UI_ALERT,
                                   font=config.UI_FONT_TINY,
                                   text="failed: %s" % failure)
                continue
            photo = ImageTk.PhotoImage(image)
            #  Held per tile: Tk keeps no reference of its own, and a local
            #  one is collected before it paints -- which shows up as a blank
            #  panel rather than as an error.
            self._photos[name] = photo
            canvas.create_image(x, y, image=photo, anchor="nw")

            live = [c for c in cols if c.active is not None]
            if live:
                canvas.create_text(x + tile_w - 4, y - 2, anchor="se",
                                   fill=config.UI_DIM,
                                   font=config.UI_FONT_DATA,
                                   text="%d/%d ACTIVE"
                                        % (live[-1].active, live[-1].total))

        if not self.expanded:
            canvas.create_text(width / 2, height - 4, anchor="s",
                               fill=config.UI_MID, font=config.UI_FONT_DATA,
                               text="CLICK A NETWORK TO EXPAND IT")
        else:
            canvas.create_text(width / 2, height - 4, anchor="s",
                               fill=config.UI_MID, font=config.UI_FONT_DATA,
                               text="CLICK ANYWHERE TO GO BACK")

    def _draw_side(self) -> None:
        side = self.side
        side.delete("all")
        width = max(side.winfo_width(), 120)
        cursor = 18
        side.create_text(12, cursor, anchor="w", fill=config.UI_ACCENT,
                         font=config.UI_FONT_TITLE, text="SYSTEM")
        cursor += 24

        sample = self.telemetry.latest()
        for label, value in (("cpu", "%.0f%%" % sample.cpu_percent),
                             ("ram", "%.0f%%" % sample.ram_percent),
                             ("gpu", sample.gpu_line(compact=True)),
                             ("threads", str(sample.threads))):
            side.create_text(16, cursor, anchor="w", fill=config.UI_MID,
                             font=config.UI_FONT_TINY, text=label.upper())
            side.create_text(width - 12, cursor, anchor="e", fill=config.UI_FG,
                             font=config.UI_FONT_DATA, text=str(value)[:22])
            cursor += 16
        cursor += 10

        if self.thought is not None and self.thought.concepts:
            side.create_text(12, cursor, anchor="w", fill=config.UI_COOL,
                             font=config.UI_FONT_SMALL,
                             text="what it made of it")
            cursor += 18
            for name, score in self.thought.concepts[:5]:
                mark = "*" if name in self.thought.extrapolated else ""
                side.create_text(16, cursor, anchor="w", fill=config.UI_FG,
                                 font=config.UI_FONT_TINY,
                                 text=(name + mark)[:22])
                side.create_rectangle(width - 74, cursor - 4, width - 14,
                                      cursor + 4, outline=config.UI_EDGE_IDLE)
                side.create_rectangle(width - 74, cursor - 4,
                                      width - 74 + 60 * max(score, 0.0),
                                      cursor + 4, fill=config.UI_ACCENT,
                                      outline="")
                cursor += 15
            side.create_text(16, cursor, anchor="w", fill=config.UI_DIM,
                             font=config.UI_FONT_TINY,
                             text="* outside the trained set")
            cursor += 20

        for group, rows in self.readouts:
            side.create_text(12, cursor, anchor="w", fill=config.UI_COOL,
                             font=config.UI_FONT_SMALL, text=group)
            cursor += 15
            for label, value in rows:
                side.create_text(16, cursor, anchor="w", fill=config.UI_MID,
                                 font=config.UI_FONT_TINY,
                                 text=str(label)[:20])
                side.create_text(width - 12, cursor, anchor="e",
                                 fill=config.UI_FG, font=config.UI_FONT_DATA,
                                 text="%.3f" % value)
                cursor += 14
            cursor += 6

        if self.engine is not None and self.engine.lesioned:
            side.create_text(12, cursor, anchor="w", fill=config.UI_ALERT,
                             font=config.UI_FONT_SMALL, text="LESIONED")
            cursor += 16
            for name in sorted(self.engine.lesioned):
                side.create_text(16, cursor, anchor="w", fill=config.UI_ALERT,
                                 font=config.UI_FONT_TINY, text=name)
                cursor += 12


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selftest", action="store_true")
    args = parser.parse_args()

    if args.selftest:
        return selftest()

    import tkinter as tk

    root = tk.Tk()
    Console(root)
    root.mainloop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
