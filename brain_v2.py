"""
brain_v2.py -- the trained v2 system, assembled and ready to answer.

WHAT THIS IS FOR.

`cortex.py` settles. `train_v2.py` trains it. Neither of them can be talked to.
This is the piece that turns a settled cortical state into an answer, and it is
where the interesting join happens: a 256-number state that means nothing to
anything outside this model has to become words.

THE BRIDGE IS THE SEMANTIC HEAD, and it only exists because training put it
there. Frontal state is trained to predict the FROZEN text encoder's embedding
of what the percept is about -- for text, its own content; for images, the
embedding of the label, which is the trick CLIP is built on. Both land in the
same 384-dimensional space, so:

    settled cortical state --> semantic vector --> nearest known concepts

Those nearest concepts, plus the classifier heads, plus the subcortical state
(how open the gate was, how salient, how novel) become a structured perceptual
report. The language model speaks from that report and from nothing else.

WHY THE LANGUAGE MODEL IS NOT THE BRAIN. Qwen is frozen and is handed a
report; it never sees the raw input. That boundary is the whole point. If the
language model read the user's text directly it would answer well and the four
lobes would be decoration -- which is precisely the failure this architecture
exists to avoid, and precisely what a demo is tempted to do. Everything the
answer is based on has to come through the cortex first.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import torch
import torch.nn as nn

import config
from cortex import Cortex

CHECKPOINT = config.PROJECT_ROOT / "checkpoints" / "cortex_v2.pt"


@dataclass
class Thought:
    """One complete pass: what was perceived, and everything behind it."""

    text: str = ""
    #  (concept, similarity) from the shared semantic space, strongest first
    concepts: list[tuple[str, float]] = field(default_factory=list)
    #  (label, probability) from the domain classifier
    predictions: list[tuple[str, float]] = field(default_factory=list)
    domain: str = ""
    modalities: list[str] = field(default_factory=list)
    #  subcortical readings, for the report and the dashboard
    gate: float = 1.0
    salience: float = 0.0
    novelty: float = 0.0
    prediction_error: float = 0.0
    settled: bool = True
    settling_ratio: float = 0.0
    #  How many video frames were actually decoded and read. 0 for a still
    #  image, which is how the dashboard can tell the two apart.
    frames_read: int = 0
    elapsed_ms: float = 0.0
    trace: object = None
    #  Concepts retrieved from OUTSIDE the set the heads were trained on.
    #  Named separately because presenting them identically would be a claim
    #  the training never made.
    extrapolated: set = field(default_factory=set)
    #  (similarity, confidence) floors, measured from this model. None falls
    #  back to the config constants.
    thresholds: tuple | None = None

    @property
    def margin(self) -> float:
        """Gap between the best and second-best label.

        The most useful of the three signals for spotting input the model has
        no business classifying, because it measures whether the classifier
        DECIDED or merely ranked. Measured: real inputs 0.95-1.00, random
        keystrokes 0.41-0.47.
        """
        if len(self.predictions) < 2:
            return self.predictions[0][1] if self.predictions else 0.0
        return self.predictions[0][1] - self.predictions[1][1]

    @property
    def recognised(self) -> bool:
        """Is this something the system knows, or the nearest available label?

        Three signals, all required: a genuinely close concept, a confident
        classifier, and a real MARGIN between first and second place. Any one
        alone is easy to satisfy by accident -- softmax always sums to one,
        and in a thousand-concept space something is always nearest.

        WHAT THIS CANNOT CATCH, stated plainly because the limit is real.
        Input that maps CONFIDENTLY onto a wrong class is invisible to every
        one of these signals. Measured: "1+1" scores similarity 0.294,
        confidence 0.982 and margin 0.976 -- indistinguishable from a correct
        recognition -- and lands on "misc.forsale". The model is not uncertain
        and does not know it is wrong; MiniLM places "1+1" somewhere, and of
        twenty topics, classifieds full of numbers and prices genuinely is
        the nearest. No confidence measure can detect that, because there is
        no lack of confidence to detect.
        
        Fixing it properly needs an explicit "none of these" class trained
        against out-of-domain text, which is a retraining job rather than a
        threshold. Until then the grounded chat carries it: the language
        model answers "1+1" correctly from its own knowledge and only reports
        the cortex when asked about it.
        """
        similarity = self.concepts[0][1] if self.concepts else 0.0
        confidence = self.predictions[0][1] if self.predictions else 0.0
        floor = self.thresholds or (config.RECOGNITION_SIMILARITY,
                                    config.RECOGNITION_CONFIDENCE)
        return (similarity >= floor[0] and confidence >= floor[1]
                and self.margin >= config.RECOGNITION_MARGIN)

    def summary(self) -> str:
        top = ", ".join("%s %.0f%%" % (c, 100 * s) for c, s in self.concepts[:3])
        return "%s | gate %.2f | novelty %.2f | %s" % (
            top or "nothing", self.gate, self.novelty,
            "settled" if self.settled else "did not settle")


class BrainV2System(nn.Module):
    """Cortex, heads, and the semantic bridge -- loaded from a checkpoint."""

    def __init__(self, device: str | None = None):
        super().__init__()
        self.device_name = device or config.resolve_device()
        self.cortex = Cortex()
        self.trained = False
        self.training_record: dict = {}

        #  Built here, not only in load(). An untrained system must still
        #  RUN -- and say clearly that it is untrained -- rather than crash on
        #  a missing attribute. "No checkpoint" is a state to report, not an
        #  exception to raise halfway through a percept.
        self.heads, self.project, self.semantic = self._build_heads(
            dict(config.DEFAULT_CLASSES))
        self.label_names: dict[str, list[str]] = {}
        self.label_space: dict[str, torch.Tensor] = {}
        #  The wide retrieval vocabulary, and how many of it the heads were
        #  actually trained against.
        self.concept_names: list[str] = []
        self.concept_space: torch.Tensor | None = None
        self.trained_concepts: int = 0
        #  Measured by calibrate(); None means "use the config fallback".
        self.recognition: tuple | None = None
        self._front = None
        self._language = None
        #  Parameters the loaded checkpoint could not fill. Empty is the
        #  healthy case; anything in it means the answer is partly random.
        self.stale_parameters: list[str] = []

    # -- loading -----------------------------------------------------------
    def load(self, path: Path | None = None) -> bool:
        path = path or CHECKPOINT
        if not path.exists():
            return False
        state = torch.load(path, map_location=self.device_name,
                           weights_only=False)
        self.heads, self.project, self.semantic = self._build_heads(
            state["classes"])

        #  A checkpoint written before a module existed is STALE, not
        #  broken -- but it must say so out loud. Loading it silently with
        #  strict=False would leave the new parameters at their random init
        #  while every report claimed a trained system, which is exactly the
        #  kind of quiet wrongness that takes a week to find. So: load what
        #  matches, name what does not, and mark the system untrained until
        #  it has actually been retrained on the current architecture.
        missing, unexpected = self.cortex.load_state_dict(
            state["cortex"], strict=False)
        self.stale_parameters = sorted(missing) + sorted(unexpected)
        if self.stale_parameters:
            print("checkpoint predates %d parameter(s) -- retrain required: %s"
                  % (len(self.stale_parameters),
                     ", ".join(self.stale_parameters[:4])))
        self.heads.load_state_dict(state["heads"])
        self.project.load_state_dict(state["project"])
        self.semantic.load_state_dict(state["semantic"])
        self.training_record = dict(state.get("training_record") or {})
        self.trained = bool(self.training_record) and not self.stale_parameters

        self.to(self.device_name)
        self.eval()
        return True

    @staticmethod
    def _build_heads(classes: dict[str, int]):
        """The read-out heads, shaped the same way train_v2.py shapes them.

        Must match exactly -- the LayerNorm, and the fact that the read-out is
        all four lobes concatenated rather than frontal alone -- or the
        checkpoint will not load into it.
        """
        dim = config.LATENT_DIM
        wide = dim * len(Cortex.LOBES)
        heads = nn.ModuleDict({
            name: nn.Sequential(nn.LayerNorm(wide), nn.Linear(wide, count))
            for name, count in classes.items()
        })
        project = nn.ModuleDict({
            name: nn.Sequential(nn.Linear(raw, dim), nn.LayerNorm(dim))
            for name, raw in config.RAW_DIMS.items()
        })
        semantic = nn.Sequential(
            nn.LayerNorm(wide), nn.Linear(wide, dim), nn.GELU(),
            nn.Linear(dim, config.TEXT_RAW_DIM))
        return heads, project, semantic

    def load_concepts(self) -> None:
        """The vocabulary the semantic space is anchored to.

        Read from the training stream cache, because those are exactly the
        concepts the semantic head was trained against. Anchoring to anything
        else would be measuring similarity in a space against points that
        space was never taught.
        """
        cache = config.PROJECT_ROOT / "data" / "v2_streams.pt"
        if not cache.exists():
            return
        streams = torch.load(cache, weights_only=False)

        #  THE WIDE VOCABULARY -- a thousand concepts rather than the hundred
        #  and thirty the heads were trained on. See train_v2 for why this is
        #  legitimate: the semantic head maps into the frozen text encoder's
        #  space, and nearest-neighbour retrieval in that space works against
        #  any points embedded the same way. What matters is not pretending
        #  the extrapolated ones carry the same weight, which is what
        #  `trained_concepts` is for.
        vocabulary = streams.get("vocabulary")
        if vocabulary is not None:
            space = vocabulary["embeddings"].to(self.device_name)
            space = space - space.mean(dim=0, keepdim=True)
            self.concept_names = list(vocabulary["names"])
            self.concept_space = nn.functional.normalize(space, dim=-1)
            self.trained_concepts = int(vocabulary["trained"])

        for modality in ("image", "text", "audio", "sensor"):
            if "label_embeddings" not in (streams.get(modality) or {}):
                continue
            names = streams[modality]["names"]
            space = streams[modality]["label_embeddings"].to(self.device_name)

            #  CENTRED before normalising, and that is not a detail. Every
            #  image label was embedded from the same template ("a photo of a
            #  ..."), so those vectors share a large common component and sit
            #  closer to everything than the text labels do -- measured, mean
            #  pairwise similarity 0.458 against 0.374. Ranking raw cosine
            #  across both then hands the top of every list to whichever
            #  modality had the more uniform prompt, and a question about a
            #  doctor comes back "bear, elephant, house". Removing each
            #  space's own mean removes exactly that shared component and
            #  leaves what distinguishes the concepts from one another.
            space = space - space.mean(dim=0, keepdim=True)
            self.label_names[modality] = names
            self.label_space[modality] = nn.functional.normalize(space, dim=-1)

    @torch.no_grad()
    def calibrate(self, samples: int = 192) -> bool:
        """Set the recognition thresholds from THIS model's own distribution.

        Hardcoding them does not survive retraining. The constants in config
        were measured against one checkpoint, where correct answers scored
        0.44-0.49 similarity; the next checkpoint's correct answers scored a
        median of 0.389, so a threshold of 0.40 rejected more than half of
        what the system got RIGHT -- including "the doctor said the treatment
        was working", classified sci.med at 100% and reported as unrecognised.

        So the threshold is measured instead: run held-out examples, keep the
        ones the classifier gets right, and take a low percentile of their
        scores. That accepts ~90% of genuine recognitions while still sitting
        far above what unrelated input produces. It is a property of the
        trained model, and it is recomputed whenever the model changes.
        """
        cache = config.PROJECT_ROOT / "data" / "v2_streams.pt"
        if not cache.exists() or "text" not in self.label_names:
            return False
        streams = torch.load(cache, weights_only=False)
        tokens, labels = streams["text"]["test"]

        similarity, confidence = [], []
        for start in range(0, min(samples, len(labels)), 64):
            chunk = tokens[start:start + 64].float().to(self.device_name)
            target = labels[start:start + 64]
            projected = self.project["text"](chunk)
            trace = self.cortex.settle({"thal.PULVINAR": projected.mean(dim=1)},
                                       sequences={"temporal": projected})
            steps = range(config.SETTLING_WARMUP, config.T_STEPS)
            wide = torch.cat(
                [torch.stack([trace.history.read(l, t) for t in steps]).mean(0)
                 for l in self.cortex.LOBES], dim=-1)
            probability = torch.softmax(self.heads["text"](wide), dim=-1)
            best, predicted = probability.max(dim=-1)
            raw = self.semantic(wide)
            vector = nn.functional.normalize(
                raw - raw.mean(dim=-1, keepdim=True), dim=-1)
            nearest = (vector @ self.concept_space.t()).max(dim=-1).values
            right = predicted.cpu() == target
            similarity += nearest[right].tolist()
            confidence += best[right].tolist()

        if len(similarity) < 20:
            return False
        similarity.sort()
        confidence.sort()
        index = max(int(0.10 * len(similarity)) - 1, 0)
        self.recognition = (similarity[index], confidence[index])
        return True

    # -- the sensory front, loaded lazily ---------------------------------
    @property
    def front(self):
        if self._front is None:
            from modules.encoders import EncoderBank
            from modules.sensory import SensoryFront
            self._front = SensoryFront(EncoderBank(), device=self.device_name)
        return self._front

    @property
    def language(self):
        if self._language is None:
            from modules.language import LanguageModule
            self._language = LanguageModule(device=self.device_name)
        return self._language

    # -- perceiving --------------------------------------------------------
    @torch.no_grad()
    def perceive(self, text: str | None = None, image=None, audio=None,
                 video=None, on_step=None) -> Thought:
        """Settle on whatever arrived, then read the state.

        Every modality is optional and they combine freely -- text alone, an
        image and a sound, all four. A missing sense is ABSENT rather than
        zero-filled: its nucleus stays silent, which is the difference between
        "nothing is there" and "something blank is there".
        """
        thought = Thought(text=text or "")
        sensory, sequences = {}, {}

        #  VIDEO IS TWO STREAMS, not a blurry photograph.
        #
        #  It used to average the sampled frames into one grid, and the cost of
        #  that was motion: after the mean, a ball flying left and a ball flying
        #  right are the same tensor, and temporal -- the one lobe with any
        #  notion of before and after -- had nothing to be recurrent ABOUT.
        #
        #  So a clip is now decomposed the way vision actually splits. The
        #  middle frame contributes its full patch grid, which is spatial
        #  detail; the frames contribute one pooled vector each IN TIME ORDER,
        #  which is motion. Both go down the visual route as positions of one
        #  sequence, so parietal can attend across frames and temporal's GRU
        #  reads them in order. Reversing the clip now changes the answer.
        #
        #  A clip also usually has a SOUNDTRACK, which was being thrown away
        #  outright. When a decoder can extract it and the caller did not
        #  supply audio separately, it goes down the auditory route -- so one
        #  video drives both sensory pathways, which is the whole point of
        #  having more than one.
        if video is not None:
            frames = self._video_frames(video)
            #  A still loaded ALONGSIDE a clip is kept, not overwritten. This
            #  assigned straight to `image` and silently dropped whatever was
            #  already there -- and the input pane lets you load both at once,
            #  so the only signal that half your input had been discarded was
            #  the modality list not mentioning it.
            if image is not None:
                frames = frames + [image]
            image = frames
            if audio is None:
                audio = self._video_audio(video)

        def deliver(modality: str, projected: torch.Tensor) -> None:
            """Send one sense to its nucleus and to every lobe.

            Must match train_v2.BrainV2.ROUTES exactly. Every lobe reads the
            sequence through its own operator, so the same input is parsed
            four different ways rather than sitting idle in three of them.
            """
            from train_v2 import BrainV2
            nucleus, lobes = BrainV2.ROUTES[modality]
            sensory["thal.%s" % nucleus] = projected.mean(dim=1)
            for lobe in lobes:
                sequences[lobe] = (
                    projected if lobe not in sequences
                    else torch.cat([sequences[lobe], projected], dim=1))

        if text:
            tokens = self.front._text_tokens([text]).to(self.device_name)
            deliver("text", self.project["text"](tokens.float()))
            thought.modalities.append("text")
            thought.domain = "text"

        if image is not None:
            if isinstance(image, list):
                #  A clip. See the note above `_video_frames` for why this is
                #  motion + detail rather than a mean over frames.
                grids = [self.front._vision_grid(f).to(self.device_name)
                         for f in image]
                motion = torch.cat([g.mean(dim=1, keepdim=True) for g in grids],
                                   dim=1)
                detail = grids[len(grids) // 2]
                grid = torch.cat([motion, detail], dim=1)
                thought.frames_read = len(grids)
                thought.modalities.append("video")
            else:
                grid = self.front._vision_grid(image).to(self.device_name)
            deliver("image", self.project["image"](grid.float()))
            if "video" not in thought.modalities:
                thought.modalities.append("image")
            thought.domain = "image"

        if audio is not None:
            #  Concatenated along the position axis by `deliver`, never
            #  summed: five text tokens and sixty-four audio frames are
            #  different lengths, and even at equal lengths summing would
            #  claim token 3 and frame 3 are the same position.
            frames = self.front._audio_frames(audio).to(self.device_name)
            deliver("audio", self.project["audio"](frames.float()))
            thought.modalities.append("audio")
            thought.domain = thought.domain or "audio"

        if not sensory:
            return thought

        trace = self.cortex.settle(sensory, sequences=sequences,
                                   on_step=on_step)
        #  The same integrated read-out training used. Reading only the final
        #  timestep would expose about thirteen active units out of 256 and
        #  put the heads back at chance -- the sparsity is real, so the
        #  read-out has to integrate over time the way training did.
        #  The same read-out training used: all four lobes, integrated over
        #  the settled window. See train_v2.readout for why it is not frontal
        #  alone, and why averaging over time alone recovers nothing.
        steps = range(config.SETTLING_WARMUP, config.T_STEPS)
        frontal = torch.cat(
            [torch.stack([trace.history.read(lobe, t) for t in steps]).mean(0)
             for lobe in self.cortex.LOBES], dim=-1)
        thought.trace = trace

        #  the semantic bridge
        #  The query is centred the same way the label spaces are, or the
        #  correction above would only be applied to one side of the compare.
        raw = self.semantic(frontal)
        vector = nn.functional.normalize(raw - raw.mean(dim=-1, keepdim=True),
                                         dim=-1)
        if self.concept_space is not None:
            similarity = (vector @ self.concept_space.t())[0]
            best = similarity.topk(min(6, similarity.numel()))
            for score, index in zip(best.values.tolist(), best.indices.tolist()):
                thought.concepts.append((self.concept_names[index], float(score)))
                if index >= self.trained_concepts:
                    thought.extrapolated.add(self.concept_names[index])
        else:
            for modality in self.label_space:
                similarity = (vector @ self.label_space[modality].t())[0]
                best = similarity.topk(min(5, similarity.numel()))
                for score, index in zip(best.values.tolist(),
                                        best.indices.tolist()):
                    thought.concepts.append(
                        (self.label_names[modality][index], float(score)))
            thought.concepts.sort(key=lambda c: -c[1])
            thought.concepts = thought.concepts[:6]

        #  the classifier for whichever domain actually arrived
        if thought.domain in self.heads:
            probabilities = torch.softmax(
                self.heads[thought.domain](frontal), dim=-1)[0]
            best = probabilities.topk(min(3, probabilities.numel()))
            names = self.label_names.get(thought.domain, [])
            for score, index in zip(best.values.tolist(), best.indices.tolist()):
                label = (names[index].replace("_", " ")
                         if index < len(names) else "class %d" % index)
                thought.predictions.append((label, float(score)))

        #  subcortical readings
        if trace.bg:
            thought.gate = trace.bg[-1].openness
        if trace.amygdala:
            thought.salience = float(trace.amygdala[-1].salience.mean())
        if trace.novelty:
            thought.novelty = trace.novelty[-1]
        if trace.cerebellum:
            thought.prediction_error = trace.cerebellum[-1].error_value
        thought.thresholds = self.recognition
        thought.settled = trace.settled()
        thought.settling_ratio = trace.settling_ratio()
        thought.elapsed_ms = trace.elapsed_ms
        return thought

    # -- speaking ----------------------------------------------------------
    @torch.no_grad()
    def _video_frames(self, path):
        """Evenly spaced frames from a clip, averaged into one visual percept.

        Reads with OpenCV when it is available and falls back to imageio;
        if neither is installed the caller is told plainly rather than the
        video silently being ignored.
        """
        from PIL import Image
        import numpy as np

        frames = []
        try:
            import cv2
            capture = cv2.VideoCapture(str(path))
            total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
            for index in np.linspace(0, max(total - 1, 0),
                                     config.VIDEO_FRAMES, dtype=int):
                capture.set(cv2.CAP_PROP_POS_FRAMES, int(index))
                ok, frame = capture.read()
                if ok:
                    frames.append(Image.fromarray(frame[:, :, ::-1]))
            capture.release()
        except ImportError:
            try:
                import imageio.v3 as iio
                clip = iio.imread(str(path), plugin="pyav")
                step = max(len(clip) // config.VIDEO_FRAMES, 1)
                frames = [Image.fromarray(f) for f in clip[::step]]
            except Exception as failure:                      # noqa: BLE001
                raise RuntimeError(
                    "video needs opencv-python or imageio[pyav]: %s" % failure)
        if not frames:
            raise RuntimeError("could not read any frames from %s" % path)
        return frames[: config.VIDEO_FRAMES]

    def _video_audio(self, path):
        """The clip's soundtrack as a mono float32 waveform, or None.

        None is a normal outcome, not a failure: plenty of clips are silent,
        and ffmpeg is not guaranteed to be present. Either way the visual
        stream still runs, so a missing soundtrack costs a pathway rather than
        the whole percept.
        """
        import subprocess
        import numpy as np

        try:
            import imageio_ffmpeg
            binary = imageio_ffmpeg.get_ffmpeg_exe()
        except Exception:                                     # noqa: BLE001
            return None
        try:
            #  Decoded straight to the rate the auditory front end wants, as
            #  raw little-endian float32 on stdout -- no temporary file, and
            #  no second resampling step to introduce its own artefacts.
            done = subprocess.run(
                [binary, "-nostdin", "-loglevel", "error", "-i", str(path),
                 "-vn", "-ac", "1", "-ar", str(config.AUDIO_SAMPLE_RATE),
                 "-f", "f32le", "-"],
                capture_output=True, timeout=120)
        except Exception:                                     # noqa: BLE001
            return None
        if done.returncode != 0 or not done.stdout:
            return None
        wave = np.frombuffer(done.stdout, dtype="<f4").astype("float32")
        if not wave.size:
            return None
        #  A track of digital silence is not a sound. ffmpeg succeeds on one
        #  and returns a valid array of zeros, and without this check the
        #  percept would claim an auditory modality that carries no
        #  information -- which then shows up in the report as a pathway that
        #  contributed, and in the dashboard as a lit AUDIO box. Reporting a
        #  sense that delivered nothing is worse than reporting no sense.
        if float(np.abs(wave).max()) < 1e-4:
            return None
        return wave

    def report(self, thought: Thought) -> "object":
        """The structured perceptual record the language model speaks from."""
        from modules.language import Perception

        #  An unrecognised percept hands over NO predictions. Passing them
        #  along with a note saying "but it is unsure" invites the language
        #  model to report the label anyway, which is exactly how "1+1"
        #  became a confident "misc.forsale".
        #  MEASURED, not asserted. This used to hand over all four lobe names
        #  unconditionally, which made "4 of 4 lobes contributed" a statement
        #  about the config file rather than about the percept. A lobe counts
        #  as having contributed when its committed state is non-zero at the
        #  final timestep.
        active = list(self.cortex.LOBES)
        if thought.trace is not None:
            active = [name for name in self.cortex.LOBES
                      if float(thought.trace.history.read(
                          name, config.T_STEPS - 1).abs().sum()) > 0.0]

        return Perception(
            modalities=list(thought.modalities),
            lobes_active=active,
            domain=thought.domain,
            predictions=list(thought.predictions) if thought.recognised else [],
            reference=list(thought.concepts[:4]) if thought.recognised else [],
            gate=thought.gate,
            allowed=thought.gate > config.BG_GATE_THRESHOLD,
            user_text=thought.text,
        )

    def say(self, thought: Thought, history=None, on_token=None) -> str:
        """Turn the thought into words.

        Under config.GROUNDED_CHAT the language model reads the user's message
        as well as the cortex's report. That makes the system answerable on
        anything, and it means the reply text alone no longer demonstrates
        that the lobes contributed -- see `contribution()`, which measures it
        directly instead of leaving it to be inferred.
        """
        return self.language.respond(self.report(thought), history=history,
                                     on_token=on_token)

    @torch.no_grad()
    def contribution(self, thought: Thought) -> dict:
        """How much the cortex actually shaped this answer.

        With the language model no longer blindfolded, "the lobes are not
        decoration" stops being self-evident from the output and has to be
        measured. These are the terms that go into the report the model is
        given, so a run where they are all empty is a run where the cortex
        contributed nothing -- and the panel will say so.
        """
        return {
            "recognised": thought.recognised,
            "concepts": len(thought.concepts),
            "top_similarity": thought.concepts[0][1] if thought.concepts else 0.0,
            "classifier": thought.predictions[0][0] if thought.predictions else "",
            "confidence": thought.predictions[0][1] if thought.predictions else 0.0,
            "settled": thought.settled,
            "in_report": bool(thought.recognised and thought.concepts),
        }


def build(device: str | None = None) -> BrainV2System:
    """Load the trained system, or say plainly that there isn't one."""
    system = BrainV2System(device=device)
    if not system.load():
        print("no trained checkpoint at %s" % CHECKPOINT)
        print("run:  python train_v2.py")
        print("until then the lobes are random and any answer is noise")
        system.to(system.device_name)
        system.eval()
    system.load_concepts()
    if system.trained and system.calibrate():
        print("recognition thresholds calibrated from this model: "
              "similarity %.3f, confidence %.3f" % system.recognition)
    return system


__all__ = ["BrainV2System", "Thought", "build", "CHECKPOINT"]
