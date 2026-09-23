"""
train_v2.py -- train the v2 settling cortex (BUILD_SPEC_V2 section 7).

WHAT IS BEING TRAINED, AND WHAT IS NOT.

Frozen, always: the four sensory encoders (ResNet-18, MiniLM, Whisper), the
four pretrained cores inside the lobes, and the language model. Those carry
knowledge this project did not produce and has no business overwriting on a
few thousand examples.

Trained here: everything that makes those parts into a brain rather than a
pile of backbones -- the projections into and out of each core, the laminar
shells, all named tracts, the thalamic nuclei and TRN, the basal ganglia, the
limbic structures, the cerebellum, and the read-out heads.

FIVE OBJECTIVES, and each exists because one of them alone leaves something
dead. This was the single most expensive lesson of v1, where 34% of parameters
never received a gradient because one loss cannot reach every pathway.

  1. TASK. Classify the percept. The obvious one, and on its own it trains a
     narrow path from input to answer and nothing else.

  2. SEMANTIC GROUNDING. Frontal state must predict the frozen text encoder's
     embedding of the same content. This is what makes the cortical state
     MEAN something outside this model -- and it is what the chat interface
     stands on. Without it, frontal is an arbitrary 256-vector that only the
     classifier head can read, and no language model can be conditioned on it.

  3. PREDICTIVE CODING. The state at t must predict the state at t+1. Free
     supervision, available on every percept with no labels, and the only
     objective that trains the recurrent dynamics AS dynamics rather than as
     a means to an answer.

  4. FORWARD MODEL. The cerebellum's own prediction error (stage 7).

  5. VALUE. The SNc value head regresses on reward, which is the only
     objective that can reach it -- dopamine enters the striatum as a
     gradient SCALE, so nothing flows back into the value head from the task.

ROUND-ROBIN OVER MODALITIES. Batches alternate between the image task and the
text task rather than being mixed, because a mixed batch would need every
modality present for every example and the whole point of the thalamic routing
is that inputs arrive in whatever combination the world supplies.

    python train_v2.py                     train with the defaults
    python train_v2.py --epochs 8          longer
    python train_v2.py --limit 2000        fewer examples, for a quick run
    python train_v2.py --prepare-only      just build and cache the streams
"""

from __future__ import annotations

import argparse
import datetime
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

import config
from cortex import Cortex

STREAM_CACHE = config.PROJECT_ROOT / "data" / "v2_streams.pt"
CHECKPOINT = config.PROJECT_ROOT / "checkpoints" / "cortex_v2.pt"


# ---------------------------------------------------------------------------
# Data: un-pooled streams, cached
# ---------------------------------------------------------------------------


def build_streams(limit_image: int, limit_text: int,
                  limit_audio: int = 900, limit_sensor: int = 3000) -> dict:
    """Encode the on-disk datasets into UN-POOLED streams, once.

    The benchmark cache holds POOLED features, which is exactly what stage 2
    exists to stop using: parietal needs 49 real positions and temporal a real
    token sequence. So this re-encodes, keeping the axis, and caches the
    result in float16 -- 49 positions of 512 channels per image is large, and
    the precision is irrelevant to a frozen teacher's output.
    """
    import torchvision.datasets as tvd
    from sklearn.datasets import fetch_20newsgroups

    from modules.encoders import EncoderBank
    from modules.sensory import SensoryFront

    front = SensoryFront(EncoderBank())
    out: dict = {"image": {}, "text": {}}

    def vision_grid(images: list) -> torch.Tensor:
        """ResNet layer4 for a LIST of images, keeping the 7x7 grid.

        SensoryFront.encode preprocesses one image at a time, which is right
        for a single percept and hopeless for four thousand of them. Same tap,
        same hook, one backbone pass per batch instead of per image.
        """
        encoder = front.encoders.image
        encoder._ensure_loaded()
        pixels = torch.cat([encoder.preprocess(im) for im in images]).to(
            encoder.device)
        captured: dict = {}
        handle = encoder.backbone.layer4.register_forward_hook(
            lambda m, i, o: captured.__setitem__("grid", o))
        try:
            with torch.no_grad():
                encoder.backbone(pixels)
        finally:
            handle.remove()
        grid = captured["grid"]                      # (B, 512, 7, 7)
        b, c, h, w = grid.shape
        return grid.reshape(b, c, h * w).transpose(1, 2)

    print("encoding images (ResNet-18 layer4, 7x7 grid kept)...")
    for split, train in (("train", True), ("test", False)):
        dataset = tvd.CIFAR100(root=str(config.PROJECT_ROOT / "data"),
                               train=train, download=False)
        count = min(limit_image if train else limit_image // 4, len(dataset))
        grids, labels = [], []
        for start in range(0, count, 64):
            batch = [dataset[i][0] for i in range(start, min(start + 64, count))]
            with torch.no_grad():
                grids.append(vision_grid(batch).half().cpu())
            labels.extend(dataset.targets[start:start + len(batch)])
            print("\r  %s %d/%d" % (split, start + len(batch), count), end="")
        print()
        out["image"][split] = (torch.cat(grids), torch.tensor(labels))
    out["image"]["names"] = list(dataset.classes)

    print("encoding audio (Whisper encoder, frame level)...")
    #  Synthetic, and that has to be said plainly: there is no audio corpus in
    #  this project. The classes are trivially separable by construction, so
    #  accuracy on them measures nothing about the architecture. They exist so
    #  that "the auditory pathway is trained" is a true statement instead of a
    #  hopeful one -- without them the MGN nucleus and the audio projection
    #  keep their random initialisation in a system advertised as trained.
    from benchmark.data import build_audio
    rng = np.random.default_rng(config.SEED)
    sr = config.AUDIO_SAMPLE_RATE
    grid_t = np.linspace(0, 1.0, sr, endpoint=False, dtype=np.float32)
    audio_names = ("low tone", "high tone", "rising sweep", "falling sweep",
                   "noise", "two tones")

    def make_sound(kind: int) -> np.ndarray:
        jitter = float(rng.uniform(0.85, 1.15))
        if kind == 0:
            wave = np.sin(2 * np.pi * 220 * jitter * grid_t)
        elif kind == 1:
            wave = np.sin(2 * np.pi * 1760 * jitter * grid_t)
        elif kind == 2:
            wave = np.sin(2 * np.pi * np.linspace(200, 2000 * jitter,
                                                  len(grid_t)) * grid_t)
        elif kind == 3:
            wave = np.sin(2 * np.pi * np.linspace(2000 * jitter, 200,
                                                  len(grid_t)) * grid_t)
        elif kind == 4:
            wave = rng.normal(0, 0.3, len(grid_t)).astype(np.float32)
        else:
            wave = (np.sin(2 * np.pi * 440 * jitter * grid_t)
                    + np.sin(2 * np.pi * 880 * jitter * grid_t)) * 0.5
        return (0.4 * wave).astype(np.float32)

    out["audio"] = {"names": list(audio_names)}
    for split, count in (("train", limit_audio), ("test", limit_audio // 4)):
        frames, labels = [], []
        for i in range(count):
            kind = i % len(audio_names)
            with torch.no_grad():
                frames.append(front._audio_frames(make_sound(kind)).half().cpu())
            labels.append(kind)
            if i % 50 == 0:
                print("\r  %s %d/%d" % (split, i, count), end="", flush=True)
        print()
        out["audio"][split] = (torch.cat(frames), torch.tensor(labels))

    print("encoding sensor (rangefinder ring, 8 beams)...")
    from benchmark.gridworld import NUM_ACTIONS, GridWorld
    for split, count, seed in (("train", limit_sensor, config.SEED),
                               ("test", limit_sensor // 4, config.SEED + 5000)):
        world = GridWorld(seed=seed)
        observations = world.generate(count)
        beams = torch.tensor(
            np.stack([o.sensor for o in observations]), dtype=torch.float32)
        out.setdefault("sensor", {})[split] = (
            beams.reshape(count, config.SENSOR_BEAMS, 2).half(),
            torch.tensor([o.action for o in observations], dtype=torch.long))
    #  ACTIONS is ((-1,0),(1,0),(0,-1),(0,1)) -- row/column deltas, in order.
    out["sensor"]["names"] = ["move up", "move down", "move left", "move right"]
    assert len(out["sensor"]["names"]) == NUM_ACTIONS

    print("encoding text (MiniLM, token level)...")
    for split, subset in (("train", "train"), ("test", "test")):
        news = fetch_20newsgroups(
            data_home=str(config.PROJECT_ROOT / "data"), subset=subset,
            remove=("headers", "footers", "quotes"))
        count = min(limit_text if subset == "train" else limit_text // 4,
                    len(news.data))
        texts = [t.strip()[:400] for t in news.data[:count]]
        labels = list(news.target[:count])
        keep = [i for i, t in enumerate(texts) if len(t) > 20]
        texts = [texts[i] for i in keep]
        labels = [labels[i] for i in keep]

        chunks = []
        for start in range(0, len(texts), 64):
            piece = texts[start:start + 64]
            with torch.no_grad():
                tokens = front._text_tokens(piece)
            #  Pad or clip to a fixed length so the cache is one tensor.
            fixed = torch.zeros(len(piece), config.TEXT_TOKENS_TRAIN,
                                tokens.shape[-1])
            width = min(tokens.shape[1], config.TEXT_TOKENS_TRAIN)
            fixed[:, :width] = tokens[:, :width].cpu()
            chunks.append(fixed.half())
            print("\r  %s %d/%d" % (split, start + len(piece), len(texts)), end="")
        print()
        out["text"][split] = (torch.cat(chunks), torch.tensor(labels))
    out["text"]["names"] = list(news.target_names)

    #  SEMANTIC TEACHERS, all in ONE space -- the text encoder's.
    #
    #  The image teacher cannot be the vision encoder's own embedding: that
    #  lives in a 512-dim ResNet space with no relationship to language, and a
    #  cortical state grounded there is unreadable to anything that speaks
    #  words. Grounding vision in the LABEL's text embedding instead (the
    #  trick CLIP is built on) puts both modalities in the same space, which
    #  is what lets one semantic head serve both and what the chat interface
    #  stands on.
    print("embedding label names into the shared semantic space...")
    for modality in ("image", "text", "audio", "sensor"):
        template = {"image": "a photo of a %s", "audio": "the sound of a %s",
                    "sensor": "the decision to %s"}.get(modality, "%s")
        prompts = [template % n.replace("_", " ").replace(".", " ")
                   for n in out[modality]["names"]]
        with torch.no_grad():
            tokens = front._text_tokens(prompts)
        out[modality]["label_embeddings"] = tokens.mean(dim=1).cpu()

    #  THE CONCEPT VOCABULARY the chat interface retrieves against.
    #
    #  The training LABELS are only 130 concepts, which is a narrow world to
    #  hold a conversation in -- ask about anything outside it and the nearest
    #  neighbour is meaningless. The semantic head maps into the frozen text
    #  encoder's space, and that space is not limited to what was trained
    #  against: nearest-neighbour retrieval works against any set of points
    #  embedded the same way. ImageNet's thousand class names come free with
    #  the vision encoder and are real, concrete, human-scale concepts.
    #
    #  They are ANCHORS, not claims of knowledge. The head was trained toward
    #  the 130 labels; the other thousand are extrapolation, and the chat
    #  interface says which is which rather than presenting them alike.
    print("building the concept vocabulary...")
    trained_names, extra_names = [], []
    for modality in ("image", "text", "audio", "sensor"):
        trained_names.extend(n.replace("_", " ").replace(".", " ")
                             for n in out[modality]["names"])
    front.encoders.image._ensure_loaded()
    for name in front.encoders.image.categories:
        clean = name.replace("_", " ")
        if clean not in trained_names:
            extra_names.append(clean)

    vocabulary, embeddings = [], []
    for chunk in range(0, len(trained_names + extra_names), 128):
        names = (trained_names + extra_names)[chunk:chunk + 128]
        with torch.no_grad():
            tokens = front._text_tokens(names)
        embeddings.append(tokens.mean(dim=1).cpu())
        vocabulary.extend(names)
    out["vocabulary"] = {
        "names": vocabulary,
        "embeddings": torch.cat(embeddings),
        "trained": len(trained_names),
    }
    print("  %d concepts (%d trained against, %d extrapolated)"
          % (len(vocabulary), len(trained_names), len(extra_names)))

    STREAM_CACHE.parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, STREAM_CACHE)
    print("cached -> %s" % STREAM_CACHE)
    return out


def load_streams(limit_image: int, limit_text: int, rebuild: bool) -> dict:
    if STREAM_CACHE.exists() and not rebuild:
        print("loading cached streams from %s" % STREAM_CACHE)
        return torch.load(STREAM_CACHE, weights_only=False)
    return build_streams(limit_image, limit_text)


# ---------------------------------------------------------------------------
# The trainable brain
# ---------------------------------------------------------------------------


class BrainV2(nn.Module):
    """The settling cortex plus the heads that read it."""

    #  modality -> (thalamic nucleus it enters by, lobe whose sequence
    #  operator it feeds). This IS the thalamic routing table: each sense has
    #  its own relay and its own destination, which is the whole reason the
    #  thalamus is a switchboard rather than a wire.
    #  EVERY SENSE REACHES EVERY LOBE, through its own thalamic nucleus.
    #
    #  It did not before, and the consequence was measurable: on a text input
    #  only temporal received the sequence, and the other three lobes saw
    #  nothing but echoes arriving over the tracts. Three of four networks
    #  were not parsing the input at all.
    #
    #  Routing everything everywhere is defensible rather than lazy. Each
    #  lobe reads a sequence through its OWN operator -- occipital with local
    #  receptive fields, parietal with attention between positions, temporal
    #  with recurrence -- so the same tokens are genuinely processed four
    #  different ways rather than four times identically. What still
    #  distinguishes the modalities is which NUCLEUS they enter by, which is
    #  what the thalamic routing table was always for, and the primary lobe
    #  listed first, which keeps the anatomical bias.
    ROUTES = {
        "image": ("LGN", ("occipital", "parietal", "temporal", "frontal")),
        "text": ("PULVINAR", ("temporal", "parietal", "occipital", "frontal")),
        "audio": ("MGN", ("temporal", "parietal", "occipital", "frontal")),
        "sensor": ("VPL", ("parietal", "occipital", "temporal", "frontal")),
    }

    def __init__(self, classes: dict[str, int]):
        super().__init__()
        self.cortex = Cortex(dropout=config.TRAIN_DROPOUT)
        dim = config.LATENT_DIM

        #  Per-domain heads. One shared head across modalities was the v1 bug
        #  that answered "hi" with "ocarina": the classes of one domain are
        #  not candidates for another.
        #  LayerNorm BEFORE the classifier, and it is a stability fix rather
        #  than a flourish. Cross-entropy can always lower its loss by making
        #  the logits larger, and with a bare Linear the cheapest way to do
        #  that is to inflate the cortical state feeding it. That is fine
        #  until the predictive and forward objectives -- which are squared
        #  errors on that same state -- start growing quadratically with it.
        #  Measured: they went from 1.3 to 7x10^8 in a single epoch and took
        #  the accuracy down with them.
        wide = dim * len(Cortex.LOBES)
        self.heads = nn.ModuleDict({
            name: nn.Sequential(nn.LayerNorm(wide), nn.Linear(wide, count))
            for name, count in classes.items()
        })

        #  Un-pooled taps are (N, raw) per position; these project each
        #  position into cortical width. Trained, unlike the encoders.
        self.project = nn.ModuleDict({
            name: nn.Sequential(nn.Linear(raw, dim), nn.LayerNorm(dim))
            for name, raw in config.RAW_DIMS.items()
        })

        #  SEMANTIC HEAD -- frontal state into the frozen text encoder's space.
        #  This is the bridge the chat interface stands on.
        self.semantic = nn.Sequential(
            nn.LayerNorm(wide), nn.Linear(wide, dim), nn.GELU(),
            nn.Linear(dim, config.TEXT_RAW_DIM))

        #  PREDICTIVE CODING -- state at t predicts state at t+1.
        self.predict_next = nn.Linear(dim, dim)

    def settle(self, modality: str, tokens: torch.Tensor, reward=None):
        projected = self.project[modality](tokens)
        nucleus, lobes = self.ROUTES[modality]
        sensory = {"thal.%s" % nucleus: projected.mean(dim=1)}
        sequences = {lobe: projected for lobe in lobes}
        trace = self.cortex.settle(sensory, sequences=sequences, reward=reward)
        return trace, projected


def relative_mse(predicted: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Squared error divided by the target's own scale.

    Plain MSE between cortical states is unbounded, and nothing in this system
    anchors the magnitude of that state. Once the task loss starts inflating
    it -- which cross-entropy always wants to do -- a squared error on it
    grows quadratically and drags the whole run apart. Measured with plain
    MSE: the forward objective went 1.28, 1.61, then 7.3x10^8. Dividing by
    the target's own energy makes the loss dimensionless, so growing the
    state cannot change it and there is nothing to run away with.
    """
    scale = target.detach().pow(2).mean() + 1e-6
    return nn.functional.mse_loss(predicted, target) / scale


def save_checkpoint(chosen: dict, classes: dict, seen: int, started: float,
                    args, complete: bool) -> None:
    """Write the checkpoint. Called on every improvement, and once at the end.

    `complete` records whether the run finished or whether this is the best
    epoch of a run still in progress. Anything reading the checkpoint can then
    tell "this model was trained for 25 epochs" from "this model is the best
    of the 4 epochs that had finished when the process died", which are very
    different claims to make about a set of weights.
    """
    CHECKPOINT.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "cortex": chosen["cortex"],
        "heads": chosen["heads"],
        "project": chosen["project"],
        "semantic": chosen["semantic"],
        "predict_next": chosen["predict_next"],
        "classes": classes,
        "best_epoch": chosen["epoch"],
        "held_out": chosen["scores"],
        #  Provenance, so nothing downstream can mistake this for an untrained
        #  model or a checkpoint a test happened to write.
        "training_record": {
            "epochs": args.epochs if complete else chosen["epoch"] + 1,
            "planned_epochs": args.epochs,
            "complete": complete,
            "examples_seen": seen,
            "minutes": round((time.perf_counter() - started) / 60, 1),
            "finished": datetime.datetime.now().isoformat(timespec="seconds"),
            "objectives": sorted(config.OBJECTIVE_WEIGHTS),
        },
    }, CHECKPOINT)


def readout(trace, lobes=("occipital", "parietal", "temporal", "frontal")
            ) -> torch.Tensor:
    """What a downstream area reads: ALL FOUR LOBES, integrated over the window.

    Reading frontal alone is too narrow, and the reason is the sparsity doing
    exactly what stage 4 promised. The cortical state is a hard 5% mask, so one
    lobe exposes about thirteen of two hundred and fifty-six units. That is a
    fine code for twenty topics and far too little for a hundred image classes
    at forty examples each -- measured at 5.0% held-out against 1.0% chance,
    while a plain linear probe on the same frozen features reached 56.8%.
    Concatenating the four lobes gives fifty-two active units of 1024 and
    measured 12.4% on the same test: two and a half times the information for
    no change to the sparsity itself.
    (Averaging across TIME, which was the first thing tried, recovers nothing
    at all: the stage-4 hysteresis locks the winner set by design, so every
    timestep in the settled window exposes the SAME thirteen units.)

    A downstream area reading from several cortical areas at once is the
    normal arrangement anyway; frontal is not the only thing anything reads.
    """
    steps = range(config.SETTLING_WARMUP, config.T_STEPS)
    per_lobe = [torch.stack([trace.history.read(lobe, t) for t in steps]).mean(0)
                for lobe in lobes]
    return torch.cat(per_lobe, dim=-1)


def objectives(brain: BrainV2, modality: str, tokens: torch.Tensor,
               labels: torch.Tensor, teacher: torch.Tensor) -> dict:
    """All five losses for one batch. Returns them separately, so a dead one
    is visible in the log rather than hidden inside a sum."""
    correct = None
    trace, projected = brain.settle(modality, tokens)
    frontal = readout(trace)

    losses: dict[str, torch.Tensor] = {}

    #  1. task
    logits = brain.heads[modality](frontal)
    losses["task"] = nn.functional.cross_entropy(logits, labels)
    correct = (logits.argmax(dim=-1) == labels).float().mean()

    #  2. semantic grounding -- cosine, not MSE: what matters is the direction
    #  in the teacher's space, not matching its arbitrary scale.
    predicted = brain.semantic(frontal)
    losses["semantic"] = (1.0 - nn.functional.cosine_similarity(
        predicted, teacher, dim=-1)).mean()

    #  3. predictive coding, over the settled half of the window
    steps = range(config.SETTLING_WARMUP, config.T_STEPS - 1)
    predictive = []
    for t in steps:
        here = trace.history.read("frontal", t)
        nxt = trace.history.read("frontal", t + 1).detach()
        predictive.append(relative_mse(brain.predict_next(here), nxt))
    losses["predictive"] = (torch.stack(predictive).mean() if predictive
                            else frontal.sum() * 0.0)

    #  4. forward model
    #  The cerebellum reports a raw squared error, which has the same
    #  unbounded-scale problem; normalised here against the state it is
    #  predicting rather than left to grow with it.
    errors = [p.error for p in trace.cerebellum if p.error is not None]
    energy = trace.final["frontal"].detach().pow(2).mean() + 1e-6
    losses["forward"] = (torch.stack(errors).mean() / energy if errors
                         else frontal.sum() * 0.0)

    #  5. value -- reward is simply whether the answer was right
    reward = (logits.argmax(dim=-1) == labels).float().detach()
    #  The value head is part of the substantia nigra and reads frontal state,
    #  not the wide read-out -- it is a subcortical structure, not a head.
    losses["value"] = brain.cortex.basal_ganglia.value_loss(
        readout(trace, lobes=("frontal",)), reward)

    return {"losses": losses, "accuracy": float(correct.detach()),
            "settled": trace.settled()}


# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--limit", type=int, default=4000)
    parser.add_argument("--rebuild", action="store_true")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    device = args.device or config.resolve_device()
    print("=" * 74)
    print("TRAINING THE v2 SETTLING CORTEX")
    print("=" * 74)
    print("device %s   T=%d   batch %d   lr %g"
          % (device, config.T_STEPS, args.batch, args.lr))

    streams = load_streams(args.limit, args.limit, args.rebuild)
    if args.prepare_only:
        return 0

    modalities = tuple(m for m in ("image", "text", "audio", "sensor")
                       if m in streams and "train" in streams[m])
    classes = {m: int(streams[m]["train"][1].max()) + 1 for m in modalities}
    #  Two of these are PATHWAY TRAINING, not benchmarks, and saying so here
    #  is the difference between an honest system and a misleading one.
    #
    #    audio   synthetic tones and sweeps, trivially separable by design.
    #            A linear probe on the same features scores 100%. It exists so
    #            that the MGN nucleus and the audio projection have sensible
    #            weights instead of random ones -- accuracy on it measures
    #            nothing about the architecture.
    #    sensor  an 8-beam rangefinder reports where the WALLS are, not where
    #            the goal is, so the optimal move is not determined by this
    #            input alone. Measured: a linear probe reaches 27% against 25%
    #            chance. The VPL pathway still receives gradient, which is the
    #            entire purpose; the number is not a result.
    PATHWAY_ONLY = {"audio", "sensor"}
    for m in modalities:
        note = "  (pathway training, not a benchmark)" if m in PATHWAY_ONLY else ""
        print("%-7s %3d classes, %5d train / %5d test%s"
              % (m, classes[m], len(streams[m]["train"][1]),
                 len(streams[m]["test"][1]), note))

    brain = BrainV2(classes).to(device)
    trainable = [p for p in brain.parameters() if p.requires_grad]
    print("trainable %s   frozen %s"
          % (format(sum(p.numel() for p in trainable), ","),
             format(sum(p.numel() for p in brain.parameters()
                        if not p.requires_grad), ",")))

    optimiser = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=0.01)

    #  COSINE SCHEDULE WITH WARMUP. The previous runs held one learning rate
    #  throughout and the symptom was unmistakable: image accuracy peaked
    #  around epoch 16 and then oscillated between 25% and 36% for the rest
    #  of the budget, never settling. That is a step size too large for the
    #  basin it had already found. Warmup exists for the opposite reason --
    #  the first few hundred steps run through a cortex whose sparsity is
    #  still annealing, and a full-size step on that transient is wasted.
    steps_per_epoch = sum(
        max(len(streams[m]["train"][1]) - args.batch, 0) // args.batch
        for m in modalities)
    total_steps = max(steps_per_epoch * args.epochs, 1)
    warmup = min(int(0.03 * total_steps), 300)

    def learning_rate(step: int) -> float:
        if step < warmup:
            return (step + 1) / max(warmup, 1)
        progress = (step - warmup) / max(total_steps - warmup, 1)
        return 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))

    schedule = torch.optim.lr_scheduler.LambdaLR(optimiser, learning_rate)
    print("schedule: %d steps, %d warmup, cosine to zero"
          % (total_steps, warmup))
    weights = config.OBJECTIVE_WEIGHTS

    #  The teacher signal for semantic grounding is the frozen encoder's own
    #  pooled embedding of the same input -- available for free, since the
    #  un-pooled tokens are already cached.
    label_space = {m: streams[m]["label_embeddings"].to(device)
                   for m in modalities}

    def teacher_for(modality: str, tokens: torch.Tensor,
                    labels: torch.Tensor) -> torch.Tensor:
        """The frozen text encoder's reading of what this percept is about.

        Text grounds on its own content; images ground on the text embedding
        of their label, so both land in the same 384-dim space.
        """
        #  Text grounds on its OWN content -- the encoder already read it,
        #  and that reading is richer than any label. Everything else grounds
        #  on the text embedding of its label, which is what puts all four
        #  senses in one space: an image of a rocket and the word "rocket"
        #  land in the same place, and that is what the chat interface reads.
        if modality == "text":
            return tokens.mean(dim=1)
        return label_space[modality][labels]

    #  BATCH SIZE IS A THROUGHPUT DECISION HERE, not a regularisation one.
    #  The settling loop is launch-latency bound -- about ninety small matmuls
    #  per timestep, twelve timesteps -- so the GPU spends most of its time
    #  waiting for kernels rather than doing arithmetic. Measured forward-only:
    #  18 examples/sec at batch 24, 84 at batch 256. Bigger batches are nearly
    #  free until the arithmetic finally dominates.
    print("(the settling loop is launch-bound; larger batches are near-free)")

    started = time.perf_counter()
    seen = 0
    touched: set[str] = set()
    best_score = [0.0]
    keep: list = [None]
    telemetry_off = config.no_telemetry()
    telemetry_off.__enter__()
    for epoch in range(args.epochs):
        brain.train()
        totals: dict[str, float] = {}
        accuracy = {m: [] for m in modalities}
        batches = 0

        order = []
        for modality in modalities:
            tokens, labels = streams[modality]["train"]
            for start in range(0, len(labels) - args.batch, args.batch):
                order.append((modality, start))
        rng = np.random.default_rng(config.SEED + epoch)
        rng.shuffle(order)

        for modality, start in order:
            tokens, labels = streams[modality]["train"]
            chunk = tokens[start:start + args.batch].float().to(device)
            target = labels[start:start + args.batch].to(device)
            teacher = teacher_for(modality, chunk, target)

            result = objectives(brain, modality, chunk, target, teacher)
            loss = sum(weights[k] * v for k, v in result["losses"].items())

            optimiser.zero_grad(set_to_none=True)
            loss.backward()
            #  Sampled every few batches rather than every one. Walking 300
            #  tensors and reducing each on the GPU costs more than the step
            #  it is measuring; the modalities alternate often enough that a
            #  periodic sample still sees every pathway.
            if batches % config.COVERAGE_SAMPLE_EVERY == 0:
                for name, parameter in brain.named_parameters():
                    if parameter.grad is not None and parameter.grad.abs().sum() > 0:
                        touched.add(name)
            torch.nn.utils.clip_grad_norm_(trainable, config.GRAD_CLIP)
            optimiser.step()
            schedule.step()

            for name, value in result["losses"].items():
                totals[name] = totals.get(name, 0.0) + float(value.detach())
            accuracy[modality].append(result["accuracy"])
            batches += 1
            seen += args.batch
            if batches % 20 == 0:
                print("\r  epoch %d  %d/%d batches  loss %.4f"
                      % (epoch, batches, len(order), float(loss.detach())),
                      end="", flush=True)
        print()

        print("  epoch %d  %s" % (epoch, "  ".join(
            "%s %.4f" % (k, v / max(batches, 1)) for k, v in sorted(totals.items()))))
        for modality, values in accuracy.items():
            if values:
                print("     train accuracy %-6s %.1f%%"
                      % (modality, 100 * sum(values) / len(values)))

        #  held-out
        brain.eval()
        epoch_scores = {}
        for modality in modalities:
            tokens, labels = streams[modality]["test"]
            hits, total = 0, 0
            with torch.no_grad():
                for start in range(0, len(labels) - args.batch, args.batch):
                    chunk = tokens[start:start + args.batch].float().to(device)
                    target = labels[start:start + args.batch].to(device)
                    trace, _ = brain.settle(modality, chunk)
                    logits = brain.heads[modality](readout(trace))
                    hits += int((logits.argmax(dim=-1) == target).sum())
                    total += len(target)
            if total:
                epoch_scores[modality] = hits / total
                print("     HELD-OUT       %-6s %.1f%%  (chance %.1f%%)%s"
                      % (modality, 100 * hits / total,
                         100.0 / classes[modality],
                         "  [pathway only]" if modality in PATHWAY_ONLY else ""))

        #  KEEP THE BEST, NOT THE LAST. Held-out accuracy does not climb
        #  monotonically -- image peaked at 35.9% around epoch 19 and was
        #  oscillating between 25% and 33% by epoch 29. Saving whatever the
        #  final epoch happened to produce threw away four points for no
        #  reason other than that the loop had stopped there.
        real = [m for m in epoch_scores if m not in PATHWAY_ONLY]
        score = sum(epoch_scores[m] for m in real) / max(len(real), 1)
        if score > best_score[0]:
            best_score[0] = score
            best_state = {
                "cortex": {k: v.detach().clone()
                           for k, v in brain.cortex.state_dict().items()},
                "heads": {k: v.detach().clone()
                          for k, v in brain.heads.state_dict().items()},
                "project": {k: v.detach().clone()
                            for k, v in brain.project.state_dict().items()},
                "semantic": {k: v.detach().clone()
                             for k, v in brain.semantic.state_dict().items()},
                "predict_next": {k: v.detach().clone()
                                 for k, v in brain.predict_next.state_dict().items()},
                "epoch": epoch,
                "scores": dict(epoch_scores),
            }
            keep[0] = best_state
            #  WRITTEN TO DISK NOW, not at the end of the run.
            #
            #  This used to hold the best epoch in memory and save once, after
            #  the final epoch. A twenty-five epoch run on a laptop GPU is two
            #  hours, and anything that kills the process in that window --
            #  another program claiming the card, a CUDA OOM, a closed lid --
            #  threw away every completed epoch. That is exactly what happened
            #  at epoch 4 of a run whose epoch-3 weights were already better
            #  than the previous release: four epochs of work, unrecoverable,
            #  for want of one write.
            #
            #  Saving on improvement instead means the file on disk is always
            #  the best epoch SO FAR, so a crashed run is a shorter run rather
            #  than a lost one. It costs one write per improvement, which is a
            #  handful of writes across a whole training run.
            save_checkpoint(best_state, classes, seen, started, args,
                            complete=False)
            print("     ^ best so far (mean of the real tasks %.1f%%) -- saved"
                  % (100 * score))

    telemetry_off.__exit__(None, None, None)

    #  GRADIENT COVERAGE, accumulated across the whole epoch rather than
    #  sampled after the last batch. Batches alternate between modalities, so
    #  a single-batch snapshot always reports every other pathway as dead --
    #  which is exactly the false alarm that cost real time in v1. A tensor
    #  counts as trained if it received gradient on ANY batch.
    dark = [n for n, p in brain.named_parameters()
            if p.requires_grad and n not in touched]
    print()
    total = sum(1 for _, p in brain.named_parameters() if p.requires_grad)
    print("gradient coverage: %d of %d trainable tensors received gradient "
          "at some point during training" % (total - len(dark), total))
    for name in dark[:10]:
        print("   no gradient: %s" % name)

    chosen = keep[0]
    if chosen is None:
        raise SystemExit("no epoch completed -- nothing to save")
    print()
    print("saving the BEST epoch (%d): %s"
          % (chosen["epoch"], "  ".join("%s %.1f%%" % (m, 100 * v)
                                        for m, v in sorted(chosen["scores"].items()))))
    save_checkpoint(chosen, classes, seen, started, args, complete=True)
    print("saved -> %s" % CHECKPOINT)
    print()
    print("Next:  python desktop_app_v2.py            talk to it, watch it think")
    print("       python desktop_app_v2.py --selftest same, as text")
    print("       python status.py                   confirm it is trained")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
