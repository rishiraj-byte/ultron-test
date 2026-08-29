"""
train.py -- multi-task pretraining for the lobes. PROJECT.md build order step 7.

WHAT CHANGED AND WHY. The first version of this file trained the image path
only. Measured consequence: 34% of the trainable parameters -- the whole
temporal lobe, the text and audio projections, and the MGN and pulvinar relay
nuclei -- never received a single gradient. A checkpoint from it described a
system as "trained" while a third of it was still at its random
initialisation. This version drives every pathway.

THE FOUR OBJECTIVES, and which parts of the brain each one reaches:

  vision      CIFAR-100 images -> the frozen encoder's own ImageNet logits.
              Distillation, so no hand-labelling: the teacher head already
              exists inside ResNet-18. Trains LGN, occipital, parietal,
              frontal, the image projection and the main output head.

  language    20 Newsgroups -> topic. Trains pulvinar, temporal, frontal and
              the text projection.

  navigation  GridNav (image + rangefinder + instruction) -> action. The only
              objective that drives VPL, the sensor projection, and frontal
              fusion with three simultaneous streams.

  acoustic    A small SYNTHETIC tone task. Trains MGN, the audio projection
              and temporal. It is pathway initialisation, not an audio
              benchmark, and is labelled as such wherever it appears.

AUXILIARY HEADS. Only the vision objective writes to the system's real output
head, which stays ImageNet-1k -- that is the answer space the dashboard
reports. The other three objectives use small heads that exist for training
only and are NOT part of the shipped system. Their job is to put a gradient
through lobes that would otherwise stay random; the representations they shape
are the point, not their predictions.

Training runs entirely on cached frozen-encoder features, so the expensive
backbones run once (see benchmark/data.py) and this is minutes on CPU.

Usage
    python train.py                       all four objectives
    python train.py --epochs 30
    python train.py --tasks vision language
"""

from __future__ import annotations

import argparse
import random
import sys
import time
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import config
from benchmark import data as benchdata
from modules.pipeline import BrainAI, build_brain

#  How hard to push the go/no-go gate relative to the task losses. Low enough
#  that gating never overwhelms the objective it is gating.
GATE_WEIGHT = 0.3
GATE_BATCH = 64      # examples used for the gate's extra forward pass

#  objective -> the answer space it teaches the system to speak
DOMAIN_FOR = {
    "language": "topic",
    "navigation": "action",
    "acoustic": "acoustic",
}

#  objective -> human-readable class names for that answer space
LABELS = {
    "language": lambda t: _news_labels(t.num_classes),
    "navigation": lambda t: list(_action_labels()),
    "acoustic": lambda t: ["low tone", "high tone", "sweep up", "sweep down",
                           "noise", "two tones"][: t.num_classes],
}


def _news_labels(n: int) -> list[str]:
    try:
        from sklearn.datasets import fetch_20newsgroups
        names = fetch_20newsgroups(subset="train", data_home=str(config.PROJECT_ROOT / "data")).target_names
        return [x.replace("_", " ") for x in names][:n]
    except Exception:
        return [f"topic {i}" for i in range(n)]



def _unclassifiable_texts(n: int = 900, seed: int = config.SEED) -> list[str]:
    """Short, empty or nonsense strings, for an explicit "none of the above".

    20 Newsgroups documents have a median length of 86 words and only 1% are
    under five. The topic head therefore never saw a greeting, and -- having no
    abstain class -- had to assign one of 20 topics to everything. Measured
    result: "hi" came back as comp.windows.x at 93% confidence.

    The earlier attempt to solve this at the GATE failed, and the reason is
    worth recording: it trained on dimension-shuffled embeddings, but "hi" is
    not a corrupted embedding. It is a perfectly valid MiniLM encoding of a
    real string that simply carries no topic. The negatives have to be real
    TEXT of the kind that should be refused, not damaged vectors.
    """
    rng = random.Random(seed)
    letters = "abcdefghijklmnopqrstuvwxyz"
    greetings = ["hi", "hey", "hello", "ok", "okay", "yes", "no", "thanks", "sure",
                 "yo", "hm", "huh", "wow", "lol", "bye", "please", "maybe", "what",
                 "cool", "nice", "yeah", "nope", "hi there", "good morning",
                 "how are you", "thank you", "see you", "never mind"]
    out: list[str] = []
    while len(out) < n:
        kind = rng.random()
        if kind < 0.35:
            out.append(rng.choice(greetings))
        elif kind < 0.6:                       # random letter salad
            out.append(" ".join(
                "".join(rng.choice(letters) for _ in range(rng.randint(3, 8)))
                for _ in range(rng.randint(1, 4))))
        elif kind < 0.8:                       # digits and punctuation
            out.append(" ".join(str(rng.randint(0, 9999)) for _ in range(rng.randint(1, 5))))
        else:                                  # a single common word, no context
            out.append(rng.choice(
                ["the", "and", "it", "that", "thing", "stuff", "one", "some",
                 "here", "there", "now", "then", "this"]))
    return out[:n]


def _action_labels():
    from benchmark.gridworld import ACTION_NAMES
    return ACTION_NAMES


#  objective -> (cached task name, which modalities it feeds)
OBJECTIVES = {
    "vision": ("cifar100", ("image",)),
    "language": ("news20", ("text",)),
    "navigation": ("gridnav", ("image", "text", "sensor")),
    "acoustic": ("audiotone", ("audio",)),
}


@dataclass
class Objective:
    name: str
    modalities: tuple[str, ...]
    features: dict[str, torch.Tensor]
    labels: torch.Tensor
    head: nn.Module | None          # None => use the system's own output head
    teacher: torch.Tensor | None    # distillation targets, vision only


# ---------------------------------------------------------------------------
# Forward pass through the REAL modules, with gradients
# ---------------------------------------------------------------------------


def forward_from_features(
    brain: BrainAI, features: dict[str, torch.Tensor], return_gate: bool = False
):
    """Cached encoder features -> the gated frontal decision.

    Mirrors pipeline.perceive() exactly: projection -> thalamic relay ->
    lobes (with parietal fed by occipital and/or VPL) -> frontal fusion ->
    basal-ganglia gate. If this ever diverges from perceive(), the system
    trains one topology and runs another, so the two must be read together.

    The backbones are skipped because `features` are already their output.
    """
    relayed: dict[str, list[torch.Tensor]] = {}
    for modality, values in features.items():
        embedding = brain.encoders[modality].project_features(values)
        routing = brain.thalamus(modality, embedding)
        target = routing.destination
        relayed.setdefault(target, []).append(routing.payload[target])

    merged = {
        lobe: (t[0] if len(t) == 1 else torch.stack(t).mean(0))
        for lobe, t in relayed.items()
    }

    streams: dict[str, torch.Tensor] = {}
    if "occipital" in merged:
        streams["occipital"] = brain.cortex.occipital(merged["occipital"])

    parietal_sources = []
    if "parietal" in merged:                      # VPL / somatosensory
        parietal_sources.append(merged["parietal"])
    if "occipital" in streams:                    # dorsal stream
        parietal_sources.append(streams["occipital"])
    if parietal_sources:
        driven = (parietal_sources[0] if len(parietal_sources) == 1
                  else torch.stack(parietal_sources).mean(0))
        streams["parietal"] = brain.cortex.parietal(driven)

    if "temporal" in merged:
        streams["temporal"] = brain.cortex.temporal(merged["temporal"])

    decision = brain.cortex.frontal(streams)

    striatal = brain.basal_ganglia.striatum(decision)
    gate = torch.sigmoid(
        brain.basal_ganglia.direct(striatal) - brain.basal_ganglia.indirect(striatal)
    )
    return (decision * gate, gate) if return_gate else decision * gate



def gate_loss(brain: BrainAI, features: dict[str, torch.Tensor]) -> torch.Tensor:
    """Teach the basal ganglia to CLOSE on input that carries no signal.

    Without this the gate is decorative. Trained only on valid examples it
    learns "always allow", and measurement bore that out: the gate sat at
    1.000 for the gibberish "asdkjh qwe zxcvb" and 0.977 for a real sentence --
    fractionally MORE open on nonsense than on signal. A gate that never closes
    is not a gate, and PROJECT.md section 2.1 asks it to allow OR suppress.

    So each batch is paired with a corrupted twin: the same features shuffled
    independently per dimension, which preserves each dimension's marginal
    statistics while destroying the joint structure that carries meaning. The
    gate is pushed open on the real batch and shut on the twin. Shuffling
    rather than adding noise matters -- a noise blob is trivially separable by
    magnitude alone, and the gate would learn a norm threshold instead of
    learning what coherent input looks like.
    """
    _, real = forward_from_features(brain, features, return_gate=True)

    corrupted = {}
    for name, values in features.items():
        idx = torch.stack(
            [torch.randperm(values.shape[0], device=values.device)
             for _ in range(values.shape[1])], dim=1
        )
        corrupted[name] = torch.gather(values, 0, idx)
    _, fake = forward_from_features(brain, corrupted, return_gate=True)

    return (
        F.binary_cross_entropy(real.clamp(1e-6, 1 - 1e-6), torch.ones_like(real))
        + F.binary_cross_entropy(fake.clamp(1e-6, 1 - 1e-6), torch.zeros_like(fake))
    )


def distillation_loss(student, teacher, temperature: float = 4.0):
    """KL(student || teacher) at a temperature.

    T > 1 softens both distributions so the student learns the teacher's
    ranking over all 1000 classes, not just its argmax -- which is the whole
    reason distillation transfers more than a hard label. T^2 keeps gradient
    magnitude comparable across temperatures.
    """
    s = F.log_softmax(student / temperature, dim=-1)
    t = F.softmax(teacher / temperature, dim=-1)
    return F.kl_div(s, t, reduction="batchmean") * (temperature ** 2)


# ---------------------------------------------------------------------------


def load_objectives(brain: BrainAI, wanted: list[str], limit: int | None) -> list[Objective]:
    objectives: list[Objective] = []

    for name in wanted:
        task_name, modalities = OBJECTIVES[name]
        task = benchdata.load(task_name)
        if task is None:
            print(f"  !! no cache for {task_name}; skipping {name}")
            print(f"     build it:  python -c \"from benchmark import data;"
                  f" data.get('{task_name}')\"")
            continue

        n = task.size("train")
        keep = slice(0, min(n, limit) if limit else n)
        # Cached features come off disk as numpy, i.e. on the CPU. The
        # projections they feed live wherever the brain does, so they must be
        # moved or the first matmul raises a device mismatch.
        device = brain.device_name
        features = {
            m: torch.from_numpy(task.features["train"][m][keep]).to(device)
            for m in modalities
        }
        labels = torch.from_numpy(task.labels["train"][keep]).to(device)
        n_classes = task.num_classes

        if name == "language":
            # Give the head somewhere to put input that has no topic.
            from modules.encoders import TextEncoder

            encoder = TextEncoder()
            encoder._ensure_loaded()
            junk = _unclassifiable_texts()
            with torch.no_grad():
                junk_feats = encoder.encode_raw(junk).to(device)
            features = {"text": torch.cat([features["text"], junk_feats])}
            labels = torch.cat([labels, torch.full((len(junk),), n_classes,
                                                   dtype=labels.dtype,
                                                   device=device)])
            n_classes += 1
            print(f"  {'':11} + {len(junk)} unclassifiable examples "
                  f"-> explicit abstain class")

        teacher = None
        head: nn.Module | None = None
        if name == "vision":
            # The teacher head lives inside the frozen encoder already.
            # 50k x 1000 float32 is 200 MB held for the whole run, on a
            # machine already at 82% RAM. These are soft targets, never
            # differentiated, so half precision costs nothing and halves it.
            with torch.no_grad():
                teacher = brain.encoders.image.reference_head(
                    features["image"]
                ).half()
        else:
            # Registered ON the output generator, so it ships in the checkpoint
            # and the running system can answer in this space. Previously these
            # were throwaway scaffolding, which is why text had no vocabulary.
            names = LABELS[name](task)
            if name == "language":
                names = list(names) + ["(no clear topic)"]
            head = brain.output.add_head(DOMAIN_FOR[name], names).to(brain.device_name)

        objectives.append(
            Objective(name, modalities, features, labels, head, teacher)
        )
        print(f"  {name:<11} {len(labels):>6} samples  "
              f"modalities {list(modalities)}  "
              f"{'distillation -> main head' if head is None else f'aux head {task.num_classes}-way'}")
    return objectives


def note_gradients(brain: BrainAI, seen: set[str]) -> None:
    """Record which tensors have a live gradient right now.

    Must be ACCUMULATED across a whole epoch, never read after a single batch:
    each batch belongs to one objective and drives only that objective's
    pathway, so a one-batch snapshot reports every other pathway as dead.
    """
    for name, p in brain.named_parameters():
        if not p.requires_grad or name in seen:
            continue
        if p.grad is not None and p.grad.abs().sum() > 0:
            seen.add(name)


def coverage_report(brain: BrainAI, seen: set[str]) -> tuple[int, int, list[str]]:
    """(params that trained, total trainable, names that never did).

    This is the check the first version of this file lacked, and its absence is
    why 34% of the system sat at its random initialisation while the checkpoint
    claimed to be trained.
    """
    total = trained = 0
    missing: list[str] = []
    for name, p in brain.named_parameters():
        if not p.requires_grad:
            continue
        total += p.numel()
        if name in seen:
            trained += p.numel()
        else:
            missing.append(name)
    return trained, total, missing


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", nargs="+", default=list(OBJECTIVES),
                        choices=list(OBJECTIVES))
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--limit", type=int, help="cap samples per objective")
    parser.add_argument("--out", default=str(config.DEFAULT_CHECKPOINT))
    args = parser.parse_args()

    torch.manual_seed(config.SEED)
    np.random.seed(config.SEED)
    random.seed(config.SEED)

    print("=" * 78)
    print("MULTI-TASK PRETRAINING -- every pathway, not just vision")
    print("=" * 78)

    # Dropout in the lobes. Without it the language objective drove its loss
    # to 0.0016 -- memorisation -- and a memorised head is confident about
    # everything, including input it has no basis to judge.
    brain = build_brain(load_checkpoint=False, dropout=0.15)
    print(f"device: {brain.device_name}")
    print("\nobjectives:")
    objectives = load_objectives(brain, args.tasks, args.limit)
    if not objectives:
        print("\nno objectives could be loaded.")
        return 1

    # The extra heads now live on the output generator, so trainable_parameters
    # already includes them -- collected after load_objectives for that reason.
    params = list(brain.trainable_parameters())

    optimiser = torch.optim.AdamW(params, lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimiser, T_max=max(args.epochs, 1)
    )

    counts = brain.parameter_count()
    print(f"\ntrainable in the brain: {sum(counts.values()):,}")
    print(f"auxiliary head params : "
          f"{sum(p.numel() for o in objectives if o.head for p in o.head.parameters()):,}"
          f"  (training scaffolding, not shipped)")

    header = "epoch  " + "".join(f"{o.name:>13}" for o in objectives) + f"{'lr':>11}"
    print(f"\n{header}\n" + "-" * len(header))

    brain.train()
    started = time.perf_counter()
    seen_gradients: set[str] = set()

    for epoch in range(1, args.epochs + 1):
        losses = {o.name: [] for o in objectives}

        # Round-robin over objectives within the epoch, so no single task's
        # gradients dominate a contiguous stretch of training.
        batches = []
        for objective in objectives:
            order = torch.randperm(len(objective.labels))
            for i in range(0, len(order), args.batch_size):
                batches.append((objective, order[i : i + args.batch_size]))
        random.shuffle(batches)

        for objective, index in batches:
            features = {m: v[index] for m, v in objective.features.items()}
            decision = forward_from_features(brain, features)

            if objective.head is None:
                loss = distillation_loss(
                    brain.output(decision), objective.teacher[index].float()
                )
            else:
                loss = F.cross_entropy(
                    objective.head(decision), objective.labels[index]
                )

            # The gate is trained on EVERY objective, so it learns what
            # coherent input looks like across all four modalities rather than
            # just one.
            # Capped subsample: the gate is a scalar decision, so a slice is
            # as informative as the full batch and costs a fraction of the
            # activations. The full-batch version was enough to OOM.
            sub = {k: v[:GATE_BATCH] for k, v in features.items()}
            loss = loss + GATE_WEIGHT * gate_loss(brain, sub)

            optimiser.zero_grad(set_to_none=True)
            loss.backward()
            if epoch == 1:
                note_gradients(brain, seen_gradients)
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            optimiser.step()
            losses[objective.name].append(float(loss.detach()))

        scheduler.step()
        row = f"{epoch:<7}" + "".join(
            f"{np.mean(losses[o.name]):>13.4f}" for o in objectives
        )
        print(row + f"{scheduler.get_last_lr()[0]:>11.2e}")

    print("-" * len(header))
    print(f"trained in {time.perf_counter() - started:.1f}s")

    # ---- did every pathway actually train? --------------------------
    trained_n, total_n, missing = coverage_report(brain, seen_gradients)
    print("\nGRADIENT COVERAGE (the check the first version lacked)")
    print(f"  {trained_n:,} / {total_n:,} trainable parameters "
          f"({100 * trained_n / max(total_n, 1):.1f}%) received a gradient")
    if missing:
        print(f"  {len(missing)} tensors never did:")
        for name in missing[:12]:
            print(f"    {name}")
    else:
        print("  every trainable tensor in the brain trained")

    brain.trained = True
    brain.eval()
    out = type(config.DEFAULT_CHECKPOINT)(args.out)
    brain.save(out)
    print(f"\nsaved -> {out}")
    print("The lobes are pretrained on all four pathways. Try:")
    print("    python verify.py --all")
    print("    python desktop_app.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
