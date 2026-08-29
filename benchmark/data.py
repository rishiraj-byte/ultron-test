"""
data.py -- task definitions and frozen-encoder feature caching.

Every model in the comparison consumes the SAME cached features, produced once
by the same frozen encoders. That is deliberate and it is what makes the
experiment about architecture:

  * the frozen backbones are identical for every variant, so no result can be
    attributed to one model getting a better feature extractor;
  * caching means the expensive part runs once, so five seeds x five variants
    x several ablations is minutes of CPU rather than hours;
  * the cache is keyed by task and split and stored as .npz, so a run is
    reproducible and interruptible.

TASKS

  gridnav    embodied navigation. image + sensor + text -> 4 actions.
             The only task that exercises every lobe. See gridworld.py for why
             no single modality can solve it.
  cifar100   vision only. 100 classes. Included precisely because the modular
             system is NOT expected to win here -- with one modality there is
             nothing to route, and reporting that honestly is what makes the
             multimodal result credible.
  news20     text only. 20 classes. Same role as cifar100, on the other side.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

import config

CACHE_DIR = config.PROJECT_ROOT / "data" / "features"
DATA_ROOT = str(config.PROJECT_ROOT / "data")


@dataclass
class TaskData:
    """Cached features for one task, already split."""

    name: str
    num_classes: int
    modalities: tuple[str, ...]
    #  split -> modality -> (N, raw_dim) float32
    features: dict[str, dict[str, np.ndarray]]
    #  split -> (N,) int64
    labels: dict[str, np.ndarray]
    #  split -> (N, num_classes) bool; several actions can be optimal at once
    accepts: dict[str, np.ndarray] | None = None
    #  split -> (N, flat_dim) float32, the monolithic baseline's input
    flat: dict[str, np.ndarray] | None = None

    def raw_dims(self) -> dict[str, int]:
        first = self.features["train"]
        return {m: int(first[m].shape[1]) for m in self.modalities}

    def size(self, split: str) -> int:
        return len(self.labels[split])


# ---------------------------------------------------------------------------
# Encoding helpers -- the frozen backbones, batched
# ---------------------------------------------------------------------------


def _encode_images(images, batch: int = 64, note: str = "") -> np.ndarray:
    """PIL images / arrays -> frozen ResNet-18 features (N, 512)."""
    from modules.encoders import VisionEncoder

    encoder = VisionEncoder()
    encoder._ensure_loaded()
    out: list[np.ndarray] = []
    started = time.perf_counter()

    for i in range(0, len(images), batch):
        chunk = images[i : i + batch]
        pixels = torch.cat([encoder.preprocess(im) for im in chunk])
        with torch.no_grad():
            out.append(encoder.backbone(pixels).numpy())
        done = min(i + batch, len(images))
        rate = done / max(time.perf_counter() - started, 1e-9)
        print(f"\r  {note} images {done}/{len(images)}  ({rate:.0f}/s)",
              end="", flush=True)
    print()
    return np.concatenate(out).astype(np.float32)


def _encode_texts(texts: list[str], batch: int = 256, note: str = "") -> np.ndarray:
    """Strings -> frozen MiniLM sentence embeddings (N, 384)."""
    from modules.encoders import TextEncoder

    encoder = TextEncoder()
    encoder._ensure_loaded()
    out: list[np.ndarray] = []
    started = time.perf_counter()

    for i in range(0, len(texts), batch):
        with torch.no_grad():
            out.append(encoder.encode_raw(texts[i : i + batch]).numpy())
        done = min(i + batch, len(texts))
        rate = done / max(time.perf_counter() - started, 1e-9)
        print(f"\r  {note} texts {done}/{len(texts)}  ({rate:.0f}/s)",
              end="", flush=True)
    print()
    return np.concatenate(out).astype(np.float32)


# ---------------------------------------------------------------------------
# Tasks
# ---------------------------------------------------------------------------


def build_gridnav(n_train: int = 12000, n_test: int = 3000, seed: int = 0) -> TaskData:
    from benchmark.gridworld import NUM_ACTIONS, GridWorld, flat_features

    splits = {}
    labels = {}
    accepts = {}
    flats = {}

    for split, n, world_seed in (("train", n_train, seed), ("test", n_test, seed + 5000)):
        world = GridWorld(seed=world_seed)
        obs = world.generate(n)
        print(f"  {split}: {n} navigation states")

        splits[split] = {
            "image": _encode_images([o.image for o in obs], note=f"{split}"),
            "text": _encode_texts([o.text for o in obs], note=f"{split}"),
            "sensor": np.stack([o.sensor for o in obs]).astype(np.float32),
        }
        labels[split] = np.array([o.action for o in obs], dtype=np.int64)

        mask = np.zeros((n, NUM_ACTIONS), dtype=bool)
        for i, o in enumerate(obs):
            mask[i, list(o.optimal)] = True
        accepts[split] = mask
        flats[split] = np.stack([flat_features(o) for o in obs]).astype(np.float32)

    return TaskData(
        name="gridnav",
        num_classes=NUM_ACTIONS,
        modalities=("image", "text", "sensor"),
        features=splits,
        labels=labels,
        accepts=accepts,
        flat=flats,
    )


def build_cifar100(n_train: int | None = None, n_test: int | None = None) -> TaskData:
    import torchvision.datasets as tvd

    splits, labels, flats = {}, {}, {}
    for split, train, limit in (("train", True, n_train), ("test", False, n_test)):
        ds = tvd.CIFAR100(root=DATA_ROOT, train=train, download=False)
        images = [ds[i][0] for i in range(len(ds))]
        targets = np.array(ds.targets, dtype=np.int64)
        if limit:
            images, targets = images[:limit], targets[:limit]
        print(f"  {split}: {len(images)} images")

        splits[split] = {"image": _encode_images(images, note=split)}
        labels[split] = targets
        # A single-modality task: the monolithic baseline sees the same frozen
        # features, so the only difference is what happens downstream.
        flats[split] = splits[split]["image"]

    return TaskData(
        name="cifar100",
        num_classes=100,
        modalities=("image",),
        features=splits,
        labels=labels,
        flat=flats,
    )


def build_news20() -> TaskData:
    from sklearn.datasets import fetch_20newsgroups

    splits, labels, flats = {}, {}, {}
    for split, subset in (("train", "train"), ("test", "test")):
        bunch = fetch_20newsgroups(
            subset=subset,
            remove=("headers", "footers", "quotes"),
            data_home=DATA_ROOT,
        )
        # A handful of documents are empty once headers/quotes are stripped;
        # they carry no signal and would only add label noise.
        keep = [i for i, d in enumerate(bunch.data) if d.strip()]
        docs = [bunch.data[i] for i in keep]
        targets = np.array(bunch.target, dtype=np.int64)[keep]
        print(f"  {split}: {len(docs)} documents")

        splits[split] = {"text": _encode_texts(docs, note=split)}
        labels[split] = targets
        flats[split] = splits[split]["text"]

    return TaskData(
        name="news20",
        num_classes=20,
        modalities=("text",),
        features=splits,
        labels=labels,
        flat=flats,
    )


def build_audio(n_train: int = 480, n_test: int = 160, seed: int = 0) -> TaskData:
    """A small SYNTHETIC acoustic task, for pathway pretraining only.

    There is no audio corpus in this project, and the MGN relay plus the audio
    projection would otherwise never receive a gradient -- roughly 100k
    parameters left at their random initialisation in a system advertised as
    trained. This gives that pathway sensible weights.

    It is NOT an audio benchmark and must never be reported as one. The classes
    are trivially separable by design (tones, sweeps, noise), so accuracy here
    measures nothing about the architecture. It exists so that "the audio path
    is trained" is a true statement rather than a hopeful one.
    """
    rng = np.random.default_rng(seed)
    sr = config.AUDIO_SAMPLE_RATE
    duration = 1.0
    t = np.linspace(0, duration, int(sr * duration), endpoint=False, dtype=np.float32)
    names = ("low tone", "high tone", "sweep up", "sweep down", "noise", "two tones")

    def make(kind: int) -> np.ndarray:
        jitter = float(rng.uniform(0.85, 1.15))
        if kind == 0:
            wave = np.sin(2 * np.pi * 220 * jitter * t)
        elif kind == 1:
            wave = np.sin(2 * np.pi * 1760 * jitter * t)
        elif kind == 2:
            f = np.linspace(200, 2000 * jitter, len(t))
            wave = np.sin(2 * np.pi * f * t / sr * sr / len(t) * len(t))
        elif kind == 3:
            f = np.linspace(2000 * jitter, 200, len(t))
            wave = np.sin(2 * np.pi * f * t / sr * sr / len(t) * len(t))
        elif kind == 4:
            wave = rng.standard_normal(len(t))
        else:
            wave = (np.sin(2 * np.pi * 330 * jitter * t)
                    + np.sin(2 * np.pi * 990 * jitter * t))
        wave = wave + 0.02 * rng.standard_normal(len(t))
        peak = np.abs(wave).max()
        return (0.5 * wave / peak).astype(np.float32) if peak > 0 else wave.astype(np.float32)

    from modules.encoders import AudioEncoder

    encoder = AudioEncoder()
    encoder._ensure_loaded()

    splits, labels = {}, {}
    for split, n in (("train", n_train), ("test", n_test)):
        kinds = rng.integers(0, len(names), size=n)
        vectors = []
        started = time.perf_counter()
        for i, kind in enumerate(kinds, 1):
            with torch.no_grad():
                vectors.append(encoder.encode_raw(make(int(kind))).numpy())
            if i % 40 == 0 or i == n:
                rate = i / max(time.perf_counter() - started, 1e-9)
                print(f"\r{split} clips {i}/{n}  ({rate:.1f}/s)", end="", flush=True)
        print()
        splits[split] = {"audio": np.concatenate(vectors).astype(np.float32)}
        labels[split] = kinds.astype(np.int64)

    return TaskData(
        name="audiotone",
        num_classes=len(names),
        modalities=("audio",),
        features=splits,
        labels=labels,
    )



BUILDERS = {
    "gridnav": build_gridnav,
    "cifar100": build_cifar100,
    "news20": build_news20,
    "audiotone": build_audio,
}


# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------


def _cache_path(name: str) -> Path:
    return CACHE_DIR / f"{name}.npz"


def save(task: TaskData) -> Path:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    blob: dict[str, np.ndarray] = {}
    for split, mods in task.features.items():
        for modality, values in mods.items():
            blob[f"x/{split}/{modality}"] = values
    for split, values in task.labels.items():
        blob[f"y/{split}"] = values
    if task.accepts:
        for split, values in task.accepts.items():
            blob[f"a/{split}"] = values
    if task.flat:
        for split, values in task.flat.items():
            blob[f"f/{split}"] = values
    blob["meta/num_classes"] = np.array([task.num_classes])
    blob["meta/modalities"] = np.array(list(task.modalities))

    path = _cache_path(task.name)
    np.savez_compressed(path, **blob)
    return path


def load(name: str) -> TaskData | None:
    path = _cache_path(name)
    if not path.exists():
        return None
    blob = np.load(path, allow_pickle=False)

    features: dict[str, dict[str, np.ndarray]] = {}
    labels, accepts, flat = {}, {}, {}
    for key in blob.files:
        parts = key.split("/")
        if parts[0] == "x":
            features.setdefault(parts[1], {})[parts[2]] = blob[key]
        elif parts[0] == "y":
            labels[parts[1]] = blob[key]
        elif parts[0] == "a":
            accepts[parts[1]] = blob[key]
        elif parts[0] == "f":
            flat[parts[1]] = blob[key]

    return TaskData(
        name=name,
        num_classes=int(blob["meta/num_classes"][0]),
        modalities=tuple(str(m) for m in blob["meta/modalities"]),
        features=features,
        labels=labels,
        accepts=accepts or None,
        flat=flat or None,
    )


def get(name: str, rebuild: bool = False, **kwargs) -> TaskData:
    """Load from cache, or build and cache."""
    if not rebuild:
        cached = load(name)
        if cached is not None:
            return cached
    if name not in BUILDERS:
        raise KeyError(f"unknown task {name!r}; known: {sorted(BUILDERS)}")
    print(f"building {name} (frozen encoders run once, then cached)...")
    task = BUILDERS[name](**kwargs)
    path = save(task)
    print(f"cached -> {path}  ({path.stat().st_size / 1e6:.1f} MB)")
    return task


__all__ = ["TaskData", "get", "load", "save", "BUILDERS", "CACHE_DIR"]
