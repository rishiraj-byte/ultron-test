"""
data.py -- what v2.B learns from and is tested on.

    images  CIFAR-100: 100 classes of 32x32 photos, on disk. Three disjoint
            sets: 250/class to train, 50/class held out from CIFAR's own
            training images to VALIDATE (this fits the memory readout and the
            amygdala, which must not see the test set), and CIFAR's 10,000
            test images to TEST.
    sounds  ESC-50 when present: 2,000 labelled 5-second environmental clips,
            50 classes (CC BY-NC 3.0). Folds 1-3 train, 4 validates, 5 tests.
            Otherwise SYNTHETIC tones -- trivially separable, so a number on
            them only says the auditory path is wired.
"""

from __future__ import annotations

import collections
import csv
import os
import pickle
import warnings
from pathlib import Path

import numpy as np

from v2b import config_v2b as C

TONE_NAMES = ("low tone", "high tone", "rising sweep", "falling sweep",
              "hiss of noise", "two-note chord")


# ---------------------------------------------------------------------------
# Images
# ---------------------------------------------------------------------------


def cifar_names() -> list[str]:
    with open(C.CIFAR_DIR / "meta", "rb") as handle:
        meta = pickle.load(handle, encoding="latin1")
    return [name.replace("_", " ") for name in meta["fine_label_names"]]


def _cifar_raw(split: str):
    with open(C.CIFAR_DIR / split, "rb") as handle, warnings.catch_warnings():
        #  CIFAR's pickles predate NumPy 2.4 and trip a harmless deprecation.
        warnings.simplefilter("ignore")
        raw = pickle.load(handle, encoding="latin1")
    images = raw["data"].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
    return np.ascontiguousarray(images), np.asarray(raw["fine_labels"], np.int64)


def cifar(part: str, per_class: int | None = None, seed: int = C.SEED):
    """part: "train" | "val" | "extra" | "test" -> (uint8 N,32,32,3, labels, ids).

    train, val and extra are disjoint slices of CIFAR's training images
    (extra is stored in memory, never trained on); test is CIFAR's own
    test set. Source ids ("cifar-train:1234") let the dashboard
    show the actual picture behind any memory.
    """
    if part == "test":
        images, labels = _cifar_raw("test")
        keep = np.arange(len(labels))
        if per_class is not None:
            rng = np.random.default_rng(seed)
            keep = np.sort(np.concatenate([
                rng.permutation(np.flatnonzero(labels == c))[:per_class]
                for c in range(100)]))
        return images[keep], labels[keep], ["cifar-test:%d" % i for i in keep]
    images, labels = _cifar_raw("train")
    rng = np.random.default_rng(seed)
    start, take = {"train": (0, C.TRAIN_PER_CLASS),
                   "val": (C.TRAIN_PER_CLASS, C.VAL_PER_CLASS),
                   "extra": (C.TRAIN_PER_CLASS + C.VAL_PER_CLASS, C.EXTRA_PER_CLASS)}[part]
    keep = np.sort(np.concatenate([
        rng.permutation(np.flatnonzero(labels == c))[start:start + take]
        for c in range(100)]))
    return images[keep], labels[keep], ["cifar-train:%d" % i for i in keep]


def cifar_image(source: str) -> np.ndarray | None:
    """The actual 32x32 picture behind a source id, for the memory view."""
    try:
        split, index = source.split(":")
        images, _ = _cifar_cached(split.replace("cifar-", ""))
        return images[int(index)]
    except Exception:                                         # noqa: BLE001
        return None


# ---------------------------------------------------------------------------
# Tiny-ImageNet: 200 ImageNet classes, 64x64 (Hugging Face: zh-plus/tiny-imagenet)
# ---------------------------------------------------------------------------

_TINY: dict = {}


def _tiny_files():
    from huggingface_hub import snapshot_download
    try:
        root = Path(snapshot_download("zh-plus/tiny-imagenet", repo_type="dataset",
                                      local_files_only=True))
    except Exception:                                         # noqa: BLE001
        return None
    return root


def have_tiny() -> bool:
    return _tiny_files() is not None


def _tiny_classes() -> dict:
    """Tiny-ImageNet label position -> vocabulary index, via WordNet lemmas."""
    import ast
    import json
    root = _tiny_files()
    wnids = next(iter(json.load(open(root / "dataset_infos.json")).values()))[
        "features"]["label"]["names"]
    source = (root / "classes.py").read_text(encoding="utf-8")
    lemmas = ast.literal_eval(source[source.index("{"):source.index("}") + 1])
    vocab = {n.lower(): i for i, n in enumerate(image_vocabulary())}
    out = {}
    for position, wnid in enumerate(wnids):
        if wnid in IMAGENETTE:                  # keep the never-trained test clean
            continue
        for lemma in lemmas[wnid].split(","):
            if lemma.strip().lower() in vocab:
                out[position] = vocab[lemma.strip().lower()]
                break
    return out


def _tiny_table(split: str):
    import pyarrow.parquet as pq
    if split not in _TINY:
        root = _tiny_files()
        path = next((root / "data").glob("%s-*.parquet" % split))
        _TINY[split] = pq.read_table(path)
    return _TINY[split]


def _tiny_decode(cell) -> np.ndarray:
    import io
    from PIL import Image
    data = cell["bytes"] if isinstance(cell, dict) else cell
    return np.asarray(Image.open(io.BytesIO(data)).convert("RGB"), np.uint8)


def tiny_pixels(sources: list[str]) -> np.ndarray:
    """Decode just these pictures (uint8 N,64,64,3). Encoding decodes batch by
    batch: all 90,000 at once needed ~2.2 GB of RAM at the peak, and this
    machine had 2.5 GB free."""
    for split in {s[5:].split(":")[0] for s in sources if s.startswith("tiny-")}:
        _tiny_table(split)          # in bulk, one full read beats 90,000 row-group reads
    return np.stack([picture(s) for s in sources])


_TINY_FILES: dict = {}


def _tiny_row(split: str, index: int):
    """One picture's bytes from the row group that holds it (~15 MB, ~34 ms),
    not the whole split (616 MB, 1.7 s): the dashboard froze for that long
    on its first hover over a Tiny-ImageNet memory."""
    import pyarrow.parquet as pq
    if split not in _TINY_FILES:
        path = next((_tiny_files() / "data").glob("%s-*.parquet" % split))
        handle = pq.ParquetFile(path)
        starts = np.cumsum([0] + [handle.metadata.row_group(g).num_rows
                                  for g in range(handle.num_row_groups)])
        _TINY_FILES[split] = (handle, starts, {})
    handle, starts, last = _TINY_FILES[split]
    group = int(np.searchsorted(starts, index, "right") - 1)
    if group not in last:
        last.clear()                # keep one row group: neighbours are often wanted together
        last[group] = handle.read_row_group(group, columns=["image"]).column("image")
    return last[group][index - int(starts[group])].as_py()


def tiny_imagenet(part: str, seed: int = C.SEED, decode: bool = True):
    """part: "train" | "val" | "test" -> (uint8 N,64,64,3, vocabulary indices, ids).
    With decode=False the first item is None -- use tiny_pixels(ids) per batch.

    train and val are disjoint slices of Tiny-ImageNet's training pictures
    (450 + 50 per class); test is its own 10,000-picture validation set.
    """
    classes = _tiny_classes()
    table = _tiny_table("valid" if part == "test" else "train")
    labels = table.column("label").to_numpy()
    rng = np.random.default_rng(seed)
    keep = []
    for position in classes:
        rows = np.flatnonzero(labels == position)
        if part != "test":
            rows = rng.permutation(rows)
            rows = (rows[:C.TINY_TRAIN_PER_CLASS] if part == "train"
                    else rows[C.TINY_TRAIN_PER_CLASS:C.TINY_TRAIN_PER_CLASS + C.TINY_VAL_PER_CLASS])
        keep.extend(rows.tolist())
    keep = np.sort(np.asarray(keep))
    split = "valid" if part == "test" else "train"
    ids = ["tiny-%s:%d" % (split, i) for i in keep]
    pictures = tiny_pixels(ids) if decode else None
    return pictures, np.asarray([classes[int(labels[i])] for i in keep], np.int64), ids


# ---------------------------------------------------------------------------
# ImageNet photos: the ImageNet classes v2.B had never trained on, as real
# photos (256 px; Hugging Face: benjamin-paine/imagenet-1k-256x256)
# ---------------------------------------------------------------------------

_INET: dict = {}


def have_inet() -> bool:
    return all((C.INET_DIR / ("%s.parquet" % s)).exists() for s in ("train", "test"))


def _inet_excluded() -> set:
    """ImageNet classes never fetched: Tiny-ImageNet's 200 (its test pictures
    come from ImageNet, so training on these classes' photos could leak them)
    and Imagenette's 10 (the never-trained test must stay never-trained)."""
    names = image_vocabulary()
    inet = imagenet_names()
    tiny = set(_tiny_classes().values()) if have_tiny() else set()
    #  Also any class without a name of its own: the merge is case-blind, so
    #  ImageNet's "cardigan" (the sweater) became "Cardigan" (the dog), and
    #  its photos would teach the dog's name for a sweater.
    return ({i for i, n in enumerate(inet) if n not in names or names.index(n) in tiny}
            | set(IMAGENETTE.values()) | set(C.INET_SKIP))


def fetch_imagenet(log=print, more: bool = False) -> None:
    """Fetch the first photos of each wanted class, row group by row group.

    The shards are sorted by class (shard 0 holds classes 0-24, ~1,300
    photos each), so a few whole shards would cover a few dozen classes;
    all 40 are 16.5 GB. A row group holds ~100 photos, so reading only each
    class's first row group over HTTP covers every class in ~1.3 GB.
    """
    import os
    import pyarrow as pa
    import pyarrow.parquet as pq
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")    # the xet backend stalled here
    from huggingface_hub import HfApi, HfFileSystem
    skip = _inet_excluded()
    wanted = {"train": C.INET_TRAIN_PER_CLASS + C.INET_VAL_PER_CLASS,
              "test": C.INET_TEST_PER_CLASS}
    #  more=True: the NEXT photos of each class, after the ones already
    #  fetched (the shards are sorted, so "first N" is deterministic).
    passed = {"more": C.INET_TRAIN_PER_CLASS + C.INET_VAL_PER_CLASS}
    if more:
        wanted = {"more": C.INET_MORE_PER_CLASS}
    files = [s.rfilename for s in HfApi().dataset_info(C.INET_REPO).siblings]
    fs = HfFileSystem()
    C.INET_DIR.mkdir(parents=True, exist_ok=True)
    plan = (("more", "train"),) if more else (("train", "train"), ("test", "validation"))
    for ours, theirs in plan:
        out = C.INET_DIR / ("%s.parquet" % ours)
        if out.exists():
            continue
        #  RESUMABLE: each shard's photos are saved as soon as it is read. The
        #  first full fetch took ~35 min and was killed at shard 37 of 40 with
        #  nothing written; a restart now skips the shards it already has.
        parts = C.INET_DIR / "parts"
        parts.mkdir(exist_ok=True)
        need = {c: wanted[ours] for c in range(1000) if c not in skip}
        before = {c: passed.get(ours, 0) for c in need}     # photos to pass over first
        shards = sorted(f for f in files if f.startswith("data/%s-" % theirs))
        done = []
        for shard in shards:
            saved = parts / ("%s-%s" % (ours, shard.split("/")[-1]))
            if saved.exists():
                for label in pq.read_table(saved, columns=["label"]).column("label").to_pylist():
                    need[label] = need.get(label, 0) - 1
                #  a resumed shard passed over its share of already-fetched photos too
                for c in list(before):
                    if c in need and need[c] < wanted[ours]:
                        before[c] = 0
                done.append(saved)
                continue
            images, labels = [], []
            with fs.open("datasets/%s/%s" % (C.INET_REPO, shard), "rb",
                         block_size=2 ** 20) as handle:
                parquet = pq.ParquetFile(handle)
                #  By name: "image" is a struct (bytes, path), so the label is
                #  leaf column 2, not field 1.
                column = next(i for i in range(parquet.metadata.num_columns)
                              if parquet.metadata.schema.column(i).path == "label")
                for g in range(parquet.num_row_groups):
                    stats = parquet.metadata.row_group(g).column(column).statistics
                    span = range(int(stats.min), int(stats.max) + 1)
                    if not any(need.get(c, 0) > 0 for c in span):
                        continue
                    rows = parquet.read_row_group(g, columns=["image", "label"])
                    for cell, label in zip(rows.column("image").to_pylist(),
                                           rows.column("label").to_pylist()):
                        if before.get(label, 0) > 0:
                            before[label] -= 1
                            continue
                        if need.get(label, 0) > 0:
                            need[label] -= 1
                            images.append(cell["bytes"])
                            labels.append(label)
            table = pa.table({"image": pa.array(images, pa.binary()),
                              "label": pa.array(labels, pa.int64())})
            pq.write_table(table, saved.with_suffix(".tmp"))
            os.replace(saved.with_suffix(".tmp"), saved)
            done.append(saved)
            log("  %s: %s -- %d photos, %d classes still short"
                % (ours, shard.split("/")[-1], len(labels), sum(1 for v in need.values() if v > 0)))
        table = pa.concat_tables([pq.read_table(f) for f in done])
        #  Small row groups: the dashboard reads one photo at a time. Written
        #  aside and renamed, so an existing file always means a complete one.
        pq.write_table(table, out.with_suffix(".tmp"), row_group_size=256)
        os.replace(out.with_suffix(".tmp"), out)
        if ours == "train" and (C.INET_DIR / "more.parquet").exists():
            #  "more" passed over THIS file's photos; a new one invalidates it.
            (C.INET_DIR / "more.parquet").unlink()
        for f in done:
            f.unlink()
        log("  %s: %d photos of %d classes -> %s"
            % (ours, table.num_rows, len(set(table.column("label").to_pylist())), out))


def _inet_table(split: str):
    import pyarrow.parquet as pq
    if split not in _INET:
        _INET[split] = pq.read_table(C.INET_DIR / ("%s.parquet" % split))
    return _INET[split]


def _inet_decode(data: bytes) -> np.ndarray:
    """256x256 RGB. A photo that is not exactly 256 px square is resized,
    so every batch stacks."""
    import io
    from PIL import Image
    image = Image.open(io.BytesIO(data)).convert("RGB")
    if image.size != (256, 256):
        image = image.resize((256, 256), Image.BICUBIC)
    return np.asarray(image, np.uint8)


def imagenet_photos(part: str, seed: int = C.SEED):
    """part: "train" | "val" | "test" -> (vocabulary indices, ids).

    train and val split each class's fetched photos (INET_TRAIN_PER_CLASS +
    INET_VAL_PER_CLASS); test is ImageNet's own validation photos.
    """
    names, inet = image_vocabulary(), imagenet_names()
    split = {"test": "test", "more": "more"}.get(part, "train")
    labels = _inet_table(split).column("label").to_numpy()
    rng = np.random.default_rng(seed)
    keep = []
    skip = _inet_excluded()
    for c in np.unique(labels):
        if int(c) in skip:
            continue
        rows = np.flatnonzero(labels == c)
        if part in ("train", "val"):
            rows = rng.permutation(rows)
            rows = (rows[:C.INET_TRAIN_PER_CLASS] if part == "train"
                    else rows[C.INET_TRAIN_PER_CLASS:])
        keep.extend(rows.tolist())
    keep = np.sort(np.asarray(keep, np.int64))
    vocab = np.asarray([names.index(inet[int(labels[i])]) for i in keep], np.int64)
    return vocab, ["inet-%s:%d" % (split, i) for i in keep]


def have_inet_more() -> bool:
    return (C.INET_DIR / "more.parquet").exists()


def inet_more_stamp() -> str:
    import hashlib
    path = C.INET_DIR / "more.parquet"
    parts = [str(sorted(_inet_excluded())),
             str((C.INET_MORE_PER_CLASS, C.INET_TRAIN_PER_CLASS, C.INET_VAL_PER_CLASS, C.SEED)),
             str(path.stat().st_size if path.exists() else "missing"),
             str((C.INET_DIR / "train.parquet").stat().st_size
                 if (C.INET_DIR / "train.parquet").exists() else "missing")]
    return hashlib.sha1("|".join(parts).encode()).hexdigest()[:16]


def inet_stamp() -> str:
    """What the encoded ImageNet splits were made from: the fetched files, the
    exclusions and the per-class counts. A re-fetch or a rule change must
    re-encode them -- their ids are row numbers in these files."""
    import hashlib
    parts = [str(sorted(_inet_excluded())), str((C.INET_TRAIN_PER_CLASS, C.INET_VAL_PER_CLASS,
                                                 C.INET_TEST_PER_CLASS, C.SEED))]
    for split in ("train", "test"):
        path = C.INET_DIR / ("%s.parquet" % split)
        stat = path.stat() if path.exists() else None
        parts.append("%s:%s" % (split, stat.st_size if stat else "missing"))
    return hashlib.sha1("|".join(parts).encode()).hexdigest()[:16]


def inet_pixels(sources: list[str]) -> np.ndarray:
    for split in {s[5:].split(":")[0] for s in sources if s.startswith("inet-")}:
        _inet_table(split)          # in bulk, one full read; one at a time, a row group
    return np.stack([picture(s) for s in sources])


_INET_FILES: dict = {}


def _inet_row(split: str, index: int) -> bytes:
    """One photo from its 256-row group, for the dashboard: the whole train
    table is ~0.6 GB, and loading it on a hover froze the UI (as Tiny's did)."""
    import pyarrow.parquet as pq
    if split not in _INET_FILES:
        handle = pq.ParquetFile(C.INET_DIR / ("%s.parquet" % split))
        starts = np.cumsum([0] + [handle.metadata.row_group(g).num_rows
                                  for g in range(handle.num_row_groups)])
        _INET_FILES[split] = (handle, starts)
    handle, starts = _INET_FILES[split]
    group = int(np.searchsorted(starts, index, "right") - 1)
    column = handle.read_row_group(group, columns=["image"]).column("image")
    return column[index - int(starts[group])].as_py()


def picture(source: str) -> np.ndarray | None:
    """The actual picture behind any stored source id (CIFAR or Tiny-ImageNet)."""
    if source.startswith("cifar"):
        return cifar_image(source)
    if source.startswith("tiny-"):
        try:
            split, index = source[5:].split(":")
            cell = (_TINY[split].column("image")[int(index)].as_py() if split in _TINY
                    else _tiny_row(split, int(index)))
            return _tiny_decode(cell)
        except Exception:                                     # noqa: BLE001
            return None
    if source.startswith("inet-"):
        try:
            split, index = source[5:].split(":")
            cell = (_INET[split].column("image")[int(index)].as_py() if split in _INET
                    else _inet_row(split, int(index)))
            return _inet_decode(cell)
        except Exception:                                     # noqa: BLE001
            return None
    return None


_CIFAR_CACHE: dict = {}


def _cifar_cached(split: str):
    if split not in _CIFAR_CACHE:
        _CIFAR_CACHE[split] = _cifar_raw(split)
    return _CIFAR_CACHE[split]


# ---------------------------------------------------------------------------
# Sounds
# ---------------------------------------------------------------------------


def have_esc50() -> bool:
    return (C.ESC50_DIR / "meta" / "esc50.csv").exists()


def audio_names() -> list[str]:
    if not have_esc50():
        return list(TONE_NAMES)
    rows = _esc50_rows()
    names = {}
    for row in rows:
        names[int(row["target"])] = row["category"].replace("_", " ")
    return [names[i] for i in range(len(names))]


def _esc50_rows() -> list[dict]:
    with open(C.ESC50_DIR / "meta" / "esc50.csv", newline="") as handle:
        return list(csv.DictReader(handle))


def audio(part: str) -> tuple[list[np.ndarray], np.ndarray, list[str]]:
    """part: "train" | "val" | "test" -> (waves at 16 kHz, labels, source ids)."""
    if not have_esc50():
        count = dict(zip(("train", "val", "test"), C.AUDIO_TONES))[part]
        seed = {"train": C.SEED, "val": C.SEED + 1, "test": C.SEED + 2}[part]
        waves, labels = tones(count, seed)
        return waves, labels, ["tone:%s:%d" % (part, i) for i in range(count)]
    from v2b.senses import read_audio
    folds = dict(zip(("train", "val", "test"), C.ESC50_FOLDS))[part]
    waves, labels, sources = [], [], []
    for row in _esc50_rows():
        if int(row["fold"]) in folds:
            path = C.ESC50_DIR / "audio" / row["filename"]
            waves.append(read_audio(path))
            labels.append(int(row["target"]))
            sources.append("esc50:%s" % row["filename"])
    return waves, np.asarray(labels, np.int64), sources


# ---------------------------------------------------------------------------
# More sounds: AudioSet's balanced set, labelled by v2.B's own sound names
# ---------------------------------------------------------------------------

_SND: dict = {}


def _hf_local(repo: str, pattern: str) -> list[Path]:
    from huggingface_hub import snapshot_download
    try:
        root = Path(snapshot_download(repo, repo_type="dataset", local_files_only=True))
    except Exception:                                         # noqa: BLE001
        return []
    return sorted(root.glob(pattern))


def have_more_sounds() -> bool:
    return bool(_hf_local(C.AUDIOSET_REPO, "data/*.parquet")
                and _hf_local(C.ONTOLOGY_REPO, "**/ontology.json")
                and _hf_local(C.FSD50K_REPO, "ground_truth/dev.csv"))


def _esc50_held_sources() -> set:
    """Freesound recordings behind ESC-50's validation and test folds. 229
    FSD50K clips are cut from these same recordings; training on them would
    be training on the test."""
    held = {f for part in ("val", "test") for f in dict(zip(("train", "val", "test"),
                                                           C.ESC50_FOLDS))[part]}
    return {row["src_file"] for row in _esc50_rows() if int(row["fold"]) in held}


def _fsd_rows():
    """Every usable FSD50K row: (split "train"/"val"/"eval", file, row, fname, [ids])."""
    import csv
    import pyarrow.parquet as pq
    if "fsd" not in _SND:
        root = _hf_local(C.FSD50K_REPO, "ground_truth/dev.csv")[0].parent.parent
        truth = {}
        for name, split_of in (("dev", None), ("eval", "eval")):
            with open(root / "ground_truth" / ("%s.csv" % name), newline="") as handle:
                for row in csv.DictReader(handle):
                    truth[(name, row["fname"])] = (split_of or row["split"], row["mids"].split(","))
        leaked = _esc50_held_sources()
        #  FSD50K's val/test must not hold the recordings ESC-50's TRAINING
        #  clips were cut from either: the model trained on those clips.
        every = {row["src_file"] for row in _esc50_rows()}
        rows = []
        for path in sorted((root / "data").glob("*.parquet")):
            names = pq.read_table(path, columns=["name"]).column("name").to_pylist()
            for r, name in enumerate(names):
                folder, file = name.split("/")
                fname = file.rsplit("_", 1)[0]
                key = ("dev" if folder.startswith("dev") else "eval", fname)
                if key in truth:
                    split, mids = truth[key]
                    if fname in (leaked if split == "train" else every):
                        continue
                    rows.append((split, path.name, r, fname, mids))
        _SND["fsd"] = rows
    return _SND["fsd"]


def _ontology():
    """AudioSet's ontology: node list (the label indices point into it),
    id -> node, and id -> every ancestor id."""
    import json
    if "onto" not in _SND:
        nodes = json.loads(_hf_local(C.ONTOLOGY_REPO, "**/ontology.json")[0].read_text())
        by_id = {n["id"]: n for n in nodes}
        parents: dict = {}
        for n in nodes:
            for child in n["child_ids"]:
                parents.setdefault(child, set()).add(n["id"])
        ancestors = {}
        for node_id in by_id:
            seen, todo = set(), list(parents.get(node_id, ()))
            while todo:
                p = todo.pop()
                if p not in seen:
                    seen.add(p)
                    todo.extend(parents.get(p, ()))
            ancestors[node_id] = seen
        _SND["onto"] = (nodes, by_id, parents, ancestors)
    return _SND["onto"]


CONDITIONS = "/t/dd00123"          # AudioSet: "Channel, environment and background"


def _sound_slot() -> dict:
    """Our sound name (lowercase) -> its vocabulary index. Cached: building
    the vocabulary reads AST's config, and one_name() asks per clip."""
    if "slot" not in _SND:
        offset = sound_offset()
        _SND["slot"] = {n.lower(): offset + i for i, n in enumerate(sound_vocabulary())}
    return _SND["slot"]


def _our_name(node: dict) -> str:
    """AudioSet's display name, cut as audioset_names() cuts it -- and, for
    the same sound under another word, ESC-50's name (SOUND_SYNONYMS)."""
    name = node["name"].split(",")[0].strip().lower()
    return C.SOUND_SYNONYMS.get(name, name)


def one_name(node_ids) -> int | None:
    """A clip's AudioSet labels -> ONE vocabulary index, or None.

    Clips carry several labels (a dog barking outside: Dog, Bark, Outside).
    Scene tags go ("Inside, small room" would become "inside" and merge
    different rooms); a label implied by a more specific one goes; Speech,
    Music, Noise and Silence give way to anything specific; what is left must
    agree on one name, or share an ancestor at most two levels up that is a
    vocabulary name. Anything else is dropped rather than guessed.
    """
    nodes, by_id, parents, ancestors = _ontology()
    slot = _sound_slot()
    esc = sound_offset() + len(audio_names())
    ids = sorted(i for i in set(node_ids) if i in by_id
                 and not by_id[i]["name"].startswith(("Inside,", "Outside,")))
    #  Recording conditions (echo, field recording, radio...) are how a sound
    #  was captured, not what it is: they give way to any other label.
    condition = [i for i in ids if CONDITIONS in ancestors[i]]
    if len(condition) < len(ids):
        ids = [i for i in ids if i not in condition]
    #  An ESC-50 name beats its own subtypes ({Sheep, Bleat} is "sheep",
    #  {Siren, Police car} is "siren"): the test asks for the ESC-50 name.
    esc_ids = [i for i in ids if slot.get(_our_name(by_id[i]), esc) < esc]
    ids = [i for i in ids if not any(e in ancestors[i] for e in esc_ids)]
    ids = [i for i in ids if not any(i in ancestors[j] for j in ids if j != i)]
    named = [i for i in ids if _our_name(by_id[i]) in slot]
    specific = [i for i in named if _our_name(by_id[i]) not in ("speech", "music", "noise",
                                                                  "silence")]
    named = specific or named
    names = {_our_name(by_id[i]) for i in named}
    if len(names) == 1:
        return slot[names.pop()]
    if len(names) > 1:
        def near(i):
            out, frontier = {}, {i}
            for depth in (1, 2):
                frontier = set().union(*(parents.get(f, set()) for f in frontier))
                for f in frontier:
                    out.setdefault(f, depth)
            return out
        reach = [near(i) for i in named]
        common = set(reach[0]).intersection(*reach[1:])
        common = sorted(c for c in common if _our_name(by_id[c]) in slot)
        if common:
            #  Ties broken by id: set order changed labels between processes.
            best = min(common, key=lambda c: (max(r[c] for r in reach), c))
            return slot[_our_name(by_id[best])]
    return None


def _audioset_rows():
    """Every AudioSet row: (split "train"/"eval", file, row, youtube id, [node ids])."""
    import pyarrow.parquet as pq
    if "aset" not in _SND:
        nodes = _ontology()[0]
        rows = []
        for path in _hf_local(C.AUDIOSET_REPO, "data/*.parquet"):
            split = "train" if path.name.startswith("train") else "eval"
            table = pq.read_table(path, columns=["path", "label"])
            for r, (p, labels) in enumerate(zip(table.column("path").to_pylist(),
                                                table.column("label").to_pylist())):
                yt = p.rsplit("/", 1)[-1][1:].rsplit(".", 1)[0]
                rows.append((split, path.name, r, yt, [nodes[k]["id"] for k in labels]))
        _SND["aset"] = rows
    return _SND["aset"]


def _held_out(key: str, seed: int) -> float:
    """A stable 0..1 number per clip: which held-out clips are val, which test."""
    import hashlib
    return int(hashlib.sha1(("%d:%s" % (seed, key)).encode()).hexdigest()[:8], 16) / 2 ** 32


def more_sounds(part: str, seed: int = C.SEED):
    """part: "train" | "val" | "atest" | "ftest" -> (vocabulary indices, ids).

    train: AudioSet's balanced training segments + FSD50K's dev-train clips,
    in one hash-shuffled order, at most SOUND_CAP per name (SOUND_BROAD_CAP
    for catch-alls) so both sources share each name. val: SOUND_VAL from
    AudioSet's evaluation segments + SOUND_VAL from FSD50K's dev-val. atest /
    ftest: SOUND_TEST from the rest of AudioSet's evaluation segments / from
    FSD50K's eval set -- two separate tests.
    """
    labels, ids = [], []
    count: dict = {}
    broad = {_sound_slot()[n] for n in C.SOUND_BROAD if n in _sound_slot()}
    aset = [("aset",) + r for r in _audioset_rows()]
    fsd = [("fsd",) + r for r in _fsd_rows()]
    held = sorted((r for r in aset if r[1] == "eval"), key=lambda r: _held_out(r[4], seed))
    cut = int(len(held) * 0.3)
    if part == "train":
        pool = [r for r in aset if r[1] == "train"] + [r for r in fsd if r[1] == "train"]
        groups = [sorted(pool, key=lambda r: _held_out(r[4], seed))]
    elif part == "val":
        groups = [held[:cut], sorted((r for r in fsd if r[1] == "val"),
                                     key=lambda r: _held_out(r[4], seed))]
    elif part == "atest":
        groups = [held[cut:]]
    else:
        groups = [sorted((r for r in fsd if r[1] == "eval"), key=lambda r: _held_out(r[4], seed))]
    limit = {"val": C.SOUND_VAL, "atest": C.SOUND_TEST, "ftest": C.SOUND_TEST}.get(part)
    for group in groups:
        taken = 0
        for source, split, name, r, key, node_ids in group:
            slot = one_name(node_ids)
            if slot is None:
                continue
            if part == "train":
                cap = C.SOUND_BROAD_CAP if slot in broad else C.SOUND_CAP
                if count.get(slot, 0) >= cap:
                    continue
                count[slot] = count.get(slot, 0) + 1
            labels.append(slot)
            ids.append("%s:%s:%d:%s" % (source, name, r, key))
            taken += 1
            if limit and taken >= limit:
                break
    return np.asarray(labels, np.int64), ids


def have_extra_sounds() -> bool:
    files = [p.name for p in _hf_local(C.AUDIOSET_EXTRA_REPO, "data/*.parquet")]
    return bool(files) and all(f.split("/")[-1] in files for f in C.AUDIOSET_EXTRA_FILES)


def extra_sounds(seed: int = C.SEED):
    """Rare names only: clips from AudioSet's unbalanced set whose one name has
    fewer than SOUND_EXTRA_UPTO training clips so far, up to that many. No
    catch-alls, no clip whose YouTube id is in the balanced or eval lists."""
    import pyarrow.parquet as pq
    slot = _sound_slot()
    nodes = _ontology()[0]
    taken = collections.Counter(more_sounds("train")[0].tolist())
    broad = {slot[n] for n in C.SOUND_BROAD + ("speech", "music", "silence", "noise")
             if n in slot}
    seen_yt = {r[3] for r in _audioset_rows()}
    rows = []
    for path in _hf_local(C.AUDIOSET_EXTRA_REPO, "data/*.parquet"):
        if not any(path.name == f.split("/")[-1] for f in C.AUDIOSET_EXTRA_FILES):
            continue
        table = pq.read_table(path, columns=["path", "label"])
        for r, (p, labels) in enumerate(zip(table.column("path").to_pylist(),
                                            table.column("label").to_pylist())):
            yt = p.rsplit("/", 1)[-1][1:].rsplit(".", 1)[0]
            if yt not in seen_yt:
                rows.append((path.name, r, yt, [nodes[k]["id"] for k in labels]))
    labels, ids = [], []
    for name, r, yt, node_ids in sorted(rows, key=lambda x: _held_out(x[2], seed)):
        s = one_name(node_ids)
        if s is None or s in broad or taken[s] >= C.SOUND_EXTRA_UPTO:
            continue
        taken[s] += 1
        labels.append(s)
        ids.append("asetu:%s:%d:%s" % (name, r, yt))
    return np.asarray(labels, np.int64), ids


def extra_sounds_stamp() -> str:
    import hashlib
    files = [p for p in _hf_local(C.AUDIOSET_EXTRA_REPO, "data/*.parquet")
             if any(p.name == f.split("/")[-1] for f in C.AUDIOSET_EXTRA_FILES)]
    parts = [more_sounds_stamp(), str(C.SOUND_EXTRA_UPTO), str(C.SOUND_BROAD)]
    parts += ["%s:%d" % (f.name, f.stat().st_size) for f in files]
    return hashlib.sha1("|".join(parts).encode()).hexdigest()[:16]


def more_sounds_stamp() -> str:
    """What the encoded sound splits were made from (files, rules, counts)."""
    import hashlib
    files = (_hf_local(C.AUDIOSET_REPO, "data/*.parquet")
             + sorted((_hf_local(C.FSD50K_REPO, "ground_truth/dev.csv")[0].parent.parent
                       / "data").glob("*.parquet")))
    parts = [C.SOUND_RULES, str(C.ESC50_FOLDS), str(sorted(C.SOUND_SYNONYMS.items())),
             str((C.SOUND_CAP, C.SOUND_BROAD_CAP, C.SOUND_BROAD, C.SOUND_VAL, C.SOUND_TEST,
                  C.SOUND_SECONDS, C.SEED))]
    parts += ["%s:%d" % (f.name, f.stat().st_size) for f in files]
    return hashlib.sha1("|".join(parts).encode()).hexdigest()[:16]


def _unused_more_sounds_v1(part: str, seed: int = C.SEED):
    """(the AudioSet-only first version, kept out of use)"""
    labels, ids = [], []
    count: dict = {}
    broad = set()
    order, limit = [], None
    for split, name, r, yt, node_ids in order:
        slot = one_name(node_ids)
        if slot is None:
            continue
        if part == "train":
            cap = C.SOUND_BROAD_CAP if slot in broad else C.SOUND_CAP
            if count.get(slot, 0) >= cap:
                continue
            count[slot] = count.get(slot, 0) + 1
        labels.append(slot)
        ids.append("aset:%s:%d:%s" % (name, r, yt))
        if limit and len(ids) >= limit:
            break
    return np.asarray(labels, np.int64), ids


def more_waves(sources: list[str]) -> list[np.ndarray]:
    """Decode these clips to float32 mono at AUDIO_SAMPLE_RATE."""
    import io
    import pyarrow.parquet as pq
    import soundfile
    from v2b.senses import _resample
    files = {("aset", p.name): p for p in _hf_local(C.AUDIOSET_REPO, "data/*.parquet")}
    fsd_root = _hf_local(C.FSD50K_REPO, "ground_truth/dev.csv")
    if fsd_root:
        files.update({("fsd", p.name): p
                      for p in (fsd_root[0].parent.parent / "data").glob("*.parquet")})
    files.update({("asetu", p.name): p for p in _hf_local(C.AUDIOSET_EXTRA_REPO, "data/*.parquet")})
    column = {"aset": "opus", "fsd": "audio", "asetu": "opus"}
    out = []
    for source in sources:
        kind, name, row = source.split(":")[:3]
        key = (kind, name)
        if key not in _SND:
            handle = pq.ParquetFile(files[key])
            starts = np.cumsum([0] + [handle.metadata.row_group(g).num_rows
                                      for g in range(handle.num_row_groups)])
            _SND[key] = (handle, starts, {})
        handle, starts, cache = _SND[key]
        group = int(np.searchsorted(starts, int(row), "right") - 1)
        if group not in cache:
            cache.clear()
            cache[group] = handle.read_row_group(group, columns=[column[kind]]).column(column[kind])
        cell = cache[group][int(row) - int(starts[group])].as_py()
        data = cell["bytes"] if isinstance(cell, dict) else cell
        wave, rate = soundfile.read(io.BytesIO(data), dtype="float32")
        if wave.ndim > 1:
            wave = wave.mean(axis=1)
        wave = np.asarray(wave[:int(C.SOUND_SECONDS * rate)], np.float32)
        out.append(_resample(wave, int(rate), C.AUDIO_SAMPLE_RATE))
    return out


def tones(count: int, seed: int) -> tuple[list[np.ndarray], np.ndarray]:
    """Synthetic sounds, 0.6-1.6 s, jittered pitch and loudness."""
    rng = np.random.default_rng(seed)
    rate = C.AUDIO_SAMPLE_RATE
    waves, labels = [], []
    for index in range(count):
        kind = index % len(TONE_NAMES)
        t = np.arange(int(rate * rng.uniform(0.6, 1.6)), dtype=np.float32) / rate
        jitter = float(rng.uniform(0.85, 1.15))
        if kind == 0:
            wave = np.sin(2 * np.pi * 220 * jitter * t)
        elif kind == 1:
            wave = np.sin(2 * np.pi * 1760 * jitter * t)
        elif kind == 2:
            wave = np.sin(2 * np.pi * np.cumsum(np.linspace(200, 2000 * jitter, t.size)) / rate)
        elif kind == 3:
            wave = np.sin(2 * np.pi * np.cumsum(np.linspace(2000 * jitter, 200, t.size)) / rate)
        elif kind == 4:
            wave = rng.normal(0, 0.5, t.size)
        else:
            wave = 0.5 * (np.sin(2 * np.pi * 440 * jitter * t) + np.sin(2 * np.pi * 660 * jitter * t))
        waves.append((float(rng.uniform(0.15, 0.6)) * wave).astype(np.float32))
        labels.append(kind)
    return waves, np.asarray(labels, np.int64)


# ---------------------------------------------------------------------------
# The vocabulary: every name frontal can answer with
# ---------------------------------------------------------------------------


def imagenet_names() -> list[str]:
    """The 1,000 ImageNet class names, as torchvision ships them."""
    from torchvision.models import ResNet50_Weights
    return list(ResNet50_Weights.IMAGENET1K_V1.meta["categories"])


def audioset_names() -> list[str]:
    """AudioSet's 527 sound classes -- read from AST's config, already on disk.
    Only the first phrase of each is kept: "Male speech, man speaking" ->
    "male speech"."""
    try:
        from transformers import AutoConfig
        config = AutoConfig.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593",
                                            local_files_only=True)
    except Exception:                                         # noqa: BLE001
        return []
    return [config.id2label[i].split(",")[0].strip().lower()
            for i in range(len(config.id2label))]


def _merge(first: list[str], more: list[str]) -> list[str]:
    seen = {n.lower() for n in first}
    out = list(first)
    for name in more:
        if name.lower() not in seen:
            seen.add(name.lower())
            out.append(name)
    return out


def image_vocabulary() -> list[str]:
    names = cifar_names()                   # first, so CIFAR label i is index i
    if "imagenet" in C.IMAGE_VOCABULARY:
        names = _merge(names, imagenet_names())
    return names


def sound_vocabulary() -> list[str]:
    names = audio_names()                   # first, so ESC-50 target t is offset + t
    if "audioset" in C.SOUND_VOCABULARY:
        names = _merge(names, audioset_names())
    return names


def vocabulary() -> tuple[list[str], list[str]]:
    """The names frontal starts with: things to see, then sounds to hear."""
    images, sounds = image_vocabulary(), sound_vocabulary()
    return images + sounds, ["image"] * len(images) + ["sound"] * len(sounds)


def sound_offset() -> int:
    """Where the sound names begin in the vocabulary."""
    return len(image_vocabulary())


# ---------------------------------------------------------------------------
# The open-vocabulary test: pictures of things v2.B never trained on
# ---------------------------------------------------------------------------

#  Imagenette's ten classes, by WordNet id -> ImageNet class index. None of the
#  ten is a CIFAR-100 class, so v2.B has never been trained on any of them.
IMAGENETTE = {"n01440764": 0, "n02102040": 217, "n02979186": 482, "n03000684": 491,
              "n03028079": 497, "n03394916": 566, "n03417042": 569, "n03425413": 571,
              "n03445777": 574, "n03888257": 701}


OPEN_PER_CLASS = 200          # pictures per never-trained class in the test


def _imagenette_parquet():
    """Imagenette (320 px) as one parquet file from the Hugging Face hub --
    fast.ai's own bucket delivered 12 KB/s here, the hub ~1.5 MB/s."""
    from huggingface_hub import hf_hub_download
    try:
        parquet = hf_hub_download("johnowhitaker/imagenette2-320",
                                  "data/train-00000-of-00001.parquet",
                                  repo_type="dataset", local_files_only=True)
        info = hf_hub_download("johnowhitaker/imagenette2-320", "dataset_infos.json",
                               repo_type="dataset", local_files_only=True)
        return parquet, info
    except Exception:                                         # noqa: BLE001
        return None, None


def have_imagenette() -> bool:
    return _imagenette_parquet()[0] is not None


def imagenette(per_class: int = OPEN_PER_CLASS, seed: int = C.SEED):
    """Pictures of ten things v2.B never trained on -> (PIL images, vocabulary
    indices, source ids). None of the ten is a CIFAR-100 class."""
    import io
    import json
    import pyarrow.parquet as pq
    from PIL import Image

    parquet, info = _imagenette_parquet()
    features = next(iter(json.load(open(info)).values()))["features"]
    label_names = features["label"]["names"]
    names = image_vocabulary()
    imagenet = imagenet_names()
    wanted = {}
    for position, label in enumerate(label_names):
        #  Labels are WordNet ids (n01440764) or readable names; map either.
        if label in IMAGENETTE:
            wanted[position] = names.index(imagenet[IMAGENETTE[label]])
        elif label in names:
            wanted[position] = names.index(label)
    table = pq.read_table(parquet).to_pandas()
    rng = np.random.default_rng(seed)
    images, labels, sources = [], [], []
    for position, slot in sorted(wanted.items()):
        rows = np.flatnonzero(table["label"].to_numpy() == position)
        for row in np.sort(rng.permutation(rows)[:per_class]):
            cell = table.iloc[int(row)]["image"]
            data = cell["bytes"] if isinstance(cell, dict) else cell
            images.append(Image.open(io.BytesIO(data)).convert("RGB"))
            labels.append(slot)
            sources.append("imagenette:%d" % int(row))
    return images, np.asarray(labels, np.int64), sources
