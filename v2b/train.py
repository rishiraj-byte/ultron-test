"""
train.py -- train v2.B's small parts. The pretrained networks never change.

    python -m v2b.train                       encode (cached) + all stages
    python -m v2b.train --fresh               re-encode first
    python -m v2b.train --watch               ...and open the live monitor window
    python -m v2b.train --cut "frontal->POT"  a variant trained WITHOUT a
                                              connection -- the honest way to
                                              measure what it contributes

FOUR STAGES

  1. ENCODE   every image and sound through the frozen networks, once, to
              data/v2b_cache/. Frozen networks give the same output every
              time, so running them every epoch would redo identical work.
  2. CORTEX   thalamic relays + TRN feedback, dorsal, STS, POT. Cross-entropy
              on frontal's scores after every pass (the last counts most).
  3. MEMORY   the hippocampus. Its store is every TRAINING example's POT
              vector. The entorhinal projection and CA1's sharpness are
              trained leave-one-out: each training item must be recognised by
              its neighbours with itself removed -- exactly the situation a
              new input is in. Then CA1's trust (how much frontal weighs
              memory against cortex) is fitted on the VALIDATION set, because
              on training data cortex is overconfident and memory would look
              useless.
  4. VALUE    the amygdala, on the validation set: predict whether the answer
              v2.B is about to give is correct.

  All numbers are then measured on the TEST set, which no stage has seen.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict

import numpy as np
import torch
import torch.nn.functional as F

from v2b import config_v2b as C
from v2b import data, senses
from v2b.cortex import Cortex, Dims, Features
from v2b.memory import CA1, Amygdala, Entorhinal
from v2b.wiring import NAMES, Wiring

SPLITS = ("train", "val", "test")
IMAGE_SPLITS = SPLITS + ("extra",)          # CIFAR slices
TINY_SPLITS = ("tiny_train", "tiny_val", "tiny_test")   # Tiny-ImageNet slices
INET_SPLITS = ("inet_train", "inet_val", "inet_test")   # ImageNet photos
REND_SPLITS = ("imr_train", "imr_test", "sketch_train", "sketch_test")   # renditions
SOUND_SPLITS = ("snd_train", "snd_val", "snd_atest", "snd_ftest")   # AudioSet + FSD50K
LIVE = C.ROOT / "results" / "v2b_live.jsonl"
_last_emit = [0.0]


def emit(event: dict, every: float = 0.0) -> None:
    """One line of live progress for the training monitor (v2b/monitor.py).
    `every` rate-limits chatty events (seconds between writes)."""
    now = time.time()
    if every and now - _last_emit[0] < every:
        return
    _last_emit[0] = now
    event["t"] = now
    LIVE.parent.mkdir(exist_ok=True)
    with open(LIVE, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(event) + "\n")


# ---------------------------------------------------------------------------
# Stage 1: encode once
# ---------------------------------------------------------------------------


def _fields(dims: Dims) -> tuple[dict, dict]:
    g = C.GRID * C.GRID
    image = {"occ_grid": (g, dims.vision), "it_grid": (g, dims.vision),
             "identity": (dims.embed,), "par_cells": (g, dims.parietal),
             "image_stats": (C.IMAGE_STATS,)}
    audio = {"aud_bins": (C.TEMPORAL_BINS, dims.audio), "aud_identity": (dims.audio,),
             "aud_joint": (dims.sound_embed,), "audio_stats": (dims.audio_stats,)}
    return image, audio


def _path(name: str):
    return C.CACHE_DIR / name


def _memmap(name, shape, count, mode):
    dtype = (np.int64 if name.endswith("labels") else
             np.float32 if name.endswith("stats") else np.float16)
    return np.lib.format.open_memmap(_path(name + ".npy"), mode=mode, dtype=dtype,
                                     shape=(count,) + tuple(shape))


def _manifest_wanted() -> dict:
    from v2b.brain import BrainV2B
    return {"models": BrainV2B.models(), "audio": "esc50" if data.have_esc50() else "tones",
            "grid": C.GRID, "occipital_share": C.OCCIPITAL_SHARE,
            "vocabulary": len(data.vocabulary()[0]),
            "train_per_class": C.TRAIN_PER_CLASS, "val_per_class": C.VAL_PER_CLASS}


def _save_sources(sources: dict) -> None:
    """Written aside and renamed: a full disk mid-write must not leave a torn
    sources.json, which every cache read depends on."""
    import os
    tmp = _path("sources.json.tmp")
    tmp.write_text(json.dumps(sources))
    os.replace(tmp, _path("sources.json"))


def _round5_ready() -> None:
    """Refuse to train 'with' round-5 data that is missing or still arriving,
    and refuse to start without disk headroom (D: shares the page file)."""
    import shutil
    if C.TRAIN_ON_INET and C.INET_MORE_PER_CLASS and not data.have_inet_more():
        fetching = any((C.INET_DIR / "parts").glob("more-*")) if (C.INET_DIR / "parts").exists() else False
        raise SystemExit("INET_MORE_PER_CLASS is set but data/imagenet_subset/more.parquet is %s "
                         "-- finish data.fetch_imagenet(more=True) or set it to 0"
                         % ("still being fetched" if fetching else "missing"))
    if (C.TRAIN_ON_MORE_SOUNDS and C.AUDIOSET_EXTRA_FILES and not data.have_extra_sounds()
            and not _path("aud_rare_train.labels.npy").exists()):
        raise SystemExit("AUDIOSET_EXTRA_FILES are set but not downloaded -- fetch them or set ()")
    if C.TRAIN_ON_INET and C.INET_MORE2_PER_CLASS and not data.have_inet_more2():
        fetching = any((C.INET_DIR / "parts").glob("more2-*")) if (C.INET_DIR / "parts").exists() else False
        raise SystemExit("INET_MORE2_PER_CLASS is set but data/imagenet_subset/more2.parquet is %s "
                         "-- finish data.fetch_imagenet(more=\"more2\") or set it to 0"
                         % ("still being fetched" if fetching else "missing"))
    if (C.TRAIN_ON_RENDITIONS and not data.have_renditions()
            and not all(_path("img_%s.labels.npy" % p).exists() for p in REND_SPLITS)):
        raise SystemExit("TRAIN_ON_RENDITIONS is set but imr.parquet / sketch.parquet are not "
                         "fetched -- run data.fetch_renditions() or set it to False")
    if (C.TRAIN_ON_MORE_SOUNDS and C.VGGSOUND_NAMES and not data.have_vggsound()
            and not _path("aud_vgg_train.labels.npy").exists()):
        raise SystemExit("VGGSOUND_NAMES is set but VGGSound is not fetched -- run "
                         "data.fetch_vggsound() or set it to {}")
    free = shutil.disk_usage(C.CACHE_DIR).free / 1e9
    if free < C.DISK_HEADROOM_GB:
        raise SystemExit("only %.1f GB free on the cache drive; need %.1f GB of headroom"
                         % (free, C.DISK_HEADROOM_GB))


def _complete() -> bool:
    need = [("img", p) for p in IMAGE_SPLITS] + [("aud", p) for p in SPLITS]
    if data.have_tiny():
        need += [("img", p) for p in TINY_SPLITS]
    if data.have_inet():
        need += [("img", p) for p in INET_SPLITS]
        #  Encoded from other photos or rules: not complete (encode() re-does them).
        listed = (json.loads(_path("sources.json").read_text())
                  if _path("sources.json").exists() else {})
        if listed.get("inet_stamp") != data.inet_stamp():
            return False
    if C.TRAIN_ON_MORE_SOUNDS and data.have_more_sounds():
        need += [("aud", p) for p in SOUND_SPLITS]
        listed = (json.loads(_path("sources.json").read_text())
                  if _path("sources.json").exists() else {})
        if listed.get("snd_stamp") != data.more_sounds_stamp():
            return False
    listed = (json.loads(_path("sources.json").read_text())
              if _path("sources.json").exists() else {})
    if C.INET_MORE_PER_CLASS and data.have_inet_more():
        need.append(("img", "inet_more"))
        if listed.get("inet_more_stamp") != data.inet_more_stamp():
            return False
    if C.AUDIOSET_EXTRA_FILES and data.have_extra_sounds():
        need.append(("aud", "rare_train"))
        if listed.get("rare_stamp") != data.extra_sounds_stamp():
            return False
    if C.INET_MORE2_PER_CLASS and data.have_inet_more2():
        need.append(("img", "inet_more2"))
        if listed.get("inet_more2_stamp") != data.inet_more2_stamp():
            return False
    if C.VGGSOUND_NAMES and data.have_vggsound():
        need.append(("aud", "vgg_train"))
        if listed.get("vgg_stamp") != data.vggsound_stamp(_audioset_videos(listed)):
            return False
    if C.TRAIN_ON_RENDITIONS and data.have_renditions():
        stamp = data.renditions_stamp()
        for part in REND_SPLITS:
            need.append(("img", part))
            if listed.get("stamp_img_%s" % part) != stamp:
                return False
        for part in ("imr_test", "sketch_test"):
            if (listed.get("flip_img_%s" % part) != stamp
                    or not _path("img_%s.identity_flip.npy" % part).exists()):
                return False
    if data.have_imagenette():
        need.append(("img", "open"))
    return all(_path("%s_%s.labels.npy" % (k, p)).exists() for k, p in need)


def encode(fresh: bool = False) -> dict:
    """Fill the cache if it is missing or was built for different networks."""
    C.CACHE_DIR.mkdir(parents=True, exist_ok=True)
    manifest_path = _path("manifest.json")
    wanted = _manifest_wanted()
    keep = False
    if not fresh and manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if all(manifest.get(k) == v for k, v in wanted.items()):
            if _complete():
                print("cache matches the configured networks (%s)" % C.CACHE_DIR)
                compress_cache()        # a float16 grid an interrupted run left
                return manifest
            keep = True                 # same networks: only encode what is missing
            print("cache matches; encoding the splits it is missing")
        else:
            print("cache was built for other networks -- re-encoding")
    if not keep:
        #  The raw AudioSet / FSD50K / unbalanced-AudioSet files were deleted
        #  once encoded (C: had no room); wiping would lose ~57,000 encoded
        #  clips for good.
        gone = [name for name, have in (("aud_snd_train", data.have_more_sounds()),
                                        ("aud_rare_train", data.have_extra_sounds()),
                                        ("aud_vgg_train", data.have_vggsound()))
                if _path("%s.labels.npy" % name).exists() and not have]
        if gone:
            raise SystemExit("re-encoding everything would delete %s, whose source audio is no "
                             "longer on disk -- fetch it again first" % ", ".join(gone))
        for old in C.CACHE_DIR.glob("*.npy"):
            old.unlink()
    sources = (json.loads(_path("sources.json").read_text())
               if keep and _path("sources.json").exists() else {})

    from v2b.brain import BrainV2B
    brain = BrainV2B(checkpoint=None, memory_dir=C.CACHE_DIR / "no_memory")
    dims = brain.dims
    image_fields, audio_fields = _fields(dims)
    started = time.time()
    for part in IMAGE_SPLITS:
        if keep and _path("img_%s.labels.npy" % part).exists():
            continue
        images, labels, src = data.cifar(part)
        n = len(labels)
        out = _image_out(part, n, image_fields)
        for start in range(0, n, C.ENCODE_BATCH):
            chunk = torch.from_numpy(images[start:start + C.ENCODE_BATCH]).to(brain.device)
            seen = brain.see(senses.retina(chunk))
            for key in image_fields:
                out[key][start:start + len(chunk)] = seen[key].cpu().numpy()
            print("\r  images %-5s %d/%d  %.0fs" % (part, start + len(chunk), n,
                                                   time.time() - started), end="", flush=True)
            emit({"stage": "encode", "split": "CIFAR " + part, "done": start + len(chunk),
                  "total": n}, every=1.0)
        print()
        _settle(out)
        _memmap("img_%s.labels" % part, (), n, "w+")[:] = labels
        sources["img_%s" % part] = src

    #  TINY-IMAGENET: 200 ImageNet classes at 64x64 -- more, and sharper,
    #  pictures to learn from than CIFAR alone.
    if data.have_tiny():
        for part in TINY_SPLITS:
            if keep and _path("img_%s.labels.npy" % part).exists():
                continue
            _, labels, src = data.tiny_imagenet(part.split("_")[1], decode=False)
            n = len(labels)
            out = _image_out(part, n, image_fields)
            for start in range(0, n, C.ENCODE_BATCH):
                pixels = data.tiny_pixels(src[start:start + C.ENCODE_BATCH])
                chunk = torch.from_numpy(pixels).to(brain.device)
                seen = brain.see(senses.retina(chunk))
                for key in image_fields:
                    out[key][start:start + len(chunk)] = seen[key].cpu().numpy()
                print("\r  images %-10s %d/%d  %.0fs" % (part, start + len(chunk), n,
                                                        time.time() - started), end="", flush=True)
                emit({"stage": "encode", "split": part.replace("_", " "),
                      "done": start + len(chunk), "total": n}, every=1.0)
            print()
            _settle(out)
            _memmap("img_%s.labels" % part, (), n, "w+")[:] = labels
            sources["img_%s" % part] = src
            _save_sources(sources)

    #  IMAGENET PHOTOS: real 256 px photos of the ~790 ImageNet classes that
    #  neither CIFAR nor Tiny covers.
    if data.have_inet():
        stamp = data.inet_stamp()
        if sources.get("inet_stamp") not in (None, stamp):
            for old in C.CACHE_DIR.glob("img_inet_*.npy"):
                old.unlink()                     # made from other photos or rules
        sources["inet_stamp"] = stamp
        for part in INET_SPLITS:
            if keep and _path("img_%s.labels.npy" % part).exists():
                continue
            labels, src = data.imagenet_photos(part.split("_")[1])
            n = len(labels)
            out = _image_out(part, n, image_fields)
            for start in range(0, n, C.ENCODE_BATCH):
                pixels = data.inet_pixels(src[start:start + C.ENCODE_BATCH])
                chunk = torch.from_numpy(pixels).to(brain.device)
                seen = brain.see(senses.retina(chunk))
                for key in image_fields:
                    out[key][start:start + len(chunk)] = seen[key].cpu().numpy()
                print("\r  images %-10s %d/%d  %.0fs" % (part, start + len(chunk), n,
                                                        time.time() - started), end="", flush=True)
                emit({"stage": "encode", "split": part.replace("_", " "),
                      "done": start + len(chunk), "total": n}, every=1.0)
            print()
            _settle(out)
            _memmap("img_%s.labels" % part, (), n, "w+")[:] = labels
            sources["img_%s" % part] = src
            _save_sources(sources)

    #  THE OPEN-VOCABULARY TEST: pictures of ten things v2.B never trained on,
    #  labelled with their ImageNet names. Encoded, never trained on, never
    #  stored in memory -- only measured.
    if data.have_imagenette() and not (keep and _path("img_open.labels.npy").exists()):
        pictures, labels, src = data.imagenette()
        n = len(labels)
        out = _image_out("open", n, image_fields)
        for start in range(0, n, C.ENCODE_BATCH):
            seen = brain.see(senses.retina(pictures[start:start + C.ENCODE_BATCH]))
            for key in image_fields:
                out[key][start:start + C.ENCODE_BATCH] = seen[key].cpu().numpy()
        _settle(out)
        _memmap("img_open.labels", (), n, "w+")[:] = labels
        sources["img_open"] = src
        print("  open-vocabulary pictures: %d" % n)

    offset = data.sound_offset()                        # sounds follow the image names
    for part in SPLITS:
        if keep and _path("aud_%s.labels.npy" % part).exists():
            continue
        waves, labels, src = data.audio(part)
        n = len(labels)
        out = {k: _memmap("aud_%s.%s" % (part, k), s, n, "w+") for k, s in audio_fields.items()}
        for start in range(0, n, 16):
            heard = brain.hear(waves[start:start + 16])
            for key in audio_fields:
                out[key][start:start + 16] = heard[key].cpu().numpy()
            print("\r  sounds %-5s %d/%d  %.0fs" % (part, min(start + 16, n), n,
                                                   time.time() - started), end="", flush=True)
        print()
        for array in out.values():
            array.flush()
        _memmap("aud_%s.labels" % part, (), n, "w+")[:] = labels + offset
        sources["aud_%s" % part] = src

    #  MORE SOUNDS: AudioSet's balanced set + FSD50K, decoded batch by batch
    #  (45,000 ten-second clips at 48 kHz would be ~90 GB as one list).
    if C.TRAIN_ON_MORE_SOUNDS and data.have_more_sounds():
        stamp = data.more_sounds_stamp()
        if sources.get("snd_stamp") not in (None, stamp):
            for old in C.CACHE_DIR.glob("aud_snd_*.npy"):
                old.unlink()                     # made from other clips or rules
        sources["snd_stamp"] = stamp
        for part in SOUND_SPLITS:
            if keep and sources.get("snd_stamp") == stamp and _path("aud_%s.labels.npy" % part).exists():
                continue
            labels, src = data.more_sounds(part.split("_")[1])
            n = len(labels)
            out = {k: _memmap("aud_%s.%s" % (part, k), s_, n, "w+") for k, s_ in audio_fields.items()}
            for start in range(0, n, 16):
                heard = brain.hear(data.more_waves(src[start:start + 16]))
                for key in audio_fields:
                    out[key][start:start + 16] = heard[key].cpu().numpy()
                print("\r  sounds %-10s %d/%d  %.0fs" % (part, min(start + 16, n), n,
                                                        time.time() - started), end="", flush=True)
                emit({"stage": "encode", "split": part.replace("_", " "),
                      "done": min(start + 16, n), "total": n}, every=1.0)
            print()
            for array in out.values():
                array.flush()
            _memmap("aud_%s.labels" % part, (), n, "w+")[:] = labels      # already vocabulary indices
            sources["aud_%s" % part] = src
            _save_sources(sources)

    #  ROUND 5: the next photos of each ImageNet class, and rare sounds from
    #  AudioSet's unbalanced set -- each its own split, so nothing encoded
    #  before is redone.
    if C.INET_MORE_PER_CLASS and data.have_inet_more():
        stamp = data.inet_more_stamp()
        if sources.get("inet_more_stamp") != stamp or not _path("img_inet_more.labels.npy").exists():
            labels, src = data.imagenet_photos("more")
            n = len(labels)
            out = _image_out("inet_more", n, image_fields)
            for start in range(0, n, C.ENCODE_BATCH):
                pixels = data.inet_pixels(src[start:start + C.ENCODE_BATCH])
                seen = brain.see(senses.retina(torch.from_numpy(pixels).to(brain.device)))
                for key in image_fields:
                    out[key][start:start + len(pixels)] = seen[key].cpu().numpy()
                print("\r  images inet_more   %d/%d  %.0fs" % (start + len(pixels), n,
                                                            time.time() - started), end="", flush=True)
                emit({"stage": "encode", "split": "inet more", "done": start + len(pixels),
                      "total": n}, every=1.0)
            print()
            _settle(out)
            _memmap("img_inet_more.labels", (), n, "w+")[:] = labels
            sources["img_inet_more"], sources["inet_more_stamp"] = src, stamp
            _save_sources(sources)
    if C.AUDIOSET_EXTRA_FILES and data.have_extra_sounds():
        stamp = data.extra_sounds_stamp()
        if sources.get("rare_stamp") != stamp or not _path("aud_rare_train.labels.npy").exists():
            labels, src = data.extra_sounds()
            n = len(labels)
            out = {k: _memmap("aud_rare_train.%s" % k, s_, n, "w+") for k, s_ in audio_fields.items()}
            for start in range(0, n, 16):
                heard = brain.hear(data.more_waves(src[start:start + 16]))
                for key in audio_fields:
                    out[key][start:start + 16] = heard[key].cpu().numpy()
                print("\r  sounds rare_train  %d/%d  %.0fs" % (min(start + 16, n), n,
                                                           time.time() - started), end="", flush=True)
                emit({"stage": "encode", "split": "rare sounds", "done": min(start + 16, n),
                      "total": n}, every=1.0)
            print()
            for array in out.values():
                array.flush()
            _memmap("aud_rare_train.labels", (), n, "w+")[:] = labels
            sources["aud_rare_train"], sources["rare_stamp"] = src, stamp
            _save_sources(sources)

    #  ROUND 7: the next 88 photos of every ImageNet class, written straight to
    #  int8, and VGGSound's clips for names v2.B already has.
    if C.INET_MORE2_PER_CLASS and data.have_inet_more2():
        stamp = data.inet_more2_stamp()
        if sources.get("inet_more2_stamp") != stamp or not _path("img_inet_more2.labels.npy").exists():
            labels, src = data.imagenet_photos("more2")
            n = len(labels)
            out = _image_out("inet_more2", n, image_fields)
            for start in range(0, n, C.ENCODE_BATCH):
                pixels = data.inet_pixels(src[start:start + C.ENCODE_BATCH])
                seen = brain.see(senses.retina(torch.from_numpy(pixels).to(brain.device)))
                for key in image_fields:
                    out[key][start:start + len(pixels)] = seen[key].cpu().numpy()
                print("\r  images inet_more2  %d/%d  %.0fs" % (start + len(pixels), n,
                                                            time.time() - started), end="", flush=True)
                emit({"stage": "encode", "split": "inet more2", "done": start + len(pixels),
                      "total": n}, every=1.0)
            print()
            _settle(out)
            _memmap("img_inet_more2.labels", (), n, "w+")[:] = labels
            sources["img_inet_more2"], sources["inet_more2_stamp"] = src, stamp
            _save_sources(sources)
            data._INET.clear()          # ~1.8 GB of photos
    if C.VGGSOUND_NAMES and data.have_vggsound():
        videos = _audioset_videos(sources)
        stamp = data.vggsound_stamp(videos)
        if sources.get("vgg_stamp") != stamp or not _path("aud_vgg_train.labels.npy").exists():
            labels, src = data.vggsound(videos)
            n = len(labels)
            out = {k: _memmap("aud_vgg_train.%s" % k, s_, n, "w+") for k, s_ in audio_fields.items()}
            for start in range(0, n, 16):
                heard = brain.hear(data.more_waves(src[start:start + 16]))
                for key in audio_fields:
                    out[key][start:start + 16] = heard[key].cpu().numpy()
                print("\r  sounds vgg_train   %d/%d  %.0fs" % (min(start + 16, n), n,
                                                            time.time() - started), end="", flush=True)
                emit({"stage": "encode", "split": "vggsound", "done": min(start + 16, n),
                      "total": n}, every=1.0)
            print()
            for array in out.values():
                array.flush()
            _memmap("aud_vgg_train.labels", (), n, "w+")[:] = labels
            sources["aud_vgg_train"], sources["vgg_stamp"] = src, stamp
            _save_sources(sources)

    #  RENDITIONS: art, toys, origami, sketches... of ImageNet's classes. One
    #  stamp per split, so an interrupted run resumes where it stopped.
    if C.TRAIN_ON_RENDITIONS and data.have_renditions():
        stamp = data.renditions_stamp()
        for part in REND_SPLITS:
            if (sources.get("stamp_img_%s" % part) == stamp
                    and _path("img_%s.labels.npy" % part).exists()):
                continue
            labels, src = data.renditions(*part.split("_"))
            n = len(labels)
            out = _image_out(part, n, image_fields)
            for start in range(0, n, C.ENCODE_BATCH):
                pixels = data.inet_pixels(src[start:start + C.ENCODE_BATCH])
                seen = brain.see(senses.retina(torch.from_numpy(pixels).to(brain.device)))
                for key in image_fields:
                    out[key][start:start + len(pixels)] = seen[key].cpu().numpy()
                print("\r  images %-12s %d/%d  %.0fs" % (part, start + len(pixels), n,
                                                          time.time() - started), end="", flush=True)
                emit({"stage": "encode", "split": part.replace("_", " "),
                      "done": start + len(pixels), "total": n}, every=1.0)
            print()
            _settle(out)
            _memmap("img_%s.labels" % part, (), n, "w+")[:] = labels
            sources["img_%s" % part], sources["stamp_img_%s" % part] = src, stamp
            _save_sources(sources)
        data._INET.clear()

    #  MIRRORED COPIES: ImageNet-R and ImageNet-Sketch hold left-right flipped
    #  copies of one picture (a broom, a liner, one frog filed as "bullfrog" in
    #  one copy and "tailed frog" in the other), at identity cosine 0.965-0.970
    #  -- under IMAGE_NEAR_COPY, so they passed as different pictures. Each
    #  rendition TEST picture is also seen mirrored (identity only), and every
    #  copy check in train() compares against both.
    if C.TRAIN_ON_RENDITIONS and data.have_renditions():
        for part in ("imr_test", "sketch_test"):
            stamp = sources.get("stamp_img_%s" % part)
            if (sources.get("flip_img_%s" % part) == stamp
                    and _path("img_%s.identity_flip.npy" % part).exists()):
                continue
            src = sources["img_%s" % part]
            n = len(src)
            flip = _memmap("img_%s.identity_flip" % part, image_fields["identity"], n, "w+")
            for start in range(0, n, C.ENCODE_BATCH):
                pixels = np.ascontiguousarray(data.inet_pixels(src[start:start + C.ENCODE_BATCH])[:, :, ::-1])
                seen = brain.see(senses.retina(torch.from_numpy(pixels).to(brain.device)))
                flip[start:start + len(pixels)] = seen["identity"].cpu().numpy()
                print("\r  mirrored %-12s %d/%d" % (part, start + len(pixels), n), end="", flush=True)
            print()
            flip.flush()
            del flip
            sources["flip_img_%s" % part] = stamp
            _save_sources(sources)
        data._INET.clear()

    names, kinds = data.vocabulary()
    np.save(_path("words.npy"), brain.frontal.words.cpu().numpy())
    np.save(_path("lexicon.npy"), brain.cortex.lexicon.cpu().numpy())
    manifest = dict(wanted, dims=asdict(dims), scale=brain.frontal.scale,
                    sound_scale=brain.temporal.scale,
                    names=names, kinds=kinds, minutes=round((time.time() - started) / 60, 2))
    _save_sources(sources)
    manifest_path.write_text(json.dumps(manifest, indent=1))
    compress_cache()                    # anything a crash left at float16
    del brain
    data._INET.clear()                  # ~0.6 GB of photos not needed while training
    data._TINY.clear()
    torch.cuda.empty_cache()
    return manifest


class _Q8:
    """An int8 grid with one scale per (item, place), read back as float32, with
    the split's few outlier channels kept at 16 bits beside it. Indexing returns
    floats, so every reader of Cache.arrays works unchanged."""

    def __init__(self, q, scale, keep, kept):
        self.q, self.scale, self.keep, self.kept = q, scale, np.asarray(keep), kept
        self.shape, self.dtype = q.shape, np.dtype(np.float32)

    def __len__(self) -> int:
        return len(self.q)

    def __getitem__(self, index):
        x = (np.asarray(self.q[index], np.float32)
             * (np.asarray(self.scale[index], np.float32)[..., None] / 127.0))
        x[..., self.keep] = np.asarray(self.kept[index], np.float32)
        return x


def _quantize(x: np.ndarray, keep: np.ndarray) -> tuple:
    """float (n, places, width) -> int8 values and a float16 scale per place, with
    the `keep` channels taken out first and returned as float16."""
    kept = x[..., keep].astype(np.float16)
    x = x.copy()
    x[..., keep] = 0.0
    scale = np.abs(x).max(-1)
    q = np.rint(x / np.maximum(scale, 1e-8)[..., None] * 127.0).clip(-127, 127).astype(np.int8)
    return q, scale.astype(np.float16), kept


def compress_split(name: str, log=print) -> bool:
    """img_<part>.<grid>.npy (float16) -> .q8 / .scale / .keep / .kept .npy, then
    the float16 file is deleted -- only after a sample reads back at cosine
    >= C.QUANT_MIN_COSINE. Returns False (everything left as it was) if the disk
    is too full."""
    import os
    import shutil
    src_path = _path(name + ".npy")
    if not src_path.exists():
        return True
    src = np.load(src_path, mmap_mode="r")
    if src.dtype != np.float16:
        return True
    n, places, width = src.shape
    need = src.size + n * places * (2 + 2 * C.QUANT_KEEP) + 3e8
    if shutil.disk_usage(C.CACHE_DIR).free < need:
        log("  not enough disk yet to compress %s (%.1f GB needed)" % (name, need / 1e9))
        return False
    rng = np.random.default_rng(0)
    probe = np.sort(rng.choice(n, min(512, n), replace=False))
    peak = np.abs(np.asarray(src[probe], np.float32)).reshape(-1, width).max(0)
    keep = np.sort(np.argsort(peak)[::-1][:C.QUANT_KEEP])
    files = {part: _path("%s.%s.tmp.npy" % (name, part)) for part in ("q8", "scale", "kept")}
    q = np.lib.format.open_memmap(files["q8"], mode="w+", dtype=np.int8, shape=src.shape)
    s = np.lib.format.open_memmap(files["scale"], mode="w+", dtype=np.float16, shape=(n, places))
    k = np.lib.format.open_memmap(files["kept"], mode="w+", dtype=np.float16,
                                  shape=(n, places, len(keep)))
    for start in range(0, n, 1024):
        q[start:start + 1024], s[start:start + 1024], k[start:start + 1024] = _quantize(
            np.asarray(src[start:start + 1024], np.float32), keep)
    q.flush()
    s.flush()
    k.flush()
    rows = np.sort(rng.choice(n, min(256, n), replace=False))
    a = np.asarray(src[rows], np.float32).reshape(len(rows), -1)
    b = _Q8(q, s, keep, k)[rows].reshape(len(rows), -1)
    cos = (a * b).sum(-1) / np.maximum(np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1), 1e-12)
    del q, s, k, src
    if cos.min() < C.QUANT_MIN_COSINE:
        for f in files.values():
            f.unlink()
        raise SystemExit("compressing %s read back at cosine %.5f -- left as float16"
                         % (name, cos.min()))
    np.save(_path(name + ".keep.npy"), keep)
    for part in ("scale", "kept", "q8"):    # q8 last: Cache reads int8 once it exists
        os.replace(files[part], _path("%s.%s.npy" % (name, part)))
    src_path.unlink()
    log("  compressed %s (min cosine %.6f)" % (name, cos.min()))
    return True


def compress_cache(log=print) -> None:
    """Every float16 picture grid, smallest first: each one frees room for the next."""
    if not C.CACHE_INT8:
        return
    todo = sorted((p for k in C.QUANTIZED for p in C.CACHE_DIR.glob("img_*.%s.npy" % k)),
                  key=lambda p: p.stat().st_size)
    for _ in range(2):                       # a second sweep picks up what lacked room
        for p in todo:
            compress_split(p.name[:-len(".npy")], log)


def _audioset_videos(sources: dict) -> set:
    """The YouTube ids behind every AudioSet clip in the cache (each id ends
    in one). VGGSound is YouTube too: a clip from one of these videos could be
    the very sound an AudioSet test clip was cut from."""
    return {x.rsplit(":", 1)[1] for key in ("aud_snd_train", "aud_snd_val", "aud_snd_atest",
                                             "aud_snd_ftest", "aud_rare_train")
            for x in sources.get(key, ()) if x.startswith(("aset:", "asetu:"))}


def _shared_keep(field: str):
    """The QUANT_KEEP outlier channels most compressed splits agree on, or None.
    They are SigLIP's own (514, 875 and 1009 lead in every split; 14 of 16
    occ_grid channels are shared by all 12), so a NEW split written straight to
    int8 can use them before it has a sample of its own."""
    import collections
    votes = collections.Counter()
    for path in C.CACHE_DIR.glob("img_*.%s.keep.npy" % field):
        votes.update(np.load(path).tolist())
    if not votes:
        return None
    return np.sort(np.asarray(sorted(votes, key=lambda c: (-votes[c], c))[:C.QUANT_KEEP]))


class _Q8Writer:
    """A new split's grid written straight to int8 (+ scale + kept channels),
    every batch checked against QUANT_MIN_COSINE. At float16 first, round 7's
    69k photos would need 15.6 GB before compress_split could halve them --
    more than D: had once they and VGGSound were fetched."""

    def __init__(self, name: str, shape: tuple, keep: np.ndarray):
        self.name, self.keep = name, keep
        self.files = {part: _path("%s.%s.tmp.npy" % (name, part)) for part in ("q8", "scale", "kept")}
        self.q = np.lib.format.open_memmap(self.files["q8"], mode="w+", dtype=np.int8, shape=shape)
        self.s = np.lib.format.open_memmap(self.files["scale"], mode="w+", dtype=np.float16,
                                           shape=shape[:2])
        self.k = np.lib.format.open_memmap(self.files["kept"], mode="w+", dtype=np.float16,
                                           shape=shape[:2] + (len(keep),))

    def __setitem__(self, index, x) -> None:
        x = np.asarray(x, np.float32)
        q, scale, kept = _quantize(x, self.keep)
        back = q.astype(np.float32) * (scale.astype(np.float32)[..., None] / 127.0)
        back[..., self.keep] = kept.astype(np.float32)
        a, b = x.reshape(len(x), -1), back.reshape(len(x), -1)
        cos = (a * b).sum(-1) / np.maximum(np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1),
                                           1e-12)
        if cos.min() < C.QUANT_MIN_COSINE:
            raise SystemExit("%s: a batch quantised at cosine %.5f" % (self.name, cos.min()))
        self.q[index], self.s[index], self.k[index] = q, scale, kept

    def flush(self) -> None:
        """Close and rename into place, q8 last: Cache reads int8 once it exists."""
        import os
        self.q.flush()
        self.s.flush()
        self.k.flush()
        del self.q, self.s, self.k           # Windows will not rename a mapped file
        np.save(_path(self.name + ".keep.npy"), self.keep)
        for part in ("scale", "kept", "q8"):
            os.replace(self.files[part], _path("%s.%s.npy" % (self.name, part)))


def _image_out(part: str, n: int, image_fields: dict) -> dict:
    """Output arrays for a NEW picture split: its grids straight to int8 when
    the cache is int8, everything else as _memmap writes it."""
    out = {}
    for k, shape in image_fields.items():
        keep = _shared_keep(k) if C.CACHE_INT8 and k in C.QUANTIZED else None
        stem = "img_%s.%s" % (part, k)
        if keep is not None:
            if _path(stem + ".npy").exists():
                _path(stem + ".npy").unlink()       # compress_cache would put it back over ours
            out[k] = _Q8Writer(stem, (n,) + tuple(shape), keep)
            continue
        for old in ("q8", "scale", "keep", "kept"):      # Cache would prefer a stale int8
            if _path("%s.%s.npy" % (stem, old)).exists():
                _path("%s.%s.npy" % (stem, old)).unlink()
        out[k] = _memmap(stem, shape, n, "w+")
    return out


def _identities(cache) -> list:
    """A split's SigLIP identities, plus their mirrored twins where encoded."""
    found = [cache.arrays["identity"]]
    flip = _path("%s.identity_flip.npy" % getattr(cache, "stem", ""))
    if flip.exists():
        found.append(np.load(flip, mmap_mode="r"))
    return found


def _round6_pictures(dims) -> list:
    """The picture splits round 6 trained on -- and so stored in its memory."""
    parts = ["train"] + (["extra"] if C.TRAIN_ON_EXTRA else []) + ["tiny_train", "inet_train",
                                                                    "inet_more"]
    return [Cache("img", p, dims) for p in parts if _path("img_%s.labels.npy" % p).exists()]


def rendition_tests(dims, device, rend: dict | None = None) -> tuple[dict, dict]:
    """imr_test / sketch_test minus every picture whose SigLIP identity (or its
    mirror's) is >= REND_NEAR_COPY like a picture round 6 trained on: ImageNet scraped
    drawings, toys and paintings too, and either round's memory would recall
    such a copy. The round-6 baseline uses this same function, so both models
    are scored on identical rows."""
    rend = rend or {p: Cache("img", p, dims) for p in ("imr_test", "sketch_test")}
    stored = [c.arrays["identity"] for c in _round6_pictures(dims)]
    out, dropped = {}, {}
    with torch.no_grad():
        for key in ("imr", "sketch"):
            test = rend["%s_test" % key]
            best = torch.full((len(test),), -1.0, device=device)
            for seen in _identities(test):                     # the picture, and mirrored
                q = F.normalize(torch.from_numpy(np.asarray(seen, np.float32)).to(device), dim=-1)
                for ident in stored:
                    for start in range(0, len(ident), 32768):
                        k = F.normalize(torch.from_numpy(np.asarray(ident[start:start + 32768],
                                                                    np.float32)).to(device), dim=-1)
                        for s in range(0, len(q), 2048):       # 2,048 x 32,768: 0.27 GB
                            best[s:s + 2048] = torch.maximum(best[s:s + 2048],
                                                             (q[s:s + 2048] @ k.T).max(-1).values)
            rows = np.flatnonzero(best.cpu().numpy() < C.REND_NEAR_COPY)
            out[key], dropped[key] = Subset(test, rows), len(test) - len(rows)
    torch.cuda.empty_cache()
    return out, dropped


def _settle(out: dict) -> None:
    """A finished picture split: flush, let go of every mapping (Windows will
    not rename or delete a mapped file -- WinError 32), then compress whatever
    of it is still float16. Per split, so --fresh never holds every grid at
    float16 at once: ~63 GB for round 6 alone, more than D: has."""
    for key in list(out):
        out.pop(key).flush()
    compress_cache()


class Cache:
    """One split of cached lobe output, read in batches."""

    def __init__(self, kind: str, part: str, dims: Dims):
        self.image = kind == "img"
        self.stem = "%s_%s" % (kind, part)
        self.dims = dims
        image_fields, audio_fields = _fields(dims)
        self.fields = image_fields if self.image else audio_fields
        self.labels = np.load(_path("%s_%s.labels.npy" % (kind, part)))
        self.arrays = {}
        for k in self.fields:
            stem = "%s_%s.%s" % (kind, part, k)
            if _path(stem + ".q8.npy").exists():
                self.arrays[k] = _Q8(*(np.load(_path("%s.%s.npy" % (stem, part)), mmap_mode="r")
                                       for part in ("q8", "scale", "keep", "kept")))
            else:
                self.arrays[k] = np.load(_path(stem + ".npy"), mmap_mode="r")
        self.sources = json.loads(_path("sources.json").read_text())["%s_%s" % (kind, part)]

    def __len__(self) -> int:
        return len(self.labels)

    def batch(self, index, device) -> tuple[Features, torch.Tensor]:
        index = np.sort(np.asarray(index))
        n = len(index)
        image_fields, audio_fields = _fields(self.dims)
        shapes = dict(image_fields, **audio_fields)

        def take(key):
            if key in self.arrays:
                return torch.from_numpy(np.asarray(self.arrays[key][index], np.float32)).to(device)
            return torch.zeros((n,) + shapes[key], device=device)

        f = Features(take("occ_grid"), take("it_grid"), take("identity"), take("par_cells"),
                     take("aud_bins"), take("aud_identity"), take("aud_joint"), take("image_stats"),
                     take("audio_stats"), torch.full((n,), self.image, device=device),
                     torch.full((n,), not self.image, device=device))
        return f, torch.from_numpy(self.labels[index]).to(device)


class Subset:
    """Some rows of a cache -- e.g. new sounds minus near-copies of a test clip."""

    def __init__(self, cache, rows):
        self.cache, self.rows = cache, np.asarray(rows, np.int64)
        self.image = cache.image
        self.labels = cache.labels[self.rows]
        self.sources = [cache.sources[i] for i in self.rows]
        self.arrays = cache.arrays

    def __len__(self) -> int:
        return len(self.rows)

    def batch(self, index, device):
        return self.cache.batch(self.rows[np.asarray(index)], device)


class Joined:
    """Several cached splits read as one (e.g. CIFAR + Tiny-ImageNet)."""

    def __init__(self, parts: list):
        self.parts = [q for q in parts if q is not None]
        self.image = self.parts[0].image
        self.labels = np.concatenate([q.labels for q in self.parts])
        self.sources = [x for q in self.parts for x in q.sources]
        self.offsets = np.cumsum([0] + [len(q) for q in self.parts])

    def __len__(self) -> int:
        return int(self.offsets[-1])

    def batch(self, index, device):
        index = np.sort(np.asarray(index))
        feats, labels = [], []
        for q, lo, hi in zip(self.parts, self.offsets[:-1], self.offsets[1:]):
            local = index[(index >= lo) & (index < hi)] - lo
            if len(local):
                f, y = q.batch(local, device)
                feats.append(f)
                labels.append(y)
        f = feats[0]
        for more in feats[1:]:
            f = Features.join(f, more)
        return f, torch.cat(labels)


# ---------------------------------------------------------------------------
# Running the cortex over a split
# ---------------------------------------------------------------------------


@torch.no_grad()
def run(cortex: Cortex, cache: Cache, words, device, keep_logits: bool = True):
    """-> final logits (N,V), POT vectors (N,E), per-pass accuracy, labels.
    keep_logits=False drops the (N,V) scores: for the 136,000 training
    examples they are ~0.9 GB of RAM and only the vectors are needed."""
    cortex.eval()
    logits, vectors, right = [], [], np.zeros(C.FRONTAL_PASSES)
    for start in range(0, len(cache), 512):
        f, y = cache.batch(np.arange(start, min(start + 512, len(cache))), device)
        out = cortex(f, words)
        for k, p in enumerate(out.passes):
            right[k] += float((p.logits.argmax(-1) == y).sum())
        if keep_logits:
            logits.append(out.passes[-1].logits.cpu())
        vectors.append(out.passes[-1].embedding.cpu())
    return (torch.cat(logits) if keep_logits else None, torch.cat(vectors),
            (right / len(cache) * 100).tolist(),
            torch.from_numpy(cache.labels))


def episode_similarity(vectors: torch.Tensor, labels: torch.Tensor, pairs: int = 40000,
                       seed: int = C.SEED) -> float:
    """The taught-episode bar: EPISODE_QUANTILE of the cosine between two
    DIFFERENT validation items of the same class (see EPISODE_SIMILARITY)."""
    v = F.normalize(vectors.float(), dim=-1)
    y = labels.numpy() if torch.is_tensor(labels) else np.asarray(labels)
    rng = np.random.default_rng(seed)
    by = {}
    for i, c in enumerate(y):
        by.setdefault(int(c), []).append(i)
    groups = [np.asarray(g) for g in by.values() if len(g) > 1]
    sims = []
    for _ in range(pairs):
        g = groups[rng.integers(len(groups))]
        i, j = rng.choice(g, 2, replace=False)
        sims.append(float(v[i] @ v[j]))
    return float(min(max(np.quantile(sims, C.EPISODE_QUANTILE), C.EPISODE_SIMILARITY),
                     C.EPISODE_MAX))


def _drop_features(f: Features, p: float) -> Features:
    """Training only: dropout on the input grids (see FEATURE_DROPOUT)."""
    grids = ("occ_grid", "it_grid", "par_cells", "aud_bins")
    return Features(*(F.dropout(getattr(f, k), p, training=True) if k in grids
                      else getattr(f, k) for k in f.__dataclass_fields__))


def train_cortex(cortex, words, caches, device, epochs, names=None, watch=None) -> int:
    """`watch(epoch)` is called after every epoch, for the live monitor."""
    rng = np.random.default_rng(C.SEED)
    optimiser = torch.optim.AdamW(cortex.parameters(), lr=C.LR, weight_decay=C.WEIGHT_DECAY)
    more = caches.get("aud_more")
    #  ESC-50 keeps exactly its old share of every batch; the new sounds come
    #  on top of it, out of the pictures' share.
    image_share = C.BATCH - C.AUDIO_SHARE - (C.MORE_AUDIO_SHARE if more is not None else 0)
    steps = math.ceil(len(caches["img"]) / image_share)
    schedule = torch.optim.lr_scheduler.OneCycleLR(optimiser, max_lr=C.LR,
                                                   total_steps=epochs * steps, pct_start=0.1)
    weights = torch.tensor(C.PASS_WEIGHTS, device=device)
    seen = 0
    started = time.time()
    for epoch in range(epochs):
        cortex.train()
        order = rng.permutation(len(caches["img"]))
        running = 0.0
        for index in range(steps):
            chosen = order[index * image_share:(index + 1) * image_share]
            f1, y1 = caches["img"].batch(chosen, device)
            f2, y2 = caches["aud"].batch(rng.choice(len(caches["aud"]), C.AUDIO_SHARE,
                                                    replace=False), device)
            if more is not None:
                f3, y3 = more.batch(rng.choice(len(more), C.MORE_AUDIO_SHARE, replace=False),
                                    device)
                f2, y2 = Features.join(f2, f3), torch.cat([y2, y3])
            f, y = Features.join(f1, f2), torch.cat([y1, y2])
            if C.FEATURE_DROPOUT:
                f = _drop_features(f, C.FEATURE_DROPOUT)
            out = cortex(f, words)
            loss = sum(wt * F.cross_entropy(p.logits, y, label_smoothing=C.LABEL_SMOOTHING)
                       for wt, p in zip(weights, out.passes)) / weights.sum()
            optimiser.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(cortex.parameters(), 1.0)
            optimiser.step()
            schedule.step()
            running += float(loss.detach())
            seen += len(y)
            if index % 20 == 0:
                print("\r  cortex epoch %d/%d  step %d/%d  loss %.3f"
                      % (epoch + 1, epochs, index, steps, running / (index + 1)),
                      end="", flush=True)
            if index % 5 == 0:
                done = epoch * steps + index + 1
                emit({"stage": "cortex", "epoch": epoch + 1, "epochs": epochs, "step": index + 1,
                      "steps": steps, "loss": float(loss.detach()),
                      "lr": schedule.get_last_lr()[0], "seen": seen,
                      "rate": seen / max(time.time() - started, 1e-6),
                      "eta": (time.time() - started) / done * (epochs * steps - done)})
            if index % 40 == 0 and names is not None:
                #  A few of the pictures in this very batch, and what v2.B
                #  currently answers for them.
                with torch.no_grad():
                    probs = out.passes[-1].logits[:len(y1)].softmax(-1)
                    top = probs.max(-1)
                ordered = np.sort(np.asarray(chosen))
                pick = np.linspace(0, len(y1) - 1, 8).astype(int)
                emit({"stage": "samples", "items": [
                    {"source": caches["img"].sources[int(ordered[i])],
                     "truth": names[int(y1[i])], "guess": names[int(top.indices[i])],
                     "p": float(top.values[i])} for i in pick]})
        print("\r  cortex epoch %d/%d  loss %.3f%s" % (epoch + 1, epochs, running / steps, " " * 20))
        if watch is not None:
            watch(epoch + 1, running / steps)
            cortex.train()
    return seen


# ---------------------------------------------------------------------------
# Stage 3: the hippocampus
# ---------------------------------------------------------------------------


def _recall(entorhinal, ca1, bank, bank_labels, queries, vocab, exclude_self=False):
    """Differentiable recall: queries (N,E) against the bank (M,E) -> memory logp."""
    keys = entorhinal(bank)
    q = entorhinal(queries)
    sim = q @ keys.T
    if exclude_self:
        sim = sim - 10.0 * torch.eye(len(q), len(keys), device=sim.device)
    top, index = sim.topk(C.MEMORY_K, dim=-1)
    return ca1.vote(top, bank_labels[index], torch.ones_like(top), vocab), top[:, 0]


def train_memory(entorhinal, ca1, bank, bank_labels, vocab, device):
    """Leave-one-out: every training item recognised by its neighbours alone.

    The bank is embedded once per step and each query's OWN row is masked.
    It used to copy the whole bank into a reordered tensor and build a
    batch x bank identity to find "self": at 219,000 memories that was ~3 GB
    a step, the 6 GB card spilled into shared memory, and one memory epoch
    took 23 minutes. Same arithmetic, same gradients (checked against it)."""
    params = list(entorhinal.parameters()) + [ca1.log_sharpness]
    optimiser = torch.optim.AdamW(params, lr=1e-3, weight_decay=1e-4)
    rng = np.random.default_rng(C.SEED)
    n = len(bank)
    rows = max(64, min(512, C.MEMORY_SIM_BUDGET // n))
    for epoch in range(C.MEMORY_EPOCHS):
        order, running, steps = rng.permutation(n), 0.0, 0
        for start in range(0, n, rows):
            idx = torch.from_numpy(order[start:start + rows]).to(device)
            keys = entorhinal(bank)                       # N,K: gradient into every key
            sim = keys[idx] @ keys.T
            sim = sim.index_put((torch.arange(len(idx), device=device), idx),
                                torch.tensor(-10.0, device=device), accumulate=True)
            top, index = sim.topk(C.MEMORY_K, dim=-1)
            logp = ca1.vote(top, bank_labels[index], torch.ones_like(top), vocab)
            loss = F.nll_loss(logp, bank_labels[idx])
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
            running, steps = running + float(loss.detach()), steps + 1
        print("  memory epoch %d/%d  leave-one-out loss %.3f  sharpness %.1f"
              % (epoch + 1, C.MEMORY_EPOCHS, running / steps, float(ca1.sharpness.detach())))


def set_familiarity(ca1, closest: torch.Tensor) -> None:
    """Familiar = the validation set's own closest-memory similarities. The
    gate is half-open a little below their 5th percentile and fades over the
    spread between their 5th and 25th percentiles."""
    p5, p25 = torch.quantile(closest.float(), torch.tensor([0.05, 0.25], device=closest.device))
    width = max(float(p25 - p5) / 2, 1e-3)
    with torch.no_grad():
        ca1.familiar_at.fill_(float(p5) - width)
        ca1.familiar_width.fill_(width)


def fit_trust(ca1, cortex_logits, memory_logp, labels):
    optimiser = torch.optim.Adam([ca1.log_trust], lr=0.05)
    for _ in range(300):
        loss = F.cross_entropy(cortex_logits + ca1.trust * memory_logp, labels)
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()
    return float(ca1.trust.detach())


def fit_value(amygdala, features, correct):
    optimiser = torch.optim.AdamW(amygdala.parameters(), lr=3e-3, weight_decay=1e-3)
    for _ in range(C.VALUE_EPOCHS):
        loss = F.binary_cross_entropy(amygdala(features), correct)
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()
    return float(loss.detach())


# ---------------------------------------------------------------------------
# The whole run
# ---------------------------------------------------------------------------


def _fingerprint(manifest: dict, caches: dict, cut: tuple) -> dict:
    """What a trained cortex depends on. --refit refuses a cortex whose
    fingerprint differs: a review found it would otherwise load whatever
    v2b_cortex_best.pt held -- e.g. a lesioned cortex from a --cut run."""
    listed = (json.loads(_path("sources.json").read_text())
              if _path("sources.json").exists() else {})
    return {"models": manifest["models"], "dims": manifest["dims"],
            "vocabulary": len(manifest["names"]), "inet_stamp": listed.get("inet_stamp"),
            "snd_stamp": listed.get("snd_stamp"),
            "round7": [listed.get("inet_more2_stamp"), listed.get("vgg_stamp"),
                       [listed.get("stamp_img_%s" % p) for p in REND_SPLITS]],
            "train_on": [C.TRAIN_ON_EXTRA, C.TRAIN_ON_TINY, C.TRAIN_ON_INET],
            "train_images": len(caches[("img", "train")]),
            "train_sounds": len(caches[("aud", "train")]), "cut": sorted(cut)}


def train(epochs: int = C.EPOCHS, cut: tuple = (), refit: bool = False,
          candidate: bool = False) -> dict:
    torch.manual_seed(C.SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    manifest = json.loads(_path("manifest.json").read_text())
    dims = Dims(**manifest["dims"])
    words = torch.from_numpy(np.load(_path("words.npy"))).to(device)
    vocab = len(words)
    caches = {(k, p): Cache(k, p, dims) for k in ("img", "aud") for p in SPLITS}
    tiny = ({p: Cache("img", p, dims) for p in TINY_SPLITS}
            if all(_path("img_%s.labels.npy" % p).exists() for p in TINY_SPLITS) else {})
    extra = Cache("img", "extra", dims) if C.TRAIN_ON_EXTRA else None
    #  MORE DATA: training = all of CIFAR's training pictures + Tiny-ImageNet;
    #  validation = CIFAR val + Tiny val. The CIFAR test set stays the headline
    #  "images" number, comparable with earlier builds; Tiny's test is reported
    #  beside it.
    learn_tiny = C.TRAIN_ON_TINY and bool(tiny)
    inet = ({p: Cache("img", p, dims) for p in INET_SPLITS}
            if all(_path("img_%s.labels.npy" % p).exists() for p in INET_SPLITS) else {})
    learn_inet = C.TRAIN_ON_INET and bool(inet)
    inet_more = (Cache("img", "inet_more", dims)
                 if learn_inet and C.INET_MORE_PER_CLASS and _path("img_inet_more.labels.npy").exists()
                 else None)
    if learn_inet and C.INET_MORE_PER_CLASS and data.have_inet_more() and inet_more is None:
        raise SystemExit("the extra ImageNet photos are fetched but not encoded -- run encode first")
    if inet_more is not None:
        print("  more ImageNet photos: %d" % len(inet_more))
    inet_more2 = (Cache("img", "inet_more2", dims)
                  if learn_inet and C.INET_MORE2_PER_CLASS
                  and _path("img_inet_more2.labels.npy").exists() else None)
    if learn_inet and C.INET_MORE2_PER_CLASS and data.have_inet_more2() and inet_more2 is None:
        raise SystemExit("the round-7 ImageNet photos are fetched but not encoded -- run encode first")
    if inet_more2 is not None:
        print("  round-7 ImageNet photos: %d" % len(inet_more2))
    rend = ({p: Cache("img", p, dims) for p in REND_SPLITS}
            if all(_path("img_%s.labels.npy" % p).exists() for p in REND_SPLITS) else {})
    if C.TRAIN_ON_RENDITIONS and data.have_renditions() and not rend:
        raise SystemExit("the renditions are fetched but not encoded -- run encode first")
    rend_train, rend_dropped, more2_dropped = [], 0, 0
    learn_rend = C.TRAIN_ON_RENDITIONS and bool(rend)
    if inet_more2 is not None or learn_rend:
        #  LEAK BACKSTOP for pictures: a NEW training picture whose SigLIP
        #  identity is >= IMAGE_NEAR_COPY like any validation or test picture's
        #  is dropped -- ImageNet's cross-class duplicates, ImageNet-R's
        #  repeated artwork. Round 6's training splits are left as they were.
        held = [caches[("img", "val")], caches[("img", "test")], tiny.get("tiny_val"),
                tiny.get("tiny_test"), inet.get("inet_val"), inet.get("inet_test")]
        if learn_rend:
            held += [rend["imr_test"], rend["sketch_test"]]
        if _path("img_open.labels.npy").exists():
            held.append(Cache("img", "open", dims))
        held_img = F.normalize(torch.cat([torch.from_numpy(np.asarray(a, np.float32))
                                          for t in held if t is not None
                                          for a in _identities(t)]).to(device), dim=-1)

        def unlike_image_tests(cache, bar):
            ident, rows = cache.arrays["identity"], []
            for start in range(0, len(cache), 1024):      # 1,024 x ~58,000 held: 0.24 GB
                chunk = torch.from_numpy(np.asarray(ident[start:start + 1024], np.float32)).to(device)
                close = (F.normalize(chunk, dim=-1) @ held_img.T).max(-1).values.cpu().numpy()
                rows.extend((start + np.flatnonzero(close < bar)).tolist())
            return Subset(cache, rows), len(cache) - len(rows)

        if inet_more2 is not None:
            inet_more2, more2_dropped = unlike_image_tests(inet_more2, C.IMAGE_NEAR_COPY)
            print("  round-7 ImageNet photos: %d kept (%d copies of a val/test picture dropped)"
                  % (len(inet_more2), more2_dropped))
        if learn_rend:
            for part in ("imr_train", "sketch_train"):
                kept, dropped = unlike_image_tests(rend[part], C.REND_NEAR_COPY)
                rend_train.append(kept)
                rend_dropped += dropped
                print("  renditions %s: %d training pictures (%d copies of a val/test picture "
                      "dropped)" % (part, len(kept), dropped))
        del held_img
        torch.cuda.empty_cache()
    caches[("img", "train")] = Joined([caches[("img", "train")], extra,
                                       tiny.get("tiny_train") if learn_tiny else None,
                                       inet.get("inet_train") if learn_inet else None,
                                       inet_more, inet_more2] + rend_train)
    caches[("img", "val")] = Joined([caches[("img", "val")],
                                     tiny.get("tiny_val") if learn_tiny else None,
                                     inet.get("inet_val") if learn_inet else None])
    tiny_test = tiny.get("tiny_test")
    inet_test = inet.get("inet_test")
    sound = ({p: Cache("aud", p, dims) for p in SOUND_SPLITS}
             if all(_path("aud_%s.labels.npy" % p).exists() for p in SOUND_SPLITS) else {})
    learn_sound = C.TRAIN_ON_MORE_SOUNDS and bool(sound)
    if C.TRAIN_ON_MORE_SOUNDS and not sound:
        raise SystemExit("TRAIN_ON_MORE_SOUNDS is set but the AudioSet/FSD50K sounds are not "
                         "encoded -- refusing to train without them silently")
    #  LEAK BACKSTOP: FSD50K clips cut from ESC-50's validation/test
    #  recordings were removed by freesound ID; this catches what IDs miss (the
    #  same sound uploaded twice). A training clip whose CLAP vector is
    #  >= SOUND_NEAR_COPY like any ESC-50 val/test clip is dropped.
    held = torch.cat([torch.from_numpy(np.asarray(caches[("aud", p)].arrays["aud_joint"],
                                                  np.float32)) for p in ("val", "test")])
    held = F.normalize(held.to(device), dim=-1)
    #  VGGSound is YouTube, like AudioSet: its clips are also checked against
    #  every AudioSet and FSD50K validation and test clip.
    every_held = (F.normalize(torch.cat([held.cpu()] + [
        torch.from_numpy(np.asarray(sound[p].arrays["aud_joint"], np.float32))
        for p in ("snd_val", "snd_atest", "snd_ftest")]).to(device), dim=-1)
        if sound else held)

    def unlike_tests(cache, against=None):
        against = held if against is None else against
        joint = cache.arrays["aud_joint"]
        rows = []
        for start in range(0, len(cache), 4096):
            chunk = torch.from_numpy(np.asarray(joint[start:start + 4096], np.float32)).to(device)
            close = (F.normalize(chunk, dim=-1) @ against.T).max(-1).values.cpu().numpy()
            rows.extend((start + np.flatnonzero(close < C.SOUND_NEAR_COPY)).tolist())
        return Subset(cache, rows), len(cache) - len(rows)

    more_train, near_copies, rare_count, vgg_count, vgg_dropped = None, 0, 0, 0, 0
    if learn_sound:
        more_train, near_copies = unlike_tests(sound["snd_train"])
        print("  more sounds: %d training clips (%d near-copies of ESC-50 val/test dropped)"
              % (len(more_train), near_copies))
        caches[("aud", "val")] = Joined([caches[("aud", "val")], sound["snd_val"]])
        if C.AUDIOSET_EXTRA_FILES and _path("aud_rare_train.labels.npy").exists():
            rare, dropped = unlike_tests(Cache("aud", "rare_train", dims))
            rare_count, near_copies = len(rare), near_copies + dropped
            more_train = Joined([more_train, rare])
            print("  rare sounds: %d training clips (%d near-copies dropped)" % (len(rare), dropped))
        elif C.AUDIOSET_EXTRA_FILES and data.have_extra_sounds():
            raise SystemExit("rare sounds are downloaded but not encoded -- run encode first")
        if C.TRAIN_ON_VGGSOUND and C.VGGSOUND_NAMES and _path("aud_vgg_train.labels.npy").exists():
            vgg, vgg_dropped = unlike_tests(Cache("aud", "vgg_train", dims), every_held)
            vgg_count = len(vgg)
            more_train = Joined([more_train, vgg])
            print("  VGGSound: %d training clips (%d near-copies of a val/test clip dropped)"
                  % (vgg_count, vgg_dropped))
        elif C.TRAIN_ON_VGGSOUND and C.VGGSOUND_NAMES and data.have_vggsound():
            raise SystemExit("VGGSound is fetched but not encoded -- run encode first")
    esc_train = caches[("aud", "train")]
    if more_train is not None:
        caches[("aud", "train")] = Joined([esc_train, more_train])
    if C.TRAIN_ON_INET and not inet:
        raise SystemExit("TRAIN_ON_INET is set but the ImageNet photos are not encoded "
                         "(fetch with data.fetch_imagenet(), then encode) -- refusing to "
                         "overwrite the checkpoint with a run that silently left them out")
    open_test = Cache("img", "open", dims) if _path("img_open.labels.npy").exists() else None
    names = manifest["names"]
    emit({"stage": "start", "train_images": len(caches[("img", "train")]),
          "train_sounds": len(caches[("aud", "train")]), "epochs": epochs,
          "vocabulary": vocab, "models": manifest["models"]})

    wiring = Wiring()
    for name in cut:
        wiring.lesion(name)
    cortex = Cortex(dims, manifest["scale"], wiring,
                    sound_scale=manifest.get("sound_scale", 1.0)).to(device)
    cortex.lexicon = torch.from_numpy(np.load(_path("lexicon.npy"))).to(device)
    trainable = sum(p.numel() for p in cortex.parameters())
    untrained = {kind: run(cortex, caches[(kind, "test")], words, device)[2][-1]
                 for kind in ("img", "aud")}
    #  Tiny-ImageNet zero-shot is the floor its training must beat. Without
    #  it, "88.3% on Tiny" was unreadable: zero-shot scores 56.8%, and the
    #  CIFAR-only recipe dragged it DOWN to 33.0% -- a loss the 10-class
    #  never-trained test (79.5 -> 73.8) had hidden.
    if tiny_test is not None:
        untrained["tiny"] = run(cortex, tiny_test, words, device, keep_logits=False)[2][-1]
    if inet_test is not None:
        untrained["inet"] = run(cortex, inet_test, words, device, keep_logits=False)[2][-1]
    rend_tests, rend_test_dropped = rendition_tests(dims, device, rend) if learn_rend else ({}, {})
    for key, n in rend_test_dropped.items():
        print("  %s test: %d pictures (%d copies of a round-6 training picture left out)"
              % (key, len(rend_tests[key]), n))
    for key, cache in rend_tests.items():
        untrained[key] = run(cortex, cache, words, device, keep_logits=False)[2][-1]
    for key, part in (("aset", "snd_atest"), ("fsd", "snd_ftest")):
        if part in sound:
            untrained[key] = run(cortex, sound[part], words, device, keep_logits=False)[2][-1]
    open_before = run(cortex, open_test, words, device)[2][-1] if open_test else None
    print("untrained (zero-shot through POT): image %.1f%%  audio %.1f%%%s"
          % (untrained["img"], untrained["aud"],
             "  tiny-imagenet %.1f%%" % untrained["tiny"] if "tiny" in untrained else ""))

    emit({"stage": "untrained", "val_image": untrained["img"], "val_sound": untrained["aud"],
          "open": open_before})

    best_epoch = {"score": -1.0, "epoch": 0, "state": None}
    epoch_curve = []

    def watch(epoch, loss):
        """After every epoch: validation accuracy (and the never-trained test,
        shown only -- nothing is chosen by it). The epoch with the best pooled
        validation accuracy -- every validation picture and sound counted
        once -- is the one kept: on the old data validation flattened by
        epoch 4 while the training loss kept falling."""
        val = {k: run(cortex, caches[(k, "val")], words, device)[2][-1] for k in ("img", "aud")}
        seen_open = run(cortex, open_test, words, device)[2][-1] if open_test else None
        n_img, n_aud = len(caches[("img", "val")]), len(caches[("aud", "val")])
        pooled = (val["img"] * n_img + val["aud"] * n_aud) / (n_img + n_aud)
        epoch_curve.append({"epoch": epoch, "loss": round(loss, 4), "val_image": val["img"],
                            "val_sound": val["aud"], "open": seen_open})
        if pooled > best_epoch["score"]:
            best_epoch.update(score=pooled, epoch=epoch,
                              state={k: v.detach().clone() for k, v in cortex.state_dict().items()})
        emit({"stage": "epoch", "epoch": epoch, "loss": loss, "val_image": val["img"],
              "val_sound": val["aud"], "open": seen_open, "best_epoch": best_epoch["epoch"]})

    started = time.time()
    #  --candidate: every output gets "_candidate", so the deployed model is
    #  never overwritten; it is promoted by renaming only if it wins on
    #  validation (and costs no extra disk: D: had ~1.8 GB free).
    tag = (("_without_" + "+".join(n.replace("->", "-") for n in cut)) if cut
           else "_candidate" if candidate else "")
    fingerprint = _fingerprint(manifest, caches, cut)
    if refit:
        #  REFIT: the cortex the last run kept, unchanged -- only the stages
        #  after it run (keep, memory, expertise, amygdala, measurement). The
        #  cortex is deterministic: retraining it would reproduce it in 90 min.
        source = C.CHECKPOINT.with_name("v2b_cortex_best%s.pt" % tag)
        kept = torch.load(source, map_location=device, weights_only=False)
        if kept.get("fingerprint") != fingerprint:
            raise SystemExit("refusing --refit: %s was trained on something else\n  it: %s\n"
                             "  now: %s" % (source, kept.get("fingerprint"), fingerprint))
        cortex.load_state_dict(kept["cortex"])
        best_epoch["epoch"] = kept["epoch"]
        epoch_curve.extend(kept["epoch_curve"])
        epochs, seen, cortex_minutes = kept["epochs"], kept["examples_seen"], kept["cortex_minutes"]
        print("  refit: the cortex kept at epoch %d, not retrained" % best_epoch["epoch"])
    else:
        seen = train_cortex(cortex, words, {"img": caches[("img", "train")], "aud": esc_train,
                                            "aud_more": more_train}, device, epochs,
                            names=names, watch=watch)
        cortex_minutes = (time.time() - started) / 60
    if best_epoch["state"] is not None:
        cortex.load_state_dict(best_epoch["state"])
        print("  kept epoch %d of %d: best pooled validation accuracy (%.2f%%)"
              % (best_epoch["epoch"], epochs, best_epoch["score"]))
        torch.save({"cortex": best_epoch["state"], "epoch": best_epoch["epoch"],
                    "fingerprint": fingerprint, "epoch_curve": epoch_curve, "epochs": epochs,
                    "examples_seen": seen, "cortex_minutes": cortex_minutes},
                   C.CHECKPOINT.with_name("v2b_cortex_best%s.pt" % tag))

    # -- how much of the correction to keep (validation sets only) ---------------
    full_open = run(cortex, open_test, words, device)[2][-1] if open_test else None
    curve = {}
    for keep in C.KEEP_GRID:
        with torch.no_grad():
            cortex.pot.keep.fill_(keep)
        curve[keep] = {k: run(cortex, caches[(k, "val")], words, device)[2][-1]
                       for k in ("img", "aud")}
    best = curve[1.0]
    chosen = min(k for k, v in curve.items()
                 if v["img"] >= best["img"] - C.KEEP_TOLERANCE
                 and v["aud"] >= best["aud"] - C.KEEP_TOLERANCE)
    with torch.no_grad():
        cortex.pot.keep.fill_(chosen)
    emit({"stage": "keep", "curve": {str(k): v for k, v in curve.items()}, "chosen": chosen})
    print("  correction kept: %.2f of full strength (validation: images %.1f%% vs %.1f%%, "
          "sounds %.1f%% vs %.1f%%)" % (chosen, curve[chosen]["img"], best["img"],
                                        curve[chosen]["aud"], best["aud"]))

    # -- every split through the trained cortex ------------------------------
    emit({"stage": "memory", "note": "encoding memories"})
    outs = {key: run(cortex, cache, words, device, keep_logits=key[1] != "train")
            for key, cache in caches.items()}
    if tiny_test is not None:
        outs[("img", "tiny_test")] = run(cortex, tiny_test, words, device)
    if inet_test is not None:
        outs[("img", "inet_test")] = run(cortex, inet_test, words, device)
    for key, cache in rend_tests.items():
        outs[("img", "%s_test" % key)] = run(cortex, cache, words, device)
    for part in ("snd_atest", "snd_ftest"):
        if part in sound:
            outs[("aud", part)] = run(cortex, sound[part], words, device)

    def both(part, i):
        return torch.cat([outs[("img", part)][i], outs[("aud", part)][i]])

    trained_bank = both("train", 1).to(device)
    trained_labels = both("train", 3).to(device)
    #  The store is every training example -- CIFAR, Tiny-ImageNet, ImageNet
    #  photos and sounds.
    bank, bank_labels = trained_bank, trained_labels

    episode_bar = episode_similarity(both("val", 1), both("val", 3))
    print("  taught-episode bar: %.3f (q%.1f of same-class, different-item similarity)"
          % (episode_bar, 100 * C.EPISODE_QUANTILE))

    # -- hippocampus -----------------------------------------------------------
    #  Entorhinal keys learn from the TRAINING examples only (leave-one-out);
    #  the extra examples are stored, not learned from.
    entorhinal = Entorhinal(dims.embed).to(device)
    ca1 = CA1().to(device)
    train_memory(entorhinal, ca1, trained_bank, trained_labels, vocab, device)
    torch.cuda.empty_cache()

    def recall_all(vectors):
        """_recall in 1,024-query chunks (topk is per row: identical results)."""
        rows = max(64, min(1024, 2 * C.MEMORY_SIM_BUDGET // len(bank)))
        with torch.no_grad():
            logp, closest = [], []
            for start in range(0, len(vectors), rows):
                lp, cl = _recall(entorhinal, ca1, bank, bank_labels,
                                 vectors[start:start + rows].to(device), vocab)
                logp.append(lp)
                closest.append(cl)
            return torch.cat(logp), torch.cat(closest)

    def memory_of(part, store, store_labels):
        rows = max(64, min(1024, 2 * C.MEMORY_SIM_BUDGET // len(store)))
        with torch.no_grad():
            logp, closest = [], []
            vectors = both(part, 1).to(device)
            for start in range(0, len(vectors), rows):
                lp, cl = _recall(entorhinal, ca1, store, store_labels,
                                 vectors[start:start + rows], vocab)
                logp.append(lp)
                closest.append(cl)
            return torch.cat(logp), torch.cat(closest)

    val_logits, val_labels = both("val", 0).to(device), both("val", 3).to(device)
    test_logits, test_labels = both("test", 0).to(device), both("test", 3).to(device)
    n_img = len(caches[("img", "test")])

    #  CA1's familiarity gate, from the validation set's similarities.
    _, val_closest_all = memory_of("val", bank, bank_labels)
    set_familiarity(ca1, val_closest_all)
    print("  CA1 familiarity: half-open at similarity %.3f, fading over %.3f"
          % (float(ca1.familiar_at), float(ca1.familiar_width)))

    # -- EXPERTISE: how much correction a FAMILIAR answer gets ----------------
    #  The cautious keep (above) answers first. If that answer is a name the
    #  store holds memories of, the input is re-read at a stronger keep,
    #  chosen here on validation together with CA1's trust. See EXPERTISE in
    #  config_v2b.py for the measurement that motivated it.
    known = torch.zeros(vocab, dtype=torch.bool, device=device)
    known[bank_labels.unique()] = True

    def at_keep(keep, parts):
        """Final-pass logits of these caches with `keep` of the correction."""
        with torch.no_grad():
            cortex.pot.keep.fill_(keep)
        logits = torch.cat([run(cortex, c, words, device)[0] for c in parts]).to(device)
        with torch.no_grad():
            cortex.pot.keep.fill_(chosen)
        return logits

    #  Memory reaches frontal only through both of these, as in brain.perceive.
    use = float(wiring.open("POT->hippocampus") and wiring.open("hippocampus->frontal"))

    def decide(lo, hi, memory, gate, trust):
        """-> (cortex logits that answered, fused, which rows used expertise)."""
        gate = gate * use
        first = lo + trust * gate * memory
        if hi is None:
            return lo, first, torch.zeros(len(lo), dtype=torch.bool, device=lo.device)
        expert = known[first.argmax(-1)]
        used = torch.where(expert[:, None], hi, lo)
        return used, used + trust * gate * memory, expert

    with torch.no_grad():
        val_memory, val_closest = memory_of("val", bank, bank_labels)
        val_gate = ca1.familiarity(val_closest)[:, None]

    def fit_rule(hi):
        """CA1's trust for this rule, on validation (as fit_trust does)."""
        with torch.no_grad():
            ca1.log_trust.zero_()
        optimiser = torch.optim.Adam([ca1.log_trust], lr=0.05)
        for _ in range(300):
            loss = F.cross_entropy(decide(val_logits, hi, val_memory, val_gate, ca1.trust)[1],
                                   val_labels)
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
        with torch.no_grad():
            fused = decide(val_logits, hi, val_memory, val_gate, ca1.trust)[1]
        return float(ca1.trust.detach()), float((fused.argmax(-1) == val_labels).float().mean() * 100)

    val_parts = [caches[("img", "val")], caches[("aud", "val")]]
    candidates = [None]
    if C.EXPERTISE and use and wiring.open("hippocampus->POT"):
        candidates += [k for k in C.KEEP_GRID if k > chosen]
    scores = {}
    for k in candidates:
        hi = at_keep(k, val_parts) if k is not None else None
        scores[k] = fit_rule(hi)
        del hi
    keep_expert = max(candidates, key=lambda k: scores[k][1])
    val_hi = at_keep(keep_expert, val_parts) if keep_expert is not None else None
    trust, _ = fit_rule(val_hi)
    expertise = {"keep": keep_expert, "cautious_keep": chosen,
                 "validation": {("off" if k is None else str(k)): round(v[1], 2)
                                for k, v in scores.items()}}
    emit({"stage": "expertise", "keep": keep_expert, "validation": expertise["validation"]})
    print("  expertise: familiar answers re-read at keep %s (pooled validation %s)"
          % (keep_expert, "  ".join("%s %.2f%%" % kv for kv in expertise["validation"].items())))

    # -- amygdala ------------------------------------------------------------------
    amygdala = Amygdala().to(device)
    with torch.no_grad():
        val_used, val_fused, val_expert = decide(val_logits, val_hi, val_memory, val_gate, ca1.trust)
        val_features = Amygdala.features(val_used, val_memory, val_fused, val_closest)
        val_correct = (val_fused.argmax(-1) == val_labels).float()
        expertise["validation_share_expert"] = round(float(val_expert.float().mean() * 100), 1)
    del val_hi
    fit_value(amygdala, val_features, val_correct)

    def accuracy(scores, y):
        return float((scores.argmax(-1) == y).float().mean() * 100)

    # -- measure on the test set ------------------------------------------------
    results = {}
    with torch.no_grad():
        test_memory, test_closest = memory_of("test", bank, bank_labels)
        test_gate = ca1.familiarity(test_closest)[:, None]
    test_hi = (at_keep(keep_expert, [caches[("img", "test")], caches[("aud", "test")]])
               if keep_expert is not None else None)
    with torch.no_grad():
        labels = test_labels
        used, fused, expert = decide(test_logits, test_hi, test_memory, test_gate, ca1.trust)
        value = amygdala(Amygdala.features(used, test_memory, fused, test_closest))
        for kind, part in (("image", slice(0, n_img)), ("audio", slice(n_img, None))):
            y = labels[part]
            right = (fused[part].argmax(-1) == y)
            answered = value[part] >= C.ABSTAIN_BELOW
            results[kind] = {
                "cortex": accuracy(used[part], y),
                "cortex_cautious": accuracy(test_logits[part], y),
                "expert_share": float(expert[part].float().mean() * 100),
                "memory": accuracy(test_memory[part], y),
                "fused": float(right.float().mean() * 100),
                "answered": float(answered.float().mean() * 100),
                "accuracy_when_answered": float(right[answered].float().mean() * 100)
                if answered.any() else 0.0,
                "accuracy_when_abstaining": float(right[~answered].float().mean() * 100)
                if (~answered).any() else 0.0,
                "per_pass": outs[("img" if kind == "image" else "aud", "test")][2],
            }
    del test_hi
    growth = {"every training example": {
        "memories": len(bank), "trust": round(trust, 4),
        "image": {k: results["image"][k] for k in ("memory", "fused")},
        "audio": {k: results["audio"][k] for k in ("memory", "fused")}}}

    def measure(lo_logits, vectors, labels, cache):
        """A further test set through the same memory, expertise and scoring."""
        with torch.no_grad():
            mem, closest = recall_all(vectors)
            gate = ca1.familiarity(closest)[:, None]
        hi = at_keep(keep_expert, [cache]) if keep_expert is not None else None
        with torch.no_grad():
            lo = lo_logits.to(device)
            used, fused, expert = decide(lo, hi, mem, gate, ca1.trust)
            y = labels.to(device)
            return {"cortex": accuracy(used, y), "cortex_cautious": accuracy(lo, y),
                    "expert_share": float(expert.float().mean() * 100),
                    "memory": accuracy(mem, y), "fused": accuracy(fused, y)}

    if tiny_test is not None:
        t_logits, t_vectors, _, t_labels = outs[("img", "tiny_test")]
        results["tiny"] = dict(measure(t_logits, t_vectors, t_labels, tiny_test),
                               pictures=len(tiny_test))

    if inet_test is not None:
        i_logits, i_vectors, _, i_labels = outs[("img", "inet_test")]
        results["inet"] = dict(measure(i_logits, i_vectors, i_labels, inet_test),
                               pictures=len(inet_test),
                               classes=int(len(np.unique(inet_test.labels))))

    for key, cache in rend_tests.items():
        r_logits, r_vectors, _, r_labels = outs[("img", "%s_test" % key)]
        results[key] = dict(measure(r_logits, r_vectors, r_labels, cache), pictures=len(cache),
                            classes=int(len(np.unique(cache.labels))))

    for key, part in (("aset", "snd_atest"), ("fsd", "snd_ftest")):
        if part in sound:
            s_logits, s_vectors, _, s_labels = outs[("aud", part)]
            results[key] = dict(measure(s_logits, s_vectors, s_labels, sound[part]),
                                clips=len(sound[part]),
                                names=int(len(np.unique(sound[part].labels))))

    #  OPEN VOCABULARY after training: did learning these names damage the
    #  ones never trained on? Measured on pictures of never-trained classes.
    open_result = None
    if open_test is not None:
        o_logits, o_vectors, _, o_labels = run(cortex, open_test, words, device)
        open_result = dict(measure(o_logits, o_vectors, o_labels, open_test),
                           pictures=len(open_test), classes=10,
                           #  Trained ImageNet photos make neighbours of these ten
                           #  (other spaniels, trucks, brass) trained classes, and
                           #  this set has informed design choices: a dev set now.
                           note=("not comparable with runs before 2026-10-04: ~780 newly "
                                 "trained ImageNet classes include neighbours of these 10; "
                                 "used in design decisions, so a development set")
                           if learn_inet else "",
                           untrained=open_before, full_correction=full_open)

    coverage = gradient_coverage(cortex, caches, words, device)
    cuts = {} if cut else ablation(cortex, caches, words, device)
    record = {
        "finished": time.strftime("%Y-%m-%dT%H:%M:%S"), "epochs": epochs,
        "examples_seen": seen, "cortex_minutes": round(cortex_minutes, 2),
        "trainable": {"cortex": trainable,
                      "memory": sum(p.numel() for p in entorhinal.parameters()) + 2,
                      "amygdala": sum(p.numel() for p in amygdala.parameters())},
        "untrained": {"image": untrained["img"], "audio": untrained["aud"],
                      "tiny": untrained.get("tiny"), "inet": untrained.get("inet"),
                      "aset": untrained.get("aset"), "fsd": untrained.get("fsd"),
                      "imr": untrained.get("imr"), "sketch": untrained.get("sketch")},
        "rendition_pictures": sum(len(r) for r in rend_train),
        "rendition_near_copies_dropped": rend_dropped,
        "rendition_test_copies_dropped": rend_test_dropped,
        "sound_near_copies_dropped": near_copies, "rare_sounds": rare_count,
        "inet_more_photos": len(inet_more) if inet_more is not None else 0,
        "inet_more2_photos": len(inet_more2) if inet_more2 is not None else 0,
        "inet_more2_copies_dropped": more2_dropped,
        "vgg_sounds": vgg_count, "vgg_near_copies_dropped": vgg_dropped,
        "best_epoch": best_epoch["epoch"], "epoch_curve": epoch_curve,
        "test": results, "trust": trust, "sharpness": float(ca1.sharpness.detach()),
        "memory_growth": growth, "open_vocabulary": open_result,
        "correction_kept": chosen, "expertise": expertise, "refit": refit,
        "candidate": bool(candidate),
        #  The epoch_curve "loss" is the smoothed loss on dropped-out inputs: its
        #  floor is ~1.06 with these settings, so it does not compare with runs
        #  trained without them.
        "label_smoothing": C.LABEL_SMOOTHING, "feature_dropout": C.FEATURE_DROPOUT,
        "episode_similarity": episode_bar,
        "familiarity": {"half_open_at": float(ca1.familiar_at), "width": float(ca1.familiar_width)},
        "keep_curve_validation": {str(k): v for k, v in curve.items()},
        "vocabulary": vocab,
        "gradient_coverage": coverage, "ablation": cuts, "trained_without": list(cut),
        "models": manifest["models"], "audio_data": manifest["audio"],
        "data": {"image": "CIFAR-100 + Tiny-ImageNet + ImageNet photos: %d train, %d val; "
                          "tests: CIFAR %d, Tiny %d, ImageNet photos %d"
                          % (len(caches[("img", "train")]), len(caches[("img", "val")]),
                             len(caches[("img", "test")]), len(tiny_test) if tiny_test else 0,
                             len(inet_test) if inet_test else 0),
                 "audio": "%s + AudioSet + FSD50K%s: %d train, %d val; tests: ESC-50 %d, "
                          "AudioSet %d, FSD50K %d"
                          % (manifest["audio"], " + VGGSound" if vgg_count else "",
                             len(caches[("aud", "train")]),
                             len(caches[("aud", "val")]), len(caches[("aud", "test")]),
                             len(sound.get("snd_atest", [])), len(sound.get("snd_ftest", [])))},
    }
    path = C.CHECKPOINT.with_name("v2b%s.pt" % tag)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"models": manifest["models"], "dims": manifest["dims"],
                "cortex": cortex.state_dict(), "entorhinal": entorhinal.state_dict(),
                "ca1": ca1.state_dict(), "amygdala": amygdala.state_dict(),
                "keep_expert": keep_expert, "episode_similarity": episode_bar,
                "names": manifest["names"], "kinds": manifest["kinds"], "record": record}, path)
    if not cut:
        sources = caches[("img", "train")].sources + caches[("aud", "train")].sources
        trained = np.ones(len(bank), bool)
        import os
        target = C.MEMORY_BANK.with_name("v2b_memory%s.npz" % tag)
        part = target.with_suffix(".tmp.npz")
        np.savez(part, vectors=bank.cpu().numpy().astype(np.float16),
                 labels=bank_labels.cpu().numpy(), sources=np.array(sources), trained=trained)
        os.replace(part, target)                 # never a half-written bank
    (C.ROOT / "results").mkdir(exist_ok=True)
    (C.ROOT / "results" / ("v2b_train%s.json" % tag)).write_text(json.dumps(record, indent=2))
    emit({"stage": "done", "test": results, "open": open_result, "kept": chosen, "tag": tag,
          "minutes": round((time.time() - started) / 60, 1)})
    report(record)
    return record


def gradient_coverage(cortex, caches, words, device) -> dict:
    cortex.train()
    cortex.zero_grad()
    f1, y1 = caches[("img", "train")].batch(np.arange(64), device)
    f2, y2 = caches[("aud", "train")].batch(np.arange(32), device)
    out = cortex(Features.join(f1, f2), words)
    sum(F.cross_entropy(p.logits, torch.cat([y1, y2])) for p in out.passes).backward()
    missing = [n for n, p in cortex.named_parameters()
               if p.grad is None or float(p.grad.abs().sum()) == 0.0]
    total = sum(1 for _ in cortex.parameters())
    cortex.zero_grad()
    cortex.eval()
    return {"covered": total - len(missing), "total": total, "missing": missing}


def ablation(cortex, caches, words, device) -> dict:
    """Accuracy lost (cortex path) when each connection is cut AFTER training.
    Overstates importance -- see --cut for the honest version."""
    base = {k: run(cortex, caches[(k, "test")], words, device)[2][-1] for k in ("img", "aud")}
    table = {}
    for name in NAMES:
        if name.startswith(("POT->hippocampus", "POT->amygdala", "hippocampus->",
                            "amygdala->", "frontal->motor")):
            continue
        cortex.wiring.lesion(name)
        table[name] = {("image" if k == "img" else "audio"):
                       round(base[k] - run(cortex, caches[(k, "test")], words, device)[2][-1], 2)
                       for k in ("img", "aud")}
        cortex.wiring.lesion(name, cut=False)
    return table


def report(record: dict) -> None:
    print("\n" + "=" * 74)
    print("v2.B TRAINED%s  (%d epochs, %.1f min for cortex)"
          % (" WITHOUT " + ", ".join(record["trained_without"]) if record["trained_without"]
             else " -- CANDIDATE (saved as *_candidate; the dashboard still loads v2b.pt)"
             if record.get("candidate") else "",
             record["epochs"], record["cortex_minutes"]))
    print("models: %s" % "  ".join("%s=%s" % kv for kv in record["models"].items()))
    print("=" * 74)
    print("TEST SET            untrained   cortex@keep  cortex(as answered)   memory   final")
    for kind in ("image", "audio"):
        t = record["test"][kind]
        print("  %-8s          %7.1f%%     %6.1f%%        %6.1f%%          %6.1f%%  %6.1f%%"
              % (kind, record["untrained"][kind], t.get("cortex_cautious", t["cortex"]),
                 t["cortex"], t["memory"], t["fused"]))
    if "tiny" in record["test"]:
        t = record["test"]["tiny"]
        print("  tiny-imagenet (%d pictures)  untrained %.1f%%  cortex %.1f%%  memory %.1f%%  "
              "final %.1f%%" % (t["pictures"], record["untrained"].get("tiny") or 0,
                                t["cortex"], t["memory"], t["fused"]))
    if "inet" in record["test"]:
        t = record["test"]["inet"]
        print("  imagenet photos (%d, %d classes)  untrained %.1f%%  cortex %.1f%%  memory %.1f%%  "
              "final %.1f%%" % (t["pictures"], t["classes"], record["untrained"].get("inet") or 0,
                                t["cortex"], t["memory"], t["fused"]))
    for key, title in (("imr", "imagenet-r renditions"), ("sketch", "imagenet-sketch")):
        if key in record["test"]:
            t = record["test"][key]
            print("  %s (%d, %d classes)  untrained %.1f%%  cortex %.1f%%  memory %.1f%%  "
                  "final %.1f%%" % (title, t["pictures"], t["classes"],
                                    record["untrained"].get(key) or 0, t["cortex"], t["memory"],
                                    t["fused"]))
    for key, title in (("aset", "audioset eval"), ("fsd", "fsd50k eval")):
        if key in record["test"]:
            t = record["test"][key]
            print("  %s (%d clips, %d names)  untrained %.1f%%  cortex %.1f%%  memory %.1f%%  "
                  "final %.1f%%" % (title, t["clips"], t["names"],
                                    record["untrained"].get(key) or 0, t["cortex"], t["memory"],
                                    t["fused"]))
    if record.get("best_epoch"):
        print("  kept epoch %d of %d (best on validation)%s" % (
            record["best_epoch"], record["epochs"], "  [refit]" if record.get("refit") else ""))
    e = record.get("expertise") or {}
    if e:
        print("  expertise: familiar answers at keep %s, others at %.2f; validation %s"
              % (e.get("keep"), e.get("cautious_keep", 0),
                 "  ".join("%s %.2f%%" % kv for kv in e.get("validation", {}).items())))
    for kind in ("image", "audio"):
        t = record["test"][kind]
        print("  %s: amygdala lets %.0f%% through -> %.1f%% right; the %.0f%% it holds back are %.1f%% right"
              % (kind, t["answered"], t["accuracy_when_answered"], 100 - t["answered"],
                 t["accuracy_when_abstaining"]))
        print("  %s by frontal pass (cautious keep): %s"
              % (kind, "  ".join("%.1f%%" % v for v in t["per_pass"])))
    print("CA1 trust in memory %.3f, sharpness %.1f" % (record["trust"], record["sharpness"]))
    o = record.get("open_vocabulary")
    if o:
        print("OPEN VOCABULARY (%d pictures of %d never-trained classes, out of %d names):"
              % (o["pictures"], o["classes"], record.get("vocabulary", 0)))
        print("  untrained %.1f%%   full correction %.1f%%   kept %.2f: cortex %.1f%%  "
              "cortex+memory %.1f%%" % (o["untrained"], o.get("full_correction") or 0,
                                        record.get("correction_kept", 1.0), o["cortex"], o["fused"]))
    for store, g in record.get("memory_growth", {}).items():
        print("  memory store %-28s %6d memories  trust %.3f  image %.1f%% -> %.1f%%  audio %.1f%% -> %.1f%%"
              % (store, g["memories"], g["trust"], g["image"]["memory"], g["image"]["fused"],
                 g["audio"]["memory"], g["audio"]["fused"]))
    cov = record["gradient_coverage"]
    print("gradient coverage: %d / %d%s" % (cov["covered"], cov["total"],
                                            "" if not cov["missing"] else "  MISSING %s" % cov["missing"]))
    if record["ablation"]:
        print("points lost when cut AFTER training (overstates; use --cut):")
        for name, cost in record["ablation"].items():
            print("  %-20s image %6.2f   audio %6.2f" % (name, cost["image"], cost["audio"]))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--fresh", action="store_true")
    parser.add_argument("--epochs", type=int, default=C.EPOCHS)
    parser.add_argument("--cut", action="append", default=[], metavar="CONNECTION")
    parser.add_argument("--watch", action="store_true",
                        help="open the live training monitor window")
    parser.add_argument("--refit", action="store_true",
                        help="reuse the cortex the last run kept; redo the stages after it")
    parser.add_argument("--candidate", action="store_true",
                        help="write every output as *_candidate: the deployed model is untouched")
    args = parser.parse_args()
    LIVE.parent.mkdir(exist_ok=True)
    LIVE.write_text("")                          # a fresh run, a fresh live log
    if args.watch:
        import subprocess
        import sys
        subprocess.Popen([sys.executable, "-m", "v2b.monitor"], cwd=str(C.ROOT))
    _round5_ready()
    encode(fresh=args.fresh)
    train(args.epochs, tuple(args.cut), refit=args.refit, candidate=args.candidate)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
