"""
senses.py -- turn files into the signals the thalamus receives.

Three input kinds, and video is not a fourth: a clip is split into evenly
spaced frames (which go where images go) and its soundtrack (which goes where
audio goes). Motion between frames is not modelled -- there is no area MT here.

Everything in this file is decoding and bookkeeping. No learning, no network.
(The cochlea -- waveform to spectrogram -- now lives with each auditory
network in lobes.py, because each family expects its own spectrogram.)
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from v2b import config_v2b as C


# ---------------------------------------------------------------------------
# Eyes: one retina for both visual lobes
# ---------------------------------------------------------------------------


def load_image(path):
    from PIL import Image
    return Image.open(path).convert("RGB")


def retina(images) -> torch.Tensor:
    """PIL images (or a uint8 N,H,W,3 array) -> float N,3,224,224 in [0, 1].

    Shortest side to 224, then a centre crop. Done ONCE, here, so occipital
    and parietal see the same pixels and grid cell (i, j) means the same patch
    of the world in both -- the dorsal stream depends on that.
    """
    if isinstance(images, (np.ndarray, torch.Tensor)):
        batch = torch.as_tensor(images).permute(0, 3, 1, 2).float() / 255.0
        return _square(batch)
    tensors = []
    for image in images:
        array = np.array(image.convert("RGB"), dtype=np.uint8)
        tensors.append(_square(
            torch.from_numpy(array).permute(2, 0, 1)[None].float() / 255.0))
    return torch.cat(tensors)


def _square(batch: torch.Tensor) -> torch.Tensor:
    size = C.RETINA_SIZE
    h, w = batch.shape[-2:]
    scale = size / min(h, w)
    nh, nw = max(size, round(h * scale)), max(size, round(w * scale))
    batch = F.interpolate(batch, size=(nh, nw), mode="bicubic",
                          align_corners=False).clamp(0, 1)
    top, left = (nh - size) // 2, (nw - size) // 2
    return batch[:, :, top:top + size, left:left + size].contiguous()


def image_stats(pixels: torch.Tensor) -> torch.Tensor:
    """What a relay can know about an image before cortex looks at it.

    Colour means and spreads, overall contrast, and edge energy -- cheap
    signal statistics, the kind of thing a thalamic relay's gain could follow.
    """
    mean = pixels.mean(dim=(2, 3))
    spread = pixels.std(dim=(2, 3))
    luminance = (0.299 * pixels[:, 0] + 0.587 * pixels[:, 1]
                 + 0.114 * pixels[:, 2])
    contrast = luminance.std(dim=(1, 2))[:, None]
    edges = ((luminance[:, :, 1:] - luminance[:, :, :-1]).abs().mean(dim=(1, 2))
             + (luminance[:, 1:] - luminance[:, :-1]).abs().mean(dim=(1, 2)))
    return torch.cat([mean, spread, contrast, edges[:, None]], dim=1)


# ---------------------------------------------------------------------------
# Ears
# ---------------------------------------------------------------------------


def _resample(wave: np.ndarray, rate: int, target: int) -> np.ndarray:
    #  Linear is enough: the next stop is a 25 ms mel window. The RATE is what
    #  matters -- a 44.1 kHz file read as 16 kHz moves every formant.
    if rate == target or wave.size == 0:
        return wave
    count = max(int(round(wave.size * target / float(rate))), 1)
    source = np.linspace(0.0, wave.size - 1, wave.size)
    wanted = np.linspace(0.0, wave.size - 1, count)
    return np.interp(wanted, source, wave).astype(np.float32)


def _ffmpeg_audio(path) -> np.ndarray:
    import imageio_ffmpeg
    done = subprocess.run(
        [imageio_ffmpeg.get_ffmpeg_exe(), "-nostdin", "-loglevel", "error",
         "-i", str(path), "-vn", "-ac", "1", "-ar", str(C.AUDIO_SAMPLE_RATE),
         "-f", "f32le", "-"], capture_output=True, timeout=120)
    if done.returncode != 0:
        raise RuntimeError("ffmpeg could not decode %s" % Path(path).name)
    return np.frombuffer(done.stdout, dtype="<f4").copy()


def read_audio(path) -> np.ndarray:
    """Any audio file -> float32 mono at AUDIO_SAMPLE_RATE (48 kHz: CLAP's rate)."""
    try:
        import soundfile
        wave, rate = soundfile.read(str(path), dtype="float32")
        if wave.ndim > 1:
            wave = wave.mean(axis=1)
        return _resample(np.asarray(wave, np.float32), int(rate),
                         C.AUDIO_SAMPLE_RATE)
    except Exception:                                         # noqa: BLE001
        #  soundfile reads WAV/FLAC/OGG; ffmpeg covers mp3, m4a and the rest.
        return _ffmpeg_audio(path)


def read_video(path) -> tuple[list, np.ndarray | None]:
    """A clip -> (evenly spaced frames as PIL images, soundtrack or None)."""
    import cv2
    from PIL import Image

    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError("could not open video %s" % Path(path).name)
    total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
    frames = []
    for index in np.linspace(0, max(total - 1, 0), C.VIDEO_FRAMES, dtype=int):
        capture.set(cv2.CAP_PROP_POS_FRAMES, int(index))
        ok, frame = capture.read()
        if ok:
            frames.append(Image.fromarray(frame[:, :, ::-1].copy()))
    capture.release()
    try:
        sound = _ffmpeg_audio(path)
        sound = sound if sound.size > C.AUDIO_SAMPLE_RATE // 10 else None
    except Exception:                                         # noqa: BLE001
        sound = None                                          # silent clip
    return frames, sound
