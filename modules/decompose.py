"""
decompose.py -- input decomposition. PROJECT.md section 8.

"Decomposition sits BEFORE the thalamus."

    video -> frames + audio track
    image -> pixels + detected text regions
    text found inside an image branches to TEMPORAL for detection and
    identification, then to FRONTAL for processing.

That last rule is why this module exists rather than being folded into the
encoders: one file can yield inputs for TWO modalities, which the thalamus
then relays to two different lobes, and frontal fuses. It is the same
multi-stream case the section 4 fusion fix was built for.

Honest limit, stated rather than faked: text-region DETECTION is implemented
(MSER, no model download, works offline). Text IDENTIFICATION -- actual OCR --
is not: no OCR engine is installed. When regions are found, this module
reports WHERE and HOW MUCH text there is, not WHAT it says. `text_payload`
makes that explicit in the string it hands to temporal, so nothing downstream
can mistake a layout description for a transcript.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

import numpy as np

import config

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp", ".tif", ".tiff"}
VIDEO_SUFFIXES = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"}
AUDIO_SUFFIXES = {".wav", ".mp3", ".m4a", ".flac", ".ogg"}


@dataclass
class TextRegion:
    x: int
    y: int
    w: int
    h: int

    @property
    def area(self) -> int:
        return self.w * self.h


@dataclass
class Decomposition:
    """What one file became. `inputs` is ready to hand to BrainAI.perceive()."""

    source: str
    kind: str                                   # image | video | audio | text
    inputs: dict[str, Any] = field(default_factory=dict)
    regions: list[TextRegion] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    frame_count: int = 0
    # every sampled frame, for video; `inputs` carries only the one that drives
    # this pass, so a clip can be re-run frame by frame without re-decoding
    frames: list[Any] = field(default_factory=list)

    def describe(self) -> str:
        parts = [f"{self.kind}: {self.source}"]
        parts += [f"  -> {m}" for m in self.inputs]
        parts += [f"  note: {n}" for n in self.notes]
        return "\n".join(parts)


# ---------------------------------------------------------------------------
# Text regions inside an image (PROJECT.md section 8)
# ---------------------------------------------------------------------------


def detect_text_regions(
    image: np.ndarray,
    min_area: int = 60,
    max_regions: int = 400,
) -> list[TextRegion]:
    """MSER text-like region detection. No model, no network, no OCR.

    MSER finds Maximally Stable Extremal Regions -- blobs whose shape survives
    a wide range of intensity thresholds. Glyphs are the textbook example.
    Filtered by aspect ratio and area to drop the obvious non-glyphs.
    """
    try:
        import cv2
    except ImportError:
        return []

    grey = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if image.ndim == 3 else image

    mser = cv2.MSER_create()
    try:
        regions, _ = mser.detectRegions(grey)
    except Exception:
        return []

    found: list[TextRegion] = []
    for points in regions:
        x, y, w, h = cv2.boundingRect(points.reshape(-1, 1, 2))
        if w * h < min_area or h == 0:
            continue
        aspect = w / h
        if not (0.1 < aspect < 12.0):    # glyphs and words, not smears
            continue
        if h > grey.shape[0] * 0.6:      # taller than most of the frame
            continue
        found.append(TextRegion(int(x), int(y), int(w), int(h)))

    found.sort(key=lambda r: r.area, reverse=True)
    found = found[:max_regions]

    # MSER finds stable blobs, and a blob is not a letter. On a photograph or a
    # 3D render it fires on highlights, folds and edges -- measured 139 "text
    # regions" on a Batman render containing no text at all, which then injected
    # a junk stream into temporal and destroyed the image answer.
    #
    # Real text has a property isolated blobs do not: many glyphs of SIMILAR
    # HEIGHT sitting on SHARED BASELINES. Requiring that is what separates a
    # line of type from a field of speckle.
    return _keep_textlike(found)


def _keep_textlike(regions: list[TextRegion], min_run: int = 4) -> list[TextRegion]:
    """Keep only regions that sit in rows of similar-height neighbours.

    A run of >= min_run boxes with comparable height sharing a baseline is
    text. Anything else is texture, and passing texture to the language lobe is
    worse than passing nothing.
    """
    if len(regions) < min_run:
        return []

    heights = np.array([r.h for r in regions], dtype=np.float32)
    typical = float(np.median(heights))
    if typical < 5:                        # too small to be legible glyphs
        return []

    # Bucket by ABSOLUTE vertical position, in units of the typical glyph
    # height. Bucketing by each region's own height (the first attempt) put
    # same-line glyphs of different sizes in different rows, which found rows
    # in noise and missed real ones.
    band = max(typical * 0.7, 1.0)
    rows: dict[int, list[TextRegion]] = {}
    for r in regions:
        if not (0.5 * typical <= r.h <= 2.0 * typical):
            continue                       # not glyph-sized for this image
        rows.setdefault(int((r.y + r.h / 2) // band), []).append(r)

    kept: list[TextRegion] = []
    for row in rows.values():
        if len(row) < min_run:
            continue
        # Glyphs on a line are also spread horizontally, not stacked.
        xs = sorted(r.x for r in row)
        if xs[-1] - xs[0] < typical * min_run * 0.5:
            continue
        kept.extend(row)
    return kept


def text_payload(regions: list[TextRegion], shape: tuple[int, int]) -> str:
    """Turn detected regions into the string that goes to TEMPORAL.

    Deliberately worded as a layout description, never as a transcript,
    because no OCR ran. See the module docstring.
    """
    h, w = shape[:2]
    if not regions:
        return "no text regions detected in this image"

    xs = [r.x + r.w / 2 for r in regions]
    ys = [r.y + r.h / 2 for r in regions]
    coverage = sum(r.area for r in regions) / float(w * h)

    def band(v: float, size: int) -> str:
        f = v / size
        return "upper" if f < 0.33 else ("middle" if f < 0.66 else "lower")

    vertical = band(float(np.mean(ys)), h)
    horizontal = {"upper": "left", "middle": "centre", "lower": "right"}[
        band(float(np.mean(xs)), w)
    ]
    return (
        f"detected {len(regions)} text-like regions, mostly {vertical} "
        f"{horizontal}, covering {coverage:.1%} of the frame "
        f"(layout only -- no OCR engine installed, so the wording is unknown)"
    )


# ---------------------------------------------------------------------------
# Per-kind decomposition
# ---------------------------------------------------------------------------


#  Text-in-image branching is OFF by default. See decompose_image.
DETECT_TEXT_IN_IMAGES = False


def decompose_image(path: str | Path, find_text: bool | None = None) -> Decomposition:
    """image -> pixels (+ text regions -> temporal, when enabled).

    THE TEXT BRANCH IS OFF BY DEFAULT, and that is a correction rather than a
    limitation being hidden.

    PROJECT.md section 8 specifies that text found inside an image should branch
    to temporal. Detection without RECOGNITION cannot honour that: with no OCR
    engine the only thing this module can hand to the language lobe is a
    description of the layout ("N text-like regions, mostly centre"), which is
    metadata about the picture, not content from it.

    Measured harm, on a 3D character render containing no text at all: MSER
    reported 139 regions, the layout string was routed to temporal, frontal
    fused it with the visual stream, and the answer degraded from
    `cuirass 32%` (correct -- the subject wears armour) to `trench coat 7%`.
    A stream that is confidently wrong is worse than an absent one, and
    frontal's zero-padding already handles absence correctly.

    Tightening the detector was tried first and does not rescue it: requiring
    aligned rows of similar-height glyphs still missed rendered text while
    firing on specular highlights. The blocker is the missing recogniser, not
    the threshold.

    Install an OCR engine (pytesseract or easyocr), make `text_payload` return
    the transcript, and set `find_text=True` -- then the branch carries content
    and section 8 is satisfied properly.
    """
    from PIL import Image

    if find_text is None:
        find_text = DETECT_TEXT_IN_IMAGES

    path = Path(path)
    image = Image.open(path).convert("RGB")
    result = Decomposition(source=str(path), kind="image", frame_count=1)
    result.inputs["image"] = image

    if not find_text:
        return result

    array = np.asarray(image)
    regions = detect_text_regions(array)
    result.regions = regions
    if regions:
        # THE BRANCH from PROJECT.md section 8: text found inside an image goes
        # to temporal. The thalamus routes it; frontal fuses the two streams.
        result.inputs["text"] = text_payload(regions, array.shape)
        result.notes.append(
            f"{len(regions)} text-like regions -> temporal (section 8 branch)"
        )
    else:
        result.notes.append("no text regions; image path only")
    return result


def decompose_video(
    path: str | Path,
    max_frames: int = 8,
    find_text: bool = False,
) -> Decomposition:
    """video -> frames (+ audio track, when a decoder exists).

    Frames are sampled evenly across the whole clip rather than taken from the
    head, so a slow opening does not decide what the system sees.
    """
    try:
        import cv2
    except ImportError as exc:
        raise NotImplementedError("video decomposition needs opencv-python") from exc

    from PIL import Image

    path = Path(path)
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise ValueError(f"could not open video: {path}")

    total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    result = Decomposition(source=str(path), kind="video")

    wanted = (
        np.linspace(0, total - 1, num=min(max_frames, total), dtype=int)
        if total > 0
        else np.arange(max_frames)
    )

    frames: list[Image.Image] = []
    for index in wanted:
        capture.set(cv2.CAP_PROP_POS_FRAMES, int(index))
        ok, frame = capture.read()
        if not ok:
            break
        frames.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
    capture.release()

    if not frames:
        raise ValueError(f"no frames decoded from {path}")

    result.frame_count = len(frames)
    result.inputs["image"] = frames[len(frames) // 2]   # the middle frame
    result.notes.append(
        f"sampled {len(frames)} frames evenly across {total or 'unknown'} total; "
        f"the middle frame drives this pass -- use iter_video_frames() for all"
    )
    result.frames = frames

    # OpenCV decodes video, never audio. Say so instead of silently dropping it.
    result.notes.append(
        "audio track NOT extracted: opencv decodes video only. To include it:\n"
        f"    ffmpeg -i \"{path.name}\" -ac 1 -ar {config.AUDIO_SAMPLE_RATE} track.wav\n"
        "    then load track.wav as the audio input"
    )

    if find_text:
        array = np.asarray(result.inputs["image"])
        regions = detect_text_regions(array)
        if regions:
            result.regions = regions
            result.inputs["text"] = text_payload(regions, array.shape)
    return result


def iter_video_frames(path: str | Path, stride: int = 1) -> Iterator[Any]:
    """Every nth frame, as PIL images. For streaming a clip through the brain."""
    import cv2
    from PIL import Image

    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise ValueError(f"could not open video: {path}")
    index = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if index % stride == 0:
                yield Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            index += 1
    finally:
        capture.release()


def decompose_audio(path: str | Path) -> Decomposition:
    from modules.encoders import AudioEncoder

    path = Path(path)
    result = Decomposition(source=str(path), kind="audio")
    result.inputs["audio"] = AudioEncoder.load_waveform(path)
    seconds = len(result.inputs["audio"]) / config.AUDIO_SAMPLE_RATE
    result.notes.append(f"{seconds:.1f}s mono @ {config.AUDIO_SAMPLE_RATE} Hz")
    if seconds > config.AUDIO_MAX_SECONDS:
        result.notes.append(
            f"only the first {config.AUDIO_MAX_SECONDS}s reach whisper's window"
        )
    return result


def decompose(path_or_text: str | Path, find_text: bool | None = None) -> Decomposition:
    """The one entry point. Dispatches on suffix; anything else is raw text."""
    raw = str(path_or_text)
    candidate = Path(raw)
    suffix = candidate.suffix.lower()

    if suffix in IMAGE_SUFFIXES and candidate.exists():
        return decompose_image(candidate, find_text=find_text)
    if suffix in VIDEO_SUFFIXES and candidate.exists():
        return decompose_video(candidate, find_text=find_text)
    if suffix in AUDIO_SUFFIXES and candidate.exists():
        return decompose_audio(candidate)

    result = Decomposition(source=raw[:60], kind="text")
    result.inputs["text"] = raw
    return result


__all__ = [
    "Decomposition",
    "TextRegion",
    "decompose",
    "decompose_image",
    "decompose_video",
    "decompose_audio",
    "detect_text_regions",
    "DETECT_TEXT_IN_IMAGES",
    "iter_video_frames",
    "text_payload",
]
