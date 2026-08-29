"""
encoders.py -- frozen pretrained backbones + trainable projections.

PROJECT.md section 5, the hybrid position:

    [ frozen pretrained encoder ] -> [ projection ] -> [ your lobes ] -> [ output ]
           not trained              trainable, ->256    trained by you

The backbone is borrowed and frozen. The projection is yours and trains.
Nothing here ever hardcodes a latent size: every output width comes from
config.LATENT_DIM.

All three encoders are real. PROJECT.md originally listed text and audio as
stubs that raise; the backbones were verified to load and to produce exactly
the dims section 5 predicted (512 / 384 / 384), so they are implemented.
Backbone loading is lazy -- constructing an encoder is cheap, the weights
arrive on first use -- so the image path never pays for whisper.
"""

from __future__ import annotations

import io
import math
import wave
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn

import config


# ---------------------------------------------------------------------------
# Projection -- the piece that makes any off-the-shelf model plug in
# ---------------------------------------------------------------------------


class Projection(nn.Module):
    """raw_dim -> LATENT_DIM. The only trainable part of an encoder.

    LayerNorm on the way out keeps the three modalities on a comparable scale
    before they hit the thalamus; without it, whichever backbone happens to
    have the largest activations would dominate frontal fusion for reasons
    that have nothing to do with content.
    """

    def __init__(self, raw_dim: int, latent_dim: int = config.LATENT_DIM):
        super().__init__()
        self.raw_dim = raw_dim
        self.latent_dim = latent_dim
        self.proj = nn.Linear(raw_dim, latent_dim)
        self.norm = nn.LayerNorm(latent_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] != self.raw_dim:
            raise ValueError(
                f"projection expected last dim {self.raw_dim}, got {tuple(x.shape)}"
            )
        return self.norm(self.proj(x))


class BaseEncoder(nn.Module):
    """Common contract: .encode(raw_input) -> (B, LATENT_DIM)."""

    modality: str = "unknown"
    raw_dim: int = 0

    def __init__(self, raw_dim: int, device: str | None = None):
        super().__init__()
        self.raw_dim = raw_dim
        self.projection = Projection(raw_dim)
        #  Where the frozen backbone and its inputs must live. Resolved once,
        #  here, because a backbone left on the CPU while the lobes sit on the
        #  GPU does not run slowly -- it raises a device-mismatch error on the
        #  first matmul.
        self.device = device or config.resolve_device()

    def to_device(self, device: str) -> None:
        """Move this encoder, backbone included, to `device`."""
        self.device = device
        self.to(device)
        for attr in ("_backbone", "_head", "_model"):
            module = getattr(self, attr, None)
            if module is not None:
                module.to(device)

    # -- subclasses implement this --
    def encode_raw(self, raw_input: Any) -> torch.Tensor:
        raise NotImplementedError

    def forward(self, raw_input: Any) -> torch.Tensor:
        features = self.encode_raw(raw_input)
        if features.dim() == 1:
            features = features.unsqueeze(0)
        return self.project_features(features)

    def project_features(self, features: torch.Tensor) -> torch.Tensor:
        """Everything TRAINABLE that sits between the frozen backbone and the
        latent space.

        Training runs from cached backbone outputs, so it needs to enter the
        encoder at exactly this point. Calling `self.projection` directly would
        silently skip any trainable stage an encoder adds in front of it --
        which is what left the sensor encoder's stem untrained.
        """
        return self.projection(features)

    @staticmethod
    def _freeze(module: nn.Module) -> nn.Module:
        for p in module.parameters():
            p.requires_grad_(False)
        module.eval()
        return module

    def trainable_parameters(self):
        return self.projection.parameters()


# ---------------------------------------------------------------------------
# Vision -- frozen ResNet-18
# ---------------------------------------------------------------------------


class VisionEncoder(BaseEncoder):
    """Frozen ResNet-18. 512-d pooled features -> LATENT_DIM.

    Chosen over MobileNetV3 because it is convolutional with wide, clean
    feature maps, so the layer taps in activations.py produce genuine spatial
    heatmaps rather than noise (PROJECT.md section 5).

    The original 1000-way ImageNet head is kept but NOT used in the lobe path.
    It is exposed as `reference_logits` so the dashboard can show what the
    borrowed encoder believes, clearly separated from what the lobes decide,
    and so train.py has a distillation target.
    """

    modality = "image"

    def __init__(self, device: str | None = None) -> None:
        super().__init__(config.VISION_RAW_DIM, device=device)
        self._backbone: nn.Module | None = None
        self._head: nn.Module | None = None
        self._categories: list[str] | None = None
        self._transform = None

    # -- lazy load ----------------------------------------------------------
    def _ensure_loaded(self) -> None:
        if self._backbone is not None:
            return
        from torchvision.models import ResNet18_Weights, resnet18

        weights = ResNet18_Weights.DEFAULT
        net = resnet18(weights=weights)

        if net.fc.in_features != self.raw_dim:
            raise RuntimeError(
                f"config.VISION_RAW_DIM={self.raw_dim} but ResNet-18 exposes "
                f"{net.fc.in_features}. Fix config, not this file."
            )

        self._head = self._freeze(net.fc).to(self.device)
        net.fc = nn.Identity()            # backbone now ends at the 512-d pool
        self._backbone = self._freeze(net).to(self.device)
        self._categories = list(weights.meta["categories"])
        self._transform = weights.transforms()

    @property
    def backbone(self) -> nn.Module:
        self._ensure_loaded()
        assert self._backbone is not None
        return self._backbone

    @property
    def categories(self) -> list[str]:
        self._ensure_loaded()
        assert self._categories is not None
        return self._categories

    # -- preprocessing ------------------------------------------------------
    def preprocess(self, image: Any) -> torch.Tensor:
        """PIL.Image | path | (H,W,3) array | already-batched tensor -> (B,3,H,W)."""
        self._ensure_loaded()
        from PIL import Image

        if isinstance(image, torch.Tensor):
            t = image
            if t.dim() == 3:
                t = t.unsqueeze(0)
            if t.dim() != 4:
                raise ValueError(f"image tensor must be 3-D or 4-D, got {tuple(t.shape)}")
            return t.float()

        if isinstance(image, (str, Path)):
            image = Image.open(image)
        elif isinstance(image, np.ndarray):
            image = Image.fromarray(image.astype(np.uint8))
        elif isinstance(image, (bytes, bytearray)):
            image = Image.open(io.BytesIO(image))

        image = image.convert("RGB")
        return self._transform(image).unsqueeze(0)

    # -- forward ------------------------------------------------------------
    def encode_raw(self, raw_input: Any) -> torch.Tensor:
        pixels = self.preprocess(raw_input).to(self.device)
        with torch.no_grad():
            return self.backbone(pixels)

    @torch.no_grad()
    def reference_head(self, features: torch.Tensor) -> torch.Tensor:
        """The frozen ImageNet head applied to already-computed features.

        Callers that have run the backbone should use this rather than
        reference_logits, which would run the whole of ResNet-18 again.
        """
        self._ensure_loaded()
        assert self._head is not None
        return self._head(features.to(self.device))

    @torch.no_grad()
    def reference_logits(self, raw_input: Any) -> torch.Tensor:
        """What the BORROWED encoder thinks. Not the lobes' answer."""
        return self.reference_head(self.encode_raw(raw_input))


# ---------------------------------------------------------------------------
# Text -- frozen all-MiniLM-L6-v2
# ---------------------------------------------------------------------------


class TextEncoder(BaseEncoder):
    """Frozen MiniLM sentence embeddings (384-d) -> LATENT_DIM.

    Loaded through raw transformers rather than sentence-transformers so the
    pooling is explicit and visible: mean over tokens, mask-weighted, then
    L2 normalise -- exactly what the sentence-transformers wrapper does.
    """

    modality = "text"

    def __init__(self, device: str | None = None) -> None:
        super().__init__(config.TEXT_RAW_DIM, device=device)
        self._model = None
        self._tokenizer = None

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        from transformers import AutoModel, AutoTokenizer

        self._tokenizer = AutoTokenizer.from_pretrained(config.TEXT_MODEL)
        model = AutoModel.from_pretrained(config.TEXT_MODEL)
        hidden = model.config.hidden_size
        if hidden != self.raw_dim:
            raise RuntimeError(
                f"config.TEXT_RAW_DIM={self.raw_dim} but {config.TEXT_MODEL} "
                f"exposes {hidden}. Fix config, not this file."
            )
        self._model = self._freeze(model).to(self.device)

    def encode_raw(self, raw_input: Any) -> torch.Tensor:
        self._ensure_loaded()
        texts = [raw_input] if isinstance(raw_input, str) else list(raw_input)
        if not texts:
            raise ValueError("text encoder received an empty batch")

        batch = self._tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=config.TEXT_MAX_TOKENS,
            return_tensors="pt",
        )
        batch = {k: v.to(self.device) for k, v in batch.items()}
        with torch.no_grad():
            out = self._model(**batch).last_hidden_state       # (B, T, 384)

        mask = batch["attention_mask"].unsqueeze(-1).float()   # (B, T, 1)
        pooled = (out * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
        return torch.nn.functional.normalize(pooled, p=2, dim=1)


# ---------------------------------------------------------------------------
# Audio -- frozen whisper-tiny encoder
# ---------------------------------------------------------------------------


class AudioEncoder(BaseEncoder):
    """Frozen whisper-tiny encoder (d_model 384) -> LATENT_DIM.

    Only the ENCODER half of whisper is used: this system wants an acoustic
    embedding, not a transcript. The decoder is never constructed.

    Waveform loading uses the stdlib `wave` module, so .wav works with no extra
    dependency. Other containers (mp3, m4a, ogg) need a decoder that is not
    installed; load_waveform says so plainly rather than returning silence.
    """

    modality = "audio"

    def __init__(self, device: str | None = None) -> None:
        super().__init__(config.AUDIO_RAW_DIM, device=device)
        self._model = None
        self._features = None

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        from transformers import WhisperFeatureExtractor, WhisperModel

        self._features = WhisperFeatureExtractor.from_pretrained(config.AUDIO_MODEL)
        full = WhisperModel.from_pretrained(config.AUDIO_MODEL)
        if full.config.d_model != self.raw_dim:
            raise RuntimeError(
                f"config.AUDIO_RAW_DIM={self.raw_dim} but {config.AUDIO_MODEL} "
                f"exposes {full.config.d_model}. Fix config, not this file."
            )
        self._model = self._freeze(full.get_encoder()).to(self.device)

    # -- waveform loading ---------------------------------------------------
    @staticmethod
    def load_waveform(path: str | Path) -> np.ndarray:
        """Read an audio file to mono float32 at AUDIO_SAMPLE_RATE."""
        path = Path(path)
        if path.suffix.lower() != ".wav":
            raise NotImplementedError(
                f"{path.suffix} decoding needs soundfile/librosa/torchaudio, none "
                f"of which are installed. Convert to 16-bit PCM .wav first, e.g.\n"
                f"    ffmpeg -i \"{path}\" -ac 1 -ar {config.AUDIO_SAMPLE_RATE} out.wav"
            )

        with wave.open(str(path), "rb") as wf:
            channels = wf.getnchannels()
            width = wf.getsampwidth()
            rate = wf.getframerate()
            frames = wf.readframes(wf.getnframes())

        dtype = {1: np.uint8, 2: np.int16, 4: np.int32}.get(width)
        if dtype is None:
            raise NotImplementedError(f"unsupported WAV sample width: {width} bytes")

        data = np.frombuffer(frames, dtype=dtype).astype(np.float32)
        if dtype is np.uint8:
            data = (data - 128.0) / 128.0
        else:
            data /= float(np.iinfo(dtype).max)

        if channels > 1:
            data = data.reshape(-1, channels).mean(axis=1)

        if rate != config.AUDIO_SAMPLE_RATE:
            # Linear resample. Adequate for an embedding; not audiophile grade.
            n_out = int(round(len(data) * config.AUDIO_SAMPLE_RATE / rate))
            if n_out <= 1:
                raise ValueError(f"audio file too short to resample: {path}")
            src = np.linspace(0.0, len(data) - 1, num=n_out, dtype=np.float64)
            data = np.interp(src, np.arange(len(data)), data).astype(np.float32)

        return data

    def encode_raw(self, raw_input: Any) -> torch.Tensor:
        self._ensure_loaded()

        if isinstance(raw_input, (str, Path)):
            wave_data = self.load_waveform(raw_input)
        elif isinstance(raw_input, torch.Tensor):
            wave_data = raw_input.detach().cpu().numpy().astype(np.float32)
        else:
            wave_data = np.asarray(raw_input, dtype=np.float32)

        wave_data = np.squeeze(wave_data)
        if wave_data.ndim != 1:
            raise ValueError(f"audio must be mono 1-D, got shape {wave_data.shape}")

        limit = config.AUDIO_SAMPLE_RATE * config.AUDIO_MAX_SECONDS
        wave_data = wave_data[:limit]

        inputs = self._features(
            wave_data,
            sampling_rate=config.AUDIO_SAMPLE_RATE,
            return_tensors="pt",
        )
        with torch.no_grad():
            hidden = self._model(
                inputs.input_features.to(self.device)
            ).last_hidden_state                                  # (B,T,384)
        return hidden.mean(dim=1)


# ---------------------------------------------------------------------------
# Proprioception -- a rangefinder ring. No pretrained backbone, by design.
# ---------------------------------------------------------------------------


class SensorEncoder(BaseEncoder):
    """Embodied proximity readings -> LATENT_DIM.

    The other three encoders borrow a frozen backbone because pixels, waveforms
    and characters need a lot of learned machinery before they mean anything.
    A rangefinder does not: the reading already IS the measurement. There is no
    pretrained model worth borrowing here, so the whole encoder is trainable --
    a small MLP over (distance, hit) pairs from a ring of beams.

    Stating that plainly matters for the paper: this modality is NOT carried by
    a large pretrained model, so any navigation performance on it is genuinely
    attributable to the architecture rather than to a borrowed backbone.
    """

    modality = "sensor"

    def __init__(self, device: str | None = None) -> None:
        super().__init__(config.SENSOR_RAW_DIM, device=device)
        self.beams = config.SENSOR_BEAMS
        self.stem = nn.Sequential(
            nn.Linear(config.SENSOR_RAW_DIM, config.SENSOR_RAW_DIM * 8),
            nn.GELU(),
            nn.Linear(config.SENSOR_RAW_DIM * 8, config.SENSOR_RAW_DIM),
        )

    def project_features(self, features: torch.Tensor) -> torch.Tensor:
        """The sensor has no frozen backbone, so its cached "features" are the
        raw beam readings and the stem is part of the trainable path."""
        return self.projection(self.stem(features))

    def encode_raw(self, raw_input: Any) -> torch.Tensor:
        if isinstance(raw_input, torch.Tensor):
            readings = raw_input.float()
        else:
            readings = torch.as_tensor(np.asarray(raw_input), dtype=torch.float32)
        if readings.dim() == 1:
            readings = readings.unsqueeze(0)
        readings = readings.to(self.device)
        if readings.shape[-1] != config.SENSOR_RAW_DIM:
            raise ValueError(
                f"sensor expected {config.SENSOR_RAW_DIM} values "
                f"({self.beams} beams x 2), got {tuple(readings.shape)}"
            )
        return readings          # stem is applied by project_features

    def trainable_parameters(self):
        yield from self.stem.parameters()
        yield from self.projection.parameters()


# ---------------------------------------------------------------------------
# Bundle
# ---------------------------------------------------------------------------


class EncoderBank(nn.Module):
    """One encoder per modality, addressed by the same keys the thalamus uses."""

    def __init__(self, device: str | None = None) -> None:
        super().__init__()
        device = device or config.resolve_device()
        self.image = VisionEncoder(device=device)
        self.text = TextEncoder(device=device)
        self.audio = AudioEncoder(device=device)
        self.sensor = SensorEncoder(device=device)

    def to_device(self, device: str) -> None:
        for name in config.MODALITIES:
            self[name].to_device(device)

    def __getitem__(self, modality: str) -> BaseEncoder:
        try:
            return getattr(self, modality)
        except AttributeError as exc:
            raise KeyError(
                f"no encoder for modality {modality!r}; known: {config.MODALITIES}"
            ) from exc

    def encode(self, modality: str, raw_input: Any) -> torch.Tensor:
        return self[modality](raw_input)

    def trainable_parameters(self):
        for name in config.MODALITIES:
            yield from self[name].trainable_parameters()


__all__ = [
    "Projection",
    "BaseEncoder",
    "VisionEncoder",
    "TextEncoder",
    "AudioEncoder",
    "SensorEncoder",
    "EncoderBank",
]
