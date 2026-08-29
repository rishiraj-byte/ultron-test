"""
sensory.py -- un-pooled encoder taps (BUILD_SPEC_V2 section 5, stage 2).

THE PROBLEM THIS FIXES, and why the spec calls it the highest-value change.

v1 pooled every encoder to a single vector before the lobes saw anything.
ResNet's 7x7 feature map was averaged away; MiniLM's token sequence was
mean-pooled; Whisper's frames were collapsed. What reached parietal was one
256-vector with no positions in it, and what reached temporal was one
256-vector with no sequence in it.

Parietal's whole reason to exist is attention BETWEEN POSITIONS. Temporal's is
recurrence OVER ORDER. Both were handed inputs from which position and order
had already been deleted, then asked to exploit them. v1 worked around this by
reinterpreting the 256 vector as "16 positions of 16 dims" -- an arbitrary
reshape with no relationship to anything spatial. That is almost certainly why
neither lobe showed measurable benefit in the v1 ablations: they were given a
capability and no data on which to use it.

WHAT THE TAPS NOW DELIVER

    vision   ResNet-18 layer4   (B, 512, 7, 7)  -> 49 REAL spatial positions
    text     MiniLM             (B, T, 384)     -> real token sequence
    audio    Whisper encoder    (B, 1500, 384)  -> real frame sequence
    sensor   raw beams          (B, 8, 2)       -> 8 real beam directions

Each position is projected independently to LATENT_DIM, so a `SensoryStream` is
(B, N, LATENT_DIM): N genuine places or moments, not one summary.

AUDIO IS SUBSAMPLED, and that is a choice worth stating. Whisper emits 1500
frames for 30 s of audio at ~50 fps. Feeding all 1500 into a GRU every one of
T settling timesteps is 12000 recurrent steps per percept, which dominates the
whole model's cost for a signal that is heavily redundant frame to frame.
Strided mean-pooling to AUDIO_FRAMES keeps the sequence -- order, onsets and
duration all survive -- while making the cost sane. This is downsampling, not
pooling-to-a-point: the thing v1 destroyed is preserved.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

import config


@dataclass
class SensoryStream:
    """One modality as a sequence of positions, already in LATENT_DIM.

    `positions` is what makes this different from v1: N is the number of real
    places (image patches), moments (audio frames) or tokens, and it varies by
    modality and by input.
    """

    modality: str
    #  (B, N, LATENT_DIM)
    tokens: torch.Tensor
    #  what the N axis means, for the dashboard and for error messages
    kind: str = "sequence"

    @property
    def positions(self) -> int:
        return int(self.tokens.shape[1])

    @property
    def batch(self) -> int:
        return int(self.tokens.shape[0])

    def summary(self) -> torch.Tensor:
        """Mean over positions -- (B, LATENT_DIM).

        Used only where a module genuinely needs one vector (frontal fusion,
        the thalamic gate). The sequence itself always reaches the lobe whose
        operator can use it.
        """
        return self.tokens.mean(dim=1)

    def __repr__(self) -> str:
        return (f"SensoryStream({self.modality}, {self.positions} {self.kind}, "
                f"{tuple(self.tokens.shape)})")


class SpatialProjection(nn.Module):
    """raw_dim -> LATENT_DIM, applied INDEPENDENTLY at every position.

    Independence is the point: a shared per-position map preserves the N axis.
    Flattening N into the projection would collapse exactly the structure this
    module exists to keep.
    """

    def __init__(self, raw_dim: int, dim: int = config.LATENT_DIM):
        super().__init__()
        self.raw_dim = raw_dim
        self.proj = nn.Linear(raw_dim, dim)
        self.norm = nn.LayerNorm(dim)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        if tokens.shape[-1] != self.raw_dim:
            raise ValueError(
                f"spatial projection expected last dim {self.raw_dim}, "
                f"got {tuple(tokens.shape)}"
            )
        return self.norm(self.proj(tokens))


class SensoryFront(nn.Module):
    """Frozen backbones tapped BEFORE pooling, plus per-position projections."""

    def __init__(self, encoders, device: str | None = None):
        super().__init__()
        self.encoders = encoders          # the existing EncoderBank, reused
        self.device_name = device or config.resolve_device()
        self.projections = nn.ModuleDict({
            "image": SpatialProjection(config.VISION_RAW_DIM),
            "text": SpatialProjection(config.TEXT_RAW_DIM),
            "audio": SpatialProjection(config.AUDIO_RAW_DIM),
            "sensor": SpatialProjection(2),      # (distance, hit) per beam
        })
        self._vision_tap: dict[str, torch.Tensor] = {}

    # -- per-modality un-pooled taps ------------------------------------
    @torch.no_grad()
    def _vision_grid(self, image) -> torch.Tensor:
        """ResNet-18 layer4, keeping the 7x7 grid. (B, 49, 512)."""
        encoder = self.encoders.image
        encoder._ensure_loaded()
        pixels = encoder.preprocess(image).to(encoder.device)

        captured: dict[str, torch.Tensor] = {}
        handle = encoder.backbone.layer4.register_forward_hook(
            lambda m, i, o: captured.__setitem__("grid", o)
        )
        try:
            encoder.backbone(pixels)
        finally:
            handle.remove()

        grid = captured["grid"]                       # (B, 512, 7, 7)
        b, c, h, w = grid.shape
        return grid.reshape(b, c, h * w).transpose(1, 2)   # (B, 49, 512)

    @torch.no_grad()
    def _text_tokens(self, text) -> torch.Tensor:
        """MiniLM token states, pre-pooling. (B, T, 384)."""
        encoder = self.encoders.text
        encoder._ensure_loaded()
        texts = [text] if isinstance(text, str) else list(text)
        batch = encoder._tokenizer(
            texts, padding=True, truncation=True,
            max_length=config.TEXT_MAX_TOKENS, return_tensors="pt",
        )
        batch = {k: v.to(encoder.device) for k, v in batch.items()}
        return encoder._model(**batch).last_hidden_state

    @torch.no_grad()
    def _audio_frames(self, waveform) -> torch.Tensor:
        """Whisper encoder frames, pre-pooling, strided down. (B, F, 384)."""
        import numpy as np

        encoder = self.encoders.audio
        encoder._ensure_loaded()
        if isinstance(waveform, torch.Tensor):
            wave = waveform.detach().cpu().numpy()
        else:
            wave = np.asarray(waveform, dtype=np.float32)
        wave = np.squeeze(wave)[: config.AUDIO_SAMPLE_RATE * config.AUDIO_MAX_SECONDS]

        inputs = encoder._features(
            wave, sampling_rate=config.AUDIO_SAMPLE_RATE, return_tensors="pt"
        )
        frames = encoder._model(
            inputs.input_features.to(encoder.device)
        ).last_hidden_state                                # (B, 1500, 384)

        # Strided mean-pool to a workable frame count. Order and duration
        # survive; only redundancy between adjacent frames is removed.
        stride = max(frames.shape[1] // config.AUDIO_FRAMES, 1)
        usable = (frames.shape[1] // stride) * stride
        pooled = frames[:, :usable].reshape(
            frames.shape[0], -1, stride, frames.shape[2]
        ).mean(dim=2)
        return pooled[:, : config.AUDIO_FRAMES]

    def _sensor_beams(self, readings) -> torch.Tensor:
        """Rangefinder ring as 8 beam positions of (distance, hit). (B, 8, 2)."""
        import numpy as np

        if isinstance(readings, torch.Tensor):
            values = readings.float()
        else:
            values = torch.as_tensor(np.asarray(readings), dtype=torch.float32)
        if values.dim() == 1:
            values = values.unsqueeze(0)
        return values.reshape(values.shape[0], config.SENSOR_BEAMS, 2).to(
            self.projections["sensor"].proj.weight.device
        )

    # -- public ----------------------------------------------------------
    def encode(self, modality: str, raw) -> SensoryStream:
        tap = {
            "image": (self._vision_grid, "spatial positions"),
            "text": (self._text_tokens, "tokens"),
            "audio": (self._audio_frames, "frames"),
            "sensor": (self._sensor_beams, "beams"),
        }
        if modality not in tap:
            raise KeyError(f"no sensory tap for {modality!r}; known: {sorted(tap)}")
        fn, kind = tap[modality]
        tokens = fn(raw)
        projected = self.projections[modality](tokens.to(
            self.projections[modality].proj.weight.device
        ))
        return SensoryStream(modality=modality, tokens=projected, kind=kind)

    def encode_all(self, inputs: dict) -> dict[str, SensoryStream]:
        return {m: self.encode(m, raw) for m, raw in inputs.items()}


__all__ = ["SensoryStream", "SensoryFront", "SpatialProjection"]
