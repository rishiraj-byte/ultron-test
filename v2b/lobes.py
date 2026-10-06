"""
lobes.py -- the pretrained networks, frozen, placed where the anatomy puts them.

    VentralVision   ONE vision network spanning the ventral stream:
                      blocks 1-6       occipital cortex, V1 -> V4
                      blocks 7-12+head inferotemporal cortex (IT), temporal lobe
    Parietal        a depth network: how near, which way surfaces face
    Auditory        the temporal lobe's auditory cortex: CLAP (or AST / Whisper)
    Frontal         the text half of the vision-language model: turns names
                    into vectors in the same space IT answers in

WHY SPLIT ONE NETWORK ACROSS TWO LOBES. In a primate, recognising an object is
not finished in occipital cortex: V1 -> V2 -> V4 build up features, and
inferotemporal cortex, in the temporal lobe, is where "that is a face" or
"that is a cup" is decided. Earlier v2.B gave all of CLIP to occipital, which
made occipital do temporal's job. Splitting at the midpoint puts identity in
IT and hands the dorsal stream V4-level features, which is what V4 sends to
parietal. It is still ONE network -- the ventral stream IS one pathway.

FROZEN MEANS FROZEN. `requires_grad = False` does not stop a network switching
to training behaviour; in v2 that overwrote BatchNorm statistics and cost 46
points. Every lobe refuses to leave eval mode (`_Frozen`).

LIVE ACTIVITY. With `record=True` each network also returns what every block
did: how strongly each patch was represented, where attention went, and the
fraction of MLP neurons that fired. That is what the dashboard's ACTIVITY view
draws -- real activations, captured by forward hooks, never synthesised.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from v2b import config_v2b as C


def load(loader, name, **kw):
    """Load from the local cache first; only go online if the model is missing.

    Everything v2.B uses is normally on disk already, and the online check
    produced a rate-limit warning on every start for no benefit.
    """
    from transformers.utils import logging
    logging.set_verbosity_error()
    logging.disable_progress_bar()
    try:
        return loader.from_pretrained(name, local_files_only=True, **kw)
    except (OSError, ValueError):
        return loader.from_pretrained(name, **kw)


#  AST's 128-band filter bank at 16 kHz leaves one band empty; transformers
#  warns on every call. Known and harmless for this checkpoint.
warnings.filterwarnings("ignore", message="At least one mel filter has all zero values")


class _Frozen(nn.Module):
    """A pretrained network that never trains, whatever its parent asks."""

    def _freeze(self, net: nn.Module) -> nn.Module:
        for parameter in net.parameters():
            parameter.requires_grad = False
        return net.eval()

    def train(self, mode: bool = True):                      # noqa: D401
        super().train(False)
        return self

    #  The wrapper reports eval too; nn.Module starts every module with
    #  training=True, which the check suite caught on the wrapper itself.
    @property
    def training(self) -> bool:
        return False

    @training.setter
    def training(self, value: bool) -> None:
        pass

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    @property
    def dtype(self) -> torch.dtype:
        return next(self.parameters()).dtype


def _normalise(pixels, mean, std):
    mean = torch.tensor(mean, device=pixels.device, dtype=pixels.dtype)
    std = torch.tensor(std, device=pixels.device, dtype=pixels.dtype)
    return (pixels - mean[:, None, None]) / std[:, None, None]


def _grid(tokens: torch.Tensor, side: int) -> torch.Tensor:
    """N, side*side, W patch tokens -> N, GRID*GRID, W (area-pooled)."""
    n, _, w = tokens.shape
    square = tokens.transpose(1, 2).reshape(n, w, side, side)
    return F.adaptive_avg_pool2d(square, C.GRID).flatten(2).transpose(1, 2)


def _activation_of(block: nn.Module) -> nn.Module | None:
    """The MLP's nonlinearity inside a transformer block -- where a neuron
    either fires or does not. Found by type, so any HF family works."""
    for name, module in block.named_modules():
        kind = type(module).__name__.lower()
        if name.endswith(("activation_fn", "activation", "act")) or "gelu" in kind:
            if not list(module.children()):
                return module
    return None


class _Recorder:
    """Forward hooks on each block's MLP nonlinearity, only while recording."""

    def __init__(self, blocks):
        self.blocks = list(blocks)
        self.handles = []
        self.fired: list = []

    def __enter__(self):
        self.fired = [None] * len(self.blocks)
        for index, block in enumerate(self.blocks):
            act = _activation_of(block)
            if act is None:
                continue

            def hook(module, inputs, output, index=index):
                out = output.float()
                #  Per neuron: mean activation over tokens (sample 0), and the
                #  share of (token, neuron) pairs above zero -- "firing".
                self.fired[index] = ((out[0] > 0).float().mean().item(),
                                     out[0].mean(0).cpu().numpy())
            self.handles.append(act.register_forward_hook(hook))
        return self

    def __exit__(self, *exc):
        for handle in self.handles:
            handle.remove()
        self.handles = []


@dataclass
class Live:
    """What every block of one network did, for sample 0. Display only."""

    blocks: list = field(default_factory=list)   # per block: 2-D map (patches or time)
    attention: list = field(default_factory=list)  # per block: 2-D attention map
    firing: list = field(default_factory=list)   # per block: share of neurons > 0
    neurons: list = field(default_factory=list)  # per block: per-neuron mean activation
    extra: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Vision-language: the ventral stream and frontal share one pretrained model
# ---------------------------------------------------------------------------


def vision_language():
    """Load the configured vision-language model once, for both halves."""
    from transformers import AutoImageProcessor, AutoModel
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    model = load(AutoModel, C.VISION_MODEL, dtype=dtype)
    processor = load(AutoImageProcessor, C.VISION_MODEL)
    family = "siglip" if "siglip" in model.config.model_type else "clip"
    return model, processor, family


@dataclass
class VisionOut:
    occ_grid: torch.Tensor      # N,49,W  V4-level features, for the dorsal stream
    it_grid: torch.Tensor       # N,49,W  IT-level features, for the ventral stream
    identity: torch.Tensor      # N,E     what it is, in frontal's naming space
    occipital: Live | None = None
    inferotemporal: Live | None = None


class VentralVision(_Frozen):
    def __init__(self, model, processor, family: str):
        super().__init__()
        self.family = family
        #  Kept OUTSIDE the module tree (object.__setattr__), so the text tower
        #  it also contains is not registered twice. Used only to switch the
        #  attention kernel, which must be done on the parent model.
        object.__setattr__(self, "_parent", model)
        self.vision = self._freeze(model.vision_model)
        self.project = (self._freeze(model.visual_projection)
                        if hasattr(model, "visual_projection") else None)
        self.mean = tuple(processor.image_mean)
        self.std = tuple(processor.image_std)
        cfg = model.config.vision_config
        self.side = cfg.image_size // cfg.patch_size
        self.width = cfg.hidden_size
        self.blocks = list(self.vision.encoder.layers)
        self.cls = 1 if family == "clip" else 0    # CLIP carries a class token
        #  Where occipital hands over to the ventral stream's IT end: the same
        #  share of the depth whatever the network (12 blocks -> 6, 27 -> 13).
        self.split = max(1, int(len(self.blocks) * C.OCCIPITAL_SHARE))

    @property
    def occipital_blocks(self):
        return self.blocks[:self.split]

    @property
    def it_blocks(self):
        return self.blocks[self.split:]

    @torch.no_grad()
    def forward(self, pixels: torch.Tensor, record: bool = False) -> VisionOut:
        x = _normalise(pixels.to(self.device, self.dtype), self.mean, self.std)
        if record:
            #  Attention maps exist only in the eager kernel; the fast one
            #  fuses them away. Switched for this one call, then back.
            self._parent.set_attn_implementation("eager")
            try:
                with _Recorder(self.blocks) as recorder:
                    out = self.vision(pixel_values=x, output_hidden_states=True,
                                      output_attentions=True)
            finally:
                self._parent.set_attn_implementation("sdpa")
        else:
            out = self.vision(pixel_values=x, output_hidden_states=True)
        states = out.hidden_states                        # embeddings + each block
        occ = states[self.split][:, self.cls:].float()
        last = out.last_hidden_state[:, self.cls:].float()
        pooled = out.pooler_output
        identity = (self.project(pooled) if self.project is not None else pooled).float()
        result = VisionOut(_grid(occ, self.side), _grid(last, self.side), identity)
        if record:
            maps, attention = [], []
            for index in range(1, len(states)):
                tokens = states[index][0, self.cls:].float()
                maps.append(tokens.norm(dim=-1).reshape(self.side, self.side).cpu().numpy())
                a = out.attentions[index - 1][0].float().mean(0)   # heads averaged
                #  CLIP: where the class token looked. SigLIP (no class token):
                #  how much attention each patch received from all the others.
                seen = a[0, self.cls:] if self.cls else a.mean(0)
                attention.append(seen.reshape(self.side, self.side).cpu().numpy())
            fired = recorder.fired
            split = self.split

            def live(a, b):
                return Live(maps[a:b], attention[a:b],
                            [f[0] if f else 0.0 for f in fired[a:b]],
                            [f[1] if f else None for f in fired[a:b]])
            result.occipital = live(0, split)
            result.inferotemporal = live(split, len(maps))
        return result


class Frontal(_Frozen):
    """The names frontal can answer with, as vectors. Open vocabulary:
    `learn_word` adds any phrase at any time, with no training."""

    def __init__(self, model, family: str):
        super().__init__()
        from transformers import AutoTokenizer
        self.family = family
        self.text = self._freeze(model.text_model)
        self.project = (self._freeze(model.text_projection)
                        if hasattr(model, "text_projection") else None)
        self.tokenizer = load(AutoTokenizer, C.VISION_MODEL)
        scale = getattr(model, "logit_scale", None)
        self.scale = float(scale.detach().exp()) if scale is not None else C.LOGIT_SCALE
        self.names: list[str] = []
        self.kinds: list[str] = []
        self.words = torch.zeros(0, 1)

    @torch.no_grad()
    def encode(self, phrases: list[str]) -> torch.Tensor:
        if self.family == "siglip":
            #  SigLIP was trained on lower-cased text padded to 64 tokens;
            #  anything else gives quietly worse vectors.
            tokens = self.tokenizer([p.lower() for p in phrases], padding="max_length",
                                    max_length=64, truncation=True, return_tensors="pt")
        else:
            tokens = self.tokenizer(phrases, padding=True, truncation=True,
                                    return_tensors="pt")
        tokens = {k: v.to(self.device) for k, v in tokens.items()}
        out = self.text(**tokens)
        vectors = out.pooler_output
        if self.project is not None:
            vectors = self.project(vectors)
        return F.normalize(vectors.float(), dim=-1)

    @staticmethod
    def phrase(name: str, kind: str) -> str:
        template = C.SOUND_TEMPLATE if kind == "sound" else C.IMAGE_TEMPLATE
        return template.format(name.replace("_", " "))

    def set_vocabulary(self, names: list[str], kinds: list[str]) -> None:
        phrases = [self.phrase(n, k) for n, k in zip(names, kinds)]
        vectors = [self.encode(phrases[i:i + 128]) for i in range(0, len(phrases), 128)]
        self.names, self.kinds = list(names), list(kinds)
        self.words = torch.cat(vectors)

    def learn_word(self, name: str, kind: str = "image") -> int:
        if name in self.names:
            return self.names.index(name)
        vector = self.encode([self.phrase(name, kind)])
        self.names.append(name)
        self.kinds.append(kind)
        self.words = torch.cat([self.words.to(vector.device), vector])
        return len(self.names) - 1


# ---------------------------------------------------------------------------
# Parietal -- a depth network
# ---------------------------------------------------------------------------


@dataclass
class ParietalOut:
    cells: torch.Tensor         # N,49,388  backbone features + depth geometry
    depth: torch.Tensor         # N,224,224 0 far .. 1 near
    live: Live | None = None


class Parietal(_Frozen):
    def __init__(self):
        super().__init__()
        from transformers import AutoModelForDepthEstimation
        dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        self.net = self._freeze(load(AutoModelForDepthEstimation, C.PARIETAL_MODEL,
                                     dtype=dtype))
        self.blocks = list(self.net.backbone.encoder.layer)
        self.width = self.net.config.backbone_config.hidden_size
        self.cell = self.width + C.DEPTH_STATS

    @torch.no_grad()
    def forward(self, pixels: torch.Tensor, record: bool = False) -> ParietalOut:
        x = _normalise(pixels.to(self.device, self.dtype), C.IMAGENET_MEAN, C.IMAGENET_STD)
        if record:
            with _Recorder(self.blocks) as recorder:
                out = self.net(pixel_values=x, output_hidden_states=True)
        else:
            out = self.net(pixel_values=x, output_hidden_states=True)
        n = x.shape[0]
        depth = out.predicted_depth.float()
        low = depth.flatten(1).min(1).values[:, None, None]
        high = depth.flatten(1).max(1).values[:, None, None]
        depth = (depth - low) / (high - low).clamp_min(1e-6)

        #  Geometry per cell: how near, how varied, and which way the surface
        #  slopes -- orientation read straight off the depth map.
        d = depth[:, None]
        slope_x = F.pad(d[..., :, 1:] - d[..., :, :-1], (0, 1))
        slope_y = F.pad(d[..., 1:, :] - d[..., :-1, :], (0, 0, 0, 1))
        near = F.adaptive_avg_pool2d(d, C.GRID)
        spread = (F.adaptive_avg_pool2d(d * d, C.GRID) - near ** 2).clamp_min(0).sqrt()
        geometry = torch.cat([near, spread, F.adaptive_avg_pool2d(slope_x, C.GRID) * 50,
                              F.adaptive_avg_pool2d(slope_y, C.GRID) * 50], 1)
        tokens = out.hidden_states[-1][:, 1:].float()
        side = int(math.isqrt(tokens.shape[1]))
        cells = torch.cat([_grid(tokens, side).transpose(1, 2).reshape(n, -1, C.GRID, C.GRID),
                           geometry], 1).flatten(2).transpose(1, 2)
        result = ParietalOut(cells, depth)
        if record:
            maps = [s[0, 1:].float().norm(dim=-1).reshape(side, side).cpu().numpy()
                    for s in out.hidden_states[1:]]
            fired = recorder.fired
            result.live = Live(maps, [], [f[0] if f else 0.0 for f in fired],
                               [f[1] if f else None for f in fired])
        return result


# ---------------------------------------------------------------------------
# Auditory cortex -- a sound network (temporal lobe)
# ---------------------------------------------------------------------------


@dataclass
class AuditoryOut:
    bins: torch.Tensor          # N,16,W   the sound in 16 slices of time
    identity: torch.Tensor      # N,W      the whole sound, one vector
    joint: torch.Tensor | None  # N,J      the sound in CLAP's sound-word space
    stats: torch.Tensor         # N,bands+2 what the MGN relay can know
    live: Live | None = None


def _resample(wave: np.ndarray, rate: int, target: int) -> np.ndarray:
    if rate == target or wave.size == 0:
        return wave
    count = max(int(round(wave.size * target / float(rate))), 1)
    return np.interp(np.linspace(0, wave.size - 1, count), np.arange(wave.size),
                     wave).astype(np.float32)


class Auditory(_Frozen):
    """The temporal lobe's auditory cortex. Three families are supported:

      clap     LAION CLAP: sounds and their descriptions share one space, so a
               sound can be named from words alone -- by the temporal lobe's own
               sound lexicon (`lexicon`), which reaches frontal directly.
      ast      Audio Spectrogram Transformer: strong features, fixed labels.
      whisper  Whisper's encoder: a speech model.
    """

    def __init__(self):
        super().__init__()
        from transformers import AutoFeatureExtractor, AutoModel
        dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        model = load(AutoModel, C.AUDITORY_MODEL, dtype=dtype)
        kind = model.config.model_type
        self.family = ("clap" if kind == "clap" else
                       "ast" if "audio-spectrogram" in kind else "whisper")
        self.extractor = load(AutoFeatureExtractor, C.AUDITORY_MODEL)
        self.rate = int(self.extractor.sampling_rate)
        self.joint_width, self.scale = 1, 1.0    # clap: its sound-word space
        if self.family == "clap":
            self.net = self._freeze(model.audio_model)
            self.project = self._freeze(model.audio_projection)
            #  CLAP's own text network builds the sound lexicon once; it is
            #  kept outside the module tree so it is not counted as a lobe.
            object.__setattr__(self, "_text", (model.text_model, model.text_projection))
            ac = model.config.audio_config
            self.width, self.bands = ac.hidden_size, ac.num_mel_bins
            self.joint_width = model.config.projection_dim
            self.scale = float(model.logit_scale_a.detach().exp())
            self.blocks = [b for layer in self.net.audio_encoder.layers for b in layer.blocks]
        else:
            self.net = self._freeze(model.encoder if self.family == "whisper" else model)
            self.project = None
            cfg = model.config
            self.width = cfg.d_model if self.family == "whisper" else cfg.hidden_size
            self.bands = cfg.num_mel_bins
            found = getattr(self.net, "layers", None)
            self.blocks = list(found if found is not None else self.net.encoder.layer)
            if self.family == "ast":
                #  AST cuts the spectrogram into overlapping 16x16 patches with
                #  stride 10: this many rows of frequency, columns of time.
                self.freq_rows = (cfg.num_mel_bins - 16) // cfg.frequency_stride + 1
                self.time_cols = (cfg.max_length - 16) // cfg.time_stride + 1
                self.frames_per_col = cfg.time_stride
                self.tokens_before = 2

    @property
    def stats_width(self) -> int:
        return self.bands + 2

    # -- the sound lexicon: Wernicke's area -------------------------------------
    @torch.no_grad()
    def _clap_text(self, phrases: list[str]) -> torch.Tensor:
        from transformers import AutoTokenizer
        text, projection = self._text
        tokenizer = load(AutoTokenizer, C.AUDITORY_MODEL)
        text, projection = text.to(self.device), projection.to(self.device)
        out = []
        for start in range(0, len(phrases), 128):
            tokens = tokenizer(phrases[start:start + 128], padding=True, truncation=True,
                               return_tensors="pt").to(self.device)
            pooled = text(**tokens).pooler_output
            out.append(F.normalize(projection(pooled).float(), dim=-1))
        return torch.cat(out)

    def lexicon(self, names: list[str], kinds: list[str]) -> torch.Tensor:
        """Every SOUND name as a vector in CLAP's own sound-word space; zero rows
        for picture names. This is the temporal lobe's sound vocabulary --
        Wernicke's area, where heard sounds meet their meanings.

        Why not one shared vocabulary: a linear bridge from CLAP's space into
        frontal's, fitted on text pairs, named ESC-50 test sounds 17.5% right
        where CLAP in its own space got 89.8%. Sounds and words are not in the
        same place even inside CLAP (its "modality gap"), so a map learned on
        words alone did not carry sounds. The lexicon stays in CLAP's space.
        """
        if self.family != "clap":
            return torch.zeros(len(names), 1, device=self.device)
        rows = [i for i, k in enumerate(kinds) if k == "sound"]
        table = torch.zeros(len(names), self.joint_width, device=self.device)
        if rows:
            vectors = self._clap_text(["the sound of %s" % names[i].replace("_", " ")
                                       for i in rows])
            table[rows] = vectors.to(self.device)
        return table

    # -- hearing -------------------------------------------------------------------
    @torch.no_grad()
    def forward(self, waves: list[np.ndarray], record: bool = False) -> AuditoryOut:
        limit = C.AUDIO_SAMPLE_RATE * C.AUDIO_MAX_SECONDS
        captured = [np.asarray(w, np.float32)[:limit] for w in waves]
        heard = [_resample(w, C.AUDIO_SAMPLE_RATE, self.rate) for w in captured]
        feats = self.extractor(heard, sampling_rate=self.rate, return_tensors="pt")
        if self.family == "clap":
            spectrum = feats["input_features"]                     # N,1,1001,64
            kwargs = {"input_features": spectrum.to(self.device, self.dtype),
                      "is_longer": feats["is_longer"].to(self.device)}
        else:
            spectrum = feats.input_features if self.family == "whisper" else feats.input_values
            kwargs = {"input_features" if self.family == "whisper" else "input_values":
                      spectrum.to(self.device, self.dtype)}
        if record:
            with _Recorder(self.blocks) as recorder:
                out = self.net(**kwargs, output_hidden_states=True)
        else:
            out = self.net(**kwargs, output_hidden_states=True)

        bins, identity, stats, keep = [], [], [], []
        for index, wave in enumerate(captured):
            seconds = max(wave.size / C.AUDIO_SAMPLE_RATE, 0.05)
            share = min(seconds / C.AUDIO_MAX_SECONDS, 1.0)
            if self.family == "clap":
                #  HTSAT ends on a 2 (frequency) x 32 (time) grid over a 10 s
                #  window; a shorter sound is repeated to fill it, so only the
                #  first share of the time steps is the sound itself.
                grid = out.last_hidden_state[index].float()           # W,2,32
                count = max(int(math.ceil(grid.shape[-1] * share)), 1)
                frames = grid.mean(1)[:, :count].T                    # time, W
                mel = spectrum[index, 0, :max(int(spectrum.shape[2] * share), 1)].float().T
                ident = out.pooler_output[index].float()
            elif self.family == "whisper":
                count = min(max(int(math.ceil(seconds * 50)), 1), 1500)
                frames = out.last_hidden_state[index, :count].float()
                mel = spectrum[index, :, :count * 2].float()
                ident = frames.mean(0)
            else:
                tokens = out.last_hidden_state[index, self.tokens_before:].float()
                count = min(max(int(math.ceil(seconds * 100 / self.frames_per_col)), 1),
                            self.time_cols)
                frames = tokens.reshape(self.freq_rows, self.time_cols, -1)[:, :count].mean(0)
                mel = spectrum[index, :int(seconds * 100)].float().T
                ident = out.pooler_output[index].float()
            bins.append(F.adaptive_avg_pool1d(frames.T[None], C.TEMPORAL_BINS)[0].T)
            identity.append(ident)
            loud = float(np.sqrt(np.mean(wave ** 2))) if wave.size else 0.0
            stats.append(torch.cat([mel.mean(1).cpu(), torch.tensor([loud, share])]))
            keep.append((mel, count))
        identity = torch.stack(identity)
        joint = (F.normalize(self.project(out.pooler_output).float(), dim=-1)
                 if self.family == "clap" else None)
        result = AuditoryOut(torch.stack(bins), identity, joint, torch.stack(stats).float())
        if record:
            mel, count = keep[0]
            maps, labels = [], []
            if self.family == "clap":
                for stage, state in enumerate(out.hidden_states):   # N,C,H,W per stage
                    maps.append(state[0].float().norm(dim=0).cpu().numpy())
                    labels.append("stage %d" % stage)
            else:
                for state in out.hidden_states[1:]:
                    if self.family == "whisper":
                        maps.append(state[0, :count].float().norm(dim=-1)[None].cpu().numpy())
                    else:
                        grid = state[0, self.tokens_before:].float().reshape(
                            self.freq_rows, self.time_cols, -1)[:, :count]
                        maps.append(grid.norm(dim=-1).cpu().numpy())
            fired = recorder.fired
            result.live = Live(maps, [], [f[0] if f else 0.0 for f in fired],
                               [f[1] if f else None for f in fired],
                               {"spectrogram": mel.cpu().numpy(), "map_labels": labels})
        return result
