"""
pretrained_lobes.py -- real pretrained networks inside the four lobes.

WHAT CHANGED AND WHY IT MATTERS.

Until now every lobe's operator was randomly initialised. The architecture was
real -- signals flowed, lesions bit, gradients reached everything -- but the
lobes themselves knew nothing at the start of training. Only the ENCODERS were
pretrained, and they sit outside cortex.

Each lobe now runs a genuinely pretrained network at its core, chosen to match
what that lobe is for rather than assigned arbitrarily:

    occipital   ResNet-18 layer4          convolutional, local receptive
                                          fields, trained on ImageNet
    parietal    MiniLM layer 3            mid-depth self-attention, which is
                                          relation-between-positions
    temporal    Whisper encoder layer     trained on sequences unfolding in
                                          time, which is what temporal does
    frontal     MiniLM final layer        the most abstract representation in
                                          the stack, for fusion and decision

The occipital choice is the neatest of the four and not a coincidence worth
hiding: ResNet-18's layer4 expects exactly 256 input channels, and LATENT_DIM
is 256, so cortical state enters a real ImageNet convolution stack without any
adapter at all on the input side.

FROZEN, following the same rule as the encoders. The pretrained blocks are
knowledge, not capacity to be re-fit on whatever small task is at hand; the
trainable part is the pair of projections that map cortical state into the
teacher's space and back out. That also keeps the gradient-coverage gate
meaningful -- a frozen tensor is excluded from the count by declaration rather
than showing up as a dead parameter someone has to explain.

LOADED ONCE. Four lobes asking for two MiniLM layers would otherwise load
MiniLM twice, and every gate run constructs a Cortex. The cache below is keyed
by model name so the weights are fetched at most once per process, and nothing
is loaded at all until a lobe is actually built with pretraining enabled.
"""

from __future__ import annotations

import torch
import torch.nn as nn

import config

#  name -> loaded backbone. Populated on first use, never evicted.
_CACHE: dict[str, object] = {}


def _minilm_layers():
    """The transformer stack from all-MiniLM-L6-v2, loaded once."""
    if "minilm" not in _CACHE:
        from transformers import AutoModel
        model = AutoModel.from_pretrained(config.TEXT_MODEL)
        model.eval()
        _CACHE["minilm"] = model.encoder.layer
    return _CACHE["minilm"]


def _whisper_layers():
    """A Whisper encoder stack, loaded once.

    Separate from the ENCODER whisper used at the sensory front, which stays
    whisper-tiny: that one only has to turn audio into frames, and a bigger
    model there costs time on every percept. This one is temporal's core, and
    a deeper layer is worth the memory because it runs on cortical state.
    """
    if "whisper" not in _CACHE:
        from transformers import WhisperModel
        model = WhisperModel.from_pretrained(config.TEMPORAL_MODEL)
        model.eval()
        _CACHE["whisper"] = (model.encoder.layers, model.config.d_model)
    return _CACHE["whisper"]


def _resnet_layer4():
    """A ResNet's final convolutional stage, loaded once.

    ResNet-50 by default. Occipital's inductive bias is CONVOLUTION with
    local receptive fields, so a deeper convolutional stage is the honest
    upgrade here -- swapping in a vision transformer would give it more
    parameters and take away the thing that makes it occipital.
    """
    if "resnet" not in _CACHE:
        if config.OCCIPITAL_RESNET == 50:
            from torchvision.models import ResNet50_Weights, resnet50
            net = resnet50(weights=ResNet50_Weights.DEFAULT)
        else:
            from torchvision.models import ResNet18_Weights, resnet18
            net = resnet18(weights=ResNet18_Weights.DEFAULT)
        net.eval()
        #  layer4's INPUT width is what cortical state has to be projected to.
        _CACHE["resnet"] = (net.layer4, net.layer4[0].conv1.in_channels,
                            net.layer4[-1].bn3.num_features
                            if hasattr(net.layer4[-1], "bn3")
                            else net.layer4[-1].bn2.num_features)
    return _CACHE["resnet"]


def _dinov2_blocks():
    """DINOv2's transformer stack: attention over IMAGE PATCHES, loaded once.

    The right prior for parietal, and the previous choice was wrong. Parietal
    attends over 49 real image positions; it was running a MiniLM layer, which
    is attention over WORDS. Same operation, entirely the wrong data to have
    learned it on. DINOv2 attends over image patches, self-supervised, which
    is the same operation on the same kind of thing.
    """
    if "dinov2" not in _CACHE:
        from transformers import AutoModel
        model = AutoModel.from_pretrained(config.PARIETAL_MODEL)
        model.eval()
        _CACHE["dinov2"] = (model.encoder.layer, model.config.hidden_size)
    return _CACHE["dinov2"]


def _qwen_layers():
    """Qwen2.5's decoder stack and its rotary embedder, loaded once.

    The rotary embedder comes along because a Qwen decoder layer cannot be
    called without it: unlike a BERT layer, position is not baked into the
    hidden states, it is applied to queries and keys inside attention. Taking
    the layer and leaving the embedder behind gives a layer that raises.
    """
    if "qwen" not in _CACHE:
        from transformers import AutoModelForCausalLM
        model = AutoModelForCausalLM.from_pretrained(config.LANGUAGE_MODEL)
        model.eval()
        _CACHE["qwen"] = (model.model.layers, model.model.rotary_emb,
                          model.config.hidden_size)
    return _CACHE["qwen"]


def _freeze(module: nn.Module) -> nn.Module:
    for p in module.parameters():
        p.requires_grad_(False)
    module.eval()
    return module


class PretrainedCore(nn.Module):
    """A frozen pretrained block, wrapped so cortical state can pass through it.

    Shape discipline: everything in this system is LATENT_DIM wide, and the
    teachers are not. `into` and `back` are the only trainable parts, and they
    are what learns to speak the teacher's language -- the teacher itself is
    fixed, so what it knows cannot be quietly retrained away on a small task.
    """

    def __init__(self, teacher: nn.Module, teacher_dim: int,
                 dim: int = config.LATENT_DIM, adapt_input: bool = True,
                 name: str = "pretrained", input_dim: int | None = None):
        super().__init__()
        self.teacher = _freeze(teacher)
        #  A readable name for the layer diagram. `type(teacher).__name__`
        #  gives "Sequential" for the ResNet stage, which tells a reader
        #  nothing about what it is or where it came from.
        self.teacher_name = name
        self.teacher_dim = teacher_dim
        self.dim = dim
        #  Identity when the widths already agree, which is the case for
        #  occipital: a real ImageNet convolution takes cortical state as-is.
        #  A convolutional stage takes a different width IN than it gives
        #  OUT, so the input projection targets `input_dim` where one is
        #  supplied and `teacher_dim` otherwise.
        width_in = input_dim if input_dim is not None else teacher_dim
        self.into = (nn.Linear(dim, width_in)
                     if adapt_input and width_in != dim else nn.Identity())
        self.back = (nn.Linear(teacher_dim, dim) if teacher_dim != dim
                     else nn.Identity())
        self.norm = nn.LayerNorm(dim)

    def train(self, mode: bool = True):
        """Put the wrapper in training mode and the TEACHER never.

        `nn.Module.train()` recurses into every child, so one call to
        `brain.train()` was flipping the frozen ResNet's ten BatchNorm layers
        into training mode -- where they update their running statistics from
        whatever they are fed. What they were being fed is cortical state,
        not ImageNet activations, so every training batch overwrote a little
        more of the pretrained normalisation.

        The damage is invisible in the loss and specific to vision: DINOv2,
        Whisper and Qwen use LayerNorm, which has no running statistics, so
        only the convolutional core was affected. Measured, that was the
        difference between 26% and 1.8% on CIFAR-100 -- a probe on the
        UNTRAINED cortex scored 13.3%, so training was making the vision
        pathway worse than not training it at all.

        Freezing parameters is not enough. `requires_grad_(False)` stops
        gradients; it does nothing about buffers.
        """
        super().train(mode)
        self.teacher.eval()
        return self

    def _run(self, hidden: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(B, dim) or (B, N, dim) in, same shape out."""
        vector = x.dim() == 2
        sequence = x.unsqueeze(1) if vector else x
        hidden = self.into(sequence)
        out = self.back(self._run(hidden))
        return self.norm(out.squeeze(1) if vector else out)


class TransformerCore(PretrainedCore):
    """One pretrained transformer layer -- BERT-style or Whisper-style.

    Both take (B, T, D) and return it, but they differ in what they return it
    IN: a tuple for some versions, a bare tensor for others, and the tuple's
    length has changed across transformers releases. Unwrapping defensively
    here is cheaper than pinning a version.
    """

    #  How many positional arguments this teacher wanted, once discovered.
    _arity: int = 0

    def _run(self, hidden: torch.Tensor) -> torch.Tensor:
        #  BertLayer takes (hidden_states); WhisperEncoderLayer requires
        #  (hidden_states, attention_mask) positionally and raises TypeError
        #  without it. Rather than pin a transformers version or special-case
        #  each teacher by name, find the arity once and remember it.
        if self._arity:
            out = self.teacher(hidden, *([None] * (self._arity - 1)))
        else:
            for extra in range(0, 3):
                try:
                    out = self.teacher(hidden, *([None] * extra))
                except TypeError as exc:
                    if "positional argument" not in str(exc):
                        raise
                    continue
                self._arity = extra + 1
                break
            else:
                raise TypeError("could not call %s with 1-3 positional args"
                                % type(self.teacher).__name__)
        while isinstance(out, (tuple, list)):
            out = out[0]
        return out


class QwenCore(PretrainedCore):
    """One decoder layer out of Qwen2.5, running inside a cortical lobe.

    WHY THIS IS DIFFERENT FROM THE OTHER CORES. ResNet, MiniLM and Whisper
    contribute perceptual features -- what an image looks like, what a
    sentence is about. A Qwen decoder layer contributes something else: it is
    a slice of a model trained to CONTINUE text, so what it carries is
    structure about how one thing follows from another. Putting that inside
    frontal, the lobe whose job is deciding what follows, is the closest thing
    in this system to giving a lobe a reasoning prior rather than a sensory
    one.

    Two mechanical differences from a BERT layer, both of which raise if
    ignored. Position is applied inside attention through rotary embeddings
    rather than being added to the hidden states, so the embedder has to come
    with the layer. And the weights are bfloat16, so the input has to be cast
    to meet them and the output cast back.
    """

    def __init__(self, index: int, dim: int = config.LATENT_DIM):
        layers, rotary, hidden = _qwen_layers()
        super().__init__(layers[index], teacher_dim=hidden, dim=dim,
                         name="Qwen2.5 L%d" % index)
        self.rotary = _freeze(rotary)
        self.layer_dtype = next(self.teacher.parameters()).dtype

    def _run(self, hidden: torch.Tensor) -> torch.Tensor:
        positions = torch.arange(hidden.shape[1], device=hidden.device)
        positions = positions.unsqueeze(0).expand(hidden.shape[0], -1)
        cast = hidden.to(self.layer_dtype)
        embeddings = self.rotary(cast, positions)
        out = self.teacher(cast, position_embeddings=embeddings,
                           attention_mask=None)
        while isinstance(out, (tuple, list)):
            out = out[0]
        return out.to(hidden.dtype)


class ConvolutionalCore(PretrainedCore):
    """ResNet-18 layer4, run on cortical state as a spatial map.

    A vector becomes a 1x1 map and a sequence of N positions becomes a square
    grid where one exists -- 49 positions is exactly the 7x7 that ResNet's own
    layer4 operates on, so the geometry the encoder produced is the geometry
    this block sees, rather than an arbitrary reshape.
    """

    def _run(self, hidden: torch.Tensor) -> torch.Tensor:
        b, n, d = hidden.shape
        side = int(round(n ** 0.5))
        square = side * side == n
        if square:
            grid = hidden.transpose(1, 2).reshape(b, d, side, side)
        else:
            grid = hidden.transpose(1, 2).reshape(b, d, 1, n)
        out = self.teacher(grid)                       # (B, 512, h, w)
        return out.flatten(2).transpose(1, 2)          # (B, h*w, 512)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        vector = x.dim() == 2
        sequence = x.unsqueeze(1) if vector else x
        hidden = self.into(sequence)
        out = self.back(self._run(hidden))
        #  layer4 is stride-2, so a 7x7 grid comes back as 4x4 and the
        #  position count changes. Cortex expects the axis it sent, so the
        #  spatial summary is broadcast back over the original positions
        #  rather than silently changing N downstream.
        if not vector and out.shape[1] != sequence.shape[1]:
            out = out.mean(dim=1, keepdim=True).expand(-1, sequence.shape[1], -1)
        return self.norm(out.squeeze(1) if vector else out)


def build_core(lobe: str, dim: int = config.LATENT_DIM) -> PretrainedCore:
    """The pretrained network for one lobe, matched to what that lobe does."""
    if lobe == "occipital":
        stage, channels_in, channels_out = _resnet_layer4()
        #  ResNet-18's layer4 takes 256 channels, which happens to equal
        #  LATENT_DIM, so it needed no input projection. ResNet-50's takes
        #  1024, so it does -- the coincidence was convenient, not load-
        #  bearing, and pretending otherwise would silently reshape state.
        return ConvolutionalCore(
            stage, teacher_dim=channels_out, dim=dim,
            adapt_input=channels_in != dim, input_dim=channels_in,
            name="ResNet%d layer4" % config.OCCIPITAL_RESNET)
    if lobe == "parietal":
        blocks, hidden = _dinov2_blocks()
        return TransformerCore(
            blocks[min(config.PARIETAL_LAYER, len(blocks) - 1)],
            teacher_dim=hidden, dim=dim,
            name="DINOv2 L%d" % config.PARIETAL_LAYER)
    if lobe == "temporal":
        layers, hidden = _whisper_layers()
        return TransformerCore(
            layers[min(config.TEMPORAL_LAYER, len(layers) - 1)],
            teacher_dim=hidden, dim=dim,
            name="Whisper-s L%d" % config.TEMPORAL_LAYER)
    if lobe == "frontal":
        if config.FRONTAL_USES_QWEN:
            #  Frontal decides what follows. A decoder layer from a model
            #  trained to continue text is the nearest thing to a prior for
            #  that, and it is a far larger and better-trained block than the
            #  sentence-encoder layer it replaces.
            return QwenCore(config.PRETRAINED_QWEN_LAYER, dim=dim)
        layers = _minilm_layers()
        return TransformerCore(
            layers[config.PRETRAINED_FRONTAL_LAYER],
            teacher_dim=config.TEXT_RAW_DIM, dim=dim,
            name="MiniLM L%d" % config.PRETRAINED_FRONTAL_LAYER)
    raise KeyError("no pretrained core defined for lobe %r" % lobe)


def describe(cortex=None) -> str:
    """What is actually inside each lobe.

    Built by ASKING the cores rather than from a table. The table version
    went stale the moment frontal's core changed and cheerfully reported
    MiniLM for a lobe running Qwen -- a description that has to be kept in
    sync by hand is a description that will eventually lie.
    """
    why = {
        "occipital": "convolution, local receptive fields",
        "parietal": "attention between positions",
        "temporal": "sequence unfolding in time",
        "frontal": "deciding what follows",
    }
    lines = ["%-11s%-22s%12s  %s"
             % ("lobe", "pretrained core", "frozen", "why it is there")]
    for lobe in ("occipital", "parietal", "temporal", "frontal"):
        if cortex is not None:
            core = getattr(cortex.lobes[lobe], "core", None)
        else:
            core = build_core(lobe)
        if core is None:
            lines.append("%-11s%-22s%12s  %s"
                         % (lobe, "randomly initialised", "-", why[lobe]))
            continue
        frozen = sum(p.numel() for p in core.teacher.parameters())
        lines.append("%-11s%-22s%12s  %s"
                     % (lobe, core.teacher_name, format(frozen, ","), why[lobe]))
    return "\n".join(lines)


__all__ = ["build_core", "PretrainedCore", "TransformerCore",
           "ConvolutionalCore", "QwenCore", "describe"]
