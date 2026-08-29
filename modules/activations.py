"""
activations.py -- forward hooks, heatmap conversion, and the hand-off queue.

PROJECT.md section 6.

Three jobs, kept separate on purpose:

  1. ActivationRecorder -- registers forward hooks and captures raw tensors.
     Hooks are callbacks that fire when data passes through a layer, handing
     you the intermediate tensor. The model itself is never modified.

  2. to_heatmap -- turns a captured tensor into a normalised 2-D grid.
        conv (C,H,W)   -> mean across channels -> the classic "looking here" map
        flat (LATENT)  -> reshape to LATENT_GRID_SIDE square
     Brightness is normalised PER LAYER, so contrast stays visible even on
     small-magnitude weights.

  3. ActivationBus -- the queue. THE THREADING RULE: never render inside the
     inference loop. Inference pushes; the UI drains on its own timer. The
     queue is deliberately shallow (config.ACTIVATION_QUEUE_MAXSIZE) and drops
     the OLDEST frame when full -- a deep queue lets the display drift behind
     reality.
"""

from __future__ import annotations

import queue
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable

import numpy as np
import torch
import torch.nn as nn

import config
from modules.topology import IN_SUFFIX


# ---------------------------------------------------------------------------
# 1. Capture
# ---------------------------------------------------------------------------


class ActivationRecorder:
    """Registers forward hooks on named modules and keeps the latest tensors."""

    def __init__(self) -> None:
        self._activations: dict[str, torch.Tensor] = {}
        self._handles: list[torch.utils.hooks.RemovableHandle] = []
        self._order: list[str] = []
        self._lock = threading.Lock()

    def _hook(self, name: str, capture_input: bool) -> Callable:
        def fn(module: nn.Module, inputs: Any, output: Any) -> None:
            tensor = output[0] if isinstance(output, (tuple, list)) else output
            if isinstance(tensor, torch.Tensor):
                with self._lock:
                    self._activations[name] = tensor.detach().to("cpu")
            if capture_input and inputs:
                first = inputs[0]
                if isinstance(first, torch.Tensor):
                    with self._lock:
                        self._activations[name + IN_SUFFIX] = first.detach().to("cpu")
        return fn

    def watch(
        self, name: str, module: nn.Module, capture_input: bool = False
    ) -> None:
        """Hook `module`. With capture_input, also record what went IN.

        The input tensor is what gives a layer diagram a real input column --
        otherwise the drawing has to start at the first weight matrix and the
        network appears to have one layer fewer than it has.
        """
        if name in self._order:
            raise ValueError(f"already watching {name!r}")
        self._handles.append(
            module.register_forward_hook(self._hook(name, capture_input))
        )
        self._order.append(name)
        if capture_input:
            self._order.append(name + IN_SUFFIX)

    def watch_many(self, modules: Iterable[tuple]) -> None:
        """Accepts (name, module) or (name, module, capture_input)."""
        for entry in modules:
            if len(entry) == 3:
                self.watch(entry[0], entry[1], entry[2])
            else:
                self.watch(entry[0], entry[1])

    def snapshot(self) -> dict[str, torch.Tensor]:
        with self._lock:
            return dict(self._activations)

    def clear(self) -> None:
        with self._lock:
            self._activations.clear()

    def remove(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
        self._order.clear()

    @property
    def names(self) -> list[str]:
        return list(self._order)

    def __enter__(self) -> "ActivationRecorder":
        return self

    def __exit__(self, *exc: object) -> None:
        self.remove()


# ---------------------------------------------------------------------------
# 2. Tensor -> heatmap
# ---------------------------------------------------------------------------


def to_heatmap(tensor: torch.Tensor, grid_side: int = config.LATENT_GRID_SIDE) -> np.ndarray:
    """Any activation tensor -> a float32 2-D array normalised to 0..1.

    Per PROJECT.md section 6.2:
      convolutional (B,C,H,W) or (C,H,W) -> mean across channels -> (H,W)
      token (B,T,D)                      -> mean across dim      -> (T-ish square)
      flat (B,N) or (N,)                 -> square reshape
    """
    t = tensor.detach().float()
    if t.dim() == 4:
        t = t[0]
    if t.dim() == 3:
        if t.shape[0] > 1 and t.shape[1] > 1 and t.shape[2] > 1:
            grid = t.mean(dim=0)                       # (C,H,W) -> (H,W)
        else:
            grid = t.reshape(t.shape[-2], t.shape[-1])
    elif t.dim() == 2:
        if t.shape[0] == 1:
            grid = _square(t[0], grid_side)            # (1,N) batch of one
        else:
            grid = t                                   # already 2-D
    elif t.dim() == 1:
        grid = _square(t, grid_side)
    else:
        grid = t.reshape(1, -1)

    return _normalise(grid.numpy())


def to_vector(tensor: torch.Tensor) -> np.ndarray:
    """Any flat activation -> a 1-D float32 array, RAW, in network order.

    No reshaping and no normalisation. Neuron i stays at index i, and the sign
    is preserved -- which is what lets the diagram distinguish a unit that is
    genuinely OFF (GELU output ~0) from one that is merely small.
    """
    t = tensor.detach().float()
    if t.dim() > 1:
        t = t[0]                       # batch of one
    return t.reshape(-1).numpy().astype(np.float32)   # (C,1) flattens to (C,)



def _square(flat: torch.Tensor, grid_side: int) -> torch.Tensor:
    """Flat vector -> squarest 2-D grid, zero-padded if it does not divide."""
    n = flat.numel()
    side = grid_side if grid_side * grid_side == n else int(np.ceil(np.sqrt(n)))
    pad = side * side - n
    if pad:
        flat = torch.cat([flat, torch.zeros(pad, dtype=flat.dtype)])
    return flat.reshape(side, side)


def _normalise(grid: np.ndarray) -> np.ndarray:
    """Per-layer min-max to 0..1. Standard heatmap practice (section 6.2)."""
    lo = float(np.nanmin(grid))
    hi = float(np.nanmax(grid))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi - lo < 1e-12:
        return np.zeros_like(grid, dtype=np.float32)
    return ((grid - lo) / (hi - lo)).astype(np.float32)


def heatmap_stats(tensor: torch.Tensor) -> dict[str, float]:
    """Raw numeric scale, kept alongside the normalised map so the dashboard
    can show that two equally-bright heatmaps are not equally active."""
    t = tensor.detach().float()
    return {
        "min": float(t.min()),
        "max": float(t.max()),
        "mean": float(t.mean()),
        "l2": float(t.norm()),
    }


# ---------------------------------------------------------------------------
# 3. The hand-off queue
# ---------------------------------------------------------------------------


@dataclass
class ActivationFrame:
    """One inference pass, packaged for the UI. Plain data -- no tensors."""

    sequence: int
    modality: str
    timestamp: float
    #  layer name -> normalised 0..1 grid (convolutional taps only)
    heatmaps: dict[str, np.ndarray] = field(default_factory=dict)
    #  layer name -> RAW per-neuron activation vector, in network order.
    #  This is what the layer diagrams draw. Unlike a heatmap it is not
    #  reshaped, not rearranged and not normalised -- neuron i is at index i.
    vectors: dict[str, np.ndarray] = field(default_factory=dict)
    #  layer name -> raw scale stats
    stats: dict[str, dict[str, float]] = field(default_factory=dict)
    #  which graph edges carried signal this pass
    active_edges: list[tuple[str, str]] = field(default_factory=list)
    active_nodes: list[str] = field(default_factory=list)
    #  free-form extras (gate value, routing, answer text)
    meta: dict[str, Any] = field(default_factory=dict)


class ActivationBus:
    """Thread-safe hand-off. Inference pushes, the UI drains on its own timer.

    Drops the oldest frame when full, never blocks the producer. A stalled UI
    must not become a stalled model (PROJECT.md section 6.3).
    """

    def __init__(self, maxsize: int = config.ACTIVATION_QUEUE_MAXSIZE):
        self._q: queue.Queue[ActivationFrame] = queue.Queue(maxsize=maxsize)
        self._sequence = 0
        self._dropped = 0
        self._lock = threading.Lock()

    def next_sequence(self) -> int:
        with self._lock:
            self._sequence += 1
            return self._sequence

    def publish(self, frame: ActivationFrame) -> None:
        while True:
            try:
                self._q.put_nowait(frame)
                return
            except queue.Full:
                try:
                    self._q.get_nowait()
                    with self._lock:
                        self._dropped += 1
                except queue.Empty:
                    pass

    def drain(self, limit: int = 8) -> list[ActivationFrame]:
        frames: list[ActivationFrame] = []
        for _ in range(limit):
            try:
                frames.append(self._q.get_nowait())
            except queue.Empty:
                break
        return frames

    def latest(self) -> ActivationFrame | None:
        frames = self.drain(limit=self._q.maxsize or 32)
        return frames[-1] if frames else None

    @property
    def dropped(self) -> int:
        with self._lock:
            return self._dropped

    def __len__(self) -> int:
        return self._q.qsize()


def build_frame(
    recorder: ActivationRecorder,
    modality: str,
    sequence: int,
    active_edges: list[tuple[str, str]] | None = None,
    active_nodes: list[str] | None = None,
    meta: dict[str, Any] | None = None,
) -> ActivationFrame:
    """Convert captured tensors into a UI-ready frame. Called on the inference
    thread -- conversion is cheap, rendering is not, and only rendering is
    banned from this side of the queue."""
    captured = recorder.snapshot()

    #  Convolutional taps stay spatial -- a (C,H,W) map genuinely has a layout,
    #  so averaging channels into an image is meaningful. Everything else is a
    #  flat layer, and is kept as an ordered vector for the layer diagrams.
    heatmaps: dict[str, np.ndarray] = {}
    vectors: dict[str, np.ndarray] = {}
    for name, tensor in captured.items():
        # "Spatial" means at least two non-singleton spatial axes. Counting raw
        # dimensions is not enough: occipital's grouped hidden layer is a
        # Conv1d, so it emits (B, C, 1) -- three dims describing a plain vector.
        # Filing that as a heatmap silently drops it out of the layer diagrams.
        if tensor.squeeze().dim() >= 3:
            heatmaps[name] = to_heatmap(tensor)
        else:
            vectors[name] = to_vector(tensor)

    return ActivationFrame(
        sequence=sequence,
        modality=modality,
        timestamp=time.time(),
        heatmaps=heatmaps,
        vectors=vectors,
        stats={name: heatmap_stats(t) for name, t in captured.items()},
        active_edges=list(active_edges or []),
        active_nodes=list(active_nodes or []),
        meta=dict(meta or {}),
    )


__all__ = [
    "ActivationRecorder",
    "ActivationBus",
    "ActivationFrame",
    "build_frame",
    "to_heatmap",
    "to_vector",
    "heatmap_stats",
]
