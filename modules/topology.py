"""
topology.py -- the real layer structure of every network, read off the model.

Why this exists: a 16x16 heatmap of a 256-vector is an arbitrary square
reshape. Neuron 0 and neuron 16 end up adjacent for no reason other than the
grid width. It shows you *that* something changed, never *what the network
is*.

This module reads the actual `nn.Module` tree and reports, for each network,
the ordered chain of layers with their true widths -- input layer, hidden
layers, output layer -- so the dashboard can draw the network the way a neural
network is actually drawn, and light up real per-neuron activations on it.

Nothing here is hardcoded. Widths are read from `in_features` / `out_features`
/ `normalized_shape` on the live modules, so if you change LATENT_DIM or the
lobe width, the diagram follows automatically.

Layer chain per network, as built:

    lobe          in 256 -> norm 256 -> linear 512 -> GELU 512 -> linear 256
                  -> out 256 (residual)
    frontal       streams 768 -> fuse 256 -> norm 256 -> then as a lobe
    relay nucleus in 256 -> relay 256 / gate 256 -> out 256
    basal ganglia in 256 -> striatum 256 -> GELU 256 -> go 1 / nogo 1
    output head   in 256 -> logits 1000
"""

from __future__ import annotations

from dataclasses import dataclass

import torch.nn as nn

import config

#  Hook keys use "::in" to mean "the tensor that went INTO this module",
#  as opposed to the plain name which means what came out.
IN_SUFFIX = "::in"


@dataclass
class LayerSpec:
    """One column of the network diagram."""

    key: str            # activation-frame key holding this layer's values
    label: str          # what to print under the column
    size: int           # true neuron count -- never rounded for display
    kind: str           # input | linear | gelu | norm | sigmoid | output | scalar
    primary: bool = False   # part of the compact input/hidden/output view

    @property
    def is_activation(self) -> bool:
        """Layers where a unit can genuinely be OFF, not just small.

        GELU drives negative pre-activations to ~0, and sigmoid gates saturate
        shut. Everything else is signed and centred, so "deactivated" is not a
        meaningful word for it.
        """
        return self.kind in ("gelu", "sigmoid")


def _width(module: nn.Module, fallback: int) -> int:
    if isinstance(module, nn.Linear):
        return module.out_features
    if isinstance(module, nn.Conv1d):
        return module.out_channels          # occipital's grouped bands
    if isinstance(module, nn.LayerNorm):
        return int(module.normalized_shape[0])
    return fallback


def lobe_layers(prefix: str, lobe: nn.Module, is_frontal: bool = False) -> list[LayerSpec]:
    """The chain through one cortical lobe."""
    dim = lobe.dim
    layers: list[LayerSpec] = []

    if is_frontal:
        # Frontal's input is the concatenated stream block, not a single
        # LATENT_DIM vector -- that is the whole point of PROJECT.md section 4.
        fuse = lobe.fuse
        n_in = fuse.in_features if isinstance(fuse, nn.Linear) else dim * len(lobe.slots)
        layers.append(
            LayerSpec(f"{prefix}.fuse{IN_SUFFIX}", "streams in", n_in, "input", True)
        )
        if isinstance(fuse, nn.Linear):
            layers.append(
                LayerSpec(f"{prefix}.fuse", "fused", fuse.out_features, "linear", True)
            )
        layers.append(
            LayerSpec(f"{prefix}.fusion_norm", "fusion norm",
                      _width(lobe.fusion_norm, dim), "norm")
        )
    else:
        layers.append(LayerSpec(f"{prefix}{IN_SUFFIX}", "input", dim, "input", True))

    layers.append(LayerSpec(f"{prefix}.norm", "norm", _width(lobe.norm, dim), "norm"))

    # The lobe-specific operator, which is the whole point of the architecture.
    # Each lobe exposes a different one, so the diagram shows what actually
    # distinguishes them rather than three identical chains.
    if hasattr(lobe, "attn"):                       # parietal: relational
        layers.append(
            LayerSpec(f"{prefix}.attn", f"attend {lobe.positions}x{lobe.slot}",
                      dim, "linear", True)
        )
    elif hasattr(lobe, "rnn"):                      # temporal: sequential
        layers.append(
            LayerSpec(f"{prefix}.rnn", f"GRU {lobe.steps} steps",
                      dim, "linear", True)
        )

    core = getattr(lobe, "core", None)
    if core is not None:
        # The operator is a FROZEN PRETRAINED NETWORK, so the diagram shows
        # what it actually is -- the teacher's name and width -- rather than a
        # generic hidden layer. `lobe.net` does not exist in this mode, and
        # reaching for it is what used to crash this function.
        teacher = getattr(core, "teacher_name", type(core.teacher).__name__)
        #  Names must resolve against the real module tree, and widths must be
        #  what those modules actually EMIT. Declaring the teacher's internal
        #  width for the wrapper was wrong twice over: `core` returns cortical
        #  width, and `core.in` / `core.out` are not modules at all -- they are
        #  called `into` and `back`.
        if isinstance(core.into, nn.Linear):
            #  The projection's OWN output width. A convolutional stage takes
            #  a different width in than it gives out -- ResNet-50's layer4
            #  takes 1024 and returns 2048 -- so using teacher_dim here
            #  declared a column that never matched what was captured.
            layers.append(
                LayerSpec(f"{prefix}.core.into", "into %s" % teacher,
                          core.into.out_features, "linear", True)
            )
        layers.append(
            LayerSpec(f"{prefix}.core.back", "%s (frozen)" % teacher,
                      dim, "gelu", True)
        )
        layers.append(
            LayerSpec(f"{prefix}.core", "back to cortex", dim, "linear")
        )
    else:
        hidden_kind = "linear"
        hidden_label = "hidden (pre)"
        if isinstance(lobe.net[0], nn.Conv1d):      # occipital: grouped/local
            hidden_label = f"hidden x{lobe.net[0].groups} local"
        layers.append(
            LayerSpec(f"{prefix}.net.0", hidden_label, _width(lobe.net[0], dim),
                      hidden_kind)
        )
        layers.append(
            LayerSpec(f"{prefix}.net.1", "hidden GELU", _width(lobe.net[0], dim),
                      "gelu", True)
        )
        layers.append(
            LayerSpec(f"{prefix}.net.3", "project", _width(lobe.net[3], dim),
                      "linear")
        )
    if hasattr(lobe, "mix"):                        # occipital: cross-band mix
        layers.append(
            LayerSpec(f"{prefix}.mix", "mix bands", _width(lobe.mix, dim), "linear")
        )
    layers.append(LayerSpec(prefix, "output (+res)", dim, "output", True))
    return layers


def nucleus_layers(prefix: str, nucleus: nn.Module) -> list[LayerSpec]:
    """One thalamic relay nucleus. The gate column is where signal is held back."""
    dim = _width(nucleus.relay, config.LATENT_DIM)
    return [
        LayerSpec(f"{prefix}.relay{IN_SUFFIX}", "input", nucleus.relay.in_features,
                  "input", True),
        LayerSpec(f"{prefix}.relay", "relay", dim, "linear", True),
        LayerSpec(f"{prefix}.gate", "gate (sigmoid)", _width(nucleus.gate, dim),
                  "sigmoid", True),
        LayerSpec(f"{prefix}.norm", "output", _width(nucleus.norm, dim), "output", True),
    ]


def basal_layers(prefix: str, module: nn.Module) -> list[LayerSpec]:
    striatum = module.striatum
    dim = module.dim
    return [
        LayerSpec(f"{prefix}.striatum{IN_SUFFIX}", "input", dim, "input", True),
        LayerSpec(f"{prefix}.striatum.1", "striatum", _width(striatum[1], dim),
                  "linear"),
        LayerSpec(f"{prefix}.striatum.2", "striatum GELU", _width(striatum[1], dim),
                  "gelu", True),
        LayerSpec(f"{prefix}.direct", "GO (direct)", _width(module.direct, 1),
                  "scalar", True),
        LayerSpec(f"{prefix}.indirect", "NoGo (indirect)", _width(module.indirect, 1),
                  "scalar", True),
    ]


def output_layers(prefix: str, module: nn.Module) -> list[LayerSpec]:
    head = module.head
    return [
        LayerSpec(f"{prefix}.head{IN_SUFFIX}", "input", module.dim, "input", True),
        LayerSpec(f"{prefix}.head.0", "norm", _width(head[0], module.dim), "norm"),
        LayerSpec(f"{prefix}.head.1", f"logits ({module.num_classes})",
                  _width(head[1], module.num_classes), "output", True),
    ]


def build_topology(brain) -> dict[str, list[LayerSpec]]:
    """Every network in the system -> its ordered layer chain.

    Keys match the dashboard's node keys, so a node click maps straight to a
    diagram.
    """
    topology: dict[str, list[LayerSpec]] = {}

    for name in config.RELAY_NUCLEI:
        topology[f"thalamus.{name}"] = nucleus_layers(
            f"thalamus.{name}", brain.thalamus.nuclei[name]
        )

    for name in config.LOBES:
        topology[name] = lobe_layers(
            f"lobe.{name}", brain.cortex[name], is_frontal=(name == "frontal")
        )

    topology["basal_ganglia"] = basal_layers("basal_ganglia", brain.basal_ganglia)
    topology["output"] = output_layers("output", brain.output)
    return topology


def hook_targets(brain) -> list[tuple[str, nn.Module, bool]]:
    """(hook name, module, capture_input) for every layer in the topology.

    `capture_input` marks the modules whose INPUT tensor is also needed -- that
    is what gives the diagram a real input column rather than starting at the
    first weight matrix.
    """
    targets: list[tuple[str, nn.Module, bool]] = []
    seen: set[str] = set()

    def add(name: str, module: nn.Module, want_input: bool) -> None:
        if name in seen:
            return
        seen.add(name)
        targets.append((name, module, want_input))

    def resolve(root: nn.Module, dotted: str) -> nn.Module | None:
        node = root
        for part in dotted.split("."):
            if part.isdigit():
                node = node[int(part)]
            else:
                node = getattr(node, part, None)
            if node is None:
                return None
        return node

    roots = {
        **{f"thalamus.{n}": brain.thalamus.nuclei[n] for n in config.RELAY_NUCLEI},
        **{f"lobe.{n}": brain.cortex[n] for n in config.LOBES},
        "basal_ganglia": brain.basal_ganglia,
        "output": brain.output,
    }

    for layers in build_topology(brain).values():
        for spec in layers:
            key = spec.key
            want_input = key.endswith(IN_SUFFIX)
            base = key[: -len(IN_SUFFIX)] if want_input else key

            root_name = max(
                (r for r in roots if base == r or base.startswith(r + ".")),
                key=len,
                default=None,
            )
            if root_name is None:
                continue
            remainder = base[len(root_name):].lstrip(".")
            module = roots[root_name] if not remainder else resolve(
                roots[root_name], remainder
            )
            if module is not None:
                add(base, module, want_input)

    return targets


__all__ = [
    "LayerSpec",
    "build_topology",
    "hook_targets",
    "lobe_layers",
    "nucleus_layers",
    "basal_layers",
    "output_layers",
    "IN_SUFFIX",
]
