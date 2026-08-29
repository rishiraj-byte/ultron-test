"""
pipeline.py -- end-to-end orchestration. The signal path of PROJECT.md 2.1.

    image / audio / text
             |
        [ THALAMUS ]              modality-keyed relay, not a fixed pipeline
         /         \\
   image           audio, text
       |                |
  [ OCCIPITAL ]    [ TEMPORAL ]
       |    \\           |
  [ PARIETAL ] \\        |
        \\       \\      |
         +--------+-----+
                  |
             [ FRONTAL ]          convergence, fusion, decision
                  |
        [ BASAL GANGLIA ]         gate: allow or suppress
                  |
        [ OUTPUT GENERATOR ]

Occipital delivers to frontal DIRECTLY as well as via parietal. That is the
two-stream case the frontal fusion fix exists for (PROJECT.md section 4).

Every stage appends to a BrainTrace, so verify.py can print shapes at every
stage and the dashboard can light up exactly the edges that carried signal.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn as nn

import config
from modules.activations import ActivationBus, ActivationRecorder, build_frame
from modules.basal_ganglia import BasalGanglia, GateDecision
from modules.encoders import EncoderBank
from modules.lobes import Cortex
from modules.output_generator import Answer, OutputGenerator, Prediction
from modules.thalamus import Routing, Thalamus
from modules.topology import build_topology, hook_targets


@dataclass
class Stage:
    """One step of the signal path, recorded for inspection."""

    name: str
    shape: tuple[int, ...]
    note: str = ""

    def __str__(self) -> str:
        shape = "x".join(str(d) for d in self.shape) if self.shape else "-"
        line = f"{self.name:<34} {shape:>12}"
        return f"{line}   {self.note}" if self.note else line


@dataclass
class BrainTrace:
    """Everything one pass through the brain produced."""

    modalities: list[str] = field(default_factory=list)
    stages: list[Stage] = field(default_factory=list)
    routings: list[Routing] = field(default_factory=list)
    streams: dict[str, torch.Tensor] = field(default_factory=dict)
    decision: torch.Tensor | None = None
    gate: GateDecision | None = None
    logits: torch.Tensor | None = None
    answer: Answer | None = None
    active_nodes: list[str] = field(default_factory=list)
    active_edges: list[tuple[str, str]] = field(default_factory=list)
    #  which read-out answered, e.g. "imagenet" / "topic" / "action"
    domain: str = "imagenet"
    elapsed_ms: float = 0.0

    def add(self, name: str, tensor: torch.Tensor | None, note: str = "") -> None:
        shape = tuple(tensor.shape) if tensor is not None else ()
        self.stages.append(Stage(name, shape, note))

    def report(self) -> str:
        return "\n".join(str(s) for s in self.stages)


class BrainAI(nn.Module):
    """The whole system. One object, four lobes, one thalamus, one gate."""

    def __init__(self, device: str | None = None, dropout: float = 0.0):
        super().__init__()
        self.device_name = device or config.resolve_device()

        self.encoders = EncoderBank(device=self.device_name)
        self.thalamus = Thalamus()
        self.cortex = Cortex(dropout=dropout)
        self.basal_ganglia = BasalGanglia()
        self.output = OutputGenerator()

        # True once a checkpoint is loaded. Drives the honesty in Answer.
        self.trained = False

        self.bus = ActivationBus()
        self._recorder: ActivationRecorder | None = None
        #  network key -> ordered layer chain, read off the live modules
        self.topology = build_topology(self)

        self.to(self.device_name)
        # nn.Module.to() only moves registered submodules. The frozen backbones
        # are loaded lazily and held as plain attributes, so they need moving
        # explicitly or they stay on the CPU while the lobes sit on the GPU.
        self.encoders.to_device(self.device_name)

    # ------------------------------------------------------------------
    # Labels
    # ------------------------------------------------------------------
    def load_labels(self) -> None:
        """Inherit ImageNet-1k class names from the frozen vision encoder."""
        if config.LABEL_SET != "imagenet1k":
            return
        self.output.set_labels(self.encoders.image.categories)

    # ------------------------------------------------------------------
    # Activation capture (step 4)
    # ------------------------------------------------------------------
    def attach_recorder(self, include_vision_taps: bool = True) -> ActivationRecorder:
        """Hook EVERY layer of every network, not just each module's output.

        The layer list comes from `topology.hook_targets`, which reads the
        actual nn.Module tree -- so the dashboard can draw each network as a
        real input / hidden / output diagram with per-neuron activations,
        rather than a square reshape of the final vector.

        Module inputs are captured too where the topology asks for them; that
        is what gives each diagram a genuine input column.
        """
        if self._recorder is not None:
            self._recorder.remove()

        recorder = ActivationRecorder()
        watched: list[tuple] = list(hook_targets(self))

        if include_vision_taps:
            # These stay spatial: a conv feature map genuinely has a layout, so
            # it is drawn as an image, not as a column of neurons.
            backbone = self.encoders.image.backbone
            for tap in config.VISION_TAPS:
                module = getattr(backbone, tap, None)
                if module is not None:
                    watched.append((f"vision.{tap}", module))

        recorder.watch_many(watched)
        self._recorder = recorder
        self.topology = build_topology(self)
        return recorder

    def detach_recorder(self) -> None:
        if self._recorder is not None:
            self._recorder.remove()
            self._recorder = None


    # ------------------------------------------------------------------
    def answer_domain(self, modalities) -> str:
        """Pick the read-out that suits the input that actually arrived.

        Forcing everything through the ImageNet head is what made the text
        "hi" come back as "ocarina": the routing was right, but a sentence has
        no meaningful answer in 1000 object categories. The upstream modules
        are shared; only this final read-out changes.
        """
        present = set(modalities)
        if "sensor" in present:
            domain = "action"          # embodied navigation
        elif "image" in present:
            domain = "imagenet"        # anything with a picture in it
        elif "audio" in present:
            domain = "acoustic"
        elif "text" in present:
            domain = "topic"
        else:
            domain = "imagenet"
        return domain if self.output.has_head(domain) else "imagenet"

    # ------------------------------------------------------------------
    # The signal path
    # ------------------------------------------------------------------
    def perceive(
        self,
        inputs: dict[str, Any] | tuple[str, Any],
        publish: bool = False,
    ) -> BrainTrace:
        """Run one pass. `inputs` maps modality -> raw input.

        More than one modality is allowed and is the normal case once
        decompose.py splits a video into frames + audio, or finds text inside
        an image. Streams that land on the same lobe are averaged at the relay.
        """
        if isinstance(inputs, tuple):
            inputs = {inputs[0]: inputs[1]}
        if not inputs:
            raise ValueError("perceive() received no inputs")

        started = time.perf_counter()
        trace = BrainTrace(modalities=list(inputs))
        #  Frozen 512-d vision features, kept for the reference readout so the
        #  backbone is never run twice in one pass.
        vision_features: torch.Tensor | None = None
        if self._recorder is not None:
            self._recorder.clear()

        # -- 1. encode -------------------------------------------------
        relayed_to: dict[str, list[torch.Tensor]] = {}
        for modality, raw in inputs.items():
            if modality == "image":
                # Run the frozen backbone ONCE and keep its 512-d features.
                # The reference readout needs the same features, and calling
                # reference_logits() separately re-ran the whole of ResNet-18 --
                # which was doubling the cost of every image pass.
                vision_features = self.encoders.image.encode_raw(raw)
                embedding = self.encoders.image.projection(vision_features)
            else:
                embedding = self.encoders.encode(modality, raw)
            embedding = embedding.to(self.device_name)
            trace.add(
                f"encoder.{modality}",
                embedding,
                f"frozen {self._encoder_name(modality)} -> projection",
            )

            # -- 2. thalamic relay -------------------------------------
            routing = self.thalamus(modality, embedding)
            trace.routings.append(routing)
            trace.add(
                f"thalamus.{routing.nucleus}",
                routing.payload[routing.destination],
                f"{modality} -> {routing.destination}  gate={routing.gate_mean:.3f}",
            )
            relayed_to.setdefault(routing.destination, []).append(
                routing.payload[routing.destination]
            )
            trace.active_nodes += ["thalamus", routing.destination]
            trace.active_edges.append(("thalamus", routing.destination))

        # Two modalities on one lobe (audio + text both land on temporal):
        # average at the relay rather than silently keeping only the last.
        merged = {
            lobe: (tensors[0] if len(tensors) == 1
                   else torch.stack(tensors, dim=0).mean(dim=0))
            for lobe, tensors in relayed_to.items()
        }
        for lobe, tensors in relayed_to.items():
            if len(tensors) > 1:
                trace.add(
                    f"relay.merge.{lobe}", merged[lobe],
                    f"{len(tensors)} modalities averaged onto {lobe}",
                )

        # -- 3. cortex -------------------------------------------------
        #
        # A lobe's input is whatever reached it: thalamic relays, cortical
        # predecessors, or BOTH. Parietal is the case that forces this to be
        # general -- it takes occipital's dorsal stream and, when an embodied
        # sensor is present, VPL's somatosensory relay as well. Sources are
        # averaged so that adding a modality does not change the input scale
        # the lobe was trained at.
        streams: dict[str, torch.Tensor] = {}

        def drive(lobe: str, extra: torch.Tensor | None = None) -> torch.Tensor | None:
            sources = []
            if lobe in merged:
                sources.append(merged[lobe])
            if extra is not None:
                sources.append(extra)
            if not sources:
                return None
            return sources[0] if len(sources) == 1 else torch.stack(sources).mean(0)

        occipital_in = drive("occipital")
        if occipital_in is not None:
            occipital = self.cortex.occipital(occipital_in)
            streams["occipital"] = occipital
            trace.add("lobe.occipital", occipital, "local feature extraction")
            trace.active_nodes.append("occipital")
            trace.active_edges += [
                ("occipital", "parietal"),
                ("occipital", "frontal"),   # the DIRECT delivery -- section 2.1
            ]

        # Dorsal "where" stream from occipital, plus VPL if a body is attached.
        parietal_in = drive("parietal", extra=streams.get("occipital"))
        if parietal_in is not None:
            parietal = self.cortex.parietal(parietal_in)
            streams["parietal"] = parietal
            sources = [s for s in ("occipital", "sensor")
                       if (s == "occipital" and "occipital" in streams)
                       or (s == "sensor" and "parietal" in merged)]
            trace.add("lobe.parietal", parietal,
                      f"spatial relations (from {' + '.join(sources)})")
            trace.active_nodes.append("parietal")
            trace.active_edges.append(("parietal", "frontal"))

        temporal_in = drive("temporal")
        if temporal_in is not None:
            temporal = self.cortex.temporal(temporal_in)
            streams["temporal"] = temporal
            trace.add("lobe.temporal", temporal, "meaning, language, memory")
            trace.active_nodes.append("temporal")
            trace.active_edges.append(("temporal", "frontal"))

        for lobe in merged:
            if lobe not in ("occipital", "parietal", "temporal"):
                raise RuntimeError(
                    f"thalamus routed to {lobe!r}, which the cortex stage does not "
                    f"handle; check config.THALAMIC_ROUTES"
                )

        trace.streams = streams

        # -- 4. frontal convergence (the section 4 fix) ----------------
        filled = [s for s in config.FRONTAL_STREAM_SLOTS if s in streams]
        padded = [s for s in config.FRONTAL_STREAM_SLOTS if s not in streams]
        decision = self.cortex.frontal(streams)
        trace.decision = decision
        trace.active_nodes.append("frontal")
        trace.add(
            "lobe.frontal",
            decision,
            f"fused {len(filled)} stream(s) {filled}"
            + (f", zero-padded {padded}" if padded else ""),
        )

        # -- 5. basal ganglia gate -------------------------------------
        gate = self.basal_ganglia(decision)
        trace.gate = gate
        trace.active_nodes.append("basal_ganglia")
        trace.active_edges.append(("frontal", "basal_ganglia"))
        trace.add("basal_ganglia", gate.gated, gate.describe())

        # -- 6. output -------------------------------------------------
        domain = self.answer_domain(inputs)
        logits = self.output.read(gate.gated, domain)
        trace.logits = logits
        trace.domain = domain
        trace.active_nodes.append("output")
        trace.active_edges.append(("basal_ganglia", "output"))
        trace.add("output.logits", logits,
                  f"{domain} head -- {self.output.DOMAINS.get(domain, '')}")

        predictions = self.output.top_k_in(logits, domain)
        reference = self._reference(inputs, vision_features)

        trace.answer = Answer(
            predictions=predictions,
            suppressed=not gate.allowed,
            gate=gate.gate,
            trained=self.trained,
            reference=reference,
        )

        trace.elapsed_ms = (time.perf_counter() - started) * 1000.0
        trace.active_nodes = list(dict.fromkeys(trace.active_nodes))
        trace.active_edges = list(dict.fromkeys(trace.active_edges))

        if publish and self._recorder is not None:
            self.bus.publish(
                build_frame(
                    self._recorder,
                    modality="+".join(trace.modalities),
                    sequence=self.bus.next_sequence(),
                    active_edges=trace.active_edges,
                    active_nodes=trace.active_nodes,
                    meta={
                        "gate": gate.gate,
                        "allowed": gate.allowed,
                        "answer": trace.answer.verbalise(),
                        "elapsed_ms": trace.elapsed_ms,
                        "routings": [
                            (r.modality, r.nucleus, r.destination, r.gate_mean)
                            for r in trace.routings
                        ],
                        "top": [(p.label, p.probability) for p in predictions],
                        "reference": [(p.label, p.probability) for p in reference],
                    },
                )
            )

        return trace

    # ------------------------------------------------------------------
    def _reference(
        self, inputs: dict[str, Any], features: torch.Tensor | None = None
    ) -> list[Prediction]:
        """What the frozen encoder believes. Shown for contrast, never as the
        lobes' answer. Only vision has a usable pretrained head.

        `features` are the backbone outputs already computed this pass. Passing
        them in is what keeps this from re-running ResNet-18.
        """
        if "image" not in inputs:
            return []
        try:
            if features is not None:
                with torch.no_grad():
                    logits = self.encoders.image.reference_head(features)
            else:
                logits = self.encoders.image.reference_logits(inputs["image"])
        except Exception:
            return []
        probs = torch.softmax(logits[0], dim=-1)
        values, indices = torch.topk(probs, min(config.TOP_K, probs.numel()))
        categories = self.encoders.image.categories
        return [
            Prediction(int(i), categories[int(i)], float(v))
            for v, i in zip(values.tolist(), indices.tolist())
        ]

    @staticmethod
    def _encoder_name(modality: str) -> str:
        return {
            "image": config.VISION_MODEL,
            "text": config.TEXT_MODEL.split("/")[-1],
            "audio": config.AUDIO_MODEL.split("/")[-1],
        }.get(modality, modality)

    # ------------------------------------------------------------------
    # Persistence -- only the parts that are actually yours
    # ------------------------------------------------------------------
    def architecture_signature(self) -> str:
        """A fingerprint of the structure a checkpoint was trained against.

        Shapes alone are not enough: the lobes now carry different operators
        (grouped / attention / recurrent), and swapping one for another keeps
        every tensor the same size while changing what the weights mean. The
        signature covers the routing table and the lobe classes too, so a
        stale checkpoint is refused with a readable message instead of a
        state_dict error listing forty missing keys.
        """
        parts = [
            f"latent={config.LATENT_DIM}",
            f"hidden={config.LOBE_HIDDEN_DIM}",
            f"classes={config.NUM_CLASSES}",
            f"fusion={config.FRONTAL_FUSION_MODE}",
            f"slots={'+'.join(config.FRONTAL_STREAM_SLOTS)}",
            "routes=" + ",".join(
                f"{m}>{n}>{d}" for m, (n, d) in sorted(config.THALAMIC_ROUTES.items())
            ),
            "lobes=" + ",".join(
                f"{name}:{type(self.cortex[name]).__name__}" for name in config.LOBES
            ),
        ]
        return "|".join(parts)

    def trainable_state(self) -> dict[str, Any]:
        return {
            "signature": self.architecture_signature(),
            "latent_dim": config.LATENT_DIM,
            "num_classes": config.NUM_CLASSES,
            "label_set": config.LABEL_SET,
            "fusion_mode": config.FRONTAL_FUSION_MODE,
            "thalamus": self.thalamus.state_dict(),
            "cortex": self.cortex.state_dict(),
            "basal_ganglia": self.basal_ganglia.state_dict(),
            "output": self.output.state_dict(),
            "output_labels": self.output.extra_label_state(),
            "projections": {
                m: self.encoders[m].projection.state_dict() for m in config.MODALITIES
            },
        }

    def save(self, path=None) -> None:
        path = path or config.DEFAULT_CHECKPOINT
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.trainable_state(), path)

    def load(self, path=None, strict: bool = True) -> bool:
        path = path or config.DEFAULT_CHECKPOINT
        if not path.exists():
            return False
        state = torch.load(path, map_location=self.device_name, weights_only=False)

        want = self.architecture_signature()
        have = state.get("signature")
        if have != want:
            raise ValueError(
                "checkpoint does not match the current architecture, so its "
                "weights would mean something different.\n"
                f"  checkpoint : {have}\n"
                f"  this build : {want}\n"
                "Retrain with:  python train.py --images <folder>\n"
                f"or delete {path} to start from an untrained system."
            )

        self.thalamus.load_state_dict(state["thalamus"], strict=strict)
        self.cortex.load_state_dict(state["cortex"], strict=strict)
        self.basal_ganglia.load_state_dict(state["basal_ganglia"], strict=strict)
        # Recreate the extra read-outs before loading, or their tensors have
        # nowhere to land.
        self.output.restore_extra_labels(state.get("output_labels", {}))
        self.output.load_state_dict(state["output"], strict=strict)
        for modality, proj in state.get("projections", {}).items():
            self.encoders[modality].projection.load_state_dict(proj, strict=strict)

        # The extra read-out heads are CONSTRUCTED here, by
        # restore_extra_labels, which is after __init__ already moved
        # everything to the device. They are therefore born on the CPU, and
        # without this the first text or audio query dies on a device
        # mismatch inside their LayerNorm. Anything a load() creates has to be
        # moved by the load().
        self.to(self.device_name)
        self.encoders.to_device(self.device_name)

        self.trained = True
        return True

    def trainable_parameters(self):
        """Everything that is yours. The backbones are frozen and excluded."""
        yield from self.thalamus.parameters()
        yield from self.cortex.parameters()
        yield from self.basal_ganglia.parameters()
        yield from self.output.parameters()
        yield from self.encoders.trainable_parameters()

    def parameter_count(self) -> dict[str, int]:
        def count(module: nn.Module) -> int:
            return sum(p.numel() for p in module.parameters() if p.requires_grad)

        return {
            "thalamus": count(self.thalamus),
            "occipital": count(self.cortex.occipital),
            "parietal": count(self.cortex.parietal),
            "temporal": count(self.cortex.temporal),
            "frontal": count(self.cortex.frontal),
            "basal_ganglia": count(self.basal_ganglia),
            "output": count(self.output),
            "projections": sum(
                count(self.encoders[m].projection) for m in config.MODALITIES
            ),
        }


def build_brain(
    load_checkpoint: bool = True,
    device: str | None = None,
    dropout: float = 0.0,
) -> BrainAI:
    """The one constructor everything else should call.

    `dropout` is for training only; inference builds leave it at zero.
    """
    torch.manual_seed(config.SEED)
    brain = BrainAI(device=device, dropout=dropout)
    brain.load_labels()
    if load_checkpoint:
        brain.load()
    brain.eval()
    return brain


__all__ = ["BrainAI", "BrainTrace", "Stage", "build_brain"]
