"""
output_generator.py -- turns frontal's gated decision into an answer.

PROJECT.md section 7.1: text only in v1.
PROJECT.md section 11 flagged two things this file has to be honest about:

  1. The classifier head becomes the bottleneck the moment you want language
     out. It is still a classifier here; `Answer.text` is a small templated
     verbalisation, not generation, and it says so.

  2. On a fresh system the head is randomly initialised, so its answer is
     noise. Rather than dress that up, `Answer` carries `trained=False` and
     the verbalisation states it outright. A separate `reference` field
     carries what the BORROWED encoder believes, clearly labelled, so the
     dashboard has something real to show on day one.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn as nn

import config


@dataclass
class Prediction:
    index: int
    label: str
    probability: float


@dataclass
class Answer:
    """Everything the UI needs to render one response."""

    predictions: list[Prediction]
    suppressed: bool = False
    gate: float = 1.0
    trained: bool = False
    # What the frozen encoder thinks, for contrast. Never the lobes' answer.
    reference: list[Prediction] = field(default_factory=list)

    @property
    def top(self) -> Prediction | None:
        return self.predictions[0] if self.predictions else None

    def verbalise(self) -> str:
        """Template, not generation. See the module docstring."""
        if self.suppressed:
            return (
                f"[basal ganglia suppressed the response -- gate {self.gate:.2f} "
                f"below threshold {config.BG_GATE_THRESHOLD:.2f}]"
            )
        if not self.predictions:
            return "[no prediction]"

        top = self.predictions[0]
        if not self.trained:
            lines = [
                f"lobes (UNTRAINED, this answer is noise): {top.label} "
                f"{top.probability:.1%}"
            ]
            if self.reference:
                ref = self.reference[0]
                lines.append(
                    f"frozen encoder reference: {ref.label} {ref.probability:.1%}"
                )
            lines.append("run  python train.py  to make the lobes' answer mean something")
            return "\n".join(lines)

        # An explicit abstain class is an answer, but "I identify this as
        # (no clear topic)" is not how you say it.
        if top.label.startswith("("):
            return ("Nothing to classify here -- the input carries no topic "
                    f"I can recognise ({top.probability:.0%} sure of that).")

        rest = ", ".join(
            f"{p.label} {p.probability:.1%}" for p in self.predictions[1:3]
        )
        line = f"I identify this as {top.label} ({top.probability:.1%})."
        return f"{line}  Alternatives: {rest}" if rest else line


class OutputGenerator(nn.Module):
    """LATENT_DIM -> an answer, in an answer space that suits the input.

    ONE HEAD IS NOT ENOUGH, and shipping only the ImageNet head was a real
    defect rather than a simplification. Every input, whatever its modality,
    was forced into 1000 visual categories -- so the text "hi" came back as
    "ocarina (16.7%)". The routing was correct and the lobes were fine; the
    system simply had no vocabulary in which a sentence could have an answer.

    So the generator carries several heads and the pipeline selects one from
    the modalities that actually arrived. The heads share every upstream
    module: the lobes, the fusion and the gate are common, and only the final
    read-out differs. That is also how cortex works -- one representation,
    several downstream read-outs.
    """

    #  head name -> (class count, what it answers)
    DOMAINS = {
        "imagenet": "what object is this",
        "topic": "what is this text about",
        "acoustic": "what kind of sound is this",
        "action": "which way should I move",
    }

    def __init__(
        self,
        dim: int = config.LATENT_DIM,
        num_classes: int = config.NUM_CLASSES,
        labels: list[str] | None = None,
    ):
        super().__init__()
        self.dim = dim
        self.num_classes = num_classes
        self.head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes),
        )
        self._labels = labels
        #  name -> head module, and name -> label list
        self.extra_heads = nn.ModuleDict()
        self._extra_labels: dict[str, list[str]] = {}

    # -- additional answer spaces ------------------------------------------
    def add_head(self, name: str, labels: list[str]) -> nn.Module:
        """Register (or replace) a read-out for one answer space."""
        head = nn.Sequential(nn.LayerNorm(self.dim), nn.Linear(self.dim, len(labels)))
        self.extra_heads[name] = head
        self._extra_labels[name] = list(labels)
        return head

    def has_head(self, name: str) -> bool:
        return name == "imagenet" or name in self.extra_heads

    def read(self, decision: torch.Tensor, domain: str = "imagenet") -> torch.Tensor:
        if domain == "imagenet" or domain not in self.extra_heads:
            return self.forward(decision)
        if decision.dim() == 1:
            decision = decision.unsqueeze(0)
        return self.extra_heads[domain](decision)

    def labels_for(self, domain: str) -> list[str] | None:
        if domain == "imagenet":
            return self._labels
        return self._extra_labels.get(domain)

    def top_k_in(self, logits: torch.Tensor, domain: str,
                 k: int = config.TOP_K) -> list["Prediction"]:
        names = self.labels_for(domain)
        if logits.dim() == 1:
            logits = logits.unsqueeze(0)
        probs = torch.softmax(logits[0], dim=-1)
        k = min(k, probs.numel())
        values, indices = torch.topk(probs, k)
        return [
            Prediction(int(i), names[int(i)] if names else f"class_{i}", float(v))
            for v, i in zip(values.tolist(), indices.tolist())
        ]

    # -- labels -------------------------------------------------------------
    def extra_label_state(self) -> dict[str, list[str]]:
        return {k: list(v) for k, v in self._extra_labels.items()}

    def restore_extra_labels(self, state: dict[str, list[str]]) -> None:
        for name, labels in state.items():
            if name not in self.extra_heads:
                self.add_head(name, labels)
            else:
                self._extra_labels[name] = list(labels)

    def set_labels(self, labels: list[str]) -> None:
        if len(labels) != self.num_classes:
            raise ValueError(
                f"got {len(labels)} labels but config.NUM_CLASSES={self.num_classes}"
            )
        self._labels = list(labels)

    def label_for(self, index: int) -> str:
        if self._labels is None:
            return f"class_{index}"
        return self._labels[index]

    # -- forward ------------------------------------------------------------
    def forward(self, decision: torch.Tensor) -> torch.Tensor:
        if decision.dim() == 1:
            decision = decision.unsqueeze(0)
        if decision.shape[-1] != self.dim:
            raise ValueError(
                f"output generator expected LATENT_DIM={self.dim}, "
                f"got {tuple(decision.shape)}"
            )
        return self.head(decision)

    # -- reading the result -------------------------------------------------
    def top_k(self, logits: torch.Tensor, k: int = config.TOP_K) -> list[Prediction]:
        if logits.dim() == 1:
            logits = logits.unsqueeze(0)
        probs = torch.softmax(logits[0], dim=-1)
        k = min(k, probs.numel())
        values, indices = torch.topk(probs, k)
        return [
            Prediction(int(i), self.label_for(int(i)), float(v))
            for v, i in zip(values.tolist(), indices.tolist())
        ]


__all__ = ["OutputGenerator", "Answer", "Prediction"]
