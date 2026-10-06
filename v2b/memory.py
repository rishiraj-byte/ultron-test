"""
memory.py -- hippocampus (retrieval-augmented memory) and amygdala (value).

HIPPOCAMPUS, as retrieval-augmented recognition (RAG), in the anatomy's order:

    POT -> ENTORHINAL cortex   a trained projection of what POT sent frontal
                               into a memory key space. Trained so that things
                               of the same kind land near each other -- this is
                               what makes recall generalise past the exact
                               picture it stored.
        -> DG / CA3            the store, searched by similarity (faiss):
                               every training example, plus examples it was
                               never trained on (knowledge by retrieval alone)
        -> CA1 / subiculum     the readout: the k nearest memories vote for
                               their labels, sharper for closer matches
                               (a trained temperature), and the vote reaches
                               frontal with a trained weight -- how much
                               frontal should trust memory over cortex.

    EPISODES -- what YOU taught it -- take a separate, direct path: a
    stored episode that is (nearly) this same input votes for its label with
    a strength that fades with similarity. It has to be separate: the trust
    CA1 learns for consolidated memory is small (memory of what cortex trained
    on adds little), and through it a correction could never win.

AMYGDALA, as learned value. It reads how sure cortex is, how sure memory is,
and whether the two agree, and outputs the probability that the answer about
to be given is GOOD (correct). Low value makes motor cortex say "not sure"
instead of asserting. Your GOOD/BAD tags ride on top: an answer you marked bad
for something that looked like this is pushed away.

The trained parts here are small nn.Modules (entorhinal, CA1, amygdala); the
store itself is data, not weights -- which is why teaching it a new thing
needs no training at all.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from v2b import config_v2b as C


# ---------------------------------------------------------------------------
# The trained parts
# ---------------------------------------------------------------------------


class Entorhinal(nn.Module):
    """POT's vector -> a unit-length memory key."""

    def __init__(self, embed: int):
        super().__init__()
        self.embed = embed
        self.project = nn.Linear(embed, C.KEY_DIM)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.project(x), dim=-1)


class CA1(nn.Module):
    """The readout: recalled memories -> votes for frontal."""

    def __init__(self):
        super().__init__()
        #  log-parameterised so both stay positive while training.
        self.log_sharpness = nn.Parameter(torch.tensor(math.log(C.MEMORY_SHARPNESS)))
        self.log_trust = nn.Parameter(torch.tensor(0.0))
        #  FAMILIARITY (CA1 as a match / mismatch detector). How close the
        #  nearest memory is, turned into a 0..1 gate on the vote. Without it
        #  memory voted as confidently for a church -- whose nearest stored
        #  pictures are CIFAR castles at similarity ~0.58 -- as for a familiar
        #  picture (~0.89), and dragged never-trained classes from 76.5% to
        #  67.8%. Set from the VALIDATION set's own similarities (train.py).
        self.register_buffer("familiar_at", torch.tensor(-10.0))
        self.register_buffer("familiar_width", torch.tensor(1.0))

    def familiarity(self, closest: torch.Tensor) -> torch.Tensor:
        """Nearest-memory similarity -> 1 familiar .. 0 novel."""
        return torch.sigmoid((closest - self.familiar_at) / self.familiar_width)

    @property
    def sharpness(self) -> torch.Tensor:
        return self.log_sharpness.exp()

    @property
    def trust(self) -> torch.Tensor:
        return self.log_trust.exp()

    def vote(self, similarity: torch.Tensor, labels: torch.Tensor,
             weights: torch.Tensor, vocab: int) -> torch.Tensor:
        """similarity, labels, weights: N,k -> memory log-probabilities N,V."""
        strength = torch.softmax(similarity * self.sharpness, dim=-1) * weights
        votes = torch.zeros(similarity.shape[0], vocab, device=similarity.device)
        votes.scatter_add_(1, labels, strength)
        votes = votes / votes.sum(-1, keepdim=True).clamp_min(1e-8)
        return torch.log(votes + C.MEMORY_FLOOR)


class Amygdala(nn.Module):
    """How likely the answer about to be given is GOOD. Learned."""

    FEATURES = ("cortex sure", "cortex margin", "cortex spread", "memory sure",
                "closest memory", "cortex = memory", "fused sure")

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(len(self.FEATURES), C.VALUE_HIDDEN),
                                 nn.GELU(), nn.Linear(C.VALUE_HIDDEN, 1))

    @staticmethod
    def features(cortex_logits, memory_logp, fused_logits, closest):
        p = cortex_logits.softmax(-1)
        top2 = p.topk(2, dim=-1).values
        spread = -(p * p.clamp_min(1e-9).log()).sum(-1) / math.log(p.shape[-1])
        m = memory_logp.exp()
        agree = (p.argmax(-1) == m.argmax(-1)).float()
        return torch.stack([top2[:, 0], top2[:, 0] - top2[:, 1], spread,
                            m.max(-1).values, closest, agree,
                            fused_logits.softmax(-1).max(-1).values], dim=-1)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.net(features)).squeeze(-1)


# ---------------------------------------------------------------------------
# The store
# ---------------------------------------------------------------------------


@dataclass
class Episode:
    id: int
    label: str
    kind: str            # "image" or "sound"
    valence: float       # +1 good, -1 bad, 0 not rated
    source: str
    when: float


class Hippocampus:
    """Consolidated memories (what it was trained on) + episodes (yours)."""

    def __init__(self, entorhinal: Entorhinal, directory=C.MEMORY_DIR):
        self.entorhinal = entorhinal
        self.directory = directory
        self.embed = entorhinal.embed
        # consolidated: POT vectors of every training example
        self.bank_vectors = np.zeros((0, self.embed), np.float32)
        self.bank_labels = np.zeros(0, np.int64)
        self.bank_sources: list[str] = []
        self.bank_trained = np.zeros(0, bool)
        self.known: set = set()
        #  The taught-episode bar; brain.py sets the trained model's value.
        self.similar_at = C.EPISODE_SIMILARITY
        # episodic: what you taught it
        self.kept: list[Episode] = []
        self.kept_vectors = np.zeros((0, self.embed), np.float32)
        self.scratch: dict[int, tuple[Episode, np.ndarray]] = {}
        self._next = 1
        self._keys = None
        self._index = None
        self._label_of: dict[str, int] = {}
        self.load_episodes()

    # -- loading -------------------------------------------------------------
    def load_bank(self, path=C.MEMORY_BANK) -> None:
        if not path.exists():
            return
        saved = np.load(path, allow_pickle=False)
        if saved["vectors"].shape[1:] != (self.embed,):
            return
        self.bank_vectors = saved["vectors"].astype(np.float32)
        self.bank_labels = saved["labels"].astype(np.int64)
        self.bank_sources = [str(s) for s in saved["sources"]]
        #  Which memories cortex also trained on, and which arrived by storage
        #  alone -- shown in the dashboard's database view.
        self.bank_trained = (saved["trained"].astype(bool) if "trained" in saved.files
                             else np.ones(len(self.bank_labels), bool))
        self._where = None
        #  The names the store holds memories of: an answer among them is
        #  familiar, and is read with expertise (brain.perceive).
        self.known = set(np.unique(self.bank_labels).tolist())
        self.reindex()

    def index_of(self, source: str) -> int:
        """A stored memory's position, from its source id (each id is unique)."""
        if getattr(self, "_where", None) is None:
            self._where = {s: i for i, s in enumerate(self.bank_sources)}
        return self._where.get(source, -1)

    def load_episodes(self) -> None:
        meta = self.directory / "episodes.json"
        vectors = self.directory / "episodes.npy"
        if meta.exists() and vectors.exists():
            kept = [Episode(**row) for row in json.loads(meta.read_text())]
            saved = np.load(vectors).astype(np.float32)
            if saved.shape[1:] != (self.embed,):
                #  Taught under a different vision model: those vectors live
                #  in another space and would recall nonsense. Set aside.
                return
            self.kept, self.kept_vectors = kept, saved
            self._next = max([e.id for e in self.kept] + [0]) + 1

    def save_episodes(self) -> None:
        self.directory.mkdir(parents=True, exist_ok=True)
        (self.directory / "episodes.json").write_text(
            json.dumps([asdict(e) for e in self.kept], indent=1))
        np.save(self.directory / "episodes.npy", self.kept_vectors)

    @property
    def size(self) -> tuple[int, int]:
        return len(self.bank_labels), len(self.kept)

    # -- keys and the index -------------------------------------------------
    @torch.no_grad()
    def keys(self, vectors: np.ndarray) -> np.ndarray:
        device = next(self.entorhinal.parameters()).device
        out = []
        for start in range(0, len(vectors), 4096):
            chunk = torch.from_numpy(vectors[start:start + 4096]).to(device)
            out.append(self.entorhinal(chunk).cpu().numpy())
        return (np.concatenate(out) if out
                else np.zeros((0, C.KEY_DIM), np.float32)).astype(np.float32)

    def reindex(self) -> None:
        """Rebuild the searchable store after the bank changes."""
        self._keys = self.keys(self.bank_vectors)
        try:
            import faiss
            index = faiss.IndexFlatIP(C.KEY_DIM)
            if len(self._keys):
                index.add(self._keys)
            self._index = index
        except Exception:                                     # noqa: BLE001
            self._index = None

    # -- recall --------------------------------------------------------------
    def recall(self, vectors: np.ndarray, names: list[str], k: int = C.MEMORY_K):
        """Consolidated memory -> list per query of [(similarity, label, 1, ("bank", source))]."""
        if self._keys is None or not len(self._keys):
            return [[] for _ in range(len(vectors))]
        queries = self.keys(np.atleast_2d(vectors).astype(np.float32))
        k = min(k, len(self._keys))
        if self._index is not None:
            similarity, index = self._index.search(queries, k)
        else:
            scores = queries @ self._keys.T
            index = np.argsort(-scores, axis=1)[:, :k]
            similarity = np.take_along_axis(scores, index, axis=1)
        return [[(float(s), int(self.bank_labels[i]), 1.0,
                  ("bank", self.bank_sources[i] if self.bank_sources else ""))
                 for s, i in zip(row_s, row_i) if i >= 0]
                for row_s, row_i in zip(similarity, index)]

    def activation(self, vector: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """This percept's entorhinal key, and its cosine with EVERY stored key.

        The store read as one layer of units whose weights are the stored
        keys -- what the dashboard's RAG view draws. recall() searches these
        same keys exactly (faiss IndexFlatIP), so its k nearest are this
        vector's k largest.
        """
        if self._keys is None or not len(self._keys):
            return np.zeros(C.KEY_DIM, np.float32), np.zeros(0, np.float32)
        key = self.keys(np.atleast_2d(np.asarray(vector, np.float32)))[0]
        return key, self._keys @ key

    def episode_scores(self, vector: np.ndarray) -> np.ndarray:
        """Cosine between this input and every episode you taught."""
        if not self.kept:
            return np.zeros(0, np.float32)
        v = np.asarray(vector, np.float32).reshape(-1)
        v = v / max(float(np.linalg.norm(v)), 1e-8)
        return self.kept_vectors @ v

    def episodes_like(self, vector: np.ndarray) -> list:
        """Your rated episodes that are (nearly) this same input: [(Episode, cosine)].

        Raw cosine between POT vectors, not entorhinal keys: the keys were
        trained to pull a whole CLASS together, and an episode must match the
        thing you taught, not everything of its kind.
        """
        scores = self.episode_scores(vector)
        return [(e, float(s)) for e, s in zip(self.kept, scores) if s >= self.similar_at]

    def strength(self, similarity: float) -> float:
        """1 for the same thing again, fading to 0 at the episode bar."""
        floor = self.similar_at
        return max(similarity - floor, 0.0) / (1.0 - floor)

    # -- episodes ------------------------------------------------------------
    def witness(self, vector: np.ndarray, label: str, kind: str, source: str) -> int:
        """Hold a new, unrated episode briefly. Unrated episodes fade."""
        episode = Episode(self._next, label, kind, 0.0, source, time.time())
        self._next += 1
        self.scratch[episode.id] = (episode, _unit(vector))
        while len(self.scratch) > C.SCRATCH_EPISODES:
            self.scratch.pop(next(iter(self.scratch)))
        return episode.id

    def consolidate(self, episode_id: int, valence: float,
                    label: str | None = None) -> Episode:
        if episode_id not in self.scratch:
            raise KeyError("episode %d has faded or was already kept" % episode_id)
        episode, vector = self.scratch.pop(episode_id)
        episode.valence = float(valence)
        if label:
            episode.label = label
        self._keep(episode, vector)
        return episode

    def remember(self, vector, label: str, kind: str, valence: float,
                 source: str) -> Episode:
        episode = Episode(self._next, label, kind, float(valence), source, time.time())
        self._next += 1
        self._keep(episode, _unit(vector))
        return episode

    def _keep(self, episode: Episode, vector: np.ndarray) -> None:
        self.kept.append(episode)
        self.kept_vectors = np.vstack([self.kept_vectors, vector[None]])
        self.save_episodes()

    def forget_episodes(self) -> None:
        self.kept, self.scratch = [], {}
        self.kept_vectors = np.zeros((0, self.embed), np.float32)
        self.save_episodes()


def _unit(vector) -> np.ndarray:
    vector = np.asarray(vector, np.float32).reshape(-1)
    return vector / max(float(np.linalg.norm(vector)), 1e-8)
