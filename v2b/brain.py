"""
brain.py -- v2.B assembled. `BrainV2B.perceive()` is the entry point.

  senses -> thalamus -> occipital -> IT          (ventral: what)
                     -> occipital + parietal     (dorsal: where)
                     -> temporal (auditory cortex)
         -> STS (sight meets sound) + dorsal -> POT <-> frontal, 3 passes
         -> hippocampus: recall similar memories (RAG), CA1 votes
         -> amygdala: how likely is this answer good? + your good/bad tags
         -> motor: the answer, or "not sure"

THE ONLY WAY OUT IS THROUGH FRONTAL. Motor sees only frontal's final scores;
frontal sees only POT, memory and the amygdala. Cut POT->frontal and there is
no answer. (v2's frontal contributed nothing because its readout bypassed it.)
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field

import numpy as np
import torch
import torch.nn.functional as F

from v2b import config_v2b as C
from v2b import data, senses
from v2b.cortex import Cortex, Dims, Features
from v2b.lobes import Auditory, Frontal, Parietal, VentralVision, vision_language
from v2b.memory import CA1, Amygdala, Entorhinal, Hippocampus
from v2b.wiring import Wiring


@dataclass
class Thought:
    """One percept, with everything the dashboard needs to replay it."""

    answer: str
    confidence: float
    value: float                    # amygdala: P(this answer is good)
    abstained: bool                 # value too low -> "not sure"
    top: list                       # [(name, p)] final, fused
    cortex_top: list                # [(name, p)] cortex alone
    memory_top: list                # [(name, p)] memory alone
    per_pass: list                  # per pass: [(name, p)] top 3 (cortex)
    gains: list                     # per pass: relay -> gain
    modalities: list
    recalled: list                  # [(similarity, name, weight, what)]
    tags: list                      # [(episode, similarity)] your taught memories that applied
    feeling: float                  # -1 bad .. +1 good, as shown
    episode: int
    kind: str
    elapsed_ms: float
    trust: float                    # CA1's learned weight on memory
    source: str = ""
    retina: np.ndarray | None = None
    depth: np.ndarray | None = None
    wave: np.ndarray | None = None
    attention: list = field(default_factory=list)   # per pass: 1+49+67
    carried: list = field(default_factory=list)     # per pass: connection -> value
    live: dict = field(default_factory=dict)        # lobe -> Live
    layers: dict = field(default_factory=dict)      # node -> [(label, vector)]
    pass_layers: list = field(default_factory=list) # per pass: node -> columns
    amygdala_inputs: dict = field(default_factory=dict)
    decision: str = "GO"            # amygdala -> motor: GO or HOLD, with `value` as confidence
    familiarity: float = 0.0        # CA1: 1 = seen things like this, 0 = novel
    rag: dict = field(default_factory=dict)         # every state and gate, POT -> motor
    expertise: bool = False         # re-read with the expert correction (familiar answer)

    def summary(self) -> str:
        if not self.top:
            return "no answer -- the path to motor cortex is cut"
        head = ("HOLD, not sure -- maybe %s (%.0f%%)" % (self.answer, 100 * self.confidence)
                if self.abstained else "GO: %s (%.0f%%)" % (self.answer, 100 * self.confidence))
        others = ", ".join("%s %.0f%%" % (n, 100 * p) for n, p in self.top[1:3])
        memory = ("memory says %s" % self.memory_top[0][0]) if self.memory_top else "no memory"
        return "%s. also: %s. %s; amygdala confidence %.2f." % (
            head, others or "-", memory, self.value)


class BrainV2B:
    def __init__(self, device: str | None = None, checkpoint=C.CHECKPOINT,
                 memory_dir=C.MEMORY_DIR, bank=C.MEMORY_BANK):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        model, processor, family = vision_language()
        self.vision = VentralVision(model, processor, family).to(self.device)
        self.frontal = Frontal(model, family).to(self.device)
        self.parietal = Parietal().to(self.device)
        #  The temporal lobe's auditory cortex. (The key "auditory" in models()
        #  is kept: checkpoints record it, and renaming it would orphan them.)
        self.temporal = Auditory().to(self.device)
        self.dims = Dims(vision=self.vision.width,
                         embed=self.frontal.encode(["probe"]).shape[-1],
                         parietal=self.parietal.cell, audio=self.temporal.width,
                         audio_stats=self.temporal.stats_width,
                         sound_embed=self.temporal.joint_width)
        self.wiring = Wiring()
        self.cortex = Cortex(self.dims, self.frontal.scale, self.wiring,
                             sound_scale=self.temporal.scale).to(self.device).eval()
        self.entorhinal = Entorhinal(self.dims.embed).to(self.device).eval()
        self.ca1 = CA1().to(self.device).eval()
        self.amygdala = Amygdala().to(self.device).eval()
        self.record: dict = {}
        self.keep_expert = None         # the stronger keep for familiar answers, if trained
        self.episode_bar = C.EPISODE_SIMILARITY
        self.mismatch = ""

        names, kinds = data.vocabulary()
        if checkpoint is not None and checkpoint.exists():
            saved = torch.load(checkpoint, map_location=self.device, weights_only=False)
            if saved.get("models") == self.models() and saved.get("dims") == asdict(self.dims):
                self.cortex.load_state_dict(saved["cortex"])
                self.entorhinal.load_state_dict(saved["entorhinal"])
                self.ca1.load_state_dict(saved["ca1"])
                self.amygdala.load_state_dict(saved["amygdala"])
                names, kinds = saved["names"], saved["kinds"]
                self.record = saved.get("record", {})
                self.keep_expert = saved.get("keep_expert")
                self.episode_bar = saved.get("episode_similarity", C.EPISODE_SIMILARITY)
            else:
                #  Trained for different lobes: loading it would mean reading
                #  one network's output with weights learned on another's.
                self.mismatch = ("checkpoint was trained with %s; this config loads %s"
                                 % (saved.get("models"), self.models()))
        self.frontal.set_vocabulary(names, kinds)
        self.hippocampus = Hippocampus(self.entorhinal, memory_dir)
        self.hippocampus.similar_at = self.episode_bar
        if self.record and bank is not None:
            self.hippocampus.load_bank(bank)
        for episode in self.hippocampus.kept:
            self.frontal.learn_word(episode.label, episode.kind)
        self._build_lexicon()
        self.hippocampus.reindex()

    def _build_lexicon(self) -> None:
        """The temporal lobe's sound vocabulary, matching frontal's names row for row."""
        self.cortex.lexicon = self.temporal.lexicon(self.frontal.names, self.frontal.kinds)

    @staticmethod
    def models() -> dict:
        return {"vision": C.VISION_MODEL, "parietal": C.PARIETAL_MODEL,
                "auditory": C.AUDITORY_MODEL}

    @property
    def trained(self) -> bool:
        return bool(self.record)

    def lesion(self, name: str, cut: bool = True) -> None:
        self.wiring.lesion(name, cut)

    # -- the sensory lobes ----------------------------------------------------
    @torch.no_grad()
    def see(self, pixels: torch.Tensor, record: bool = False) -> dict:
        vis = self.vision(pixels, record=record)
        par = self.parietal(pixels, record=record)
        out = {"occ_grid": vis.occ_grid, "it_grid": vis.it_grid, "identity": vis.identity,
               "par_cells": par.cells,
               "image_stats": senses.image_stats(pixels.float()).to(self.device)}
        if record:
            out.update(depth=par.depth, live={"occipital": vis.occipital,
                                              "ventral": vis.inferotemporal,
                                              "parietal": par.live})
        return out

    @torch.no_grad()
    def hear(self, waves: list[np.ndarray], record: bool = False) -> dict:
        aud = self.temporal(waves, record=record)
        joint = (aud.joint if aud.joint is not None
                 else torch.zeros(len(waves), self.dims.sound_embed, device=self.device))
        out = {"aud_bins": aud.bins, "aud_identity": aud.identity, "aud_joint": joint,
               "audio_stats": aud.stats.to(self.device)}
        if record:
            out["live"] = {"temporal": aud.live}
        return out

    def features(self, frames: list | None, wave, record: bool = False):
        """One percept's lobe outputs. Video frames are averaged: no motion."""
        d, dims, g = self.device, self.dims, C.GRID * C.GRID
        extra: dict = {"live": {}}
        if frames:
            pixels = senses.retina(frames)
            seen = self.see(pixels, record=record)
            parts = {k: seen[k].mean(0, keepdim=True) for k in
                     ("occ_grid", "it_grid", "identity", "par_cells", "image_stats")}
            if record:
                middle = len(frames) // 2
                extra["retina"] = (pixels[middle].permute(1, 2, 0).numpy() * 255).astype(np.uint8)
                extra["depth"] = seen["depth"][middle].cpu().numpy()
                extra["live"].update(seen["live"])
        else:
            parts = {"occ_grid": torch.zeros(1, g, dims.vision, device=d),
                     "it_grid": torch.zeros(1, g, dims.vision, device=d),
                     "identity": torch.zeros(1, dims.embed, device=d),
                     "par_cells": torch.zeros(1, g, dims.parietal, device=d),
                     "image_stats": torch.zeros(1, C.IMAGE_STATS, device=d)}
        if wave is not None and np.size(wave):
            heard = self.hear([wave], record=record)
            parts.update({k: heard[k] for k in ("aud_bins", "aud_identity", "aud_joint",
                                                 "audio_stats")})
            if record:
                extra["live"].update(heard["live"])
        else:
            parts.update(aud_bins=torch.zeros(1, C.TEMPORAL_BINS, dims.audio, device=d),
                         aud_identity=torch.zeros(1, dims.audio, device=d),
                         aud_joint=torch.zeros(1, dims.sound_embed, device=d),
                         audio_stats=torch.zeros(1, dims.audio_stats, device=d))
        f = Features(parts["occ_grid"], parts["it_grid"], parts["identity"],
                     parts["par_cells"], parts["aud_bins"], parts["aud_identity"],
                     parts["aud_joint"], parts["image_stats"], parts["audio_stats"],
                     torch.tensor([bool(frames)], device=d),
                     torch.tensor([wave is not None and bool(np.size(wave))], device=d))
        return f, extra

    # -- memory and value, shared with training ----------------------------------
    def remember_and_value(self, cortex_logits, embedding, recalled_batch):
        """cortex logits + recalled memories -> (memory logp, fused, value).

        The same function scores training batches and live percepts, so what
        the dashboard shows is exactly what was measured.
        """
        vocab = cortex_logits.shape[-1]
        k = max(1, max((len(r) for r in recalled_batch), default=1))
        sim = torch.zeros(len(recalled_batch), k, device=cortex_logits.device)
        lab = torch.zeros(len(recalled_batch), k, dtype=torch.long, device=cortex_logits.device)
        wt = torch.zeros(len(recalled_batch), k, device=cortex_logits.device)
        for row, found in enumerate(recalled_batch):
            for col, (s, label, weight, _) in enumerate(found[:k]):
                sim[row, col], lab[row, col], wt[row, col] = s, label, weight
        if not any(recalled_batch):
            memory = torch.full_like(cortex_logits, float(np.log(1.0 / vocab)))
        else:
            memory = self.ca1.vote(sim, lab, wt, vocab)
        w = self.wiring
        use = float(w.open("POT->hippocampus") and w.open("hippocampus->frontal"))
        closest = sim[:, 0] if any(recalled_batch) else torch.zeros(len(recalled_batch),
                                                                     device=sim.device)
        #  Memory only votes as strongly as the input is familiar.
        familiar = self.ca1.familiarity(closest) if any(recalled_batch) else torch.zeros_like(closest)
        self.last_familiarity = familiar
        fused = cortex_logits + use * self.ca1.trust * familiar[:, None] * memory
        #  The amygdala hears memory's opinion straight from the hippocampus
        #  (CA1/subiculum -> basolateral amygdala). Cut that and it judges on
        #  cortex alone: memory looks uniform and nothing looks familiar.
        if not (w.open("POT->hippocampus") and w.open("hippocampus->amygdala")):
            memory_seen = torch.full_like(cortex_logits, float(np.log(1.0 / vocab)))
            closest = torch.zeros_like(closest)
        else:
            memory_seen = memory
        features = Amygdala.features(cortex_logits, memory_seen, fused, closest)
        return memory, fused, self.amygdala(features), features

    # -- one percept --------------------------------------------------------------
    @torch.no_grad()
    def perceive(self, image=None, audio=None, video=None, source: str = "") -> Thought:
        """image: PIL image or path. audio: waveform or path. video: path."""
        start = time.perf_counter()
        frames, wave, modalities = None, None, []
        if video is not None:
            frames, wave = senses.read_video(video)
            modalities.append("video")
        if image is not None:
            frames = [senses.load_image(image) if isinstance(image, (str, bytes))
                      or hasattr(image, "__fspath__") else image]
            modalities.append("image")
        if audio is not None:
            wave = (senses.read_audio(audio) if isinstance(audio, (str, bytes))
                    or hasattr(audio, "__fspath__") else np.asarray(audio, np.float32))
            modalities.append("audio")
        if not frames and wave is None:
            raise ValueError("nothing to perceive: give an image, audio or video")

        f, extra = self.features(frames, wave, record=True)
        words = self.frontal.words.to(self.device)
        out = self.cortex(f, words, record=True)
        final = out.passes[-1]
        names, w = self.frontal.names, self.wiring

        recalled = []
        if w.open("POT->hippocampus"):
            recalled = self.hippocampus.recall(final.embedding.cpu().numpy(), names)[0]
        memory, fused, value, seen = self.remember_and_value(final.logits, final.embedding, [recalled])

        #  EXPERTISE (hippocampus->POT): an answer the store holds memories of
        #  is re-read with the stronger correction chosen on validation. The
        #  memory term still comes from the cautious embedding -- the space the
        #  store was built in. Exactly train.py's rule.
        expert = False
        keep_lo = float(self.cortex.pot.keep)
        if (self.keep_expert is not None and self.keep_expert > keep_lo and recalled
                and w.open("hippocampus->POT")
                and int(fused[0].argmax()) in self.hippocampus.known):
            with torch.no_grad():
                self.cortex.pot.keep.fill_(self.keep_expert)
            try:
                out = self.cortex(f, words, record=True)
            finally:
                with torch.no_grad():
                    self.cortex.pot.keep.fill_(keep_lo)
            expert = True
            memory, fused, value, seen = self.remember_and_value(
                out.passes[-1].logits, final.embedding, [recalled])
        answering = out.passes[-1]

        # EPISODIC memory: what you taught it, for (nearly) this same input.
        # Good episodes pull toward their label (hippocampus); bad ones push
        # away from theirs (amygdala). Both fade with similarity.
        tags = self.hippocampus.episodes_like(final.embedding[0].cpu().numpy())
        hippocampal = w.open("POT->hippocampus") and w.open("hippocampus->frontal")
        amygdalar = w.open("POT->amygdala") and w.open("amygdala->frontal")
        for episode, similarity in tags:
            if episode.label not in names:
                continue
            pull = self.hippocampus.strength(similarity)
            #  The episode's own sense: "train", "hen", "harmonica"... are both
            #  a picture name and a sound name, and names.index() finds the
            #  picture first -- a taught sound would have voted for a picture.
            kinds = self.frontal.kinds
            same = [i for i, n in enumerate(names) if n == episode.label and kinds[i] == episode.kind]
            slot = same[0] if same else names.index(episode.label)
            if episode.valence > 0 and hippocampal:
                fused[0, slot] += C.EPISODE_VOTE * pull
            elif episode.valence < 0 and amygdalar:
                fused[0, slot] -= C.AMYGDALA_PENALTY * pull
        amygdala_on = w.open("POT->amygdala")
        value_now = float(value[0]) if amygdala_on else 1.0
        #  Having been told what this very thing is, is grounds for confidence:
        #  without this the amygdala, reading only cortex and consolidated
        #  memory, said "not sure" about the answer you had just taught it.
        chosen = names[int(fused[0].argmax())]
        taught = [self.hippocampus.strength(s) for e, s in tags
                  if e.valence > 0 and e.label == chosen]
        raised = False
        if taught and w.open("POT->hippocampus") and w.open("hippocampus->amygdala"):
            raised = max(taught) > value_now
            value_now = max(value_now, max(taught))
        if tags and amygdala_on:
            mood = sum(e.valence * s for e, s in tags) / sum(s for _, s in tags)
            feeling = float(np.clip(0.5 * (2 * value_now - 1) + 0.5 * mood, -1, 1))
        else:
            feeling = 2 * value_now - 1

        def top(logits, k=5):
            p = logits[0].softmax(-1)
            best = p.topk(min(k, p.numel()))
            return [(names[i], float(v)) for v, i in zip(best.values.tolist(), best.indices.tolist())]

        answer_top = top(fused) if (w.open("frontal->motor") and w.open("POT->frontal")) else []
        #  THE AMYGDALA'S VERDICT gates motor: GO (say it) or HOLD (say "not
        #  sure"), with its value as the confidence. Cut amygdala->motor and
        #  motor always goes.
        decision = "GO" if (value_now >= C.ABSTAIN_BELOW or not amygdala_on) else "HOLD"
        drive = max(float(g[0]) for g in out.passes[-1].gains.values())
        if drive < C.MIN_DRIVE:                   # nothing got through the thalamus
            decision, value_now, feeling = "HOLD", 0.0, -1.0
        if not w.open("amygdala->motor"):
            decision = "GO"
        abstained = bool(answer_top) and decision == "HOLD"
        kind = "image" if frames else "sound"
        answer = answer_top[0][0] if answer_top else ""
        said = decision if answer_top else "-"
        rag = self.trace_memory(answering.logits, final.embedding, memory, fused, seen, recalled,
                                value_now, said, answer,
                                {"amygdala_on": amygdala_on, "taught_raised": raised,
                                 "low_drive": drive < C.MIN_DRIVE})
        rag["expertise"] = {"used": expert, "keep": self.keep_expert, "cautious_keep": keep_lo}
        episode = self.hippocampus.witness(final.embedding[0].cpu().numpy(), answer, kind, source)
        thought = Thought(
            answer=answer, confidence=answer_top[0][1] if answer_top else 0.0,
            value=value_now, abstained=abstained, top=answer_top,
            cortex_top=top(answering.logits), memory_top=top(memory) if recalled else [],
            per_pass=[top(p.logits, 3) for p in out.passes],
            gains=[{k: float(v[0]) for k, v in p.gains.items()} for p in out.passes],
            modalities=modalities,
            recalled=[(s, names[l] if l < len(names) else "?", wt, what)
                      for s, l, wt, what in recalled],
            tags=tags, feeling=feeling, episode=episode, kind=kind,
            elapsed_ms=(time.perf_counter() - start) * 1000,
            trust=float(self.ca1.trust.detach()), source=source,
            familiarity=float(self.last_familiarity[0]) if recalled else 0.0,
            retina=extra.get("retina"), depth=extra.get("depth"), wave=wave,
            attention=[p.attention[0].cpu().numpy() for p in out.passes],
            live=extra["live"],
            amygdala_inputs=dict(zip(Amygdala.FEATURES, seen[0].tolist())),
            decision=said, rag=rag, expertise=expert)
        self._record_views(thought, out, f)
        return thought

    #  The connections the RAG view draws, in the order the signal takes them.
    MEMORY_PATHS = ("POT->hippocampus", "hippocampus->frontal", "hippocampus->amygdala",
                    "POT->amygdala", "amygdala->frontal", "amygdala->motor",
                    "POT->frontal", "frontal->motor", "hippocampus->POT")

    @torch.no_grad()
    def trace_memory(self, cortex_logits, embedding, memory, fused, features, recalled,
                     value_now: float, decision: str, answer: str, causes: dict) -> dict:
        """Every state and gate between POT and motor for one percept -- what
        the dashboard's RAG view draws. Nothing is estimated: each number is
        read back from the tensors perceive() used, or recomputed by the same
        modules, and `causes` are the branches perceive() actually took (a
        review found the first version inferring them, and blaming a taught
        episode whenever POT->amygdala was cut)."""
        hippo, ca1, names = self.hippocampus, self.ca1, self.frontal.names
        gates = {name: bool(self.wiring.open(name)) for name in self.MEMORY_PATHS}
        query = embedding[0].float().cpu().numpy()
        familiar = float(self.last_familiarity[0]) if recalled else 0.0
        use = gates["POT->hippocampus"] and gates["hippocampus->frontal"]
        trace = {"query": query, "bank": len(hippo.bank_labels), "gates": gates, "use": use,
                 "k": C.MEMORY_K, "sharpness": float(ca1.sharpness), "trust": float(ca1.trust),
                 "familiar_at": float(ca1.familiar_at),
                 "familiar_width": float(ca1.familiar_width), "familiarity": familiar,
                 "closest": float(recalled[0][0]) if recalled else 0.0}

        # ENTORHINAL -> the store, read as one layer of units.
        key, similarity = (hippo.activation(query) if gates["POT->hippocampus"]
                           else (np.zeros(C.KEY_DIM, np.float32), np.zeros(0, np.float32)))
        trace["key"], trace["similarity"] = key, similarity.astype(np.float32)

        # CA1: the k recalled memories and the share of the vote each carried
        # (softmax of similarity x sharpness, exactly as CA1.vote computes it).
        sims = np.array([s for s, *_ in recalled], np.float32)
        strength = np.exp((sims - sims.max()) * trace["sharpness"]) if len(sims) else sims
        strength = strength / strength.sum() if len(sims) else strength
        trace["recalled"] = [
            {"index": hippo.index_of(what[1]), "similarity": float(s), "strength": float(g),
             "label": names[label] if label < len(names) else "?", "source": what[1]}
            for (s, label, _, what), g in zip(recalled, strength)]
        votes: dict = {}
        for row in trace["recalled"]:
            votes[row["label"]] = votes.get(row["label"], 0.0) + row["strength"]
        trace["votes"] = sorted(votes.items(), key=lambda kv: -kv[1])[:8]

        # The sum frontal makes, name by name: cortex + memory term (+ episodes).
        # Rows: frontal's leaders, cortex's, and what memory actually voted
        # for -- not topk of a flat memory, which is arbitrary ties.
        vocab = fused.shape[-1]
        candidates = set(fused[0].topk(min(8, vocab)).indices.tolist())
        candidates.update(cortex_logits[0].topk(min(5, vocab)).indices.tolist())
        if recalled:
            voted = (memory[0] > float(np.log(C.MEMORY_FLOOR)) + 1e-3).nonzero().flatten()
            candidates.update(voted[memory[0, voted].argsort(descending=True)[:5]].tolist())
        ledger = []
        for i in candidates:
            c, m, final = float(cortex_logits[0, i]), float(memory[0, i]), float(fused[0, i])
            term = float(use) * trace["trust"] * familiar * m
            ledger.append({"name": names[i], "cortex": c, "memory_logp": m, "memory": term,
                           "episodes": final - c - term, "final": final})
        trace["ledger"] = sorted(ledger, key=lambda r: -r["final"])

        # AMYGDALA: its 7 inputs, its hidden layer, its own value, the verdict.
        net = self.amygdala.net
        hidden = net[1](net[0](features))
        value_net = float(torch.sigmoid(net[2](hidden))[0])
        reasons = []
        if decision == "-":
            reasons += ["%s cut: no answer" % n for n in ("POT->frontal", "frontal->motor")
                        if not gates[n]]
        if causes["low_drive"]:
            reasons.append("no thalamic drive: value 0")
        elif not causes["amygdala_on"]:
            reasons.append("POT->amygdala cut: value fixed at 1, amygdala not used")
        elif causes["taught_raised"]:
            reasons.append("a taught episode matched: value raised")
        if not gates["amygdala->motor"] and decision != "-":
            reasons.append("amygdala->motor cut: motor always goes")
        trace["amygdala"] = {"inputs": dict(zip(Amygdala.FEATURES, features[0].tolist())),
                             "hidden": hidden[0].float().cpu().numpy(), "value_net": value_net,
                             "value": value_now, "threshold": C.ABSTAIN_BELOW,
                             "decision": decision, "reasons": reasons, "answer": answer,
                             "used": causes["amygdala_on"],
                             "hears_memory": gates["POT->hippocampus"]
                             and gates["hippocampus->amygdala"]}

        # EPISODES you taught: their direct path, every one, judged by the
        # same rules perceive() applied (near enough, a known name, its path open).
        hippocampal = gates["POT->hippocampus"] and gates["hippocampus->frontal"]
        amygdalar = gates["POT->amygdala"] and gates["amygdala->frontal"]
        trace["episodes"] = []
        for e, s in zip(hippo.kept, hippo.episode_scores(query)):
            route = hippocampal if e.valence > 0 else amygdalar if e.valence < 0 else False
            state = ("too far" if float(s) < hippo.similar_at
                     else "unknown name" if e.label not in names
                     else "unrated" if e.valence == 0
                     else "applies" if route else "path cut")
            trace["episodes"].append({"label": e.label, "kind": e.kind, "valence": e.valence,
                                      "cosine": float(s), "strength": hippo.strength(float(s)),
                                      "state": state, "applies": state == "applies"})
        trace["episode_threshold"] = hippo.similar_at
        return trace

    def _record_views(self, thought: Thought, out, f: Features) -> None:
        """The columns the network diagrams draw, and what each arrow carried."""
        def v(x):
            if x is None:
                return None
            return np.asarray(x.detach().float().cpu().numpy() if torch.is_tensor(x) else x,
                              np.float32).reshape(-1)

        sees, hears = bool(f.has_image[0]), bool(f.has_audio[0])
        live = thought.live

        def lobe(name, first):
            L = live.get(name)
            if L is None:
                return [(first, None)] + [("block %d" % i, None) for i in range(1, 4)]
            cols = [(first, v(L.blocks[0]))]
            step = max(1, len(L.neurons) // 5)
            for i in range(0, len(L.neurons), step):
                cols.append(("block %d" % (i + 1), v(L.neurons[i])))
            return cols

        act0 = out.activity[0]
        thought.layers = {
            "pulvinar": [("image stats", v(f.image_stats[0]) if sees else None),
                         ("hidden", v(act0["relay.hidden"]["pulvinar"][0]) if sees else None),
                         ("gain", v([thought.gains[-1]["pulvinar"]]))],
            "LGN": [("image stats", v(f.image_stats[0]) if sees else None),
                    ("hidden", v(act0["relay.hidden"]["LGN"][0]) if sees else None),
                    ("gain", v([thought.gains[-1]["LGN"]]))],
            "MGN": [("sound stats", v(f.audio_stats[0]) if hears else None),
                    ("hidden", v(act0["relay.hidden"]["MGN"][0]) if hears else None),
                    ("gain", v([thought.gains[-1]["MGN"]]))],
            "occipital": lobe("occipital", "V1 input") + [("V4 grid", v(f.occ_grid[0].norm(dim=-1)) if sees else None)],
            "ventral": lobe("ventral", "from V4") + [("identity", v(f.identity[0]) if sees else None)],
            "parietal": lobe("parietal", "input") + [("depth 7x7", v(f.par_cells[0, :, -C.DEPTH_STATS]) if sees else None)],
            "temporal": lobe("temporal", "spectrogram") + [("identity", v(f.aud_identity[0]) if hears else None)],
            "hippocampus": [("recalled", v([r[0] for r in thought.recalled]) if thought.recalled else None),
                            ("memory vote", v([p for _, p in thought.memory_top]) if thought.memory_top else None)],
            "amygdala": [("inputs", v(list(thought.amygdala_inputs.values()))),
                         ("value", v([thought.value]))],
        }
        thought.pass_layers, thought.carried = [], []
        for index, (p, act, carried) in enumerate(zip(out.passes, out.activity, out.carried)):
            probs = p.logits[0].softmax(-1)
            thought.pass_layers.append({
                "dorsal": [("V4+depth in", v(torch.cat([f.occ_grid[0].norm(dim=-1),
                                                        f.par_cells[0].norm(dim=-1)])) if sees else None),
                           ("hidden", v(act["dorsal.hidden"][0]) if sees else None),
                           ("where tokens", v(act["dorsal.tokens"][0]) if sees else None)],
                "STS": [("sight", v(f.identity[0]) if sees else None),
                        ("sound", v(f.aud_bins[0].norm(dim=-1)) if hears else None),
                        ("what tokens", v(act["STS.tokens"][0]))],
                "POT": [("frontal query", v(p.query[0])), ("attention", v(p.attention[0])),
                        ("fused", v(act["POT.fused"][0])), ("to frontal", v(p.embedding[0]))],
                "frontal": [("from POT", v(p.embedding[0])), ("name scores", v(probs)),
                            ("top %d held" % C.FRONTAL_TOP_K, v(probs.topk(C.FRONTAL_TOP_K).values))],
                "thalamus": [("TRN feedback", v(act["thalamus.feedback"][0]))],
            })
            row = {k: float(t[0]) for k, t in carried.items()}
            last = index == len(out.passes) - 1
            row["POT->hippocampus"] = float(last and bool(thought.recalled))
            row["hippocampus->frontal"] = float(last and bool(thought.recalled)
                                                and self.wiring.open("hippocampus->frontal"))
            row["POT->amygdala"] = float(last and self.wiring.open("POT->amygdala"))
            row["amygdala->frontal"] = float(last and self.wiring.open("amygdala->frontal"))
            row["hippocampus->amygdala"] = float(last and bool(thought.recalled)
                                                 and self.wiring.open("hippocampus->amygdala"))
            row["amygdala->motor"] = float(last and bool(thought.top)
                                           and self.wiring.open("amygdala->motor"))
            row["frontal->motor"] = float(last and bool(thought.top))
            row["hippocampus->POT"] = float(last and thought.expertise)
            thought.carried.append(row)

    # -- your tags --------------------------------------------------------------------
    def rate(self, thought: Thought, good: bool) -> None:
        self.hippocampus.consolidate(thought.episode, 1.0 if good else -1.0)

    def correct(self, thought: Thought, label: str) -> None:
        """"No, it's a <label>." Tags the wrong answer bad, keeps the right one.
        A new word is learned on the spot -- frontal's text network can place
        any phrase in its space, so there is nothing to train."""
        label = label.strip()
        if not label:
            return
        known = label in self.frontal.names
        self.frontal.learn_word(label, thought.kind)
        if not known:
            self._build_lexicon()             # a new sound word joins the lexicon too
        if thought.episode in self.hippocampus.scratch:
            vector = self.hippocampus.scratch[thought.episode][1].copy()
            if thought.answer and thought.answer != label:
                self.hippocampus.consolidate(thought.episode, -1.0)
            else:
                self.hippocampus.scratch.pop(thought.episode)
            self.hippocampus.remember(vector, label, thought.kind, 1.0, thought.source)

    # -- the networks, for the weights view -------------------------------------------
    def networks(self) -> dict:
        """Region -> [(prefix, module)] -- every weight in v2.B, by where it sits."""
        v = self.vision.vision
        occ = [("embeddings", v.embeddings)] + [
            ("block%02d" % (i + 1), b) for i, b in enumerate(self.vision.occipital_blocks)]
        it = [("block%02d" % (i + 1 + self.vision.split), b)
              for i, b in enumerate(self.vision.it_blocks)]
        it += [(n, getattr(v, n)) for n in ("post_layernorm", "head") if hasattr(v, n)]
        if self.vision.project is not None:
            it.append(("projection", self.vision.project))
        frontal = [("text", self.frontal.text)]
        if self.frontal.project is not None:
            frontal.append(("projection", self.frontal.project))
        th = self.cortex.thalamus
        return {"pulvinar": [("relay", th.pulvinar)], "LGN": [("relay", th.LGN)],
                "MGN": [("relay", th.MGN)], "thalamus": [("trn", th.trn)],
                "occipital": occ, "ventral": it, "parietal": [("depth", self.parietal.net)],
                "temporal": [("sound", self.temporal.net)],
                "dorsal": [("dorsal", self.cortex.dorsal)], "STS": [("sts", self.cortex.sts)],
                "POT": [("pot", self.cortex.pot)], "frontal": frontal,
                "hippocampus": [("entorhinal", self.entorhinal), ("ca1", self.ca1)],
                "amygdala": [("value", self.amygdala)]}

    def save(self, path=C.CHECKPOINT, record: dict | None = None) -> None:
        base = len(data.vocabulary()[0])
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"models": self.models(), "dims": asdict(self.dims),
                    "cortex": self.cortex.state_dict(),
                    "entorhinal": self.entorhinal.state_dict(), "ca1": self.ca1.state_dict(),
                    "amygdala": self.amygdala.state_dict(),
                    "names": self.frontal.names[:base], "kinds": self.frontal.kinds[:base],
                    "record": record or self.record}, path)
