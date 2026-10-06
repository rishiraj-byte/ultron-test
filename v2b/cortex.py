"""
cortex.py -- the small parts that learn: thalamic relays and their feedback,
the dorsal stream, the STS, and the POT junction in its loop with frontal.

Everything here trains (~2-3M weights). Everything in lobes.py is frozen.

THE CIRCUIT, one pass:

  thalamus   pulvinar, LGN, MGN each set a gain from cheap statistics of the
             raw signal. Routing is fixed wiring -- eyes never reach the MGN.
             From frontal, via the thalamic reticular nucleus (TRN), comes a
             second factor: frontal can turn a sense up or down for the next
             pass. That corticothalamic loop is real; in the brain there are
             more fibres running cortex -> thalamus than the other way.
  dorsal     occipital's V4-level grid + parietal's depth, cell by cell:
             what is WHERE, and how far.
  STS        superior temporal sulcus: IT's object identity + IT's grid +
             auditory cortex's sound over time, mixed by one attention layer.
             This is where sight and sound meet in the temporal lobe.
  POT        attends over dorsal + STS with a query from frontal, and sends
             frontal one vector in its naming space. Frontal names it, forms
             an expectation, and that becomes the next query -- and the next
             pass's thalamic gains. Three passes.

WORKS BEFORE TRAINING. STS carries IT's identity vector -- already in the
space frontal names in -- straight to POT, and POT's learned correction starts
at exactly zero. Untrained, v2.B is its vision model's zero-shot classifier.
Sound gets its head start differently: the temporal lobe names the sound in
its own lexicon (CLAP's sound-word space) and sends those scores to frontal over
the arcuate fasciculus. (A linear bridge into frontal's word space was tried
and named ESC-50 sounds 17.5% right against CLAP's own 89.8% -- abandoned.)
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F

from v2b import config_v2b as C
from v2b.wiring import Wiring


@dataclass(frozen=True)
class Dims:
    """Widths read from the loaded pretrained networks -- never assumed."""

    vision: int          # ventral vision network hidden width
    embed: int           # frontal's naming space
    parietal: int        # parietal cell: backbone width + depth geometry
    audio: int           # auditory network hidden width
    audio_stats: int     # MGN's statistics: mel bands + loudness + duration
    sound_embed: int = 1 # CLAP's sound-word space (1 when the sound network has none)


@dataclass
class Features:
    """What the sensory lobes delivered, batched. Absent senses are zeros,
    flagged by has_image / has_audio, so one batch can mix sights and sounds."""

    occ_grid: torch.Tensor       # N,49,vision   occipital (V4-level)
    it_grid: torch.Tensor        # N,49,vision   inferotemporal
    identity: torch.Tensor       # N,embed       IT's "what", in naming space
    par_cells: torch.Tensor      # N,49,parietal
    aud_bins: torch.Tensor       # N,16,audio
    aud_identity: torch.Tensor   # N,audio
    aud_joint: torch.Tensor      # N,sound_embed the sound in CLAP's sound-word space
    image_stats: torch.Tensor    # N,8
    audio_stats: torch.Tensor    # N,audio_stats
    has_image: torch.Tensor      # N bool
    has_audio: torch.Tensor      # N bool

    def to(self, device) -> "Features":
        return Features(*(getattr(self, k).to(device) for k in self.__dataclass_fields__))

    @property
    def size(self) -> int:
        return self.has_image.shape[0]

    @staticmethod
    def join(a: "Features", b: "Features") -> "Features":
        return Features(*(torch.cat([getattr(a, k), getattr(b, k)])
                          for k in Features.__dataclass_fields__))


@dataclass
class PassOut:
    embedding: torch.Tensor      # N,E   what POT sent frontal
    logits: torch.Tensor         # N,V   frontal's scores
    attention: torch.Tensor      # N,1+49+67 where POT looked
    query: torch.Tensor          # N,E   what frontal asked for
    gains: dict                  # relay -> N, this pass


@dataclass
class CortexOut:
    passes: list
    activity: list = field(default_factory=list)   # per pass: name -> tensor
    carried: list = field(default_factory=list)    # per pass: connection -> N


# ---------------------------------------------------------------------------
# Thalamus
# ---------------------------------------------------------------------------


class Relay(nn.Module):
    """One thalamic nucleus: signal statistics in, a gain in (0, 1) out."""

    def __init__(self, stats: int):
        super().__init__()
        self.hidden = nn.Linear(stats, C.GATE_HIDDEN)
        self.gain = nn.Linear(C.GATE_HIDDEN, 1)
        nn.init.constant_(self.gain.bias, C.GATE_OPEN_BIAS)

    def forward(self, stats):
        hidden = F.gelu(self.hidden(stats))
        return torch.sigmoid(self.gain(hidden)).squeeze(-1), hidden


class Thalamus(nn.Module):
    NUCLEI = ("pulvinar", "LGN", "MGN")

    def __init__(self, dims: Dims):
        super().__init__()
        self.pulvinar = Relay(C.IMAGE_STATS)     # eyes -> parietal
        self.LGN = Relay(C.IMAGE_STATS)          # eyes -> occipital
        self.MGN = Relay(dims.audio_stats)       # ears -> auditory cortex
        #  Frontal -> TRN -> each relay. Zero weights: exactly 1x at start.
        self.trn = nn.Linear(dims.embed, len(self.NUCLEI))
        nn.init.zeros_(self.trn.weight)
        nn.init.zeros_(self.trn.bias)

    def feedback(self, query):
        return C.TRN_RANGE * torch.sigmoid(self.trn(query))


# ---------------------------------------------------------------------------
# Dorsal stream, STS, POT
# ---------------------------------------------------------------------------


class Dorsal(nn.Module):
    """V4 cell (i, j) + parietal cell (i, j) -> one "what is where" token."""

    def __init__(self, dims: Dims):
        super().__init__()
        width = dims.vision + dims.parietal
        self.norm = nn.LayerNorm(width)
        self.hidden = nn.Linear(width, C.STREAM_DIM)
        self.out = nn.Linear(C.STREAM_DIM, C.STREAM_DIM)
        self.place = nn.Parameter(torch.randn(C.GRID * C.GRID, C.STREAM_DIM) * 0.02)

    def forward(self, occ, par):
        hidden = F.gelu(self.hidden(self.norm(torch.cat([occ, par], -1))))
        return self.out(hidden) + self.place, hidden


class STS(nn.Module):
    """Superior temporal sulcus: IT's sight and auditory cortex's sound, mixed.

    Tokens: IT identity (1), IT grid (49), sound identity (1), sound over
    time (16) -- 67 in all.
    """

    SIGHT = 1 + C.GRID * C.GRID
    SOUND = 1 + C.TEMPORAL_BINS

    def __init__(self, dims: Dims):
        super().__init__()
        self.identity = nn.Linear(dims.embed, C.STREAM_DIM)
        self.it_norm = nn.LayerNorm(dims.vision)
        self.it = nn.Linear(dims.vision, C.STREAM_DIM)
        self.sound_norm = nn.LayerNorm(dims.audio)
        self.sound = nn.Linear(dims.audio, C.STREAM_DIM)
        self.kind = nn.Parameter(torch.randn(4, C.STREAM_DIM) * 0.02)
        self.place = nn.Parameter(torch.randn(C.GRID * C.GRID, C.STREAM_DIM) * 0.02)
        self.when = nn.Parameter(torch.randn(C.TEMPORAL_BINS, C.STREAM_DIM) * 0.02)
        self.mix = nn.TransformerEncoderLayer(
            C.STREAM_DIM, C.STREAM_HEADS, 2 * C.STREAM_DIM, dropout=0.1,
            batch_first=True, norm_first=True)

    def forward(self, identity, it_grid, aud_identity, aud_bins, sight_off, sound_off):
        tokens = torch.cat([
            self.identity(identity)[:, None] + self.kind[0],
            self.it(self.it_norm(it_grid)) + self.kind[1] + self.place,
            self.sound(self.sound_norm(aud_identity))[:, None] + self.kind[2],
            self.sound(self.sound_norm(aud_bins)) + self.kind[3] + self.when], 1)
        absent = torch.cat([sight_off[:, None].expand(-1, self.SIGHT),
                            sound_off[:, None].expand(-1, self.SOUND)], 1)
        #  A row with nothing present would divide by zero in attention; such
        #  rows are masked out at POT anyway, so let them attend freely here.
        empty = absent.all(dim=1, keepdim=True)
        return self.mix(tokens, src_key_padding_mask=absent & ~empty), absent


class POT(nn.Module):
    """Temporo-parieto-occipital junction: fuses both streams for frontal.

    Its output is a correction added to IT's own identity vector, and the
    layer producing it starts at zero -- so untrained, POT hands frontal
    exactly what the vision network saw.
    """

    def __init__(self, dims: Dims):
        super().__init__()
        self.resting = nn.Parameter(torch.randn(dims.embed) * 0.02)
        self.ask = nn.Linear(dims.embed, C.STREAM_DIM)
        self.nothing = nn.Parameter(torch.randn(1, C.STREAM_DIM) * 0.02)
        self.source = nn.Parameter(torch.randn(2, C.STREAM_DIM) * 0.02)
        self.attend = nn.MultiheadAttention(C.STREAM_DIM, C.STREAM_HEADS, batch_first=True)
        self.norm = nn.LayerNorm(C.STREAM_DIM)
        self.think = nn.Sequential(nn.Linear(C.STREAM_DIM, 2 * C.STREAM_DIM), nn.GELU(),
                                   nn.Linear(2 * C.STREAM_DIM, C.STREAM_DIM))
        self.correct = nn.Linear(C.STREAM_DIM, dims.embed)
        nn.init.zeros_(self.correct.weight)
        nn.init.zeros_(self.correct.bias)
        #  How much of the learned correction is applied (config KEEP_GRID):
        #  chosen after training, saved with the weights.
        self.register_buffer("keep", torch.ones(()))

    def forward(self, query, tokens, absent, identity):
        n = query.shape[0]
        ask = self.ask(query)[:, None]
        keys = torch.cat([self.nothing.expand(n, 1, -1), tokens], 1)
        mask = torch.cat([torch.zeros(n, 1, dtype=torch.bool, device=absent.device), absent], 1)
        fused, weights = self.attend(ask, keys, keys, key_padding_mask=mask)
        fused = fused + ask
        fused = (fused + self.think(self.norm(fused)))[:, 0]
        embedding = F.normalize(identity + self.keep * self.correct(fused), dim=-1)
        return embedding, weights[:, 0], fused


# ---------------------------------------------------------------------------
# The whole trainable circuit
# ---------------------------------------------------------------------------


class Cortex(nn.Module):
    def __init__(self, dims: Dims, scale: float, wiring: Wiring | None = None,
                 sound_scale: float = 1.0):
        super().__init__()
        self.dims = dims
        self.scale = scale
        self.sound_scale = sound_scale
        #  How much frontal weighs the temporal lobe's own naming of a sound.
        #  Starts at 1 (CLAP's own temperature), then trained.
        self.log_arcuate = nn.Parameter(torch.zeros(()))
        #  The temporal lobe's sound lexicon (V x sound_embed; zero rows for
        #  picture names). Set by whoever builds the brain; not trained.
        self.register_buffer("lexicon", torch.zeros(0, 1), persistent=False)
        self.thalamus = Thalamus(dims)
        self.dorsal = Dorsal(dims)
        self.sts = STS(dims)
        self.pot = POT(dims)
        self.wiring = wiring or Wiring()

    def forward(self, f: Features, words: torch.Tensor, record: bool = False,
                sound_words: torch.Tensor | None = None) -> CortexOut:
        if sound_words is None and self.lexicon.numel():
            sound_words = self.lexicon
        w = self.wiring
        o = {name: float(w.open(name)) for name in w.names}
        image, audio = f.has_image.float(), f.has_audio.float()

        base = {"pulvinar": self.thalamus.pulvinar(f.image_stats),
                "LGN": self.thalamus.LGN(f.image_stats),
                "MGN": self.thalamus.MGN(f.audio_stats)}
        present = {"pulvinar": image * o["pulvinar->parietal"],
                   "LGN": image * o["LGN->occipital"],
                   "MGN": audio * o["MGN->temporal"]}

        query = F.normalize(self.pot.resting, dim=-1).expand(f.size, -1)
        out = CortexOut([])
        for _ in range(C.FRONTAL_PASSES):
            # -- thalamus: own gain x frontal's feedback via the TRN --------
            if o["frontal->thalamus"]:
                feedback = self.thalamus.feedback(query)
            else:
                feedback = torch.ones(f.size, 3, device=query.device)
            gains = {name: base[name][0] * feedback[:, i] * present[name]
                     for i, name in enumerate(Thalamus.NUCLEI)}
            g_pul, g_lgn, g_mgn = gains["pulvinar"], gains["LGN"], gains["MGN"]
            ventral = o["occipital->ventral"]         # V4 -> IT, inside one network

            # -- dorsal stream -------------------------------------------------
            occ = f.occ_grid * (g_lgn * o["occipital->dorsal"])[:, None, None]
            par = f.par_cells * (g_pul * o["parietal->dorsal"])[:, None, None]
            dorsal, d_hidden = self.dorsal(occ, par)
            dorsal_off = ~(((g_lgn * o["occipital->dorsal"]) > 0)
                           | ((g_pul * o["parietal->dorsal"]) > 0))
            if not o["dorsal->POT"]:
                dorsal_off = torch.ones_like(dorsal_off)

            # -- STS: sight (from IT) meets sound (from auditory cortex) ------
            sight = g_lgn * ventral * o["ventral->STS"]
            sound = g_mgn * o["temporal->STS"]
            identity_in = f.identity * sight[:, None]
            sts, sts_absent = self.sts(identity_in, f.it_grid * sight[:, None, None],
                                       f.aud_identity * sound[:, None],
                                       f.aud_bins * sound[:, None, None],
                                       sight == 0, sound == 0)
            #  The zero-shot shortcut for sight: IT's identity already lives
            #  where frontal names things. (Sound's is the arcuate, below.)
            identity = F.normalize(f.identity, dim=-1) * sight[:, None]
            if not o["STS->POT"]:
                sts_absent = torch.ones_like(sts_absent)
                identity = identity * 0

            tokens = torch.cat([dorsal + self.pot.source[0], sts + self.pot.source[1]], 1)
            absent = torch.cat([dorsal_off[:, None].expand(-1, C.GRID * C.GRID), sts_absent], 1)

            # -- POT -> frontal --------------------------------------------------
            embedding, attention, fused = self.pot(query, tokens, absent, identity)
            to_frontal = embedding * o["POT->frontal"]
            logits = self.scale * to_frontal @ words.T
            #  THE ARCUATE FASCICULUS: the temporal lobe names the sound in its
            #  own lexicon (Wernicke's area) and sends those scores straight to
            #  frontal (Broca's). Only sound names have lexicon rows.
            arcuate = torch.zeros(f.size, device=logits.device)
            if sound_words is not None and o["temporal->frontal"] and sound_words.shape[1] > 1:
                heard = F.normalize(f.aud_joint, dim=-1) * (sound > 0).float()[:, None]
                logits = logits + (self.sound_scale * self.log_arcuate.exp()
                                   * heard @ sound_words.T)
                arcuate = (sound > 0).float()
            out.passes.append(PassOut(embedding, logits, attention, query, gains))
            if record:
                out.activity.append({
                    "dorsal.hidden": d_hidden.mean(1), "dorsal.tokens": dorsal.mean(1),
                    "STS.tokens": sts.mean(1), "POT.fused": fused,
                    "thalamus.feedback": feedback,
                    "relay.hidden": {k: base[k][1] for k in base}})
                out.carried.append({
                    "pulvinar->parietal": g_pul, "LGN->occipital": g_lgn,
                    "MGN->temporal": g_mgn,
                    "occipital->ventral": g_lgn * ventral,
                    "occipital->dorsal": occ.abs().mean((1, 2)),
                    "parietal->dorsal": par.abs().mean((1, 2)),
                    "ventral->STS": sight, "temporal->STS": sound,
                    "dorsal->POT": (~dorsal_off).float(),
                    "STS->POT": (~sts_absent).float().mean(1),
                    "POT->frontal": to_frontal.norm(dim=-1),
                    "frontal->POT": query.norm(dim=-1) * o["frontal->POT"],
                    "frontal->thalamus": (feedback - 1).abs().mean(1) * o["frontal->thalamus"],
                    "temporal->frontal": arcuate})

            # -- frontal's expectation becomes the next question --------------
            if o["frontal->POT"]:
                p = logits.softmax(-1)
                top, index = p.topk(min(C.FRONTAL_TOP_K, p.shape[-1]), dim=-1)
                expected = F.normalize((top[..., None] * words[index]).sum(1), dim=-1)
                query = F.normalize(C.FRONTAL_EXPECTATION * expected
                                    + (1 - C.FRONTAL_EXPECTATION) * embedding, dim=-1)
        return out
