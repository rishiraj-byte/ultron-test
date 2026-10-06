"""
verify_v2b.py -- does v2.B do what its diagram says?

    python -m v2b.verify_v2b

Every check runs the real system, and each is phrased as a claim a plausible
bug would break. Numbers that describe rather than test are printed as [info].
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

from v2b import config_v2b as C
from v2b import data

RESULTS: list[bool] = []


def check(claim: str, passed: bool, detail: str = "") -> None:
    RESULTS.append(bool(passed))
    print("[%s] %-60s %s" % ("  ok  " if passed else " FAIL ", claim, detail))


def note(claim: str, detail: str) -> None:
    print("[ info ] %-60s %s" % (claim, detail))


def section(title: str) -> None:
    print("\n-- %s --" % title)


def _clip(folder: Path, frame: np.ndarray, wave: np.ndarray) -> Path:
    """A 2-second video: one picture, and a real sound as its soundtrack."""
    import cv2
    import imageio_ffmpeg
    import soundfile
    silent = folder / "silent.mp4"
    writer = cv2.VideoWriter(str(silent), cv2.VideoWriter_fourcc(*"mp4v"), 10, (96, 96))
    for _ in range(20):
        writer.write(cv2.resize(frame[:, :, ::-1], (96, 96), interpolation=cv2.INTER_NEAREST))
    writer.release()
    wav = folder / "sound.wav"
    soundfile.write(str(wav), wave[:C.AUDIO_SAMPLE_RATE * 2], C.AUDIO_SAMPLE_RATE)
    clip = folder / "clip.mp4"
    subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error", "-i", str(silent),
                    "-i", str(wav), "-c:v", "copy", "-c:a", "aac", "-shortest", str(clip)],
                   check=True, timeout=60)
    return clip


def main() -> int:
    print("=" * 76)
    print("v2.B VERIFICATION -- is the build the architecture that was drawn?")
    print("=" * 76)
    from PIL import Image
    from v2b.brain import BrainV2B
    from v2b.cortex import Cortex

    scratch = Path(tempfile.mkdtemp(prefix="v2b_verify_"))
    brain = BrainV2B(memory_dir=scratch / "memory")
    images, labels, _ = data.cifar("test", per_class=2)
    names = data.cifar_names()
    by_class = {c: np.flatnonzero(labels == c) for c in range(100)}
    picture = Image.fromarray(images[by_class[3][0]])        # a bear
    other = Image.fromarray(images[by_class[3][1]])          # a different bear
    waves, sound_labels, _ = data.audio("test")

    # ------------------------------------------------------------------
    section("1. PRETRAINED, OPEN-WEIGHT, FROZEN")
    lobes = {"ventral vision": brain.vision, "parietal": brain.parietal,
             "temporal": brain.temporal, "frontal": brain.frontal}
    sizes = {n: sum(p.numel() for p in m.parameters()) / 1e6 for n, m in lobes.items()}
    trainable = sum(p.numel() for m in lobes.values() for p in m.parameters() if p.requires_grad)
    check("no pretrained weight can receive a gradient", trainable == 0,
          "  ".join("%s %.0fM" % kv for kv in sizes.items()))
    for lobe in lobes.values():
        lobe.train()
    brain.cortex.train()
    stuck = all(not m.training for lobe in lobes.values() for m in lobe.modules())
    brain.cortex.eval()
    check("asking a lobe to train leaves it in eval mode", stuck, "(the BatchNorm trap from v2)")
    loaded = {type(m).__name__ for lobe in lobes.values() for m in lobe.modules()}
    llm = sorted(n for n in loaded if any(w in n for w in
                                          ("CausalLM", "Qwen", "Llama", "Gemma", "Mistral")))
    check("no language model anywhere", not llm, llm or "text is only ever encoded, never generated")
    small = sum(p.numel() for m in (brain.cortex, brain.entorhinal, brain.ca1, brain.amygdala)
                for p in m.parameters())
    check("the trained part is small next to the frozen part",
          small < 0.02 * sum(sizes.values()) * 1e6,
          "%.2fM trained vs %.0fM frozen" % (small / 1e6, sum(sizes.values())))
    note("networks in use", "  ".join("%s=%s" % kv for kv in brain.models().items()))

    # ------------------------------------------------------------------
    section("2. ANATOMY")
    blocks = len(brain.vision.blocks)
    split = brain.vision.split
    check("one vision network spans occipital -> ventral stream (IT)",
          len(brain.vision.occipital_blocks) == split
          and len(brain.vision.it_blocks) == blocks - split and 0 < split < blocks,
          "occipital blocks 1-%d, ventral stream (IT) blocks %d-%d" % (split, split + 1, blocks))
    f, _ = brain.features([picture], None)
    occ = torch.nn.functional.normalize(f.occ_grid[0].mean(0), dim=-1)
    it = torch.nn.functional.normalize(f.it_grid[0].mean(0), dim=-1)
    check("the dorsal stream gets V4-level features, not IT's answer",
          float((occ * it).sum()) < 0.95, "cosine(V4 grid, IT grid) %.3f" % float((occ * it).sum()))
    seen = brain.perceive(image=picture)
    heard = brain.perceive(audio=waves[0])
    g_seen, g_heard = seen.gains[0], heard.gains[0]
    check("an image reaches LGN and pulvinar, never MGN",
          g_seen["LGN"] > 0 and g_seen["pulvinar"] > 0 and g_seen["MGN"] == 0,
          "LGN %.2f  pulvinar %.2f  MGN %.2f" % (g_seen["LGN"], g_seen["pulvinar"], g_seen["MGN"]))
    check("a sound reaches MGN only",
          g_heard["MGN"] > 0 and g_heard["LGN"] == 0 and g_heard["pulvinar"] == 0,
          "MGN %.2f" % g_heard["MGN"])
    fresh = Cortex(brain.dims, brain.frontal.scale, brain.wiring,
                   sound_scale=brain.temporal.scale).to(brain.device).eval()
    fresh.lexicon = brain.cortex.lexicon
    feedback = fresh.thalamus.feedback(torch.randn(4, brain.dims.embed, device=brain.device))
    check("untrained corticothalamic feedback changes nothing (1x)",
          torch.allclose(feedback, torch.ones_like(feedback)))
    gains = [p["LGN"] for p in seen.gains]
    note("trained TRN: LGN gain over the 3 passes", "  ".join("%.3f" % g for g in gains))

    # ------------------------------------------------------------------
    section("3. THE LOBES HAND ON GRIDS, NOT SINGLE VECTORS")
    fa, _ = brain.features([picture], waves[0])
    check("occipital + IT: 7x7 grids", tuple(fa.occ_grid.shape[1:2]) == (49,)
          and tuple(fa.it_grid.shape[1:2]) == (49,), "%s" % (tuple(fa.it_grid.shape),))
    check("parietal: a 7x7 grid with depth geometry per cell",
          fa.par_cells.shape[1:] == (49, brain.dims.parietal), str(tuple(fa.par_cells.shape)))
    check("temporal (auditory cortex): 16 slices of time", fa.aud_bins.shape[1] == C.TEMPORAL_BINS,
          str(tuple(fa.aud_bins.shape)))
    check("parietal's depth map is a real map, not a constant",
          seen.depth is not None and float(seen.depth.std()) > 0.05,
          "depth spread %.3f" % float(seen.depth.std()))

    # ------------------------------------------------------------------
    section("4. UNTRAINED, POT PASSES IT STRAIGHT THROUGH (zero-shot)")
    with torch.no_grad():
        out = fresh(f.to(brain.device), brain.frontal.words.to(brain.device))
        agree = float((out.passes[-1].embedding[0]
                       * torch.nn.functional.normalize(f.identity[0], dim=-1)).sum())
    check("untrained POT output == IT's own identity vector", agree > 0.999, "cosine %.5f" % agree)

    # ------------------------------------------------------------------
    section("5. THE ONLY WAY OUT IS THROUGH FRONTAL")
    for name in ("POT->frontal", "frontal->motor"):
        brain.lesion(name)
        silent = brain.perceive(image=picture)
        brain.lesion(name, cut=False)
        check("cut %s: no answer at all" % name, not silent.top)

    # ------------------------------------------------------------------
    section("6. THE FRONTAL LOOP")
    with torch.no_grad():
        queries = [p.query for p in brain.cortex(f, brain.frontal.words.to(brain.device)).passes]
        moved = [float(1 - (a * b).sum()) for a, b in zip(queries, queries[1:])]
    check("frontal asks POT a different question each pass", all(m > 1e-4 for m in moved),
          "change %s" % "  ".join("%.4f" % m for m in moved))
    brain.lesion("frontal->POT")
    with torch.no_grad():
        queries = [p.query for p in brain.cortex(f, brain.frontal.words.to(brain.device)).passes]
    brain.lesion("frontal->POT", cut=False)
    check("cut frontal->POT: every pass asks the same question",
          all(torch.equal(queries[0], q) for q in queries))

    # ------------------------------------------------------------------
    section("7. TRAINING (test set, never seen by any stage)")
    record = brain.record
    if not record:
        check("a trained checkpoint exists for these networks", False,
              brain.mismatch or "run: python -m v2b.train")
    else:
        t = record["test"]
        sounds = len(data.audio_names())
        check("training beat zero-shot on held-out images", t["image"]["fused"] > record["untrained"]["image"],
              "%.1f%% -> %.1f%%  (chance 1%%)" % (record["untrained"]["image"], t["image"]["fused"]))
        check("hearing was learned", t["audio"]["fused"] > 100 / sounds + 30,
              "%.1f%% on %s (chance %.1f%%)" % (t["audio"]["fused"], record["audio_data"], 100 / sounds))
        cov = record["gradient_coverage"]
        check("every trainable tensor received gradient", not cov["missing"],
              "%d / %d" % (cov["covered"], cov["total"]))
        for kind in ("image", "audio"):
            note("%s: cortex / memory / cortex+memory" % kind, "%.1f%% / %.1f%% / %.1f%%"
                 % (t[kind]["cortex"], t[kind]["memory"], t[kind]["fused"]))
        o = record.get("open_vocabulary")
        if o:
            check("it names things it never trained on (open vocabulary)",
                  o["fused"] > 50,
                  "%d pictures of %d never-trained classes, out of %d names: %.1f%% "
                  "(untrained %.1f%%)" % (o["pictures"], o["classes"], record["vocabulary"],
                                          o["fused"], o["untrained"]))

    # ------------------------------------------------------------------
    section("7b. TEMPORAL -> FRONTAL: THE ARCUATE FASCICULUS")
    check("frontal can name %d sounds and %d sights" % (
              brain.frontal.kinds.count("sound"), brain.frontal.kinds.count("image")),
          brain.frontal.kinds.count("sound") > 100 and "web site" in brain.frontal.names)
    brain.lesion("temporal->frontal")
    unheard = brain.perceive(audio=waves[0])
    brain.lesion("temporal->frontal", cut=False)
    heard_again = brain.perceive(audio=waves[0])
    check("the sound lexicon reaches frontal only over the arcuate",
          heard_again.carried[-1]["temporal->frontal"] == 1.0
          and unheard.carried[-1]["temporal->frontal"] == 0.0
          and heard_again.cortex_top != unheard.cortex_top,
          "with it: '%s' %.0f%%; cut: '%s' %.0f%%" % (
              heard_again.cortex_top[0][0], 100 * heard_again.cortex_top[0][1],
              unheard.cortex_top[0][0], 100 * unheard.cortex_top[0][1]))

    # ------------------------------------------------------------------
    section("8. HIPPOCAMPUS -- RETRIEVAL-AUGMENTED MEMORY")
    bank, _ = brain.hippocampus.size
    check("the consolidated memory is loaded", bank > 1000, "%s memories" % f"{bank:,}")
    hits = 0
    for c in range(0, 100, 2):
        thought = brain.perceive(image=Image.fromarray(images[by_class[c][1]]))
        hits += bool(thought.memory_top) and thought.memory_top[0][0] == names[c]
    check("memory generalises: recall names unseen test pictures", hits >= 25,
          "memory alone right on %d / 50 test pictures" % hits)
    if record:
        t = record["test"]["image"]
        check("adding memory does not make cortex worse", t["fused"] >= t["cortex"] - 0.3,
              "cortex %.1f%% -> with memory %.1f%% (trust %.2f)"
              % (t["cortex"], t["fused"], record["trust"]))

    # ------------------------------------------------------------------
    section("9. TEACHING IT, AND THE AMYGDALA")
    first = brain.perceive(image=picture)
    brain.correct(first, "garden gnome")
    again = brain.perceive(image=picture)
    check("a NEW word, taught once, is used for that picture", again.answer == "garden gnome",
          "'%s' -> '%s'" % (first.answer, again.answer))
    elsewhere = brain.perceive(image=other)
    check("...without relabelling other pictures of the same class",
          elsewhere.answer != "garden gnome", "another bear is still '%s'" % elsewhere.answer)
    brain.rate(again, good=False)
    third = brain.perceive(image=picture)
    check("rated bad, the amygdala pushes frontal off that answer",
          third.answer != "garden gnome", "now '%s'" % third.answer)
    kept = len(brain.hippocampus.kept)
    for _ in range(3):
        brain.perceive(image=other)
    check("unrated percepts are not written to long-term memory",
          len(brain.hippocampus.kept) == kept, "%d kept before and after" % kept)
    for name in ("hippocampus->frontal", "amygdala->frontal"):
        brain.lesion(name)
    plain = brain.perceive(image=picture)
    brain.wiring.restore()
    check("cut both memory paths: back to what cortex alone says",
          plain.answer == plain.cortex_top[0][0], "'%s'" % plain.answer)
    bias = brain.cortex.thalamus.MGN.gain.bias
    saved = bias.detach().clone()
    with torch.no_grad():
        bias.fill_(-20.0)                       # the ears' relay forced shut
    deaf = brain.perceive(audio=waves[0])
    brain.lesion("amygdala->motor")
    ungated = brain.perceive(audio=waves[0])
    brain.lesion("amygdala->motor", cut=False)
    with torch.no_grad():
        bias.copy_(saved)
    check("no thalamic drive -> amygdala says HOLD, never a confident name",
          deaf.abstained and deaf.decision == "HOLD",
          "gate %.6f, %s %.2f" % (deaf.gains[-1]["MGN"], deaf.decision, deaf.value))
    check("cut amygdala->motor: the verdict no longer gates the answer",
          not ungated.abstained, "motor said '%s' anyway" % ungated.answer)
    heard_all = brain.perceive(image=other)
    brain.lesion("hippocampus->amygdala")
    heard_none = brain.perceive(image=other)
    brain.lesion("hippocampus->amygdala", cut=False)
    check("the amygdala hears memory directly from the hippocampus",
          heard_all.amygdala_inputs["closest memory"] > 0
          and heard_none.amygdala_inputs["closest memory"] == 0,
          "closest memory it sees: %.2f intact, %.2f with hippocampus->amygdala cut"
          % (heard_all.amygdala_inputs["closest memory"],
             heard_none.amygdala_inputs["closest memory"]))
    if record:
        t = record["test"]["image"]
        check("the amygdala's value is informative",
              t["accuracy_when_answered"] > t["accuracy_when_abstaining"] + 20,
              "answers %.0f%% of images at %.1f%% right; held-back ones are %.1f%% right"
              % (t["answered"], t["accuracy_when_answered"], t["accuracy_when_abstaining"]))

    # ------------------------------------------------------------------
    section("10. VIDEO IS FRAMES + SOUNDTRACK")
    try:
        clip = _clip(scratch, images[by_class[3][0]], waves[0])
        watched = brain.perceive(video=clip)
        check("a clip reaches the eyes AND the ears",
              watched.gains[0]["LGN"] > 0 and watched.gains[0]["MGN"] > 0,
              "LGN %.2f  MGN %.2f" % (watched.gains[0]["LGN"], watched.gains[0]["MGN"]))
    except Exception as failure:                              # noqa: BLE001
        check("a clip reaches the eyes AND the ears", False, str(failure)[:60])

    # ------------------------------------------------------------------
    section("11. EVERYTHING THE DASHBOARD SHOWS IS REAL")
    lives = [seen.live.get(k) for k in ("occipital", "ventral", "parietal")] + [heard.live.get("temporal")]
    check("per-block activity recorded for all four sensory networks",
          all(l is not None and len(l.blocks) and len(l.neurons) for l in lives),
          "blocks: %s" % [len(l.blocks) for l in lives if l is not None])
    check("attention maps captured from the vision network",
          len(seen.live["ventral"].attention) == len(brain.vision.it_blocks))
    rag = seen.rag
    recalled = {r["index"] for r in rag.get("recalled", [])}
    similarity = np.asarray(rag.get("similarity", []), np.float32)
    strongest = set(np.argsort(-similarity)[:len(recalled)].tolist()) if len(similarity) else set()
    check("RAG view: its store layer is the store recall() searched",
          len(recalled) == C.MEMORY_K and recalled == strongest,
          "the %d most active of %s units are the %d recalled" % (
              len(strongest), f"{len(similarity):,}", len(recalled)))
    #  Not tautological: "final" is read from the scores frontal actually
    #  summed, "cortex" from cortex's own, and the memory term is rebuilt from
    #  trust x familiarity x vote. With no episode applying, they must add up.
    ledger = rag.get("ledger", [])
    episodes_apply = any(e["applies"] for e in rag.get("episodes", []))
    worst = max((abs(r["final"] - r["cortex"] - r["memory"]) for r in ledger), default=1.0)
    check("RAG view: cortex + trust x familiarity x vote = what frontal summed",
          bool(ledger) and ledger[0]["name"] == seen.answer
          and (episodes_apply or worst < 1e-3),
          "top '%s' = the answer; worst row off by %.1e" % (
              ledger[0]["name"] if ledger else "-", worst))
    regions = brain.networks()
    empty = [r for r, parts in regions.items() if not sum(p.numel() for _, m in parts for p in m.parameters())]
    check("every region's weights are reachable for the WEIGHTS view", not empty,
          "%d regions, %s numbers" % (len(regions), f"{sum(p.numel() for parts in regions.values() for _, m in parts for p in m.parameters()):,}"))

    shutil.rmtree(scratch, ignore_errors=True)
    passed = sum(RESULTS)
    print("\n" + "=" * 76)
    print("%d passed, %d failed" % (passed, len(RESULTS) - passed))
    print("=" * 76)
    return 0 if passed == len(RESULTS) else 1


if __name__ == "__main__":
    sys.exit(main())
