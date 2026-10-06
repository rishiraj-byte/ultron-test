"""
lessons.py -- learn how v2.B works by poking it. Run one lesson at a time:

    python -m v2b.lessons              list the lessons
    python -m v2b.lessons 1            what is inside a network
    python -m v2b.lessons 2            encode: image, sound, text -> vectors
    python -m v2b.lessons 3            vectors: compare, rank, do arithmetic
    python -m v2b.lessons 4            change a weight, watch the answer change
    python -m v2b.lessons 5            train: loss, gradient, step
    python -m v2b.lessons 6            save, load, and swap a whole network
    python -m v2b.lessons 7            memory: retrieval-augmented recognition (RAG)

Read each lesson's source next to its output -- the code IS the lesson.
Every lesson restores what it changes. LEARN.md explains the ideas in words;
the dashboard's WEIGHTS and ACTIVITY tabs show the same things live.
"""

from __future__ import annotations

import sys

import numpy as np
import torch
import torch.nn.functional as F

from v2b import config_v2b as C
from v2b import data


def say(text: str = "") -> None:
    print(text)


def heading(text: str) -> None:
    print("\n" + "=" * 72 + "\n" + text + "\n" + "=" * 72)


def _picture(cls: int = 3, which: int = 0):
    from PIL import Image
    images, labels, _ = data.cifar("test", per_class=2)
    index = np.flatnonzero(labels == cls)[which]
    return Image.fromarray(images[index]), data.cifar_names()[cls]


def _brain():
    from v2b.brain import BrainV2B
    import tempfile
    from pathlib import Path
    #  A throwaway memory folder: lessons never touch what you taught it.
    return BrainV2B(memory_dir=Path(tempfile.mkdtemp()))


# ---------------------------------------------------------------------------


def lesson_1() -> None:
    heading("LESSON 1 -- what is inside a network")
    from v2b.lobes import VentralVision, vision_language
    model, processor, family = vision_language()
    vision = VentralVision(model, processor, family).vision

    say("A network is a stack of layers. Each layer is a few WEIGHT TENSORS --")
    say("grids of numbers -- and a rule for combining its input with them.")
    say("The ventral vision network (%s), top level:\n" % C.VISION_MODEL)
    for name, child in vision.named_children():
        count = sum(p.numel() for p in child.parameters())
        say("   %-16s %-30s %12s numbers" % (name, type(child).__name__, f"{count:,}"))
    blocks = vision.encoder.layers
    split = max(1, int(len(blocks) * C.OCCIPITAL_SHARE))
    say("\nIts encoder is %d identical blocks. v2.B calls blocks 1-%d OCCIPITAL"
        % (len(blocks), split))
    say("(V1 -> V4) and blocks %d-%d the VENTRAL STREAM's end, inferotemporal"
        % (split + 1, len(blocks)))
    say("cortex (IT). One block holds:\n")
    for name, tensor in blocks[0].named_parameters():
        say("   %-28s shape %s" % (name, tuple(tensor.shape)))
    weight = blocks[0].self_attn.q_proj.weight
    say("\nA weight is a number you can read. Row 0 of block 1's query matrix:")
    say("   %s ..." % np.array2string(weight[0, :6].detach().float().cpu().numpy(), precision=4))
    say("\nEvery weight is addressed by name:")
    say("   vision.state_dict()['encoder.layers.0.self_attn.q_proj.weight']")
    say("\nSee all of them, as pictures, in the dashboard: WEIGHTS tab, FOCUS 'occipital'.")


def lesson_2() -> None:
    heading("LESSON 2 -- encoding: an image, a sound, a word -> vectors")
    from v2b import senses
    brain = _brain()
    picture, truth = _picture()

    say("ENCODING = run an input through a network and keep what comes out: a")
    say("VECTOR, a list of numbers. Similar inputs give similar vectors.\n")
    pixels = senses.retina([picture])
    say("1. RETINA: a %s picture -> tensor %s (colour, height, width), 0..1."
        % (picture.size, tuple(pixels.shape[1:])))
    seen = brain.see(pixels)
    say("2. OCCIPITAL (V4 level): a 7x7 grid, %s -- one vector per place."
        % (tuple(seen["occ_grid"].shape[1:]),))
    say("   IT: another 7x7 grid %s, and ONE identity vector of %d numbers:"
        % (tuple(seen["it_grid"].shape[1:]), seen["identity"].shape[-1]))
    say("   %s ..." % np.array2string(seen["identity"][0, :8].cpu().numpy(), precision=3))
    say("   PARIETAL: %s per place (backbone features + nearness, spread, slope)"
        % (tuple(seen["par_cells"].shape[1:]),))
    waves, labels, _ = data.audio("test")
    heard = brain.hear(waves[:1])
    say("3. AUDITORY: a %.1f s sound of '%s' -> %s: 16 slices of time."
        % (waves[0].size / C.AUDIO_SAMPLE_RATE, data.audio_names()[labels[0]],
           tuple(heard["aud_bins"].shape[1:])))
    words = brain.frontal.encode(["a photo of a %s." % truth, "a photo of a car."])
    say("4. FRONTAL: a PHRASE -> a vector in the SAME space as IT's identity.")
    image = F.normalize(seen["identity"][0].float(), dim=-1)
    say("   similarity(picture, '%s') = %.3f" % (truth, float(image @ words[0])))
    say("   similarity(picture, 'car')  = %.3f" % float(image @ words[1]))
    say("\nThat shared space is why v2.B can name pictures with no training at all.")


def lesson_3() -> None:
    heading("LESSON 3 -- vectors: compare, rank, and add them")
    from v2b import senses
    from v2b.lobes import Frontal, VentralVision, vision_language
    model, processor, family = vision_language()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    frontal = Frontal(model, family).to(device)
    vision = VentralVision(model, processor, family).to(device)

    say("COSINE SIMILARITY: do two vectors point the same way? For unit-length")
    say("vectors it is a dot product: a @ b. 1 = same direction.\n")
    phrases = ["a photo of a dog.", "a photo of a wolf.", "a photo of a cat.",
               "a photo of a truck.", "a photo of a bus."]
    table = (frontal.encode(phrases) @ frontal.encode(phrases).T).cpu().numpy()
    short = [p.split(" a ")[-1].rstrip(".") for p in phrases]
    say("          " + "".join("%8s" % s for s in short))
    for s, row in zip(short, table):
        say("   %-6s " % s + "".join("%8.3f" % v for v in row))
    say("\nRANKING: frontal's whole job. Compare one vector with every name it knows:")
    names, kinds = data.vocabulary()
    frontal.set_vocabulary(names, kinds)
    picture, truth = _picture()
    image = F.normalize(vision(senses.retina([picture])).identity, dim=-1)[0]
    best = (frontal.words @ image).topk(5)
    say("   a picture of a %s ranks: %s" % (truth, ", ".join(
        "%s %.3f" % (names[i], v) for v, i in zip(best.values.tolist(), best.indices.tolist()))))
    say("\nARITHMETIC: add two concept vectors, look at what is nearest the sum:")
    blend = F.normalize(frontal.encode(["a photo of a lion."])[0]
                        + frontal.encode(["a photo of a house."])[0], dim=-1)
    say("   lion + house is nearest to: %s" % ", ".join(
        names[i] for i in (frontal.words @ blend).topk(4).indices.tolist()))
    say("\nNORMALISE first (F.normalize) so similarity measures direction only.")


def lesson_4() -> None:
    heading("LESSON 4 -- change a weight, watch the answer change")
    from v2b import senses
    brain = _brain()
    picture, truth = _picture()
    waves, labels, _ = data.audio("test")
    sound = waves[0]

    say("The pattern for any edit: save = w.detach().clone(); change w inside")
    say("torch.no_grad(); look; w.copy_(save). Three edits, each undone.\n")
    say("EDIT 1 -- a TRAINED weight: the MGN relay's output bias.")
    bias = brain.cortex.thalamus.MGN.gain.bias
    before = brain.perceive(audio=sound)
    saved = bias.detach().clone()
    with torch.no_grad():
        bias.fill_(-20.0)                      # sigmoid(-20) ~ 0: the gate shuts
    after = brain.perceive(audio=sound)
    with torch.no_grad():
        bias.copy_(saved)
    say("   sound of '%s': gate %.2f -> '%s'" % (data.audio_names()[labels[0]],
                                                  before.gains[0]["MGN"], before.answer))
    say("   bias set to -20: gate %.6f -> %s"
        % (after.gains[0]["MGN"], "NOTHING PERCEIVED (motor: not sure)" if after.abstained
           else "'%s' (value %.2f)" % (after.answer, after.value)))
    say("   One number made it deaf. With no thalamic drive there is no percept, so")
    say("   motor cortex says so instead of naming something. (Before that rule")
    say("   existed it answered 'car horn' here, and the amygdala -- never trained")
    say("   on silence -- rated that 0.66. A network always answers unless built not to.)\n")

    say("EDIT 2 -- PRETRAINED weights: silence half the attention heads in IT.")
    pixels = senses.retina([picture])
    intact = F.normalize(brain.vision(pixels).identity, dim=-1)
    first = brain.perceive(image=picture)
    saved = [b.self_attn.out_proj.weight.detach().clone() for b in brain.vision.it_blocks]
    with torch.no_grad():
        for b in brain.vision.it_blocks:
            w = b.self_attn.out_proj.weight
            w[:, :w.shape[1] // 2] = 0
    edited = F.normalize(brain.vision(pixels).identity, dim=-1)
    second = brain.perceive(image=picture)
    with torch.no_grad():
        for b, w in zip(brain.vision.it_blocks, saved):
            b.self_attn.out_proj.weight.copy_(w)
    say("   a %s. IT's identity vector kept %.0f%% of its direction;"
        % (truth, 100 * float((intact * edited).sum())))
    say("   answer '%s' %.0f%% -> '%s' %.0f%%" % (first.answer, 100 * first.confidence,
                                                second.answer, 100 * second.confidence))
    say("   Networks are redundant: damage moves the vector more than the answer.\n")

    say("EDIT 3 -- zero a whole layer: POT's correction (= back to pure zero-shot).")
    layer = brain.cortex.pot.correct
    saved = (layer.weight.detach().clone(), layer.bias.detach().clone())
    with torch.no_grad():
        layer.weight.zero_()
        layer.bias.zero_()
    raw = brain.perceive(image=picture)
    with torch.no_grad():
        layer.weight.copy_(saved[0])
        layer.bias.copy_(saved[1])
    say("   trained: '%s' %.0f%%   correction zeroed: '%s' %.0f%%"
        % (first.answer, 100 * first.confidence, raw.answer, 100 * raw.confidence))
    say("\nThe dashboard does the same: WEIGHTS tab -> pick a tensor -> 'zero it'.")


def lesson_5() -> None:
    heading("LESSON 5 -- training: loss, gradient, step")
    import json
    from v2b.cortex import Cortex, Dims
    from v2b.train import Cache, _path
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        manifest = json.loads(_path("manifest.json").read_text())
    except FileNotFoundError:
        say("No feature cache yet -- run  python -m v2b.train  first.")
        return
    dims = Dims(**manifest["dims"])
    words = torch.from_numpy(np.load(_path("words.npy"))).to(device)
    cache = Cache("img", "train", dims)
    cortex = Cortex(dims, manifest["scale"],                    # a FRESH, untrained copy
                    sound_scale=manifest.get("sound_scale", 1.0)).to(device)
    cortex.lexicon = torch.from_numpy(np.load(_path("lexicon.npy"))).to(device)
    optimiser = torch.optim.AdamW(cortex.parameters(), lr=1e-4)

    say("Training repeats three moves:")
    say("  1. LOSS      run a batch; one number says how wrong the answers were")
    say("  2. GRADIENT  loss.backward(): for EVERY weight, which way lowers the loss")
    say("  3. STEP      optimiser.step(): move every weight a little that way\n")
    rng = np.random.default_rng(0)
    probe = cortex.pot.think[0].weight
    for step in range(31):
        f, y = cache.batch(rng.choice(len(cache), 128, replace=False), device)
        logits = cortex(f, words).passes[-1].logits
        loss = F.cross_entropy(logits, y)
        optimiser.zero_grad()
        loss.backward()
        if step in (0, 1, 5, 10, 20, 30):
            say("   step %2d   loss %.3f   batch accuracy %5.1f%%   |gradient| on one POT weight %.2e"
                % (step, float(loss.detach()), 100 * float((logits.argmax(-1) == y).float().mean()),
                   float(probe.grad.abs().mean())))
        optimiser.step()
    say("\nWhy is the gradient 0 at step 0? POT's last layer starts at exactly zero")
    say("(so untrained v2.B = zero-shot), and a zero layer passes no gradient back.")
    say("One step makes it nonzero and the layers before it wake up.")
    say("\nTo FINE-TUNE part of a pretrained network, unfreeze just that part:")
    say("   for p in brain.vision.it_blocks[-1].parameters(): p.requires_grad = True")
    say("and give only those to the optimiser, with a small rate (1e-5).")


def lesson_6() -> None:
    heading("LESSON 6 -- save, load, and swap a whole network")
    import json
    import tempfile
    from pathlib import Path
    from v2b.cortex import Cortex, Dims
    from v2b.train import _path
    try:
        manifest = json.loads(_path("manifest.json").read_text())
    except FileNotFoundError:
        say("No feature cache yet -- run  python -m v2b.train  first.")
        return
    dims = Dims(**manifest["dims"])
    a, b = Cortex(dims, manifest["scale"]), Cortex(dims, manifest["scale"])
    path = Path(tempfile.mkdtemp()) / "cortex.pt"
    say("Weights are a dictionary: name -> tensor. Save writes it; load copies")
    say("every tensor back BY NAME.\n")
    state = a.state_dict()
    say("   %d named tensors, e.g. %s" % (len(state), ", ".join(list(state)[:3])))
    torch.save(state, path)
    same_before = torch.equal(a.pot.ask.weight, b.pot.ask.weight)
    b.load_state_dict(torch.load(path))
    say("   two random cortices match? before load: %s   after: %s"
        % (same_before, torch.equal(a.pot.ask.weight, b.pot.ask.weight)))
    say("\nSWAP A NETWORK: one line in v2b/config_v2b.py, then retrain the small parts:")
    say("   VISION_MODEL   = '%s'" % C.VISION_MODEL)
    say("   PARIETAL_MODEL = '%s'" % C.PARIETAL_MODEL)
    say("   AUDITORY_MODEL = '%s'" % C.AUDITORY_MODEL)
    say("   python -m v2b.train --fresh")
    say("Widths are read from the loaded model, so nothing else changes. The")
    say("checkpoint records which networks it was trained with and refuses to")
    say("load into different ones -- their outputs would mean different things.")


def lesson_7() -> None:
    heading("LESSON 7 -- memory: retrieval-augmented recognition (RAG)")
    brain = _brain()
    names = brain.frontal.names
    bank, _ = brain.hippocampus.size
    say("RAG = before answering, RETRIEVE similar stored examples and let them")
    say("vote. v2.B's store holds every training example (%s of them) as the" % f"{bank:,}")
    say("vector POT sent frontal; ENTORHINAL maps it to a memory key, and the")
    say("search is one matrix product (faiss).\n")
    if not bank:
        say("No consolidated memory yet -- run  python -m v2b.train  first.")
        return
    picture, truth = _picture(cls=90, which=1)                # a train, from the TEST set
    thought = brain.perceive(image=picture)
    say("   a TEST picture of a %s (never stored) recalls:" % truth)
    for sim, label, weight, what in thought.recalled[:8]:
        say("        %-14s similarity %.3f   (%s)" % (label, sim, what[1]))
    say("\n   CA1 vote (sharpness %.1f): %s" % (float(brain.ca1.sharpness.detach()), ", ".join(
        "%s %.0f%%" % (n, 100 * p) for n, p in thought.memory_top[:3])))
    say("   cortex alone:            %s" % ", ".join(
        "%s %.0f%%" % (n, 100 * p) for n, p in thought.cortex_top[:3]))
    say("   CA1 familiarity %.2f (%s): how much memory is allowed to vote"
        % (thought.familiarity, "familiar" if thought.familiarity >= 0.5 else "novel"))
    say("   final = cortex + %.2f trust x %.2f familiarity x memory: %s"
        % (thought.trust, thought.familiarity, ", ".join(
        "%s %.0f%%" % (n, 100 * p) for n, p in thought.top[:3])))
    say("   amygdala value %.2f -> %s" % (thought.value,
                                         "not sure" if thought.abstained else "answers"))
    say("\nTEACHING it adds a memory -- no training. Correct it once:")
    brain.correct(thought, "my model train")
    again = brain.perceive(image=picture)
    say("   same picture now: '%s'" % again.answer)
    other, _ = _picture(cls=90, which=0)
    say("   a different train picture: '%s' (one memory must not relabel a class)"
        % brain.perceive(image=other).answer)


LESSONS = {"1": lesson_1, "2": lesson_2, "3": lesson_3, "4": lesson_4,
           "5": lesson_5, "6": lesson_6, "7": lesson_7}


def main() -> int:
    if len(sys.argv) < 2 or sys.argv[1] not in LESSONS:
        print(__doc__)
        return 0
    LESSONS[sys.argv[1]]()
    return 0


if __name__ == "__main__":
    sys.exit(main())
