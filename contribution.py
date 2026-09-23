"""
contribution.py -- does each of the four lobes actually do any work?

WHY THIS TOOL EXISTS, given that ablate.py already sweeps lesions.

ablate.py measures DISPLACEMENT: how far the settled cortical state moves when
a pathway is cut. That is a real number and it is not the number that matters.
Cutting the arcuate fasciculus moves the state by 134% and costs 0.0 accuracy
points -- the system reorganises around the wound and answers exactly as well
as before. Displacement says a part is connected. It does not say a part is
USED.

So this asks the harder question directly, in the only currency that cannot be
gamed: held-out accuracy on the real tasks.

Two different questions, and they can disagree:

  READING    Zero one lobe's slice of the readout and re-measure. This asks
             whether the answer depends on what that lobe concluded. A lobe
             that can be blanked at the readout with no cost is not
             contributing to the output, however busy its state looks.

  PARSING    Withhold the sensory sequence from one lobe and re-measure. This
             asks whether that lobe's own reading of the INPUT matters, as
             opposed to it merely relaying what arrives over the tracts. Until
             the routing table was fixed, three of the four lobes scored zero
             here on a text input for the simple reason that they never
             received the input at all.

A lobe that costs nothing on both is decoration, and this tool is written to
be capable of saying so. If that is the answer, it is the answer; the point of
building the measurement is that it can come out badly.

    python contribution.py
    python contribution.py --modality text --batch 64
"""

from __future__ import annotations

import argparse

import torch

import config


def accuracy(brain, modality: str, streams, batch: int,
             drop_readout: str | None = None,
             drop_sequence: str | None = None,
             limit: int | None = None) -> float:
    """Held-out accuracy, optionally with one lobe silenced.

    `drop_readout` zeroes that lobe's slice of the concatenated readout, so
    the head sees the lobe as reporting nothing. `drop_sequence` withholds the
    sensory sequence from that lobe, so it can still receive tract input but
    never reads the input itself.
    """
    from train_v2 import readout

    lobes = tuple(brain.cortex.LOBES)
    tokens, labels = streams[modality]["test"]
    if limit:
        tokens, labels = tokens[:limit], labels[:limit]

    device = next(brain.parameters()).device
    hits = total = 0
    with torch.no_grad():
        for start in range(0, max(len(labels) - batch, 0), batch):
            chunk = tokens[start:start + batch].float().to(device)
            target = labels[start:start + batch].to(device)

            projected = brain.project[modality](chunk)
            nucleus, destinations = brain.ROUTES[modality]
            sensory = {"thal.%s" % nucleus: projected.mean(dim=1)}
            sequences = {lobe: projected for lobe in destinations
                         if lobe != drop_sequence}
            trace = brain.cortex.settle(sensory, sequences=sequences)

            state = readout(trace)
            if drop_readout is not None:
                #  Zero exactly this lobe's slice, leaving the head's input
                #  the same SHAPE. Dropping the columns instead would change
                #  the head's geometry and measure a different model.
                width = state.shape[-1] // len(lobes)
                index = lobes.index(drop_readout)
                state = state.clone()
                state[..., index * width:(index + 1) * width] = 0.0

            logits = brain.heads[modality](state)
            hits += int((logits.argmax(dim=-1) == target).sum())
            total += len(target)
    return hits / total if total else 0.0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--modality", default=None,
                        help="text or image; default measures both")
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--limit", type=int, default=2000,
                        help="held-out examples per measurement")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    import train_v2

    device = args.device or config.resolve_device()
    print("=" * 72)
    print("DOES EACH LOBE DO ANY WORK?")
    print("=" * 72)
    print("device %s   %d held-out examples per cell" % (device, args.limit))
    print()

    #  Same construction the trainer uses, so the heads have the same shape
    #  the checkpoint was written with. Reading the class count out of the
    #  labels rather than assuming it keeps this tool correct if the datasets
    #  are ever rebuilt at a different size.
    streams = train_v2.load_streams(0, 0, rebuild=False)
    modalities = [m for m in ("image", "text", "audio", "sensor")
                  if m in streams and "train" in streams[m]]
    classes = {m: int(streams[m]["train"][1].max()) + 1 for m in modalities}
    brain = train_v2.BrainV2(classes).to(device)

    state = torch.load(train_v2.CHECKPOINT, map_location=device,
                       weights_only=False)
    brain.cortex.load_state_dict(state["cortex"])
    brain.heads.load_state_dict(state["heads"])
    brain.project.load_state_dict(state["project"])
    brain.eval()

    record = state.get("training_record") or {}
    print("checkpoint: %s" % ("  ".join("%s=%s" % kv
                                        for kv in sorted(record.items()))
                              or "NO TRAINING RECORD -- weights are random"))
    print()

    wanted = ([args.modality] if args.modality
              else [m for m in ("image", "text") if m in modalities])

    for modality in wanted:
        intact = accuracy(brain, modality, streams, args.batch,
                          limit=args.limit)
        chance = 100.0 / classes[modality]
        print("-" * 72)
        print("%s   intact %.1f%%   (chance %.1f%%)"
              % (modality.upper(), 100 * intact, chance))
        print("-" * 72)
        print("%-11s %14s %14s" % ("lobe", "reading", "parsing"))
        print("%-11s %14s %14s" % ("", "blanked", "starved"))

        for lobe in brain.cortex.LOBES:
            blanked = accuracy(brain, modality, streams, args.batch,
                               drop_readout=lobe, limit=args.limit)
            starved = accuracy(brain, modality, streams, args.batch,
                               drop_sequence=lobe, limit=args.limit)
            print("%-11s %+13.1f %+13.1f"
                  % (lobe, 100 * (blanked - intact), 100 * (starved - intact)))

        print()
        print("   Negative means removing the lobe HURT, which is the lobe")
        print("   earning its place. A row of zeros is a lobe the answer does")
        print("   not depend on -- report it, do not explain it away.")
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
