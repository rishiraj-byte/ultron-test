"""
guide.py -- the GUIDE tab's text: for every region, what it does in a real
brain, what v2.B puts there, and where the two differ. Plain words, kept
next to the code so a change to one is a reminder to change the other.
"""

from __future__ import annotations

OVERVIEW = """\
HOW v2.B WORKS, ONE PERCEPT

 1  INPUT      an image, a sound, or a video (split into frames + soundtrack)
 2  THALAMUS   routes each sense to its cortex and sets how strongly to pass it
 3  VENTRAL    one vision network: occipital (V1->V4) then inferotemporal (IT),
               which decides WHAT the thing is
 4  DORSAL     V4's grid + parietal's depth map: WHERE things are, how far
 5  TEMPORAL   its auditory cortex (CLAP) hears the sound; its lexicon (Wernicke's
               area) names it and sends those scores to frontal over the arcuate
               fasciculus. (IT, step 3, is in the temporal lobe too.)
 6  STS        sight (from IT) and sound (from temporal) meet
 7  POT        fuses dorsal + STS into one vector for frontal
 8  FRONTAL    compares that vector with every name it knows, forms a guess,
               and sends the guess back to POT -- and to the thalamus. 3 passes.
 9  MEMORY     hippocampus queries its RAG DATABASE (237,001 stored examples)
               for the 32 most similar; they vote. CA1 sets how much the vote
               counts. Episodes you taught it match directly.
10  AMYGDALA   hears cortex's answer AND the hippocampus's recall directly,
               and decides GO or HOLD, with a confidence score (0-1).
11  MOTOR      GO: the answer goes to you. HOLD: "not sure -- maybe X".
               You press GOOD/BAD or type the right name, and it is
               remembered -- no retraining.

WHAT LEARNS: only the small parts (thalamus, dorsal, STS, POT, entorhinal,
CA1, amygdala). The five pretrained networks are frozen.

Click any box in ROUTING to focus it; ACTIVITY, WEIGHTS and this GUIDE follow.
"""

REGIONS = {
    "input": ("The senses: retina and cochlea.",
              "A picture is cropped once to 224x224 so every visual area sees the same "
              "pixels. A sound is resampled to 48 kHz (CLAP's rate). A video is 8 evenly spaced frames "
              "(averaged) plus its soundtrack.",
              "No eye movements, no motion (area MT is not modelled), no binaural hearing."),
    "pulvinar": ("A thalamic nucleus with strong links to posterior parietal cortex; it "
                 "carries visual signals that bypass V1 and helps steer attention.",
                 "A tiny network (161 weights): cheap image statistics in, a gain out. "
                 "Frontal can scale the gain on each pass (via the TRN).",
                 "Real pulvinar is also driven heavily by cortex itself; here it only "
                 "sees the raw image statistics and frontal's feedback."),
    "LGN": ("Lateral geniculate nucleus: the relay from the eyes to primary visual "
            "cortex (V1). About 90% of the optic nerve ends here.",
            "A tiny gate network: image statistics in, a gain (0-1) out, scaled by "
            "frontal's feedback. Routing is fixed: eyes -> LGN -> occipital.",
            "Real LGN has layers (magno/parvo/konio) and gets more input FROM cortex "
            "than from the eyes. Here it is one number per percept."),
    "MGN": ("Medial geniculate nucleus: the thalamic relay for hearing, to primary "
            "auditory cortex (A1).",
            "A tiny gate network on the spectrum's statistics, scaled by frontal.",
            "Same simplification as the LGN: one gain, no tonotopic layers."),
    "thalamus": ("The thalamic reticular nucleus (TRN) wraps the thalamus and lets "
                 "cortex (layer 6) turn each relay up or down -- attention at the gate.",
                 "A linear map from frontal's current guess to a 0-2x factor per relay, "
                 "starting at exactly 1x. Its measured contribution is in the README.",
                 "Real corticothalamic feedback is massive and topographic; this is "
                 "three numbers per pass."),
    "occipital": ("Occipital lobe: V1 -> V2 -> V4. Edges, then textures and shapes, "
                  "then object parts. Retinotopic: neighbouring neurons see neighbouring "
                  "places.",
                  "The first 13 of 27 blocks of SigLIP 2 So400m's vision network (Google, "
                  "Apache-2.0). Its 16x16 patch grid is retinotopic in the same sense "
                  "(pooled to 7x7 for the circuit).",
                  "Where V4 'ends' inside a transformer is a choice (half way, block 13), "
                  "not a measurement. Real V1 is not a transformer and learned from your life."),
    "ventral": ("The VENTRAL STREAM, the 'what' pathway: from occipital (V1 -> V4) into "
                "inferotemporal cortex (IT), in the temporal lobe, where it ends. "
                "Inferotemporal cortex, in the temporal lobe: the end of the ventral 'what' "
           "stream. Neurons here respond to whole objects and faces.",
           "Blocks 14-27 of the same SigLIP 2 network plus its pooling head. Its output "
           "lives in the same space as the names frontal holds, so it can be named "
           "with no training at all.",
           "SigLIP 2 learned from image-text pairs on the web, so its 'objects' are "
           "shaped by captions, not by handling things."),
    "parietal": ("Posterior parietal cortex: the dorsal 'where/how' stream -- location, "
                 "depth, and planning actions toward things.",
                 "Depth Anything V2 Small (Apache-2.0): a depth map, read into nearness, "
                 "spread and surface slope for each of 7x7 places, plus its backbone's "
                 "features.",
                 "Measured with earlier networks: on naming objects, its help came from its "
                 "backbone's 'what' features, not from depth. Real parietal also does touch, body "
                 "position and reaching; none of that is here."),
    "dorsal": ("The dorsal stream proper: V4/MT -> posterior parietal. Combines what "
               "is seen with where it is.",
               "A small trained network: for each of 49 places, occipital's V4 features "
               "+ parietal's depth -> one token.",
               "No motion input (MT), no eye position, no action output."),
    "temporal": ("The temporal lobe's auditory cortex, on the superior temporal gyrus "
                 "(A1 + belt), and Wernicke's area behind it, where heard sounds meet "
                 "their meanings. (IT, the end of the ventral stream, is in the temporal "
                 "lobe too -- it has its own box.)",
                 "LAION CLAP (Apache-2.0): it hears the spectrogram and places the sound "
                 "in a space shared with descriptions of sounds. Its LEXICON -- every "
                 "sound name as a CLAP vector -- names the sound with no training, and "
                 "those scores go straight to frontal over the arcuate fasciculus. It "
                 "also sends 16 slices of time to the STS.",
                 "One ear, no pitch tracking, no speech pathway. The lexicon lives in "
                 "CLAP's space because a map into frontal's space failed (17.5% vs 89.8%)."),
    "STS": ("Superior temporal sulcus: where sight and sound of the same event come "
            "together (a voice with a face, a bark with a dog).",
            "One trained attention layer over 67 tokens: IT's identity and 7x7 grid, "
            "and the sound's identity and 16 time slices.",
            "Real STS integrates over time and binds by synchrony; here the sight is a "
            "still (or frame average)."),
    "POT": ("The temporo-parieto-occipital junction: association cortex where the "
            "streams converge into one percept.",
            "A trained attention readout: frontal's question picks what to attend to "
            "among dorsal and STS tokens; out comes one vector for frontal. Its last "
            "layer starts at zero, so untrained it passes IT straight through.",
            "One junction instead of a web of association areas."),
    "frontal": ("Prefrontal and inferior frontal cortex: choosing, naming, holding a "
                "guess in mind, and sending expectations back down.",
                "SigLIP 2's text network turns every name into a vector once; frontal "
                "scores POT's vector against them, holds its top 5 guesses, and sends "
                "that expectation back to POT and the thalamus. Any new word can be "
                "added instantly.",
                "Real frontal plans, decides and produces language; here it only picks "
                "a name. The loop is 3 steps, not continuous."),
    "hippocampus": ("Stores episodes and recalls them by similarity (pattern "
                    "completion). Entorhinal cortex is its gateway; CA1/subiculum "
                    "send what is recalled back to cortex.",
                    "Retrieval-augmented recognition. ENTORHINAL: a trained projection "
                    "into memory keys. STORE: 237,001 examples -- every picture and sound "
                    "cortex trained on -- searched with faiss. CA1: the 32 "
                    "nearest vote, with a trained sharpness and a trained trust. "
                    "EPISODES you taught it vote directly when they match this same input. "
                    "CA1 also judges FAMILIARITY: the vote fades for inputs unlike anything "
                    "stored. Imperfectly: before ImageNet photos, a church (never "
                    "trained) found Tiny-ImageNet altars at 0.76, the gate stayed open and "
                    "memory flipped 'church' to 'altar' (-4.4 points on never-trained "
                    "classes). Now trust is 0.075 and memory barely moves an answer.",
                    "No consolidation over sleep, no forgetting, no replay. The trust CA1 "
                    "learned is small (0.075): memory of what cortex already knows adds "
                    "little on top of cortex -- its value is as a second opinion and as "
                    "the place new things are learned without training."),
    "amygdala": ("Assigns value -- good or bad, safe or threatening -- makes important "
                 "things memorable, and can stop an action (its central nucleus drives "
                 "freezing via the brainstem). It is directly and reciprocally wired to "
                 "the hippocampus.",
                 "A small trained network (289 weights). It hears cortex (via POT) and "
                 "the hippocampus's recall DIRECTLY, and outputs a confidence that the "
                 "answer is right. That becomes a verdict to motor: GO (say it) or HOLD "
                 "(say 'not sure'), below 0.5. Your BAD tags push an answer away for "
                 "(nearly) the same input.",
                 "Real amygdala reacts fast to raw input via the thalamus (the 'low "
                 "road'); that path is not modelled. Value here means 'likely "
                 "correct', which is closer to what orbitofrontal cortex tracks."),
    "motor": ("Motor and premotor cortex, in the frontal lobe: the output.",
              "Not a network: frontal's answer, gated by the amygdala's GO/HOLD.",
              "No speech, no hand -- a text line."),
}


def text(region: str) -> str:
    if region not in REGIONS:
        return OVERVIEW
    brain, model, gap = REGIONS[region]
    return ("%s\n\nIN THE BRAIN\n%s\n\nIN v2.B\n%s\n\nWHERE THEY DIFFER\n%s"
            % (region.upper(), brain, model, gap))
