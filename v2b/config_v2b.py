"""
config_v2b.py -- every constant v2.B uses, and nothing else.

v2.B is self-contained: it imports nothing from v1, v2 or v3 and changes
nothing outside `v2b/`. If a number matters it lives here.

THE ARCHITECTURE (README.md has the picture):

  input: image / audio / video (-> sampled frames + soundtrack)
  thalamus    pulvinar, LGN, MGN: tiny gate networks, and frontal can turn
              each one up or down (corticothalamic feedback via the TRN)
  VENTRAL     one vision network spans the whole stream, as the anatomy does:
              blocks 1-6 = OCCIPITAL (V1 -> V4), blocks 7-12 + head =
              INFEROTEMPORAL cortex (IT, temporal lobe) -- where "what is it"
              is actually settled in a primate brain
  DORSAL      occipital's V4-level grid + PARIETAL (a depth network)
  AUDITORY    auditory cortex in the TEMPORAL lobe (a sound network)
  STS         sight and sound meet (superior temporal sulcus)
  POT         the junction that fuses everything, in a loop with FRONTAL
  frontal     a text network that holds the names it can answer with
  hippocampus retrieval memory (RAG); amygdala: learned value (good / bad)
  motor       the answer, or "not sure", to the dashboard
"""

from __future__ import annotations

from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parent.parent
V2B = Path(__file__).resolve().parent
CIFAR_DIR = ROOT / "data" / "cifar-100-python"
ESC50_DIR = ROOT / "data" / "ESC-50-master"
IMAGENETTE_DIR = ROOT / "data" / "imagenette2-160"      # open-vocabulary test only
CACHE_DIR = ROOT / "data" / "v2b_cache"
CHECKPOINT = ROOT / "checkpoints" / "v2b.pt"            # small trained parts
MEMORY_BANK = ROOT / "checkpoints" / "v2b_memory.npz"   # consolidated memories
MEMORY_DIR = V2B / "memory"                             # what YOU taught it

SEED = 7

# ---------------------------------------------------------------------------
# The pretrained networks (frozen, open-weight)
# ---------------------------------------------------------------------------
#  Swapping one is a one-line change here plus `python -m v2b.train --fresh`.
#  Sizes are read from the loaded model, never assumed, so a swap cannot
#  leave a stale width behind. Supported families, and their licences:
#    vision + frontal:  openai/clip-*  (MIT)   google/siglip2-*  (Apache-2.0)
#    parietal:          depth-anything/*-Small-hf (Apache-2.0; Base/Large are
#                       CC-BY-NC, which is why Small)
#    auditory:          openai/whisper-*  (MIT)
#                       MIT/ast-finetuned-audioset-*  (BSD-3)
#
#  Measured 2026-09-27 on this machine's connection: Hugging Face delivered
#  ~54 KB/s, so SigLIP 2 (1.5 GB) and AST (350 MB) are downloading in the
#  background. CLIP and Whisper are what is on disk today. The research
#  behind the upgrade: SigLIP 2 ViT-B/16 reaches 79.1% zero-shot ImageNet vs
#  ~63% for CLIP ViT-B/32; AST reaches 95.7% on ESC-50 (BEATs 98.1% is better
#  but ships outside Hugging Face).

#  2026-09-27, second upgrade. Vision: SigLIP 2 So400m (~1B params, Apache-2.0),
#  the larger member of the family the base model came from. Sound: LAION CLAP
#  larger_clap_general (Apache-2.0), 90.5% ZERO-SHOT on ESC-50 -- it can name a
#  sound from words alone, as SigLIP names pictures, where AST only knew the
#  labels it was trained on.
VISION_MODEL = "google/siglip2-so400m-patch14-224"   # ventral stream + frontal
PARIETAL_MODEL = "depth-anything/Depth-Anything-V2-Small-hf"
AUDITORY_MODEL = "laion/larger_clap_general"

#  WHERE OCCIPITAL ENDS AND INFEROTEMPORAL BEGINS in a 12-block vision
#  network. The first half is taken as V1 -> V4 (occipital), the second half
#  plus the pooling head as IT (temporal lobe). Not a measured boundary -- a
#  12-block network has no labelled V4 -- but it puts object identity where
#  the anatomy puts it, and hands the dorsal stream mid-level features rather
#  than a finished answer, which is what V4 -> parietal carries.
OCCIPITAL_SHARE = 0.5            # first half of the blocks: occipital

#  One retina for all visual lobes: every grid cell (i, j) is the same patch
#  of the world in occipital, IT and parietal.
RETINA_SIZE = 224
GRID = 7                         # every visual grid is pooled to 7x7
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
DEPTH_STATS = 4                  # per cell: nearness, spread, slope x, slope y

#  Sound is captured at 48 kHz, CLAP's own rate; a network wanting less
#  (AST, Whisper: 16 kHz) is handed a resampled copy. Capturing at 16 kHz and
#  upsampling for CLAP would throw away everything above 8 kHz first.
AUDIO_SAMPLE_RATE = 48_000
AUDIO_MAX_SECONDS = 10
TEMPORAL_BINS = 16               # sound is handed on as 16 slices of time
VIDEO_FRAMES = 8

# ---------------------------------------------------------------------------
# Thalamus
# ---------------------------------------------------------------------------

IMAGE_STATS = 8
GATE_HIDDEN = 16
GATE_OPEN_BIAS = 2.0             # sigmoid(2) = 0.88: relays start mostly open
#  Corticothalamic feedback (frontal -> TRN -> relay) can scale each relay
#  between 0 and 2x its own gain. It starts at exactly 1x (zero weights), so
#  it changes nothing until training finds a use for it.
TRN_RANGE = 2.0

# ---------------------------------------------------------------------------
# Streams, POT, and the frontal loop
# ---------------------------------------------------------------------------

STREAM_DIM = 256
STREAM_HEADS = 4
FRONTAL_PASSES = 3
FRONTAL_EXPECTATION = 0.5        # share of the next query that is frontal's guess
FRONTAL_TOP_K = 5
IMAGE_TEMPLATE = "a photo of a {}."
SOUND_TEMPLATE = "the sound of {}."
#  THE VOCABULARY is what frontal can answer with. It used to be the 150 names
#  v2.B trained on, so a screenshot of a web page got "not sure -- maybe
#  table": correct caution, useless answer. Now it also holds every ImageNet
#  class (1,000, incl. "web site", "monitor") and every AudioSet sound class
#  (527). Names it never trained on are recognised zero-shot.
IMAGE_VOCABULARY = ("cifar100", "imagenet")
SOUND_VOCABULARY = ("esc50", "audioset")

# ---------------------------------------------------------------------------
# Hippocampus (RAG) and amygdala (value)
# ---------------------------------------------------------------------------

KEY_DIM = 256                    # entorhinal key width
MEMORY_K = 32                    # memories recalled per percept
MEMORY_SHARPNESS = 20.0          # CA1 starting temperature (then trained)
MEMORY_FLOOR = 1e-4              # smallest memory probability (log of it is the floor)
#  EPISODIC memory -- what YOU taught it -- has its own path, separate from
#  the consolidated RAG vote. It had to: the trust CA1 learns for the
#  consolidated vote came out at 0.031 (memory of the training set is
#  redundant with a cortex trained on the same set), and routed through that,
#  a correction you typed moved frontal's scores by under 0.3 logits -- it
#  could never win. An episode instead votes directly, fading linearly from
#  EPISODE_VOTE logits at cosine 1.0 (the same thing again) to nothing at
#  EPISODE_SIMILARITY.
EPISODE_SIMILARITY = 0.80        # fallback only: train.py sets it per model, below
#  A taught episode must fire for THIS input (or a near-copy), not for every
#  picture of its class. A fixed 0.80 could not tell them apart: two
#  different same-class validation pictures cleared it 21% of the time with
#  round 5's cortex and 72% with round 6's (label smoothing pulls classes
#  together; median 0.70 -> 0.88), so teaching one bear renamed other bears.
#  train.py sets the bar at this quantile of same-class, different-item
#  similarity on validation, and stores it in the checkpoint.
EPISODE_QUANTILE = 0.995
EPISODE_MAX = 0.99
EPISODE_VOTE = 40.0              # a good episode's pull toward its label
AMYGDALA_PENALTY = 40.0          # a bad episode's push away from its label
SCRATCH_EPISODES = 200           # unrated episodes fade after this many
VALUE_HIDDEN = 32
ABSTAIN_BELOW = 0.5              # amygdala value under this -> "not sure"
#  With every thalamic relay shut there is no percept to name. Found by the
#  lessons: with the MGN gate forced shut, v2.B still answered "car horn" and
#  the amygdala (never trained on silence) rated it 0.66. No drive, no answer.
MIN_DRIVE = 0.05

# ---------------------------------------------------------------------------
# Training -- only the small parts learn
# ---------------------------------------------------------------------------

TRAIN_PER_CLASS = 250            # CIFAR-100 images per class ("train" slice)
VAL_PER_CLASS = 50               # held out from CIFAR train, to fit memory + value
EXTRA_PER_CLASS = 200            # the rest of CIFAR train ("extra" slice)
#  MORE DATA (2026-09-29). v2.B's trained parts had learned from 25,000 blurry
#  32x32 CIFAR pictures of 150 names. Now they learn from all of CIFAR's
#  training pictures (the "extra" slice was memory-only before) plus
#  Tiny-ImageNet: 200 ImageNet classes at 64x64 -- twice the resolution, all
#  already in the vocabulary, none of them an Imagenette test class.
TRAIN_ON_EXTRA = True
TRAIN_ON_TINY = True             # False reproduces the CIFAR-only recipe, for comparison
TINY_TRAIN_PER_CLASS = 450       # Tiny-ImageNet train: 450 train + 50 val per class
TINY_VAL_PER_CLASS = 50          # (its own 10,000-picture validation set is the TEST)
#  MORE DIVERSE DATA (2026-10-04): real photos. CIFAR (32 px) and Tiny (64 px)
#  covered 290 picture names; the vocabulary holds all 1,000 ImageNet classes.
#  These are the ~790 ImageNet classes v2.B had never trained on, as 256 px
#  photos -- Tiny's 200 are left out (its test photos come from ImageNet) and
#  so are Imagenette's 10 (the never-trained test). ImageNet terms:
#  non-commercial research, like Tiny-ImageNet.
INET_REPO = "benjamin-paine/imagenet-1k-256x256"
INET_DIR = ROOT / "data" / "imagenet_subset"
INET_TRAIN_PER_CLASS = 48        # first 54 photos of each class, fetched by row group:
INET_VAL_PER_CLASS = 6           #   48 to train on, 6 to validate on
INET_TEST_PER_CLASS = 10         # from ImageNet's own validation photos: the TEST
TRAIN_ON_INET = True
#  ROUND 5 (2026-10-05): the next photos of each class, as a split of their own
#  (inet_more) so nothing already encoded is redone. 8, not more: D: had 3.9 GB
#  free (the page file had grown), and 8 x 782 photos are ~1.7 GB of cache.
INET_MORE_PER_CLASS = 8
#  ImageNet classes left out besides Tiny's and Imagenette's: TWINS of a
#  held-out class, which would steal its test pictures (a review found them
#  by name, 2026-10-04) -- 848 tape player (Imagenette's cassette player),
#  836 sunglass, 657 missile, 876 tub, 282 tiger cat, 898 water bottle (Tiny's
#  sunglasses, projectile, bathtub, tabby/Egyptian cat, water jug) -- and 673
#  mouse, the COMPUTER mouse, which the merged vocabulary files under CIFAR's
#  rodent. 218 Welsh springer stays: a real breed, not a twin.
INET_SKIP = (848, 836, 657, 876, 282, 898, 673)
#  MORE SOUNDS (2026-10-04). Sounds trained on 1,200 ESC-50 clips of 50
#  names; the vocabulary holds 554. AudioSet's balanced set (YouTube, labels
#  CC-BY-4.0, audio research-use like every AudioSet mirror) covers ~480 of
#  them. A survey + adversarial check chose it (and FSD50K, whose clips share
#  119 freesound recordings with ESC-50's test fold -- removed by ID).
AUDIOSET_REPO = "danjacobellis/audioset_opus_24kbps_balanced"
ONTOLOGY_REPO = "agkphysics/AudioSet"
FSD50K_REPO = "philgzl/fsd50k"   # freesound, CC-BY-4.0 (per-clip CC licences)
SOUND_SECONDS = 10               # FSD50K clips run to 30 s: the first 10, as AudioSet's
MORE_AUDIO_SHARE = 40            # new sounds per training batch, beside ESC-50's AUDIO_SHARE
SOUND_NEAR_COPY = 0.95
#  RARE SOUNDS (round 5): AudioSet's unbalanced set, only for names with fewer
#  than SOUND_EXTRA_UPTO training clips. Two of its 96 files: C: had 3.9 GB
#  free. It cannot help ESC-50's "washing machine" -- AudioSet has no such
#  class -- and much of it is music (genres and instruments are rare names).
AUDIOSET_EXTRA_REPO = "danjacobellis/audioset_opus_24kbps"
AUDIOSET_EXTRA_FILES = ("data/part_00-00000-of-00001.parquet",
                        "data/part_01-00000-of-00001.parquet")
SOUND_EXTRA_UPTO = 300
#  Refuse to start below this much free space on the cache drive: it shares
#  D: with the page file, which grew by GBs under memory pressure (2026-10-05).
DISK_HEADROOM_GB = 1.5           # CLAP cosine at which a training clip counts as a test clip's copy
TRAIN_ON_MORE_SOUNDS = True
SOUND_CAP = 600                  # training clips per name: no name drowns the rest
SOUND_BROAD_CAP = 300            # ...and fewer for catch-alls (music, speech, ...)
SOUND_BROAD = ("music", "speech", "musical instrument", "bowed string instrument",
               "wind instrument", "plucked string instrument", "percussion")
#  The same sound under AudioSet's word and ESC-50's: trained as two names, a
#  test clip of "insects" answered "insect" would be scored wrong while the
#  model was right. Only unambiguous pairs; AudioSet's word -> ESC-50's.
SOUND_SYNONYMS = {
    "insect": "insects", "cattle": "cow", "moo": "cow", "oink": "pig", "caw": "crow",
    "croak": "frog", "crowing": "rooster", "chirp": "chirping birds", "waves": "sea waves",
    "drip": "water drops", "fire": "crackling fire", "computer keyboard": "keyboard typing",
    "knock": "door wood knock", "shatter": "glass breaking", "baby cry": "crying baby",
    "cough": "coughing", "sneeze": "sneezing", "laughter": "laughing", "walk": "footsteps",
    "toothbrush": "brushing teeth", "creak": "door wood creaks", "cluck": "hen", "meow": "cat",
    "bark": "dog", "bow-wow": "dog", "yip": "dog", "thunder": "thunderstorm",
    "vehicle horn": "car horn", "sawing": "hand saw", "church bell": "church bells",
    "pour": "pouring water", "cricket": "crickets", "tick": "clock tick",
    "tick-tock": "clock tick", "alarm clock": "clock alarm",
    #  AudioSet's "Fixed-wing aircraft, airplane" keeps only its first phrase.
    "fixed-wing aircraft": "airplane"}
#  The labelling rules' version: part of the cache stamp, so a change to
#  one_name() or the FSD50K filter re-encodes the sound splits (a review found
#  the stamp ignored the code, and fixes would silently reuse old labels).
#  rules:2 -- ESC-50 names beat their own subtypes (sheep had 0 clips, siren 25
#  against 123 under police car etc.); recording-condition tags give way;
#  deterministic tie-breaks; FSD50K val/test drop every ESC-50 recording.
SOUND_RULES = "rules:2"
SOUND_VAL = 2000                 # from each set's held-out part: validation
SOUND_TEST = 4000                # ...and test (the rest is not encoded)
AUDIO_TONES = (480, 160, 160)    # synthetic tones (train, val, test) if no ESC-50
ESC50_FOLDS = ((1, 2, 3), (4,), (5,))   # train, val, test folds
BATCH = 256
#  Sounds per training batch. With 135,000 pictures an epoch is ~600 batches;
#  at 32 sounds each the 1,200 training clips would be seen 16 times per
#  epoch. 8 keeps it near 4.
AUDIO_SHARE = 8
#  6 epochs flattened by epoch 4 on the old data (validation pictures 84.4,
#  87.0, 87.9, 88.6, 88.7, 88.7) while training loss kept falling (0.27 ->
#  0.12): more epochs alone would only memorise. With ~790 new classes there
#  is more to learn, and train() keeps the epoch best on validation, so the
#  extra epochs cannot make the result worse.
#  16 (round 5): in both 12-epoch runs the best epoch was the last or one
#  before it, so validation had not clearly stopped rising.
EPOCHS = 16
#  REGULARISATION (round 6). Round 5 kept epoch 5 of 16: past it the training
#  loss fell to 0.08 while validation stayed flat -- memorising the cached
#  features, which are identical every epoch. Label smoothing stops the
#  scores chasing 100% on training labels; dropping 10% of each input grid
#  (occipital, IT, parietal, sound slices) makes every epoch's copy of a
#  picture a little different. The identity vectors are left whole: they are
#  the zero-shot path.
LABEL_SMOOTHING = 0.1
FEATURE_DROPOUT = 0.1
LR = 3e-4
WEIGHT_DECAY = 0.01
PASS_WEIGHTS = (0.25, 0.5, 1.0)
LOGIT_SCALE = 100.0              # CLIP's temperature; read from the model if it has one
#  HOW MUCH OF POT'S LEARNED CORRECTION TO KEEP. Trained at full strength,
#  the correction specialised v2.B to its 150 training names and wrecked the
#  other 1,479: pictures of never-trained classes fell from 79.5% (zero-shot)
#  to 29.6%. Scaling the correction showed why -- right direction, far too
#  strong: at 0.3 of its strength CIFAR kept 86.8% (vs 90.2%) and the open
#  vocabulary 80.6%. (Blending toward the zero-shot model is WiSE-FT,
#  Wortsman et al. 2022.) The strength is chosen WITHOUT the open-vocabulary
#  test: the weakest correction within KEEP_TOLERANCE points of full strength
#  on the VALIDATION sets. Retrained on 340 names with Tiny-ImageNet
#  (2026-09-29), full strength fell to 33.4% on them; 0.40 was kept again,
#  and cortex scored 79.3%.
KEEP_GRID = (0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.75, 1.0)
KEEP_TOLERANCE = 3.0
#  EXPERTISE. With ImageNet photos ~1,070 of the 1,075 picture names are
#  trained, and the rule above (made to protect ~1,290 untrained names) kept
#  0.30 -- which cost every trained class ~3 validation points (0.30 -> 85.0%,
#  0.60 -> 88.0%). The cautious keep stays for answers the hippocampus has no
#  memories of; an answer it does hold memories of is re-read at a stronger
#  keep, chosen on validation (hippocampus->POT, cuttable). Measured on the
#  trained checkpoint first (2026-10-04), trust refitted on validation:
#  pooled validation 87.46 -> 87.71, never-trained 78.5 -> 81.9 (a dev set
#  by now), no test worse. A stronger keep for EVERY answer scored higher on
#  validation (88.23) but cost the never-trained classes 2.8 points.
EXPERTISE = True
MEMORY_EPOCHS = 4
VALUE_EPOCHS = 200
ENCODE_BATCH = 48               # So400m returns 28 layers of activations per image

# ---------------------------------------------------------------------------
# Dashboard -- the v2 console's look, kept
# ---------------------------------------------------------------------------

UI_BG = "#05070a"
UI_PANEL = "#080b10"
UI_EDGE_IDLE = "#1b2530"
UI_DIM = "#3d4a5a"
UI_MID = "#7e93a8"
UI_FG = "#d8e4f0"
UI_ACCENT = "#ff9020"
UI_ACCENT_DIM = "#6b4413"
UI_ALERT = "#ff4a28"
UI_COOL = "#4fd8ff"
UI_GOOD = "#5fd068"
UI_NEG = "#3a8dde"               # negative weights in heatmaps
UI_FONT_SMALL = ("Consolas", 10)
UI_FONT_TINY = ("Consolas", 9)
UI_FONT_TITLE = ("Consolas", 13, "bold")
UI_FONT_LABEL = ("Consolas", 10, "bold")
UI_FONT_DATA = ("Consolas", 9, "bold")
UI_REFRESH_MS = 100
UI_PASS_MS = 700
#  The RAG tab lights its 8 stages one after another, so the eye can follow
#  the signal from POT to motor; 8 x 220 ms is about the frontal replay's length.
UI_RAG_STAGE_MS = 220
UI_MIN_WIDTH = 1360
UI_MIN_HEIGHT = 820
