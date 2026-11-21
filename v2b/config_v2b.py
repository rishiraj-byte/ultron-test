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
SOUND_NEAR_COPY = 0.95           # CLAP cosine at which a training clip counts as a test clip's copy
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
DISK_HEADROOM_GB = 1.5
#  8-BIT GRIDS (round 7). occ_grid and it_grid (49 places x 1,152 numbers per
#  picture) were 51.6 of the cache's 59 GB at 16 bits, with D: down to 1.7 GB
#  -- no room for more diverse data. Stored as int8 with one scale per place
#  they take half, read back as float32. One scale per place alone read back at
#  cosine 0.9974: SigLIP's channel 514 reaches 229 in occ_grid (median channel
#  peak 0.89) and takes the whole 8-bit range. Keeping each split's 16 largest
#  channels at 16 bits beside the int8 (+2.8%) gives 0.99999 (occ) / 0.9999 (it).
#  Even the 0.9974 version changed only 0.1% of the deployed model's answers on
#  validation (84.727% vs 84.719%), so QUANT_MIN_COSINE is a guard, not a tuning.
#  par_cells (Depth Anything, 49 x 388 per picture, 8.7 GB) joined them for
#  round 7's renditions: same scheme, worst row 0.99997 on three val splits.
CACHE_INT8 = True
QUANTIZED = ("occ_grid", "it_grid", "par_cells")
QUANT_KEEP = 16
QUANT_MIN_COSINE = 0.9995
#  ROUND 7: MORE DIVERSE TRAINING DATA, paid for by the 8-bit grids above.
#  (1) The next INET_MORE2_PER_CLASS photos of every ImageNet class, after the
#  48 + 6 + 8 already used: each class goes from 56 training photos to 144.
INET_MORE2_PER_CLASS = 88
#  (2) VGGSound (11hu83/vggsound: VGGSound's own test list, 309 labels x ~50
#  ten-second YouTube clips, CC-BY-4.0 annotations, audio at 16 kHz). Training
#  only -- validation and every test stay exactly as in round 6, so the two
#  rounds compare on identical numbers. Clips from a YouTube video that is in
#  any of our AudioSet splits are left out by id.
VGGSOUND_REPO = "11hu83/vggsound"
#  ROUND 8: trained without it. Round 7 with VGGSound: validation sounds 54.59 ->
#  53.14, ESC-50 87.5 -> 86.75, AudioSet 35.2 -> 34.6, FSD50K 65.8 -> 64.0 -- it
#  helped no sound test. The encoded split stays in the cache.
TRAIN_ON_VGGSOUND = False
VGGSOUND_CACHE = ROOT.parent / "hf_cache"     # D:; C: had 0.7 GB free
#  VGGSound label -> one v2.B sound name, by hand: a string match found 83 of
#  309 and sent "cattle, bovinae cowbell" (a cowbell) to "cow". The rules are
#  one_name()'s: ESC-50's name beats its own subtypes, else the most specific
#  name AudioSet would give ("duck quacking" is "quack", the child of "duck").
#  The first draft broke the first rule for 24 labels (fly, giggle, idling,
#  purr...): round 6 trains buzzing as "insects" (428 AudioSet/FSD50K clips,
#  0 as "fly"), and 150 VGGSound clips as fly/bee/mosquito would have taught a
#  rival answer to ESC-50's insects test -- the kind of overlap that already
#  cost ESC-50 95.2 -> 87.5. Exceptions: "cattle, bovinae cowbell" stays
#  "cowbell" (a bell, not a cow); "coyote howling" is left out (howl sits under
#  Dog in AudioSet, and a coyote is not one). A second review found the fold
#  cuts the other way too: one_name() lets the parent win only when the clip
#  ALSO carries it, and AudioSet's eval labels 76 of its 4,000 test clips by a
#  subtype alone (giggle 11, firecracker 11, hiss 11 + 17 in FSD50K, howl 9,
#  rustling leaves 8, belly laugh 7, growling 7 in FSD50K, purr 6, lawn mower
#  6). So a label goes when either name would compete with a test: those
#  subtypes; "cricket chirping" (AudioSet/FSD50K crickets are "insects",
#  ESC-50's are "crickets"); "people babbling" ("hubbub" is a condition tag
#  one_name drops); "railroad car, train wagon" (a sibling of Train, not a
#  subtype); "ocean burbling" (no test clip is "ocean", 40 are "sea waves");
#  and the songbird-like mynah, magpie and cuckoo under "bird vocalization"
#  (FSD50K's test holds 63 "chirping birds"). 221 labels, 11,049 clips, 185
#  names; 37 of ESC-50's 50 classes get clips. Labels with no name of their own (sports,
#  wild animals AudioSet lacks, "dinosaurs bellowing") are left out rather
#  than forced onto a catch-all, and so is "playing erhu": its nearest name,
#  "bowed string instrument", is one of SOUND_BROAD's capped catch-alls.
VGGSOUND_NAMES = {
    "air conditioning noise": "air conditioning", "air horn": "air horn",
    "airplane": "airplane", "airplane flyby": "airplane",
    "alarm clock ringing": "clock alarm", "ambulance siren": "siren",
    "baby babbling": "babbling", "baby crying": "crying baby", "baby laughter": "laughing",
    "baltimore oriole calling": "chirping birds", "barn swallow calling": "chirping birds",
    "basketball bounce": "basketball bounce", "bathroom ventilation fan running": "mechanical fan",
    "beat boxing": "beatboxing", "bee, wasp, etc. buzzing": "insects",
    "bird chirping, tweeting": "chirping birds", "bird squawking": "squawk",
    "bird wings flapping": "bird flight", "black capped chickadee calling": "chirping birds",
    "bull bellowing": "cow", "canary calling": "chirping birds", "cap gun shooting": "cap gun",
    "car engine idling": "engine", "car engine knocking": "engine",
    "car engine starting": "engine", "car passing by": "car passing by",
    "cat caterwauling": "cat", "cat growling": "cat",
    "cat meowing": "cat", "cattle mooing": "cow",
    "cattle, bovinae cowbell": "cowbell", "chainsawing trees": "chainsaw",
    "chicken clucking": "hen", "chicken crowing": "rooster", "child singing": "child singing",
    "child speech, kid speaking": "child speech", "children shouting": "children shouting",
    "chopping food": "chopping (food)", "chopping wood": "chop",
    "church bell ringing": "church bells", "civil defense siren": "siren",
    "cow lowing": "cow",
    "crow cawing": "crow",
    "cupboard opening or closing": "cupboard open or close",
    "cutting hair with electric trimmers": "electric shaver",
    "disc scratching": "scratching (performance technique)", "dog barking": "dog",
    "dog baying": "dog", "dog bow-wow": "dog",
    "dog whimpering": "dog", "door slamming": "slam", "driving buses": "bus",
    "driving motorcycle": "motorcycle", "duck quacking": "quack",
    "eagle screaming": "bird vocalization", "eating with cutlery": "cutlery",
    "electric shaver, electric razor shaving": "electric shaver",
    "eletric blender running": "blender", "engine accelerating, revving, vroom": "engine",
    "female singing": "female singing", "female speech, woman speaking": "female speech",
    "fire crackling": "crackling fire", "fire truck siren": "siren",
    "fireworks banging": "fireworks", "firing cannon": "artillery fire",
    "firing muskets": "gunshot", "fly, housefly buzzing": "insects", "foghorn": "foghorn",
    "footsteps on snow": "footsteps", "francolin calling": "bird vocalization",
    "frog croaking": "frog", "goat bleating": "goat", "goose honking": "honk",
    "hair dryer drying": "hair dryer", "hammering nails": "hammer",
    "heart sounds, heartbeat": "heart sounds", "helicopter": "helicopter",
    "horse clip-clop": "clip-clop", "horse neighing": "neigh",
    "ice cream truck, ice cream van": "ice cream truck",
    "lions growling": "roaring cats (lions",
    "lions roaring": "roar", "machine gun shooting": "machine gun",
    "male singing": "male singing",
    "male speech, man speaking": "male speech", "mosquito buzzing": "insects",
    "motorboat, speedboat acceleration": "motorboat", "mouse clicking": "mouse click",
    "mouse pattering": "patter", "mouse squeaking": "mouse",
    "opening or closing car electric windows": "power windows",
    "opening or closing drawers": "drawer open or close", "orchestra": "orchestra",
    "owl hooting": "hoot", "parrot talking": "bird vocalization",
    "people battle cry": "battle cry",
    "people burping": "burping", "people cheering": "cheering", "people clapping": "clapping",
    "people coughing": "coughing", "people crowd": "crowd", "people eating": "chewing",
    "people eating apple": "chewing", "people eating crisps": "chewing", "people farting": "fart",
    "people finger snapping": "finger snapping", "people gargling": "gargling",
    "people hiccup": "hiccup", "people humming": "humming",
    "people running": "run", "people screaming": "screaming", "people shuffling": "shuffle",
    "people slapping": "slap", "people slurping": "drinking sipping",
    "people sneezing": "sneezing", "people sniggering": "laughing", "people sobbing": "crying",
    "people whispering": "whispering", "people whistling": "whistling",
    "pheasant crowing": "bird vocalization", "pig oinking": "pig", "pigeon, dove cooing": "coo",
    "plastic bottle crushing": "crushing", "playing accordion": "accordion",
    "playing acoustic guitar": "acoustic guitar", "playing bagpipes": "bagpipes",
    "playing banjo": "banjo", "playing bass drum": "bass drum",
    "playing bass guitar": "bass guitar", "playing bongo": "drum",
    "playing bugle": "brass instrument", "playing cello": "cello", "playing clarinet": "clarinet",
    "playing congas": "drum", "playing cornet": "brass instrument", "playing cymbal": "cymbal",
    "playing didgeridoo": "didgeridoo", "playing djembe": "drum",
    "playing double bass": "double bass", "playing drum kit": "drum kit",
    "playing electric guitar": "electric guitar", "playing electronic organ": "electronic organ",
    "playing flute": "flute",
    "playing french horn": "french horn", "playing glockenspiel": "glockenspiel",
    "playing gong": "gong", "playing hammond organ": "hammond organ",
    "playing harmonica": "harmonica", "playing harp": "harp", "playing harpsichord": "harpsichord",
    "playing mandolin": "mandolin", "playing marimba, xylophone": "marimba",
    "playing piano": "piano", "playing saxophone": "saxophone", "playing shofar": "shofar",
    "playing sitar": "sitar", "playing snare drum": "snare drum",
    "playing steel guitar, slide guitar": "steel guitar", "playing steelpan": "steelpan",
    "playing synthesizer": "synthesizer", "playing tabla": "tabla",
    "playing tambourine": "tambourine", "playing theremin": "theremin",
    "playing timbales": "drum", "playing timpani": "timpani", "playing trombone": "trombone",
    "playing trumpet": "trumpet", "playing tuning fork": "tuning fork",
    "playing tympani": "timpani", "playing ukulele": "ukulele",
    "playing vibraphone": "vibraphone", "playing violin, fiddle": "violin",
    "playing zither": "zither", "police car (siren)": "siren", "printer printing": "printer",
    "pumping water": "pump (liquid)", "race car, auto racing": "race car",
    "raining": "rain", "rapping": "rapping",
    "reversing beeps": "reversing beeps", "ripping paper": "tearing",
    "rowboat, canoe, kayak rowing": "rowboat", "running electric fan": "mechanical fan",
    "sailing": "sailboat", "sea waves": "sea waves", "sheep bleating": "sheep",
    "singing bowl": "singing bowl", "singing choir": "choir", "skateboarding": "skateboard",
    "skidding": "skidding", "sliding door": "sliding door", "sloshing water": "slosh",
    "smoke detector beeping": "smoke detector", "snake hissing": "snake",
    "snake rattling": "rattle", "splashing water": "splash", "spraying water": "spray",
    "squishing water": "squish", "stream burbling": "stream", "subway, metro, underground": "subway",
    "tapping guitar": "tapping (guitar technique)",
    "telephone bell ringing": "telephone bell ringing", "thunder": "thunderstorm",
    "toilet flushing": "toilet flush", "train horning": "train",
    "train wheels squealing": "train wheels squealing", "train whistling": "train",
    "turkey gobbling": "gobble", "typing on computer keyboard": "keyboard typing",
    "typing on typewriter": "typewriter", "using sewing machines": "sewing machine",
    "vacuum cleaner cleaning floors": "vacuum cleaner",
    "vehicle horn, car horn, honking": "car horn", "warbler chirping": "chirping birds",
    "waterfall burbling": "waterfall", "whale calling": "whale vocalization",
    "wind chime": "wind chime", "wind noise": "wind",
    "wood thrush calling": "chirping birds", "writing on blackboard with chalk": "writing",
    "yodelling": "yodeling",
}
#  (3) RENDITIONS: every picture so far was a photo. ImageNet-R (axiong/
#  imagenet-r, MIT: 30,000 pictures of 200 ImageNet classes as art, cartoons,
#  deviantart, graffiti, embroidery, graphics, origami, paintings, patterns,
#  plastic objects, plush, sculptures, sketches, tattoos, toys and video games)
#  and ImageNet-Sketch (songweig/imagenet_sketch: ~50 pencil sketches of each
#  of the 1,000 classes; SKETCH_PER_CLASS of them, read member by member from
#  the 7.6 GB zip over HTTP). Stored at 256 px (shorter side, centre crop) in
#  data/imagenet_subset as imr.parquet / sketch.parquet, so they are read
#  like the ImageNet photos ("inet-imr:<row>"). Imagenette's 10 classes and
#  INET_SKIP stay out, as for the photos; Tiny-ImageNet's 200 do NOT -- a
#  painting of a goldfish is not one of Tiny's test photos.
TRAIN_ON_RENDITIONS = True
IMR_REPO = "axiong/imagenet-r"
SKETCH_REPO = "songweig/imagenet_sketch"
SKETCH_PER_CLASS = 15
#  Each rendition's row hashes into test (REND_TEST_SHARE) or train. There is
#  no rendition validation: selection stays on round 6's validation set, so
#  the rounds compare on identical numbers. The new tests are measured for
#  the deployed round-6 model too (results/v2b_round7_baseline.json).
REND_TEST_SHARE = 0.2
#  A NEW training picture (renditions, and the round-7 ImageNet photos) whose
#  SigLIP identity is this close to any validation or test picture's is
#  dropped. ImageNet holds the same photo under two classes (9 of the more2
#  photos are byte-identical to inet_test photos, 6 to inet_val, each under a
#  DIFFERENT label), and ImageNet-R the same artwork more than once. Measured
#  on inet_train vs inet_test: 555 pairs >= 0.95 (504 of them just same-class
#  lookalikes), 36 >= 0.97, 4 >= 0.99 -- exactly the byte-identical ones. 0.97
#  catches resized or recropped copies without stripping lookalikes.
IMAGE_NEAR_COPY = 0.97
#  Renditions need a lower bar. ImageNet-R and ImageNet-Sketch also hold
#  MIRRORED copies, often re-cropped or watermarked too; encoding each test
#  picture mirrored as well (identity_flip) moved those from 0.965-0.97 to
#  >= 0.99, but contact sheets of the nearest train/test pairs still showed
#  copies down to 0.93 (half the sketch pairs at 0.95-0.96, a few at
#  0.93-0.95: a bee eater, a torch, "hen" vs "cock", "agaric" vs "bolete").
#  At 0.93 the filter drops 813 of 23,737 ImageNet-R and 713 of 11,461 sketch
#  training pictures; the same bar on the round-7 photos would drop 6,374
#  same-class lookalikes, so they keep IMAGE_NEAR_COPY (125 dropped).
REND_NEAR_COPY = 0.93
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
#  8 (round 9). Round 8 kept epoch 8 of 16 and its curve says why: pictures
#  flat from epoch 4 (87.5, 87.7, 87.6, 87.8, 87.9 ... 87.9), sounds peaking at
#  54.7 (epoch 6) then sliding to 52.3, never-trained classes at full
#  correction falling all the way (76.3 -> 47.4 at epoch 8 -> 41.3). Keeping
#  the best epoch does not make the extra epochs free: under OneCycle the kept
#  epoch 8 was taken at high learning rate, never annealed. 8 epochs lets the
#  rate finish decaying where validation peaks, in half the time.
EPOCHS = 8
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
#  Leave-one-out training builds (queries x bank) similarities: 512 x 237,001
#  in round 6, when the 6 GB card already spilled into shared memory once.
#  Round 7's bank is ~350,000, so the query batch shrinks to keep that size.
MEMORY_SIM_BUDGET = 512 * 237_001
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
