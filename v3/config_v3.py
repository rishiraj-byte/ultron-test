"""
config_v3.py -- every v3 constant, in the units the brain uses.

WHY THE UNITS CHANGED. v2's constants were dimensionless: a delay of "1" meant
one update of a loop, a state magnitude of "4.0" meant nothing in particular.
Here a delay is milliseconds, a membrane potential is millivolts, and a time
constant is the real one measured in a real cell. That is not decoration -- it
is what makes the numbers checkable against the literature instead of against
each other.

Standing rule from v1 and v2, still in force: nothing below may be hardcoded
anywhere else. If a number matters, it lives here.
"""

from __future__ import annotations

from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CHECKPOINT_V3 = PROJECT_ROOT / "checkpoints" / "brain_v3.pt"
SEED = 20260903


def resolve_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


# ---------------------------------------------------------------------------
# TIME
#
# v2 ran twelve abstract "timesteps". v3 runs in milliseconds, because every
# other quantity here is defined against real time and a dimensionless tick
# cannot be compared to a membrane time constant.
#
# 1 ms is the standard integration step for point-neuron models: fast enough
# to resolve the 2 ms refractory period and spike-timing-dependent plasticity
# windows, slow enough to simulate a few hundred milliseconds without the step
# count exploding. It is not continuous time, and that remains a difference
# from a real brain -- but it is a discretisation of real time rather than an
# invented clock.
# ---------------------------------------------------------------------------
DT_MS = 1.0
#  How long one percept is given. Human perceptual decisions land in roughly
#  150-300 ms from stimulus onset; 300 ms leaves room for the slow loops
#  (cortico-basal-ganglia, cortico-cerebellar) to close more than once.
PERCEPT_MS = 300.0
STEPS = int(PERCEPT_MS / DT_MS)


# ---------------------------------------------------------------------------
# THE NEURON
#
# Leaky integrate-and-fire. Charge leaks toward rest; when the membrane
# crosses threshold the cell emits a spike and is pushed back to reset, and
# for a short refractory window it cannot fire again no matter what arrives.
#
# The refractory period is one of the drawbacks being carried forward
# deliberately. It caps the firing rate at 1000/REFRACTORY_MS Hz, so a neuron
# physically cannot signal more strongly than about 500 Hz however hard it is
# driven. A rate-coded float has no such ceiling, which is precisely why v2's
# cortical state could reach a magnitude of 567,699.
# ---------------------------------------------------------------------------
V_REST_MV = -65.0
V_THRESHOLD_MV = -50.0
V_RESET_MV = -65.0
#  Pyramidal cells integrate slowly; inhibitory interneurons are fast, which
#  is what lets inhibition track and contain excitation rather than lag it.
TAU_M_EXC_MS = 20.0
TAU_M_INH_MS = 10.0
REFRACTORY_MS = 2.0

#  Synaptic current decay. AMPA is fast, GABA-A slower, and the difference is
#  functional: inhibition outlasts the excitation that recruited it.
TAU_SYN_EXC_MS = 5.0
TAU_SYN_INH_MS = 10.0

#  Dendritic compartments. A pyramidal cell is not a point: feedforward input
#  arrives on basal dendrites near the soma, feedback arrives on the apical
#  tuft far away, and the tuft cannot fire the cell by itself -- it modulates
#  what the basal input does. This is thought to be how cortex keeps "what I
#  am being told" separate from "what I am predicting" inside one cell, and it
#  is the single mechanism v2 had no way to express.
APICAL_GAIN = 0.35            # how much the tuft can modulate the soma
APICAL_DRIVE_ALONE = 0.0      # feedback ALONE cannot make a cell fire


# ---------------------------------------------------------------------------
# POPULATIONS
#
# Cortex is roughly 80% excitatory pyramidal cells and 20% inhibitory
# interneurons, and that ratio is stable across areas and species. Dale's
# principle holds: a cell is one or the other for life.
# ---------------------------------------------------------------------------
EXC_PER_AREA = 1024

#  THREE INTERNEURON CLASSES, NOT ONE.
#
#  "Inhibitory neuron" is not a type, it is a category containing cells that
#  do completely different jobs, and the difference is WHERE on the pyramidal
#  cell they land:
#
#    PV   parvalbumin, fast-spiking, contacts the SOMA. Controls whether the
#         cell fires at all, and how fast. This is gain control.
#    SST  somatostatin, contacts the APICAL DENDRITE. Controls whether
#         feedback and context reach the cell -- not its output, but its
#         receptivity to prediction.
#    VIP  vasoactive intestinal peptide, contacts SST cells and little else.
#         Its job is to switch SST off.
#
#  That last one is the point. VIP inhibits SST, SST inhibits the apical
#  tuft, so driving VIP DISINHIBITS the tuft and lets top-down signals into
#  the cell. A one-type model cannot express that: it can turn cortex down,
#  but it cannot selectively open the door that context arrives through.
#  This motif is thought to be much of how attention works.
PV_PER_AREA = 128
SST_PER_AREA = 96
VIP_PER_AREA = 32
INH_PER_AREA = PV_PER_AREA + SST_PER_AREA + VIP_PER_AREA
NEURONS_PER_AREA = EXC_PER_AREA + INH_PER_AREA

#  A cortical pyramidal cell contacts a small fraction of its local
#  population, not all of it. Dense all-to-all connectivity -- what v2 had --
#  is the single least biological thing about a standard weight matrix.
P_CONNECT_LOCAL = 0.10
P_CONNECT_TRACT = 0.05

#  PERISOMATIC INHIBITION IS DENSE. A basket cell contacts a large fraction
#  of the pyramidal cells near it, right at the soma where it has the most
#  leverage -- unlike excitatory contacts, which are sparse and distal.
#
#  This asymmetry is what keeps cortex in the asynchronous irregular regime
#  instead of flipping between silence and seizure. Built with inhibition at
#  twice the excitatory connection probability, total inhibitory drive was
#  only 2.4x recurrent excitation and the network was bistable: four lobes
#  swinging between 0 Hz and 60 Hz with nothing in between. Inhibition has to
#  DOMINATE, not merely balance.
P_CONNECT_INHIBITORY = 0.40
#  SST cells contact many pyramidal apical tufts; VIP cells contact SST
#  densely and specifically, which is what makes the disinhibition
#  switch-like rather than gradual.
P_CONNECT_SST = 0.30
P_CONNECT_VIP = 0.35
#  How strongly a feedback axon drives VIP alongside the apical tuft it also
#  contacts. Modest: this opens a door, it does not push anything through it.
VIP_FEEDBACK_GAIN = 1.6

#  SYNAPTIC STRENGTH, in millivolts of postsynaptic potential per spike.
#
#  This has to be stated in real units or nothing else works. A cortical cell
#  sits 15 mV below threshold, and a single excitatory synapse moves it by
#  well under a millivolt -- so roughly 25 to 30 near-coincident inputs are
#  needed to fire it. That number is the whole reason cortex needs population
#  coding: no single presynaptic cell can command a postsynaptic one.
#
#  Getting this wrong is silent. Initialised as a fan-in-scaled rate weight,
#  the first build produced excitatory cells firing at a plausible 4 Hz while
#  the inhibitory population never fired ONCE -- the microcircuit looked
#  healthy from the outside and had no inhibition in it at all.
#
#  Inhibitory synapses are individually stronger than excitatory ones, which
#  is real: a basket cell contacting a pyramidal soma has far more leverage
#  than one distal excitatory contact.
EPSP_MV = 0.55
IPSP_MV = 1.30

#  DRIVER SYNAPSES. Not every connection is the same strength, and the
#  difference is categorical rather than gradual. Sherman and Guillery's
#  distinction: a DRIVER carries the message and is powerful enough that a
#  handful of them fire the postsynaptic cell -- retina to LGN, LGN to layer
#  4. A MODULATOR adjusts how that message is treated and is individually
#  weak -- which is what every cortico-cortical synapse above is.
#
#  Built with all synapses at cortical strength, the whole system was silent:
#  0.00 Hz in all four lobes, because 25 modulator synapses cannot move a
#  relay cell 15 mV. The fix is not "turn up the gain", it is that these are
#  a different KIND of synapse.
DRIVER_MV = 7.0

#  Target firing rate for homeostasis, in Hz. Cortical pyramidal cells fire
#  sparsely: a few Hz spontaneously, tens of Hz when driven.
TARGET_RATE_HZ = 5.0

#  BACKGROUND BOMBARDMENT -- the thing without which none of this works.
#
#  A cortical neuron in a living brain is never at rest. It receives a
#  continuous barrage from thousands of synapses, which holds its membrane a
#  few millivolts BELOW threshold with large fluctuations riding on top. It
#  then fires when a fluctuation happens to carry it over -- irregularly, at a
#  low rate, driven by variance rather than by mean.
#
#  That is the asynchronous irregular regime, and it is the operating point of
#  cortex. Without background input the network has no such regime available:
#  every cell is either far below threshold (silent) or pushed past it
#  (saturated), and the whole sheet flips between the two. Measured without
#  it, four lobes swung between 0.00 Hz and 44 Hz across successive epochs
#  with nothing stable in between, and no amount of homeostasis fixed it
#  because the problem was not the operating point but the absence of one.
#
#  The mean sits the membrane about 4 mV under threshold; sigma supplies the
#  fluctuations that occasionally close that gap.
BACKGROUND_MEAN = 0.55
BACKGROUND_SIGMA = 0.40


# ---------------------------------------------------------------------------
# CONDUCTION DELAYS, in milliseconds
#
# These are not invented. Thalamocortical afferents are fast and short; long
# association bundles crossing the hemisphere take an order of magnitude
# longer. v2 collapsed all of this into "1" or "2", which made the fast and
# slow pathways nearly indistinguishable -- the amygdala's two roads differed
# by one abstract tick.
# ---------------------------------------------------------------------------
DELAY_MS = {
    # thalamus to cortex: short, myelinated, fast
    "optic_radiation":     4.0,    # LGN  -> occipital
    "acoustic_radiation":  4.0,    # MGN  -> temporal
    "somato_radiation":    4.0,    # VPL  -> parietal
    "pulvinar_temporal":   5.0,    # pulvinar -> temporal
    "md_frontal":          5.0,    # MD   -> frontal
    "vavl_frontal":        5.0,    # VA/VL -> frontal

    # cortico-cortical association bundles: long, slower
    "occ_par":            10.0,    # dorsal stream, first leg
    "ilf":                12.0,    # ventral stream, first leg
    "slf":                12.0,    # dorsal stream, second leg
    "arcuate":            15.0,    # ventral stream, second leg
    "ifof":               18.0,    # the long shortcut, and it IS long
    "lat_par_occ":        10.0,

    # feedback: slower than the feedforward it answers
    "fb_frontal_occ":     20.0,
    "fb_frontal_par":     15.0,
    "fb_frontal_tem":     18.0,
    "fb_par_occ":         12.0,
    "fb_tem_occ":         14.0,

    # subcortical
    "low_road":            8.0,    # thalamus -> amygdala, the fast crude one
    "high_road":          25.0,    # temporal -> amygdala, slow and accurate
    "amygdala_cortex":    10.0,
    "cortico_hippocampal": 12.0,
    "fornix":             14.0,
    "corticostriatal":     8.0,
    "hyperdirect":         4.0,    # the emergency brake is the fastest route
    "striatum_gpi":        6.0,
    "striatum_gpe":        6.0,
    "gpe_stn":             4.0,
    "stn_gpi":             4.0,
    "gpi_thalamus":        5.0,
    "pontine":            10.0,
    "dentate_vl":         10.0,
    "hypothalamic":       15.0,
}


# ---------------------------------------------------------------------------
# LEARNING
#
# No loss function appears anywhere in v3. A synapse changes using three
# things it can physically sense: that the cell before it fired, that the cell
# after it fired, and what neuromodulator is currently bathing the tissue.
#
# STDP: if pre fires shortly BEFORE post, the synapse strengthens -- pre
# helped cause post. If it fires shortly after, it weakens. The asymmetry is
# what makes it causal rather than merely correlational.
# ---------------------------------------------------------------------------
STDP_TAU_PRE_MS = 20.0
STDP_TAU_POST_MS = 20.0
STDP_A_PLUS = 0.008
#  Depression slightly exceeds potentiation. Without this, any Hebbian rule
#  runs away: cells that fire together wire together, then fire together more.
STDP_A_MINUS = 0.0088

#  ELIGIBILITY TRACE. STDP alone cannot solve credit assignment across time --
#  the reward for a decision arrives long after the synapses that caused it
#  have stopped signalling. A slow trace holds the coincidence until a
#  neuromodulator arrives to confirm or cancel it. This is the three-factor
#  rule, and it is the closest thing biology has to what backprop does.
TAU_ELIGIBILITY_MS = 500.0
DOPAMINE_BASELINE = 0.0
DOPAMINE_GAIN = 1.0
#  Reward-driven rate for dopamine-gated synapses.
#
#  Measured why this had to rise: the class signal reaches motor cortex intact
#  (separability 0.55 at the receptors, 0.27 at motor -- degraded but present),
#  so the failure to learn was never a propagation problem. It was that a
#  scalar reward applied at 0.02 moves a sparse 1024x1280 projection almost
#  imperceptibly, and a thousand trials is nothing. Animals need thousands of
#  trials for tasks like this; that slowness is real and is being carried
#  forward, but the per-trial step still has to be large enough to accumulate.
LEARNING_RATE = 0.15
#  Rate for plain, unsupervised STDP on cortical and thalamocortical
#  synapses. Smaller than the reward-driven rate because it applies on every
#  millisecond rather than once per decision.
UNSUPERVISED_RATE = 0.004

WEIGHT_MIN = 0.0              # Dale's law: a sign never flips
WEIGHT_MAX = 4.0

#  HOMEOSTASIS. Real neurons defend their own firing rate over hours: too
#  quiet and they scale their inputs up, too loud and down. Without it, purely
#  Hebbian networks either fall silent or seize. It is slow on purpose.
TAU_HOMEOSTATIC_MS = 10_000.0
HOMEOSTATIC_RATE = 0.30
#  Synaptic scaling is slower than threshold adjustment in real tissue -- it
#  involves trafficking receptors, not moving an ion channel's operating
#  point. See LIFPopulation.input_scale.
SCALING_RATE = 0.35
#  How fast the rate sensor tracks. See LIFPopulation.rate_estimate for the
#  instability that comes from making this too slow.
RATE_ESTIMATE_BLEND = 0.35

#  ASTROCYTES. Glia are not passive scaffolding -- they take up glutamate,
#  release gliotransmitters, and modulate synaptic strength over seconds. This
#  is a coarse model of one effect: activity-dependent local gain control on a
#  timescale between the synapse and homeostasis.
ASTROCYTE_TAU_MS = 2000.0
ASTROCYTE_STRENGTH = 0.15


# ---------------------------------------------------------------------------
# THE DIFFUSE MODULATORY SYSTEMS
#
# Four tiny nuclei whose axons branch across most of the brain. They do not
# carry messages; they set states, and they do it on timescales between a
# second and a minute -- far slower than a percept and far faster than
# homeostasis. See modulators.py.
#
# The time constants differ on purpose. Noradrenaline tracks arousal and can
# shift within a second; serotonin moves over tens of seconds, which is part
# of why mood is not a moment-to-moment quantity.
# ---------------------------------------------------------------------------
MODULATOR_TARGET_HZ = 8.0
NORADRENALINE_TAU_MS = 1_000.0
ACETYLCHOLINE_TAU_MS = 2_000.0
SEROTONIN_TAU_MS = 20_000.0
DOPAMINE_TAU_MS = 500.0

#  How far acetylcholine can tilt cortex between trusting its senses and
#  trusting its predictions. At the extreme low end feedback dominates with
#  no sensory check on it, which is the regime that produces confident
#  nonsense -- kept, because it is a real property of the mechanism.
ACH_TILT = 0.35


# ---------------------------------------------------------------------------
# SENSORY TRANSDUCTION
#
# Frozen encoders survive into v3 at the INPUT BOUNDARY ONLY, and the reason
# is not convenience. A retina and a cochlea are fixed transducers: you did
# not learn them from experience, development specified them, and they convert
# the world into spike trains before any cortex sees anything. That is exactly
# the job these do here.
#
# Nothing downstream of the boundary sees a float. There is no pretrained
# network anywhere inside the circuit -- in particular frontal has none, which
# is the difference from v2 that this file most wants noticed.
# ---------------------------------------------------------------------------
RETINA_MODEL = "resnet50"
COCHLEA_MODEL = "openai/whisper-small"
TEXT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

#  Rate coding at the boundary: a strong feature drives a high Poisson rate.
#  Real receptors also adapt, and this keeps that -- a constant stimulus
#  produces a burst at onset and then settles, which is why you stop noticing
#  a smell.
INPUT_RATE_MAX_HZ = 80.0
ADAPTATION_TAU_MS = 100.0
ADAPTATION_STRENGTH = 0.6


# ---------------------------------------------------------------------------
# THE OUTPUT PATH
#
# The correction v2 most needed. There, the answer was read straight off
# cortical state, which bypassed the basal ganglia entirely -- the gate could
# slam shut and the answer still came out.
#
# Here there is no such tap. A decision exists only when a motor population
# crosses threshold, and motor cortex is driven through the VA/VL thalamic
# nucleus, which the basal ganglia gates. Close the gate and nothing is
# emitted. That is what a gate means.
# ---------------------------------------------------------------------------
#  Neither input alone should fire a motor cell. Cortex says which action,
#  the gated thalamus says whether any action at all, and the cell needs both
#  -- see BrainV3.step for what breaks at either extreme.
MOTOR_SELECT_GAIN = 0.20       # from frontal cortex: which
MOTOR_PERMIT_GAIN = 0.75       # from VA/VL, gated by GPi: whether

#  EXPLORATION -- how an untrained brain gets its first reward.
#
#  A brain that has learned nothing cannot select an action, so at resting
#  GPi motor cortex reaches about 19 Hz against a 40 Hz threshold and NOTHING
#  is ever emitted. Measured: 0 decisions in 8 trials. Reinforcement learning
#  then has nothing to reinforce -- it can only ever observe the penalty for
#  failing to act, which teaches it nothing about which action was right.
#
#  Infants solve this by moving before they know why: motor babbling, driven
#  by spontaneous activity rather than by any plan. The noise makes actions
#  happen, some are rewarded, and selection takes over from there. It fades
#  as competence grows, which is what the anneal below is.
EXPLORATION_DRIVE = 1.4
EXPLORATION_DECAY = 0.90       # per training epoch
#  A FLOOR, because a brain that cannot act cannot learn.
#
#  Annealed without one, exploration reached 0.06 by epoch 30 and the system
#  stopped emitting actions entirely: 0.0% decided for the last eighteen
#  epochs of a 13.5 hour run. Since reward learning only ever sees the
#  penalty for not acting, those epochs taught it nothing at all -- the run
#  effectively ended at epoch 21 and kept burning wall clock.
#
#  Animals never stop exploring completely either; the rate falls, it does
#  not reach zero.
EXPLORATION_FLOOR = 0.45
#  How hard exploratory drive pushes the direct pathway. This is the term
#  that actually opens the gate on an untrained brain.
EXPLORATION_STRIATAL = 2.2

#  How hard a teacher drives the correct response. Strong enough to fire the
#  population reliably, since an association cannot form between a cortical
#  pattern and a motor population that never spikes.
TEACHER_DRIVE = 3.0
#  Fraction of the presentation the teacher drives before letting go. The
#  remainder measures what the brain does unaided, which is what the reward
#  is then computed from.
TEACHER_FRACTION = 0.45
#  Milliseconds of silence between the teacher letting go and the decision
#  window opening, so no taught spike can leak into the measurement.
TEACHER_GAP_STEPS = 15

#  Stimuli the gate-authority test averages over. One trial flips sign
#  because of reticular coupling between nuclei -- see verify_v3.
GATE_TRIALS = 6

MOTOR_POPULATION = 64          # neurons per candidate action
#  The rate a motor population must SUSTAIN before it counts as a committed
#  action. Swept rather than guessed: with cortical selection alone motor
#  reaches about 24 Hz and with the gate also open about 59 Hz, so the
#  threshold has to sit between them or a shut gate still produces actions.
#  A higher threshold is a brain that demands more evidence before moving.
MOTOR_THRESHOLD_HZ = 40.0      # sustained rate that counts as a decision
MOTOR_WINDOW_MS = 50.0         # over which that rate is measured

#  GPi is tonically active, so the gate is CLOSED by default and everything is
#  vetoed until the direct pathway releases it.
#  HOW HARD GPi LEANS ON THE THALAMUS.
#
#  A gate has to have a working range, and this one did not. At full strength
#  the pallidal projection delivers roughly forty times the excitatory drive
#  reaching VA/VL, so at its normal tonic rate the relay is not gated but
#  annihilated: motor cortex sat at 0.3 Hz against a 40 Hz threshold and the
#  system could never act at all. Force GPi silent and the same relay hits
#  319 Hz. All or nothing, with nothing usable in between.
#
#  Real pallidal inhibition MODULATES thalamic relay. Scaling it to be
#  comparable with the cortical drive it opposes restores a middle: mostly
#  shut at rest, released when the direct pathway quiets GPi, hard shut when
#  the indirect pathway drives it.
GPI_INHIBITION_GAIN = 1.00
#
#  RESTORED TO FULL after a long detour. Weakening this gave motor cortex a
#  sensible firing range but cost the gate its authority: averaged over six
#  stimuli, "open" read 27.9 Hz against "shut" at 29.6 -- the gate had stopped
#  controlling anything, because the reticular nucleus releases the ungated
#  nuclei whenever the gated ones are suppressed, and that effect outweighed
#  a weak pallidal projection.
#
#  Gate authority and a comfortable motor dynamic range are in direct tension
#  here, and authority is the one worth keeping: it is the whole reason v3
#  exists rather than v2.

GPI_TONIC_HZ = 60.0
#  Depolarising current, in mV/ms, that produces that tonic rate. Calibrated
#  against the measured resting rate rather than assumed.
GPI_TONIC_DRIVE = 0.62
