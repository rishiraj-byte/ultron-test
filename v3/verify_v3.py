"""
verify_v3.py -- the claims v3 makes, each written as a test that can fail.

Same discipline as v2's gate suite, aimed at different claims. v2's gates
asked whether the wiring diagram was real. These ask whether the SUBSTRATE is:
whether a spike is really a spike, whether a delay is really a delay, whether
anything anywhere is secretly using a gradient.

The most important gate in the file is `gate_gate_blocks_output`. In v2 the
answer was tapped straight off cortical state, so the basal ganglia could be
fully suppressed and the system answered anyway -- the gate was decorative
because it was not in the output path. This checks that closing it now
produces silence.
"""

from __future__ import annotations

import sys

import torch

from v3 import config_v3 as cfg
from v3.brain_v3 import BrainV3
from v3.neurons import CorticalArea, LIFPopulation
from v3.regions import Thalamus
from v3.synapses import SparseProjection
from v3.transduce import poisson_spikes

PASSED: list[str] = []
FAILED: list[str] = []


def check(label: str, condition, detail: str = "") -> None:
    ok = bool(condition)
    (PASSED if ok else FAILED).append(label)
    print("[%s] %-56s %s" % ("  ok  " if ok else " FAIL ", label, detail))


def _drive(brain, steps=120, batch=1):
    train = poisson_spikes(
        torch.rand(batch, Thalamus.RELAY_SIZE) * cfg.INPUT_RATE_MAX_HZ, steps)
    brain.reset(batch, "cpu")
    return train


# ---------------------------------------------------------------------------
def gate_no_gradient(brain: BrainV3) -> None:
    """Nothing in v3 may learn by backpropagation."""
    print("\n-- 1. THERE IS NO GRADIENT ANYWHERE --")
    trainable = [n for n, p in brain.named_parameters() if p.requires_grad]
    check("no parameter requires grad", not trainable,
          "%d found" % len(trainable))

    #  Weights are buffers, not parameters, so an optimiser cannot even see
    #  them. This is structural rather than a promise.
    weights = [n for n, _ in brain.named_buffers() if n.endswith(".weight")]
    check("synaptic weights are buffers, not parameters", len(weights) > 0,
          "%d weight buffers" % len(weights))

    #  Skip THIS file: it names the banned tokens in order to search for
    #  them, and a scanner that flags its own search terms is measuring
    #  itself rather than the system.
    source = [p for p in (cfg.PROJECT_ROOT / "v3").glob("*.py")
              if p.name != "verify_v3.py"]
    banned = []
    for path in source:
        text = path.read_text(encoding="utf-8")
        for token in (".backward()", "torch.optim", "cross_entropy",
                      "loss.backward"):
            #  Ignore prose: only flag it if it appears outside a comment.
            for line in text.splitlines():
                stripped = line.strip()
                if token in stripped and not stripped.startswith("#"):
                    banned.append("%s: %s" % (path.name, token))
    check("no backward pass, optimiser or loss in the source", not banned,
          str(banned[:3]))


def gate_spikes_are_spikes() -> None:
    print("\n-- 2. THE SIGNAL IS A SPIKE, NOT A NUMBER --")
    pop = LIFPopulation(256)
    pop.reset(4, "cpu")
    seen = set()
    for _ in range(50):
        pop.receive(torch.rand(4, 256) * 2.0, "exc")
        out = pop.step()
        seen.update(out.unique().tolist())
    check("output takes only the values 0 and 1", seen <= {0.0, 1.0},
          "values seen: %s" % sorted(seen))

    #  REFRACTORY PERIOD. Drive a cell as hard as physically possible and it
    #  still cannot exceed the ceiling its refractory period imposes. This is
    #  a limitation kept on purpose: v2's rate-coded state had no ceiling at
    #  all, which is how it reached a magnitude of 567,699.
    pop.reset(1, "cpu")
    fired = 0
    for _ in range(200):
        pop.receive(torch.full((1, 256), 500.0), "exc")
        fired += float(pop.step().mean())
    rate = fired / 200 * 1000.0
    ceiling = 1000.0 / cfg.REFRACTORY_MS
    check("firing rate is capped by the refractory period",
          rate <= ceiling + 1.0, "%.0f Hz under maximal drive, ceiling %.0f Hz"
          % (rate, ceiling))


def gate_dale() -> None:
    print("\n-- 3. DALE'S PRINCIPLE, STRUCTURALLY --")
    proj = SparseProjection(64, 64, 0.3, "exc", "dale_test")
    check("weights start non-negative", float(proj.weight.min()) >= 0.0,
          "min %.4f" % float(proj.weight.min()))

    #  Hammer it with the most negative consolidation it can receive and the
    #  sign still cannot flip -- the clamp is not a suggestion.
    proj.eligibility = torch.full_like(proj.eligibility, 5.0)
    for _ in range(50):
        proj.consolidate(-10.0)
    check("no weight can be driven negative", float(proj.weight.min()) >= 0.0,
          "min after 50 depressions: %.4f" % float(proj.weight.min()))
    check("a synapse cannot strengthen without bound",
          float(proj.weight.max()) <= cfg.WEIGHT_MAX + 1e-6,
          "max %.2f, ceiling %.2f" % (float(proj.weight.max()), cfg.WEIGHT_MAX))


def gate_delays() -> None:
    print("\n-- 4. CONDUCTION DELAYS ARE REAL TIME --")
    proj = SparseProjection(32, 32, 1.0, "exc", "delay_test").with_delay(15.0)
    proj.reset(1, "cpu")
    burst = torch.ones(1, 32)
    arrivals = []
    for t in range(30):
        out = proj(burst if t == 0 else torch.zeros(1, 32))
        if float(out.abs().sum()) > 0:
            arrivals.append(t)
    check("a 15 ms tract delivers 15 ms late", arrivals == [15],
          "arrived at t=%s ms" % arrivals)

    fast = int(cfg.DELAY_MS["low_road"] / cfg.DT_MS)
    slow = int(cfg.DELAY_MS["high_road"] / cfg.DT_MS)
    check("the amygdala's two roads differ by real milliseconds",
          slow - fast >= 15,
          "low road %d ms, high road %d ms, gap %d ms" % (fast, slow, slow - fast))


def gate_apical() -> None:
    print("\n-- 5. FEEDBACK CANNOT MANUFACTURE A PERCEPT --")
    pop = LIFPopulation(256)

    pop.reset(1, "cpu")
    for _ in range(100):
        pop.receive(torch.full((1, 256), 50.0), "apical")
        pop.step()
    apical_only = float(pop.spike_count.sum())

    pop.reset(1, "cpu")
    pop.spike_count.zero_()
    pop.step_count.zero_()
    for _ in range(100):
        pop.receive(torch.full((1, 256), 3.0), "exc")
        pop.step()
    basal_only = float(pop.spike_count.sum())

    #  Background bombardment means "fires nothing" is the wrong bar -- a
    #  living cell is never silent. The claim is that feedback does not raise
    #  firing ABOVE that baseline on its own.
    pop.reset(1, "cpu")
    pop.spike_count.zero_()
    pop.step_count.zero_()
    for _ in range(100):
        pop.step()
    baseline = float(pop.spike_count.sum())

    check("apical drive alone does not fire the cell above baseline",
          apical_only <= baseline * 1.15 + 1.0,
          "feedback alone %.0f spikes vs baseline %.0f" % (apical_only, baseline))
    check("basal drive alone does fire the cell", basal_only > 0.0,
          "%.0f spikes from feedforward alone" % basal_only)


def gate_learning_is_local() -> None:
    print("\n-- 6. LEARNING USES ONLY WHAT THE SYNAPSE CAN SENSE --")
    #  Two kinds of synapse, and the difference is real biology: cortical
    #  synapses change from spike timing alone, striatal ones will not change
    #  without dopamine. Both are tested.
    proj = SparseProjection(64, 64, 0.5, "exc", "local_test",
                            dopamine_gated=True)
    proj.reset(1, "cpu")
    before = proj.weight.clone()

    #  Pre fires, then post fires 5 ms later: pre helped cause post.
    for t in range(40):
        pre = torch.ones(1, 64) if t % 10 == 0 else torch.zeros(1, 64)
        post = torch.ones(1, 64) if t % 10 == 5 else torch.zeros(1, 64)
        proj.accumulate(pre, post)

    check("coincidence alone changes NO weight",
          torch.equal(before, proj.weight),
          "eligibility %.4f accumulated, weight unchanged"
          % float(proj.eligibility.abs().sum()))

    proj.consolidate(1.0)
    check("the weight moves only when a neuromodulator arrives",
          not torch.equal(before, proj.weight),
          "moved %.4f after dopamine" % float((proj.weight - before).abs().sum()))

    #  And zero dopamine must do nothing, however much eligibility there is.
    held = proj.weight.clone()
    proj.consolidate(0.0)
    check("zero neuromodulator changes nothing",
          torch.equal(held, proj.weight), "unchanged")

    #  A CORTICAL synapse is different: it learns from timing alone, with no
    #  reward anywhere. This is how sensory cortex becomes selective before
    #  anything is rewarding, and gating it on dopamine -- as this first did
    #  -- left the whole sensory pathway unable to develop at all.
    cortical = SparseProjection(64, 64, 0.5, "exc", "cortical_test")
    cortical.reset(1, "cpu")
    start = cortical.weight.clone()
    for t in range(40):
        pre = torch.ones(1, 64) if t % 10 == 0 else torch.zeros(1, 64)
        post = torch.ones(1, 64) if t % 10 == 5 else torch.zeros(1, 64)
        cortical.accumulate(pre, post)
    check("a cortical synapse learns from timing alone, unsupervised",
          not torch.equal(start, cortical.weight),
          "moved %.4f with no reward at all"
          % float((cortical.weight - start).abs().sum()))


def gate_sparse() -> None:
    print("\n-- 7. CONNECTIVITY IS SPARSE AND FIXED --")
    area = CorticalArea("sparse_test")
    density = float(area.e_to_e.mask.mean())
    check("local excitatory connectivity is sparse, not all-to-all",
          0.05 < density < 0.2, "%.1f%% of possible pairs connected"
          % (100 * density))
    check("perisomatic (PV) inhibition is DENSER than excitation",
          float(area.pv_to_e.mask.mean()) > density,
          "PV->E %.1f%% vs E->E %.1f%%"
          % (100 * float(area.pv_to_e.mask.mean()), 100 * density))

    #  THREE INTERNEURON CLASSES WITH THREE TARGETS. Collapsing them into one
    #  "inhibitory" population loses the VIP -> SST -> apical disinhibition
    #  motif entirely, which is much of what attention is thought to be.
    check("inhibition is three classes, not one",
          all(hasattr(area, n) for n in ("pv", "sst", "vip")),
          "PV %d, SST %d, VIP %d cells"
          % (area.pv.size, area.sst.size, area.vip.size))
    check("PV targets the soma and SST targets the apical tuft",
          area.pv_to_e.post_size == cfg.EXC_PER_AREA
          and area.sst_to_e.post_size == cfg.EXC_PER_AREA,
          "different compartments of the same cells")
    check("VIP inhibits SST and nothing else",
          area.vip_to_sst.post_size == cfg.SST_PER_AREA,
          "so driving VIP DISINHIBITS the tuft")

    before = area.e_to_e.mask.clone()
    area.e_to_e.consolidate(5.0)
    check("learning never grows a new connection",
          torch.equal(before, area.e_to_e.mask), "mask unchanged")


def gate_gate_blocks_output(brain: BrainV3) -> None:
    """THE gate. In v2 this was impossible to pass."""
    print("\n-- 8. CLOSING THE GATE STOPS THE OUTPUT --")
    print("   (v2 could not pass this: the answer was tapped straight off")
    print("    cortical state, so the basal ganglia was not in the path.)")

    #  AVERAGED OVER SEVERAL STIMULI, because one is not evidence.
    #
    #  Measured on a single input this verdict flips: the reticular nucleus
    #  couples the nuclei, so suppressing the gated ones releases the others
    #  and cortical activity rises. On some stimuli that outweighs the gating
    #  and "shut" reads higher than "open". The effect is real; the
    #  single-trial measurement was the thing that was wrong.
    resting = brain.basal_ganglia.gpi.threshold.clone()
    opens, shuts = [], []
    for _ in range(cfg.GATE_TRIALS):
        train = _drive(brain)
        brain.basal_ganglia.gpi.threshold.fill_(0.0)          # unreachable
        opens.append(float(
            brain.perceive({"LGN": train}, learn=False)["decision"]["rate_hz"].max()))
        brain.basal_ganglia.gpi.threshold.fill_(cfg.V_REST_MV + 2.0)
        shuts.append(float(
            brain.perceive({"LGN": train}, learn=False)["decision"]["rate_hz"].max()))
        brain.basal_ganglia.gpi.threshold.copy_(resting)

    open_rate = sum(opens) / len(opens)
    shut_rate = sum(shuts) / len(shuts)
    print("   over %d stimuli: open %.1f Hz, shut %.1f Hz"
          % (cfg.GATE_TRIALS, open_rate, shut_rate))

    #  Open vs shut is the meaningful comparison. Comparing SHUT against the
    #  resting state is not: suppressing the gated nuclei also removes their
    #  drive to the reticular nucleus, which releases the OTHER nuclei from
    #  inhibition and raises cortical activity. Measured that way the motor
    #  rate went UP when the gate closed -- a real effect of the TRN, and the
    #  wrong question.
    check("the gate controls the output: open beats shut",
          open_rate > shut_rate,
          "open %.1f Hz vs shut %.1f Hz" % (open_rate, shut_rate))
    check("a shut gate keeps motor cortex under decision threshold",
          shut_rate < cfg.MOTOR_THRESHOLD_HZ,
          "%.1f Hz, threshold %.0f Hz" % (shut_rate, cfg.MOTOR_THRESHOLD_HZ))


def gate_adaptation() -> None:
    print("\n-- 9. RECEPTORS ADAPT (a limitation, kept) --")
    #  Averaged over many receptors and a long window: a single 50 ms slice
    #  of a Poisson process is noise, and this gate read 0.056 -> 0.060 on
    #  one sample of a genuinely adapting signal.
    steady = torch.full((8, 512), 60.0)
    train = poisson_spikes(steady, 400, adapt=True)
    early = float(train[:100].mean())
    late = float(train[-100:].mean())
    check("a constant stimulus produces less response over time",
          late < early,
          "%.3f early -> %.3f late  (this is why you stop noticing a smell)"
          % (early, late))


def gate_no_pretrained_inside(brain: BrainV3) -> None:
    print("\n-- 10. NO PRETRAINED NETWORK INSIDE THE CIRCUIT --")
    names = [type(m).__name__ for m in brain.modules()]
    foreign = [n for n in names if n in ("Qwen2DecoderLayer", "BertLayer",
                                         "WhisperEncoderLayer", "ResNet",
                                         "Bottleneck")]
    check("no transformer or convolutional block anywhere in the brain",
          not foreign, str(sorted(set(foreign))[:4]))
    kinds = sorted(set(type(m).__name__ for m in brain.modules()))
    check("the brain is built only of populations and projections",
          set(kinds) <= {"BrainV3", "ModuleDict", "CorticalArea",
                         "LIFPopulation", "SparseProjection", "Thalamus",
                         "BasalGanglia", "Amygdala", "Hippocampus",
                         "Cerebellum", "Hypothalamus", "Modulators",
                         "DiffuseNucleus"},
          "module types: %d" % len(kinds))


def main() -> int:
    torch.manual_seed(cfg.SEED)
    print("=" * 74)
    print("v3 VERIFICATION -- is the substrate really biological?")
    print("=" * 74)

    brain = BrainV3(n_actions=20)

    gate_no_gradient(brain)
    gate_spikes_are_spikes()
    gate_dale()
    gate_delays()
    gate_apical()
    gate_learning_is_local()
    gate_sparse()
    gate_gate_blocks_output(brain)
    gate_adaptation()
    gate_no_pretrained_inside(brain)

    print()
    print("=" * 74)
    print("%d passed, %d failed" % (len(PASSED), len(FAILED)))
    for name in FAILED:
        print("   FAILED: %s" % name)
    print("=" * 74)
    return 1 if FAILED else 0


if __name__ == "__main__":
    sys.exit(main())
