"""
basal_ganglia_v2.py -- the full circuit, loop closed (BUILD_SPEC_V2 3.4, stage 5).

WHAT v1 GOT ANATOMICALLY WRONG.

v1 put the basal ganglia at the END: cortex decided, BG opened or closed a gate,
and that gate multiplied the output. That is not what this structure does. BG
output goes to the THALAMUS, and the thalamus goes back to frontal cortex. The
basal ganglia do not vet an answer on its way out the door; they select what
cortex is allowed to do NEXT. Putting the gate at the output makes the whole
loop invisible, and it is the loop that does the work.

THE CIRCUIT.

    cortex --> striatum --+-- DIRECT   (D1, "Go")   --> GPi --+
                          +-- INDIRECT (D2, "NoGo") --> GPe --> STN --> GPi
    cortex ------------------ HYPERDIRECT --------------------> STN
                                                                 |
                                    GPi --> thalamus VA/VL, MD --+--> frontal
    SNc --> dopamine --> striatum

SIGNS MATTER HERE, and they are the whole mechanism. The striatum is GABAergic,
so both of its pathways INHIBIT their targets. GPi is tonically active and
inhibits the thalamus, which means the resting state of this circuit is a
closed gate. The direct pathway inhibits GPi, and inhibiting an inhibitor is
release: "Go" works by DISINHIBITION. The indirect pathway inhibits GPe, which
was itself inhibiting STN, so suppressing GPe lets STN excite GPi harder and
the gate closes further. Getting any of these signs wrong yields a circuit that
still trains and still produces numbers, and is no longer a basal ganglia.

HYPERDIRECT is cortex straight to STN, skipping the striatum entirely. It is
the fastest path through the circuit and it raises GPi, so its effect is a
global brake -- stop everything, now, before the slower selection finishes.
v1 had no such path at all.

DOPAMINE MODULATES LEARNING RATE, not merely gain. That distinction is easy to
state and easy to fake: multiplying activations by a dopamine scalar is gain,
and it is what most implementations actually do. Here the D1 and D2 populations
pass through `scale_gradient`, so dopamine multiplies the gradient that reaches
striatal weights on the backward pass. High dopamine means the striatum learns
faster from what just happened, which is what a reward prediction error is for.

INITIALISATION. The direct and indirect matrices must NOT be zero-initialised.
With both at zero their gradients are equal and opposite, they cancel in the
sum, and the striatum receives about 1e-6 of gradient forever -- a dead module
that looks alive because loss still decreases elsewhere. This cost v1 real
training time before it was found. Small random weights, and biases at +1.5
and -1.5 so the two pathways start in genuinely different regimes.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn as nn

import config


class _ScaleGradient(torch.autograd.Function):
    """Identity forward; multiplies the gradient on the way back."""

    @staticmethod
    def forward(ctx, x: torch.Tensor, factor: torch.Tensor) -> torch.Tensor:
        ctx.save_for_backward(factor)
        return x

    @staticmethod
    def backward(ctx, grad):
        (factor,) = ctx.saved_tensors
        return grad * factor, None


def scale_gradient(x: torch.Tensor, factor: torch.Tensor) -> torch.Tensor:
    """Leave the forward pass alone, scale what flows back through it.

    This is how dopamine becomes a LEARNING RATE rather than a gain. If it
    multiplied activations instead, the same scalar would change the decision
    the circuit makes right now, which is a different claim entirely -- and
    the one that tends to get made by accident.
    """
    return _ScaleGradient.apply(x, factor)


@dataclass
class BasalGangliaOutput:
    """One pass through the circuit, with every stage kept for inspection."""

    #  nucleus -> drive delivered to the thalamus (B, dim)
    thalamic_drive: dict[str, torch.Tensor] = field(default_factory=dict)
    go: torch.Tensor | None = None
    nogo: torch.Tensor | None = None
    stn: torch.Tensor | None = None
    gpe: torch.Tensor | None = None
    gpi: torch.Tensor | None = None
    dopamine: torch.Tensor | None = None
    #  mean gate opening, 0..1 -- 1 fully released, 0 fully braked
    openness: float = 0.0

    def report(self) -> str:
        rows = [("go (D1 direct)", self.go), ("nogo (D2 indirect)", self.nogo),
                ("STN", self.stn), ("GPe", self.gpe), ("GPi", self.gpi)]
        lines = ["%-22s%10s" % ("stage", "mean")]
        for name, value in rows:
            if value is not None:
                lines.append("%-22s%10.4f"
                             % (name, float(value.mean().detach())))
        lines.append("%-22s%10.4f" % ("gate openness", self.openness))
        return "\n".join(lines)


class SubstantiaNigra(nn.Module):
    """SNc -- dopamine as reward prediction error.

    Predicts the value of the current cortical state; dopamine reports how
    much better the actual reward turned out to be. With no reward supplied
    it falls back to a tonic level, because a real SNc fires tonically and
    the circuit has to work when nothing is being rewarded.
    """

    def __init__(self, dim: int = config.LATENT_DIM,
                 tonic: float = config.DOPAMINE_TONIC):
        super().__init__()
        self.value = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, 1))
        self.tonic = tonic

    def forward(self, cortical: torch.Tensor,
                reward: torch.Tensor | None = None) -> torch.Tensor:
        predicted = self.value(cortical)                      # (B, 1)
        if reward is None:
            #  Tonic firing -- still a function of the state, so the value
            #  head is never silently cut out of the graph.
            return self.tonic + 0.0 * predicted
        if reward.dim() == 1:
            reward = reward.unsqueeze(-1)
        return self.tonic + (reward - predicted)

    def predicted_value(self, cortical: torch.Tensor) -> torch.Tensor:
        return self.value(cortical)


class BasalGangliaV2(nn.Module):
    """Striatum, GPe, GPi, STN, SNc -- output goes to the thalamus."""

    #  Which thalamic nuclei this circuit gates. Both are frontal-projecting:
    #  VA/VL is the motor and premotor return, MD the prefrontal one.
    GATED_NUCLEI = ("VA_VL", "MD")

    def __init__(self, dim: int = config.LATENT_DIM):
        super().__init__()
        self.dim = dim

        #  STRIATUM. Two populations, deliberately asymmetric at init.
        self.direct = nn.Linear(dim, dim)      # D1, "Go"
        self.indirect = nn.Linear(dim, dim)    # D2, "NoGo"
        for layer, bias in ((self.direct, config.STRIATUM_D1_BIAS),
                            (self.indirect, config.STRIATUM_D2_BIAS)):
            nn.init.normal_(layer.weight, std=config.STRIATUM_INIT_STD)
            nn.init.constant_(layer.bias, bias)

        #  Downstream nuclei.
        self.gpe = nn.Linear(dim, dim)
        #  How hard the indirect pathway inhibits GPe. Softplus-bounded so it
        #  stays an inhibition; starts above one so D2 is audible from the
        #  first step rather than having to climb out of a gradient of 1e-4.
        self.nogo_strength = nn.Parameter(
            torch.tensor(config.NOGO_STRENGTH_INIT))
        self.stn = nn.Linear(dim, dim)
        self.hyperdirect = nn.Linear(dim, dim)   # cortex straight to STN
        self.gpi = nn.Linear(dim, dim)
        #  D2's short route to GPi. Small at init so the GPe/STN route stays
        #  the dominant one in the forward pass.
        self.indirect_to_gpi = nn.Linear(dim, dim)
        nn.init.normal_(self.indirect_to_gpi.weight,
                        std=config.STRIATUM_INIT_STD)
        nn.init.zeros_(self.indirect_to_gpi.bias)

        #  GPi fires TONICALLY and inhibits the thalamus, so the circuit rests
        #  CLOSED and the direct pathway has to open it.
        nn.init.constant_(self.gpi.bias, config.GPI_TONIC)

        self.snc = SubstantiaNigra(dim)
        self.to_thalamus = nn.ModuleDict(
            {n: nn.Linear(dim, dim) for n in self.GATED_NUCLEI}
        )
        self.norm = nn.LayerNorm(dim)

    def forward(
        self,
        cortical: torch.Tensor,
        reward: torch.Tensor | None = None,
        hyperdirect_input: torch.Tensor | None = None,
    ) -> BasalGangliaOutput:
        """`cortical` is frontal state arriving over the corticostriatal tract.

        `hyperdirect_input` is the same cortical signal arriving by the fast
        route. It is a separate argument so a lesion can cut one and not the
        other, which is the only way to show the two paths do different things.
        """
        out = BasalGangliaOutput()

        #  NORMALISED INPUT. The circuit's behaviour depends on the SCALE of
        #  what arrives, not only its direction: every stage is a softplus,
        #  and a softplus's gradient depends on where on the curve it sits.
        #  When the cortical cores changed, the drive's magnitude changed with
        #  them and the indirect pathway's gradient fell to exactly zero --
        #  the same dead-module failure the init warning is about, reached by
        #  drifting into a different operating regime rather than by bad
        #  initialisation. Normalising here makes the striatum see a
        #  consistent input whatever the lobes upstream are made of.
        cortical = cortical * (config.STRIATUM_INPUT_NORM
                               / cortical.norm(dim=-1, keepdim=True)
                               .clamp(min=1e-6))

        dopamine = self.snc(cortical, reward)                  # (B, 1)
        out.dopamine = dopamine

        #  Dopamine sets how strongly the striatum learns from this pass. D1 is
        #  facilitated by dopamine and D2 suppressed by it, so one scalar drives
        #  the two populations in opposite directions -- which is exactly why
        #  they must not start out identical.
        #  THE PHASIC DEVIATION, not the absolute firing level.
        #
        #  This read `dopamine` directly, and dopamine is `tonic + (reward -
        #  predicted)`. So the tonic baseline -- 0.5, a constant, carrying no
        #  information about anything -- was being fed into the learning-rate
        #  tilt as though it were a prediction error. D1 therefore ran at
        #  1 + 0.5 = 1.5x before any reward was considered, and D2 at 0.5x,
        #  and D2 only learned at all when the error was negative enough to
        #  cancel the entire tonic level first. Measured: D2's learning rate
        #  was exactly zero on 11 of 12 timesteps, and its gradient was
        #  0.000000 -- a pathway wired correctly, receiving input, and unable
        #  to learn.
        #
        #  Reward prediction error is a DEVIATION from baseline; that is what
        #  the phrase means and what the SNc reports. Subtracting the tonic
        #  level restores it: at rest both arms learn at 1.0, a positive error
        #  tilts toward D1, a negative one toward D2, which is the whole
        #  point of having two of them.
        phasic = dopamine - config.DOPAMINE_TONIC
        floor = config.DOPAMINE_LR_FLOOR
        d1_lr = (1.0 + config.DOPAMINE_SENSITIVITY * phasic).clamp(min=floor)
        d2_lr = (1.0 - config.DOPAMINE_SENSITIVITY * phasic).clamp(min=floor)

        #  SOFTPLUS, not ReLU, and this is not a style preference. Firing
        #  rates are non-negative, so a rectifier is the right shape -- but D2
        #  starts at bias -1.5 against weights of std 0.02, which puts it flat
        #  on the wrong side of a hard hinge. Measured with ReLU the indirect
        #  pathway output was exactly 0.0000 and its gradient exactly zero: a
        #  dead population, arrived at by a different route than the zero-init
        #  trap this init was chosen to avoid. Softplus keeps the same resting
        #  asymmetry (D1 loud, D2 quiet) while leaving D2 differentiable.
        #  The scale goes on the OUTPUT of each striatal projection, not its
        #  input, and the difference is the whole mechanism. Scaling the input
        #  only touches the gradient continuing upstream to cortex: a weight
        #  gradient is grad_output x input, so the striatal weights never see
        #  it. Measured that way, D1 received gradient 294.118 under high
        #  reward and 294.118 under low -- identical to six figures, dopamine
        #  doing precisely nothing. On the output it multiplies grad_output,
        #  which is exactly the term the weight gradient is built from.
        go = nn.functional.softplus(scale_gradient(self.direct(cortical), d1_lr))
        nogo = nn.functional.softplus(scale_gradient(self.indirect(cortical), d2_lr))
        out.go, out.nogo = go, nogo

        #  INDIRECT: the striatum inhibits GPe, so GPe activity goes DOWN.
        #
        #  The inhibition carries a learnable strength, and it is not
        #  decoration. D1 reaches GPi in ONE step; D2 reaches it in three
        #  (GPe, then STN, then GPi), attenuating at each softplus. Measured,
        #  that left D2 with about a six-thousandth of D1's gradient -- a
        #  pathway that exists, is wired correctly, and cannot learn. The
        #  anatomy is genuinely asymmetric like this; what it does not have
        #  is a fixed unit gain on every synapse, so this one can grow.
        gpe = nn.functional.softplus(
            self.gpe(cortical) - nn.functional.softplus(self.nogo_strength) * nogo)
        out.gpe = gpe

        #  STN: excited by the hyperdirect path, and released by the fall in GPe.
        fast = self.hyperdirect(
            cortical if hyperdirect_input is None else hyperdirect_input
        )
        stn = nn.functional.softplus(self.stn(cortical) + fast - gpe)
        out.stn = stn

        #  GPi: tonically active, driven UP by STN, inhibited by the direct
        #  pathway. This subtraction is the selection.
        #
        #  The indirect pathway also reaches GPi directly, and that term is a
        #  gradient-flow accommodation as much as an anatomical one -- so it
        #  is worth being explicit about which. The primary indirect route is
        #  still GPe -> STN -> GPi above, and it dominates the forward pass.
        #  But D1 reaches GPi in ONE step while D2 reaches it in three, and
        #  measured across batches D2's gradient swung between 6e-3 and
        #  exactly 0 depending on the input -- a pathway that trains on some
        #  batches and not others. Real striatal projections to GPi are not
        #  exclusively via GPe, so a direct term is defensible; the honest
        #  statement is that it is here because the long route could not
        #  carry gradient reliably.
        gpi = self.gpi(stn) - go + self.indirect_to_gpi(nogo)
        out.gpi = gpi

        #  GPi is inhibitory onto the thalamus, so a HIGH GPi is a CLOSED gate.
        opening = torch.sigmoid(-gpi)
        if config.COLLECT_TELEMETRY:
            out.openness = float(opening.mean().detach())

        for nucleus in self.GATED_NUCLEI:
            out.thalamic_drive[nucleus] = self.norm(
                self.to_thalamus[nucleus](cortical) * opening
            )
        return out

    def value_loss(self, cortical: torch.Tensor,
                   reward: torch.Tensor) -> torch.Tensor:
        """The SNc value head has to be trained by its OWN objective.

        It cannot be trained by the task loss, and that is deliberate rather
        than an oversight. Dopamine reaches the striatum only as a gradient
        SCALE, and `_ScaleGradient.backward` returns None for that factor, so
        no gradient flows back into the value head from the task at all. If it
        did, the network could reduce its loss by mispredicting reward -- it
        would learn to report whatever learning rate was locally convenient,
        which is precisely the failure a reward prediction error exists to
        avoid. So the value head regresses on actual reward, and nothing else.
        """
        if reward.dim() == 1:
            reward = reward.unsqueeze(-1)
        return nn.functional.mse_loss(self.snc.predicted_value(cortical), reward)

    def striatal_parameters(self):
        """The tensors stage 5's gate watches for a live gradient."""
        return list(self.direct.parameters()) + list(self.indirect.parameters())


__all__ = ["BasalGangliaV2", "BasalGangliaOutput", "SubstantiaNigra",
           "scale_gradient"]
