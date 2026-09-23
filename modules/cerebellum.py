"""
cerebellum.py -- the forward model (BUILD_SPEC_V2 3.5, stage 7).

WHAT A FORWARD MODEL IS FOR.

Everything built so far reacts: sensory input arrives, cortex settles on an
interpretation, the basal ganglia select what to do next. Nothing anywhere
anticipates. A system like that can only ever find out what happened after it
has happened, which means it cannot tell the difference between a surprise and
a normal consequence of its own action.

The cerebellum predicts the sensory consequence of frontal's decision BEFORE
the consequence arrives. Two things follow, and both matter more than the
prediction itself:

  * The prediction error is a learning signal that needs no labels. Nobody has
    to say what the right answer was -- the world says it one timestep later.
  * What is predicted can be subtracted. A sensation the system caused itself
    is expected and cancels; a sensation it did not cause does not. That is the
    difference between moving past a wall and being pushed, and it is why you
    cannot tickle yourself.

THE CIRCUIT.

    frontal --> pontine nuclei --> cerebellar cortex --> dentate --> thalamus VL
                                        ^                                |
                                        |                                v
                                  climbing fibers                     frontal
                                   (inferior olive)
                                        ^
                                        |
                            what actually happened, one step later

The inferior olive is the part that is easy to get wrong. It does not carry the
prediction and it does not carry the outcome: it carries the DIFFERENCE, and it
only fires when there is one. A cerebellum wired to receive the outcome
directly would be learning to copy the world rather than to correct itself,
and it would keep learning forever instead of going quiet once it is right.

WHY THE ERROR IS COMPUTED AGAINST t-1 AND NOT t. The prediction made at
timestep t is a claim about timestep t+1. Comparing it against t would be
comparing a prediction with the thing it was predicted FROM, which any
identity function scores perfectly. The error here is therefore always
evaluated one step late, against the state that has since arrived.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

import config


@dataclass
class CerebellarOutput:
    """One prediction, and the error on the previous one."""

    #  what the cerebellum expects the next cortical state to be (B, dim)
    prediction: torch.Tensor
    #  the correction sent onward to thalamus VL (B, dim)
    correction: torch.Tensor
    #  mean squared prediction error on the PREVIOUS prediction, or None on
    #  the first timestep when there is nothing yet to have been wrong about
    error: torch.Tensor | None = None
    #  the same error divided by the energy of what it was predicting
    relative: torch.Tensor | None = None

    @property
    def error_value(self) -> float:
        """RELATIVE error -- the raw one is not comparable across timesteps.

        Cortical state starts at rest and grows into the percept, so a squared
        error on it grows with the state whether or not the prediction is
        improving. Measured across a settling window the raw error rose
        steadily, 0.0068 to 0.0114, purely because the thing being predicted
        got bigger; dividing by the target's own energy shows what is actually
        happening to the prediction.
        """
        source = self.relative if self.relative is not None else self.error
        return 0.0 if source is None else float(source.mean().detach())

    @property
    def raw_error_value(self) -> float:
        return 0.0 if self.error is None else float(self.error.mean().detach())


class InferiorOlive(nn.Module):
    """Climbing fibres: carries the DIFFERENCE, not the outcome.

    Deliberately has no capacity to speak of. A learnable module here would be
    able to invent an error when there was none, which is exactly what a
    teaching signal must not be able to do. The single learnable term is a
    gain, so the system can decide how loudly to listen -- never what it hears.
    """

    def __init__(self):
        super().__init__()
        self.gain = nn.Parameter(torch.tensor(1.0))

    def forward(self, predicted: torch.Tensor,
                actual: torch.Tensor) -> torch.Tensor:
        return self.gain * (actual - predicted)


class Cerebellum(nn.Module):
    """Predicts the next cortical state from frontal's current decision."""

    def __init__(self, dim: int = config.LATENT_DIM):
        super().__init__()
        #  Pontine nuclei: the relay that carries a copy of the motor command
        #  (here, frontal's decision) into the cerebellum. It is a genuine
        #  stage rather than a wire -- this is where a decision becomes an
        #  efference copy.
        self.pontine = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim), nn.GELU())

        #  Cerebellar cortex: granule-cell expansion then Purkinje readout.
        #  The expansion is the point -- a very wide sparse layer makes almost
        #  any two inputs linearly separable, which is what lets a shallow
        #  readout learn arbitrary mappings quickly.
        wide = dim * config.CEREBELLAR_EXPANSION
        self.granule = nn.Linear(dim, wide)
        self.purkinje = nn.Linear(wide, dim)

        #  Dentate nucleus: the cerebellum's only output, projecting to VL.
        self.dentate = nn.Linear(dim, dim)
        self.olive = InferiorOlive()
        self.norm = nn.LayerNorm(dim)

        self._last_prediction: torch.Tensor | None = None

    def forward(self, decision: torch.Tensor,
                actual: torch.Tensor | None = None) -> CerebellarOutput:
        """`decision` is frontal's state; `actual` is what really arrived.

        `actual` is the state the PREVIOUS prediction was about, so it is the
        caller's job to pass the newer one. See the module docstring.
        """
        error = None
        if actual is not None and self._last_prediction is not None:
            error = self.olive(self._last_prediction, actual)

        efference = self.pontine(decision)
        #  Sparse expansion, then a linear readout. Sparsity here is what makes
        #  the granule layer a separator rather than just a wider layer.
        granules = torch.relu(self.granule(efference))
        prediction = self.purkinje(granules)

        #  The correction sent to VL is the prediction, adjusted by how wrong
        #  the last one turned out to be. When the model is right the error
        #  term vanishes and the cerebellum stops interfering -- which is the
        #  correct behaviour for a forward model that has nothing to add.
        correction = prediction if error is None else prediction + error
        self._last_prediction = prediction.detach()

        squared = None if error is None else error.pow(2).mean()
        relative = None
        if squared is not None and actual is not None:
            relative = squared / (actual.detach().pow(2).mean() + 1e-8)
        return CerebellarOutput(
            prediction=prediction,
            correction=self.norm(self.dentate(correction)),
            error=squared,
            relative=relative,
        )

    def reset(self) -> None:
        """Forget the outstanding prediction. Called between episodes."""
        self._last_prediction = None

    def prediction_loss(self, predicted: torch.Tensor,
                        actual: torch.Tensor) -> torch.Tensor:
        """The cerebellum's own objective: be less wrong next time."""
        return nn.functional.mse_loss(predicted, actual)


__all__ = ["Cerebellum", "CerebellarOutput", "InferiorOlive"]
