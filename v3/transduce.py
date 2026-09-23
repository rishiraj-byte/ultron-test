"""
transduce.py -- the sense organs. Where the world stops being numbers.

THE ONLY PLACE A FROZEN PRETRAINED NETWORK SURVIVES INTO v3, and the boundary
is drawn on purpose. A retina and a cochlea are fixed transducers: you did not
learn them from experience, development specified them, and they convert
physical energy into spike trains before any cortex is involved. A frozen
encoder doing the same job at the same position is defensible in a way that a
frozen network in the middle of the circuit is not.

So ResNet, Whisper and MiniLM stay, and Qwen does not. Qwen sat inside
frontal, mid-circuit, and there is no story on which prefrontal cortex arrives
pre-loaded with a language model's weights. Frontal in v3 has no pretrained
anything and learns entirely from local plasticity.

Downstream of this file there are no floats anywhere. Every signal in the
system is a spike.

RECEPTOR ADAPTATION is kept, and it is a limitation rather than a feature. A
constant stimulus produces a burst at onset and then fades, which is why you
stop noticing a smell, the weight of your clothes, or a steady hum. It costs
the system the ability to report a truly static input, and that cost is real
in you too.
"""

from __future__ import annotations

import torch

from v3 import config_v3 as cfg


def poisson_spikes(rate_hz: torch.Tensor, steps: int,
                   adapt: bool = True) -> torch.Tensor:
    """Turn per-unit firing rates into a spike train.

    `rate_hz` is (batch, units). Returns (steps, batch, units) binary.

    Poisson rather than regular firing because cortical spiking is close to
    Poisson, and because a regular train would let a downstream cell rely on
    exact periodicity that real input does not have.
    """
    batch, units = rate_hz.shape
    device = rate_hz.device
    probability = (rate_hz * cfg.DT_MS / 1000.0).clamp(0.0, 1.0)

    train = torch.zeros(steps, batch, units, device=device)
    adaptation = torch.zeros(batch, units, device=device)
    decay = cfg.DT_MS / cfg.ADAPTATION_TAU_MS

    for t in range(steps):
        effective = probability * (1.0 - cfg.ADAPTATION_STRENGTH * adaptation)
        fired = (torch.rand(batch, units, device=device) < effective).float()
        train[t] = fired
        if adapt:
            #  Each spike deepens the cell's own adaptation, which then decays.
            adaptation = adaptation + decay * (fired - adaptation)
    return train


def features_to_rates(features: torch.Tensor, units: int) -> torch.Tensor:
    """Map an encoder's output onto a population's firing rates.

    Rank-based rather than linear, so the population's overall activity is the
    same whatever the encoder's output scale happens to be. A retina does not
    change its total spike budget because the room got brighter -- it
    renormalises, and so does this.
    """
    #  POOL, DO NOT TRUNCATE.
    #
    #  This used to flatten the encoder output and keep the first `units`
    #  numbers. For a 49x512 ResNet grid that is 256 of 25,088 values -- one
    #  percent of the feature, and specifically the first few spatial
    #  positions of the image rather than a summary of it. Measured, class
    #  separability fell from 0.384 to 0.247 before a single spike was
    #  generated, and the brain downstream then sat at chance no matter how
    #  long it trained. It was not failing to learn; it was being handed an
    #  input with most of the answer already thrown away.
    #
    #  A retina does not sample one corner of the visual field and discard
    #  the rest. It pools.
    if features.dim() == 3:
        #  Average over spatial positions first: a summary of the whole grid.
        features = features.mean(dim=1)
    flat = features.reshape(features.shape[0], -1)

    if flat.shape[1] < units:
        repeats = (units + flat.shape[1] - 1) // flat.shape[1]
        flat = flat.repeat(1, repeats)[:, :units]
    elif flat.shape[1] > units:
        #  Average adjacent channels into `units` groups, so every input
        #  dimension contributes to exactly one receptor.
        group = flat.shape[1] // units
        flat = flat[:, :group * units].reshape(
            flat.shape[0], units, group).mean(dim=2)

    #  Rank within each example, scaled into a firing rate. The strongest
    #  feature drives the fastest cell.
    order = flat.argsort(dim=1).argsort(dim=1).float()
    normalised = order / max(units - 1, 1)
    return normalised * cfg.INPUT_RATE_MAX_HZ


class SenseOrgans:
    """Frozen encoders plus the rate coding that follows them."""

    def __init__(self, device: str | None = None):
        self.device = device or cfg.resolve_device()
        self._vision = None
        self._text = None

    # -- vision ---------------------------------------------------------
    def _load_vision(self):
        if self._vision is None:
            import torchvision
            model = torchvision.models.resnet50(
                weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V2)
            model.fc = torch.nn.Identity()
            model.eval().to(self.device)
            for p in model.parameters():
                p.requires_grad_(False)
            self._vision = model
        return self._vision

    def see(self, images: torch.Tensor, units: int, steps: int) -> torch.Tensor:
        with torch.no_grad():
            features = self._load_vision()(images.to(self.device))
        return poisson_spikes(features_to_rates(features, units), steps)

    # -- pre-encoded features (the cached streams v2 already built) -------
    def from_features(self, features: torch.Tensor, units: int,
                      steps: int) -> torch.Tensor:
        """Spike train from an already-computed feature vector.

        Lets v3 read exactly the same cached encoder output v2 trained on, so
        a comparison between them is about the brain and not about the eyes.
        """
        return poisson_spikes(
            features_to_rates(features.to(self.device), units), steps)


__all__ = ["SenseOrgans", "poisson_spikes", "features_to_rates"]
