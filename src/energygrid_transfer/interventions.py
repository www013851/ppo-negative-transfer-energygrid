"""Interventions applied when the rules change from world A to world B.

Two kinds:
* training-condition interventions (entropy spike, gradual shift, more energy) change how the branch is trained;
* resets re-initialise one component of the agent at the start of the branch. Every reset also replaces Adam with
  a new optimizer with the same settings, so `optimizer_reset` isolates the effect of the optimizer state alone.

Re-initialised layers get the same initialisation as a new SB3 network: orthogonal weights with gain sqrt(2) for
hidden layers, 1 for the value output and 0.01 for the policy output, zero biases. Each component uses its own
seed (offset + training seed) inside a forked torch RNG, so a reset never shifts the global random stream and the
reset branches share every other source of randomness with the plain continuation.
"""
import copy
import math

import torch
from stable_baselines3.common.callbacks import BaseCallback

from .utils import capture_rng, state_hash

TRAINING_CONDITIONS = ["baseline", "entropy_spike", "gradual_shift", "high_energy"]
RESET_COMPONENTS = {
    "optimizer_reset": [],
    "head_optimizer_reset": ["head"],
    "hidden_optimizer_reset": ["hidden"],
    "critic_optimizer_reset": ["critic"],
    "critic_head_optimizer_reset": ["critic", "head"],
}
BRANCH_CONDITIONS = TRAINING_CONDITIONS + list(RESET_COMPONENTS)


def applies_to(condition, arch):
    """A linear policy has no hidden layers, so the hidden-layer reset is undefined for it."""
    return not (condition == "hidden_optimizer_reset" and arch == "linear")


def _orthogonal(layers, gains, seed):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        for layer, gain in zip(layers, gains):
            torch.nn.init.orthogonal_(layer.weight, gain=gain)
            torch.nn.init.zeros_(layer.bias)


def _linear(seq):
    return [m for m in seq if isinstance(m, torch.nn.Linear)]


def apply_reset(model, condition, train_seed, offsets):
    """Re-initialise the components named by `condition` and install a fresh Adam. Returns an audit record."""
    components = RESET_COMPONENTS[condition]
    policy = model.policy
    before = copy.deepcopy(policy.state_dict())
    rng_before = state_hash(capture_rng())
    optimizer = policy.optimizer
    assert len(optimizer.param_groups) == 1
    params = list(policy.parameters())
    assert all(a is b for a, b in zip(params, optimizer.param_groups[0]["params"]))
    settings = {k: copy.deepcopy(v) for k, v in optimizer.param_groups[0].items() if k != "params"}
    defaults = copy.deepcopy(optimizer.defaults)

    expected = set()
    if "hidden" in components:
        hidden = _linear(policy.mlp_extractor.policy_net)
        assert hidden, "hidden-layer reset needs an MLP policy"
        _orthogonal(hidden, [math.sqrt(2)] * len(hidden), offsets["hidden_init"] + train_seed)
        expected |= {k for k in before if k.startswith("mlp_extractor.policy_net.")}
    if "critic" in components:
        hidden = _linear(policy.mlp_extractor.value_net)
        _orthogonal(hidden + [policy.value_net], [math.sqrt(2)] * len(hidden) + [1.0], offsets["critic_init"] + train_seed)
        expected |= {k for k in before if k.startswith(("mlp_extractor.value_net.", "value_net."))}
    if "head" in components:
        _orthogonal([policy.action_net], [0.01], offsets["head_init"] + train_seed)
        expected |= {"action_net.weight", "action_net.bias"}

    policy.optimizer = type(optimizer)([{"params": params, **settings}], **defaults)
    assert not policy.optimizer.state
    after = policy.state_dict()
    changed = {k for k in before if not torch.equal(before[k], after[k])}
    # Biases that were already zero stay zero, so only require that all re-initialised weights changed.
    assert changed <= expected and all(k in changed for k in expected if k.endswith("weight")), (changed, expected)
    assert state_hash(capture_rng()) == rng_before, "a reset must not consume the global random stream"
    return {"condition": condition, "changed_tensors": sorted(changed),
            "policy_before_sha256": state_hash(before), "policy_after_sha256": state_hash(after)}


def gradual_deltas(completed_steps, duration, magnitude):
    """Utilities of resources A and B after `completed_steps` of a linear change from world A to world B."""
    progress = min(1.0, max(0.0, completed_steps / float(duration)))
    return magnitude * (1 - 2 * progress), magnitude * (-1 + 2 * progress)


class GradualShift(BaseCallback):
    """Updates the (unobserved) resource utilities after every environment step."""

    def __init__(self, start, duration, magnitude):
        super().__init__()
        self.start, self.duration, self.magnitude = start, duration, magnitude

    def _on_step(self):
        self.training_env.env_method("set_resource_deltas",
                                     *gradual_deltas(self.num_timesteps - self.start, self.duration, self.magnitude))
        return True
