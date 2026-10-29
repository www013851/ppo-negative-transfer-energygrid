import pytest
import torch

from energygrid_transfer.agents import new_model
from energygrid_transfer.interventions import RESET_COMPONENTS, apply_reset, gradual_deltas
from energygrid_transfer.utils import capture_rng, load_config, state_hash

EXPECTED_PREFIXES = {
    "optimizer_reset": (),
    "head_optimizer_reset": ("action_net.",),
    "hidden_optimizer_reset": ("mlp_extractor.policy_net.",),
    "critic_optimizer_reset": ("mlp_extractor.value_net.", "value_net."),
    "critic_head_optimizer_reset": ("mlp_extractor.value_net.", "value_net.", "action_net."),
}


@pytest.mark.parametrize("condition", list(RESET_COMPONENTS))
def test_reset_changes_only_its_component_and_keeps_rng(condition):
    cfg = load_config()
    model = new_model("mlp16", 11, "B", cfg)
    model.learn(1024)  # non-empty Adam state, weights away from initialisation
    before = {k: v.clone() for k, v in model.policy.state_dict().items()}
    rng = state_hash(capture_rng())
    apply_reset(model, condition, 11, cfg["seed_offsets"])
    after = model.policy.state_dict()
    changed = {k for k in before if not torch.equal(before[k], after[k])}
    prefixes = EXPECTED_PREFIXES[condition]
    if prefixes:
        assert changed and all(k.startswith(prefixes) for k in changed)
    else:
        assert not changed
    assert not model.policy.optimizer.state
    assert state_hash(capture_rng()) == rng


def test_reset_is_deterministic():
    cfg = load_config()
    a, b = new_model("mlp16", 11, "B", cfg), new_model("mlp16", 11, "B", cfg)
    apply_reset(a, "head_optimizer_reset", 22, cfg["seed_offsets"])
    apply_reset(b, "head_optimizer_reset", 22, cfg["seed_offsets"])
    assert state_hash(a.policy.state_dict()) == state_hash(b.policy.state_dict())


def test_hidden_reset_is_undefined_for_linear_policy():
    cfg = load_config()
    with pytest.raises(AssertionError):
        apply_reset(new_model("linear", 11, "B", cfg), "hidden_optimizer_reset", 11, cfg["seed_offsets"])


def test_gradual_shift_goes_from_world_a_to_world_b():
    assert gradual_deltas(0, 100, 30) == (30, -30)
    assert gradual_deltas(50, 100, 30) == (0, 0)
    assert gradual_deltas(500, 100, 30) == (-30, 30)
