import numpy as np
import pytest

from energygrid_transfer.env import EnergyGridEnv


def test_reset_is_seeded_and_cells_are_distinct():
    a, _ = EnergyGridEnv().reset(seed=5)
    b, _ = EnergyGridEnv().reset(seed=5)
    assert np.array_equal(a, b)
    env = EnergyGridEnv()
    env.reset(seed=7)
    assert len({tuple(env.agent_pos), tuple(env.pos_a), tuple(env.pos_b)}) == 3


@pytest.mark.parametrize("phase,eat,delta", [("A", "A", 30), ("A", "B", -30), ("B", "A", -30), ("B", "B", 30)])
def test_resource_utility_is_inverted_between_worlds(phase, eat, delta):
    env = EnergyGridEnv(phase=phase, initial_energy=50)
    env.reset(seed=0)
    env.agent_pos, env.pos_a, env.pos_b = [5, 5], [9, 9], [0, 0]
    if eat == "A":
        env.pos_a = [6, 5]
    else:
        env.pos_b = [6, 5]
    _, _, _, _, info = env.step(3)  # step right onto the resource
    assert env.energy == 50 - 1 + delta
    assert info["beneficial_consumed"] == int(delta > 0)
    assert info["harmful_consumed"] == int(delta < 0)


def test_observation_does_not_reveal_the_world():
    a, b = EnergyGridEnv(phase="A"), EnergyGridEnv(phase="B")
    assert np.array_equal(a.reset(seed=3)[0], b.reset(seed=3)[0])


def test_reward_and_termination():
    env = EnergyGridEnv(initial_energy=2)
    env.reset(seed=1)
    env.pos_a, env.pos_b = [-5, -5], [-6, -6]  # unreachable cells: no food
    _, r1, t1, _, _ = env.step(0)
    _, r2, t2, _, _ = env.step(0)
    assert (r1, t1) == (1.0, False)
    assert (r2, t2) == (0.0, True)


def test_walls_block_movement():
    env = EnergyGridEnv()
    env.reset(seed=0)
    env.pos_a, env.pos_b = [9, 9], [9, 8]
    env.agent_pos = [0, 0]
    env.step(2)
    env.step(1)
    assert env.agent_pos == [0, 0]
