"""EnergyGrid: a 10x10 foraging world with two resources whose utility can be inverted.

World A: resource A restores +30 energy, resource B removes 30.  World B: the signs are swapped.
The observation never contains the sign of a resource, so a policy cannot tell the worlds apart
from a single state; it can only learn the difference from experience.

Observation (7 floats): x/9, y/9, (Ax-x)/9, (Ay-y)/9, (Bx-x)/9, (By-y)/9, energy/max_energy.
Actions: 0 up (y+1), 1 down (y-1), 2 left (x-1), 3 right (x+1); moves into a wall leave the agent in place.
Reward: +1 for every step the agent survives, so the episode return equals the episode length
minus one if the agent dies and equals the length if the 500-step limit is reached.
"""
import gymnasium as gym
from gymnasium import spaces
import numpy as np


class EnergyGridEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, phase="A", size=10, initial_energy=100, max_energy=100, step_cost=1,
                 resource_magnitude=30, max_episode_steps=500):
        super().__init__()
        assert phase in ("A", "B"), f"Invalid phase: {phase}"
        self.phase = phase
        self.resource_deltas = None  # set only by the gradual-shift schedule
        self.size = size
        self.initial_energy = initial_energy
        self.max_energy = max_energy
        self.step_cost = step_cost
        self.resource_magnitude = resource_magnitude
        self.max_episode_steps = max_episode_steps

        self.action_space = spaces.Discrete(4)
        low = np.array([0.0, 0.0, -1.0, -1.0, -1.0, -1.0, 0.0], dtype=np.float32)
        high = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float32)
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

        # Cell coordinates [x, y]; set by reset().
        self.agent_pos: list[int] = [0, 0]
        self.pos_a: list[int] = [0, 0]
        self.pos_b: list[int] = [0, 0]
        self.energy = 0
        self.step_count = 0

    def _get_obs(self):
        norm = float(self.size - 1)
        ax, ay = self.agent_pos
        pos_a_x, pos_a_y = self.pos_a
        pos_b_x, pos_b_y = self.pos_b
        return np.array([
            ax / norm,
            ay / norm,
            (pos_a_x - ax) / norm,
            (pos_a_y - ay) / norm,
            (pos_b_x - ax) / norm,
            (pos_b_y - ay) / norm,
            max(0.0, self.energy) / float(self.max_energy),
        ], dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # Agent and both resources start in three distinct random cells.
        chosen = self.np_random.choice(self.size * self.size, size=3, replace=False)
        self.agent_pos = [int(chosen[0] % self.size), int(chosen[0] // self.size)]
        self.pos_a = [int(chosen[1] % self.size), int(chosen[1] // self.size)]
        self.pos_b = [int(chosen[2] % self.size), int(chosen[2] // self.size)]
        self.energy = self.initial_energy
        self.step_count = 0
        info = {"consumed_a": 0, "consumed_b": 0, "beneficial_consumed": 0, "harmful_consumed": 0}
        return self._get_obs(), info

    def set_resource_deltas(self, delta_a, delta_b):
        """Override the energy effects (used by the gradual-shift schedule); never observable."""
        self.resource_deltas = (float(delta_a), float(delta_b))

    def get_resource_deltas(self):
        if self.resource_deltas is not None:
            return self.resource_deltas
        magnitude = self.resource_magnitude
        return (magnitude, -magnitude) if self.phase == "A" else (-magnitude, magnitude)

    def _respawn_resource(self, resource_type):
        """A consumed resource reappears in a cell free of the agent and the other resource."""
        occupied = {tuple(self.agent_pos)}
        occupied.add(tuple(self.pos_b) if resource_type == "A" else tuple(self.pos_a))
        free_cells = [(x, y) for y in range(self.size) for x in range(self.size) if (x, y) not in occupied]
        new_pos = list(free_cells[self.np_random.integers(0, len(free_cells))])
        if resource_type == "A":
            self.pos_a = new_pos
        else:
            self.pos_b = new_pos

    def step(self, action):
        self.step_count += 1
        x, y = self.agent_pos
        if action == 0:
            y = min(self.size - 1, y + 1)
        elif action == 1:
            y = max(0, y - 1)
        elif action == 2:
            x = max(0, x - 1)
        elif action == 3:
            x = min(self.size - 1, x + 1)
        self.agent_pos = [x, y]

        self.energy -= self.step_cost
        consumed_a = consumed_b = False
        delta_a, delta_b = self.get_resource_deltas()
        if self.agent_pos == self.pos_a:
            consumed_a = True
            self.energy += delta_a
            self._respawn_resource("A")
        if self.agent_pos == self.pos_b:
            consumed_b = True
            self.energy += delta_b
            self._respawn_resource("B")
        self.energy = min(self.energy, self.max_energy)

        terminated = self.energy <= 0
        truncated = (not terminated) and self.step_count >= self.max_episode_steps
        reward = 0.0 if terminated else 1.0

        info = {
            "consumed_a": int(consumed_a),
            "consumed_b": int(consumed_b),
            "beneficial_consumed": int(consumed_a and delta_a > 0) + int(consumed_b and delta_b > 0),
            "harmful_consumed": int(consumed_a and delta_a < 0) + int(consumed_b and delta_b < 0),
            "energy": self.energy,
            "step_count": self.step_count,
        }
        return self._get_obs(), reward, terminated, truncated, info
