"""PPO agents: a separate actor (linear or MLP) and a fixed 64x64 critic, Stable-Baselines3 defaults otherwise."""
import copy

import torch
from stable_baselines3 import PPO

from .env import EnergyGridEnv


def policy_kwargs(actor_layers, critic_layers, activation="Tanh", ortho_init=True):
    return dict(net_arch=dict(pi=list(actor_layers), vf=list(critic_layers)),
                activation_fn=torch.nn.Tanh if activation == "Tanh" else torch.nn.ReLU,
                ortho_init=ortho_init)


def new_model(arch, seed, world, cfg, env_overrides=None):
    """A freshly initialised PPO model; `seed` seeds the network initialisation and all global RNGs."""
    options = copy.deepcopy(cfg["ppo"])
    ortho = options.pop("ortho_init")
    env = EnergyGridEnv(phase=world, **{**cfg["environment"], **(env_overrides or {})})
    kwargs = policy_kwargs(cfg["actor_architectures"][arch], cfg["critic_architecture"], cfg["activation"], ortho)
    return PPO("MlpPolicy", env, policy_kwargs=kwargs, seed=seed, device=cfg["device"], verbose=0, **options)


def load_model(path, world, cfg, env_overrides=None, env=None):
    """Load a checkpoint and attach a fresh EnergyGrid environment of the given world."""
    env = env or EnergyGridEnv(phase=world, **{**cfg["environment"], **(env_overrides or {})})
    return PPO.load(path, env=env, device=cfg["device"])


def parameter_counts(model):
    p = model.policy
    actor = sum(t.numel() for t in p.action_net.parameters())
    actor += sum(t.numel() for t in p.mlp_extractor.policy_net.parameters())
    critic = sum(t.numel() for t in p.value_net.parameters())
    critic += sum(t.numel() for t in p.mlp_extractor.value_net.parameters())
    return {"actor": actor, "critic": critic, "total": sum(t.numel() for t in p.parameters())}


def action_probabilities(policy, obs):
    with torch.no_grad():
        return policy.get_distribution(torch.as_tensor(obs, dtype=torch.float32)).distribution.probs.cpu().numpy()
