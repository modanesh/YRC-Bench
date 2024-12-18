import logging

import torch
import gymnasium as gym
import Minigrid.minigrid as minigrid
import YRC.envs.minigrid.wrappers as wrappers
from YRC.envs.minigrid.models import MinigridModel
from YRC.envs.minigrid.policies import MinigridPolicy
from YRC.core.configs.global_configs import get_global_variable
from YRC.envs.minigrid.wrappers import StochasticActionWrapper


def create_env(name, config):
    common_config = config.common
    specific_config = getattr(config, name)

    env = gym.make(
        id=common_config.env_name,
        render_mode=specific_config.render_mode,
    )
    env = StochasticActionWrapper(env)
    env = wrappers.HardResetWrapper(env, seed=specific_config.seed)
    return env


def create_policy(env):
    model = MinigridModel(env)
    model.to(get_global_variable("device"))
    policy = MinigridPolicy(model)
    return policy


def load_policy(path, env, test_env):
    model = MinigridModel(env)
    model.to(get_global_variable("device"))
    model.eval()
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint["model_state_dict"])
    logging.info(f"Loaded model from {path}")

    policy = MinigridPolicy(model)
    policy.eval()
    return policy
