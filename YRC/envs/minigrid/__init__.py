import logging

import torch
import gymnasium as gym
import Minigrid.minigrid as minigrid
from Minigrid.minigrid.wrappers import StochasticActionWrapper
import YRC.envs.minigrid.wrappers as wrappers
from YRC.envs.minigrid.models import MinigridModel
from YRC.envs.minigrid.policies import MinigridPolicy
from YRC.core.configs.global_configs import get_global_variable


def create_env(name, config):
    common_config = config.common
    specific_config = getattr(config, name)
    envs = gym.make_vec(common_config.env_name, wrappers=(StochasticActionWrapper,), num_envs=common_config.num_envs)
    envs.reset(seed=specific_config.seed)
    envs = wrappers.HardResetWrapper(envs)
    return envs


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
    model.load_state_dict(checkpoint["model_state"])
    logging.info(f"Loaded model from {path}")

    policy = MinigridPolicy(model, env.num_envs)
    policy.eval()
    return policy
