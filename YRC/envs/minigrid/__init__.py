import logging

import torch
import gymnasium as gym
import lib.Minigrid.minigrid as minigrid
from lib.Minigrid.minigrid.wrappers import StochasticActionWrapper
import YRC.envs.minigrid.wrappers as wrappers
from YRC.envs.minigrid.models import MinigridModel
from YRC.envs.minigrid.policies import MinigridPolicy
from YRC.core.configs.global_configs import get_global_variable


def create_env(name, config):
    common_config = config.common
    specific_config = getattr(config, name)
    full_env_name = common_config.env_name + specific_config.env_name_suffix
    envs = gym.make_vec(full_env_name, wrappers=(StochasticActionWrapper,), num_envs=common_config.num_envs)
    envs.reset(seed=specific_config.seed)
    envs = wrappers.HardResetWrapper(envs)
    envs.obs_shape = envs.observation_space.spaces['image'].shape[1:]
    return envs


def load_policy(path, env):
    model = MinigridModel(env)
    model.to(get_global_variable("device"))
    model.eval()
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint["model_state"])
    logging.info(f"Loaded model from {path}")

    policy = MinigridPolicy(model, env.num_envs)
    policy.eval()
    return policy
