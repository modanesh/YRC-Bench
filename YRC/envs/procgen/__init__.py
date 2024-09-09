import logging

import torch

from procgen import ProcgenEnv
import YRC.envs.procgen.wrappers as wrappers
from YRC.envs.procgen.models import ProcgenModel
from YRC.envs.procgen.policies import ProcgenPolicy
from YRC.core.configs.global_configs import get_global_variable


def create_env(name, config):

    common_config = config.common
    specific_config = getattr(config, name)

    env = ProcgenEnv(
        env_name=common_config.env_name,
        num_envs=common_config.num_envs,
        num_threads=common_config.num_threads,
        num_levels=specific_config.num_levels,
        start_level=specific_config.start_level,
        distribution_mode=specific_config.distribution_mode,
        rand_seed=specific_config.seed,
    )

    env = wrappers.HardResetWrapper(env)
    env = wrappers.VecExtractDictObs(env, "rgb")
    if common_config.normalize_rew:
        env = wrappers.VecNormalize(
            env, ob=False
        )  # normalizing returns, but not the img frames
    env = wrappers.TransposeFrame(env)
    env = wrappers.ScaledFloatFrame(env)

    return env


def load_policy(path, env):
    model = ProcgenModel(env)
    model.to(get_global_variable("device"))
    model.eval()
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint["model_state_dict"])
    logging.info(f"Loaded model from {path}")

    policy = ProcgenPolicy(model)

    return policy
