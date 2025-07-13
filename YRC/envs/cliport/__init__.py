import logging

import re
from lib.cliport.cliport import tasks
from lib.cliport.cliport.environments.environment import Environment
import YRC.envs.cliport.wrappers as wrappers
from YRC.envs.cliport.models import CliportModel
from YRC.envs.cliport.policies import CliportPolicy, CliportPolicyOracle
from YRC.core.configs.global_configs import get_global_variable


def create_env(name, config):
    task_category = {
        "color": ["assembling-kits-seq", "packing-boxes-pairs", "put-block-in-bowl", "stack-block-pyramid-seq", "separating-piles", "towers-of-hanoi-seq"],
        "object": ["packing-google-objects-seq", "packing-google-objects-group"],
    }
    common_config = config.common
    specific_config = getattr(config, name)

    env_name = common_config.env_name
    if env_name in task_category["color"]:
        task_name = f"{env_name}-{specific_config.distribution_mode}-colors"
    elif env_name in task_category["object"]:
        task_name = f"packing-{specific_config.distribution_mode}{env_name[7:]}"

    tsk = tasks.names[task_name]()
    tsk.mode = name if name in ['train', 'test'] else 'val'
    env = Environment(common_config.assets_root, tsk, common_config.disp, common_config.shared_memory, hz=480)
    env = wrappers.HardResetWrapper(
        env,
        specific_config.start_level,
        specific_config.num_levels,
        specific_config.distribution_mode
    )
    env.obs_shape = env.observation_space.shape
    return env


def load_policy(path, env):
    if path is not None:
        pattern = r'^(multi-language-conditioned)-cliport-n(\d+)-(train|val|test)$'
        match = re.match(pattern, path.split("/")[-3])
        env_name = match.group(1)
        n_demos = match.group(2)
        name = f"{match.group(1)}-cliport-n{n_demos}"  # 'multi-language-conditioned-cliport-n1000'

        configs = {"env_name": env_name, "num_demos": n_demos}
        model = CliportModel(name, configs)

        model.load(path)
        model.to(get_global_variable("device"))
        model.eval()
        logging.info(f"Loaded model from {path}")

        policy = CliportPolicy(model)
        policy.eval()
    else:
        policy = {key: CliportPolicyOracle() for key in ["train", "val_sim", "val_true", "test"]}
    return policy
