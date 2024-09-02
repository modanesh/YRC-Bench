from __future__ import annotations

import argparse
import os
import random
import uuid
from typing import Dict

import yaml

from .global_configs import set_global_variable


class ConfigDict:

    def __init__(self, **entries):
        self._entries = []
        rec_entries = {}
        for k, v in entries.items():
            if isinstance(v, dict):
                rv = ConfigDict(**v)
            else:
                rv = v
            rec_entries[k] = rv
            self._entries.append(k)
        self.__dict__.update(rec_entries)

    def __str_helper(self, depth):
        lines = []
        for k, v in self.__dict__.items():
            if k == "_entries":
                continue
            if isinstance(v, ConfigDict):
                v_str = v.__str_helper(depth + 1)
                lines.append("%s:\n%s" % (k, v_str))
            else:
                lines.append("%s: %r" % (k, v))
        indented_lines = ["    " * depth + l for l in lines]
        return "\n".join(indented_lines)

    def __str__(self):
        return "ConfigDict {\n%s\n}" % self.__str_helper(1)

    def __repr__(self):
        return "ConfigDict(%r)" % self.__dict__

    def __getattr__(self, name):
        try:
            return self.__dict__[name]
        except KeyError:
            return None

    def to_dict(self) -> Dict:
        ret = {}
        for k in self._entries:
            v = getattr(self, k)
            if isinstance(v, ConfigDict):
                rv = v.to_dict()
            else:
                rv = v
            ret[k] = rv
        return ret

    def clone(self) -> ConfigDict:
        return ConfigDict(**self.to_dict())


def make(file_path: str = None, config_str=None):
    if file_path is not None:
        with open(file_path, "r") as f:
            config = yaml.load(f, Loader=yaml.SafeLoader)
    else:
        assert config_str is not None
        config = yaml.safe_load(config_str)
    config = ConfigDict(**config)
    return config


def merge(config: ConfigDict, args: Dict) -> ConfigDict:
    # Update config with args
    for k, v in args.items():
        if v is not None:
            keys = k.split('.')
            current = config
            for key in keys[:-1]:
                if not hasattr(current, key):
                    setattr(current, key, ConfigDict())
                current = getattr(current, key)
            setattr(current, keys[-1], v)

    # benchmark specific setup
    if config.general.benchmark == 'procgen':
        config.acting_policy.weak.file = os.path.join("YRC", "checkpoints", "procgen", config.acting_policy.weak.env_name, config.acting_policy.weak.file)
        config.acting_policy.strong.file = os.path.join("YRC", "checkpoints", "procgen", config.acting_policy.strong.env_name,
                                                        config.acting_policy.strong.file)
        for k, v in config.environments.procgen.to_dict().items():
            for kk, vv in v.items():
                if kk == "start_level":
                    if k == "val_id" or k == "val_ood":
                        setattr(getattr(config.environments.procgen, k), 'start_level', random.randint(0, 999))
                    elif k == "test":
                        setattr(getattr(config.environments.procgen, k), 'start_level', random.randint(999, 9999))

    elif config.general.benchmark == 'cliport':
        config.acting_policy.weak.file = os.path.join("YRC", "checkpoints", "cliport", config.acting_policy.weak.file)
        cliport_root = os.getenv("CLIPORT_ROOT")
        if cliport_root is None or not os.path.exists(cliport_root) or cliport_root == "":
            raise ValueError("Please set the environment variable CLIPORT_ROOT to the root directory of the Cliport repository.")
        config.environments.cliport.common.assets_root = f'{cliport_root}/cliport/environments/assets'

    uuid_stamp = str(uuid.uuid4())[:8]
    if config.general.benchmark == 'cliport':
        env_name = getattr(config.environments, config.general.benchmark).train.env_name
    else:
        env_name = getattr(config.environments, config.general.benchmark).common.env_name
    if config.algorithm.cls != "NonParam":
        run_name = f"{config.algorithm.cls}-help-{config.help_env.feature_type}-{env_name}-{uuid_stamp}"
    else:
        run_name = f"{config.algorithm.cls}-help-{config.help_policy.NonParam.type}-{env_name}-{uuid_stamp}"
    config.algorithm.run_name = run_name
    logdir = os.path.join('logs', env_name)
    if not (os.path.exists(logdir)):
        os.makedirs(logdir)
    logdir = os.path.join(logdir, run_name)
    if not (os.path.exists(logdir)):
        os.mkdir(logdir)

    if config.algorithm.cls == "PPO":
        config.algorithm.PPO.save_dir = logdir
        config.algorithm.PPO.rollout_length = int(config.algorithm.PPO.rollout_length)
        config.algorithm.PPO.mini_batch_size = int(config.algorithm.PPO.mini_batch_size)
        config.algorithm.PPO.training_steps = int(config.algorithm.PPO.training_steps)
        config.algorithm.PPO.test_steps = int(config.algorithm.PPO.test_steps)
    elif config.algorithm.cls == "DQN":
        config.algorithm.DQN.save_dir = logdir
        config.algorithm.DQN.batch_size = int(config.algorithm.DQN.batch_size)
        config.algorithm.DQN.rollout_length = int(config.algorithm.DQN.rollout_length)
        config.algorithm.DQN.target_update_frequency = int(config.algorithm.DQN.target_update_frequency)
        config.algorithm.DQN.training_steps = int(config.algorithm.DQN.training_steps)
        config.algorithm.DQN.test_steps = int(config.algorithm.DQN.test_steps)
        config.algorithm.DQN.buffer_size = int(config.algorithm.DQN.buffer_size)
    elif config.algorithm.cls == "SVDD":
        config.algorithm.SVDD.save_dir = logdir
        config.algorithm.SVDD.test_steps = int(config.algorithm.SVDD.test_steps)
        config.algorithm.SVDD.rollout_len = int(config.algorithm.SVDD.rollout_len)
    elif config.algorithm.cls == "KDE":
        config.algorithm.KDE.save_dir = logdir
        config.algorithm.KDE.test_steps = int(config.algorithm.KDE.test_steps)
        config.algorithm.KDE.rollout_len = int(config.algorithm.KDE.rollout_len)
    elif config.algorithm.cls == "NonParam":
        config.algorithm.NonParam.save_dir = logdir
        config.algorithm.NonParam.rollout_len = int(config.algorithm.NonParam.rollout_len)

    config.evaluation.validation_steps = int(config.evaluation.validation_steps)

    set_global_variable("benchmark", config.general.benchmark)
    set_global_variable("device", config.general.device)

    config.help_env.reward_max = float(config.help_env.reward_max)
    config.help_env.timeout = float(config.help_env.timeout)

    if config.help_env.feature_type == "T3":
        config.help_policy.DQN.architecture = None
        config.help_policy.PPO.architecture = None

    return config


def parse_args():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--config', type=str, default="YRC/core/configs/config.yaml", help='Path to the config file')
    parser.add_argument('--general.skyline', type=int, default=0, help='Train skyline')

    args, unknown = parser.parse_known_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # Parse remaining arguments
    for arg in unknown:
        if arg.startswith(("-", "--")):
            parser.add_argument(arg.split('=')[0])

    args = parser.parse_args()

    # Update config with command line arguments
    for arg in vars(args):
        value = getattr(args, arg)
        if value is not None and arg != 'config':
            keys = arg.split('.')
            current = config
            for key in keys[:-1]:
                current = current.setdefault(key, {})
            current[keys[-1]] = value

    return ConfigDict(**config)
