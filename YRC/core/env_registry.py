import os
import random

import yaml

from YRC.procgen_wrapper import utils


class EnvRegistry():
    def __init__(self):
        self.cfgs = None

    def register(self, cfgs):
        self.cfgs = cfgs

    def get_cfgs(self, args):
        self.cfgs.weak_model_file = args.weak_model_file
        self.cfgs.strong_model_file = args.strong_model_file
        self.cfgs.benchmark = args.benchmark
        self.cfgs.help_policy_type = args.help_policy_type

        with open(os.path.join(self.cfgs.config_path, 'config.yml')) as f:
            alg_configs = yaml.load(f, Loader=yaml.FullLoader)[self.cfgs.param_name]

        if self.cfgs.benchmark == 'procgen' and self.cfgs.strong_model_file is None:
            raise ValueError("Strong model file not provided for procgen.")

        self.cfgs.val_env_name = self.cfgs.val_env_name if self.cfgs.val_env_name else self.cfgs.env_name
        self.cfgs.start_level_val = random.randint(0, 9999)
        utils.set_global_seeds(self.cfgs.seed)
        if self.cfgs.start_level == self.cfgs.start_level_val:
            raise ValueError("Seeds for training and validation envs are equal.")
        for k, v in alg_configs.items():
            self.cfgs.policy.__dict__[k] = v

        self.cfgs.weak_model_file = os.path.join("YRC", "procgen_wrapper", "logs", self.cfgs.env_name, self.cfgs.weak_model_file)
        self.cfgs.strong_model_file = os.path.join("YRC", "procgen_wrapper", "logs", self.cfgs.env_name, self.cfgs.strong_model_file)

        return self.cfgs


# make global env registry
env_registry = EnvRegistry()
