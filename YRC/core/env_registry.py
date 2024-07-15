import random
import inspect

from YRC.core.utils import set_global_seeds


class EnvRegistry():
    def __init__(self):
        self.all_cfgs = {}
        self.cfgs = None

    def register(self, benchmark, cfgs):
        self.all_cfgs[benchmark] = cfgs

    def set_cfgs(self, name):
        self.cfgs = self.all_cfgs[name]

    def setup_cfgs(self, args):
        if args.benchmark == 'procgen':
            self.cfgs.weak_model_file = args.weak_model_file
            self.cfgs.strong_model_file = args.strong_model_file
            self.cfgs.benchmark = args.benchmark
            self.cfgs.help_policy_type = args.help_policy_type
            self.cfgs.env_name = args.env_name
            self.cfgs.strong_query_cost = args.strong_query_cost
            self.cfgs.switching_cost = args.switching_cost

            self.cfgs.val_env_name = self.cfgs.val_env_name if self.cfgs.val_env_name else self.cfgs.env_name
            self.cfgs.start_level_val = random.randint(0, 9999)
            set_global_seeds(self.cfgs.seed)
            if self.cfgs.start_level == self.cfgs.start_level_val:
                raise ValueError("Seeds for training and validation envs are equal.")

            param_name = args.param_name
            param_class = getattr(self.cfgs, param_name, None)
            print("Found param class", param_class)
            if param_class is None:
                raise ValueError(f"No configuration class found for param_name: {param_name}")
            if inspect.isclass(param_class):
                param_instance = param_class()
                self.cfgs.policy = param_instance
            else:
                raise TypeError(f"{param_name} is not a class")
            for k, v in vars(param_instance).items():
                print("Setting", k, "to", v)
                self.cfgs.policy.__dict__[k] = v

            # self.cfgs.weak_model_file = os.path.join("YRC", "procgen_wrapper", "logs", self.cfgs.env_name, self.cfgs.weak_model_file)
            # self.cfgs.strong_model_file = os.path.join("YRC", "procgen_wrapper", "logs", self.cfgs.env_name, self.cfgs.strong_model_file)
        elif args.benchmark == 'cliport':
            self.cfgs.weak_model_file = args.weak_model_file
            self.cfgs.benchmark = args.benchmark
            self.cfgs.help_policy_type = args.help_policy_type
            self.cfgs.model_path = f"{self.cfgs.results_path}/{self.cfgs.model_task}-{self.cfgs.agent}-n{self.cfgs.n_demos}-train/checkpoints"
            self.cfgs.results_path = f"{self.cfgs.results_path}/{self.cfgs.task}-{self.cfgs.agent}-n{self.cfgs.n_demos}-train/checkpoints"
        return self.cfgs


# make global env registry
env_registry = EnvRegistry()
