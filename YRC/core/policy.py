import YRC.cliport_wrapper.utils as cliport_utils
import YRC.procgen_wrapper.utils as procgen_utils
from YRC.core.utils import to_dict, algorithm_setup


class Policy:
    def __init__(self, exp_cfg):
        self.exp_cfg = exp_cfg

    def load_acting_policies(self, kwargs):
        if self.exp_cfg.benchmark == 'procgen':
            weak_agent = procgen_utils.load_policy(kwargs['obs_shape'][0], kwargs['action_size'],
                                                   self.exp_cfg.weak_model_file, self.exp_cfg.device)
            strong_agent = procgen_utils.load_policy(kwargs['obs_shape'][0], kwargs['action_size'],
                                                     self.exp_cfg.strong_model_file, self.exp_cfg.device)
            return weak_agent, strong_agent
        elif self.exp_cfg.benchmark == 'cliport':
            weak_agent = cliport_utils.load_weak_policy(self.exp_cfg)
            return weak_agent, None

    def setup_help_policy(self, env, additional_var, weak_agent, strong_agent, writer, observation_shape):
        # additional_var is "env_val" for procgen and "task" for cliport
        if self.exp_cfg.benchmark == 'procgen':
            storage = procgen_utils.ReplayBuffer(observation_shape, self.exp_cfg.policy.n_steps,
                                                 self.exp_cfg.policy.n_envs, self.exp_cfg.device)
            storage_val = procgen_utils.ReplayBuffer(observation_shape, self.exp_cfg.policy.n_steps,
                                                     self.exp_cfg.policy.n_envs, self.exp_cfg.device)

            _, help_policy = procgen_utils.define_help_policy(env, weak_agent, self.exp_cfg.help_policy_type,
                                                              self.exp_cfg.device)
            policy_cfgs = to_dict(self.exp_cfg.policy)
            help_algorithm = algorithm_setup(env, True, additional_var, help_policy, writer, storage,
                                             storage_val, self.exp_cfg.device,
                                             self.exp_cfg.num_checkpoints, hyperparameters=policy_cfgs,
                                             pi_w=weak_agent, pi_o=strong_agent,
                                             help_policy_type=self.exp_cfg.help_policy_type)
        elif self.exp_cfg.benchmark == 'cliport':
            action_shape = 2
            storage = cliport_utils.ReplayBuffer(observation_shape, action_shape, self.exp_cfg.buffer_size,
                                                 self.exp_cfg.policy.n_envs, device=self.exp_cfg.device)
            storage_val = cliport_utils.ReplayBuffer(observation_shape, action_shape, self.exp_cfg.buffer_size,
                                                     self.exp_cfg.policy.n_envs, device=self.exp_cfg.device)

            _, help_policy = cliport_utils.define_help_policy(env, weak_agent, self.exp_cfg.help_policy_type,
                                                              self.exp_cfg.device)
            policy_cfgs = to_dict(self.exp_cfg.policy)
            help_algorithm = algorithm_setup(env, False, additional_var, help_policy, writer, storage,
                                             storage_val, self.exp_cfg.device,
                                             self.exp_cfg.num_checkpoints, hyperparameters=policy_cfgs,
                                             pi_w=weak_agent, pi_o=strong_agent,
                                             help_policy_type=self.exp_cfg.help_policy_type)
        return help_algorithm
