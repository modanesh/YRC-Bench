from . import utils


class Policy:
    def __init__(self, exp_cfg):
        self.exp_cfg = exp_cfg

    def load_acting_policies(self, kwargs):
        if self.exp_cfg.benchmark == 'procgen':
            weak_agent = utils.load_policy(kwargs['obs_shape'][0], kwargs['action_size'],
                                           self.exp_cfg.weak_model_file, self.exp_cfg.device)
            strong_agent = utils.load_policy(kwargs['obs_shape'][0], kwargs['action_size'],
                                             self.exp_cfg.strong_model_file, self.exp_cfg.device)
            return weak_agent, strong_agent
        elif self.exp_cfg.benchmark == 'cliport':
            weak_agent = utils.load_weak_policy(self.exp_cfg)
            return weak_agent, None

    def setup_help_policy(self, env, env_val, task, weak_agent, writer, observation_shape):
        if self.exp_cfg.benchmark == 'procgen':
            storage = utils.ProcgenReplayBuffer(observation_shape, self.exp_cfg.policy.n_steps,
                                                self.exp_cfg.policy.n_envs, self.exp_cfg.device)
            storage_val = utils.ProcgenReplayBuffer(observation_shape, self.exp_cfg.policy.n_steps,
                                                    self.exp_cfg.policy.n_envs, self.exp_cfg.device)

            _, help_policy = utils.procgen_define_help_policy(env, weak_agent, self.exp_cfg.help_policy_type, self.exp_cfg.device)
        elif self.exp_cfg.benchmark == 'cliport':
            action_shape = 2
            storage = utils.CliportReplayBuffer(observation_shape, action_shape, self.exp_cfg.buffer_size,
                                                self.exp_cfg.policy.n_envs, device=self.exp_cfg.device)
            storage_val = utils.CliportReplayBuffer(observation_shape, action_shape, self.exp_cfg.buffer_size,
                                                    self.exp_cfg.policy.n_envs, device=self.exp_cfg.device)

            _, help_policy = utils.cliport_define_help_policy(env, weak_agent, self.exp_cfg.help_policy_type, self.exp_cfg.device)
        policy_cfgs = utils.to_dict(self.exp_cfg.policy)
        help_algorithm = utils.algorithm_setup(env, env_val, task, help_policy, writer, storage, storage_val, self.exp_cfg.device,
                                               self.exp_cfg.num_checkpoints, hyperparameters=policy_cfgs, help_policy_type=self.exp_cfg.help_policy_type)
        return help_algorithm
