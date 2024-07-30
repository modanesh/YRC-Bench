from .utils import cliport_environment_setup, procgen_environment_setup


class Environment:
    def __init__(self, exp_cfg):
        self.exp_cfg = exp_cfg

    def make(self, weak_agent, strong_agent):
        if self.exp_cfg.benchmark == 'procgen':
            reward_range = getattr(getattr(self.exp_cfg.reward_range, self.exp_cfg.env_name), self.exp_cfg.distribution_mode)
            max_rew, timeout = reward_range['max'], reward_range['timeout']
            env = procgen_environment_setup(self.exp_cfg.policy.n_steps,
                                            self.exp_cfg.env_name,
                                            self.exp_cfg.start_level,
                                            self.exp_cfg.num_levels,
                                            self.exp_cfg.distribution_mode,
                                            self.exp_cfg.num_threads,
                                            self.exp_cfg.random_percent,
                                            self.exp_cfg.step_penalty,
                                            self.exp_cfg.key_penalty,
                                            self.exp_cfg.rand_region,
                                            self.exp_cfg.policy.normalize_rew,
                                            weak_agent,
                                            strong_agent,
                                            False,
                                            self.exp_cfg.strong_query_cost,
                                            self.exp_cfg.switching_cost,
                                            max_rew,
                                            timeout,
                                            self.exp_cfg.help_policy_type,
                                            self.exp_cfg.device
                                            )
            env_val = procgen_environment_setup(self.exp_cfg.policy.n_steps,
                                            self.exp_cfg.val_env_name,
                                            self.exp_cfg.start_level_val,
                                            self.exp_cfg.num_levels,
                                            self.exp_cfg.distribution_mode,
                                            self.exp_cfg.num_threads,
                                            self.exp_cfg.random_percent,
                                            self.exp_cfg.step_penalty,
                                            self.exp_cfg.key_penalty,
                                            self.exp_cfg.rand_region,
                                            self.exp_cfg.policy.normalize_rew,
                                            weak_agent,
                                            strong_agent,
                                            False,
                                            self.exp_cfg.strong_query_cost,
                                            self.exp_cfg.switching_cost,
                                            max_rew,
                                            timeout,
                                            self.exp_cfg.help_policy_type,
                                            self.exp_cfg.device
                                            )
            return env, env_val, None
        elif self.exp_cfg.benchmark == 'cliport':
            env, task = cliport_environment_setup(self.exp_cfg.assets_root, weak_agent,
                                                  self.exp_cfg.strong_query_cost,
                                                  self.exp_cfg.switching_cost, self.exp_cfg.disp,
                                                  self.exp_cfg.shared_memory, self.exp_cfg.task,
                                                  self.exp_cfg.help_policy_type, self.exp_cfg.device
                                                  )
            return env, None, task

    def get_env_configs(self):
        if self.exp_cfg.benchmark == 'procgen':
            return procgen_environment_setup(self.exp_cfg.policy.n_steps,
                                             self.exp_cfg.env_name,
                                             self.exp_cfg.start_level,
                                             self.exp_cfg.num_levels,
                                             self.exp_cfg.distribution_mode,
                                             self.exp_cfg.num_threads,
                                             self.exp_cfg.random_percent,
                                             self.exp_cfg.step_penalty,
                                             self.exp_cfg.key_penalty,
                                             self.exp_cfg.rand_region,
                                             self.exp_cfg.policy.normalize_rew,
                                             get_configs=True
                                             )
        elif self.exp_cfg.benchmark == 'cliport':
            observation_shape = (6, 320, 160)
            action_size = 2
            return observation_shape, action_size
