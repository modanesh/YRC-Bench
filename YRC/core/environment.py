import YRC.procgen_wrapper.utils as procgen_utils


# import YRC.cliport_wrapper.utils as cliport_utils


class Environment:
    def __init__(self, exp_cfg):
        self.exp_cfg = exp_cfg

    def make(self, weak_agent, strong_agent):
        # if wrapper_type == 'cliport':
        #     return cliport_utils(*args, **kwargs)
        if self.exp_cfg.benchmark == 'procgen':
            reward_range = getattr(getattr(self.exp_cfg.reward_range, self.exp_cfg.env_name), self.exp_cfg.distribution_mode)
            max_rew, timeout = reward_range['max'], reward_range['timeout']
            env = self.create_env(weak_agent, strong_agent, max_rew, timeout, self.exp_cfg.env_name, self.exp_cfg.start_level)
            env_val = self.create_env(weak_agent, strong_agent, max_rew, timeout, self.exp_cfg.val_env_name, self.exp_cfg.start_level_val)
        return env, env_val

    def create_env(self, weak_agent, strong_agent, reward_max, timeout, env_name, start_level):
        env = procgen_utils.create_env(self.exp_cfg.policy.n_steps,
                                       env_name,
                                       start_level,
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
                                       reward_max,
                                       timeout
                                       )
        return env

    def get_env_configs(self):
        # if args.wrapper_type == 'cliport':
        #     return cliport_utils.create_env(*args, **kwargs, get_configs=True)
        # elif args.wrapper_type == 'procgen':
        if self.exp_cfg.benchmark == 'procgen':
            return procgen_utils.create_env(self.exp_cfg.policy.n_steps,
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
