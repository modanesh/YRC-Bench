# import YRC.cliport_wrapper.utils as cliport_utils
import YRC.procgen_wrapper.utils as procgen_utils
from YRC.core.utils import to_dict


class Policy:
    def __init__(self, exp_cfg):
        self.exp_cfg = exp_cfg

    def load_acting_policies(self, obs_shape, action_size):
        # if wrapper_type == 'cliport':
        #     return cliport_utils.load(*args, **kwargs)
        # elif wrapper_type == 'procgen':
        if self.exp_cfg.benchmark == 'procgen':
            weak_agent = procgen_utils.load_policy(obs_shape[0], action_size, self.exp_cfg.weak_model_file, self.exp_cfg.device)
            strong_agent = procgen_utils.load_policy(obs_shape[0], action_size, self.exp_cfg.strong_model_file, self.exp_cfg.device)
            return weak_agent, strong_agent

    def setup_help_policy(self, env, env_val, weak_agent, strong_agent, writer, observation_shape):
        # if wrapper_type == 'cliport':
        #     return cliport_utils.define(*args, **kwargs)
        # elif wrapper_type == 'procgen':
        if self.exp_cfg.benchmark == 'procgen':
            storage = procgen_utils.Storage(observation_shape, self.exp_cfg.policy.n_steps, self.exp_cfg.policy.n_envs, self.exp_cfg.device)
            storage_val = procgen_utils.Storage(observation_shape, self.exp_cfg.policy.n_steps, self.exp_cfg.policy.n_envs, self.exp_cfg.device)
            help_model, help_policy = procgen_utils.define_help_policy(env, weak_agent, self.exp_cfg.help_policy_type, self.exp_cfg.device)
            policy_cfgs = to_dict(self.exp_cfg.policy)
            help_algorithm = procgen_utils.algorithm_setup(env, env_val, help_policy, writer, storage, storage_val, self.exp_cfg.device,
                                                           self.exp_cfg.num_checkpoints, model_file=None, hyperparameters=policy_cfgs,
                                                           pi_w=weak_agent, pi_o=strong_agent, help_policy_type=self.exp_cfg.help_policy_type)
        return help_algorithm

    @staticmethod
    def setup_storage(wrapper_type, *args, **kwargs):
        # if wrapper_type == 'cliport':
        #     return cliport_utils.save(*args, **kwargs)
        # elif wrapper_type == 'procgen':
        storage = procgen_utils.Storage(*args, **kwargs)
        storage_val = procgen_utils.Storage(*args, **kwargs)
        return storage, storage_val
