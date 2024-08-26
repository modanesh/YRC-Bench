from types import SimpleNamespace

import torch
import numpy as np
from joblib import dump, load


class RandomPolicy:
    def __init__(self, config):
        self.placeholder_config = config.placeholder_config


class NonParamPolicy:
    VALID_TYPES = {"sampled_logit", "max_logit", "sampled_prob", "max_prob", "entropy"}

    def __init__(self, config):
        self.help_percentile = config.help_percentile
        self.type = config.type
        if self.type not in self.VALID_TYPES:
            raise ValueError(f"Invalid type: {self.type}. Must be one of {self.VALID_TYPES}")
        self.rollout_features = None
        self.threshold = None

    def gather_rollouts(self, rollout_len, env):
        num_envs = env.base_env.num_envs
        self.rollout_features = np.zeros(rollout_len)

        with torch.no_grad():
            obs, _ = env.reset()
            for i in range(rollout_len // num_envs):
                info = env.base_env.info
                feature_values = env.weak_policy.get_logits_probs(obs, info)
                feature_index = list(self.VALID_TYPES).index(self.type)
                self.rollout_features[i * num_envs:(i + 1) * num_envs] = feature_values[feature_index]
                obs, _, _, _, _ = env.step(np.zeros((num_envs,)))  # dummy action for gathering rollouts using the weak policy only

    def determine_sampled_logit_threshold(self):
        self.threshold = np.percentile(self.rollout_features, self.help_percentile)

    def save_model(self, save_dir, model=None):
        state_dict = {
            'class_name': self.__class__.__name__,
            'config': {
                'help_percentile': self.help_percentile,
                'type': self.type,
            },
            'rollout_features': self.rollout_features,
            'threshold': self.threshold,
        }
        dump(state_dict, f"{save_dir}/{self.type}.joblib")

    def load_model(self, load_dir):
        state_dict = load(f"{load_dir}/{self.type}.joblib")
        help_percentile = state_dict['config']['help_percentile']
        type = state_dict['config']['type']
        rollout_features = state_dict['rollout_features']
        threshold = state_dict['threshold']
        instance = NonParamPolicy(SimpleNamespace(help_percentile=help_percentile, type=type))
        instance.rollout_features = rollout_features
        instance.threshold = threshold
        return instance
