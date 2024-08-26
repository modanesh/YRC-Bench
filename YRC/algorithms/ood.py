from abc import ABC, abstractmethod
from types import SimpleNamespace

import numpy as np
import torch
from joblib import dump, load
from pyod.models.deep_svdd import DeepSVDD
from pyod.models.kde import KDE

from YRC.core.configs import get_global_variable
from cliport.utils import utils as cliport_utils


class OutlierDetector(ABC):
    def __init__(self, config, in_channels=None, additional_hidden_dim=0, feature_type="T1"):
        self.type = feature_type
        self.in_channels = in_channels
        self.additional_hidden_dim = additional_hidden_dim
        self.n_features = int(self._get_n_features())
        self.contamination = config.contamination
        self.seed = config.seed
        self.clf_name = None

    def _get_n_features(self):
        if self.type == "T3":
            return self.additional_hidden_dim
        else:
            return self.in_channels + self.additional_hidden_dim

    def _init_rollout_obs(self, rollout_len):
        return np.zeros((rollout_len, self.n_features))

    def _update_rollout_obs(self, rollout_obs, i, num_envs, obs, pi_w_hidden):
        if self.type == "T1":
            self._update_t1_rollout_obs(rollout_obs, i, num_envs, obs)
        elif self.type == "T2":
            self._update_t2_rollout_obs(rollout_obs, i, num_envs, obs, pi_w_hidden)
        elif self.type == "T3":
            self._update_t3_rollout_obs(rollout_obs, i, num_envs, pi_w_hidden)

    def _update_t1_rollout_obs(self, rollout_obs, i, num_envs, obs):
        rollout_obs[i * num_envs:(i + 1) * num_envs] = obs.reshape(num_envs, -1)

    def _update_t2_rollout_obs(self, rollout_obs, i, num_envs, obs, pi_w_hidden):
        rollout_obs[i * num_envs:(i + 1) * num_envs] = np.hstack((obs.reshape(num_envs, -1), pi_w_hidden.cpu().numpy()))

    def _update_t3_rollout_obs(self, rollout_obs, i, num_envs, pi_w_hidden):
        rollout_obs[i * num_envs:(i + 1) * num_envs] = pi_w_hidden.cpu().numpy()

    def train(self, X_train):
        clf = self.get_classifier()
        clf.fit(X_train)
        return clf

    @abstractmethod
    def get_classifier(self):
        pass

    def save_model(self, save_dir, clf):
        state_dict = {
            'clf': clf,
            'class_name': self.__class__.__name__,
            'config': {
                'contamination': self.contamination,
                'seed': self.seed,
            },
            'type': self.type,
            'n_features': self.n_features,
            'clf_name': self.clf_name,
            'in_channels': self.in_channels,
            'additional_hidden_dim': self.additional_hidden_dim,
        }
        if isinstance(self, OODDeepSVDD):
            state_dict['config']['use_ae'] = self.use_ae
        dump(state_dict, f"{save_dir}/{self.clf_name}.joblib")

    def load_model(self, load_dir):
        state_dict = load(f"{load_dir}/{self.clf_name}.joblib")
        class_name = state_dict['class_name']
        clf = state_dict['clf']
        config = state_dict['config']
        type = state_dict['type']
        n_features = state_dict['n_features']
        clf_name = state_dict['clf_name']
        in_channels = state_dict['in_channels']
        additional_hidden_dim = state_dict['additional_hidden_dim']

        if class_name == 'OODDeepSVDD':
            instance = OODDeepSVDD(SimpleNamespace(**config), in_channels, additional_hidden_dim, type)
        elif class_name == 'OODKDE':
            instance = OODKDE(SimpleNamespace(**config), in_channels, additional_hidden_dim, type)
        else:
            raise ValueError(f"Unknown class name: {class_name}")

        instance.clf = clf
        return instance

    def predict(self, obs, features):
        if self.type == "T1":
            actions = self.clf.predict(obs)
        elif self.type == "T2":
            actions = self.clf.predict(np.hstack((obs, features.cpu().numpy())))
        elif self.type == "T3":
            actions = self.clf.predict(features.cpu().numpy())
        return actions

    def gather_rollouts(self, rollout_len, env):
        num_envs = env.base_env.num_envs
        with torch.no_grad():
            obs, pi_w_hidden = env.reset()
            rollout_obs = self._init_rollout_obs(rollout_len)
            for i in range(rollout_len // num_envs):
                if get_global_variable("benchmark") == 'cliport':
                    obs_for_update = cliport_utils.get_image(obs)
                else:
                    obs_for_update = obs
                action = np.zeros((num_envs,))  # dummy action for gathering rollouts using the weak policy only
                obs_next, _, _, _, pi_w_hidden_next = env.step(action)
                self._update_rollout_obs(rollout_obs, i, num_envs, obs_for_update, pi_w_hidden)
                obs, pi_w_hidden = obs_next, pi_w_hidden_next
        return rollout_obs


class OODKDE(OutlierDetector):
    def __init__(self, config, in_channels=None, additional_hidden_dim=0, feature_type="T1"):
        super().__init__(config, in_channels, additional_hidden_dim, feature_type)
        self.clf_name = 'KDE'

    def get_classifier(self):
        return KDE(contamination=self.contamination)


class OODDeepSVDD(OutlierDetector):
    def __init__(self, config, in_channels=None, additional_hidden_dim=0, feature_type="T1"):
        super().__init__(config, in_channels, additional_hidden_dim, feature_type)
        self.use_ae = getattr(config, 'use_ae', False)  # Default to False if not provided
        self.clf_name = 'DeepSVDD'

    def get_classifier(self):
        return DeepSVDD(
            n_features=self.n_features,
            use_ae=self.use_ae,
            contamination=self.contamination,
            random_state=self.seed
        )
