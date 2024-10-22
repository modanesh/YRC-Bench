import os
import numpy as np
from copy import deepcopy as dc

import torch
import logging
from torch.distributions.categorical import Categorical
from YRC.core import Policy
from pyod.models import kde, deep_svdd
from joblib import dump, load
import torch.nn as nn
import torch.nn.functional as F


class OODPolicy(Policy):
    def __init__(self, config, env):
        self.args = config.coord_policy
        self.agent = env.weak_agent
        self.params = {"threshold": 0.0, "explore_temp": 1.0}
        self.clf = None
        self.clf_name = None

    def gather_rollouts(self, env, num_rollouts):
        assert num_rollouts % env.num_envs == 0
        observations = []
        for i in range(num_rollouts // env.num_envs):
            observations.extend(self._rollout_once(env))
        if self.args.method == "DeepSVDD":
            self.clf.n_features = observations[0].shape[0]
        return observations

    def _rollout_once(self, env):
        def sample_action(logit):
            dist = Categorical(logits=logit / self.params["explore_temp"])
            return dist.sample().cpu().numpy()

        agent = self.agent
        agent.eval()
        obs = env.reset()
        has_done = np.array([False] * env.num_envs)
        observations = []

        while not has_done.all():
            logit = agent.forward(obs["env_obs"])

            if env.num_envs == 1:
                # todo: check for cliport
                observations.append(obs["env_obs"].item())
            else:
                for i in range(env.num_envs):
                    if not has_done[i]:
                        obs_features = self.feature_extractor(obs["env_obs"])
                        observations.extend(obs_features.detach().numpy())

            action = sample_action(logit)
            obs, reward, done, info = env.step(action)
            has_done |= done

        return observations

    def update_params(self, params):
        self.params = dc(params)

    def act(self, obs, greedy=False):
        obs_features = self.feature_extractor(obs["env_obs"]).detach().numpy()
        score = self.clf.decision_function(obs_features)
        action = (score < self.params["threshold"]).astype(int)
        return action

    def initialize_ood_detector(self, args):
        if self.args.method == "KDE":
            self.clf_name = 'KDE'
            self.clf = kde.KDE(contamination=args.contamination)
        elif self.args.method == "DeepSVDD":
            self.clf_name = 'DeepSVDD'
            self.clf = deep_svdd.DeepSVDD(
                            n_features=None,
                            use_ae=args.use_ae,
                            contamination=args.contamination,
                        )
        else:
            raise ValueError(f"Unknown OOD detector type: {args.ood_detector}")

    def save_model(self, name, save_dir):
        save_path = os.path.join(save_dir, f"{name}.joblib")
        state_dict = {
            'clf': self.clf,
            'class_name': self.__class__.__name__,
            'config': {
                'contamination': self.clf.contamination,
            },
            'clf_name': self.clf_name,
        }
        if type(self.clf) == deep_svdd.DeepSVDD:
            state_dict['config']['use_ae'] = self.clf.use_ae
        dump(state_dict, save_path)
        logging.info(f"Saved model to {save_path}")

    def feature_extractor(self, obs):
        """
        Extract features from the observation using a simple CNN.

        Note that it may overlook "important" features in the observation as it is a simple CNN without any training.

        obs: input observation
        returns: extracted features (batch_size, num_features)
        """

        class SimpleCNN(nn.Module):
            def __init__(self):
                super(SimpleCNN, self).__init__()
                self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1)
                self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
                self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
                self.fc1 = nn.Linear(64 * 8 * 8, 128)  # Fully connected layer
                self.fc2 = nn.Linear(128, 64)  # Output layer for feature extraction

            def forward(self, x):
                x = F.relu(self.conv1(x))
                x = F.max_pool2d(x, 2)  # Max pooling layer, reduces size to 32x32
                x = F.relu(self.conv2(x))
                x = F.max_pool2d(x, 2)  # Reduces size to 16x16
                x = F.relu(self.conv3(x))
                x = F.max_pool2d(x, 2)  # Reduces size to 8x8
                x = x.reshape(x.size(0), -1)  # (batch_size, 64 * 8 * 8)
                x = F.relu(self.fc1(x))
                x = self.fc2(x)  # Extracted feature vector of size 64
                return x

        cnn = SimpleCNN()
        if not isinstance(obs, torch.Tensor):
            obs = torch.tensor(obs, dtype=torch.float32)
        features = cnn(obs)
        return features

