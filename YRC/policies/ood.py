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
from YRC.core.configs.global_configs import get_global_variable


class OODPolicy(Policy):
    def __init__(self, config, env):
        self.args = config.coord_policy
        self.agent = env.weak_agent
        self.params = {"threshold": 0.0, "explore_temp": 1.0}
        self.clf = None
        self.clf_name = None
        self.cnn = None

    def gather_rollouts(self, env, num_rollouts):
        assert num_rollouts % env.num_envs == 0
        observations = []
        for i in range(2):
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

            if get_global_variable("benchmark") == "cliport":
                obs_features = self.feature_extractor(obs["env_obs"]['img'])
                observations.extend(obs_features)
            else:
                for i in range(env.num_envs):
                    if not has_done[i]:
                        obs_features = self.feature_extractor(obs["env_obs"])
                        observations.extend(obs_features)

            action = sample_action(logit)
            obs, reward, done, info = env.step(action)
            has_done |= done

        return observations

    def update_params(self, params):
        self.params = dc(params)

    def act(self, obs, greedy=False):
        if get_global_variable("benchmark") == "cliport":
            obs_features = self.feature_extractor(obs["env_obs"]['img'])
        else:
            obs_features = self.feature_extractor(obs["env_obs"])
        score = self.clf.decision_function(obs_features)
        action = (score < self.params["threshold"]).astype(int)
        if 0 not in action and 1 not in action:
            print("No action is selected as OOD")
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
        if not os.path.exists(f"{save_dir}/cnn.ckpt"):
            torch.save(self.cnn.state_dict(), f"{save_dir}/cnn.ckpt")
        logging.info(f"Saved model to {save_path}")

    def feature_extractor(self, obs):
        """
        Extract features from the observation using a simple CNN.

        Note that it may overlook "important" features in the observation as it is a simple CNN without any training.

        obs: input observation
        returns: extracted features (batch_size, num_features)
        """

        class SimpleCNN(nn.Module):
            def __init__(self, input_shape):
                super(SimpleCNN, self).__init__()
                self.input_shape = input_shape  # shape as (batch_size, channels, height, width, [depth])
                channels = self.input_shape[1]

                # Define layers dynamically
                self.conv1 = nn.Conv2d(channels, 32, kernel_size=3, stride=1, padding=1)  # 32 filters
                self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)  # 64 filters
                self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

                # Compute output size after convolutional layers to flatten properly
                self.flattened_size = self._get_flattened_size()

                # Fully connected layer after the flattening
                self.fc1 = nn.Linear(self.flattened_size, 128)  # Adjust output size based on your needs

            def _get_flattened_size(self):
                # Pass a dummy input to calculate the size after convolution
                with torch.no_grad():
                    dummy_input = torch.zeros(1, *self.input_shape[1:])  # Shape without batch size
                    x = self.pool(F.relu(self.conv1(dummy_input)))
                    x = self.pool(F.relu(self.conv2(x)))
                    return x.numel()  # Flattened size

            def forward(self, x):
                # Forward pass through conv and pooling layers
                x = self.pool(F.relu(self.conv1(x)))
                x = self.pool(F.relu(self.conv2(x)))

                # Flatten before fully connected layers
                x = x.reshape(x.size(0), -1)  # Flatten the output from conv layers
                x = F.relu(self.fc1(x))

                return x

        if not isinstance(obs, torch.Tensor):
            obs = torch.tensor(obs, dtype=torch.float32)
        if get_global_variable("benchmark") == "cliport":
            obs = obs.unsqueeze(0)
            obs = obs.permute(0, 3, 1, 2)
        self.cnn = SimpleCNN(obs.shape)
        features = self.cnn(obs)
        return features.detach().numpy()

