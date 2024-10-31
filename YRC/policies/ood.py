import os
import numpy as np
from copy import deepcopy as dc

import torch
import logging
from torch.distributions.categorical import Categorical
from YRC.core import Policy
from pyod.models import deep_svdd
from joblib import dump, load
from YRC.core.configs.global_configs import get_global_variable


class OODPolicy(Policy):
    def __init__(self, config, env):
        self.args = config.coord_policy
        self.agent = env.weak_agent
        self.params = {"threshold": 0.0, "explore_temp": 1.0}
        self.clf = None
        self.clf_name = None
        # self.cnn = None

    def gather_rollouts(self, env, num_rollouts):
        assert num_rollouts % env.num_envs == 0
        observations = []
        for i in range(2):
            observations.extend(self._rollout_once(env))
        return np.array(observations)

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
                obs_features = obs["env_obs"]['img']
                observations.extend(obs_features)
            else:
                for i in range(env.num_envs):
                    if not has_done[i]:
                        obs_features = obs["env_obs"]
                        observations.extend(obs_features)

            action = sample_action(logit)
            obs, reward, done, info = env.step(action)
            has_done |= done

        return observations

    def update_params(self, params):
        self.params = dc(params)

    def act(self, obs, greedy=False):
        if get_global_variable("benchmark") == "cliport":
            score = self.clf.decision_function(obs["env_obs"]['img'])
        else:
            score = self.clf.decision_function(obs["env_obs"])
        action = (score < self.params["threshold"]).astype(int)
        if 0 not in action and 1 not in action:
            print("No action is selected as OOD")
        return action

    def initialize_ood_detector(self, args, env):
        if self.args.method == "DeepSVDD":
            dummy_obs = env.reset()['env_obs']
            dummy_obs = dummy_obs['img'] if get_global_variable("benchmark") == "cliport" else dummy_obs
            dummy_obs = torch.tensor(dummy_obs, dtype=torch.float32)
            if get_global_variable("benchmark") == "cliport":
                dummy_obs = dummy_obs.unsqueeze(0)
                dummy_obs = dummy_obs.permute(0, 3, 1, 2)
            self.clf_name = 'DeepSVDD'
            self.clf = deep_svdd.DeepSVDD(
                            n_features=args.feature_size,
                            use_ae=args.use_ae,
                            contamination=args.contamination,
                            epochs=args.epoch,
                            batch_size=args.batch_size,
                            input_shape=dummy_obs.shape,
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

    def load_model(self, load_dir):
        state_dict = load(f"{load_dir}")
        self.clf = state_dict['clf']
        return self

