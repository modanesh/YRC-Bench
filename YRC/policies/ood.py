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
        self.device = get_global_variable("device")
        self.feature_type = config.coord_policy.feature_type

    def gather_rollouts(self, env, num_rollouts):
        assert num_rollouts % env.num_envs == 0
        observations = []
        for i in range(num_rollouts // env.num_envs):
            observations.extend(self._rollout_once(env))
        if self.feature_type in ["hidden_obs", "hidden_dist"]:
            feature0_tensors = []
            feature1_tensors = []
            for i, tensor in enumerate(observations):
                if i % 2 == 0:
                    feature0_tensors.append(tensor)
                else:
                    feature1_tensors.append(tensor)
            observations = [torch.cat(feature0_tensors, dim=0), torch.cat(feature1_tensors, dim=0)]
        else:
            observations = torch.stack(observations)
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
                # todo: fix faeture type for cliport
                obs_features = obs["env_obs"]['img']
                # randomly keep 0.005 in the observations. do this for memory usage reasons
                if np.random.rand() < 0.005:
                    observations.extend(obs_features)
            else:
                for i in range(env.num_envs):
                    if not has_done[i]:
                        if self.feature_type == "obs":
                            obs_features = obs["env_obs"]
                        elif self.feature_type == "hidden":
                            obs_features = obs["weak_features"]
                        elif self.feature_type == "hidden_obs":
                            obs_features = [obs["env_obs"], obs["weak_features"]]
                        elif self.feature_type == "dist":
                            obs_features = obs["weak_logit"]
                        elif self.feature_type == "hidden_dist":
                            obs_features = [obs["weak_features"], obs["weak_logit"]]
                        else:
                            raise NotImplementedError
                        # randomly keep 0.005 in the observations. do this for memory usage reasons
                        if np.random.rand() < 0.005:
                            if not torch.is_tensor(obs["env_obs"]):
                                if self.feature_type in ["hidden_obs", "hidden_dist"]:
                                    obs_features[0] = torch.from_numpy(obs_features[0]).float().to(self.device)
                                    obs_features[1] = torch.from_numpy(obs_features[1]).float().to(self.device)
                                else:
                                    obs_features = torch.from_numpy(obs_features).float().to(self.device)
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
            if self.feature_type == "obs":
                if not torch.is_tensor(obs['env_obs']):
                    observation = torch.from_numpy(obs['env_obs']).float().to(self.device)
                else:
                    observation = obs['env_obs'].to(self.device)
            elif self.feature_type == "hidden":
                if not torch.is_tensor(obs['weak_features']):
                    observation = torch.from_numpy(obs['weak_features']).float().to(self.device)
                else:
                    observation = obs['weak_features'].to(self.device)
            elif self.feature_type == "hidden_obs":
                if not torch.is_tensor(obs['env_obs']):
                    obs['env_obs'] = torch.from_numpy(obs['env_obs']).float().to(self.device)
                if not torch.is_tensor(obs['weak_features']):
                    obs['weak_features'] = torch.from_numpy(obs['weak_features']).float().to(self.device)
                observation = [obs['env_obs'], obs['weak_features']]
            elif self.feature_type == "dist":
                if not torch.is_tensor(obs['weak_logit']):
                    observation = torch.from_numpy(obs['weak_logit']).float().to(self.device)
                else:
                    observation = obs['weak_logit'].to(self.device)
            elif self.feature_type == "hidden_dist":
                if not torch.is_tensor(obs['weak_features']):
                    obs['weak_features'] = torch.from_numpy(obs['weak_features']).float().to(self.device)
                if not torch.is_tensor(obs['weak_logit']):
                    obs['weak_logit'] = torch.from_numpy(obs['weak_logit']).float().to(self.device)
                observation = [obs['weak_features'], obs['weak_logit']]
            score = self.clf.decision_function(observation)

        action = 1 - (score < self.clf.threshold_).astype(int)
        if 0 not in action and 1 not in action:
            print("No action is selected as OOD")
        return action

    def initialize_ood_detector(self, args, env):
        if self.args.method == "DeepSVDD":
            dummy_obs = env.reset()
            if self.feature_type == "obs":
                dummy_obs = dummy_obs['env_obs']['img'] if get_global_variable("benchmark") == "cliport" else dummy_obs['env_obs']
                dummy_obs_shape = dummy_obs.shape
            elif self.feature_type == "hidden":
                dummy_obs = dummy_obs['weak_features']
                dummy_obs_shape = dummy_obs.shape
            elif self.feature_type == "hidden_obs":
                env_obs = dummy_obs['env_obs']['img'] if get_global_variable("benchmark") == "cliport" else dummy_obs['env_obs']
                hidden_obs = dummy_obs['weak_features']
                dummy_obs_shape = env_obs.shape + hidden_obs.shape[1:]
            elif self.feature_type == "dist":
                dummy_obs = dummy_obs['weak_logit']
                dummy_obs_shape = dummy_obs.shape
            elif self.feature_type == "hidden_dist":
                dummy_obs = [dummy_obs['weak_features'], dummy_obs['weak_logit']]
                dummy_obs_shape = dummy_obs[0].shape + dummy_obs[1].shape[1:]
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
                            input_shape=dummy_obs_shape,
                            feature_type=self.feature_type
                        )
            self.clf.model_.to(self.device)
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

