import importlib
import numpy as np

from YRC.algorithms.ood import OODKDE, OODDeepSVDD
from YRC.algorithms.non_parametric import RandomPolicy, NonParamPolicy
from YRC.algorithms.rl import ProcgenPPO, ProcgenDQN, CliportPPO, CliportDQN
from cliport.utils import utils as cliport_utils
from .configs.global_configs import get_global_variable


def make(config, coord_env):
    coord_policy_cls = getattr(
        importlib.import_module(f"YRC.policies"),
        config.coord_policy.cls
    )
    coord_policy = coord_policy_cls(config, coord_env)
    return coord_policy


class Policy:

    # get logit
    def forward(self, obs):
        pass

    # get action distribution
    def predict(self, obs):
        pass

    # draw an action
    def act(self, obs, greedy=False):
        pass

    # update model parameters
    def update_params(self):
        pass

    # get pre-softmax hidden features
    def get_hidden(self):
        pass

    def train(self):
        pass

    def eval(self):
        pass

    # initialization at the beginning of an episode
    def reset(self, should_reset):
        pass

    def save_model(self, name, save_dir):
        pass

    def load_model(self, load_path):
        pass


class HelpPolicy:
    def __init__(self, config, algorithm_class, env):
        self.config = config
        self.algorithm_class = algorithm_class
        self.env = env
        self.policy = self._create_policy()

    def _create_policy(self):
        config = self._get_algorithm_config()
        rl_algorithms = {"PPO": [ProcgenPPO, CliportPPO], "DQN": [ProcgenDQN, CliportDQN]}
        ood_algorithms = {"KDE": [OODKDE], "SVDD": [OODDeepSVDD]}
        non_parametric_algorithms = {"Random": RandomPolicy, "NonParam": NonParamPolicy}
        benchmark = get_global_variable('benchmark')

        if benchmark == 'procgen':
            policy = self._create_procgen_policy(config, rl_algorithms, ood_algorithms, non_parametric_algorithms)
        elif benchmark == 'cliport':
            policy = self._create_cliport_policy(config, rl_algorithms, ood_algorithms, non_parametric_algorithms)
        else:
            raise ValueError(f"Unsupported benchmark: {benchmark}")
        return policy

    def _get_algorithm_config(self):
        return getattr(self.config, self.algorithm_class)

    def _create_cliport_policy(self, config, rl_algorithms, ood_algorithms, non_parametric_algorithms):
        obs, _ = self.env.reset(need_features=False)
        img = [cliport_utils.get_image(obs)]
        in_channels = img[0].shape[-1]
        info = [self.env.base_env.info]
        pick_features, place_features = self.env.weak_policy.extract_features(img, info)
        weak_features = pick_features[0].shape[0] + place_features[0].shape[0] if self.env.feature_type in ["T2", "T3"] else 0

        if self.algorithm_class in rl_algorithms:
            policy = rl_algorithms[self.algorithm_class][1](config, in_channels, additional_hidden_dim=weak_features, feature_type=self.env.feature_type)
            return policy.to(get_global_variable('device'))
        elif self.algorithm_class in ood_algorithms:
            return ood_algorithms[self.algorithm_class][0](config, np.prod(img[0].shape), additional_hidden_dim=weak_features,
                                                           feature_type=self.env.feature_type)
        elif self.algorithm_class in non_parametric_algorithms:
            raise ValueError(f"Non-parametric algorithms are not supported for the Cliport benchmark (yet)!")  # todo: implement this
        else:
            raise ValueError(f"Unknown algorithm class: {self.algorithm_class}")

    def _create_procgen_policy(self, config, rl_algorithms, ood_algorithms, non_parametric_algorithms):
        weak_features = self.env.weak_policy.policy.embedder.fc.out_features if self.env.feature_type in ["T2", "T3"] else 0

        if self.algorithm_class in rl_algorithms:
            in_channels = self.env.base_env.observation_space.shape[0]
            policy = rl_algorithms[self.algorithm_class][0](config, in_channels, additional_hidden_dim=weak_features, feature_type=self.env.feature_type)
            return policy.to(get_global_variable('device'))
        elif self.algorithm_class in ood_algorithms:
            in_channels = np.prod(self.env.base_env.observation_space.shape)
            return ood_algorithms[self.algorithm_class][0](config, in_channels, additional_hidden_dim=weak_features, feature_type=self.env.feature_type)
        elif self.algorithm_class in non_parametric_algorithms:
            return non_parametric_algorithms[self.algorithm_class](config)
        else:
            raise ValueError(f"Unknown algorithm class: {self.algorithm_class}")

    def act(self, obs, features=None):
        # used for the RL algorithms
        return self.policy.act(obs, features)

    def gather_rollouts(self, rollout_len, env):
        # used for the OOD and NonParam algorithms
        return self.policy.gather_rollouts(rollout_len, env)

    def train(self, X_train):
        # used for the OOD algorithms
        return self.policy.train(X_train)

    # def evaluate(self, clf, x, y):
    #     # used for the OOD algorithms
    #     return self.policy.evaluate(clf, x, y)

    def load_model(self, load_dir):
        # used for the OOD algorithms
        return self.policy.load_model(load_dir)

    def save_model(self, save_dir, model=None):
        # used for the RL and OOD algorithms
        return self.policy.save_model(save_dir, model)

    def predict(self, obs, features=None):
        # used for the RL and OOD algorithms
        return self.policy.predict(obs, features)

    def update_target_network(self):
        # used for the RL algorithms
        return self.policy.update_target_network()

    def load_state_dict(self, state_dict):
        # used for the RL algorithms
        return self.policy.load_state_dict(state_dict)

    def determine_sampled_logit_threshold(self):
        # used for the NonParam algorithms
        return self.policy.determine_sampled_logit_threshold()
