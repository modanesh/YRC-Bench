from cliport.utils import utils as cliport_utils
from .models import ProcgenPPO, CliportPPO, ProcgenDQN, CliportDQN
from .configs.global_configs import get_global_variable


class HelpPolicy:
    @staticmethod
    def create_policy(config, algorithm_class, env):
        config = getattr(config, algorithm_class)
        algorithms = {"PPO": [ProcgenPPO, CliportPPO], "DQN": [ProcgenDQN, CliportDQN]}
        benchmark = get_global_variable("benchmark")
        if benchmark == 'procgen':
            in_channels = env.base_env.observation_space.shape[0]
            weak_features = env.weak_policy.policy.embedder.fc.out_features if env.policy_type in ["T2", "T3"] else 0
            policy = algorithms[algorithm_class][0](config, in_channels, additional_hidden_dim=weak_features, policy_type=env.policy_type)
        elif benchmark == 'cliport':
            obs, _ = env.reset(need_features=False)
            img = [cliport_utils.get_image(obs)]
            in_channels = img[0].shape[-1]
            info = [env.base_env.info]
            pick_features, place_features = env.weak_policy.extract_features(img, info)
            weak_features = pick_features[0].shape[0] + place_features[0].shape[0] if env.policy_type in ["T2", "T3"] else 0
            policy = algorithms[algorithm_class][1](config, in_channels, additional_hidden_dim=weak_features, policy_type=env.policy_type)
        policy.to(get_global_variable('device'))
        return policy
