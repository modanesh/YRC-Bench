from YRC.core import Environment, Policy
from YRC.core import env_registry
from YRC.core.utils import logger_setup
import argparse


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weak_model_file', type=str, required=True)
    parser.add_argument('--strong_model_file', type=str)
    parser.add_argument('--help_policy_type', type=str, choices=['T1', 'T2', 'T3'], required=True,
                        help='Type of the helper policy. T1: vanilla PPO, T2: PPO with inputs concatenated by the weak agent features (conv + mlp), '
                             'T3: PPO with inputs from the weak agent (mlp).')
    parser.add_argument('--benchmark', type=str, choices=['procgen', 'cliport'], required=True)
    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()

    # get configs
    exp_cfg = env_registry.get_cfgs(args)
    environment = Environment(exp_cfg)
    policy = Policy(exp_cfg)

    # setup logger
    logger = logger_setup(exp_cfg)

    # get env configs
    obs_shape, action_size = environment.get_env_configs()

    # load executing policies
    weak_policy, strong_policy = policy.load_acting_policies(obs_shape, action_size)

    # set up help environment
    env, env_val = environment.make(weak_policy, strong_policy)

    # set up help policy
    help_algorithm = policy.setup_help_policy(env, env_val, weak_policy, strong_policy, logger, obs_shape)

    # train the help policy
    help_algorithm.train(exp_cfg.num_timesteps)
