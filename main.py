from YRC.core import Environment, Policy
from YRC.core import env_registry
from YRC.core.utils import logger_setup
import argparse


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weak_model_file', type=str)
    parser.add_argument('--strong_model_file', type=str)
    parser.add_argument('--help_policy_type', type=str, choices=['T1', 'T2', 'T3'], required=True,
                        help='Type of the helper policy. T1: vanilla PPO, T2: PPO with inputs concatenated by the weak agent features (conv + mlp), '
                             'T3: PPO with inputs from the weak agent (mlp).')
    parser.add_argument('--benchmark', type=str, choices=['procgen', 'cliport'], required=True)
    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    env_registry.set_cfgs(name=args.benchmark)

    # get configs
    exp_cfg = env_registry.setup_cfgs(args)
    environment = Environment(exp_cfg)
    policy = Policy(exp_cfg)

    # setup logger
    logger = logger_setup(exp_cfg)

    # get env configs
    obs_shape, action_size = environment.get_env_configs()

    # load executing policies
    policy_kwargs = {'obs_shape': obs_shape, 'action_size': action_size}
    weak_policy, strong_policy = policy.load_acting_policies(policy_kwargs)

    # Set up environment and additional variables based on benchmark
    if args.benchmark == 'procgen':
        env, env_val = environment.make(weak_policy, strong_policy)
        additional_var = env_val
    elif args.benchmark == 'cliport':
        env, task = environment.make(weak_policy, strong_policy)
        additional_var = task

    # set up help policy
    help_algorithm = policy.setup_help_policy(env, additional_var, weak_policy, strong_policy, logger, obs_shape)

    # train the help policy
    help_algorithm.train(exp_cfg.num_timesteps)
