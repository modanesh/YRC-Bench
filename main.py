from YRC.core import Environment, Policy
from YRC.core import env_registry
from YRC.core.utils import logger_setup, get_args

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

    # Set up environment and additional variables based on benchmark. For procgen, the `additional_var` is the
    # validation env, and for cliport, the `additional_var` is the task.
    env, additional_var = environment.make(weak_policy, strong_policy)

    # set up help policy
    help_algorithm = policy.setup_help_policy(env, additional_var, weak_policy, strong_policy, logger, obs_shape)

    # train the help policy
    help_algorithm.train(exp_cfg.num_timesteps)

    # train the OOD detector
    