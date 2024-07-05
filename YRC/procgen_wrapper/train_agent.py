import argparse
import utils


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name', type=str, default='test', help='experiment name')
    parser.add_argument('--env_name', type=str, default='coinrun', help='environment ID')
    parser.add_argument('--val_env_name', type=str, default=None, help='optional validation environment ID')
    parser.add_argument('--start_level', type=int, default=int(0), help='start-level for environment')
    parser.add_argument('--num_levels', type=int, default=int(0), help='number of training levels for environment')
    parser.add_argument('--distribution_mode', type=str, default='easy', help='distribution mode for environment')
    parser.add_argument('--param_name', type=str, default='easy-200', help='hyper-parameter ID')
    parser.add_argument('--device', type=str, default='cuda', required=False, help='whether to use gpu')
    parser.add_argument('--num_timesteps', type=int, default=int(25000000), help='number of training timesteps')
    parser.add_argument('--seed', type=int, default=0, help='Random generator seed')
    parser.add_argument('--log_level', type=int, default=int(40), help='[10,20,30,40]')
    parser.add_argument('--num_checkpoints', type=int, default=int(1), help='number of checkpoints to store')
    parser.add_argument('--model_file', type=str)
    parser.add_argument('--random_percent', type=int, default=0, help='COINRUN: percent of environments in which coin is randomized (only for coinrun)')
    parser.add_argument('--key_penalty', type=int, default=0, help='HEIST_AISC: Penalty for picking up keys (divided by 10)')
    parser.add_argument('--step_penalty', type=int, default=0, help='HEIST_AISC: Time penalty per step (divided by 1000)')
    parser.add_argument('--rand_region', type=int, default=0, help='MAZE: size of region (in upper left corner) in which goal is sampled.')
    parser.add_argument('--config_path', type=str, default='config.yml', help='path to hyperparameter config yaml')
    parser.add_argument('--num_threads', type=int, default=8)
    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    args, hyperparameters = utils.hyperparam_setup(args)
    args, logger = utils.logger_setup(args)

    env = utils.create_env(args.n_steps, args.env_name, args.start_level, args.num_levels, args.distribution_mode, args.num_threads,
                           args.random_percent, args.step_penalty, args.key_penalty, args.rand_region, args.normalize_rew)
    env_valid = utils.create_env(args.n_steps, args.val_env_name, args.start_level_val, 0, args.distribution_mode, args.num_threads,
                                 args.random_percent, args.step_penalty, args.key_penalty, args.rand_region, args.normalize_rew)

    model, policy = utils.define_policy(env, args.device)
    storage = utils.Storage(env.observation_space.shape, args.n_steps, args.n_envs, args.device)
    storage_valid = utils.Storage(env.observation_space.shape, args.n_steps, args.n_envs, args.device)
    agent = utils.algorithm_setup(env, env_valid, policy, logger, storage, storage_valid, args.device, args.num_checkpoints, args.model_file, hyperparameters)

    agent.train(args.num_timesteps)
