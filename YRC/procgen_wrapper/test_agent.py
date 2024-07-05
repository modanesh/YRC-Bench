import argparse
import random

import numpy as np
import torch

import utils
from utils import set_global_seeds


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', type=str, default='coinrun', help='environment ID')
    parser.add_argument('--distribution_mode', type=str, default='easy', help='distribution mode for environment')
    parser.add_argument('--param_name', type=str, default='easy-200', help='hyper-parameter ID')
    parser.add_argument('--device', type=str, default='cuda', required=False, help='whether to use gpu')
    parser.add_argument('--seed', type=int, default=0, help='Random generator seed')
    parser.add_argument('--model_file', type=str, required=True)
    parser.add_argument('--random_percent', type=int, default=0, help='COINRUN: percent of environments in which coin is randomized (only for coinrun)')
    parser.add_argument('--key_penalty', type=int, default=0, help='HEIST_AISC: Penalty for picking up keys (divided by 10)')
    parser.add_argument('--step_penalty', type=int, default=0, help='HEIST_AISC: Time penalty per step (divided by 1000)')
    parser.add_argument('--rand_region', type=int, default=0, help='MAZE: size of region (in upper left corner) in which goal is sampled.')
    parser.add_argument('--config_path', type=str, default='config.yml', help='path to hyperparameter config yaml')
    parser.add_argument('--num_threads', type=int, default=8)
    parser.add_argument('--n_trials', type=int, default=1000)
    parser.add_argument('--n_steps', type=int, default=256)
    parser.add_argument('--num_levels', type=int, default=100000)
    return parser.parse_args()


def test(env, agent, n_trials):
    print('::[LOGGING]::START TESTING...')
    obs = env.reset()
    rew_tracker = []

    while len(rew_tracker) < n_trials:
        act, log_prob_act, value = agent.predict(obs)
        next_obs, rew, done, info = env.step(act)
        obs = next_obs
        done_indices = np.where(done)[0]
        for i in done_indices:
            if 'env_reward' in info[i]:
                rew_tracker.append(info[i]['env_reward'])
            else:
                rew_tracker.append(rew[i])
    print(f'::[LOGGING]::TESTING COMPLETE. Mean reward over {n_trials} trials: {np.mean(rew_tracker):.2f}+-{np.std(rew_tracker):.2f}')
    env.close()


if __name__ == '__main__':
    args = get_args()
    args.start_level = random.randint(0, 9999)
    set_global_seeds(args.seed)
    args.device = torch.device(args.device)
    environment = utils.create_env(args.n_steps, args.env_name, args.start_level, args.num_levels, args.distribution_mode, args.num_threads,
                                   args.random_percent, args.step_penalty, args.key_penalty, args.rand_region, normalize_rew=False)
    obs_size = environment.observation_space.shape[0]
    action_size = environment.action_space.n
    actor = utils.load_policy(obs_size, action_size, args.model_file, args.device)
    test(environment, actor, args.n_trials)
