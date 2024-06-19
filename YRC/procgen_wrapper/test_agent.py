import argparse
import random

import numpy as np
import torch

from models import CategoricalPolicy, ImpalaModel, PPOFrozen
from procgen import ProcgenEnv
from procgen_wrappers import VecExtractDictObs, TransposeFrame, ScaledFloatFrame
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
    return parser.parse_args()


def model_setup(env, configs):
    observation_shape = env.observation_space.shape
    in_channels = observation_shape[0]
    action_space = env.action_space
    model = ImpalaModel(in_channels=in_channels)
    action_size = action_space.n
    policy = CategoricalPolicy(model, action_size)
    policy.to(configs.device)
    policy.eval()
    agent = PPOFrozen(policy, configs.device)
    agent = load_model(agent, configs.model_file)
    return agent


def create_env(args):
    print('::[LOGGING]::INITIALIZING ENVIRONMENTS...')
    env = ProcgenEnv(num_envs=args.n_steps,
                     env_name=args.env_name,
                     num_levels=100000,
                     start_level=args.start_level,
                     distribution_mode=args.distribution_mode,
                     num_threads=args.num_threads,
                     random_percent=args.random_percent,
                     step_penalty=args.step_penalty,
                     key_penalty=args.key_penalty,
                     rand_region=args.rand_region)
    env = VecExtractDictObs(env, "rgb")
    env = TransposeFrame(env)
    env = ScaledFloatFrame(env)
    return env


def load_model(agent, model_file):
    print("Loading agent from %s" % model_file)
    checkpoint = torch.load(model_file)
    agent.policy.load_state_dict(checkpoint["model_state_dict"])
    return agent


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
    environment = create_env(args)
    actor = model_setup(environment, args)
    test(environment, actor, args.n_trials)
