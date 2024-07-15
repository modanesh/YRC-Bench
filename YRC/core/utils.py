import argparse
import csv
import inspect
import os
import random
import time
import uuid
from collections import deque

import numpy as np
import pandas as pd
import torch
import wandb

from YRC.cliport_wrapper.models import PPO as cliport_PPO
from YRC.procgen_wrapper.models import PPO as procgen_PPO
from YRC.procgen_wrapper.utils import ProcgenEnv


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weak_model_file', type=str)
    parser.add_argument('--strong_model_file', type=str)
    parser.add_argument('--help_policy_type', type=str, choices=['T1', 'T2', 'T3'], required=True,
                        help='Type of the helper policy. '
                             'T1: vanilla PPO (input is obs), '
                             'T2: PPO with inputs concatenated by the weak agent features (conv + mlp), '
                             'T3: PPO with inputs from the weak agent (mlp).')
    parser.add_argument('--benchmark', type=str, choices=['procgen', 'cliport'], required=True)
    parser.add_argument('--env_name', type=str, required=True)
    parser.add_argument('--task', type=str)
    parser.add_argument('--model_task', type=str)
    parser.add_argument('--param_name', type=str)
    parser.add_argument('--switching_cost', type=float, required=True)
    parser.add_argument('--strong_query_cost', type=float, required=True)
    args = parser.parse_args()
    verify_args(args)
    return args


def verify_args(args):
    # general checks
    if args.weak_model_file is None:
        raise ValueError("Weak model file not provided.")
    # procgen checks
    if args.benchmark == 'procgen':
        if args.strong_model_file is None:
            raise ValueError("Strong model file not provided for procgen.")
        if args.param_name is None:
            raise ValueError("Param name not provided for procgen.")
    # cliport checks
    if args.benchmark == 'cliport':
        if args.task is None:
            raise ValueError("Task not provided for cliport.")
        if args.model_task is None:
            raise ValueError("Model task not provided for cliport.")


class Logger(object):
    def __init__(self, n_envs, logdir):
        self.start_time = time.time()
        self.n_envs = n_envs
        self.logdir = logdir

        # training
        self.episode_rewards = []
        for _ in range(n_envs):
            self.episode_rewards.append([])

        self.episode_timeout_buffer = deque(maxlen=40)
        self.episode_len_buffer = deque(maxlen=40)
        self.episode_reward_buffer = deque(maxlen=40)

        # validation
        self.episode_rewards_v = []
        for _ in range(n_envs):
            self.episode_rewards_v.append([])

        self.episode_timeout_buffer_v = deque(maxlen=40)
        self.episode_len_buffer_v = deque(maxlen=40)
        self.episode_reward_buffer_v = deque(maxlen=40)

        time_metrics = ["timesteps", "wall_time", "num_episodes"]  # only collected once
        episode_metrics = ["max_episode_rewards", "mean_episode_rewards", "min_episode_rewards",
                           "max_episode_len", "mean_episode_len",
                           "min_episode_len"]  # collected for both train and val envs
        self.log = pd.DataFrame(columns=time_metrics + episode_metrics + \
                                        ["val_" + m for m in episode_metrics])

        self.timesteps = 0
        self.num_episodes = 0

    def feed_procgen(self, rew_batch, done_batch, rew_batch_v=None, done_batch_v=None):
        steps = rew_batch.shape[0]
        rew_batch = rew_batch.T
        done_batch = done_batch.T

        valid = rew_batch_v is not None and done_batch_v is not None and rew_batch_v.shape[0] > 0
        if valid:
            rew_batch_v = rew_batch_v.T
            done_batch_v = done_batch_v.T

        for i in range(self.n_envs):
            for j in range(steps):
                self.episode_rewards[i].append(rew_batch[i][j])
                if valid:
                    self.episode_rewards_v[i].append(rew_batch_v[i][j])

                if done_batch[i][j]:
                    self.episode_len_buffer.append(len(self.episode_rewards[i]))
                    self.episode_reward_buffer.append(np.sum(self.episode_rewards[i]))
                    self.episode_rewards[i] = []
                    self.num_episodes += 1
                if valid and done_batch_v[i][j]:
                    self.episode_timeout_buffer_v.append(1 if j == steps - 1 else 0)
                    self.episode_len_buffer_v.append(len(self.episode_rewards_v[i]))
                    self.episode_reward_buffer_v.append(np.sum(self.episode_rewards_v[i]))
                    self.episode_rewards_v[i] = []

        self.timesteps += (self.n_envs * steps)

    def feed_cliport(self, rew_batch, done_batch, rew_batch_v=None, done_batch_v=None):
        valid = rew_batch_v is not None and done_batch_v is not None and rew_batch_v.shape[0] > 0
        # get the length of each episode, ends when done_batch is 1
        eps_lengths = np.where(done_batch == 1)[0] + 1

        self.episode_rewards = np.split(rew_batch, eps_lengths[:-1], axis=0)
        self.episode_len_buffer = np.insert(np.diff(eps_lengths), 0, eps_lengths[0])
        self.episode_reward_buffer = np.array([np.sum(rew) for rew in self.episode_rewards])
        self.num_episodes += len(self.episode_len_buffer)
        self.episode_rewards = []

        if valid:
            eps_lengths_v = np.where(done_batch_v == 1)[0] + 1
            self.episode_rewards_v = np.split(rew_batch_v, eps_lengths_v[:-1], axis=0)
            self.episode_len_buffer_v = np.insert(np.diff(eps_lengths_v), 0, eps_lengths_v[0])
            self.episode_reward_buffer_v = np.array([np.sum(rew) for rew in self.episode_rewards_v])
            self.episode_rewards_v = []

    def dump(self):
        wall_time = time.time() - self.start_time
        episode_statistics = self._get_episode_statistics()
        episode_statistics_list = list(episode_statistics.values())
        log = [self.timesteps, wall_time, self.num_episodes] + episode_statistics_list
        self.log.loc[len(self.log)] = log

        with open(self.logdir + '/log-append.csv', 'a') as f:
            writer = csv.writer(f)
            if f.tell() == 0:
                writer.writerow(self.log.columns)
            writer.writerow(log)

        print(self.log.loc[len(self.log) - 1])
        wandb.log({k: v for k, v in zip(self.log.columns, log)})

    def _get_episode_statistics(self):
        episode_statistics = {}
        episode_statistics['Rewards/max_episodes'] = np.max(self.episode_reward_buffer, initial=0)
        episode_statistics['Rewards/mean_episodes'] = np.mean(self.episode_reward_buffer)
        episode_statistics['Rewards/min_episodes'] = np.min(self.episode_reward_buffer, initial=0)
        episode_statistics['Len/max_episodes'] = np.max(self.episode_len_buffer, initial=0)
        episode_statistics['Len/mean_episodes'] = np.mean(self.episode_len_buffer)
        episode_statistics['Len/min_episodes'] = np.min(self.episode_len_buffer, initial=0)

        # valid
        episode_statistics['[Valid] Rewards/max_episodes'] = np.max(self.episode_reward_buffer_v, initial=0)
        episode_statistics['[Valid] Rewards/mean_episodes'] = np.mean(self.episode_reward_buffer_v)
        episode_statistics['[Valid] Rewards/min_episodes'] = np.min(self.episode_reward_buffer_v, initial=0)
        episode_statistics['[Valid] Len/max_episodes'] = np.max(self.episode_len_buffer_v, initial=0)
        episode_statistics['[Valid] Len/mean_episodes'] = np.mean(self.episode_len_buffer_v)
        episode_statistics['[Valid] Len/min_episodes'] = np.min(self.episode_len_buffer_v, initial=0)
        return episode_statistics


def logger_setup(cfgs):
    uuid_stamp = str(uuid.uuid4())[:8]
    env_name = cfgs.env_name if cfgs.benchmark == 'procgen' else cfgs.task
    run_name = f"PPO-{cfgs.benchmark}-help-{env_name}-type-{cfgs.help_policy_type}-query-cost-{cfgs.strong_query_cost}-{uuid_stamp}"
    logdir = os.path.join('logs', env_name)
    if not (os.path.exists(logdir)):
        os.makedirs(logdir)
    logdir = os.path.join(logdir, run_name)
    if not (os.path.exists(logdir)):
        os.mkdir(logdir)
    print(f'Logging to {logdir}')
    vars_cfgs = to_dict(cfgs)
    wandb.init(config=vars_cfgs, resume="allow", project="YRC", name=run_name)
    writer = Logger(cfgs.policy.n_envs, logdir)
    return writer


def to_dict(cls):
    def _convert(obj):
        if isinstance(obj, (int, float, str, bool, type(None))):
            return obj
        elif isinstance(obj, dict):
            return {k: _convert(v) for k, v in obj.items()}
        elif inspect.isclass(obj):
            return {k: _convert(v) for k, v in obj.__dict__.items()
                    if not k.startswith('__') and not callable(v)}
        elif hasattr(obj, '__dict__'):
            return {k: _convert(v) for k, v in obj.__dict__.items()
                    if not k.startswith('__')}
        else:
            return str(obj)

    attributes = {}
    for name, value in inspect.getmembers(cls):
        if not name.startswith('__') and not inspect.ismethod(value) and not inspect.isfunction(value):
            attributes[name] = _convert(value)

    return attributes


def set_global_seeds(seed, torch_deterministic=True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = torch_deterministic
    torch.backends.cudnn.benchmark = not torch_deterministic


def algorithm_setup(env, is_procgen, additional_var, policy, logger, storage, storage_valid, device, num_checkpoints,
                    hyperparameters, pi_w=None, pi_o=None, help_policy_type=None):
    print('::[LOGGING]::INTIALIZING AGENT...')
    ppo_agents = {'procgen': procgen_PPO, 'cliport': cliport_PPO}
    agent_type = 'procgen' if is_procgen else 'cliport'
    agent = ppo_agents[agent_type](env, additional_var, policy, logger, storage, device,
                                   num_checkpoints,
                                   storage_valid=storage_valid,
                                   pi_w=pi_w,
                                   pi_o=pi_o,
                                   help_policy_type=help_policy_type,
                                   **hyperparameters)
    return agent
