import csv
import os
import time
import uuid
from collections import deque

import numpy as np
import pandas as pd
import wandb


class Logger(object):

    def __init__(self, logdir):
        self.start_time = time.time()
        self.logdir = logdir

        # training
        self.episode_rewards = []

        self.episode_timeout_buffer = deque(maxlen=40)
        self.episode_len_buffer = deque(maxlen=40)
        self.episode_reward_buffer = deque(maxlen=40)

        # validation
        self.episode_rewards_v = []

        self.episode_timeout_buffer_v = deque(maxlen=40)
        self.episode_len_buffer_v = deque(maxlen=40)
        self.episode_reward_buffer_v = deque(maxlen=40)

        time_metrics = ["timesteps", "wall_time", "num_episodes"]  # only collected once
        episode_metrics = ["max_episode_rewards", "mean_episode_rewards", "min_episode_rewards",
                           "max_episode_len", "mean_episode_len", "min_episode_len",
                           "mean_timeouts"]  # collected for both train and val envs
        self.log = pd.DataFrame(columns=time_metrics + episode_metrics + \
                                        ["val_" + m for m in episode_metrics])

        self.timesteps = 0
        self.num_episodes = 0

    def feed(self, rew_batch, done_batch, rew_batch_v=None, done_batch_v=None):
        steps = rew_batch.shape[0]
        rew_batch = rew_batch.T
        done_batch = done_batch.T

        valid = rew_batch_v is not None and done_batch_v is not None
        if valid:
            rew_batch_v = rew_batch_v.T
            done_batch_v = done_batch_v.T

        for i in range(steps):
            self.episode_rewards.append(rew_batch)
            if valid:
                self.episode_rewards_v.append(rew_batch_v[i])

            if done_batch[i]:
                self.episode_timeout_buffer.append(1 if i == steps - 1 else 0)
                self.episode_len_buffer.append(len(self.episode_rewards))
                self.episode_reward_buffer.append(np.sum(self.episode_rewards))
                self.num_episodes += 1
            if valid and done_batch_v[i]:
                self.episode_timeout_buffer_v.append(1 if i == steps - 1 else 0)
                self.episode_len_buffer_v.append(len(self.episode_rewards_v))
                self.episode_reward_buffer_v.append(np.sum(self.episode_rewards_v))
                self.episode_rewards_v = []

        self.timesteps += (self.n_envs * steps)

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
        episode_statistics['Len/mean_timeout'] = np.mean(self.episode_timeout_buffer)

        # valid
        episode_statistics['[Valid] Rewards/max_episodes'] = np.max(self.episode_reward_buffer_v, initial=0)
        episode_statistics['[Valid] Rewards/mean_episodes'] = np.mean(self.episode_reward_buffer_v)
        episode_statistics['[Valid] Rewards/min_episodes'] = np.min(self.episode_reward_buffer_v, initial=0)
        episode_statistics['[Valid] Len/max_episodes'] = np.max(self.episode_len_buffer_v, initial=0)
        episode_statistics['[Valid] Len/mean_episodes'] = np.mean(self.episode_len_buffer_v)
        episode_statistics['[Valid] Len/min_episodes'] = np.min(self.episode_len_buffer_v, initial=0)
        episode_statistics['[Valid] Len/mean_timeout'] = np.mean(self.episode_timeout_buffer_v)
        return episode_statistics


def logger_setup(cfgs):
    uuid_stamp = str(uuid.uuid4())[:8]
    run_name = f"PPO-cliport-help-{cfgs.task}-{uuid_stamp}"
    logdir = os.path.join('logs', 'train', cfgs.task)
    if not (os.path.exists(logdir)):
        os.makedirs(logdir)
    logdir = os.path.join(logdir, run_name)
    if not (os.path.exists(logdir)):
        os.mkdir(logdir)
    print(f'Logging to {logdir}')
    wb_resume = "allow"
    wandb.init(config=vars(cfgs), resume=wb_resume, project="YRC", name=run_name)
    writer = Logger(logdir)
    return writer

