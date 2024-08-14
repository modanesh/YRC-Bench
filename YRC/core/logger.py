import time
import csv
from collections import deque
import numpy as np
import pandas as pd


class Logger:
    def __init__(self, n_envs, logdir, benchmark):
        self.start_time = time.time()
        self.n_envs = n_envs
        self.logdir = logdir
        self.benchmark = benchmark

        self.train_data = self._initialize_data()
        self.val_data = self._initialize_data()

        time_metrics = ["timesteps", "wall_time", "num_episodes"]
        episode_metrics = [
            "mean_episode_rewards", "std_episode_rewards", "max_episode_rewards", "min_episode_rewards",
            "mean_episode_len", "std_episode_len", "max_episode_len", "min_episode_len"
        ]

        self.log = self._create_dataframe(time_metrics, episode_metrics)
        self.log_v = self._create_dataframe(time_metrics, episode_metrics, suffix="_val")

    def _initialize_data(self):
        return {
            "episode_rewards": [[] for _ in range(self.n_envs)],
            "episode_lengths": [0 for _ in range(self.n_envs)],
            "episode_len_buffer": deque(maxlen=100),
            "episode_reward_buffer": deque(maxlen=100),
            "timesteps": 0,
            "num_episodes": 0
        }

    def _create_dataframe(self, time_metrics, episode_metrics, suffix=""):
        columns = time_metrics + [f"{m}{suffix}" for m in episode_metrics]
        return pd.DataFrame(columns=columns)

    def feed(self, rew_batch, done_batch, is_val=False):
        data = self.val_data if is_val else self.train_data
        total_reward = self._process_batch(rew_batch, done_batch, data)
        return total_reward

    def _process_batch(self, rew_batch, done_batch, data):
        rew_batch, done_batch = rew_batch.T, done_batch.T
        n_envs, steps = rew_batch.shape

        for env in range(n_envs):
            episode_start = 0
            for step in range(steps):
                reward = rew_batch[env, step]
                done = done_batch[env, step]

                data["episode_rewards"][env].append(reward)
                data["episode_lengths"][env] += 1

                if done:
                    self._handle_episode_end(data, env, step - episode_start + 1)
                    episode_start = step + 1

            # If the episode didn't end, we don't count it
            if episode_start < steps:
                data["episode_rewards"][env] = []
                data["episode_lengths"][env] = 0

        data["timesteps"] += sum(data['episode_len_buffer'])

        return np.mean(data["episode_reward_buffer"])

    def _handle_episode_end(self, data, env_index, episode_length):
        episode_reward = sum(data["episode_rewards"][env_index])
        data["episode_reward_buffer"].append(episode_reward)
        data["episode_len_buffer"].append(episode_length)
        data["num_episodes"] += 1

        # Reset episode data for this environment
        data["episode_rewards"][env_index] = []
        data["episode_lengths"][env_index] = 0

    def dump(self, is_val=False):
        data = self.val_data if is_val else self.train_data
        log_df = self.log_v if is_val else self.log

        wall_time = time.time() - self.start_time
        episode_statistics = self._get_episode_statistics(data, is_val)
        log_entry = [data["timesteps"], wall_time, data["num_episodes"]] + list(episode_statistics.values())

        log_df.loc[len(log_df)] = log_entry
        self._write_to_csv(log_entry, is_val)
        self._print_log(log_df, is_val)

    def _write_to_csv(self, log_entry, is_val):
        with open(f"{self.logdir}/log-append.csv", 'a') as f:
            writer = csv.writer(f)
            if f.tell() == 0:
                writer.writerow(self.log_v.columns if is_val else self.log.columns)
            writer.writerow(log_entry)

    def _print_log(self, log_df, is_val=False):
        print(":::::::::::::EVALUATION LOG:::::::::::::" if is_val else ":::::::::::::TRAINING LOG:::::::::::::")
        print(log_df.loc[len(log_df) - 1])

    def _get_episode_statistics(self, data, is_val):
        suffix = "_val" if is_val else ""
        episode_reward_buffer = data["episode_reward_buffer"]
        episode_len_buffer = data["episode_len_buffer"]

        def get_stats(buffer):
            if len(buffer) == 0:
                return 0, 0, 0, 0
            return np.mean(buffer), np.std(buffer), np.max(buffer), np.min(buffer)

        rewards_mean, rewards_std, rewards_max, rewards_min = get_stats(episode_reward_buffer)
        len_mean, len_std, len_max, len_min = get_stats(episode_len_buffer)

        return {
            f'mean_episode_rewards{suffix}': rewards_mean,
            f'std_episode_rewards{suffix}': rewards_std,
            f'max_episode_rewards{suffix}': rewards_max,
            f'min_episode_rewards{suffix}': rewards_min,
            f'mean_episode_len{suffix}': len_mean,
            f'std_episode_len{suffix}': len_std,
            f'max_episode_len{suffix}': len_max,
            f'min_episode_len{suffix}': len_min
        }
