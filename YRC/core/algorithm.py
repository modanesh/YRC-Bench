from abc import ABC, abstractmethod
from pathlib import Path

import torch
import numpy as np
import os
from .utils import ProcgenReplayBuffer, CliportReplayBuffer
from .configs.global_configs import get_global_variable


class Algorithm(ABC):
    def __init__(self, config, logger):
        self.training_steps = config.training_steps
        self.test_steps = config.test_steps
        self.log_freq = config.log_freq
        self.save_freq = config.save_freq
        self.save_dir = config.save_dir
        self.load_best = config.load_best
        self.load_last = config.load_last
        self.load_custom_index = config.load_custom_index

        self.storage = None
        self.logger = logger

    def train(self, policy, evaluator, train_env=None, dataset=None):
        for i in range(self.training_steps):
            self.train_one_iteration(policy, train_env, dataset)
            if i % self.log_freq == 0:
                evaluator.evaluate(policy)
                if i % self.save_freq == 0 or evaluator.model_improved:
                    evaluator.best_index = i
                    policy.save_model(os.path.join(self.save_dir, f"model_{i}.pt"))
        train_env.close()
        evaluator.eval_env.close()

    def test(self, policy, test_env, best_model_index):
        policy = self.load_fresh_model(policy, best_model_index)
        policy.eval()

        num_envs = test_env.base_env.num_envs
        reward_batch, done_batch = self._run_test_episodes(policy, test_env)

        episode_stats = self._calculate_episode_stats(reward_batch, done_batch, num_envs)
        self._print_test_performance(episode_stats)

        test_env.close()

    def _run_test_episodes(self, policy, test_env):
        reward_batch = np.zeros((self.test_steps, test_env.base_env.num_envs))
        done_batch = np.zeros((self.test_steps, test_env.base_env.num_envs), dtype=bool)

        with torch.no_grad():
            obs, pi_w_hidden = test_env.reset()
            ep_steps = 0

            for i in range(self.test_steps):
                action, _, _ = policy.predict(obs, pi_w_hidden)
                obs, reward, done, info, pi_w_hidden = test_env.step(action)
                reward_batch[i], done_batch[i] = reward, done
                ep_steps += 1

                if self._should_reset_environment(done, ep_steps, test_env):
                    done_batch[i] = True
                    obs, pi_w_hidden = self._reset_environment(test_env, i)
                    ep_steps = 0

        return reward_batch, done_batch

    def _should_reset_environment(self, done, ep_steps, env):
        return (get_global_variable('benchmark') == 'cliport' and
                (done or ep_steps == env.base_env.task.max_steps))

    def _reset_environment(self, env, step):
        env.base_env.seed(env.base_env._seed + step)
        return env.reset()

    def _calculate_episode_stats(self, reward_batch, done_batch, num_envs):
        episode_ends = np.where(done_batch)[0]
        if len(episode_ends) == 0 or episode_ends[-1] != self.test_steps - 1:
            episode_ends = np.append(episode_ends, self.test_steps - 1)

        episode_starts = np.concatenate(([0], episode_ends[:-1] + 1))

        episode_rewards = [
            reward_batch[start:end + 1, env_idx].sum()
            for start, end in zip(episode_starts, episode_ends)
            for env_idx in range(num_envs)
        ]
        episode_lengths = episode_ends - episode_starts + 1

        total_reward = np.sum(reward_batch)
        num_episodes = len(episode_rewards)

        return {
            'total_reward': total_reward,
            'num_episodes': num_episodes,
            'mean_episode_reward': np.mean(episode_rewards) if num_episodes > 0 else total_reward,
            'mean_episode_length': np.mean(episode_lengths),
            'max_episode_reward': np.max(episode_rewards) if num_episodes > 0 else total_reward,
            'min_episode_reward': np.min(episode_rewards) if num_episodes > 0 else total_reward
        }

    def _print_test_performance(self, stats):
        print("Test Performance:")
        for key, value in stats.items():
            print(f"{key.replace('_', ' ').title()}: {value}")

    def load_fresh_model(self, policy, best_index):
        if self.load_best:
            model_path = os.path.join(self.save_dir, f'model_{best_index}.pt')
        elif self.load_custom_index != -1:
            model_path = os.path.join(self.save_dir, f'model_{self.load_custom_index}.pt')
        elif self.load_last:
            model_path = str(sorted(Path(self.save_dir).iterdir(), key=os.path.getmtime)[0])
        checkpoint = torch.load(model_path)
        policy.load_state_dict(checkpoint["model_state_dict"])
        return policy

    @abstractmethod
    def train_one_iteration(self, policy, train_env=None, dataset=None):
        pass


class PPOAlgorithm(Algorithm):
    def __init__(self, config, logger, train_env):
        super().__init__(config, logger)
        self.t = 0
        self.rollout_length = config.rollout_length
        self.mini_batch_size = config.mini_batch_size
        self.mini_batch_per_epoch = config.mini_batch_per_epoch
        self.epoch = config.epoch
        self.grad_clip_norm = config.grad_clip_norm
        self.value_coef = config.value_coef
        self.entropy_coef = config.entropy_coef

        obs_shape = train_env.observation_space.shape if get_global_variable("benchmark") == 'procgen' else (6, 320, 160)
        num_envs = train_env.base_env.num_envs
        self.storage = (ProcgenReplayBuffer if get_global_variable("benchmark") == 'procgen' else CliportReplayBuffer)(config.gamma, config.lmbda,
                                                                                                                       config.use_gae,
                                                                                                                       config.normalize_adv, obs_shape,
                                                                                                                       config.rollout_length, num_envs)

    def train_one_iteration(self, policy, train_env=None, dataset=None):
        obs, pi_w_hidden = train_env.reset()
        ep_steps = 0

        for _ in range(self.rollout_length):
            act, log_prob_act, value = policy.predict(obs, pi_w_hidden)
            next_obs, rew, done, info, pi_w_hidden = train_env.step(act)

            self.storage.add_transition(obs, act, log_prob_act, rew, next_obs, done, value, info)

            obs = next_obs
            ep_steps += 1

            if self._should_reset_environment(done, ep_steps, train_env):
                self.storage.store_last_done()
                obs, pi_w_hidden = self._reset_environment(train_env, ep_steps)
                ep_steps = 0

        _, _, last_val = policy.predict(obs, pi_w_hidden)
        self.storage.store_last(obs, last_val)
        self.storage.compute_estimates()

        summary = self.update_policy(policy, train_env)
        self._update_training_progress(policy)

        return summary

    def _update_training_progress(self, policy):
        self.t += self.rollout_length * self.storage.num_envs
        self.log_training_progress()
        policy.optimizer = adjust_lr(policy.optimizer, policy.learning_rate, self.t, self.training_steps)

    def update_policy(self, policy, train_env):
        pi_loss_list, value_loss_list, entropy_loss_list = [], [], []
        batch_size = self.rollout_length * self.storage.num_envs // self.mini_batch_per_epoch
        self.mini_batch_size = min(self.mini_batch_size, batch_size)
        grad_accumulation_steps = batch_size / self.mini_batch_size
        grad_accumulation_cnt = 1

        for _ in range(self.epoch):
            generator = self.storage.fetch_train_generator(mini_batch_size=self.mini_batch_size)
            for sample in generator:
                obs_batch, act_batch, done_batch, old_log_prob_act_batch, old_value_batch, return_batch, adv_batch, info_batch = sample
                pi_w_hidden_batch = train_env.get_weak_policy_features(obs_batch, info_batch)
                dist_batch, value_batch = policy(obs_batch, pi_w_hidden_batch)

                pi_loss, value_loss, entropy_loss = policy.compute_losses(dist_batch, value_batch, act_batch,
                                                                          old_log_prob_act_batch, old_value_batch,
                                                                          return_batch, adv_batch)

                loss = pi_loss + self.value_coef * value_loss - self.entropy_coef * entropy_loss
                loss.backward()

                if grad_accumulation_cnt % grad_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(policy.parameters(), self.grad_clip_norm)
                    policy.optimizer.step()
                    policy.optimizer.zero_grad()
                grad_accumulation_cnt += 1

                pi_loss_list.append(-pi_loss.item())
                value_loss_list.append(-value_loss.item())
                entropy_loss_list.append(entropy_loss.item())

        return {
            'Loss/pi': np.mean(pi_loss_list),
            'Loss/v': np.mean(value_loss_list),
            'Loss/entropy': np.mean(entropy_loss_list)
        }

    def log_training_progress(self):
        rew_batch, done_batch = self.storage.fetch_log_data()
        self.logger.feed(rew_batch, done_batch)
        self.logger.dump()


def adjust_lr(optimizer, init_lr, timesteps, max_timesteps):
    lr = init_lr * (1 - (timesteps / max_timesteps))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return optimizer
