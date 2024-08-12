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

    def train(self, policy, evaluator, train_env=None, dataset=None, weak_policy=None, strong_policy=None):
        for i in range(self.training_steps):
            if dataset is not None:
                self.train_one_iteration_offline(policy, dataset, weak_policy, strong_policy)
            elif train_env is not None:
                self.train_one_iteration_online(policy, train_env)
            else:
                raise ValueError("Either train_env or dataset should be provided for training")
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
    def train_one_iteration_online(self, policy, train_env):
        pass

    @abstractmethod
    def train_one_iteration_offline(self, policy, dataset, weak_policy, strong_policy):
        pass


class PPOAlgorithm(Algorithm):
    def __init__(self, config, logger, env):
        super().__init__(config, logger)
        self.t = 0
        self.rollout_length = config.rollout_length
        self.mini_batch_size = config.mini_batch_size
        self.mini_batch_per_epoch = config.mini_batch_per_epoch
        self.epoch = config.epoch
        self.grad_clip_norm = config.grad_clip_norm
        self.value_coef = config.value_coef
        self.entropy_coef = config.entropy_coef
        self.gamma = config.gamma
        self.lmbda = config.lmbda
        self.use_gae = config.use_gae
        self.normalize_adv = config.normalize_adv
        obs_shape = env.observation_space.shape if get_global_variable("benchmark") == 'procgen' else (6, 320, 160)
        num_envs = env.base_env.num_envs
        self.storage = (ProcgenReplayBuffer if get_global_variable("benchmark") == 'procgen' else CliportReplayBuffer)(self.gamma, self.lmbda,
                                                                                                                       self.use_gae,
                                                                                                                       self.normalize_adv, obs_shape,
                                                                                                                       config.rollout_length, num_envs)

    def train_one_iteration_online(self, policy, train_env):
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

        summary = self.update_policy_online(policy, train_env)
        self._update_training_progress(policy)

        return summary

    def train_one_iteration_offline(self, policy, dataset, weak_policy, strong_policy):
        all_obs, all_acts, all_rewards, all_infos, all_dones = [], [], [], [], []

        for i in range(dataset.n_demos):
            demonstrations, seed = dataset.load(i)
            batch_obs, batch_act, batch_reward, batch_info = zip(*demonstrations)
            batch_done = [0.] * (len(batch_obs) - 1) + [1.]
            img_batch = []
            for obs in batch_obs:
                img = dataset.get_image(obs)
                img_batch.append(img)
            all_obs.extend(img_batch)
            all_acts.extend(batch_act)
            all_rewards.extend(batch_reward)
            all_infos.extend(batch_info)
            all_dones.extend(batch_done)

        summary = self.update_policy_offline(policy, all_obs, all_acts, all_rewards, all_dones, all_infos, weak_policy)
        self._update_training_progress(policy)

        return summary

    def _compute_returns(self, rewards, last_value):
        returns = torch.zeros_like(rewards)
        running_return = last_value
        for t in reversed(range(len(rewards))):
            running_return = rewards[t] + self.gamma * running_return
            returns[t] = running_return
        return returns

    def _update_training_progress(self, policy):
        self.t += self.rollout_length * self.storage.num_envs
        self.log_training_progress()
        policy.optimizer = adjust_lr(policy.optimizer, policy.learning_rate, self.t, self.training_steps)

    def update_policy_online(self, policy, train_env):
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

    def update_policy_offline(self, policy, obs_batch, act_batch, reward_batch, done_batch, info_batch, weak_policy):
        pi_loss_list, value_loss_list, entropy_loss_list = [], [], []
        batch_size = len(obs_batch)
        self.mini_batch_size = min(self.mini_batch_size, batch_size)
        grad_accumulation_steps = batch_size / self.mini_batch_size
        grad_accumulation_cnt = 1

        # Compute features and values
        if policy.type != "T1":
            with torch.no_grad():
                pi_w_pick_hidden, pi_w_place_hidden = weak_policy.extract_features(obs_batch, info_batch)
                pi_w_pick_hidden = torch.stack(pi_w_pick_hidden) if isinstance(pi_w_pick_hidden, list) else pi_w_pick_hidden
                pi_w_place_hidden = torch.stack(pi_w_place_hidden) if isinstance(pi_w_place_hidden, list) else pi_w_place_hidden
                if pi_w_pick_hidden.dim() != 2:
                    pi_w_pick_hidden = pi_w_pick_hidden.unsqueeze(0)
                    pi_w_place_hidden = pi_w_place_hidden.unsqueeze(0)
                pi_w_hidden_batch = torch.cat([pi_w_pick_hidden, pi_w_place_hidden], dim=-1)
        else:
            pi_w_hidden_batch = None

        obs_batch = torch.FloatTensor(obs_batch).to(device=get_global_variable("device"))
        obs_batch = obs_batch.permute(0, 3, 1, 2)
        reward_batch = torch.FloatTensor(reward_batch).to(device=get_global_variable("device"))
        done_batch = torch.FloatTensor(done_batch).to(device=get_global_variable("device"))

        with torch.no_grad():
            _, value_batch = policy(obs_batch, pi_w_hidden_batch)

            # Calculate returns and advantages
            returns = torch.zeros_like(reward_batch)
            advantages = torch.zeros_like(reward_batch)
            last_value = 0
            running_return = 0
            running_advantage = 0

            for t in reversed(range(batch_size)):
                if done_batch[t]:
                    running_return = 0
                    running_advantage = 0
                    last_value = 0

                running_return = reward_batch[t] + self.gamma * (1 - done_batch[t]) * running_return
                returns[t] = running_return

                td_error = reward_batch[t] + self.gamma * (1 - done_batch[t]) * last_value - value_batch[t]
                running_advantage = td_error + self.gamma * self.lmbda * (1 - done_batch[t]) * running_advantage
                advantages[t] = running_advantage

                last_value = value_batch[t]

            # Normalize advantages
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        for _ in range(self.epoch):
            # Shuffle the data
            indices = torch.randperm(batch_size)
            obs_batch = obs_batch[indices]
            act_batch = [act_batch[i] for i in indices]
            returns = returns[indices]
            advantages = advantages[indices]
            old_value_batch = value_batch[indices]
            if pi_w_hidden_batch is not None:
                pi_w_hidden_batch = pi_w_hidden_batch[indices]

            for start_idx in range(0, batch_size, self.mini_batch_size):
                end_idx = start_idx + self.mini_batch_size
                mb_obs = obs_batch[start_idx:end_idx]
                mb_act = act_batch[start_idx:end_idx]
                mb_returns = returns[start_idx:end_idx]
                mb_advantages = advantages[start_idx:end_idx]
                mb_old_value = old_value_batch[start_idx:end_idx]
                mb_pi_w_hidden = pi_w_hidden_batch[start_idx:end_idx] if pi_w_hidden_batch is not None else None

                dist_batch, value_batch = policy(mb_obs, mb_pi_w_hidden)

                # Calculate the log probability of the actions
                log_probs = self.calculate_action_log_probs(dist_batch, mb_act)

                # Compute policy loss
                ratio = torch.exp(log_probs - self.calculate_action_log_probs(dist_batch.detach(), mb_act))
                surr1 = ratio * mb_advantages
                surr2 = torch.clamp(ratio, 1.0 - self.eps_clip, 1.0 + self.eps_clip) * mb_advantages
                pi_loss = -torch.min(surr1, surr2).mean()

                # Compute value loss
                value_loss = 0.5 * (mb_returns - value_batch).pow(2).mean()

                # Compute entropy loss
                entropy_loss = dist_batch.entropy().mean()

                loss = pi_loss + self.value_coef * value_loss - self.entropy_coef * entropy_loss
                loss.backward()

                if grad_accumulation_cnt % grad_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(policy.parameters(), self.grad_clip_norm)
                    policy.optimizer.step()
                    policy.optimizer.zero_grad()
                grad_accumulation_cnt += 1

                pi_loss_list.append(pi_loss.item())
                value_loss_list.append(value_loss.item())
                entropy_loss_list.append(entropy_loss.item())

        return {
            'Loss/pi': np.mean(pi_loss_list),
            'Loss/v': np.mean(value_loss_list),
            'Loss/entropy': np.mean(entropy_loss_list)
        }

    def calculate_action_log_probs(self, dist_batch, actions):
        log_probs = []
        for dist, action in zip(dist_batch, actions):
            pose0, pose1 = action['pose0'], action['pose1']
            pick_log_prob = dist.log_prob(torch.tensor(pose0[0] + pose0[1]).to(dist.loc.device))
            place_log_prob = dist.log_prob(torch.tensor(pose1[0] + pose1[1]).to(dist.loc.device))
            log_probs.append(pick_log_prob + place_log_prob)
        return torch.stack(log_probs)

    def log_training_progress(self):
        rew_batch, done_batch = self.storage.fetch_log_data()
        self.logger.feed(rew_batch, done_batch)
        self.logger.dump()


def adjust_lr(optimizer, init_lr, timesteps, max_timesteps):
    lr = init_lr * (1 - (timesteps / max_timesteps))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return optimizer
