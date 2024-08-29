import os
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
import torch
from tqdm import trange
from cliport.utils import utils as cliport_utils
from .configs.global_configs import get_global_variable
from .utils import ProcgenReplayBufferOnPolicy, CliportReplayBufferOnPolicy, ProcgenReplayBufferOffPolicy, CliportReplayBufferOffPolicy, \
    ReplayBufferOffPolicy


def adjust_lr(optimizer, init_lr, timesteps, max_timesteps):
    lr = init_lr * (1 - (timesteps / max_timesteps))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return optimizer


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
        self.rb = None
        self.logger = logger

    def train(self, policy, evaluator, train_env=None, dataset=None):
        for i in trange(self.training_steps):
            if dataset is not None:
                self.train_one_iteration_offline(policy, dataset)
            elif train_env is not None:
                self.train_one_iteration_online(policy, train_env)
            else:
                raise ValueError("Either train_env or dataset should be provided for training")

            if (i + 1) % self.log_freq == 0:
                evaluator.evaluate_policy(policy)
                if (i + 1) % self.save_freq == 0 or evaluator.model_improved['id']:
                    evaluator.best_id_index = i
                    policy.save_model(os.path.join(self.save_dir, f"id_model_{i}.pt"))
                if (i + 1) % self.save_freq == 0 or evaluator.model_improved['ood']:
                    evaluator.best_ood_index = i
                    policy.save_model(os.path.join(self.save_dir, f"ood_model_{i}.pt"))

        train_env.close()

    def test(self, policy, test_env, best_model_index, id_evaluated=True):
        policy = self.load_fresh_model(policy, best_model_index, id_evaluated)

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
                    obs, pi_w_hidden = self._reset_environment(test_env)
                    ep_steps = 0

        return reward_batch, done_batch

    def _should_reset_environment(self, done, ep_steps, env):
        return (get_global_variable('benchmark') == 'cliport' and (done or ep_steps == env.base_env.task.max_steps))

    def _reset_environment(self, env):
        env.base_env.seed(env.base_env._seed + 1)
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

    def load_fresh_model(self, policy, best_index, id_evaluated):
        model_name = "id_model" if id_evaluated else "ood_model"
        if self.load_best:
            model_path = os.path.join(self.save_dir, f'{model_name}_{best_index}.pt')
        elif self.load_custom_index != -1:
            model_path = os.path.join(self.save_dir, f'{model_name}_{self.load_custom_index}.pt')
        elif self.load_last:
            model_path = str(sorted(Path(self.save_dir).iterdir(), key=os.path.getmtime)[0])
        checkpoint = torch.load(model_path)
        policy.load_state_dict(checkpoint["model_state_dict"])
        return policy

    def train_one_iteration_online(self, help_policy, train_env):
        pass

    def train_one_iteration_offline(self, help_policy, dataset):
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
        obs_shape = env.observation_space.shape if get_global_variable("benchmark") == 'procgen' else (320, 160, 6)
        num_envs = env.base_env.num_envs
        self.rb = (ProcgenReplayBufferOnPolicy if get_global_variable("benchmark") == 'procgen' else CliportReplayBufferOnPolicy)(self.gamma,
                                                                                                                                  self.lmbda,
                                                                                                                                  self.use_gae,
                                                                                                                                  self.normalize_adv,
                                                                                                                                  obs_shape,
                                                                                                                                  config.rollout_length,
                                                                                                                                  num_envs)

    def train_one_iteration_online(self, help_policy, train_env):
        obs, pi_w_hidden = train_env.reset()
        ep_steps = 0

        for _ in range(self.rollout_length):
            act, log_prob_act, value = help_policy.predict(obs, pi_w_hidden)
            next_obs, rew, done, info, pi_w_hidden = train_env.step(act)

            self.rb.add_transition(obs, act, log_prob_act, rew, next_obs, done, value, info)

            obs = next_obs
            ep_steps += 1

            if self._should_reset_environment(done, ep_steps, train_env):
                self.rb.store_last_done()
                obs, pi_w_hidden = self._reset_environment(train_env)
                ep_steps = 0

        _, _, last_val = help_policy.predict(obs, pi_w_hidden)
        self.rb.store_last(obs, last_val)
        self.rb.compute_estimates()

        summary = self.update_policy_online(help_policy.policy, train_env)
        self._update_training_progress(help_policy.policy)
        self.logger.wandb_log_loss(summary)
        return summary

    def train_one_iteration_offline(self, policy, dataset):
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

        summary = self.update_policy_offline(policy, all_obs, all_acts, all_rewards, all_dones, all_infos)
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
        self.t += self.rollout_length * self.rb.num_envs
        self.log_training_progress()
        policy.optimizer = adjust_lr(policy.optimizer, policy.learning_rate, self.t, self.training_steps)

    def update_policy_online(self, policy, train_env):
        pi_loss_list, value_loss_list, entropy_loss_list = [], [], []
        batch_size = self.rollout_length * self.rb.num_envs // self.mini_batch_per_epoch
        self.mini_batch_size = min(self.mini_batch_size, batch_size)
        grad_accumulation_steps = batch_size / self.mini_batch_size
        grad_accumulation_cnt = 1

        for _ in range(self.epoch):
            generator = self.rb.fetch_train_generator(mini_batch_size=self.mini_batch_size)
            for sample in generator:
                obs_batch, act_batch, done_batch, old_log_prob_act_batch, old_value_batch, return_batch, adv_batch, info_batch = sample
                pi_w_hidden_batch = train_env.get_weak_policy_features(obs_batch, info_batch)
                if get_global_variable("benchmark") == 'cliport':
                    obs_batch = obs_batch.permute(0, 3, 1, 2)
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
        rew_batch, done_batch = self.rb.fetch_log_data()
        self.logger.feed(rew_batch, done_batch)
        self.logger.dump()


class DQNAlgorithm(Algorithm):
    def __init__(self, config, logger, env):
        super().__init__(config, logger)
        self.t = 0
        self.batch_size = config.batch_size
        self.rollout_length = config.rollout_length
        self.mini_batch_per_epoch = config.mini_batch_per_epoch
        self.target_update_frequency = config.target_update_frequency
        self.gamma = config.gamma
        self.n_step = config.n_step
        self.epoch = config.epoch
        self.prioritized_replay = config.prioritized_replay
        self.obs_shape = env.observation_space.shape if get_global_variable("benchmark") == 'procgen' else (320, 160, 6)
        self.num_actions = env.action_space.n
        self.num_envs = env.base_env.num_envs
        self.grad_clip_norm = config.grad_clip_norm

        ReplayBufferClass = ProcgenReplayBufferOffPolicy if get_global_variable("benchmark") == 'procgen' else CliportReplayBufferOffPolicy
        self.rb = ReplayBufferClass(capacity=config.buffer_size, obs_shape=self.obs_shape, action_size=self.num_actions, num_envs=self.num_envs,
                                    n_step=self.n_step, gamma=self.gamma)

    def train_one_iteration_online(self, help_policy, train_env):
        obs, pi_w_hidden = train_env.reset()
        ep_steps = 0
        episode_buffers = [[] for _ in range(self.num_envs)]
        completed_episodes = []
        while len(completed_episodes) < self.rollout_length * self.num_envs:
            action, _, q_values = help_policy.predict(obs, pi_w_hidden)
            next_obs, reward, done, info, pi_w_hidden = train_env.step(action)

            for env_idx in range(self.num_envs):
                episode_buffers[env_idx].append((
                    obs[env_idx] if self.num_envs > 1 else obs,
                    action[env_idx] if self.num_envs > 1 else action,
                    reward[env_idx] if self.num_envs > 1 else reward,
                    next_obs[env_idx] if self.num_envs > 1 else next_obs,
                    done[env_idx] if self.num_envs > 1 else done,
                    info[env_idx] if self.num_envs > 1 else (info[0] if isinstance(info, list) else info)
                ))

            if self._should_reset_environment(done, ep_steps, train_env):
                # only come here for Cliport
                idx = 0
                if self.num_envs > 1:
                    raise NotImplementedError("Resetting environment is not implemented for multiple environments for Cliport")
                else:
                    last_step = list(episode_buffers[idx][-1])
                    last_step[4] = True
                    episode_buffers[idx][-1] = tuple(last_step)  # setting the last step as done
                for i, transition in enumerate(episode_buffers[idx]):
                    completed_episodes.append(transition)
                episode_buffers[idx] = []
                obs, pi_w_hidden = self._reset_environment(train_env)
                ep_steps = 0
            elif isinstance(done, np.ndarray) and done.any():
                # only come here for Procgen
                completed_episodes_idx = done.nonzero()[0]
                for idx in completed_episodes_idx:
                    for i, transition in enumerate(episode_buffers[idx]):
                        completed_episodes.append(transition)
                    episode_buffers[idx] = []
                obs = next_obs
            else:
                obs = next_obs
                ep_steps += 1
            self.t += 1

        for transition in completed_episodes:
            self.rb.add_transition(*transition)

        summary = self.update_policy_online(help_policy.policy, train_env)
        if self.t % self.target_update_frequency == 0:
            help_policy.update_target_network()
        self._update_training_progress(help_policy.policy)
        self.logger.wandb_log_loss(summary)
        return summary

    def train_one_iteration_offline(self, policy, dataset, weak_policy, strong_policy):
        NotImplementedError("Offline training is not implemented for DQN")

    def update_policy_online(self, policy, train_env):
        value_loss_list = []
        batch_size = len(self.rb) // self.mini_batch_per_epoch
        for _ in range(self.epoch):
            for batch in range(self.mini_batch_per_epoch):
                obs_batch, act_batch, raw_reward_batch, reward_batch, next_obs_batch, done_batch, info_batch, next_info_batch = self.rb.sample(batch_size)
                next_info_batch_nones = [i for i, v in enumerate(next_info_batch) if v is None]

                pi_w_hidden_batch = train_env.get_weak_policy_features(obs_batch, info_batch)
                next_pi_w_hidden_batch = train_env.get_weak_policy_features(next_obs_batch, next_info_batch)
                if get_global_variable("benchmark") == 'cliport':
                    obs_batch = obs_batch.permute(0, 3, 1, 2)
                    next_obs_batch = next_obs_batch.permute(0, 3, 1, 2)

                current_q_dist = policy(obs_batch, pi_w_hidden_batch)
                next_q_dist = policy.target_network(next_obs_batch, next_pi_w_hidden_batch)

                # Select the distributional Q-value for the taken action
                current_q_action = current_q_dist[range(batch_size), act_batch.long()]

                # Compute the target distribution
                next_q_dist_max = next_q_dist[range(batch_size), next_q_dist.mean(dim=2).argmax(dim=1)]
                target_q_dist = self.compute_target_distribution(next_q_dist_max, reward_batch, done_batch, policy, next_info_batch_nones)

                loss = policy.compute_loss(current_q_action, target_q_dist)
                policy.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(policy.parameters(), self.grad_clip_norm)
                policy.optimizer.step()
                value_loss_list.append(loss.item())
        return {'Loss/q': np.mean(value_loss_list)}

    def compute_target_distribution(self, next_q_dist, rewards, dones, policy, terminal_indices):
        # Compute the projected distribution
        target_z = rewards.unsqueeze(-1) + (1 - dones.unsqueeze(-1)) * self.gamma ** self.n_step * policy.support.unsqueeze(0)
        target_z = target_z.clamp(policy.v_min, policy.v_max)

        # Compute the projected probabilities
        b = (target_z - policy.v_min) / ((policy.v_max - policy.v_min) / (policy.num_atoms - 1))
        l = b.floor().long()
        u = b.ceil().long()

        target_dist = torch.zeros_like(next_q_dist)
        offset = torch.linspace(0, (next_q_dist.shape[0] - 1) * policy.num_atoms, next_q_dist.shape[0]).unsqueeze(1).expand(next_q_dist.shape[0], policy.num_atoms).long().to(get_global_variable('device'))
        target_dist.view(-1).index_add_(0, (l + offset).view(-1), (next_q_dist * (u.float() - b)).view(-1))
        target_dist.view(-1).index_add_(0, (u + offset).view(-1), (next_q_dist * (b - l.float())).view(-1))

        # For terminal states, directly assign the reward to the target distribution (used only for Cliport)
        if terminal_indices and get_global_variable("benchmark") == 'cliport':
            target_dist[terminal_indices] = rewards[terminal_indices].unsqueeze(-1).expand(-1, policy.num_atoms)

        return target_dist

    def update_policy_offline(self, policy, obs_batch, act_batch, reward_batch, done_batch, info_batch, weak_policy):
        NotImplementedError("Offline training is not implemented for DQN")

    def _update_training_progress(self, policy):
        self.log_training_progress()
        policy.optimizer = adjust_lr(policy.optimizer, policy.learning_rate, self.t, self.training_steps)

    def log_training_progress(self):
        raw_rew_batch, rew_batch, done_batch = self.rb.fetch_log_data()
        self.logger.feed(raw_rew_batch, done_batch)
        self.logger.dump()


class OODAlgorithm(Algorithm):
    def __init__(self, config, logger):
        super().__init__(config, logger)
        self.rollout_len = config.rollout_len
        self.eval_rollout_len = self.rollout_len // 10
        self.test_size = config.test_size
        self.seed = config.seed

    def train(self, policy, evaluator, train_env=None, dataset=None):
        if train_env is None:
            raise ValueError("offline training should be implemented!")

        train_obs = policy.gather_rollouts(self.rollout_len, train_env)
        classifier = policy.train(train_obs)
        evaluator.evaluate_detector(policy, classifier, train_env)
        policy.save_model(self.save_dir, classifier)
        train_env.close()

    def test(self, ood_detector, test_env, best_model_index, id_evaluated=True):
        loaded_policy = ood_detector.load_model(self.save_dir)
        ood_detector.policy = loaded_policy
        num_envs = test_env.base_env.num_envs
        reward_batch, done_batch = self._run_test_episodes(ood_detector, test_env)

        episode_stats = self._calculate_episode_stats(reward_batch, done_batch, num_envs)
        self._print_test_performance(episode_stats)

        test_env.close()

    def _run_test_episodes(self, ood_detector, test_env):
        num_envs = test_env.base_env.num_envs
        reward_batch = np.zeros((self.test_steps, num_envs))
        done_batch = np.zeros((self.test_steps, num_envs), dtype=bool)

        with torch.no_grad():
            obs, pi_w_hidden = test_env.reset()
            ep_steps = 0

            for i in range(self.test_steps):
                if get_global_variable("benchmark") == 'cliport':
                    obs = cliport_utils.get_image(obs)
                obs = obs.reshape(num_envs, -1)
                action = ood_detector.predict(obs, pi_w_hidden)
                obs, reward, done, info, pi_w_hidden = test_env.step(action)
                reward_batch[i], done_batch[i] = reward, done
                ep_steps += 1

                if self._should_reset_environment(done, ep_steps, test_env):
                    done_batch[i] = True
                    obs, pi_w_hidden = self._reset_environment(test_env)
                    ep_steps = 0

        return reward_batch, done_batch

    def train_one_iteration_online(self, policy, train_env):
        NotImplementedError("Online training is not implemented for OOD")

    def train_one_iteration_offline(self, policy, dataset, weak_policy, strong_policy):
        NotImplementedError("Offline training is not implemented for OOD")


class SVDDAlgorithm(OODAlgorithm):
    def __init__(self, config, logger, env):
        super().__init__(config, logger)


class KDEAlgorithm(OODAlgorithm):
    def __init__(self, config, logger, env):
        super().__init__(config, logger)


class NonParametricAlgorithm(Algorithm):
    def __init__(self, config, logger, env):
        self.help_percentile = config.help_percentile
        self.rollout_len = config.rollout_len
        self.sampled_logit_threshold = None
        self.save_dir = config.save_dir
        self.test_steps = config.test_steps

    def train(self, help_policy, evaluator, train_env=None, dataset=None):
        help_policy.gather_rollouts(self.rollout_len, train_env)
        help_policy.determine_sampled_logit_threshold()
        evaluator.evaluate_nonparam(help_policy)
        help_policy.save_model(self.save_dir)
        train_env.close()

    def test(self, help_policy, test_env, best_model_index, id_evaluated=True):
        help_policy = help_policy.load_model(self.save_dir)
        num_envs = test_env.base_env.num_envs
        reward_batch, done_batch = self._run_test_episodes(help_policy, test_env)

        episode_stats = self._calculate_episode_stats(reward_batch, done_batch, num_envs)
        self._print_test_performance(episode_stats)

        test_env.close()

    def _run_test_episodes(self, model, test_env):
        num_envs = test_env.base_env.num_envs
        reward_batch = np.zeros((self.test_steps, num_envs))
        done_batch = np.zeros((self.test_steps, num_envs), dtype=bool)

        with torch.no_grad():
            obs, pi_w_hidden = test_env.reset()
            ep_steps = 0
            for i in range(self.test_steps):
                sampled_logits, max_logits, sampled_probs, max_probs, entropy = test_env.weak_policy.get_logits_probs(obs)
                actions = self._get_actions(model, sampled_logits, max_logits, sampled_probs, max_probs, entropy)
                obs, reward, done, info, pi_w_hidden = test_env.step(actions)
                reward_batch[i] = reward
                done_batch[i] = done
                ep_steps += 1
                if self._should_reset_environment(done, ep_steps, test_env):
                    done_batch[i] = True
                    obs, pi_w_hidden = self._reset_environment(test_env)
                    ep_steps = 0

        return reward_batch, done_batch

    def _get_actions(self, policy, sampled_logits, max_logits, sampled_probs, max_probs, entropy):
        policy_type = policy.type
        threshold = policy.threshold
        if policy_type == "sampled_logit":
            return np.where(sampled_logits < threshold, 0, 1)
        elif policy_type == "max_logit":
            return np.where(max_logits < threshold, 0, 1)
        elif policy_type == "sampled_prob":
            return np.where(sampled_probs < threshold, 0, 1)
        elif policy_type == "max_prob":
            return np.where(max_probs < threshold, 0, 1)
        elif policy_type == "entropy":
            return np.where(entropy < threshold, 0, 1)
        else:
            raise ValueError(f"Unknown policy type: {policy_type}")


class RandomAlgorithm(Algorithm):
    def __init__(self, config, logger, env):
        self.save_dir = config.save_dir
        self.test_steps = config.test_steps
        self.help_percentage = config.help_percentage

    def train(self, policy, evaluator, train_env):
        print("Random policy does not require training")
        pass

    def test(self, policy, test_env, best_model_index, id_evaluated=True):
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
                actions = np.where(np.random.rand(obs.shape[0]) < self.help_percentage, 0, 1)
                _, reward, done, _, _ = test_env.step(actions)
                reward_batch[i], done_batch[i] = reward, done
                ep_steps += 1

                if self._should_reset_environment(done, ep_steps, test_env):
                    done_batch[i] = True
                    obs, pi_w_hidden = self._reset_environment(test_env)
                    ep_steps = 0
        return reward_batch, done_batch
