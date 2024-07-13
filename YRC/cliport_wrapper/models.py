from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
import numpy as np
import torch.optim as optim
from cliport.utils import utils


class CategoricalPolicyT3(nn.Module):
    def __init__(self, action_size, weak_agent_hidden_dim):
        """
        action_size: number of the categorical actions
        """
        super(CategoricalPolicyT3, self).__init__()
        # small scale weight-initialization in policy enhances the stability
        self.fc_policy = orthogonal_init(nn.Linear(weak_agent_hidden_dim, action_size), gain=0.01)
        self.fc_value = orthogonal_init(nn.Linear(weak_agent_hidden_dim, 1), gain=1.0)

    def forward(self, pi_w_hidden):
        hidden = pi_w_hidden
        logits = self.fc_policy(hidden)
        log_probs = F.log_softmax(logits, dim=1)
        p = Categorical(logits=log_probs)
        v = self.fc_value(hidden).reshape(-1)
        return p, v


class ResidualBlock(nn.Module):
    def __init__(self, in_features):
        super(ResidualBlock, self).__init__()
        self.fc1 = nn.Linear(in_features, in_features)
        self.fc2 = nn.Linear(in_features, in_features)

    def forward(self, x):
        out = F.relu(x)
        out = self.fc1(out)
        out = F.relu(out)
        out = self.fc2(out)
        return out + x


class ImpalaBlock(nn.Module):
    def __init__(self, in_features, out_features):
        super(ImpalaBlock, self).__init__()
        self.fc = nn.Linear(in_features, out_features)
        self.res1 = ResidualBlock(out_features)
        self.res2 = ResidualBlock(out_features)

    def forward(self, x):
        x = self.fc(x)
        x = self.res1(x)
        x = self.res2(x)
        return x


class ImpalaModel(nn.Module):
    def __init__(self, input_dim, scale=128):
        super(ImpalaModel, self).__init__()
        self.block1 = ImpalaBlock(input_dim, 16 * scale)
        self.block2 = ImpalaBlock(16 * scale, 32 * scale)
        self.block3 = ImpalaBlock(32 * scale, 32 * scale)
        self.fc = nn.Linear(32 * scale, 256)

        self.output_dim = 256
        self.apply(xavier_uniform_init)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = F.relu(x)
        x = self.fc(x)
        x = F.relu(x)
        return x


def orthogonal_init(module, gain=nn.init.calculate_gain('relu')):
    if isinstance(module, nn.Linear) or isinstance(module, nn.Conv2d):
        nn.init.orthogonal_(module.weight.data, gain)
        nn.init.constant_(module.bias.data, 0)
    return module


class Flatten(nn.Module):
    def forward(self, x):
        return torch.flatten(x, start_dim=1)


def xavier_uniform_init(module, gain=1.0):
    if isinstance(module, nn.Linear) or isinstance(module, nn.Conv2d):
        nn.init.xavier_uniform_(module.weight.data, gain)
        nn.init.constant_(module.bias.data, 0)
    return module


class PPO:
    def __init__(self,
                 env,
                 policy,
                 logger,
                 storage,
                 device,
                 n_checkpoints,
                 env_valid=None,
                 storage_valid=None,
                 pi_w=None,
                 pi_o=None,
                 help_policy_type=None,
                 n_steps=128,
                 n_envs=8,
                 epoch=3,
                 mini_batch_per_epoch=8,
                 mini_batch_size=32 * 8,
                 gamma=0.99,
                 lmbda=0.95,
                 learning_rate=2.5e-4,
                 grad_clip_norm=0.5,
                 eps_clip=0.2,
                 value_coef=0.5,
                 entropy_coef=0.01,
                 normalize_adv=True,
                 normalize_rew=True,
                 use_gae=True,
                 gae_lambda=0.95,
                 n_train_episodes=2,
                 tsk=None,
                 seed=0,
                 **kwargs):

        super().__init__()
        self.env = env
        self.policy = policy
        self.logger = logger
        self.storage = storage
        self.device = device
        self.num_checkpoints = n_checkpoints
        self.env_valid = env_valid
        self.storage_valid = storage_valid
        self.t = 0
        self.n_steps = n_steps
        self.n_envs = n_envs
        self.tsk = tsk
        self.epoch = epoch
        self.mini_batch_per_epoch = mini_batch_per_epoch
        self.mini_batch_size = mini_batch_size
        self.gamma = gamma
        self.lmbda = lmbda
        self.gae_lambda = gae_lambda
        self.learning_rate = learning_rate
        self.optimizer = optim.Adam(self.policy.parameters(), lr=learning_rate, eps=1e-5)
        self.grad_clip_norm = grad_clip_norm
        self.eps_clip = eps_clip
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.normalize_adv = normalize_adv
        self.normalize_rew = normalize_rew
        self.use_gae = use_gae
        self.pi_w = pi_w
        self.pi_o = pi_o
        self.help_policy_type = help_policy_type
        self.n_train_episodes = n_train_episodes
        self.seed = seed

    def predict(self, obs, info):
        with torch.no_grad():
            dist, value = self.get_policy_output(obs, info)
            act = dist.sample()
            log_prob_act = dist.log_prob(act)
        return act.cpu().numpy(), log_prob_act.cpu().numpy(), value.cpu().numpy()

    def get_inner_values(self, obs, info):
        with torch.no_grad():
            if isinstance(obs, Dict):
                img = [utils.get_image(obs)]
                info = [info]
            else:
                img = obs
            pick_features, place_features = self.pi_w.extract_features(img, info)
            pick_features_stacked = torch.stack(pick_features)
            place_features_stacked = torch.stack(place_features)
            features = torch.cat([pick_features_stacked, place_features_stacked], dim=1).to(device=self.device)
            return features

    def get_policy_output(self, obs, info):
        # TODO: complete this function
        # if self.help_policy_type == "T1":
        #     dist, value = self.policy(obs)
        # elif self.help_policy_type in ["T2", "T3"]:
        #     pi_w_hidden, pi_w_softmax = self.pi_w.policy.get_inner_values(obs)
        #     if self.help_policy_type == "T2":
        #         dist, value = self.policy(obs, pi_w_hidden, pi_w_softmax)
        #     elif self.help_policy_type == "T3":
        #         dist, value = self.policy(pi_w_hidden)
        # else:
        #     raise ValueError("Invalid help policy type.")
        pi_w_hidden = self.get_inner_values(obs, info)
        dist, value = self.policy(pi_w_hidden)
        return dist, value

    def optimize(self, n_steps):
        pi_loss_list, value_loss_list, entropy_loss_list = [], [], []
        batch_size = n_steps // self.mini_batch_per_epoch
        if batch_size < self.mini_batch_size:
            self.mini_batch_size = batch_size
        grad_accumulation_steps = batch_size / self.mini_batch_size
        grad_accumulation_cnt = 1

        self.policy.train()
        for e in range(self.epoch):
            generator = self.storage.fetch_train_generator(mini_batch_size=self.mini_batch_size)
            for sample in generator:
                obs_batch, act_batch, done_batch, old_log_prob_act_batch, old_value_batch, return_batch, adv_batch, info_batch = sample

                dist_batch, value_batch = self.get_policy_output(obs_batch, info_batch)

                # Clipped Surrogate Objective
                log_prob_act_batch = dist_batch.log_prob(act_batch)
                ratio = torch.exp(log_prob_act_batch - old_log_prob_act_batch)
                surr1 = ratio * adv_batch
                surr2 = torch.clamp(ratio, 1.0 - self.eps_clip, 1.0 + self.eps_clip) * adv_batch
                pi_loss = -torch.min(surr1, surr2).mean()

                # Clipped Bellman-Error
                clipped_value_batch = old_value_batch + (value_batch - old_value_batch).clamp(-self.eps_clip, self.eps_clip)
                v_surr1 = (value_batch - return_batch).pow(2)
                v_surr2 = (clipped_value_batch - return_batch).pow(2)
                value_loss = 0.5 * torch.max(v_surr1, v_surr2).mean()

                # Policy Entropy
                entropy_loss = dist_batch.entropy().mean()
                loss = pi_loss + self.value_coef * value_loss - self.entropy_coef * entropy_loss
                loss.backward()

                # Let model to handle the large batch-size with small gpu-memory
                if grad_accumulation_cnt % grad_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.grad_clip_norm)
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                grad_accumulation_cnt += 1
                pi_loss_list.append(-pi_loss.item())
                value_loss_list.append(-value_loss.item())
                entropy_loss_list.append(entropy_loss.item())

        summary = {'Loss/pi': np.mean(pi_loss_list),
                   'Loss/v': np.mean(value_loss_list),
                   'Loss/entropy': np.mean(entropy_loss_list)}
        return summary

    def train(self, num_timesteps):
        print('::[LOGGING]::START TRAINING...')
        save_every = num_timesteps // self.num_checkpoints
        checkpoint_cnt = 0
        if self.env_valid is not None:
            self.env_valid.seed(self.seed + self.n_train_episodes + 1)

        while self.t < num_timesteps:
            train_steps = 0
            self.policy.eval()
            for train_run in range(self.n_train_episodes):
                total_reward = 0
                self.env.seed(self.seed + train_run)
                obs = self.env.reset()
                info = self.env.info
                print(f"episode goal: {info['lang_goal']}")
                for i in range(self.tsk.max_steps):
                    with torch.no_grad():
                        act, log_prob_act, value = self.predict(obs, info)
                        next_obs, rew, done, info = self.env.step(act)
                        print(f"info: {info['lang_goal']}")
                        img_obs = utils.get_image(obs)
                        img_next_obs = utils.get_image(next_obs)
                        self.storage.add_transition(img_obs, act, log_prob_act, rew, img_next_obs, done, value, info)
                        obs = next_obs.copy()
                        total_reward += rew
                        train_steps += 1
                    if done:
                        print(f"episode total_reward={total_reward:.3f}, episode length (of max length)={i/self.tsk.max_steps:.3f}")
                        break
                    if not done and i == self.tsk.max_steps - 1:
                        print(f"episode total_reward={total_reward:.3f}, episode length (of max length)={i/self.tsk.max_steps:.3f}")
                        self.storage._dones[i] = True  # set done=True for the last transition

            print(f"iteration={self.t}, collected data={train_steps}, total data collected so far={self.storage._size}")
            self.storage.compute_estimates(self.gamma, self.lmbda, self.use_gae, normalize_adv=True)

            # Optimize policy & values
            summary = self.optimize(train_steps)
            # Log the training-procedure
            self.t += train_steps
            rew_batch, done_batch = self.storage.fetch_log_data()
            if self.storage_valid is not None:
                rew_batch_v, done_batch_v = self.storage_valid.fetch_log_data()
            else:
                rew_batch_v = done_batch_v = None
            self.logger.feed_cliport(rew_batch, done_batch, rew_batch_v, done_batch_v)
            self.logger.dump()
            self.optimizer = adjust_lr(self.optimizer, self.learning_rate, self.t, num_timesteps)
            # Save the model
            if self.t > ((checkpoint_cnt + 1) * save_every):
                print("Saving model.")
                torch.save({'model_state_dict': self.policy.state_dict(),
                            'optimizer_state_dict': self.optimizer.state_dict()},
                           self.logger.logdir + '/model_' + str(self.t) + '.pth')
                checkpoint_cnt += 1
        self.env.close()
        if self.env_valid is not None:
            self.env_valid.close()


def adjust_lr(optimizer, init_lr, timesteps, max_timesteps):
    lr = init_lr * (1 - (timesteps / max_timesteps))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return optimizer
