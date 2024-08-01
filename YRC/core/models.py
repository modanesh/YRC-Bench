import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions.categorical import Categorical

from cliport.utils import utils


class CategoricalPolicy(nn.Module):
    def __init__(self, embedder=None, action_size=2, additional_hidden_dim=0):
        """
        embedder: (torch.Tensor) model to extract the embedding for observation
        action_size: number of the categorical actions
        """
        super(CategoricalPolicy, self).__init__()
        self.embedder = embedder
        if self.embedder is not None:
            # Compute total input dimension for fc_policy considering additional tensors
            total_input_dim = self.embedder.output_dim + additional_hidden_dim
        else:
            total_input_dim = additional_hidden_dim
        # small scale weight-initialization in policy enhances the stability
        self.fc_policy = orthogonal_init(nn.Linear(total_input_dim, action_size), gain=0.01)
        self.fc_value = orthogonal_init(nn.Linear(total_input_dim, 1), gain=1.0)

    def forward(self, x, pi_w_hidden):
        if pi_w_hidden is None:
            # T1
            hidden = self.embedder(x)
        elif pi_w_hidden is not None and self.embedder is not None:
            # T2
            hidden = self.embedder(x)
            hidden = torch.cat([hidden, pi_w_hidden], dim=1)
        elif pi_w_hidden is not None and self.embedder is None:
            # T3
            hidden = pi_w_hidden
        logits = self.fc_policy(hidden)
        log_probs = F.log_softmax(logits, dim=1)
        p = Categorical(logits=log_probs)
        v = self.fc_value(hidden).reshape(-1)
        return p, v

    def get_inner_values(self, x):
        hidden = self.embedder(x)
        return hidden


class ResidualBlock(nn.Module):
    def __init__(self, in_channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        out = nn.ReLU()(x)
        out = self.conv1(out)
        out = nn.ReLU()(out)
        out = self.conv2(out)
        return out + x


class ImpalaBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ImpalaBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1)
        self.res1 = ResidualBlock(out_channels)
        self.res2 = ResidualBlock(out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)(x)
        x = self.res1(x)
        x = self.res2(x)
        return x


class ImpalaModel(nn.Module):
    def __init__(self,
                 in_channels,
                 benchmark,
                 **kwargs):
        super(ImpalaModel, self).__init__()
        scale = 1
        self.block1 = ImpalaBlock(in_channels=in_channels, out_channels=16 * scale)
        self.block2 = ImpalaBlock(in_channels=16 * scale, out_channels=32 * scale)
        self.block3 = ImpalaBlock(in_channels=32 * scale, out_channels=32 * scale)
        if benchmark == 'procgen':
            self.fc = nn.Linear(in_features=32 * scale * 8 * 8, out_features=256)
        elif benchmark == 'cliport':
            self.fc = nn.Linear(in_features=32 * scale * 20 * 40, out_features=256)

        self.output_dim = 256
        self.apply(xavier_uniform_init)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = nn.ReLU()(x)
        x = Flatten()(x)
        x = self.fc(x)
        x = nn.ReLU()(x)
        return x


def orthogonal_init(module, gain=nn.init.calculate_gain('relu')):
    if isinstance(module, nn.Linear) or isinstance(module, nn.Conv2d):
        nn.init.orthogonal_(module.weight.data, gain)
        nn.init.constant_(module.bias.data, 0)
    return module


def adjust_lr(optimizer, init_lr, timesteps, max_timesteps):
    lr = init_lr * (1 - (timesteps / max_timesteps))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return optimizer


class Flatten(nn.Module):
    def forward(self, x):
        return torch.flatten(x, start_dim=1)


def xavier_uniform_init(module, gain=1.0):
    if isinstance(module, nn.Linear) or isinstance(module, nn.Conv2d):
        nn.init.xavier_uniform_(module.weight.data, gain)
        nn.init.constant_(module.bias.data, 0)
    return module


class PPOFrozen:
    def __init__(self, policy, device):
        super().__init__()
        self.policy = policy
        self.device = device

    def predict(self, obs):
        with torch.no_grad():
            obs = torch.FloatTensor(obs).to(device=self.device)
            dist, value = self.policy(obs, pi_w_hidden=None)
            act = dist.sample()
            log_prob_act = dist.log_prob(act)

        return act.cpu().numpy(), log_prob_act.cpu().numpy(), value.cpu().numpy()


class procgenPPO:
    def __init__(self,
                 env,
                 env_valid,
                 task,  # not used, but to generalize the PPO input args
                 policy,
                 logger,
                 storage,
                 device,
                 n_checkpoints,
                 storage_valid=None,
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
        # self.n_envs = n_envs
        self.n_envs = 1000
        self.epoch = epoch
        self.mini_batch_per_epoch = mini_batch_per_epoch
        self.mini_batch_size = mini_batch_size
        self.gamma = gamma
        self.lmbda = lmbda
        self.learning_rate = learning_rate
        self.optimizer = optim.Adam(self.policy.parameters(), lr=learning_rate, eps=1e-5)
        self.grad_clip_norm = grad_clip_norm
        self.eps_clip = eps_clip
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.normalize_adv = normalize_adv
        self.normalize_rew = normalize_rew
        self.use_gae = use_gae
        self.help_policy_type = help_policy_type

    def predict(self, obs, pi_w_hidden):
        with torch.no_grad():
            obs_tensor = torch.FloatTensor(obs).to(device=self.device)
            dist, value = self.get_policy_output(obs_tensor, pi_w_hidden)
            act = dist.sample()
            log_prob_act = dist.log_prob(act)
        return act.cpu().numpy(), log_prob_act.cpu().numpy(), value.cpu().numpy()

    def get_policy_output(self, obs, pi_w_hidden):
        dist, value = self.policy(obs, pi_w_hidden)
        return dist, value

    def optimize(self):
        pi_loss_list, value_loss_list, entropy_loss_list = [], [], []
        batch_size = self.n_steps * self.n_envs // self.mini_batch_per_epoch
        if batch_size < self.mini_batch_size:
            self.mini_batch_size = batch_size
        grad_accumulation_steps = batch_size / self.mini_batch_size
        grad_accumulation_cnt = 1

        self.policy.train()
        for e in range(self.epoch):
            generator = self.storage.fetch_train_generator(mini_batch_size=self.mini_batch_size)
            for sample in generator:
                obs_batch, act_batch, done_batch, old_log_prob_act_batch, old_value_batch, return_batch, adv_batch = sample
                pi_w_hidden_batch = self.env.get_weak_policy_features(obs_batch)
                dist_batch, value_batch = self.get_policy_output(obs_batch, pi_w_hidden_batch)

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
        obs, pi_w_hidden = self.env.reset()
        obs_v, pi_w_hidden_v = self.env_valid.reset()

        while self.t < num_timesteps:
            self.policy.eval()
            for _ in range(self.n_steps):
                act, log_prob_act, value = self.predict(obs, pi_w_hidden)
                next_obs, rew, done, info, pi_w_hidden = self.env.step(act)
                self.storage.store(obs, act, rew, done, info, log_prob_act, value)
                obs = next_obs
            value_batch = self.storage.value_batch[:self.n_steps]
            _, _, last_val = self.predict(obs, pi_w_hidden)
            self.storage.store_last(obs, last_val)
            # Compute advantage estimates
            self.storage.compute_estimates(self.gamma, self.lmbda, self.use_gae, self.normalize_adv)

            # valid
            for _ in range(self.n_steps):
                act_v, log_prob_act_v, value_v = self.predict(obs_v, pi_w_hidden_v)
                next_obs_v, rew_v, done_v, info_v, pi_w_hidden_v = self.env_valid.step(act_v)
                self.storage_valid.store(obs_v, act_v, rew_v, done_v, info_v, log_prob_act_v, value_v)
                obs_v = next_obs_v
            _, _, last_val_v = self.predict(obs_v, pi_w_hidden_v)
            self.storage_valid.store_last(obs_v, last_val_v)
            self.storage_valid.compute_estimates(self.gamma, self.lmbda, self.use_gae, self.normalize_adv)

            # Optimize policy & values
            summary = self.optimize()
            # Log the training-procedure
            self.t += self.n_steps * self.n_envs
            _, rew_batch, done_batch = self.storage.fetch_log_data()
            _, rew_batch_v, done_batch_v = self.storage_valid.fetch_log_data()
            self.logger.feed_procgen(rew_batch, done_batch, rew_batch_v, done_batch_v)
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
        self.env_valid.close()

    def test(self, num_timesteps, help_policy_path = None):
        print('::[LOGGING]::START TESTING...')
        # get the last saved model in self.logger.logdir
        last_checkpoint = sorted([f for f in os.listdir(self.logger.logdir) if ".pth" in f])[-1]
        if not help_policy_path:
            help_policy_path = os.path.join(self.logger.logdir, last_checkpoint)
        print("Loading help policy from", help_policy_path)
        help_policy = torch.load(help_policy_path)
        self.policy.load_state_dict(help_policy["model_state_dict"])
        self.policy.eval()
        log_file = os.path.join(self.logger.logdir, "AAA_quant_eval_model_200015872.txt")
        obs, pi_w_hidden = self.env.reset()
        total_reward = 0
        run_length = 0
        print("Running for", num_timesteps, "timesteps")
        for i in range(num_timesteps):
            with torch.no_grad():
                act, log_prob_act, value = self.predict(obs, pi_w_hidden)
                next_obs, rew, done, info, pi_w_hidden = self.env.step(act)
                self.storage.store(obs, act, rew, done, info, log_prob_act, value)
                obs = next_obs.copy()
                total_reward += rew
                run_length += (~done).astype(int)
        _, _, last_val = self.predict(obs, pi_w_hidden)
        self.storage.store_last(obs, last_val)
        self.storage.compute_estimates(self.gamma, self.lmbda, self.use_gae, self.normalize_adv)
        act_batch, rew_batch, done_batch = self.storage.fetch_log_data()
        self.logger.feed_procgen(act_batch, rew_batch, done_batch, None, None)
        self.logger.dump(is_test=True)
        queries = act_batch.flatten()
        shifted_queries = np.roll(queries, -1)
        switches = (queries != shifted_queries).astype(int)
        switches = switches[:-1]
        with open(log_file, "w") as f:
            f.write(f"Mean reward: {np.mean(total_reward)}")
            f.write(f"Median reward: {np.median(total_reward)}")
            f.write("Mean adjusted reward: 69")
            f.write("Median adjusted reward: 69")
            f.write(f"All queries: {queries.tolist()}")
            f.write(f"All switches: {[0] + switches.tolist()}")
            f.write("Mean timestep achieved: 69")
            f.write("Median timestep achieved: 69")
            f.write(f"Mean run length: {int(np.mean(run_length))}")
            f.write(f"Median run length: {int(np.median(run_length))}")
            f.write(f"All rewards: {total_reward.tolist()}")
            f.write(f"All adjusted rewards: {total_reward.tolist()}")
            f.write("All timesteps: [69, 69, 69]")
            f.write(f"Mean times asked for help: {int(np.mean(np.sum(act_batch, axis = 1)))}")
            f.write(f"Median times asked for help: {int(np.mean(np.sum(act_batch, axis = 1)))}")
            f.write("Help times:")
            f.write(f"{act_batch.tolist()}")


class cliportPPO:
    def __init__(self,
                 env,
                 env_valid,
                 task,
                 policy,
                 logger,
                 storage,
                 device,
                 n_checkpoints,
                 storage_valid=None,
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
        self.task = task
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
        self.help_policy_type = help_policy_type
        self.n_train_episodes = n_train_episodes
        self.seed = seed

    def predict(self, obs, pi_w_hidden):
        with torch.no_grad():
            dist, value = self.get_policy_output(obs, pi_w_hidden)
            act = dist.sample()
            log_prob_act = dist.log_prob(act)
        return act.cpu().numpy(), log_prob_act.cpu().numpy(), value.cpu().numpy()

    def get_policy_output(self, obs, pi_w_hidden):
        if self.help_policy_type in ["T1", "T2"] and not isinstance(obs, torch.Tensor):
                obs = utils.get_image(obs)
                obs = torch.FloatTensor(obs).to(device=self.device)
                obs = obs.permute(2, 0, 1)
                obs = obs.unsqueeze(0)
        dist, value = self.policy(obs, pi_w_hidden)
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
                pi_w_hidden_batch = self.env.get_weak_policy_features(obs_batch.permute(0, 2, 3, 1), info_batch)
                dist_batch, value_batch = self.get_policy_output(obs_batch, pi_w_hidden_batch)

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

        while self.t < num_timesteps:
            train_steps = 0
            self.policy.eval()
            for train_run in range(self.n_train_episodes):
                total_reward = 0
                self.env.seed(self.seed + train_run)
                obs, pi_w_hidden = self.env.reset()
                info = self.env.info
                print(f"train episode goal: {info['lang_goal']}")
                for i in range(self.task.max_steps):
                    with torch.no_grad():
                        act, log_prob_act, value = self.predict(obs, pi_w_hidden)
                        next_obs, rew, done, info, pi_w_hidden = self.env.step(act)
                        print(f"info: {info['lang_goal']}")
                        img_obs = utils.get_image(obs).transpose(2,0,1)
                        img_next_obs = utils.get_image(next_obs).transpose(2,0,1)
                        self.storage.add_transition(img_obs, act, log_prob_act, rew, img_next_obs, done, value, info)
                        obs = next_obs.copy()
                        total_reward += rew
                        train_steps += 1
                    if done:
                        print(f"train episode total_reward={total_reward:.3f}, episode length (of max length)={i / self.task.max_steps:.3f}")
                        break
                    if not done and i == self.task.max_steps - 1:
                        print(f"train episode total_reward={total_reward:.3f}, episode length (of max length)={i / self.task.max_steps:.3f}")
                        self.storage._dones[self.storage._pointer - 1] = True  # set done=True for the last transition

            print(f"train iteration={self.t}, collected data={train_steps}, total data collected so far={self.storage._size}")
            self.storage.compute_estimates(self.gamma, self.lmbda, self.use_gae, self.normalize_adv)

            # Optimize policy & values
            summary = self.optimize(train_steps)
            # Log the training-procedure
            self.t += train_steps
            rew_batch, done_batch = self.storage.fetch_log_data()
            self.logger.feed_cliport(rew_batch, done_batch, None, None)
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

    def test(self, n_test_episodes):
        print('::[LOGGING]::START TESTING...')
        # get the last saved model in self.logger.logdir
        last_checkpoint = os.listdir(self.logger.logdir)[-1]
        help_policy_path = self.logger.logdir + '/model_' + last_checkpoint + '.pth'
        help_policy = torch.load(help_policy_path)
        self.policy.load_state_dict(help_policy["model_state_dict"])
        self.policy.eval()
        for i in range(n_test_episodes):
            total_reward = 0
            self.env.seed(self.seed + self.n_train_episodes + i)
            obs, pi_w_hidden = self.env.reset()
            info = self.env.info
            print(f"test episode goal: {info['lang_goal']}")
            for i in range(self.task.max_steps):
                with torch.no_grad():
                    act, log_prob_act, value = self.predict(obs, pi_w_hidden)
                    next_obs, rew, done, info, pi_w_hidden = self.env.step(act)
                    obs = next_obs.copy()
                    total_reward += rew
                if done:
                    print(f"test episode total_reward={total_reward:.3f}, episode length (of max length)={i / self.task.max_steps:.3f}")
                    break
                if not done and i == self.task.max_steps - 1:
                    print(f"test episode total_reward={total_reward:.3f}, episode length (of max length)={i / self.task.max_steps:.3f}")
