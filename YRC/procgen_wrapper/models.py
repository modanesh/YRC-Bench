import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions.categorical import Categorical

from utils import adjust_lr


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
                 pi_w=None,
                 pi_o=None,
                 oracle_cost=0.1,
                 switching_cost=0.1,
                 reward_min=0.0,
                 reward_max=1.0,
                 timeout=1000.0,
                 help_policy_type=None,
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
        self.pi_w = pi_w
        self.pi_o = pi_o
        self.switching_cost_per_action = (reward_max / timeout) * switching_cost
        self.oracle_cost_per_action = (reward_max / timeout) * oracle_cost
        self.help_policy_type = help_policy_type

    def predict(self, obs):
        with torch.no_grad():
            obs_tensor = torch.FloatTensor(obs).to(device=self.device)
            dist, value = self.get_policy_output(obs_tensor)
            act = dist.sample()
            log_prob_act = dist.log_prob(act)
        return act.cpu().numpy(), log_prob_act.cpu().numpy(), value.cpu().numpy()

    def get_policy_output(self, obs):
        if self.help_policy_type == "T1":
            dist, value = self.policy(obs)
        elif self.help_policy_type in ["T2", "T3"]:
            pi_w_hidden, pi_w_softmax = self.pi_w.policy.get_inner_values(obs)
            if self.help_policy_type == "T2":
                dist, value = self.policy(obs, pi_w_hidden, pi_w_softmax)
            elif self.help_policy_type == "T3":
                dist, value = self.policy(pi_w_hidden)
        else:
            dist, value = self.policy(obs)
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

                dist_batch, value_batch = self.get_policy_output(obs_batch)

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

    def train(self, num_timesteps, pi_h=False):
        print('::[LOGGING]::START TRAINING...')
        save_every = num_timesteps // self.num_checkpoints
        checkpoint_cnt = 0
        obs = self.env.reset()
        done = np.zeros(self.n_envs)
        past_act = None
        if self.env_valid is not None:
            obs_v = self.env_valid.reset()
            done_v = np.zeros(self.n_envs)
            past_act_v = None

        while self.t < num_timesteps:
            # Run Policy
            self.policy.eval()
            for _ in range(self.n_steps):
                act, log_prob_act, value = self.predict(obs)

                if pi_h:
                    act_exec = self.query_agents(obs, act)
                    next_obs, rew, done, info = self.env.step(act_exec)
                    rew = self.oracle_query_cost(rew, act)
                    if past_act is not None:
                        # get the index of elements where past_act and act are different
                        switching_idx = np.where(past_act != act)[0]
                        rew = self.switching_query_cost(rew, switching_idx)
                    past_act = act
                else:
                    next_obs, rew, done, info = self.env.step(act)

                self.storage.store(obs, act, rew, done, info, log_prob_act, value)
                obs = next_obs
            value_batch = self.storage.value_batch[:self.n_steps]
            _, _, last_val = self.predict(obs)
            self.storage.store_last(obs, last_val)
            # Compute advantage estimates
            self.storage.compute_estimates(self.gamma, self.lmbda, self.use_gae, self.normalize_adv)

            # valid
            if self.env_valid is not None:
                for _ in range(self.n_steps):
                    act_v, log_prob_act_v, value_v = self.predict(obs_v)

                    if pi_h:
                        act_exec_v = self.query_agents(obs_v, act_v)
                        next_obs_v, rew_v, done_v, info_v = self.env_valid.step(act_exec_v)
                        rew_v = self.oracle_query_cost(rew_v, act_v)
                        if past_act_v is not None:
                            # get the index of elements where past_act and act are different
                            switching_idx_v = np.where(past_act_v != act_v)[0]
                            rew_v = self.switching_query_cost(rew_v, switching_idx_v)
                        past_act_v = act_v
                    else:
                        next_obs_v, rew_v, done_v, info_v = self.env_valid.step(act_v)
                    self.storage_valid.store(obs_v, act_v,
                                             rew_v, done_v, info_v,
                                             log_prob_act_v, value_v)
                    obs_v = next_obs_v
                _, _, last_val_v = self.predict(obs_v)
                self.storage_valid.store_last(obs_v, last_val_v)
                self.storage_valid.compute_estimates(self.gamma, self.lmbda, self.use_gae, self.normalize_adv)

            # Optimize policy & valueq
            summary = self.optimize()
            # Log the training-procedure
            self.t += self.n_steps * self.n_envs
            rew_batch, done_batch = self.storage.fetch_log_data()
            if self.storage_valid is not None:
                rew_batch_v, done_batch_v = self.storage_valid.fetch_log_data()
            else:
                rew_batch_v = done_batch_v = None
            self.logger.feed(rew_batch, done_batch, rew_batch_v, done_batch_v)
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

    def query_agents(self, obs, act):
        weak_act, _, _ = self.pi_w.predict(obs)
        oracle_act, _, _ = self.pi_o.predict(obs)
        act_exec = np.where(act == 0, weak_act, oracle_act)
        return act_exec

    def oracle_query_cost(self, rew, act):
        # multiply the reward with the oracle cost if the action is from the oracle, and if the reward is above zero.
        # rew is a tensor of shape (n_envs, ), so the cost should be multiplied by all the rewards in the batch.
        # Apply oracle cost to the rewards where action is from oracle
        adjusted_rew = np.where(act == 1, rew - self.oracle_cost_per_action, rew)
        return adjusted_rew

    def switching_query_cost(self, rew, switching_idx):
        # multiply the reward with the switching cost if the action is different from the previous action
        # rew is a tensor of shape (n_envs, ), so the cost should be multiplied by all the rewards in the batch.
        # Apply switching cost to the rewards where action is different from the previous action
        rew[switching_idx] -= self.switching_cost_per_action
        return rew


class PPOFrozen:
    def __init__(self, policy, device):
        super().__init__()
        self.policy = policy
        self.device = device

    def predict(self, obs):
        with torch.no_grad():
            obs = torch.FloatTensor(obs).to(device=self.device)
            dist, value = self.policy(obs)
            act = dist.sample()
            log_prob_act = dist.log_prob(act)

        return act.cpu().numpy(), log_prob_act.cpu().numpy(), value.cpu().numpy()


class CategoricalPolicy(nn.Module):
    def __init__(self, embedder, action_size):
        """
        embedder: (torch.Tensor) model to extract the embedding for observation
        action_size: number of the categorical actions
        """
        super(CategoricalPolicy, self).__init__()
        self.embedder = embedder
        # small scale weight-initialization in policy enhances the stability
        self.fc_policy = orthogonal_init(nn.Linear(self.embedder.output_dim, action_size), gain=0.01)
        self.fc_value = orthogonal_init(nn.Linear(self.embedder.output_dim, 1), gain=1.0)

    def forward(self, x):
        hidden = self.embedder(x)
        logits = self.fc_policy(hidden)
        log_probs = F.log_softmax(logits, dim=1)
        p = Categorical(logits=log_probs)
        v = self.fc_value(hidden).reshape(-1)
        return p, v

    def get_inner_values(self, x):
        hidden = self.embedder(x)
        logits = self.fc_policy(hidden)
        action_probabilities = F.softmax(logits, dim=1)
        return hidden, action_probabilities


class CategoricalPolicyT2(nn.Module):
    def __init__(self, embedder, action_size, additional_hidden_dim, additional_softmax_dim):
        """
        embedder: (torch.Tensor) model to extract the embedding for observation
        action_size: number of the categorical actions
        """
        super(CategoricalPolicyT2, self).__init__()
        self.embedder = embedder
        # Compute total input dimension for fc_policy considering additional tensors
        total_input_dim = self.embedder.output_dim + additional_hidden_dim + additional_softmax_dim
        # small scale weight-initialization in policy enhances the stability
        self.fc_policy = orthogonal_init(nn.Linear(total_input_dim, action_size), gain=0.01)
        self.fc_value = orthogonal_init(nn.Linear(total_input_dim, 1), gain=1.0)

    def forward(self, x, pi_w_hidden, pi_w_softmax):
        hidden = self.embedder(x)
        hidden = torch.cat([hidden, pi_w_hidden], dim=1)
        hidden = torch.cat([hidden, pi_w_softmax], dim=1)
        logits = self.fc_policy(hidden)
        log_probs = F.log_softmax(logits, dim=1)
        p = Categorical(logits=log_probs)
        v = self.fc_value(hidden).reshape(-1)
        return p, v


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
    def __init__(self,
                 in_channels):
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
                 **kwargs):
        super(ImpalaModel, self).__init__()
        scale = 1
        self.block1 = ImpalaBlock(in_channels=in_channels, out_channels=16 * scale)
        self.block2 = ImpalaBlock(in_channels=16 * scale, out_channels=32 * scale)
        self.block3 = ImpalaBlock(in_channels=32 * scale, out_channels=32 * scale)
        self.fc = nn.Linear(in_features=32 * scale * 8 * 8, out_features=256)

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


class Flatten(nn.Module):
    def forward(self, x):
        return torch.flatten(x, start_dim=1)


def xavier_uniform_init(module, gain=1.0):
    if isinstance(module, nn.Linear) or isinstance(module, nn.Conv2d):
        nn.init.xavier_uniform_(module.weight.data, gain)
        nn.init.constant_(module.bias.data, 0)
    return module


def orthogonal_init(module, gain=nn.init.calculate_gain('relu')):
    if isinstance(module, nn.Linear) or isinstance(module, nn.Conv2d):
        nn.init.orthogonal_(module.weight.data, gain)
        nn.init.constant_(module.bias.data, 0)
    return module
