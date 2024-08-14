import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions.categorical import Categorical

from cliport.utils import utils as cliport_utils
from .configs.global_configs import get_global_variable


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
        super().__init__()
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
    def __init__(self, in_channels, benchmark, scale=1):
        super().__init__()
        self.block1 = ImpalaBlock(in_channels=in_channels, out_channels=16 * scale)
        self.block2 = ImpalaBlock(in_channels=16 * scale, out_channels=32 * scale)
        self.block3 = ImpalaBlock(in_channels=32 * scale, out_channels=32 * scale)
        if benchmark == 'procgen':
            fc_input = 32 * scale * 8 * 8
        elif benchmark == 'cliport':
            fc_input = 32 * scale * 20 * 40

        self.fc = nn.Linear(fc_input, 256)
        self.output_dim = 256
        self.apply(xavier_uniform_init)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = nn.ReLU()(x)
        x = torch.flatten(x, start_dim=1)
        x = self.fc(x)
        x = nn.ReLU()(x)
        return x


def orthogonal_init(module, gain=nn.init.calculate_gain('relu')):
    if isinstance(module, (nn.Linear, nn.Conv2d)):
        nn.init.orthogonal_(module.weight.data, gain)
        nn.init.constant_(module.bias.data, 0)
    return module


def xavier_uniform_init(module, gain=1.0):
    if isinstance(module, (nn.Linear, nn.Conv2d)):
        nn.init.xavier_uniform_(module.weight.data, gain)
        nn.init.constant_(module.bias.data, 0)
    return module


class PPOFrozen:
    def __init__(self, policy, device):
        self.policy = policy
        self.device = device

    @torch.no_grad()
    def predict(self, obs):
        obs = torch.FloatTensor(obs).to(device=self.device)
        dist, value = self.policy(obs, pi_w_hidden=None)
        act = dist.sample()
        log_prob_act = dist.log_prob(act)
        return act.cpu().numpy(), log_prob_act.cpu().numpy(), value.cpu().numpy()


class CategoricalPolicy(nn.Module):
    def __init__(self, config, in_channels=None, action_size=2, additional_hidden_dim=0, policy_type="T1"):
        super().__init__()
        self.architecture = config.architecture
        self.eps_clip = config.eps_clip
        self.type = policy_type
        if self.architecture == 'impala':
            self.embedder = ImpalaModel(in_channels, benchmark=get_global_variable("benchmark"))
            total_input_dim = self.embedder.output_dim + additional_hidden_dim
        else:
            self.embedder = None
            total_input_dim = additional_hidden_dim

        self.fc_policy = orthogonal_init(nn.Linear(total_input_dim, action_size), gain=0.01)
        self.fc_value = orthogonal_init(nn.Linear(total_input_dim, 1), gain=1.0)

    def forward(self, x, pi_w_hidden):
        if self.type == "T3":
            hidden = pi_w_hidden
        else:
            hidden = self.embedder(x)
            if self.type == "T2":
                hidden = torch.cat([hidden, pi_w_hidden], dim=1)

        logits = self.fc_policy(hidden)
        log_probs = F.log_softmax(logits, dim=1)
        p = Categorical(logits=log_probs)
        v = self.fc_value(hidden).squeeze(-1)
        return p, v

    def extract_features(self, x):
        return self.embedder(x) if self.type != "T3" else None

    def compute_losses(self, dist, value, act, old_log_prob_act, old_value, return_batch, adv_batch):
        log_prob_act = dist.log_prob(act)
        ratio = torch.exp(log_prob_act - old_log_prob_act)
        surr1 = ratio * adv_batch
        surr2 = torch.clamp(ratio, 1.0 - self.eps_clip, 1.0 + self.eps_clip) * adv_batch
        pi_loss = -torch.min(surr1, surr2).mean()

        clipped_value = old_value + (value - old_value).clamp(-self.eps_clip, self.eps_clip)
        v_surr1 = (value - return_batch).pow(2)
        v_surr2 = (clipped_value - return_batch).pow(2)
        value_loss = 0.5 * torch.max(v_surr1, v_surr2).mean()

        entropy_loss = dist.entropy().mean()
        return pi_loss, value_loss, entropy_loss

    def save_model(self, save_path):
        print(":::::::::::::SAVING MODEL:::::::::::::")
        torch.save({
            'model_state_dict': self.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict()
        }, save_path)

    @torch.no_grad()
    def predict(self, obs, pi_w_hidden):
        dist, value = self(obs, pi_w_hidden)
        act = dist.sample()
        log_prob_act = dist.log_prob(act)
        return act.cpu().numpy(), log_prob_act.cpu().numpy(), value.cpu().numpy()

    @torch.no_grad()
    def act(self, obs, pi_w_hidden):
        dist, value = self(obs, pi_w_hidden)
        act = dist.sample()
        return act.cpu().numpy()


class ProcgenPPO(CategoricalPolicy):
    def __init__(self, config, in_channels, action_size=2, additional_hidden_dim=0, policy_type="T1"):
        super().__init__(config, in_channels, action_size, additional_hidden_dim, policy_type)
        self.learning_rate = config.learning_rate
        self.optimizer = optim.Adam(self.parameters(), lr=self.learning_rate, eps=1e-5)

    def predict(self, obs, pi_w_hidden):
        obs = torch.FloatTensor(obs).to(device=get_global_variable("device"))
        return super().predict(obs, pi_w_hidden)

    def act(self, obs, pi_w_hidden):
        obs = torch.FloatTensor(obs).to(device=get_global_variable("device"))
        return super().act(obs, pi_w_hidden)


class CliportPPO(CategoricalPolicy):
    def __init__(self, config, in_channels, action_size=2, additional_hidden_dim=0, policy_type="T1"):
        super().__init__(config, in_channels, action_size, additional_hidden_dim, policy_type)
        self.learning_rate = config.learning_rate
        self.optimizer = optim.Adam(self.parameters(), lr=self.learning_rate, eps=1e-5)

    def predict(self, obs, pi_w_hidden):
        obs = cliport_utils.get_image(obs)
        obs = torch.FloatTensor(obs).to(device=get_global_variable("device"))
        obs = obs.permute(2, 0, 1).unsqueeze(0)
        return super().predict(obs, pi_w_hidden)

    def act(self, obs, pi_w_hidden):
        obs = cliport_utils.get_image(obs)
        obs = torch.FloatTensor(obs).to(device=get_global_variable("device"))
        obs = obs.permute(2, 0, 1).unsqueeze(0)
        return super().act(obs, pi_w_hidden)


class RainbowDQN(nn.Module):
    def __init__(self, config, in_channels=None, action_size=2, additional_hidden_dim=0, policy_type="T1"):
        super().__init__()
        self.architecture = config.architecture
        self.action_size = action_size
        self.type = policy_type
        self.v_min = config.v_min
        self.v_max = config.v_max
        self.num_atoms = config.num_atoms
        self.support = torch.linspace(self.v_min, self.v_max, self.num_atoms).to(get_global_variable("device"))

        if self.architecture == 'impala':
            self.embedder = ImpalaModel(in_channels, benchmark=get_global_variable("benchmark"))
            total_input_dim = self.embedder.output_dim + additional_hidden_dim
        else:
            self.embedder = None
            total_input_dim = additional_hidden_dim

        self.advantage_stream = orthogonal_init(nn.Linear(total_input_dim, action_size * self.num_atoms), gain=0.01)
        self.value_stream = orthogonal_init(nn.Linear(total_input_dim, self.num_atoms), gain=1.0)

        self.target_network = copy.deepcopy(self)
        self.learning_rate = config.learning_rate
        self.optimizer = optim.Adam(self.parameters(), lr=self.learning_rate, eps=1e-5)

    def forward(self, x, pi_w_hidden):
        if self.type == "T3":
            hidden = pi_w_hidden
        else:
            hidden = self.embedder(x)
            if self.type == "T2":
                hidden = torch.cat([hidden, pi_w_hidden], dim=1)

        advantage = self.advantage_stream(hidden).view(-1, self.action_size, self.num_atoms)
        value = self.value_stream(hidden).view(-1, 1, self.num_atoms)
        q_dist = value + advantage - advantage.mean(dim=1, keepdim=True)
        return F.softmax(q_dist, dim=-1)

    def compute_loss(self, current_q, target_q):
        return -(target_q * current_q.log()).sum(1).mean()

    def update_target_network(self):
        self.target_network.load_state_dict(self.state_dict())

    @torch.no_grad()
    def predict(self, obs, pi_w_hidden):
        q_dist = self(obs, pi_w_hidden)
        q_values = (q_dist * self.support).sum(dim=-1)
        action = q_values.argmax(dim=-1)
        return action.cpu().numpy(), q_values.cpu().numpy()

    @torch.no_grad()
    def act(self, obs, pi_w_hidden):
        q_dist = self(obs, pi_w_hidden)
        q_values = (q_dist * self.support).sum(dim=-1)
        action = q_values.argmax(dim=-1)
        return action.cpu().numpy()

    def save_model(self, save_path):
        print(":::::::::::::SAVING MODEL:::::::::::::")
        torch.save({
            'model_state_dict': self.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict()
        }, save_path)


class ProcgenDQN(RainbowDQN):
    def predict(self, obs, pi_w_hidden):
        obs = torch.Tensor(obs).to(device=get_global_variable("device"))
        return super().predict(obs, pi_w_hidden)

    def act(self, obs, pi_w_hidden):
        obs = torch.Tensor(obs).to(device=get_global_variable("device"))
        return super().act(obs, pi_w_hidden)


class CliportDQN(RainbowDQN):
    def predict(self, obs, pi_w_hidden):
        obs = cliport_utils.get_image(obs)
        obs = torch.Tensor(obs).to(device=get_global_variable("device"))
        obs = obs.permute(2, 0, 1).unsqueeze(0)
        return super().predict(obs, pi_w_hidden)

    def act(self, obs, pi_w_hidden):
        obs = cliport_utils.get_image(obs)
        obs = torch.Tensor(obs).to(device=get_global_variable("device"))
        obs = obs.permute(2, 0, 1).unsqueeze(0)
        return super().act(obs, pi_w_hidden)
