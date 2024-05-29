import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import numpy as np
import robomimic.utils.tensor_utils as TensorUtils


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class PPOHead(nn.Module):
    def __init__(
        self,
        input_size,
        output_size,
        hidden_size=1024,
        num_layers=2,
        activation="tanh",
        loss_coef=1.0,
    ):
        super().__init__()
        self.output_size = output_size
        self.loss_coef = loss_coef

        if num_layers > 0:
            sizes = [input_size] + [hidden_size] * num_layers
            layers = []
            for i in range(num_layers):
                layers += [nn.Linear(sizes[i], sizes[i + 1]), nn.ReLU()]
            self.share = nn.Sequential(*layers)
        else:
            self.share = nn.Identity()

        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(hidden_size, hidden_size)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_size, hidden_size)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_size, output_size), std=0.01),
        )
        self.actor_logstd = nn.Parameter(torch.zeros(1, output_size))

        self.critic = nn.Sequential(
            layer_init(nn.Linear(hidden_size, hidden_size)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_size, hidden_size)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_size, 1), std=1.0),
        )

        if activation == "tanh":
            self.actv = torch.tanh
        elif activation == "relu":
            self.actv = torch.relu
        else:
            raise NotImplementedError(f"Activation '{activation}' is not implemented.")

    def forward_fn(self, x):
        share = self.share(x)
        action_mean = self.actor_mean(share)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        value = self.critic(share)
        return action_mean, action_std, value

    def forward(self, x):
        if x.ndim == 3:
            action_mean, action_std, value = TensorUtils.time_distributed(x, self.forward_fn)
        elif x.ndim < 3:
            action_mean, action_std, value = self.forward_fn(x)
        return action_mean, action_std, value

    def get_value(self, x):
        _, _, value = self.forward_fn(x)
        return value

    def get_action_and_value(self, x, action=None):
        action_mean, action_std, value = self.forward_fn(x)
        dist = Normal(action_mean, action_std)
        if action is None:
            action = dist.sample()
        return action, dist.log_prob(action).sum(-1), dist.entropy().sum(-1), value

    def loss_fn(self, action_dist, value, actions, returns, advantages, old_log_probs, clip_param=0.2, value_loss_coef=0.5, entropy_coef=0.01):
        log_probs = action_dist.log_prob(actions).sum(-1)
        ratios = torch.exp(log_probs - old_log_probs)
        surr1 = ratios * advantages
        surr2 = torch.clamp(ratios, 1.0 - clip_param, 1.0 + clip_param) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()
        value_loss = F.mse_loss(value, returns)
        entropy = action_dist.entropy().sum(-1).mean()
        loss = policy_loss + value_loss_coef * value_loss - entropy_coef * entropy
        return loss * self.loss_coef