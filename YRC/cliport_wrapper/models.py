import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical


class Agent(nn.Module):
    def __init__(self, embedder, action_size):
        """
        embedder: (torch.Tensor) model to extract the embedding for observation
        action_size: number of the categorical actions
        """
        super(Agent, self).__init__()
        self.embedder = embedder
        # small scale weight-initialization in policy enhances the stability
        self.fc_policy = orthogonal_init(nn.Linear(self.embedder.output_dim, action_size), gain=0.01)
        self.fc_value = orthogonal_init(nn.Linear(self.embedder.output_dim, 1), gain=1.0)

    def forward(self, x, action=None):
        hidden = self.embedder(x, )
        logits = self.fc_policy(hidden)
        log_probs = F.log_softmax(logits, dim=1)
        p = Categorical(logits=log_probs)
        v = self.fc_value(hidden).reshape(-1)
        if action is None:
            action = p.sample()
        return action, p.log_prob(action), p.entropy(), v


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

