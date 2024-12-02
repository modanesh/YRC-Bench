import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical

from YRC.core.configs.global_configs import get_global_variable
from YRC.models.utils import orthogonal_init, ImpalaModel


class ProcgenModel(nn.Module):
    def __init__(self, env):
        super().__init__()

        self.device = get_global_variable("device")
        self.embedder = ImpalaModel(env.obs_shape)
        self.hidden_dim = self.embedder.output_dim
        self.fc_policy = orthogonal_init(nn.Linear(self.hidden_dim, env.action_space.n), gain=0.01)
        self.fc_value = orthogonal_init(nn.Linear(self.hidden_dim, 1), gain=1.0)
        self.logit_dim = env.action_space.n

    def forward(self, obs):
        hidden = self.get_hidden(obs)
        logit = self.fc_policy(hidden)
        log_prob = F.log_softmax(logit, dim=1)
        p = Categorical(logits=log_prob)
        v = self.fc_value(hidden).reshape(-1)
        return p, v

    def get_hidden(self, obs):
        if not torch.is_tensor(obs):
            obs = torch.FloatTensor(obs).to(device=self.device)
        hidden = self.embedder(obs)
        return hidden

    def get_logit(self, obs):
        hidden = self.get_hidden(obs)
        logit = self.fc_policy(hidden)
        return logit
