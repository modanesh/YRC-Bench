import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical

from YRC.models.utils import orthogonal_init


class PPOModel(nn.Module):

    def __init__(self, model):
        super().__init__()
        self.model = model
        self.fc_value = orthogonal_init(nn.Linear(self.model.hidden_dim, 1), gain=1.0)

    def forward(self, obs):
        logit, hidden = self.model(obs, ret_hidden=True)
        log_prob = F.log_softmax(logit, dim=1)
        p = Categorical(logits=log_prob)
        v = self.fc_value(hidden).reshape(-1)
        return p, v
