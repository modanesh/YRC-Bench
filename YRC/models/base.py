import torch
import torch.nn as nn

from cliport.utils import utils as cliport_utils

from YRC.models.utils import orthogonal_init, ImpalaModel
from YRC.core.configs.global_configs import get_global_variable


class BaseCoordModel(nn.Module):

    def __init__(self, config, coord_env):
        super().__init__()

        self.device = get_global_variable("device")
        self.embedder = ImpalaModel(coord_env.base_env.obs_shape)

        self.feature_type = config.coord_policy.feature_type
        if self.feature_type == "T1":
            self.hidden_dim = self.embedder.output_dim
        elif self.feature_type == "T2":
            self.hidden_dim = weak_agent.hidden_size
        elif self.feature_type == "T3":
            self.hidden_dim = self.embedder.output_dim + coord_env.weak_agent.hidden_size
        else:
            raise NotImplementedError

        self.fc_policy = orthogonal_init(
            nn.Linear(self.hidden_dim, coord_env.action_size), gain=0.01
        )

    def forward(self, obs, ret_hidden=False):

        env_obs = obs["env_obs"]
        if not torch.is_tensor(env_obs):
            env_obs = torch.FloatTensor(env_obs).to(self.device)
        weak_features = obs["weak_features"]

        if self.feature_type == "T1":
            hidden = self.embedder(env_obs)
        elif self.feature_type == "T2":
            hidden = obs["weak_features"]
        elif self.feature_type == "T3":
            hidden = torch.cat([self.embedder(env_obs), weak_features], dim=-1)
        else:
            raise NotImplementedError

        logit = self.fc_policy(hidden)

        if ret_hidden:
            return logit, hidden

        return logit


