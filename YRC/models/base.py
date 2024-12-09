
import torch
import torch.nn as nn


from YRC.models.utils import orthogonal_init, ImpalaModel
from YRC.core.configs.global_configs import get_global_variable


class ImpalaCoordPolicyModel(nn.Module):
    def __init__(self, config, coord_env):
        super().__init__()

        self.device = get_global_variable("device")
        self.embedder = ImpalaModel(coord_env.base_env.obs_shape)

        self.feature_type = config.coord_policy.feature_type
        if self.feature_type == "obs":
            self.hidden_dim = self.embedder.output_dim
        elif self.feature_type == "hidden":
            self.hidden_dim = coord_env.weak_agent.hidden_dim
        elif self.feature_type == "hidden_obs":
            self.hidden_dim = self.embedder.output_dim + coord_env.weak_agent.hidden_dim
        elif self.feature_type == "dist":
            self.hidden_dim = coord_env.base_env.action_space.n
        elif self.feature_type == "hidden_dist":
            self.hidden_dim = coord_env.weak_agent.hidden_dim + coord_env.base_env.action_space.n
        else:
            raise NotImplementedError

        self.fc_policy = orthogonal_init(
            nn.Linear(self.hidden_dim, coord_env.action_space.n), gain=0.01
        )

    def forward(self, obs, ret_hidden=False):
        env_obs = obs["env_obs"]
        if not torch.is_tensor(env_obs):
            env_obs = torch.from_numpy(env_obs).float().to(self.device)
        weak_features = obs["weak_features"]
        if not torch.is_tensor(weak_features):
            weak_features = torch.from_numpy(weak_features).float().to(self.device)
        weak_logit = obs["weak_logit"]
        if not torch.is_tensor(weak_logit):
            weak_logit = torch.from_numpy(weak_logit).float().to(self.device)

        if self.feature_type == "obs":
            hidden = self.embedder(env_obs)
        elif self.feature_type == "hidden":
            hidden = weak_features
        elif self.feature_type == "hidden_obs":
            hidden = torch.cat([self.embedder(env_obs), weak_features], dim=-1)
        elif self.feature_type == "dist":
            hidden = weak_logit.softmax(dim=-1)
        elif self.feature_type == "hidden_dist":
            hidden = torch.cat([weak_features, weak_logit.softmax(dim=-1)], dim=-1)
        else:
            raise NotImplementedError

        logit = self.fc_policy(hidden)

        if ret_hidden:
            return logit, hidden

        return logit


class ImpalaPolicyModel(nn.Module):

    def __init__(self, config, env):
        super().__init__()
        self.device = get_global_variable("device")
        self.embedder = ImpalaModel(env.obs_shape)
        self.hidden_dim = self.embedder.output_dim
        self.fc_policy = orthogonal_init(nn.Linear(self.hidden_dim, env.action_space.n), gain=0.01)

    def forward(self, obs, ret_hidden=False):
        hidden = self.get_hidden(obs)
        logit = self.fc_policy(hidden)
        if ret_hidden:
            return logit, hidden
        return logit

    def get_hidden(self, obs):
        if not torch.is_tensor(obs):
            obs = torch.FloatTensor(obs).to(device=self.device)
        hidden = self.embedder(obs)
        return hidden
