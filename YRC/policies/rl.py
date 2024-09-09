import torch
import torch.optim as optim

from YRC.policies.base import BasePolicy
from YRC.models.rl import PPOModel
import YRC.models as models
from YRC.core.configs.global_configs import get_global_variable


class PPOPolicy(BasePolicy):

    def __init__(self, config, env):
        self.model_cls = getattr(models, config.coord_policy.model_cls)
        self.model = self.model_cls(config, env)
        self.model = PPOModel(self.model)
        self.model.to(get_global_variable("device"))
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-4, eps=1e-5)

    def forward(self, obs):
        return self.model(obs)

    def get_action_and_value(self, obs, action=None):
        dist, value = self.forward(obs)
        if action is None:
            action = dist.sample()
        return action, dist.log_prob(action), dist.entropy(), value

    def predict(self, obs):
        dist, _ = self.forward(obs)
        return dist

    def get_value(self, obs):
        _, value = self.forward(obs)
        return value

    def set_learning_rate(self, learning_rate):
        self.optimizer.param_groups[0]["lr"] = learning_rate


        """
        lr = self.init_lr * (1 - (timesteps / max_timesteps))
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr
        """
