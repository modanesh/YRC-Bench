import torch
import torch.optim as optim

from YRC.policies.base import BasePolicy
from YRC.models.rl import PPOModel
import YRC.models as models
from YRC.core.configs.global_configs import get_global_variable


class PPOPolicy(BasePolicy):

    def __init__(self, config, coord_env):
        self.model_cls = getattr(models, config.coord_policy.model_cls)
        self.model = self.model_cls(config, coord_env)
        self.model = PPOModel(self.model)
        self.model.to(get_global_variable("device"))
        self.optimizer = optim.Adam(
            self.model.parameters(), lr=config.coord_policy.lr, eps=1e-5
        )
        self.init_lr = config.coord_policy.lr

    def forward(self, obs):
        return self.model(obs)

    @torch.no_grad()
    def get_act_logprob_value(self, obs):
        p, v = self.forward(obs)
        # NOTE: action is sampled
        a = p.sample()
        lp = p.log_prob(a)
        return a.cpu().numpy(), lp.cpu().numpy(), v.cpu().numpy()

    def predict(self, obs):
        dist, _ = self.forward(obs)
        return dist

    def adjust_lr(self, timesteps, max_timesteps):
        lr = self.init_lr * (1 - (timesteps / max_timesteps))
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

