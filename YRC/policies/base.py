import numpy as np

import torch
from torch.distributions.categorical import Categorical
import torch.nn.functional as F
import torch.optim as optim

import os

import importlib
from YRC.core.policy import Policy
import YRC.models as models
from YRC.core.configs.global_configs import get_global_variable


class BasePolicy(Policy):

    def __init__(self, config, coord_env):
        self.model_cls = getattr(models, config.coord_policy.model_cls)
        self.model = self.model_cls(config, coord_env)
        self.model.to(get_global_variable("device"))

        self.optimizer = optim.Adam(
            self.model.parameters(), lr=config.coord_policy.lr, eps=1e-5
        )

    def train(self):
        self.model.train()

    def eval(self):
        self.model.eval()

    def predict(self, obs):
        logit = self.model(obs)
        log_prob = F.log_softmax(logit, dim=-1)
        return Categorical(logits=log_prob)

    def act(self, obs):
        a = self.predict(obs).probs.argmax(dim=-1)
        return a.cpu().numpy()

    def update_params(self, grad_clip_norm=None):
        if grad_clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), grad_clip_norm)
        self.optimizer.step()
        self.optimizer.zero_grad()

    def save_model(self, name, save_dir):
        save_path = os.path.join(save_dir, f"{name}.ckpt")
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
            },
            save_path,
        )

    def load_model(self, name, load_dir):
        load_path = os.path.join(save_dir, f"{name}.ckpt")
        ckpt = torch.load(load_path)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])



class AlwaysPolicy(Policy):

    WEAK = 0
    STRONG = 1

    def __init__(self, config, coord_env):
        if config.coord_policy.which == "weak":
            self.choice = self.WEAK
        elif config.coord_policy.which == "strong":
            self.choice = self.STRONG
        else:
            raise NotImplementedError

    def act(self, obs):
        return np.ones((obs["env_obs"].shape[0],), dtype=np.int64) * self.choice

