from YRC.core.policy import Policy
import torch
from YRC.core.configs import get_global_variable


class MinigridPolicy(Policy):
    def __init__(self, model, num_envs):
        self.model = model
        self.memory = torch.zeros(num_envs, self.model.memory_size, device=get_global_variable("device"))

    def forward(self, obs):
        obs = self.model.preprocess_obs(obs)
        return self.model.get_logit(obs)

    def predict(self, obs):
        obs = self.model.preprocess_obs(obs)
        dist, value, self.memory = self.model(obs, self.memory)
        return dist

    def act(self, obs, greedy=False):
        dist = self.predict(obs)
        if greedy:
            action = dist.probs.argmax(dim=-1)
        else:
            action = dist.sample()
        return action.cpu().numpy()

    def get_hidden(self, obs):
        obs = self.model.preprocess_obs(obs)
        return self.model.get_hidden(obs)

    @property
    def hidden_dim(self):
        return self.model.hidden_dim
