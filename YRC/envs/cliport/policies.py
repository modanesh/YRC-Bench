import numpy as np

from YRC.core.policy import Policy


class CliportPolicy(Policy):
    def __init__(self, model):
        self.model = model

    def forward(self, obs):
        return self.model.get_logit(obs)

    def predict(self, obs):
        dist, value = self.model(obs)
        return dist

    def act(self, obs, greedy=False):
        img = obs['img']
        info = obs['info']
        action, _, _ = self.model.act(img, info)
        if action is None:
            np_action = np.array([None] * 14)[np.newaxis, :]
        else:
            np_action = self.flatten(action)
        return np_action

    def get_hidden(self, obs):
        return self.model.get_hidden(obs)

    @property
    def hidden_dim(self):
        return self.model.hidden_dim

    @staticmethod
    def flatten(action):
        arrays = []
        for key in action:
            value = action[key]
            if isinstance(value, tuple):
                arrays.extend(value)
            else:
                arrays.append(np.array(value))
        arrays = np.concatenate(arrays)[np.newaxis, :]
        return arrays


class CliportPolicyOracle(Policy):
    def __init__(self, model):
        self.model = model

    def act(self, obs, greedy=False):
        img = obs['img']
        info = obs['info']
        action = self.model(img, info)
        if action is None:
            np_action = np.array([None] * 14)[np.newaxis, :]
        else:
            np_action = np.concatenate([array for key in action for array in action[key]])[np.newaxis, :]
        print(f"goal is: {info['lang_goal']}")
        return np_action
