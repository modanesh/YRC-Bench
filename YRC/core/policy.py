import logging
import importlib


def make(config, env):
    coord_policy_cls = getattr(
        importlib.import_module("YRC.policies"), config.coord_policy.cls
    )
    coord_policy = coord_policy_cls(config, env)
    return coord_policy


class Policy:
    # get logit
    def forward(self, obs):
        pass

    # get action distribution
    def predict(self, obs):
        pass

    # draw an action
    def act(self, obs, greedy=False):
        pass

    # update model parameters
    def update_params(self):
        pass

    # get pre-softmax hidden features
    def get_hidden(self):
        pass

    # set to training mode
    def train(self):
        pass

    # set to eval mode
    def eval(self):
        pass

    # initialization at the beginning of an episode
    def reset(self, should_reset):
        pass

    def save_model(self, name, save_dir):
        pass

    def load_model(self, load_path):
        pass
