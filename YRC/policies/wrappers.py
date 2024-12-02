import torch
from torch.distributions.uniform import Uniform
from torch.distributions.categorical import Categorical

from YRC.core import Policy
from YRC.core.configs.global_configs import get_global_variable


class ExploreWrapper(Policy):
    def __init__(self, config, env, policy):
        self.policy = policy
        self.args = config.evaluation.simulation
        self.device = get_global_variable("device")
        self.batch_size = env.num_envs
        self.num_actions = env.num_actions
        self._set_parameter_distribution()

    @property
    def hidden_dim(self):
        return self.policy.hidden_dim

    def get_hidden(self, obs):
        return self.policy.get_hidden(obs)

    def forward(self, obs):
        return self.policy.forward(obs)

    def _set_parameter_distribution(self):
        args = self.args
        if args.dist == "uniform":
            min_val = (
                torch.ones((self.batch_size,)).to(self.device).float() * args.min_val
            )
            max_val = (
                torch.ones((self.batch_size,)).to(self.device).float() * args.max_val
            )
            self.param_dist = Uniform(min_val, max_val)
        else:
            raise NotImplementedError(f"Unrecognized distribution {args.dist}")

        self.params = self.param_dist.sample()

    def reset(self, should_reset):
        new_params = self.param_dist.sample()
        self.params[should_reset] = new_params[should_reset]

    def act(self, obs, greedy=False, mask=None):
        params = self.params if mask is None else self.params[mask]
        if self.args.type == "temp":
            # softmax sampling, params is temperature
            logit = self.forward(obs)
            dist = Categorical(logits=logit / params.unsqueeze(-1))
            action = dist.sample().cpu().numpy()
        elif self.args.type == "epsilon":
            # epsilon chance of acting randomly, params is epsilon
            action = self.policy.act(obs, greedy=greedy)
            rand_action = torch.randint_like(action, 0, self.num_actions)
            mask = torch.rand_like(action) < params
            action[mask] = rand_action[mask].cpu().numpy()
        else:
            raise NotImplementedError

        return action
