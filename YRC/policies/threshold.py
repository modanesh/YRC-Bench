import os
import numpy as np
from copy import deepcopy as dc

import torch

from torch.distributions.categorical import Categorical
from YRC.core import Policy


class ThresholdPolicy(Policy):
    def __init__(self, config, env):
        self.args = config.coord_policy
        self.agent = env.weak_agent
        self.params = {"threshold": 0.0, "explore_temp": 1.0, "score_temp": 1.0}

    def act(self, obs, greedy=False):
        score = self._compute_score(obs["weak_logit"])
        # NOTE: higher score = more certain
        action = (score < self.params["threshold"]).int()
        return action.cpu().numpy()

    def generate_scores(self, env, num_rollouts):
        assert num_rollouts % env.num_envs == 0
        scores = []
        for i in range(num_rollouts // env.num_envs):
            scores.extend(self._rollout_once(env))
        return scores

    def _rollout_once(self, env):
        def sample_action(logit):
            dist = Categorical(logits=logit / self.params["explore_temp"])
            return dist.sample().cpu().numpy()

        agent = self.agent
        agent.eval()
        obs = env.reset()
        has_done = np.array([False] * env.num_envs)
        scores = []

        while not has_done.all():
            logit = agent.forward(obs["env_obs"])
            score = self._compute_score(logit)

            for i in range(env.num_envs):
                if not has_done[i]:
                    scores.append(score[i].item())

            action = sample_action(logit)
            obs, reward, done, info = env.step(action)
            has_done |= done

        return scores

    def _compute_score(self, logit):
        # NOTE: higher score = more certain
        metric = self.args.metric
        logit = logit / self.params["score_temp"]
        if metric == "max_logit":
            score = logit.max(dim=-1)[0]
        elif metric == "max_prob":
            score = logit.softmax(dim=-1).max(dim=-1)[0]
        elif metric == "margin":
            top2 = logit.softmax(dim=-1).topk(2, dim=-1)[0]
            score = top2[:, 0] - top2[:, 1]
        elif metric == "neg_entropy":
            score = -Categorical(logits=logit).entropy()
        elif metric == "neg_energy":
            score = logit.logsumexp(dim=-1)
        else:
            raise NotImplementedError(f"Unrecognized metric: {metric}")

        return score

    def update_params(self, params):
        self.params = dc(params)

    def save_model(self, name, save_dir):
        save_path = os.path.join(save_dir, f"{name}.ckpt")
        torch.save(self.params, save_path)
        logging.info(f"Saved model to {save_path}")

    def load_model(self, load_path):
        self.params = torch.load(load_path)
