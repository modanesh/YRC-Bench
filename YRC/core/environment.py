import importlib
import logging

if importlib.util.find_spec("gymnasium") is None:
    import gym
else:
    import gymnasium as gym  # used for minigrid
import numpy as np
import pprint
import json

from copy import deepcopy as dc

from YRC.core import Evaluator
from YRC.core.configs import get_global_variable


def make(config):
    base_envs = make_raw_envs(config)
    sim_weak_agent, weak_agent, strong_agent = load_agents(config, base_envs["val_sim"])

    coord_envs = {}
    for name in base_envs:
        if config.general.skyline or name not in ["train", "val_sim"]:
            if type(strong_agent) is dict:
                coord_envs[name] = CoordEnv(config.coord_env, base_envs[name], weak_agent, strong_agent[name])
            else:
                coord_envs[name] = CoordEnv(config.coord_env, base_envs[name], weak_agent, strong_agent)
        else:
            # NOTE: not skyline and name in ["train", "val_sim"]
            # use weak agent as strong agent
            # use sim_weak agent as weak agent
            coord_envs[name] = CoordEnv(config.coord_env, base_envs[name], sim_weak_agent, weak_agent)

    # set costs for getting help from strong agent
    test_eval_info = get_test_eval_info(config, coord_envs)
    for name in coord_envs:
        coord_envs[name].set_costs(test_eval_info)

    # reset
    for name in coord_envs:
        coord_envs[name].reset()

    logging.info(
        f"Strong query cost per action: {coord_envs['train'].strong_query_cost_per_action}"
    )
    logging.info(
        f"Switch agent cost per action: {coord_envs['train'].switch_agent_cost_per_action}"
    )

    check_coord_envs(coord_envs)

    return coord_envs


def check_coord_envs(envs):
    for name in envs:
        assert (
                envs[name].strong_query_cost_per_action
                == envs["train"].strong_query_cost_per_action
        )
        assert (
                envs[name].switch_agent_cost_per_action
                == envs["train"].switch_agent_cost_per_action
        )


def get_test_eval_info(config, coord_envs):
    with open("YRC/core/test_eval_info.json") as f:
        data = json.load(f)

    backup_data = dc(data)

    benchmark = config.general.benchmark
    env_name = config.environment.common.env_name

    if env_name not in data[benchmark]:
        logging.info(f"Missing info about {benchmark}-{env_name}!")
        logging.info("Calculating missing info (taking a few minutes)...")
        evaluator = Evaluator(config.evaluation)
        # eval strong agent on test environment to get statistics
        summary = evaluator.eval(
            coord_envs["test"].strong_agent,
            {"test": coord_envs["test"].base_env},
            ["test"],
            num_episodes=coord_envs["test"].num_envs,
        )["test"]
        data[benchmark][env_name] = summary

        with open("YRC/core/backup_test_eval_info.json", "w") as f:
            json.dump(backup_data, f, indent=2)
        with open("YRC/core/test_eval_info.json", "w") as f:
            json.dump(data, f, indent=2)
        logging.info("Saved info!")

    ret = data[benchmark][env_name]

    logging.info(f"{pprint.pformat(ret, indent=2)}")
    return ret


def make_raw_envs(config):
    module = importlib.import_module(f"YRC.envs.{get_global_variable('benchmark')}")
    create_fn = getattr(module, "create_env")

    envs = {}
    for name in ["train", "val_sim", "val_true", "test"]:
        if name == "train" and config.general.skyline:
            env = create_fn("test", config.environment)
        else:
            env = create_fn(name, config.environment)
        # some extra information
        env.name = config.environment.common.env_name
        envs[name] = env

    return envs


def load_agents(config, env):
    module = importlib.import_module(f"YRC.envs.{get_global_variable('benchmark')}")
    load_fn = getattr(module, "load_policy")

    sim_weak_agent = load_fn(config.agents.sim_weak, env)
    weak_agent = load_fn(config.agents.weak, env)
    strong_agent = load_fn(config.agents.strong, env)

    return sim_weak_agent, weak_agent, strong_agent


class CoordEnv(gym.Env):
    WEAK = 0
    STRONG = 1

    def __init__(self, config, base_env, weak_agent, strong_agent):
        self.args = config
        self.base_env = base_env
        if isinstance(base_env.observation_space, list):
            obs_space = base_env.observation_space[0]
        elif isinstance(base_env.observation_space, gym.spaces.Dict):
            obs_space = base_env.observation_space.spaces['image']
        else:
            obs_space = base_env.observation_space
        self.weak_agent = weak_agent
        self.strong_agent = strong_agent

        self.action_space = gym.spaces.Discrete(2)
        self.observation_space = gym.spaces.Dict(
            {
                "env_obs": obs_space,
                "weak_features": gym.spaces.Box(
                    -100, 100, shape=(weak_agent.hidden_dim,)
                ),
                "weak_logit": gym.spaces.Box(-100, 100, shape=(weak_agent.model.logit_dim,)),
            }
        )

    def set_costs(self, test_eval_info):
        length = test_eval_info["episode_length_mean"]
        reward = test_eval_info["reward_mean"]
        reward_per_action = reward / length

        self.strong_query_cost_per_action = round(
            reward_per_action * self.args.strong_query_cost_ratio, 2
        )
        self.switch_agent_cost_per_action = round(
            reward_per_action * self.args.switch_agent_cost_ratio, 2
        )

    @property
    def num_envs(self):
        return self.base_env.num_envs

    @property
    def num_actions(self):
        return self.action_space.n

    @property
    def action_shape(self):
        return self.action_space.shape

    @property
    def obs_shape(self):
        return {
            "env_obs": self.base_env.obs_shape,
            "weak_features": (self.weak_agent.hidden_dim,),
            "weak_logit": (self.base_env.action_space.n,),
        }

    def reset(self):
        self.prev_action = None
        self.env_obs = self.base_env.reset()
        self._reset_agents(np.array([True] * self.num_envs))
        return self.get_obs()

    def _reset_agents(self, done):
        self.weak_agent.reset(done)
        self.strong_agent.reset(done)

    def step(self, action):
        env_action = self._compute_env_action(action)
        self.env_obs, env_reward, done, env_info = self.base_env.step(env_action)

        info = dc(env_info)
        if len(info) == 0:
            info = [{"env_reward": 0, "env_action": 0}] * self.num_envs
        for i, item in enumerate(info):
            if "env_reward" not in item:
                item["env_reward"] = env_reward[i]
            item["env_action"] = env_action[i]

        reward = self._get_reward(env_reward, action, done)
        self._reset_agents(done)
        self.prev_action = action

        return self.get_obs(), reward, done, info

    def _compute_env_action(self, action):
        # NOTE: this method only works with non-recurrent agent models
        greedy = self.args.act_greedy
        is_weak = (action == self.WEAK)
        is_strong = ~is_weak

        if isinstance(self.env_obs, dict):
            if is_weak.any():
                env_action = self.weak_agent.act(self.env_obs, greedy=greedy)
            if is_strong.any():
                if get_global_variable('benchmark') == 'cliport':
                    env_action = self.strong_agent.act(self.env_obs, self.base_env, greedy=greedy)
                else:
                    env_action = self.strong_agent.act(self.env_obs, greedy=greedy)
        else:
            env_action = np.zeros_like(action)
            if is_weak.any():
                env_action[is_weak] = self.weak_agent.act(self.env_obs[is_weak], greedy=greedy)
            if is_strong.any():
                env_action[is_strong] = self.strong_agent.act(self.env_obs[is_strong], greedy=greedy)
        return env_action

    def get_obs(self):
        obs = {
            "env_obs": self.env_obs,
            "weak_features": self.weak_agent.get_hidden(self.env_obs).detach().cpu().numpy(),
            "weak_logit": self.weak_agent.forward(self.env_obs).detach().cpu().numpy(),
        }
        return obs

    def _get_reward(self, env_reward, action, done):
        # cost of querying strong agent
        reward = np.where(
            action == self.STRONG,
            env_reward - self.strong_query_cost_per_action,
            env_reward,
        )

        # cost of switching
        if self.prev_action is not None:
            switch_indices = ((action != self.prev_action) & (~done)).nonzero()[0]
            if switch_indices.size > 1:
                reward[switch_indices] -= self.switch_agent_cost_per_action

        return reward

    def close(self):
        return self.base_env.close()
