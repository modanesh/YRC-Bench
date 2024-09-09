import inspect
import importlib
import logging
import gym
import numpy as np
import pprint
import json

from copy import deepcopy as dc

from YRC.core import Evaluator
from YRC.policies.wrappers import ExploreWrapper
from YRC.core.configs import get_global_variable
from cliport import tasks, agents
from cliport.environments import environment as cliport_environment

# from . import procgen_wrappers


"""
def make_help_envs(config):
    benchmark = get_global_variable("benchmark")
    base_envs = make_raw_envs(benchmark, config.environments)
    obs_shape, n_actions = get_env_specs(benchmark, base_envs)

    weak_agent, strong_agent = load_agents(benchmark, config.acting_policy, obs_shape, n_actions)

    envs = {}
    env_set = ["val_id", "val_ood", "test"] if config.general.offline else ["train", "val_id", "val_ood", "test"]
    if config.general.offline:
        envs["train"] = load_dataset(config.offline)

    for name in env_set:
        current_env = base_envs[name]
        if benchmark == 'cliport':
            strong_agent = current_env.task.oracle(current_env)[0]
            config.help_env.timeout = current_env.task.max_steps
        envs[name] = HelpEnvironment(config.help_env, current_env, weak_agent, strong_agent)
    return tuple(envs.values()) if not config.general.offline else (tuple(envs.values()), weak_agent, strong_agent)

def get_env_specs(benchmark, base_envs):
    if benchmark == "procgen":
        return (
            base_envs["train"].observation_space.shape,
            base_envs["train"].action_space.n,
        )
    elif benchmark == "cliport":
        return (6, 320, 160), 2
    else:
        raise ValueError(f"Unknown benchmark: {benchmark}")

"""


def make(config):
    base_envs = make_raw_envs(config)
    weak_agent, strong_agent = load_agents(config, base_envs["train"])

    coord_envs = {}
    for name in base_envs:
        # simulated model selection
        if name == "val_id" and config.evaluation.simulation is not None:
            # use weak agent as strong agent
            # use worsened version of weak agent as weak agent
            worse_weak_agent = ExploreWrapper(config, base_envs[name], dc(weak_agent))
            coord_envs[name] = CoordEnv(
                config.coord_env,
                base_envs[name],
                worse_weak_agent,
                weak_agent
            )
        else:
            coord_envs[name] = CoordEnv(
                config.coord_env, base_envs[name], weak_agent, strong_agent
            )

    # skyline is trained on test environments
    if config.general.skyline:
        coord_envs["train"] = dc(coord_envs["test"])

    # set costs for getting help from strong agent
    test_eval_info = get_test_eval_info(config, coord_envs)
    for name in coord_envs:
        coord_envs[name].set_costs(test_eval_info)

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
    for name in ["train", "val_id", "val_ood", "test"]:
        env = create_fn(name, config.environment)
        # some extra information
        env.name = config.environment.common.env_name
        env.obs_shape = env.observation_space.shape
        env.action_shape = env.action_space.shape
        env.num_actions = env.action_space.n

        envs[name] = env

    return envs


def load_agents(config, env):
    module = importlib.import_module(f"YRC.envs.{get_global_variable('benchmark')}")
    load_fn = getattr(module, "load_policy")

    weak_agent = load_fn(config.agents.weak, env)
    strong_agent = load_fn(config.agents.strong, env)

    return weak_agent, strong_agent


def load_weak_agent(config):
    name = f"{config.weak.env_name}-{config.weak.architecture}-n{config.weak.num_demos}"
    pi_w = agents.names[config.weak.architecture](name, config.weak.to_dict())
    pi_w.load(config.weak.file)
    pi_w.eval()
    return pi_w


def to_dict(cls):
    def _convert(obj):
        if isinstance(obj, (int, float, str, bool, type(None))):
            return obj
        elif isinstance(obj, dict):
            return {k: _convert(v) for k, v in obj.items()}
        elif inspect.isclass(obj):
            return {
                k: _convert(v)
                for k, v in obj.__dict__.items()
                if not k.startswith("__") and not callable(v)
            }
        elif hasattr(obj, "__dict__"):
            return {
                k: _convert(v)
                for k, v in obj.__dict__.items()
                if not k.startswith("__")
            }
        else:
            return str(obj)

    attributes = {}
    for name, value in inspect.getmembers(cls):
        if (
            not name.startswith("__")
            and not inspect.ismethod(value)
            and not inspect.isfunction(value)
        ):
            attributes[name] = _convert(value)

    return attributes


def create_cliport_env(env_mode, common_config, specific_config):
    tsk = tasks.names[specific_config.env_name]()
    tsk.mode = env_mode
    env = cliport_environment.Environment(
        common_config.assets_root,
        tsk,
        common_config.disp,
        common_config.shared_memory,
        hz=480,
    )
    env.seed(specific_config.seed)
    return env


class CoordEnv(gym.Env):
    WEAK = 0
    STRONG = 1

    def __init__(self, config, base_env, weak_agent, strong_agent):
        self.args = config

        self.base_env = base_env
        self.weak_agent = weak_agent
        self.strong_agent = strong_agent

        self.action_space = gym.spaces.Discrete(2)
        self.observation_space = gym.spaces.Dict(
            {
                "env_obs": base_env.observation_space,
                "weak_features": gym.spaces.Box(
                    -100, 100, shape=(weak_agent.hidden_dim,)
                ),
                "weak_logit": gym.spaces.Box(-100, 100, shape=(base_env.num_actions,)),
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
            "weak_logit": (self.base_env.num_actions,),
        }

    def reset(self):
        self.prev_action = None
        self.env_obs = self.base_env.reset()
        self._reset_agents(np.array([True] * self.num_envs))
        return self._get_obs()

    def _reset_agents(self, done):
        self.weak_agent.reset(done)
        self.strong_agent.reset(done)

    def step(self, action):
        env_action = self._compute_env_action(action)
        self.env_obs, env_reward, done, env_info = self.base_env.step(env_action)
        info = {
            "env_info": env_info,
            "env_reward": env_reward,
            "env_action": env_action,
        }
        reward = self._get_reward(env_reward, action, done)
        self._reset_agents(done)
        self.prev_action = action

        return self._get_obs(), reward, done, info

    def _compute_env_action(self, action):
        # NOTE: this method only works with non-recurrent agent models
        greedy = self.args.act_greedy
        env_action = np.zeros_like(action)
        is_weak = action == self.WEAK
        if is_weak.sum() > 0:
            env_action[is_weak] = self.weak_agent.act(
                self.env_obs[is_weak], greedy=greedy, mask=is_weak
            )
        is_strong = ~is_weak
        if is_strong.sum() > 0:
            env_action[is_strong] = self.strong_agent.act(
                self.env_obs[is_strong], greedy=greedy, mask=is_strong
            )
        return env_action

    def _get_obs(self):
        obs = {
            "env_obs": self.env_obs,
            "weak_features": self.weak_agent.get_hidden(self.env_obs).detach(),
            "weak_logit": self.weak_agent.forward(self.env_obs).detach(),
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

    """


    def _step_procgen(self):
        obs = self.base_env.reset()  # Get current observation
        zero_mask = self.action == 0
        new_actions = np.empty_like(self.action)
        if np.any(zero_mask):
            new_actions[zero_mask], _, _ = self.weak_agent.predict(obs[zero_mask])
        if np.any(~zero_mask):
            new_actions[~zero_mask], _, _ = self.strong_agent.predict(obs[~zero_mask])
        self.base_env.step_async(new_actions)
        obs, reward, done, info = self.base_env.step_wait()
        reward = self.strong_query(reward)
        reward = self.switching_agent(reward, done)
        obs_tensor = torch.FloatTensor(obs).to(device=self.device)
        pi_w_hidden = self.get_weak_agent_features(obs_tensor)
        return obs, reward, done, info, pi_w_hidden


    def reset(self, need_features=True):
        obs = self.base_env.reset()
        pi_w_hidden = None
        if need_features:
            if self.benchmark == "procgen":
                obs_tensor = torch.FloatTensor(obs).to(device=self.device)
                pi_w_hidden = self.get_weak_agent_features(obs_tensor)
            elif self.benchmark == "cliport":
                pi_w_hidden = self.get_weak_agent_features(
                    [cliport_utils.get_image(obs)], [self.base_env.info]
                )
        return obs, pi_w_hidden


    def strong_query(self, rew):
        return np.where(self.action == 1, rew - self.strong_query_cost_per_action, rew)

    def switching_agent(self, rew, done):
        if self.prev_action is not None:
            switching_idx = np.where((self.action != self.prev_action) & (~done))
            if switching_idx[0].size > 0:
                rew[switching_idx] -= self.switching_agent_cost_per_action
        self.prev_action = self.action
        return rew

    def set_strong_agent(self, strong_agent):
        self.strong_agent = strong_agent[0]

    def set_costs(self, reward_max):
        self.strong_query_cost_per_action = (
            reward_max / self.timeout
        ) * self.strong_query_cost
        self.switching_agent_cost_per_action = (
            reward_max / self.timeout
        ) * self.switching_cost

    def get_weak_agent_features(self, obs, info=None):
        if self.feature_type not in ["T2", "T3"]:
            return None

        if self.benchmark == "procgen":
            return self.weak_agent.policy.extract_features(obs)
        elif self.benchmark == "cliport":
            pi_w_pick_hidden, pi_w_place_hidden = self.weak_agent.extract_features(
                obs, info
            )
            pi_w_pick_hidden = (
                torch.stack(pi_w_pick_hidden)
                if isinstance(pi_w_pick_hidden, list)
                else pi_w_pick_hidden
            )
            pi_w_place_hidden = (
                torch.stack(pi_w_place_hidden)
                if isinstance(pi_w_place_hidden, list)
                else pi_w_place_hidden
            )

            if pi_w_pick_hidden.dim() != 2:
                pi_w_pick_hidden = pi_w_pick_hidden.unsqueeze(0)
                pi_w_place_hidden = pi_w_place_hidden.unsqueeze(0)

            return torch.cat([pi_w_pick_hidden, pi_w_place_hidden], dim=-1)

    def step_async(self, actions):
        # specific for procgen
        obs = self.base_env.reset()  # Get current observation
        zero_mask = actions == 0
        new_actions = np.empty_like(actions)
        if np.any(zero_mask):
            new_actions[zero_mask], _, _ = self.weak_agent.predict(obs[zero_mask])
        if np.any(~zero_mask):
            new_actions[~zero_mask], _, _ = self.strong_agent.predict(obs[~zero_mask])
        self.action = actions
        self.base_env.step_async(new_actions)

    def step_wait(self):
        # specific for procgen
        obs, reward, done, info = self.base_env.step_wait()
        reward = self.strong_query(reward)
        reward = self.switching_agent(reward, done)
        obs_tensor = torch.FloatTensor(obs).to(device=self.device)
        pi_w_hidden = self.get_weak_agent_features(obs_tensor)
        return obs, reward, done, info, pi_w_hidden

    def step(self, action=None):
        self.action = action

        if self.benchmark == "procgen":
            return self._step_procgen()
        elif self.benchmark == "cliport":
            return self._step_cliport(action)
        else:
            raise ValueError(f"Unknown benchmark: {self.benchmark}")

    def _step_cliport(self, action):
        if action is not None:
            obs = self.base_env._get_obs()
            info = self.base_env.info
            goal = self.base_env.get_lang_goal()
            new_action = (
                self.weak_agent.act(obs, info, goal)[0]
                if action[0] == 0
                else self.strong_agent(obs, info)
            )
            obs, reward, done, info = self.base_env.step(new_action)
            reward = self.strong_query(reward)
            reward = self.switching_agent(reward, done)
            pi_w_hidden = self.get_weak_agent_features(
                [cliport_utils.get_image(obs)], [info]
            )
            info = [info]
        else:
            obs, reward, done, info = self.base_env.step(action)
            return obs, reward, done, info
        return obs, reward, done, info, pi_w_hidden

    def render(self, mode="human"):
        return self.base_env.render(mode)

    """
