import gymnasium as gym
from gymnasium.vector.vector_env import VectorEnv


class HardResetWrapper(VectorEnv):
    def __init__(self, env):
        self.env = env
        self.observation_space = env.observation_space
        self.action_space = env.action_space
        assert env.action_space.nvec.min() == env.action_space.nvec.max(), "Action space must be discrete"
        self.action_space.n = env.action_space.nvec.min()
        self.num_envs = env.num_envs

    def reset(self, **kwargs):
        obs, _ = self.env.reset(seed=self.env.np_random_seed[-1] + 1)
        return obs

    def step(self, actions):
        obs, reward, termination, truncation, info = self.env.step(actions)
        done = termination | truncation  # wrapper for gymnasium to older gym
        info = [{"env_reward": r} for r in reward]
        return obs, reward, done, info
