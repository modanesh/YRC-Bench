import gymnasium as gym
import random


class HardResetWrapper(gym.Wrapper):
    def __init__(self, env, seed):
        super().__init__(env)
        self.seed = seed

    def reset(self, **kwargs):
        new_seed = random.randint(self.seed - 1000, self.seed + 1000)
        obs, _ = self.env.reset(seed=new_seed, **kwargs)
        return obs
