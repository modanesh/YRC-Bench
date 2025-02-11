import random

from lib.cliport.cliport.utils import utils
import gym
from gym import spaces
import numpy as np


class HardResetWrapper(gym.Wrapper):
    def __init__(self, env, start_level, num_levels, distribution_mode):
        self.env = env
        self.start_level = start_level
        self.num_levels = num_levels
        self.distribution_mode = distribution_mode
        sample_obs = utils.get_image(self.env.reset())
        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=sample_obs.shape,
            dtype=sample_obs.dtype
        )

        self.action_space = env.action_space
        total_dim = 0
        for space in env.action_space.spaces.values():
            for subspace in space.spaces:
                total_dim += subspace.shape[0]
        self.action_space.shape = ()  # placeholder for the moment
        self.action_space.n = 4  # placeholder for the moment

    def reset(self):
        new_seed = random.randint(self.start_level, self.start_level + self.num_levels - 1)
        self.env.seed(new_seed)
        obs = self.env.reset()
        obs = utils.get_image(obs)
        self.env_step = 0
        obs = np.expand_dims(obs, axis=0)
        return {"image": obs, "info": self.info}

    def step(self, np_action):
        if np.any(np_action == None) or self.env_step == self.task.max_steps or len(self.env.task.goals) == 0:
            action = None
            _, reward, _, info = self.env.step(action)
            obs = self.reset()
            done = True
            return obs, np.array([reward]), np.array([done]), [info]

        np_action = np_action[0]
        pose0_pos = np_action[:3]
        pose0_ori = np_action[3:7]
        pose1_pos = np_action[7:10]
        pose1_ori = np_action[10:14]
        action = {
            'pose0': (pose0_pos, pose0_ori),
            'pose1': (pose1_pos, pose1_ori)
        }
        if len(np_action) > 14:
            pick = np_action[14:17]
            place = np_action[17:]
            action['pick'] = pick
            action['place'] = place
        obs, reward, done, info = self.env.step(action)
        obs = utils.get_image(obs)
        self.env_step += 1
        obs = np.expand_dims(obs, axis=0)
        return {"image": obs, "info": self.info}, np.array([reward]), np.array([done]), [info]
