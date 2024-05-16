from typing import List, Dict, Tuple

import gym
import numpy as np
import torch


class ReplayBuffer:
    def __init__(self, num_steps: int, num_envs: int, observation_space: gym.spaces.Space, action_space: gym.spaces.Space, device: str):
        self._buffer_size = num_steps
        self._pointer = 0
        self._size = 0
        self._obs = torch.zeros((num_steps, num_envs) + observation_space.shape).to(device)
        self._next_obs = torch.zeros((num_steps, num_envs) + observation_space.shape).to(device)
        self._actions = torch.zeros((num_steps, num_envs) + action_space.shape).to(device)
        self._logprobs = torch.zeros((num_steps, num_envs)).to(device)
        self._rewards = torch.zeros((num_steps, num_envs)).to(device)
        self._dones = torch.zeros((num_steps, num_envs)).to(device)
        self._values = torch.zeros((num_steps, num_envs)).to(device)
        self._device = device

    def _to_tensor(self, data: np.ndarray) -> torch.Tensor:
        return torch.tensor(data, dtype=torch.float32, device=self._device)

    # Loads data in d4rl format, i.e. from Dict[str, np.array].
    def load_dataset(self, data: Dict[str, np.ndarray]) -> None:
        n_transitions = data["observations"].shape[0]
        if n_transitions > self._buffer_size + self._size:
            raise ValueError("Replay buffer is smaller than the dataset you are trying to load!")
        self._obs[self._size:n_transitions + self._size] = self._to_tensor(data["observations"])
        self._actions[self._size:n_transitions + self._size] = self._to_tensor(data["actions"])
        self._rewards[self._size:n_transitions + self._size] = self._to_tensor(data["rewards"][..., None])
        self._next_obs[self._size:n_transitions + self._size] = self._to_tensor(data["next_observations"])
        self._dones[self._size:n_transitions + self._size] = self._to_tensor(data["terminals"][..., None])
        self._size += n_transitions
        self._pointer = min(self._size, n_transitions)

        print(f"Dataset size: {self._size}")

    def sample(self, batch_size: int) -> List[torch.Tensor]:
        indices = np.random.randint(0, min(self._size, self._pointer), size=batch_size)
        obs = self._obs[indices]
        actions = self._actions[indices]
        rewards = self._rewards[indices]
        next_obs = self._next_obs[indices]
        dones = self._dones[indices]
        return [obs, actions, rewards, next_obs, dones]

    def sample_by_range(self, start_i: int, end_i: int) -> List[torch.Tensor]:
        indices = np.arange(start_i, end_i)
        obs = self._obs[indices]
        actions = self._actions[indices]
        rewards = self._rewards[indices]
        next_obs = self._next_obs[indices]
        dones = self._dones[indices]
        return [obs, actions, rewards, next_obs, dones]

    def sample_by_indices(self, indices: List, obs_shape: Tuple, action_shape: Tuple) -> List[torch.Tensor]:
        obs = self._obs.reshape((-1,) + obs_shape)[indices]
        actions = self._actions.reshape((-1,) + action_shape)[indices]
        logprobs = self._logprobs.reshape(-1)[indices]
        values = self._values.reshape(-1)[indices]
        return [obs, actions, logprobs, values]

    def add_transition(self, obs: torch.Tensor, actions: torch.Tensor, rewards: torch.Tensor, next_obs: torch.Tensor, dones: torch.Tensor,
                       values: torch.Tensor = None, logprobs: torch.Tensor = None) -> None:
        self._obs[self._pointer] = obs
        self._actions[self._pointer] = actions
        self._rewards[self._pointer] = rewards
        self._next_obs[self._pointer] = next_obs
        self._dones[self._pointer] = dones
        if values is not None:
            self._values[self._pointer] = values
        if logprobs is not None:
            self._logprobs[self._pointer] = logprobs

        self._pointer = (self._pointer + 1) % self._buffer_size
        self._size = min(self._size + 1, self._buffer_size)
