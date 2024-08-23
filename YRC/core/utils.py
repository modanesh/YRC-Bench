import inspect
import os
import random

import numpy as np
import torch
import wandb
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler

from YRC.core.configs import get_global_variable
from cliport.dataset import RavensDataset, RavensMultiTaskDataset
from cliport.utils import utils as cliport_utils
from .logger import Logger


def logger_setup(config, is_test=False):
    save_dir = getattr(config.algorithm, config.algorithm.cls).save_dir
    run_name = config.algorithm.run_name
    print(f'Logging to {save_dir}')
    vars_config = config.to_dict()
    # if not is_test:
    #     wandb.init(config=vars_config, resume="allow", project="YRC", name=run_name, settings=wandb.Settings(code_dir="."))
    num_envs = int(config.environments.procgen.common.num_envs if config.general.benchmark == 'procgen' else 1)
    writer = Logger(num_envs, save_dir, config.general.benchmark)
    return writer


def to_dict(cls):
    def _convert(obj):
        if isinstance(obj, (int, float, str, bool, type(None))):
            return obj
        elif isinstance(obj, dict):
            return {k: _convert(v) for k, v in obj.items()}
        elif inspect.isclass(obj):
            return {k: _convert(v) for k, v in obj.__dict__.items()
                    if not k.startswith('__') and not callable(v)}
        elif hasattr(obj, '__dict__'):
            return {k: _convert(v) for k, v in obj.__dict__.items()
                    if not k.startswith('__')}
        else:
            return str(obj)

    attributes = {}
    for name, value in inspect.getmembers(cls):
        if not name.startswith('__') and not inspect.ismethod(value) and not inspect.isfunction(value):
            attributes[name] = _convert(value)

    return attributes


def set_global_seeds(seed, torch_deterministic=True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = torch_deterministic
    torch.backends.cudnn.benchmark = not torch_deterministic


def load_dataset(config):
    data_dir = config.data_dir
    dataset_type = config.dataset_type
    task = config.task
    num_demos = config.num_demos
    if 'multi' in dataset_type:
        train_ds = RavensMultiTaskDataset(data_dir, config.to_dict(), group=task, mode='train', n_demos=num_demos, augment=True)
    else:
        train_ds = RavensDataset(os.path.join(data_dir, '{}-train'.format(task)), config.to_dict(), n_demos=num_demos, augment=True)
    return train_ds


class ReplayBuffer:
    def __init__(self, gamma, lmbda, use_gae, normalize_adv, obs_shape, buffer_size, num_envs):
        self.obs_shape = obs_shape
        self.buffer_size = buffer_size
        self.num_envs = num_envs
        self.device = get_global_variable("device")
        self.pointer = 0
        self.size = 0
        self.gamma = gamma
        self.lmbda = lmbda
        self.use_gae = use_gae
        self.normalize_adv = normalize_adv

        # Initialize all tensors
        self.obs_batch = torch.zeros((self.buffer_size + 1, num_envs, *obs_shape), dtype=torch.float32, device=self.device)
        self.action_batch = torch.zeros((self.buffer_size, num_envs), dtype=torch.float32, device=self.device)
        self.reward_batch = torch.zeros((self.buffer_size, num_envs), dtype=torch.float32, device=self.device)
        self.done_batch = torch.zeros((self.buffer_size, num_envs), dtype=torch.float32, device=self.device)
        self.log_prob_batch = torch.zeros((self.buffer_size, num_envs), dtype=torch.float32, device=self.device)
        self.value_batch = torch.zeros((self.buffer_size + 1, num_envs), dtype=torch.float32, device=self.device)
        self.return_batch = torch.zeros((self.buffer_size, num_envs), dtype=torch.float32, device=self.device)
        self.advantage_batch = torch.zeros((self.buffer_size, num_envs), dtype=torch.float32, device=self.device)

        self.info_batch = [None] * self.buffer_size

    def _to_torch(self, data):
        return torch.as_tensor(data, dtype=torch.float32, device=self.device)

    def add_transition(self, obs, action, log_prob, reward, next_obs, done, value, info):
        self.obs_batch[self.pointer] = self._to_torch(obs)
        self.action_batch[self.pointer] = self._to_torch(action)
        self.log_prob_batch[self.pointer] = self._to_torch(log_prob)
        self.reward_batch[self.pointer] = self._to_torch(reward)
        self.obs_batch[(self.pointer + 1) % self.buffer_size] = self._to_torch(next_obs)
        self.done_batch[self.pointer] = self._to_torch(done)
        self.value_batch[self.pointer] = self._to_torch(value)
        self.info_batch[self.pointer] = info

        self.pointer = (self.pointer + 1) % self.buffer_size
        self.size = min(self.size + 1, self.buffer_size)

    def compute_estimates(self):
        with torch.no_grad():
            if self.use_gae:
                gae = 0
                for step in reversed(range(self.size)):
                    delta = (self.reward_batch[step] +
                             self.gamma * self.value_batch[(step + 1) % self.buffer_size] * (1 - self.done_batch[step]) -
                             self.value_batch[step])
                    gae = delta + self.gamma * self.lmbda * (1 - self.done_batch[step]) * gae
                    self.advantage_batch[step] = gae
                self.return_batch = self.advantage_batch + self.value_batch[:self.size]
            else:
                return_value = self.value_batch[-1]
                for step in reversed(range(self.size)):
                    return_value = self.reward_batch[step] + self.gamma * return_value * (1 - self.done_batch[step])
                    self.return_batch[step] = return_value
                self.advantage_batch = self.return_batch - self.value_batch[:self.size]

            if self.normalize_adv:
                self.advantage_batch = (self.advantage_batch - self.advantage_batch.mean()) / (self.advantage_batch.std() + 1e-8)

    def fetch_train_generator(self, mini_batch_size=None):
        batch_size = self.size * self.num_envs
        if mini_batch_size is None:
            mini_batch_size = batch_size

        sampler = BatchSampler(SubsetRandomSampler(range(batch_size)), mini_batch_size, drop_last=True)

        for indices in sampler:
            yield (
                self.obs_batch[:self.size].reshape(-1, *self.obs_shape)[indices],
                self.action_batch[:self.size].reshape(-1)[indices],
                self.done_batch[:self.size].reshape(-1)[indices],
                self.log_prob_batch[:self.size].reshape(-1)[indices],
                self.value_batch[:self.size].reshape(-1)[indices],
                self.return_batch[:self.size].reshape(-1)[indices],
                self.advantage_batch[:self.size].reshape(-1)[indices],
                [self.info_batch[i // self.num_envs][i % self.num_envs] for i in indices]
            )

    def fetch_log_data(self):
        return self.reward_batch[:self.size].cpu().numpy(), self.done_batch[:self.size].cpu().numpy()

    def store_last(self, last_obs, last_value):
        self.obs_batch[self.pointer] = self._to_torch(last_obs)
        self.value_batch[self.pointer] = self._to_torch(last_value)


class CliportReplayBufferOnPolicy(ReplayBuffer):
    def __init__(self, gamma, lmbda, use_gae, normalize_adv, obs_shape, buffer_size, num_envs):
        super().__init__(gamma, lmbda, use_gae, normalize_adv, obs_shape, buffer_size, num_envs)

    def add_transition(self, obs, action, logprob, reward, next_obs, done, value, info):
        img_obs = self._process_image(obs)
        img_next_obs = self._process_image(next_obs)
        super().add_transition(img_obs, action, logprob, reward, img_next_obs, done, value, info)

    def store_last(self, last_obs, last_value):
        img_last_obs = self._process_image(last_obs)
        super().store_last(img_last_obs, last_value)

    def store_last_done(self):
        self.done_batch[self.pointer - 1] = 1

    def _process_image(self, obs):
        obs = cliport_utils.get_image(obs)
        obs = torch.FloatTensor(obs).to(device=get_global_variable("device"))
        # obs = obs.permute(2, 0, 1)
        return obs


class ProcgenReplayBufferOnPolicy(ReplayBuffer):
    pass  # No changes needed for ProcgenReplayBufferOnPolicy


class ReplayBufferOffPolicy(ReplayBuffer):
    def __init__(self, capacity, obs_shape, action_size, num_envs=1, n_step=3, gamma=0.99):
        self.capacity = capacity
        self.obs_shape = obs_shape
        self.action_size = action_size
        self.num_envs = num_envs
        self.n_step = n_step
        self.gamma = gamma
        self.device = get_global_variable("device")

        self.obs = torch.zeros((capacity, *obs_shape), dtype=torch.float32, device=self.device)
        self.next_obs = torch.zeros((capacity, *obs_shape), dtype=torch.float32, device=self.device)
        self.actions = torch.zeros((capacity, 1), dtype=torch.long, device=self.device)
        self.raw_rewards = torch.zeros((capacity, 1), dtype=torch.float32, device=self.device)
        self.rewards = torch.zeros((capacity, 1), dtype=torch.float32, device=self.device)
        self.dones = torch.zeros((capacity, 1), dtype=torch.float32, device=self.device)
        self.infos = [None for _ in range(capacity)]
        self.next_infos = [None for _ in range(capacity)]

        self.pos = 0
        self.full = False
        self.episode_buffer = []

    def _to_tensor(self, data):
        if isinstance(data, torch.Tensor):
            return data.to(self.device)
        else:
            return torch.as_tensor(data, dtype=torch.float32, device=self.device)

    def add_transition(self, observations, actions, rewards, next_observations, dones, infos, **kwargs):
        tensors = map(self._to_tensor, (observations, actions, rewards, next_observations, dones))
        self.episode_buffer.append((*tensors, infos))

        if dones:
            self._process_episode()

    def _process_episode(self):
        episode_length = len(self.episode_buffer)

        for i in range(episode_length):
            raw_reward = self.episode_buffer[i][2].clone()
            n_step_reward = self._get_n_step_reward(i, min(self.n_step, episode_length - i))

            obs, action, _, next_obs, done, info = self.episode_buffer[i]

            self.obs[self.pos] = obs
            self.actions[self.pos] = action
            self.raw_rewards[self.pos] = raw_reward
            self.rewards[self.pos] = n_step_reward
            self.next_obs[self.pos] = next_obs
            self.dones[self.pos] = done
            self.infos[self.pos] = info
            self.next_infos[self.pos] = self.episode_buffer[i + 1][5] if i + 1 < episode_length else None

            self.pos = (self.pos + 1) % self.capacity
            self.full = self.full or self.pos == 0

        self.episode_buffer.clear()

    def _get_n_step_reward(self, start_idx, steps):
        reward = self.episode_buffer[start_idx][2]  # Initial reward

        for i in range(1, steps):
            step_reward = self.episode_buffer[start_idx + i][2]
            reward += (self.gamma ** i) * step_reward
            if self.episode_buffer[start_idx + i][4]:  # If this step is done
                break  # Stop accumulating reward if we hit the end of the episode

        return reward

    def sample(self, batch_size):
        indices = torch.randint(0, self.capacity if self.full else self.pos, (batch_size,))

        return (
            self.obs[indices],
            self.actions[indices].squeeze(-1),
            self.raw_rewards[indices].squeeze(-1),
            self.rewards[indices].squeeze(-1),
            self.next_obs[indices],
            self.dones[indices].squeeze(-1),
            [self.infos[i.item()] for i in indices],
            [self.next_infos[i.item()] for i in indices]
        )

    def __len__(self):
        return self.capacity if self.full else self.pos

    def fetch_log_data(self):
        if self.full:
            return self.raw_rewards.cpu().numpy(), self.rewards.cpu().numpy(), self.dones.cpu().numpy()
        else:
            return self.raw_rewards[:self.pos].cpu().numpy(), self.rewards[:self.pos].cpu().numpy(), self.dones[:self.pos].cpu().numpy()


class CliportReplayBufferOffPolicy(ReplayBufferOffPolicy):
    def add_transition(self, obs, action, reward, next_obs, done, info, **kwargs):
        img_obs = self._process_image(obs)
        img_next_obs = self._process_image(next_obs)
        super().add_transition(img_obs, action, reward, img_next_obs, done, info)

    def _process_image(self, obs):
        obs = cliport_utils.get_image(obs)
        obs = torch.FloatTensor(obs).to(device=get_global_variable("device"))
        # obs = obs.permute(2, 0, 1)
        return obs


class ProcgenReplayBufferOffPolicy(ReplayBufferOffPolicy):
    pass  # No changes needed
