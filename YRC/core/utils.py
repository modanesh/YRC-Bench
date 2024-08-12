import inspect
import os
import random
import numpy as np
import torch
import wandb
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
from cliport.dataset import RavensDataset, RavensMultiTaskDataset
from cliport.utils import utils as cliport_utils
from .logger import Logger
from YRC.core.configs import get_global_variable


def logger_setup(config, is_test=False):
    print(f'Logging to {config.algorithm.save_dir}')
    vars_config = config.to_dict()
    # if not is_test:
    #     wandb.init(config=vars_config, resume="allow", project="YRC", name=run_name, settings=wandb.Settings(code_dir="."))  # todo: uncomment
    num_envs = int(config.environments.procgen.common.num_envs if config.general.benchmark == 'procgen' else 1)
    writer = Logger(num_envs, config.algorithm.save_dir, config.general.benchmark)
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


class CliportReplayBuffer(ReplayBuffer):
    def __init__(self, gamma, lmbda, use_gae, normalize_adv, obs_shape, buffer_size, num_envs):
        super().__init__(gamma, lmbda, use_gae, normalize_adv, obs_shape, buffer_size, num_envs)

    def add_transition(self, state, action, logprob, reward, next_state, done, value, info):
        img_state = self._process_image(state)
        img_next_state = self._process_image(next_state)
        super().add_transition(img_state, action, logprob, reward, img_next_state, done, value, info)

    def store_last(self, last_obs, last_value):
        img_last_obs = self._process_image(last_obs)
        super().store_last(img_last_obs, last_value)

    def store_last_done(self):
        self.done_batch[self.pointer - 1] = 1

    def _process_image(self, state):
        return cliport_utils.get_image(state).transpose(2, 0, 1)


class ProcgenReplayBuffer(ReplayBuffer):
    pass  # No changes needed for ProcgenReplayBuffer
