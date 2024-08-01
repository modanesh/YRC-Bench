import argparse
import csv
import inspect
import json
import os
import random
import time
import uuid
from collections import deque

import numpy as np
import pandas as pd
import torch
import wandb
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler

from . import procgen_wrappers

from procgen import ProcgenEnv
from cliport import tasks, agents
from cliport.environments import environment
from cliport.utils import utils as cliport_utils

from .models import cliportPPO, procgenPPO, CategoricalPolicy, ImpalaModel, PPOFrozen


########################################################################
# general utils
########################################################################


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weak_model_file', type=str, help='Path to the weak model file.')
    parser.add_argument('--strong_model_file', type=str, help='Path to the strong model file, required for procgen.')
    parser.add_argument('--help_policy_type', type=str, choices=['T1', 'T2', 'T3'], required=True,
                        help='Type of the helper policy. '
                             'T1: vanilla PPO (input is obs), '
                             'T2: PPO with inputs concatenated by the weak agent features (conv + mlp), '
                             'T3: PPO with inputs from the weak agent (mlp).')
    parser.add_argument('--benchmark', type=str, choices=['procgen', 'cliport'], required=True, help='Benchmark type.')
    parser.add_argument('--env_name', type=str, help='Environment name for training for procgen.')
    parser.add_argument('--param_name', type=str, help='Parameter name used to determine the additional '
                                                       'config for env, required for procgen, e.g. easy-200')
    parser.add_argument('--switching_cost', type=float, required=True, help='Switching cost for the help policy.')
    parser.add_argument('--strong_query_cost', type=float, required=True, help='Strong query cost for the help policy.')
    parser.add_argument('--distribution_mode', type=str, default='easy', help='Distribution mode for procgen.')
    parser.add_argument('--task', type=str, help='Task name for cliport.')
    args = parser.parse_args()
    verify_args(args)
    return args


def verify_args(args):
    # general checks
    if args.weak_model_file is None:
        raise ValueError("Weak model file not provided.")
    # procgen checks
    if args.benchmark == 'procgen':
        if args.strong_model_file is None:
            raise ValueError("Strong model file not provided for procgen.")
        if args.param_name is None:
            raise ValueError("Param name not provided for procgen.")
        if args.env_name is None:
            raise ValueError("Env name not provided for procgen.")
    # cliport checks
    if args.benchmark == 'cliport':
        if args.task is None:
            raise ValueError("Task name not provided for cliport.")


class Logger(object):
    def __init__(self, n_envs, logdir, benchmark):
        self.start_time = time.time()
        self.n_envs = n_envs
        self.logdir = logdir

        # training
        self.episode_rewards = []
        for _ in range(n_envs):
            self.episode_rewards.append([])

        self.episode_timeout_buffer = deque(maxlen=40)
        self.episode_len_buffer = deque(maxlen=40)
        self.episode_reward_buffer = deque(maxlen=40)

        # validation
        self.episode_rewards_v = []
        for _ in range(n_envs):
            self.episode_rewards_v.append([])

        self.episode_timeout_buffer_v = deque(maxlen=40)
        self.episode_len_buffer_v = deque(maxlen=40)
        self.episode_reward_buffer_v = deque(maxlen=40)

        time_metrics = ["timesteps", "wall_time", "num_episodes"]  # only collected once
        episode_metrics = ["max_episode_rewards", "mean_episode_rewards", "min_episode_rewards",
                           "max_episode_len", "mean_episode_len",
                           "min_episode_len"]  # collected for both train and val envs
        self.benchmark = benchmark
        if self.benchmark == 'procgen':
            self.log = pd.DataFrame(columns=time_metrics + episode_metrics + ["val_" + m for m in episode_metrics])
        elif self.benchmark == 'cliport':
            self.log = pd.DataFrame(columns=time_metrics + episode_metrics)
        self.timesteps = 0
        self.num_episodes = 0

    def feed_procgen(self, act_batch, rew_batch, done_batch, rew_batch_v=None, done_batch_v=None):
        steps = rew_batch.shape[0]
        rew_batch = rew_batch.T
        done_batch = done_batch.T

        valid = rew_batch_v is not None and done_batch_v is not None and rew_batch_v.shape[0] > 0
        if valid:
            rew_batch_v = rew_batch_v.T
            done_batch_v = done_batch_v.T

        for i in range(self.n_envs):
            for j in range(steps):
                # TODO FOR EVAL: make it so that actions are also logged to csv. if doesn't work, just log to some text file
                self.episode_rewards[i].append(rew_batch[i][j])
                if valid:
                    self.episode_rewards_v[i].append(rew_batch_v[i][j])

                if done_batch[i][j]:
                    self.episode_len_buffer.append(len(self.episode_rewards[i]))
                    self.episode_reward_buffer.append(np.sum(self.episode_rewards[i]))
                    self.episode_rewards[i] = []
                    self.num_episodes += 1
                if valid and done_batch_v[i][j]:
                    self.episode_timeout_buffer_v.append(1 if j == steps - 1 else 0)
                    self.episode_len_buffer_v.append(len(self.episode_rewards_v[i]))
                    self.episode_reward_buffer_v.append(np.sum(self.episode_rewards_v[i]))
                    self.episode_rewards_v[i] = []

        self.timesteps += (self.n_envs * steps)

    def feed_cliport(self, rew_batch, done_batch, rew_batch_v=None, done_batch_v=None):
        # get the length of each episode, ends when done_batch is 1
        eps_lengths = np.where(done_batch == 1)[0] + 1

        self.episode_rewards = np.split(rew_batch, eps_lengths[:-1], axis=0)
        self.episode_len_buffer = np.insert(np.diff(eps_lengths), 0, eps_lengths[0])
        self.episode_reward_buffer = np.array([np.sum(rew) for rew in self.episode_rewards])
        self.num_episodes = len(self.episode_reward_buffer)
        self.episode_rewards = []

    def dump(self, is_test=False):
        wall_time = time.time() - self.start_time
        episode_statistics = self._get_episode_statistics()
        episode_statistics_list = list(episode_statistics.values())
        log = [self.timesteps, wall_time, self.num_episodes] + episode_statistics_list
        self.log.loc[len(self.log)] = log

        with open(self.logdir + '/log-append.csv', 'a') as f:
            writer = csv.writer(f)
            if f.tell() == 0:
                writer.writerow(self.log.columns)
            writer.writerow(log)

        print(self.log.loc[len(self.log) - 1])
        if not is_test:
            wandb.log({k: v for k, v in zip(self.log.columns, log)})
        else:
            for k, v in zip(self.log.columns, log):
                print(f'{k}: {v}')

    def _get_episode_statistics(self):
        episode_statistics = {}
        episode_statistics['Rewards/max_episodes'] = np.max(self.episode_reward_buffer)
        episode_statistics['Rewards/mean_episodes'] = np.mean(self.episode_reward_buffer)
        episode_statistics['Rewards/min_episodes'] = np.min(self.episode_reward_buffer)
        episode_statistics['Len/max_episodes'] = np.max(self.episode_len_buffer)
        episode_statistics['Len/mean_episodes'] = np.mean(self.episode_len_buffer)
        episode_statistics['Len/min_episodes'] = np.min(self.episode_len_buffer)

        if self.benchmark == 'procgen':
            episode_statistics['[Valid] Rewards/max_episodes'] = np.max(self.episode_reward_buffer_v, initial=0)
            episode_statistics['[Valid] Rewards/mean_episodes'] = np.mean(self.episode_reward_buffer_v)
            episode_statistics['[Valid] Rewards/min_episodes'] = np.min(self.episode_reward_buffer_v, initial=0)
            episode_statistics['[Valid] Len/max_episodes'] = np.max(self.episode_len_buffer_v, initial=0)
            episode_statistics['[Valid] Len/mean_episodes'] = np.mean(self.episode_len_buffer_v)
            episode_statistics['[Valid] Len/min_episodes'] = np.min(self.episode_len_buffer_v, initial=0)
        return episode_statistics


def logger_setup(cfgs, is_test=False):
    uuid_stamp = str(uuid.uuid4())[:8]
    env_name = cfgs.env_name if cfgs.benchmark == 'procgen' else cfgs.task
    run_name = f"PPO-{cfgs.benchmark}-help-{env_name}-type-{cfgs.help_policy_type}-query-cost-{cfgs.strong_query_cost}-{uuid_stamp}"
    logdir = os.path.join('logs', env_name)
    if not (os.path.exists(logdir)):
        os.makedirs(logdir)
    logdir = os.path.join(logdir, run_name)
    if not (os.path.exists(logdir)):
        os.mkdir(logdir)
    print(f'Logging to {logdir}')
    vars_cfgs = to_dict(cfgs)
    if not is_test:
        wandb.init(config=vars_cfgs, resume="allow", project="YRC", name=run_name)
    writer = Logger(cfgs.policy.n_envs, logdir, cfgs.benchmark)
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


def algorithm_setup(env, env_val, task, policy, logger, storage, storage_valid, device, num_checkpoints,
                    hyperparameters, help_policy_type=None):
    print('::[LOGGING]::INTIALIZING AGENT...')
    ppo_agents = {'procgen': procgenPPO, 'cliport': cliportPPO}
    agent_type = 'procgen' if isinstance(env, procgen_wrappers.HelpEnvWrapper) else 'cliport'
    agent = ppo_agents[agent_type](env, env_val, task, policy, logger, storage, device,
                                   num_checkpoints,
                                   storage_valid=storage_valid,
                                   help_policy_type=help_policy_type,
                                   **hyperparameters)
    return agent


########################################################################
# cliport utils
########################################################################


class CliportReplayBuffer:
    def __init__(self, state_dim, action_dim, buffer_size, num_envs, device='cpu'):
        self._buffer_size = buffer_size
        self._pointer = 0
        self._size = 0
        self._device = device
        self.num_envs = num_envs
        self._states = torch.zeros((buffer_size,) + state_dim, dtype=torch.float32, device=device)
        self._actions = torch.zeros((buffer_size, self.num_envs), dtype=torch.float32, device=device)
        self._rewards = torch.zeros((buffer_size, self.num_envs), dtype=torch.float32, device=device)
        self._dones = torch.zeros((buffer_size, self.num_envs), dtype=torch.float32, device=device)
        self._logprobs = torch.zeros((buffer_size, self.num_envs), dtype=torch.float32, device=device)
        self._values = torch.zeros((buffer_size, self.num_envs), dtype=torch.float32, device=device)
        self._next_states = torch.zeros((buffer_size,) + state_dim, dtype=torch.float32, device=device)
        self._adv = torch.zeros((buffer_size, self.num_envs), dtype=torch.float32, device=device)
        self._returns = torch.zeros((buffer_size, self.num_envs), dtype=torch.float32, device=device)
        self._info = [None] * buffer_size

    def _to_torch(self, data):
        return torch.tensor(data, dtype=torch.float32, device=self._device)

    def sample(self, batch_size):
        indices = np.random.randint(0, self._size, size=batch_size)
        return (
            self._states[indices],
            self._actions[indices],
            self._logprobs[indices],
            self._rewards[indices],
            self._next_states[indices],
            self._dones[indices],
            self._values[indices],
            self._adv[indices],
            self._returns[indices],
            np.array(self._info)[indices]
        )

    def add_transition(self, state, action, logprob, reward, next_state, done, value, info):
        self._states[self._pointer] = self._to_torch(state)
        self._actions[self._pointer] = self._to_torch(action)
        self._logprobs[self._pointer] = self._to_torch(logprob)
        self._rewards[self._pointer] = self._to_torch(reward)
        self._next_states[self._pointer] = self._to_torch(next_state)
        self._dones[self._pointer] = self._to_torch(done)
        self._values[self._pointer] = self._to_torch(value)
        self._info[self._pointer] = info

        self._pointer = (self._pointer + self.num_envs) % self._buffer_size
        self._size = min(self._size + 1, self._buffer_size)

    def get_info(self):
        return self._info[:self._size]

    def fetch_train_generator(self, mini_batch_size=None):
        batch_size = self._size
        if mini_batch_size is None:
            mini_batch_size = batch_size

        sampler = BatchSampler(SubsetRandomSampler(range(batch_size)),
                               mini_batch_size,
                               drop_last=True)

        for indices in sampler:
            yield (
                self._states[indices],
                self._actions[indices].squeeze(1),
                self._dones[indices],
                self._logprobs[indices].squeeze(1),
                self._values[indices].squeeze(1),
                self._returns[indices].squeeze(1),
                self._adv[indices].squeeze(1),
                [self._info[i] for i in indices]
            )

    def compute_estimates(self, gamma=0.99, lmbda=0.95, use_gae=True, normalize_adv=True):
        with torch.no_grad():
            last_value = self._values[(self._pointer - 1) % self._buffer_size]
            advantages = torch.zeros_like(self._rewards)
            returns = torch.zeros_like(self._rewards)

            if use_gae:
                gae = 0
                for step in reversed(range(self._size)):
                    if step == self._size - 1:
                        next_value = last_value
                    else:
                        next_value = self._values[(step + 1) % self._buffer_size]

                    delta = self._rewards[step] + gamma * next_value * (1 - self._dones[step]) - self._values[step]
                    gae = delta + gamma * lmbda * (1 - self._dones[step]) * gae
                    advantages[step] = gae
                    returns[step] = advantages[step] + self._values[step]
            else:
                running_return = last_value
                for step in reversed(range(self._size)):
                    running_return = self._rewards[step] + gamma * running_return * (1 - self._dones[step])
                    returns[step] = running_return
                advantages = returns - self._values[:self._size]

            if normalize_adv:
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            # Update the stored advantages and returns
            self._adv[:self._size] = advantages[:self._size]
            self._returns[:self._size] = returns[:self._size]

    def fetch_log_data(self):
        rew_batch = self._rewards.cpu().numpy()[:self._size]
        done_batch = self._dones.cpu().numpy()[:self._size]
        return rew_batch, done_batch


def cliport_environment_setup(assets_root, weak_policy, strong_query_cost, switching_agent_cost, disp, shared_memory, task,
                              help_policy_type=None, device='cuda'):
    tsk = tasks.names[task]()
    timeout = tsk.max_steps
    env = environment.HelpEnvWrapper(assets_root, weak_policy, None, timeout, strong_query_cost, switching_agent_cost, help_policy_type,
                                     device, tsk, disp=disp, shared_memory=shared_memory, hz=480)
    env.set_task(tsk)
    _ = env.reset(need_features=False)
    reward_max = 0
    for goal in env.task.goals:
        reward_max += goal[-1]
    env.set_costs(reward_max)
    strong_policy = tsk.oracle(env)
    env.set_strong_policy(strong_policy)
    return env, tsk


def load_weak_policy(cfgs):
    name = f'{cfgs.task}-{cfgs.agent}-n{cfgs.weak_n_demos}'
    pi_w = agents.names[cfgs.agent](name, to_dict(cfgs))
    pi_w.load(cfgs.weak_model_file)
    pi_w.eval()
    return pi_w


def cliport_define_help_policy(env, weak_agent, help_policy_type, device):
    obs, _ = env.reset(need_features=False)
    img = [cliport_utils.get_image(obs)]
    info = [env.info]
    pick_features, place_features = weak_agent.extract_features(img, info)
    hidden_size = pick_features[0].shape[0] + place_features[0].shape[0] if help_policy_type != "T1" else 0
    model = ImpalaModel(img[0].shape[-1], benchmark='cliport') if help_policy_type != "T3" else None
    action_size = 2
    policy = CategoricalPolicy(embedder=model, action_size=action_size, additional_hidden_dim=hidden_size)
    policy.to(device)
    return None, policy


########################################################################
# procgen utils
########################################################################


class ProcgenReplayBuffer:
    def __init__(self, obs_shape, num_steps, num_envs, device):
        print('::[LOGGING]::INITIALIZING STORAGE...')
        self.obs_shape = obs_shape
        self.num_steps = num_steps
        self.num_envs = num_envs
        self.device = device
        self.reset()

    def reset(self):
        self.obs_batch = torch.zeros(self.num_steps + 1, self.num_envs, *self.obs_shape)
        self.act_batch = torch.zeros(self.num_steps, self.num_envs)
        self.rew_batch = torch.zeros(self.num_steps, self.num_envs)
        self.done_batch = torch.zeros(self.num_steps, self.num_envs)
        self.log_prob_act_batch = torch.zeros(self.num_steps, self.num_envs)
        self.value_batch = torch.zeros(self.num_steps + 1, self.num_envs)
        self.return_batch = torch.zeros(self.num_steps, self.num_envs)
        self.adv_batch = torch.zeros(self.num_steps, self.num_envs)
        self.info_batch = deque(maxlen=self.num_steps)
        self.step = 0

    def store(self, obs, act, rew, done, info, log_prob_act, value):
        self.obs_batch[self.step] = torch.from_numpy(obs.copy())
        self.act_batch[self.step] = torch.from_numpy(act.copy())
        self.rew_batch[self.step] = torch.from_numpy(rew.copy())
        self.done_batch[self.step] = torch.from_numpy(done.copy())
        self.log_prob_act_batch[self.step] = torch.from_numpy(log_prob_act.copy())
        self.value_batch[self.step] = torch.from_numpy(value.copy())
        self.info_batch.append(info)

        self.step = (self.step + 1) % self.num_steps

    def store_last(self, last_obs, last_value):
        self.obs_batch[-1] = torch.from_numpy(last_obs.copy())
        self.value_batch[-1] = torch.from_numpy(last_value.copy())

    def compute_estimates(self, gamma=0.99, lmbda=0.95, use_gae=True, normalize_adv=True):
        rew_batch = self.rew_batch
        if use_gae:
            A = 0
            for i in reversed(range(self.num_steps)):
                rew = rew_batch[i]
                done = self.done_batch[i]
                value = self.value_batch[i]
                next_value = self.value_batch[i + 1]

                delta = (rew + gamma * next_value * (1 - done)) - value
                self.adv_batch[i] = A = gamma * lmbda * A * (1 - done) + delta
        else:
            G = self.value_batch[-1]
            for i in reversed(range(self.num_steps)):
                rew = rew_batch[i]
                done = self.done_batch[i]

                G = rew + gamma * G * (1 - done)
                self.return_batch[i] = G

        self.return_batch = self.adv_batch + self.value_batch[:-1]
        if normalize_adv:
            self.adv_batch = (self.adv_batch - torch.mean(self.adv_batch)) / (torch.std(self.adv_batch) + 1e-8)

    def fetch_train_generator(self, mini_batch_size=None):
        batch_size = self.num_steps * self.num_envs
        if mini_batch_size is None:
            mini_batch_size = batch_size
        sampler = BatchSampler(SubsetRandomSampler(range(batch_size)),
                               mini_batch_size,
                               drop_last=True)
        for indices in sampler:
            obs_batch = torch.FloatTensor(self.obs_batch[:-1]).reshape(-1, *self.obs_shape)[indices].to(self.device)
            act_batch = torch.FloatTensor(self.act_batch).reshape(-1)[indices].to(self.device)
            done_batch = torch.FloatTensor(self.done_batch).reshape(-1)[indices].to(self.device)
            log_prob_act_batch = torch.FloatTensor(self.log_prob_act_batch).reshape(-1)[indices].to(self.device)
            value_batch = torch.FloatTensor(self.value_batch[:-1]).reshape(-1)[indices].to(self.device)
            return_batch = torch.FloatTensor(self.return_batch).reshape(-1)[indices].to(self.device)
            adv_batch = torch.FloatTensor(self.adv_batch).reshape(-1)[indices].to(self.device)
            yield obs_batch, act_batch, done_batch, log_prob_act_batch, value_batch, return_batch, adv_batch

    def fetch_log_data(self):
        if 'env_reward' in self.info_batch[0][0]:
            rew_batch = []
            for step in range(self.num_steps):
                infos = self.info_batch[step]
                rew_batch.append([info['env_reward'] for info in infos])
            rew_batch = np.array(rew_batch)
        else:
            rew_batch = self.rew_batch.numpy()
        if 'env_done' in self.info_batch[0][0]:
            done_batch = []
            for step in range(self.num_steps):
                infos = self.info_batch[step]
                done_batch.append([info['env_done'] for info in infos])
            done_batch = np.array(done_batch)
        else:
            done_batch = self.done_batch.numpy()
        return self.act_batch.numpy(), rew_batch, done_batch


def procgen_environment_setup(n_steps, env_name, start_level, num_levels, distribution_mode, num_threads,
                              random_percent, step_penalty, key_penalty, rand_region, normalize_rew, weak_policy=None,
                              strong_policy=None, get_configs=False, strong_query_cost=0.0, switching_agent_cost=0.0,
                              reward_max=1.0, timeout=1000, help_policy_type=None, device='cuda'):
    print('::[LOGGING]::INITIALIZING ENVIRONMENTS...')
    env = ProcgenEnv(num_envs=n_steps,
                     env_name=env_name,
                     num_levels=num_levels,
                     start_level=start_level,
                     distribution_mode=distribution_mode,
                     num_threads=num_threads,
                     random_percent=random_percent,
                     step_penalty=step_penalty,
                     key_penalty=key_penalty,
                     rand_region=rand_region)
    env = procgen_wrappers.VecExtractDictObs(env, "rgb")
    if normalize_rew:
        env = procgen_wrappers.VecNormalize(env, ob=False)  # normalizing returns, but not
        # the img frames
    env = procgen_wrappers.TransposeFrame(env)
    env = procgen_wrappers.ScaledFloatFrame(env)
    if strong_policy is not None and weak_policy is not None:
        env = procgen_wrappers.HelpEnvWrapper(env, weak_policy, strong_policy, strong_query_cost, switching_agent_cost,
                                              reward_max, timeout, help_policy_type, device)
    if get_configs:
        obs_shape = env.observation_space.shape
        action_size = env.action_space.n
        env.close()
        return obs_shape, action_size
    return env


def load_model(agent, model_file, frozen=False):
    print("Loading agent from %s" % model_file)
    checkpoint = torch.load(model_file)
    agent.policy.load_state_dict(checkpoint["model_state_dict"])
    if not frozen:
        agent.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return agent


def load_policy(obs_size, action_size, model_file, device):
    model = ImpalaModel(in_channels=obs_size, benchmark='procgen')
    policy = CategoricalPolicy(embedder=model, action_size=action_size)
    policy.to(device)
    policy.eval()
    agent = PPOFrozen(policy, device)
    agent = load_model(agent, model_file, frozen=True)
    return agent


def procgen_define_help_policy(env, weak_agent, help_policy_type, device):
    action_size = 2
    model, _ = model_setup(env) if help_policy_type != "T3" else (None, None)
    hidden_size = weak_agent.policy.embedder.output_dim if help_policy_type != "T1" else 0
    policy = CategoricalPolicy(embedder=model, action_size=action_size, additional_hidden_dim=hidden_size)
    policy.to(device)
    return model, policy


def model_setup(env):
    in_channels = env.observation_space.shape[0]
    model = ImpalaModel(in_channels=in_channels, benchmark='procgen')
    action_size = env.action_space.n
    return model, action_size

