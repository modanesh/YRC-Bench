from collections import deque

import numpy as np
import torch
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler

from YRC.core.models import CategoricalPolicyT1, CategoricalPolicyT2, CategoricalPolicyT3, ImpalaModel
from procgen import ProcgenEnv
from .models import PPOFrozen
from .procgen_wrappers import VecExtractDictObs, TransposeFrame, ScaledFloatFrame, VecNormalize, HelpEnvWrapper


class ReplayBuffer:
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
        return rew_batch, done_batch


def environment_setup(n_steps, env_name, start_level, num_levels, distribution_mode, num_threads, random_percent,
                      step_penalty, key_penalty, rand_region,
                      normalize_rew, weak_policy=None, strong_policy=None, get_configs=False, strong_query_cost=0.0,
                      switching_agent_cost=0.0, reward_max=1.0,
                      timeout=1000):
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
    env = VecExtractDictObs(env, "rgb")
    if normalize_rew:
        env = VecNormalize(env, ob=False)  # normalizing returns, but not
        # the img frames
    env = TransposeFrame(env)
    env = ScaledFloatFrame(env)
    if strong_policy is not None and weak_policy is not None:
        env = HelpEnvWrapper(env, weak_policy, strong_policy, strong_query_cost, switching_agent_cost, reward_max,
                             timeout)
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
    model = ImpalaModel(in_channels=obs_size)
    policy = CategoricalPolicyT1(model, action_size)
    policy.to(device)
    policy.eval()
    agent = PPOFrozen(policy, device)
    agent = load_model(agent, model_file, frozen=True)
    return agent


def define_help_policy(env, weak_agent, help_policy_type, device):
    in_channels = env.observation_space.shape[0]
    model = ImpalaModel(in_channels=in_channels)
    action_size = 2
    hidden_size = weak_agent.policy.embedder.output_dim
    if help_policy_type == "T1":
        policy = CategoricalPolicyT1(model, action_size)
    elif help_policy_type == "T2":
        policy = CategoricalPolicyT2(model, action_size, hidden_size)
    elif help_policy_type == "T3":
        policy = CategoricalPolicyT3(action_size, hidden_size)
    policy.to(device)
    return model, policy
