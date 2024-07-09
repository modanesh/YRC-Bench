import csv
import os
import random
import time
import uuid
from collections import deque

import numpy as np
import pandas as pd
import torch
import wandb
import yaml
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler

from procgen import ProcgenEnv
from .models import CategoricalPolicyT1, CategoricalPolicyT2, CategoricalPolicyT3, ImpalaModel, PPO, PPOFrozen
from .procgen_wrappers import VecExtractDictObs, TransposeFrame, ScaledFloatFrame, VecNormalize, HelpEnvWrapper


def hyperparam_setup(args):
    print('::[LOGGING]::INITIALIZING PARAMS & HYPERPARAMETERS...')
    args.val_env_name = args.val_env_name if args.val_env_name else args.env_name
    args.start_level_val = random.randint(0, 9999)
    set_global_seeds(args.seed)
    if args.start_level == args.start_level_val:
        raise ValueError("Seeds for training and validation envs are equal.")
    with open(f'./configs/{args.config_path}', 'r') as f:
        hyperparameters = yaml.safe_load(f)[args.param_name]
    args.n_envs = hyperparameters.get('n_envs', 256)
    args.n_steps = hyperparameters.get('n_steps', 256)
    args.device = torch.device(args.device)
    return args, hyperparameters


def logger_setup(args, hyperparameters):
    print('::[LOGGING]::INITIALIZING LOGGER...')
    uuid_stamp = str(uuid.uuid4())[:8]
    run_name = f"PPO-procgen-{args.env_name}-{args.param_name}-{uuid_stamp}"
    logdir = os.path.join('logs', 'train', args.env_name)
    if not (os.path.exists(logdir)):
        os.mkdir(logdir)
    logdir_folders = [os.path.join(logdir, d) for d in os.listdir(logdir)]
    if args.model_file == "auto":  # try to figure out which file to load
        logdirs_with_model = [d for d in logdir_folders if any(['model' in filename for filename in os.listdir(d)])]
        if len(logdirs_with_model) > 1:
            raise ValueError("Received args.model_file = 'auto', but there are multiple experiments"
                             f" with saved models under experiment_name {args.exp_name}.")
        elif len(logdirs_with_model) == 0:
            raise ValueError("Received args.model_file = 'auto', but there are"
                             f" no saved models under experiment_name {args.exp_name}.")
        model_dir = logdirs_with_model[0]
        args.model_file = os.path.join(model_dir, get_latest_model(model_dir))
        logdir = model_dir  # reuse logdir
    else:
        logdir = os.path.join(logdir, run_name)
    if not (os.path.exists(logdir)):
        os.mkdir(logdir)

    print(f'Logging to {logdir}')
    cfg = vars(args)
    cfg.update(hyperparameters)

    wb_resume = "allow" if args.model_file is None else "must"
    wandb.init(config=cfg, resume=wb_resume, project="YRC", name=run_name, dir=os.getcwd().split("YRC")[0])
    logger = Logger(args.n_envs, logdir)
    for key, value in cfg.items():
        print(f"{key} : {value}")
    return args, logger


def create_env(n_steps, env_name, start_level, num_levels, distribution_mode, num_threads, random_percent, step_penalty, key_penalty, rand_region,
               normalize_rew, weak_policy=None, strong_policy=None, get_configs=False, strong_query_cost=0.0, switching_agent_cost=0.0, reward_max=1.0,
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
        env = HelpEnvWrapper(env, weak_policy, strong_policy, strong_query_cost, switching_agent_cost, reward_max, timeout)
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


def define_help_policy(env, weak_agent, help_policy_type, device):
    model, _ = model_setup(env)
    action_size = 2
    hidden_size = weak_agent.policy.embedder.output_dim
    softmax_size = weak_agent.policy.fc_policy.out_features
    if help_policy_type == "T1":
        policy = CategoricalPolicyT1(model, action_size)
    elif help_policy_type == "T2":
        policy = CategoricalPolicyT2(model, action_size, hidden_size, softmax_size)
    elif help_policy_type == "T3":
        policy = CategoricalPolicyT3(action_size, hidden_size)
    else:
        raise ValueError("Invalid help policy type.")
    policy.to(device)
    return model, policy


def define_policy(env, device):
    model, action_size = model_setup(env)
    policy = CategoricalPolicyT1(model, action_size)
    policy.to(device)
    return model, policy


def load_policy(obs_size, action_size, model_file, device):
    model = ImpalaModel(in_channels=obs_size)
    policy = CategoricalPolicyT1(model, action_size)
    policy.to(device)
    policy.eval()
    agent = PPOFrozen(policy, device)
    agent = load_model(agent, model_file, frozen=True)
    return agent


def model_setup(env):
    in_channels = env.observation_space.shape[0]
    model = ImpalaModel(in_channels=in_channels)
    action_size = env.action_space.n
    return model, action_size


def algorithm_setup(env, env_valid, policy, logger, storage, storage_valid, device, num_checkpoints, model_file, hyperparameters, pi_w=None, pi_o=None,
                    help_policy_type=None):
    print('::[LOGGING]::INTIALIZING AGENT...')
    agent = PPO(env, policy, logger, storage, device,
                num_checkpoints,
                env_valid=env_valid,
                storage_valid=storage_valid,
                pi_w=pi_w,
                pi_o=pi_o,
                help_policy_type=help_policy_type,
                **hyperparameters)
    if model_file is not None:
        agent = load_model(agent, model_file)
    return agent


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


def set_global_seeds(seed, torch_deterministic=True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = torch_deterministic
    torch.backends.cudnn.benchmark = not torch_deterministic


def get_latest_model(model_dir):
    """given model_dir with files named model_n.pth where n is an integer,
    return the filename with largest n"""
    steps = [int(filename[6:-4]) for filename in os.listdir(model_dir) if filename.startswith("model_")]
    return list(os.listdir(model_dir))[np.argmax(steps)]


class Logger(object):

    def __init__(self, n_envs, logdir):
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
                           "max_episode_len", "mean_episode_len", "min_episode_len",
                           "mean_timeouts"]  # collected for both train and val envs
        self.log = pd.DataFrame(columns=time_metrics + episode_metrics + \
                                        ["val_" + m for m in episode_metrics])

        self.timesteps = 0
        self.num_episodes = 0

    def feed(self, rew_batch, done_batch, rew_batch_v=None, done_batch_v=None):
        steps = rew_batch.shape[0]
        rew_batch = rew_batch.T
        done_batch = done_batch.T

        valid = rew_batch_v is not None and done_batch_v is not None
        if valid:
            rew_batch_v = rew_batch_v.T
            done_batch_v = done_batch_v.T

        for i in range(self.n_envs):
            for j in range(steps):
                self.episode_rewards[i].append(rew_batch[i][j])
                if valid:
                    self.episode_rewards_v[i].append(rew_batch_v[i][j])

                if done_batch[i][j]:
                    self.episode_timeout_buffer.append(1 if j == steps - 1 else 0)
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

    def dump(self):
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
        wandb.log({k: v for k, v in zip(self.log.columns, log)})

    def _get_episode_statistics(self):
        episode_statistics = {}
        episode_statistics['Rewards/max_episodes'] = np.max(self.episode_reward_buffer, initial=0)
        episode_statistics['Rewards/mean_episodes'] = np.mean(self.episode_reward_buffer)
        episode_statistics['Rewards/min_episodes'] = np.min(self.episode_reward_buffer, initial=0)
        episode_statistics['Len/max_episodes'] = np.max(self.episode_len_buffer, initial=0)
        episode_statistics['Len/mean_episodes'] = np.mean(self.episode_len_buffer)
        episode_statistics['Len/min_episodes'] = np.min(self.episode_len_buffer, initial=0)
        episode_statistics['Len/mean_timeout'] = np.mean(self.episode_timeout_buffer)

        # valid
        episode_statistics['[Valid] Rewards/max_episodes'] = np.max(self.episode_reward_buffer_v, initial=0)
        episode_statistics['[Valid] Rewards/mean_episodes'] = np.mean(self.episode_reward_buffer_v)
        episode_statistics['[Valid] Rewards/min_episodes'] = np.min(self.episode_reward_buffer_v, initial=0)
        episode_statistics['[Valid] Len/max_episodes'] = np.max(self.episode_len_buffer_v, initial=0)
        episode_statistics['[Valid] Len/mean_episodes'] = np.mean(self.episode_len_buffer_v)
        episode_statistics['[Valid] Len/min_episodes'] = np.min(self.episode_len_buffer_v, initial=0)
        episode_statistics['[Valid] Len/mean_timeout'] = np.mean(self.episode_timeout_buffer_v)
        return episode_statistics


def logger_setup(cfgs):
    uuid_stamp = str(uuid.uuid4())[:8]
    run_name = f"PPO-procgen-help-{cfgs.env_name}-type{cfgs.help_policy_type}-{uuid_stamp}"
    logdir = os.path.join('logs', 'train', cfgs.env_name)
    if not (os.path.exists(logdir)):
        os.makedirs(logdir)
    logdir = os.path.join(logdir, run_name)
    if not (os.path.exists(logdir)):
        os.mkdir(logdir)
    print(f'Logging to {logdir}')
    wandb.init(config=vars(cfgs), resume="allow", project="YRC", name=run_name)
    writer = Logger(cfgs.n_envs, logdir)
    return writer
