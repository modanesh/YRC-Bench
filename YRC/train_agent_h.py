import argparse
import os
import random
import uuid

import torch
import wandb
import yaml

from procgen_wrapper import setup_training_steps as procgen_setup
from procgen_wrapper import utils, logger


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name', type=str, default='test', help='experiment name')
    parser.add_argument('--env_name', type=str, default='coinrun', help='environment ID')
    parser.add_argument('--val_env_name', type=str, default=None, help='optional validation environment ID')
    parser.add_argument('--start_level', type=int, default=int(0), help='start-level for environment')
    parser.add_argument('--num_levels', type=int, default=int(0), help='number of training levels for environment')
    parser.add_argument('--distribution_mode', type=str, default='easy', help='distribution mode for environment')
    parser.add_argument('--param_name', type=str, default='easy-200', help='hyper-parameter ID')
    parser.add_argument('--device', type=str, default='cuda', required=False, help='whether to use gpu')
    parser.add_argument('--num_timesteps', type=int, default=int(25000000), help='number of training timesteps')
    parser.add_argument('--seed', type=int, default=0, help='Random generator seed')
    parser.add_argument('--log_level', type=int, default=int(40), help='[10,20,30,40]')
    parser.add_argument('--num_checkpoints', type=int, default=int(1), help='number of checkpoints to store')
    parser.add_argument('--weak_model_file', type=str, required=True)
    parser.add_argument('--oracle_model_file', type=str, required=True)
    parser.add_argument('--random_percent', type=int, default=0, help='COINRUN: percent of environments in which coin is randomized (only for coinrun)')
    parser.add_argument('--key_penalty', type=int, default=0, help='HEIST_AISC: Penalty for picking up keys (divided by 10)')
    parser.add_argument('--step_penalty', type=int, default=0, help='HEIST_AISC: Time penalty per step (divided by 1000)')
    parser.add_argument('--rand_region', type=int, default=0, help='MAZE: size of region (in upper left corner) in which goal is sampled.')
    parser.add_argument('--config_path', type=str, default='config.yml', help='path to hyperparameter config yaml')
    parser.add_argument('--num_threads', type=int, default=8)
    return parser.parse_args()


def config_merger(cfgs, config_path):
    with open(os.path.join(config_path, 'config.yml')) as f:
        specific_cfgs = yaml.load(f, Loader=yaml.FullLoader)[cfgs.param_name]
    cfgs.val_env_name = cfgs.val_env_name if cfgs.val_env_name else cfgs.env_name
    cfgs.start_level_val = random.randint(0, 9999)
    utils.set_global_seeds(cfgs.seed)
    if cfgs.start_level == cfgs.start_level_val:
        raise ValueError("Seeds for training and validation envs are equal.")
    cfgs.device = torch.device(cfgs.device)
    cfgs.update(specific_cfgs)
    return cfgs


def load_task(cfgs):
    env = procgen_setup.create_env(cfgs)
    env_valid = procgen_setup.create_env(cfgs, is_valid=True)
    return env, env_valid


def logger_setup(cfgs):
    uuid_stamp = str(uuid.uuid4())[:8]
    run_name = f"PPO-procgen-help-{cfgs.env_name}-w{cfgs.weak_model_file}-o{cfgs.oracle_model_file}-{uuid_stamp}"
    logdir = os.path.join('logs', 'train', cfgs.env_name)
    if not (os.path.exists(logdir)):
        os.makedirs(logdir)
    logdir = os.path.join(logdir, run_name)
    if not (os.path.exists(logdir)):
        os.mkdir(logdir)
    print(f'Logging to {logdir}')
    wb_resume = "allow" if cfgs.model_file is None else "must"
    wandb.init(config=vars(cfgs), resume=wb_resume, project="YRC", name=run_name)
    writer = logger.Logger(cfgs.n_envs, logdir)
    return writer


if __name__ == '__main__':
    configs = get_args()
    configs = config_merger(configs, './procgen_wrapper/configs')
    writer = logger_setup(configs)

    task, task_valid = load_task(configs)
    weak_agent, oracle_agent = procgen_setup.model_setup(task, task_valid, configs, trainable=False)

    model, policy = procgen_setup.model_setup(task, task_valid, configs, trainable=True, helper_policy=True)
    storage = utils.Storage(task.observation_space.shape, configs.n_steps, configs.n_envs, configs.device)
    storage_valid = utils.Storage(task.observation_space.shape, configs.n_steps, configs.n_envs, configs.device)
    agent = procgen_setup.agent_setup(task, task_valid, policy, writer, storage, storage_valid, configs.device, configs.num_checkpoints,
                                      configs.model_file, pi_w=weak_agent, pi_o=oracle_agent)

    agent.train(configs.num_timesteps, pi_h=True)
