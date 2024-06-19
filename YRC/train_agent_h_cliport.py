import argparse
import os
import random

import yaml

import wandb
import uuid
import torch
from cliport_wrapper import utils, logger
from cliport.environments.environment import Environment
from cliport_wrapper import setup_training_steps as cliport_setup
from cliport import tasks


def logger_setup(cfgs):
    uuid_stamp = str(uuid.uuid4())[:8]
    # run_name = f"PPO-procgen-help-{cfgs.env_name}-w{cfgs.weak_model_file}-o{cfgs.oracle_model_file}-{uuid_stamp}"
    run_name = f"PPO-procgen-help-{cfgs.env_name}-{uuid_stamp}"
    logdir = os.path.join('logs', 'train', cfgs.env_name)
    if not (os.path.exists(logdir)):
        os.makedirs(logdir)
    logdir = os.path.join(logdir, run_name)
    if not (os.path.exists(logdir)):
        os.mkdir(logdir)
    print(f'Logging to {logdir}')
    wb_resume = "allow"
    wandb.init(config=vars(cfgs), resume=wb_resume, project="YRC", name=run_name)
    writer = logger.Logger(cfgs.n_envs, logdir)
    return writer


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name', type=str, default='test', help='experiment name')
    parser.add_argument('--env_name', type=str, default='coinrun', help='environment ID')
    parser.add_argument('--weak_model_file', type=str, required=True)
    parser.add_argument('--config_path', type=str, default='./cliport_wrapper/configs', help='path to hyperparameter config yaml')
    parser.add_argument('--switching_cost', type=float, default=0.2)
    parser.add_argument('--oracle_cost', type=float, default=0.8)
    return parser.parse_args()


def config_merger(cfgs):
    with open(os.path.join(cfgs.config_path, 'reward_range.yml')) as f:
        env_cfgs = yaml.load(f, Loader=yaml.FullLoader)[cfgs.env_name.split("_")[0]][cfgs.distribution_mode]
    cfgs.val_env_name = cfgs.val_env_name if cfgs.val_env_name else cfgs.env_name
    cfgs.start_level_val = random.randint(0, 9999)
    utils.set_global_seeds(cfgs.seed)
    if cfgs.start_level == cfgs.start_level_val:
        raise ValueError("Seeds for training and validation envs are equal.")
    cfgs.device = torch.device(cfgs.device)
    return cfgs, env_cfgs


if __name__ == '__main__':
    configs = get_args()
    if configs.oracle_cost < 0 or configs.switching_cost < 0 or configs.oracle_cost + configs.switching_cost > 1:
        raise ValueError("Invalid values for switching_cost and oracle_cost. Please ensure that they are positive and "
                         "their sum is less than 1.")
    configs, env_info = config_merger(configs)
    writer = logger_setup(configs)

    env = Environment(
        configs['assets_root'],
        disp=configs['disp'],
        shared_memory=configs['shared_memory'],
        hz=480,
        record_cfg=configs['record']
    )
    task = tasks.names[configs['task']]()
    task.mode = configs['mode']

    # load oracle agent
    oracle_agent = task.oracle(env)

    # load weak agent
    checkpoint_path = os.path.join(configs['train']['train_dir'], 'checkpoints')
    last_checkpoint_path = os.path.join(checkpoint_path, 'last.ckpt')
    last_checkpoint = last_checkpoint_path if os.path.exists(last_checkpoint_path) and configs['train']['load_from_last_ckpt'] else None
    weak_agent = torch.load(last_checkpoint)

    # define the helper policy
    env.set_task(task)
    model, policy = cliport_setup.model_setup(task, configs)
    storage = utils.Storage(task.observation_space.shape, configs.n_steps, configs.n_envs, configs.device)
    storage_valid = utils.Storage(task.observation_space.shape, configs.n_steps, configs.n_envs, configs.device)

    agent = cliport_setup.agent_setup(env, policy, writer, storage, storage_valid, configs.device,
                                      configs.num_checkpoints,
                                      configs.model_file, pi_w=weak_agent, pi_o=oracle_agent)

    agent.train(configs.num_timesteps)