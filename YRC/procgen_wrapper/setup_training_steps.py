import os
import random
import uuid

import wandb
import yaml

from logger import Logger
from models import CategoricalPolicy, ImpalaModel, PPO
from procgen import ProcgenEnv
from procgen_wrappers import *
from utils import set_global_seeds, get_latest_model


def hyperparam_setup(args):
    print('::[LOGGING]::INITIALIZING PARAMS & HYPERPARAMETERS...')
    args.val_env_name = args.val_env_name if args.val_env_name else args.env_name
    args.start_level_val = random.randint(0, 9999)
    set_global_seeds(args.seed)
    if args.start_level == args.start_level_val:
        raise ValueError("Seeds for training and validation envs are equal.")
    with open(f'./hyperparams/{args.config_path}', 'r') as f:
        hyperparameters = yaml.safe_load(f)[args.param_name]
    args.n_envs = hyperparameters.get('n_envs', 256)
    args.n_steps = hyperparameters.get('n_steps', 256)
    args.device = torch.device(args.device)
    return args, hyperparameters


def logger_setup(args):
    print('::[LOGGING]::INITIALIZING LOGGER...')
    uuid_stamp = str(uuid.uuid4())[:8]
    run_name = f"PPO-procgen-{args.env_name}-{uuid_stamp}"
    logdir = os.path.join('logs', 'train', args.env_name)
    if not (os.path.exists(logdir)):
        os.makedirs(logdir)
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
    wb_resume = "allow" if args.model_file is None else "must"
    wandb.init(config=cfg, resume=wb_resume, project="YRC", name=run_name)
    logger = Logger(args.n_envs, logdir)
    for key, value in cfg.items():
        print(f"{key} : {value}")
    return args, logger


def create_env(args, is_valid=False):
    print('::[LOGGING]::INITIALIZING ENVIRONMENTS...')
    env = ProcgenEnv(num_envs=args.n_steps,
                     env_name=args.val_env_name if is_valid else args.env_name,
                     num_levels=0 if is_valid else args.num_levels,
                     start_level=args.start_level_val if is_valid else args.start_level,
                     distribution_mode=args.distribution_mode,
                     num_threads=args.num_threads,
                     random_percent=args.random_percent,
                     step_penalty=args.step_penalty,
                     key_penalty=args.key_penalty,
                     rand_region=args.rand_region)
    env = VecExtractDictObs(env, "rgb")
    if args.normalize_rew:
        env = VecNormalize(env, ob=False)  # normalizing returns, but not
        # the img frames
    env = TransposeFrame(env)
    env = ScaledFloatFrame(env)
    return env


def load_model(env, env_valid, model_file, policy, device):
    agent = PPO(env, policy, device, env_valid=env_valid)
    print("Loading agent from %s" % model_file)
    checkpoint = torch.load(model_file)
    agent.policy.load_state_dict(checkpoint["model_state_dict"])
    agent.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return agent


def model_setup(env, env_valid, configs):
    observation_shape = env.observation_space.shape
    in_channels = observation_shape[0]
    action_space = env.action_space

    model = ImpalaModel(in_channels=in_channels)

    recurrent = configs.get('recurrent', False)
    action_size = action_space.n
    policy = CategoricalPolicy(model, recurrent, action_size)
    policy.to(configs['device'])

    weak_agent = load_model(env, env_valid, configs['weak_model_file'], policy, configs['device'])
    oracle_agent = load_model(env, env_valid, configs['oracle_model_file'], policy, configs['device'])

    return weak_agent, oracle_agent


def agent_setup(env, env_valid, policy, logger, storage, storage_valid, device, num_checkpoints, model_file, hyperparameters):
    print('::[LOGGING]::INTIALIZING AGENT...')
    agent = PPO(env, policy, logger, storage, device,
                num_checkpoints,
                env_valid=env_valid,
                storage_valid=storage_valid,
                **hyperparameters)
    if model_file is not None:
        print("Loading agent from %s" % model_file)
        checkpoint = torch.load(model_file)
        agent.policy.load_state_dict(checkpoint["model_state_dict"])
        agent.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return agent
