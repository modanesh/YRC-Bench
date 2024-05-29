import os
import random

import numpy as np
import torch
from libero.libero import benchmark, get_libero_path, envs
from models import Agent


def set_seeds(seed, torch_deterministic=True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = torch_deterministic
    torch.backends.cudnn.benchmark = not torch_deterministic


def extract_image_obs(obs):
    extracted_obs = np.concatenate((obs['agentview_image'], obs['robot0_eye_in_hand_image']), axis=0)
    return extracted_obs


def extract_vector_obs(obs):
    extracted_obs = []
    for key, value in obs.items():
        if key.__contains__("joint") or key.__contains__("gripper") or key in ["robot0_eef_pos", "robot0_eef_quat"]:
            extracted_obs.extend(value)
    return np.array(extracted_obs)


def evaluate(model_path, task_suite_name, task_id, camera_heights, camera_widths, eval_episodes, device):
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[task_suite_name]()

    task = task_suite.get_task(task_id)
    task_bddl_file = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)

    env_args = {
        "bddl_file_name": task_bddl_file,
        "camera_heights": camera_heights,
        "camera_widths": camera_widths
    }
    env = envs.OffScreenRenderEnv(**env_args)
    full_obs = env.reset()
    image_obs_shape = extract_image_obs(full_obs).shape
    vector_obs_shape = extract_vector_obs(full_obs).shape
    action_shape = env.env.action_dim

    agent = Agent(image_obs_shape, vector_obs_shape, action_shape).to(device)
    agent.load_state_dict(torch.load(model_path, map_location=device))
    agent.eval()

    episodic_returns = []
    horizon = 1000
    while len(episodic_returns) < eval_episodes:
        ep_step = 0
        done = False
        ep_return = 0
        next_full_obs = env.reset()
        next_image_obs = torch.Tensor(extract_image_obs(next_full_obs)).to(device)
        next_vector_obs = torch.Tensor(extract_vector_obs(next_full_obs)).to(device)
        while not done and ep_step < horizon:
            action, _, _, _ = agent.get_action_and_value(next_image_obs, next_vector_obs)
            next_full_obs, reward, done, infos = env.step(action.cpu().numpy().flatten())
            next_image_obs = torch.Tensor(extract_image_obs(next_full_obs)).to(device)
            next_vector_obs = torch.Tensor(extract_vector_obs(next_full_obs)).to(device)
            ep_return += reward
            ep_step += 1
        episodic_returns.append(ep_return)
    return episodic_returns









