import argparse
import os
import random
import time
import uuid
from typing import Tuple

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import wandb
from procgen import ProcgenEnv
from torch.utils.tensorboard import SummaryWriter

from models import PPOAgent
from utils import ReplayBuffer


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_name", type=str, default="ppo")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--torch_deterministic", action="store_false")
    parser.add_argument("--cuda", action="store_false")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb_project_name", type=str, default="RYC")
    parser.add_argument("--wandb_entity", type=str, default=None)
    parser.add_argument("--capture_video", action="store_true")
    parser.add_argument("--env_id", type=str, default="starpilot")
    parser.add_argument("--total_timesteps", type=int, default=int(25e6))
    parser.add_argument("--learning_rate", type=float, default=5e-4)
    parser.add_argument("--num_envs", type=int, default=64)
    parser.add_argument("--num_steps", type=int, default=256)
    parser.add_argument("--gamma", type=float, default=0.999)
    parser.add_argument("--gae_lambda", type=float, default=0.95)
    parser.add_argument("--num_minibatches", type=int, default=8)
    parser.add_argument("--update_epochs", type=int, default=3)
    parser.add_argument("--eval_episodes", type=int, default=10)
    parser.add_argument("--clip_coef", type=float, default=0.2)
    parser.add_argument("--ent_coef", type=float, default=0.01)
    parser.add_argument("--vf_coef", type=float, default=0.5)
    parser.add_argument("--max_grad_norm", type=float, default=0.5)
    return parser.parse_args()


def tracker(args: argparse.Namespace, name: str) -> SummaryWriter:
    if args.wandb:
        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=name,
            monitor_gym=True,
            save_code=True,
        )
    summary_writer = SummaryWriter(name)
    summary_writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{k}|{v}|" for k, v in vars(args).items()])),
    )
    return summary_writer


def set_seed(seed: int, torch_deterministic: bool, cuda: bool) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available() and cuda:
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = torch_deterministic
    torch.backends.cudnn.benchmark = not torch_deterministic


def env_setup(num_envs: int, env_id: str, gamma: float, capture_video: bool, name: str) -> ProcgenEnv:
    procgen_envs = ProcgenEnv(num_envs=num_envs, env_name=env_id, num_levels=0, start_level=0, distribution_mode="easy")

    # TODO: check for reward range for each environment
    procgen_envs.reward_range = (-float('inf'), float('inf'))

    procgen_envs = gym.wrappers.TransformObservation(procgen_envs, lambda observation: observation["rgb"])
    procgen_envs.single_action_space = procgen_envs.action_space
    procgen_envs.single_observation_space = procgen_envs.observation_space["rgb"]
    procgen_envs.is_vector_env = True
    procgen_envs = gym.wrappers.RecordEpisodeStatistics(procgen_envs)
    if capture_video:
        procgen_envs = gym.wrappers.RecordVideo(procgen_envs, f"videos/{name}")
    procgen_envs = gym.wrappers.NormalizeReward(procgen_envs, gamma=gamma)
    procgen_envs = gym.wrappers.TransformReward(procgen_envs, lambda reward: np.clip(reward, -10, 10))
    assert isinstance(procgen_envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"
    return procgen_envs


def train_ppo(rl_agent: PPOAgent, observations: torch.Tensor, actions: torch.Tensor, log_probs: torch.Tensor, inds: np.ndarray, returns: torch.Tensor,
              values: torch.Tensor, optimizer: optim.Optimizer, b_advantages: torch.Tensor, **kwargs) \
        -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    _, new_logprob, entropy, new_value = rl_agent.get_action_and_value(observations, actions.long())
    logratio = new_logprob - log_probs
    ratio = logratio.exp()

    with torch.no_grad():
        # calculate approx_kl http://joschu.net/blog/kl-approx.html
        old_approx_kl = (-logratio).mean()
        approx_kl = ((ratio - 1) - logratio).mean()

    mb_advantages = b_advantages[inds]
    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

    # Policy loss
    pg_loss1 = -mb_advantages * ratio
    pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - kwargs["clip_coef"], 1 + kwargs["clip_coef"])
    pg_loss = torch.max(pg_loss1, pg_loss2).mean()

    # Value loss
    new_value = new_value.view(-1)
    v_loss_unclipped = (new_value - returns[inds]) ** 2
    v_clipped = values + torch.clamp(
        new_value - values,
        -kwargs["clip_coef"],
        kwargs["clip_coef"],
    )
    v_loss_clipped = (v_clipped - returns[inds]) ** 2
    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
    v_loss = 0.5 * v_loss_max.mean()

    ent_loss = entropy.mean()
    loss = pg_loss - kwargs["ent_coef"] * ent_loss + v_loss * kwargs["vf_coef"]

    optimizer.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(rl_agent.parameters(), kwargs["max_grad_norm"])
    optimizer.step()
    return approx_kl, v_loss, pg_loss, ent_loss, old_approx_kl


def record_stats(summary_writer: SummaryWriter, global_step: int, **kwargs) -> None:
    for k, v in kwargs.items():
        if isinstance(v, torch.Tensor):
            v = v.item()
        summary_writer.add_scalar(f"charts/{k}", v, global_step)


def bootstrap_value(observation: torch.Tensor, rb: ReplayBuffer, rl_agent: PPOAgent, device: torch.device, next_done: torch.Tensor, num_steps: int,
                    gamma: float, gae_lambda: float) -> torch.Tensor:
    with torch.no_grad():
        next_value = rl_agent.get_value(observation).reshape(1, -1)
        advantages = torch.zeros_like(rb._rewards).to(device)
        lastgaelam = 0
        for t in reversed(range(num_steps)):
            if t == num_steps - 1:
                next_nonterminal = 1.0 - next_done
                next_values = next_value
            else:
                next_nonterminal = 1.0 - rb._dones[t + 1]
                next_values = rb._values[t + 1]
            delta = rb._rewards[t] + gamma * next_values * next_nonterminal - rb._values[t]
            advantages[t] = lastgaelam = delta + gamma * gae_lambda * next_nonterminal * lastgaelam
    return advantages


def evaluate(agent: PPOAgent, env_id: str, eval_episodes: int, run_name: str, device: torch.device, capture_video: bool, gamma: float) -> np.array:
    envs = env_setup(1, env_id, gamma, capture_video, run_name)
    agent.eval()
    obs = torch.Tensor(envs.reset()).to(device)
    episodic_returns = []
    while len(episodic_returns) < eval_episodes:
        actions, _, _, _ = agent.get_action_and_value(torch.Tensor(obs).to(device))
        next_obs, reward, done, info = envs.step(actions.cpu().numpy())
        reward, done, info = reward[0], done[0], info[0]
        if done:
            print(f"eval_episode={len(episodic_returns)}, episodic_return={info['episode']['r']}")
            episodic_returns += [info["episode"]["r"]]
        obs = next_obs
    agent.train()
    print(f"mean eval perforamnce: {np.mean(episodic_returns)}")
    return np.array(episodic_returns)


def variable_initialization(args: argparse.Namespace) -> Tuple[int, int, int, str, str, float, SummaryWriter, torch.device]:
    b_size = int(args.num_envs * args.num_steps)
    num_iter = args.total_timesteps // b_size
    mini_b_size = b_size // args.num_minibatches
    uuid_code = str(uuid.uuid4())[:8]
    run = f"./runs/{args.env_id}/{args.exp_name}/{uuid_code}"
    if not os.path.exists(run):
        os.makedirs(run)
    model_dir = f"{run}/best.pt"
    best_p = -np.infty
    summary_writer = tracker(args, run)
    set_seed(args.seed, args.torch_deterministic, args.cuda)
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    return batch_size, num_iter, mini_b_size, run, model_dir, best_p, summary_writer, device


if __name__ == "__main__":
    input_args = get_args()
    batch_size, num_iterations, minibatch_size, run_name, model_path, best_performance, writer, device = variable_initialization(input_args)

    envs = env_setup(input_args.num_envs, input_args.env_id, input_args.gamma, input_args.capture_video, run_name)
    agent = PPOAgent(envs).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=input_args.learning_rate, eps=1e-5)

    # ALGO Logic: Storage setup
    replay_buffer = ReplayBuffer(input_args.num_steps, input_args.num_envs, envs.single_observation_space, envs.single_action_space, device)

    # TRY NOT TO MODIFY: start the game
    global_step = 0
    start_time = time.time()
    obs = torch.Tensor(envs.reset()).to(device)
    dones = torch.zeros(input_args.num_envs).to(device)

    ppo_kwargs = {
        "clip_coef": input_args.clip_coef,
        "ent_coef": input_args.ent_coef,
        "vf_coef": input_args.vf_coef,
        "max_grad_norm": input_args.max_grad_norm,
    }

    for iteration in range(1, num_iterations + 1):
        for step in range(0, input_args.num_steps):
            global_step += input_args.num_envs

            # ALGO LOGIC: action logic
            with torch.no_grad():
                actions, logprobs, _, value = agent.get_action_and_value(obs)
                values = value.flatten()

            # TRY NOT TO MODIFY: execute the game and log data.
            next_obs, rewards, dones, info = envs.step(actions.cpu().numpy())
            rewards = torch.tensor(rewards).to(device).view(-1)
            next_obs = torch.Tensor(next_obs).to(device)
            dones = torch.Tensor(dones).to(device)

            replay_buffer.add_transition(obs, actions, rewards, next_obs, dones, values, logprobs)

            obs = torch.Tensor(next_obs).to(device)
            for item in info:
                if "episode" in item.keys():
                    writer.add_scalar("charts/episodic_return", item["episode"]["r"], global_step)
                    writer.add_scalar("charts/episodic_length", item["episode"]["l"], global_step)
                    break

        # bootstrap value if not done
        advantages = bootstrap_value(obs, replay_buffer, agent, device, dones, input_args.num_steps, input_args.gamma, input_args.gae_lambda)
        returns = advantages + values
        b_returns = returns.reshape(-1)

        # Optimizing the policy and value network
        b_inds = np.arange(batch_size)
        for epoch in range(input_args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, batch_size, minibatch_size):
                end = start + minibatch_size
                mb_inds = b_inds[start:end]
                rb_obs, rb_actions, rb_logprobs, rb_values = replay_buffer.sample_by_indices(mb_inds, envs.single_observation_space.shape,
                                                                                             envs.single_action_space.shape)

                kl, value_loss, policy_loss, entropy_loss, old_kl = train_ppo(agent, rb_obs, rb_actions, rb_logprobs, mb_inds, b_returns, rb_values,
                                                                              optimizer, advantages.reshape(-1), **ppo_kwargs)
        explained_var = torch.nan if torch.var(b_returns) == 0 else 1 - torch.var(b_returns - replay_buffer._values.reshape(-1)) / torch.var(b_returns)
        record_kwargs = {
            "value_loss": value_loss,
            "policy_loss": policy_loss,
            "entropy_loss": entropy_loss,
            "old_approx_kl": old_kl,
            "approx_kl": kl,
            "explained_variance": explained_var
        }
        record_stats(writer, global_step, **record_kwargs)

        episodic_returns = evaluate(agent, input_args.env_id, input_args.eval_episodes, f"{run_name}-eval", device, input_args.capture_video,
                                    input_args.gamma)
        if episodic_returns.mean() > best_performance:
            torch.save(agent.state_dict(), model_path)
            print(f"best model saved to {model_path} - performance: {episodic_returns.mean()}")
        for idx, episodic_return in enumerate(episodic_returns):
            writer.add_scalar("eval/episodic_return", episodic_return, idx)

    envs.close()
    writer.close()
