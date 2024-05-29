import argparse
import os
import uuid

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import wandb

from libero.libero import benchmark, get_libero_path, envs
from models import Agent
from utils import extract_image_obs, extract_vector_obs, set_seeds, evaluate
from torch.utils.tensorboard import SummaryWriter


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task_suite_name', type=str, required=True, choices=["libero_10", "libero_spatial", "libero_object", "libero_goal"])
    parser.add_argument('--task_id', type=int, required=True)
    parser.add_argument('--init_state_id', type=int, default=0)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument("--camera_heights", type=int, default=128)
    parser.add_argument("--camera_widths", type=int, default=128)
    parser.add_argument("--cuda", action="store_true")
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--num_steps", type=int, default=2048)
    parser.add_argument("--num_iterations", type=int, default=1_000)
    parser.add_argument("--total_timesteps", type=int, default=1_000_000)
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae_lambda", type=float, default=0.95)
    parser.add_argument("--update_epochs", type=int, default=10)
    parser.add_argument("--clip_coef", type=float, default=0.2)
    parser.add_argument("--ent_coef", type=float, default=0.01)
    parser.add_argument("--vf_coef", type=float, default=0.5)
    parser.add_argument("--max_grad_norm", type=float, default=0.5)
    parser.add_argument("--model_file", type=str)
    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    set_seeds(args.seed)
    uuid_stamp = str(uuid.uuid4())[:8]
    wb_resume = "allow" if args.model_file is None else "must"
    run_name = f"PPO-LIBERO-{args.task_suite_name}-{args.task_id}-{uuid_stamp}"
    wandb.init(config=vars(args), resume=wb_resume, project="YRC", name=run_name)
    writer = SummaryWriter(f"logs/{run_name}")

    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[args.task_suite_name]()

    # retrieve a specific task
    task = task_suite.get_task(args.task_id)
    task_name = task.name
    task_description = task.language
    task_bddl_file = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
    print(f"[info] retrieving task {args.task_id} from suite {args.task_suite_name}, the " + \
          f"language instruction is {task_description}, and the bddl file is {task_bddl_file}")

    # step over the environment
    env_args = {
        "bddl_file_name": task_bddl_file,
        "camera_heights": args.camera_heights,
        "camera_widths": args.camera_widths
    }
    env = envs.OffScreenRenderEnv(**env_args)




    full_obs = env.reset()
    image_obs_shape = extract_image_obs(full_obs).shape
    vector_obs_shape = extract_vector_obs(full_obs).shape
    action_shape = env.env.action_dim
    init_states = task_suite.get_task_init_states(args.task_id)
    env.set_init_state(init_states[args.init_state_id])

    agent = Agent(image_obs_shape, vector_obs_shape, action_shape).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    image_obs = torch.zeros((args.num_steps,) + image_obs_shape).to(device)
    vector_obs = torch.zeros((args.num_steps,) + vector_obs_shape).to(device)
    actions = torch.zeros(args.num_steps, action_shape).to(device)
    logprobs = torch.zeros(args.num_steps, ).to(device)
    rewards = torch.zeros(args.num_steps, ).to(device)
    dones = torch.zeros(args.num_steps, ).to(device)
    values = torch.zeros(args.num_steps, ).to(device)

    global_step = 0
    next_full_obs = env.reset()
    next_image_obs = torch.Tensor(extract_image_obs(next_full_obs)).to(device)
    next_vector_obs = torch.Tensor(extract_vector_obs(next_full_obs)).to(device)
    next_done = 0

    for iteration in range(1, args.num_iterations + 1):
        for step in range(0, args.num_steps):
            global_step += 1
            image_obs[step] = next_image_obs
            vector_obs[step] = next_vector_obs
            dones[step] = next_done

            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_image_obs, next_vector_obs)
                values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob

            next_full_obs, reward, next_done, infos = env.step(action.cpu().numpy().flatten())
            next_image_obs = torch.Tensor(extract_image_obs(next_full_obs)).to(device)
            next_vector_obs = torch.Tensor(extract_vector_obs(next_full_obs)).to(device)
            rewards[step] = torch.tensor(reward).to(device).view(-1)
            next_done = 1 if next_done else 0
            next_image_obs, next_vector_obs = torch.Tensor(next_image_obs).to(device), torch.Tensor(next_vector_obs).to(device)

            if "final_info" in infos:
                for info in infos["final_info"]:
                    if info and "episode" in info:
                        print(f"global_step={global_step}, episodic_return={info['episode']['r']}")
                        writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
                        writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)

        # bootstrap value if not done
        with torch.no_grad():
            next_value = agent.get_value(next_image_obs, next_vector_obs).reshape(1, -1)
            advantages = torch.zeros_like(rewards).to(device)
            lastgaelam = 0
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
                advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
            returns = advantages + values

        # Optimizing the policy and value network
        b_inds = np.arange(min(args.batch_size, args.num_steps))
        clipfracs = []
        minibatch_size = min(args.batch_size, args.num_steps) // 16
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, min(args.batch_size, args.num_steps), minibatch_size):
                end = start + minibatch_size
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue = agent.get_action_and_value(image_obs[mb_inds], vector_obs[mb_inds], actions[mb_inds])
                logratio = newlogprob - logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]

                mb_advantages = advantages[mb_inds]
                mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                newvalue = newvalue.view(-1)
                v_loss_unclipped = (newvalue - returns[mb_inds]) ** 2
                v_clipped = values[mb_inds] + torch.clamp(
                    newvalue - values[mb_inds],
                    -args.clip_coef,
                    args.clip_coef,
                )
                v_loss_clipped = (v_clipped - returns[mb_inds]) ** 2
                v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                v_loss = 0.5 * v_loss_max.mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

        y_pred, y_true = values.cpu().numpy(), returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
        writer.add_scalar("losses/explained_variance", explained_var, global_step)

        model_path = f"logs/{run_name}/model.pt"
        torch.save(agent.state_dict(), model_path)
        print(f"model saved to {model_path}")

        episodic_returns = evaluate(
            model_path,
            args.task_suite_name,
            args.task_id,
            args.camera_heights,
            args.camera_widths,
            eval_episodes=10,
            device=device,
        )
        for idx, episodic_return in enumerate(episodic_returns):
            writer.add_scalar("eval/episodic_return", episodic_return, idx)

    writer.close()