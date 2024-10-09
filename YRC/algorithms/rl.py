import numpy as np

import torch

from YRC.core.configs.global_configs import get_global_variable
from YRC.core import Algorithm


class PPOAlgorithm(Algorithm):
    def __init__(self, config, env):
        self.args = config
        self.args.num_envs = env.num_envs
        self.obs_shape = env.obs_shape
        self.action_shape = env.action_shape

    def init(self, policy, envs):
        args = self.args
        args.batch_size = int(args.num_envs * args.num_steps)
        args.minibatch_size = int(args.batch_size // args.num_minibatches)
        args.num_iterations = args.total_timesteps // args.batch_size

        device = get_global_variable("device")
        # Initialize all tensors
        if isinstance(self.obs_shape, dict):
            self.obs = {}
            for k, shape in self.obs_shape.items():
                self.obs[k] = torch.zeros((args.num_steps, args.num_envs) + shape).to(device)
        else:
            self.obs = torch.zeros((args.num_steps, args.num_envs) + self.obs_shape).to(device)
        self.actions = torch.zeros((args.num_steps, args.num_envs) + self.action_shape).to(device)
        self.logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
        self.rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
        self.dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
        self.values = torch.zeros((args.num_steps, args.num_envs)).to(device)

        self.global_step = 0

    def _wrap_obs(self, obs):
        device = get_global_variable("device")
        if isinstance(self.obs_shape, dict):
            ret = {}
            for k, shape in self.obs_shape.items():
                ret[k] = torch.Tensor(obs[k]).to(device)
            return ret
        return torch.Tensor(obs).to(device)

    def _add_obs(self, step, next_obs):
        if isinstance(self.obs_shape, dict):
            for k, shape in self.obs_shape.items():
                self.obs[k][step] = next_obs[k]
        else:
            self.obs[step] = next_obs

    def _flatten_obs(self):
        if isinstance(self.obs_shape, dict):
            ret = {}
            for k, shape in self.obs_shape.items():
                ret[k] = self.obs[k].reshape((-1,) + shape)
            return ret
        return self.obs.reshape((-1,) + self.obs_shape)

    def _slice_obs(self, b_obs, indices):
        if isinstance(self.obs_shape, dict):
            ret = {}
            for k, shape in self.obs_shape.items():
                ret[k] = b_obs[k][indices]
            return ret
        return b_obs[indices]

    def train_one_iteration(self, iteration, policy, train_env=None, dataset=None):
        args = self.args
        device = get_global_variable("device")

        # Annealing the rate if instructed to do so.
        if args.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / args.num_iterations
            lrnow = frac * args.learning_rate
            policy.set_learning_rate(lrnow)

        next_obs = self._wrap_obs(train_env.reset())
        next_done = torch.zeros(args.num_envs).to(device)

        for step in range(0, args.num_steps):
            self.global_step += args.num_envs
            self._add_obs(step, next_obs)
            self.dones[step] = next_done

            # ALGO LOGIC: action logic
            with torch.no_grad():
                action, logprob, _, value = policy.get_action_and_value(next_obs)
                self.values[step] = value.flatten()
            self.actions[step] = action
            self.logprobs[step] = logprob

            # TRY NOT TO MODIFY: execute the game and log data.
            next_obs, reward, next_done, info = train_env.step(action.cpu().numpy())
            self.rewards[step] = torch.tensor(reward).to(device).view(-1)
            next_obs, next_done = (
                self._wrap_obs(next_obs),
                torch.Tensor(next_done).to(device),
            )

        # bootstrap value if not done
        with torch.no_grad():
            next_value = policy.get_value(next_obs).reshape(1, -1)
            advantages = torch.zeros_like(self.rewards).to(device)
            lastgaelam = 0
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - self.dones[t + 1]
                    nextvalues = self.values[t + 1]
                delta = (
                    self.rewards[t]
                    + args.gamma * nextvalues * nextnonterminal
                    - self.values[t]
                )
                advantages[t] = lastgaelam = (
                    delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
                )
            returns = advantages + self.values

        # flatten the batch
        b_obs = self._flatten_obs()
        b_logprobs = self.logprobs.reshape(-1)
        b_actions = self.actions.reshape((-1,) + self.action_shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = self.values.reshape(-1)

        # Optimizing the policy and value network
        b_inds = np.arange(args.batch_size)
        clipfracs = []
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue = policy.get_action_and_value(
                    self._slice_obs(b_obs, mb_inds), b_actions.long()[mb_inds]
                )
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    # old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [
                        ((ratio - 1.0).abs() > args.clip_coef).float().mean().item()
                    ]

                mb_advantages = b_advantages[mb_inds]
                if args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (
                        mb_advantages.std() + 1e-8
                    )

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(
                    ratio, 1 - args.clip_coef, 1 + args.clip_coef
                )
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                newvalue = newvalue.view(-1)
                if args.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -args.clip_coef,
                        args.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef
                loss.backward()
                print(loss)

                policy.update_params(args.max_grad_norm)

            if args.target_kl is not None and approx_kl > args.target_kl:
                break
