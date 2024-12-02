import logging
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
        self.total_reward = {
            "reward": [0.] * args.num_envs,
            "env_reward": [0.] * args.num_envs
        }

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
                ret[k] = torch.from_numpy(obs[k]).to(device).float()
            return ret
        return torch.from_numpy(obs).to(device).float()

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
        log = {}

        pc_steps = self.args.pretrain_critic_steps
        pretrain_critic = pc_steps is not None and pc_steps > 0 and self.global_step < pc_steps

        # Annealing the rate if instructed to do so.
        lrnow = args.learning_rate
        if args.anneal_lr:
            lrnow *= 1 - self.global_step / args.total_timesteps
        policy.set_learning_rate(lrnow)
        log["lr"] = lrnow

        next_obs = self._wrap_obs(train_env.get_obs())
        next_done = torch.zeros(args.num_envs).to(device)

        log["reward"], log["env_reward"] = [], []
        log["action_1"] = []
        log["action_prob"] = []

        # NOTE: set policy to eval mode when collecting trajectories
        policy.eval()

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

            log["action_1"].extend((action == 1).long().tolist())
            log["action_prob"].extend(logprob.exp().tolist())

            #action = torch.ones_like(action)

            # TRY NOT TO MODIFY: execute the game and log data.
            next_obs, reward, next_done, info = train_env.step(action.cpu().numpy())

            # keep track of episode reward
            for i in range(args.num_envs):
                self.total_reward["reward"][i] += reward[i]
                if "env_reward" in info[i]:
                    self.total_reward["env_reward"][i] += info[i]["env_reward"]
                if next_done[i]:
                    #print("===>", step, reward[i], np.mean(log["reward"]))
                    log["reward"].append(self.total_reward["reward"][i])
                    log["env_reward"].append(self.total_reward["env_reward"][i])
                    self.total_reward["reward"][i] = 0
                    self.total_reward["env_reward"][i] = 0

            self.rewards[step] = torch.from_numpy(reward).to(device).float().view(-1)
            next_obs, next_done = (
                self._wrap_obs(next_obs),
                torch.from_numpy(next_done).to(device).float(),
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
        log["pg_loss"] = []
        log["v_loss"] = []
        log["ent_loss"] = []
        log["loss"] = []
        log["advantage"] = []
        log["value"] = []

        policy.train()

        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue = policy.get_action_and_value(
                    self._slice_obs(b_obs, mb_inds), b_actions.long()[mb_inds]
                )
                log["value"].extend(newvalue.tolist())

                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                mb_advantages = b_advantages[mb_inds]
                log["advantage"].extend(mb_advantages.tolist())
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

                if pretrain_critic:
                    loss = v_loss
                else:
                    loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef
                loss.backward()

                policy.update_params(args.max_grad_norm)

                # log
                log["pg_loss"].append(pg_loss.item())
                log["v_loss"].append(v_loss.item())
                log["ent_loss"].append(entropy_loss.item())
                log["loss"].append(loss.item())

        return log

    def aggregate_log(self, log, new_log):
        for k, v in new_log.items():
            if isinstance(v, list):
                if k not in log:
                    log[k] = v
                else:
                    log[k].extend(v)
            elif isinstance(v, float) or isinstance(v, int):
                log[k] = v
            else:
                raise NotImplementedError

    def summarize(self, log):
        return {
            "lr": log["lr"],
            "reward_mean": float(np.mean(log["reward"])),
            "reward_std": float(np.std(log["reward"])),
            "env_reward_mean": float(np.mean(log["env_reward"])),
            "env_reward_std": float(np.std(log["env_reward"])),
            "pg_loss": float(np.mean(log["pg_loss"])),
            "v_loss": float(np.mean(log["v_loss"])),
            "ent_loss": float(np.mean(log["ent_loss"])),
            "loss": float(np.mean(log["loss"])),
            "advantage_mean": float(np.mean(log["advantage"])),
            "advantage_std": float(np.std(log["advantage"])),
            "value_mean": float(np.mean(log["value"])),
            "value_std": float(np.std(log["value"])),
            "action_1": float(np.mean(log["action_1"])),
            "action_prob": float(np.mean(log["action_prob"]))
        }

    def write_summary(self, summary):

        log_str = "\n"
        log_str += "   Reward:     "
        log_str += f"mean {summary['reward_mean']:7.2f} ± {summary['reward_std']:7.2f}\n"
        log_str += "   Env Reward: "
        log_str += f"mean {summary['env_reward_mean']:7.2f} ± {summary['env_reward_std']:7.2f}\n"

        log_str += "   Loss:       "
        log_str += f"pg_loss {summary['pg_loss']:7.4f}  "
        log_str += f"v_loss {summary['v_loss']:7.4f}  "
        log_str += f"ent_loss {summary['ent_loss']:7.4f}  "
        log_str += f"loss {summary['loss']:7.4f}\n"

        log_str += "   Others:     "
        log_str += f"advantage {summary['advantage_mean']:7.4f} ± {summary['advantage_std']:7.4f}  "
        log_str += f"value {summary['value_mean']:7.4f} ± {summary['value_std']:7.4f}\n"

        log_str += f"   Action 1 frac: {summary['action_1']:7.2f}\n"
        log_str += f"   Action prob: {summary['action_prob']:7.2f}"


        logging.info(log_str)

        return summary