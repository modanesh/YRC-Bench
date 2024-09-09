import copy
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions.categorical import Categorical

from YRC.core.configs.global_configs import get_global_variable
from cliport.utils import utils as cliport_utils
from YRC.core import Algorithm
from .utils import ReplayBuffer


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
                self.obs[k] = torch.zeros((args.num_steps, args.num_envs) + shape).to(
                    device
                )
        else:
            self.obs = torch.zeros((args.num_steps, args.num_envs) + self.obs_shape).to(
                device
            )
        self.actions = torch.zeros(
            (args.num_steps, args.num_envs) + self.action_shape
        ).to(device)
        self.logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
        self.rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
        self.dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
        self.values = torch.zeros((args.num_steps, args.num_envs)).to(device)

        self.global_step = 0
        # self.next_obs = torch.Tensor(envs["train"].reset()).to(device)

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
            # obs[step] = next_obs
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
            next_obs, next_done = self._wrap_obs(next_obs), torch.Tensor(next_done).to(
                device
            )

            # for item in info:
            #    if "episode" in item.keys():
            #        print(f"global_step={global_step}, episodic_return={item['episode']['r']}")
            #        writer.add_scalar("charts/episodic_return", item["episode"]["r"], global_step)
            #        writer.add_scalar("charts/episodic_length", item["episode"]["l"], global_step)
            #        break

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
        # b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
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
                    old_approx_kl = (-logratio).mean()
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

                # optimizer.zero_grad()
                # nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                # optimizer.step()

            if args.target_kl is not None and approx_kl > args.target_kl:
                break


class OldPPOAlgorithm(Algorithm):

    def __init__(self, config, env):

        self.t = 0

        self.training_steps = config.training_steps
        self.log_freq = config.log_freq
        self.save_freq = config.save_freq
        self.save_dir = config.save_dir

        self.rollout_length = config.rollout_length
        self.mini_batch_size = config.mini_batch_size
        self.mini_batch_per_epoch = config.mini_batch_per_epoch
        self.epoch = config.epoch
        self.grad_clip_norm = config.grad_clip_norm
        self.value_coef = config.value_coef
        self.entropy_coef = config.entropy_coef
        self.gamma = config.gamma
        self.lmbda = config.lmbda
        self.use_gae = config.use_gae
        self.normalize_adv = config.normalize_adv
        self.eps_clip = config.eps_clip

        self.rb = ReplayBuffer(
            self.gamma,
            self.lmbda,
            self.use_gae,
            self.normalize_adv,
            env.obs_size,
            config.rollout_length,
            env.num_envs,
        )

    def train_one_iteration(self, policy, train_env=None, dataset=None):

        assert train_env is not None and dataset is None

        policy.train()
        obs = train_env.reset()
        ep_steps = 0

        for _ in range(self.rollout_length):
            act, log_prob_act, value = policy.get_act_logprob_value(obs)
            next_obs, rew, done, info = train_env.step(act)

            self.rb.add_transition(
                obs, act, log_prob_act, rew, next_obs, done, value, info
            )
            obs = next_obs
            ep_steps += 1

        _, _, last_val = policy.get_act_logprob_value(obs)

        self.rb.store_last(obs, last_val)
        self.rb.compute_estimates()

        summary = self.update_policy_online(policy, train_env)

        self.update_training_progress(policy)
        # self.logger.wandb_log_loss(summary)

        return summary

    def update_training_progress(self, policy):
        self.t += self.rollout_length * self.rb.num_envs
        # self.log_training_progress()
        policy.adjust_lr(self.t, self.training_steps)

    def update_policy_online(self, policy, train_env):

        pi_loss_list, value_loss_list, entropy_loss_list = [], [], []
        batch_size = self.rollout_length * self.rb.num_envs // self.mini_batch_per_epoch
        self.mini_batch_size = min(self.mini_batch_size, batch_size)
        grad_accumulation_steps = batch_size // self.mini_batch_size
        grad_accumulation_cnt = 1

        for _ in range(self.epoch):
            generator = self.rb.fetch_train_generator(
                mini_batch_size=self.mini_batch_size
            )
            for sample in generator:
                (
                    obs_batch,
                    act_batch,
                    done_batch,
                    old_log_prob_act_batch,
                    old_value_batch,
                    return_batch,
                    adv_batch,
                    info_batch,
                ) = sample

                # pi_w_hidden_batch = train_env.get_weak_policy_features(
                #    obs_batch, info_batch
                # )

                # if get_global_variable("benchmark") == "cliport":
                #    obs_batch = obs_batch.permute(0, 3, 1, 2)

                dist_batch, value_batch = policy.forward(obs_batch)

                pi_loss, value_loss, entropy_loss = self.compute_losses(
                    dist_batch,
                    value_batch,
                    act_batch,
                    old_log_prob_act_batch,
                    old_value_batch,
                    return_batch,
                    adv_batch,
                )

                loss = (
                    pi_loss
                    + self.value_coef * value_loss
                    - self.entropy_coef * entropy_loss
                )
                print(loss)
                loss.backward()

                if grad_accumulation_cnt % grad_accumulation_steps == 0:
                    policy.update_params(self.grad_clip_norm)

                    # torch.nn.utils.clip_grad_norm_(
                    #    policy.parameters(), self.grad_clip_norm
                    # )
                    # policy.optimizer.step()
                    # policy.optimizer.zero_grad()

                grad_accumulation_cnt += 1

                pi_loss_list.append(-pi_loss.item())
                value_loss_list.append(-value_loss.item())
                entropy_loss_list.append(entropy_loss.item())

        return {
            "Loss/pi": np.mean(pi_loss_list),
            "Loss/v": np.mean(value_loss_list),
            "Loss/entropy": np.mean(entropy_loss_list),
        }

    def compute_losses(
        self, dist, value, act, old_log_prob_act, old_value, return_batch, adv_batch
    ):
        log_prob_act = dist.log_prob(act)
        ratio = torch.exp(log_prob_act - old_log_prob_act)
        surr1 = ratio * adv_batch
        surr2 = torch.clamp(ratio, 1.0 - self.eps_clip, 1.0 + self.eps_clip) * adv_batch
        pi_loss = -torch.min(surr1, surr2).mean()

        clipped_value = old_value + (value - old_value).clamp(
            -self.eps_clip, self.eps_clip
        )
        v_surr1 = (value - return_batch).pow(2)
        v_surr2 = (clipped_value - return_batch).pow(2)
        value_loss = 0.5 * torch.max(v_surr1, v_surr2).mean()

        entropy_loss = dist.entropy().mean()
        return pi_loss, value_loss, entropy_loss


class PPO(nn.Module):
    def __init__(
        self,
        config,
        in_channels=None,
        action_size=2,
        additional_hidden_dim=0,
        feature_type="T1",
    ):
        super(PPO, self).__init__()
        self.architecture = config.architecture
        self.eps_clip = config.eps_clip
        self.type = feature_type
        if self.architecture == "impala":
            self.embedder = ImpalaModel(
                in_channels, benchmark=get_global_variable("benchmark")
            )
            total_input_dim = self.embedder.output_dim + additional_hidden_dim
        else:
            self.embedder = None
            total_input_dim = additional_hidden_dim

        self.fc_policy = orthogonal_init(
            nn.Linear(total_input_dim, action_size), gain=0.01
        )
        self.fc_value = orthogonal_init(nn.Linear(total_input_dim, 1), gain=1.0)

    def forward(self, x, pi_w_hidden):
        if self.type == "T3":
            hidden = pi_w_hidden
        else:
            hidden = self.embedder(x)
            if self.type == "T2":
                hidden = torch.cat([hidden, pi_w_hidden], dim=1)

        if torch.isnan(hidden).any():
            print("Hidden shape:", hidden.shape)
            print("Hidden contains NaN:", torch.isnan(hidden).any())

        logits = self.fc_policy(hidden)
        if torch.isnan(logits).any():
            print("Logits shape:", logits.shape)
            print("Logits contains NaN:", torch.isnan(logits).any())

        log_probs = F.log_softmax(logits, dim=1)
        if torch.isnan(log_probs).any():
            print("Log_probs shape:", log_probs.shape)
            print("Log_probs contains NaN:", torch.isnan(log_probs).any())
        p = Categorical(logits=log_probs)
        v = self.fc_value(hidden).reshape(-1)
        return p, v

    def extract_features(self, x):
        return self.embedder(x) if self.type != "T3" else None

    def compute_losses(
        self, dist, value, act, old_log_prob_act, old_value, return_batch, adv_batch
    ):
        log_prob_act = dist.log_prob(act)
        ratio = torch.exp(log_prob_act - old_log_prob_act)
        surr1 = ratio * adv_batch
        surr2 = torch.clamp(ratio, 1.0 - self.eps_clip, 1.0 + self.eps_clip) * adv_batch
        pi_loss = -torch.min(surr1, surr2).mean()

        clipped_value = old_value + (value - old_value).clamp(
            -self.eps_clip, self.eps_clip
        )
        v_surr1 = (value - return_batch).pow(2)
        v_surr2 = (clipped_value - return_batch).pow(2)
        value_loss = 0.5 * torch.max(v_surr1, v_surr2).mean()

        entropy_loss = dist.entropy().mean()
        return pi_loss, value_loss, entropy_loss

    def save_model(self, save_path, model):
        print(":::::::::::::SAVING MODEL:::::::::::::")
        torch.save(
            {
                "model_state_dict": self.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
            },
            save_path,
        )

    @torch.no_grad()
    def predict(self, obs, pi_w_hidden):
        dist, value = self(obs, pi_w_hidden)
        act = dist.sample()
        log_prob_act = dist.log_prob(act)
        return act.cpu().numpy(), log_prob_act.cpu().numpy(), value.cpu().numpy()

    @torch.no_grad()
    def act(self, obs, pi_w_hidden):
        dist, value = self(obs, pi_w_hidden)
        act = dist.sample()
        return act.cpu().numpy()


class ProcgenPPO(PPO):
    def __init__(
        self,
        config,
        in_channels,
        action_size=2,
        additional_hidden_dim=0,
        feature_type="T1",
    ):
        super().__init__(
            config, in_channels, action_size, additional_hidden_dim, feature_type
        )
        self.learning_rate = config.learning_rate
        self.optimizer = optim.Adam(self.parameters(), lr=self.learning_rate, eps=1e-5)

    def predict(self, obs, pi_w_hidden):
        obs = torch.FloatTensor(obs).to(device=get_global_variable("device"))
        return super().predict(obs, pi_w_hidden)

    def act(self, obs, pi_w_hidden):
        obs = torch.FloatTensor(obs).to(device=get_global_variable("device"))
        return super().act(obs, pi_w_hidden)


class CliportPPO(PPO):
    def __init__(
        self,
        config,
        in_channels,
        action_size=2,
        additional_hidden_dim=0,
        feature_type="T1",
    ):
        super().__init__(
            config, in_channels, action_size, additional_hidden_dim, feature_type
        )
        self.learning_rate = config.learning_rate
        self.optimizer = optim.Adam(self.parameters(), lr=self.learning_rate, eps=1e-5)

    def predict(self, obs, pi_w_hidden):
        obs = cliport_utils.get_image(obs)
        obs = torch.FloatTensor(obs).to(device=get_global_variable("device"))
        obs = obs.permute(2, 0, 1).unsqueeze(0)
        return super().predict(obs, pi_w_hidden)

    def act(self, obs, pi_w_hidden):
        obs = cliport_utils.get_image(obs)
        obs = torch.FloatTensor(obs).to(device=get_global_variable("device"))
        obs = obs.permute(2, 0, 1).unsqueeze(0)
        return super().act(obs, pi_w_hidden)


class ResidualBlock(nn.Module):
    def __init__(self, in_channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        self.conv2 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=3,
            stride=1,
            padding=1,
        )

    def forward(self, x):
        out = nn.ReLU()(x)
        out = self.conv1(out)
        out = nn.ReLU()(out)
        out = self.conv2(out)
        return out + x


class ImpalaBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ImpalaBlock, self).__init__()
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        self.res1 = ResidualBlock(out_channels)
        self.res2 = ResidualBlock(out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)(x)
        x = self.res1(x)
        x = self.res2(x)
        return x


class ImpalaModel(nn.Module):
    def __init__(self, in_channels, benchmark, scale=1):
        super(ImpalaModel, self).__init__()
        self.block1 = ImpalaBlock(in_channels=in_channels, out_channels=16 * scale)
        self.block2 = ImpalaBlock(in_channels=16 * scale, out_channels=32 * scale)
        self.block3 = ImpalaBlock(in_channels=32 * scale, out_channels=32 * scale)
        if benchmark == "procgen":
            fc_input = 32 * scale * 8 * 8
        elif benchmark == "cliport":
            fc_input = 32 * scale * 20 * 40

        self.fc = nn.Linear(in_features=fc_input, out_features=256)
        self.output_dim = 256
        self.apply(xavier_uniform_init)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = nn.ReLU()(x)
        # x = torch.flatten(x, start_dim=1)
        x = Flatten()(x)
        x = self.fc(x)
        x = nn.ReLU()(x)
        if torch.isnan(x).any():
            print("ImpalaModel output shape:", x.shape)
            print("ImpalaModel output contains NaN:", torch.isnan(x).any())
        return x


class PPOFrozen:
    def __init__(self, policy, device):
        self.policy = policy
        self.device = device

    @torch.no_grad()
    def predict(self, obs):
        obs = torch.FloatTensor(obs).to(device=self.device)
        dist, value = self.policy(obs, pi_w_hidden=None)
        act = dist.sample()
        log_prob_act = dist.log_prob(act)
        return act.cpu().numpy(), log_prob_act.cpu().numpy(), value.cpu().numpy()

    @torch.no_grad()
    def get_logits_probs(self, obs):
        obs = torch.FloatTensor(obs).to(device=self.device)
        hidden = self.policy.embedder(obs)
        logits = self.policy.fc_policy(hidden)
        log_probs = F.log_softmax(logits, dim=1)
        dist = Categorical(log_probs)
        action = dist.sample()
        sampled_logits = (
            torch.gather(logits, 1, action.unsqueeze(1)).squeeze().cpu().numpy()
        )
        max_logits = logits.max(dim=1).values.cpu().numpy()
        sampled_probs = (
            torch.gather(log_probs, 1, action.unsqueeze(1)).squeeze().cpu().numpy()
        )
        max_probs = log_probs.max(dim=1).values.cpu().numpy()
        entropy = dist.entropy().cpu().numpy()
        return sampled_logits, max_logits, sampled_probs, max_probs, entropy


def orthogonal_init(module, gain=nn.init.calculate_gain("relu")):
    if isinstance(module, (nn.Linear, nn.Conv2d)):
        nn.init.orthogonal_(module.weight.data, gain)
        nn.init.constant_(module.bias.data, 0)
    return module


def xavier_uniform_init(module, gain=1.0):
    if isinstance(module, (nn.Linear, nn.Conv2d)):
        nn.init.xavier_uniform_(module.weight.data, gain)
        nn.init.constant_(module.bias.data, 0)
    return module


class Flatten(nn.Module):
    def forward(self, x):
        return torch.flatten(x, start_dim=1)


class RainbowDQN(nn.Module):
    def __init__(
        self,
        config,
        in_channels=None,
        action_size=2,
        additional_hidden_dim=0,
        feature_type="T1",
    ):
        super().__init__()
        self.architecture = config.architecture
        self.action_size = action_size
        self.type = feature_type
        self.v_min = config.v_min
        self.v_max = config.v_max
        self.num_atoms = config.num_atoms
        self.support = torch.linspace(self.v_min, self.v_max, self.num_atoms).to(
            get_global_variable("device")
        )

        if self.architecture == "impala":
            self.embedder = ImpalaModel(
                in_channels, benchmark=get_global_variable("benchmark")
            )
            total_input_dim = self.embedder.output_dim + additional_hidden_dim
        else:
            self.embedder = None
            total_input_dim = additional_hidden_dim

        self.advantage_stream = orthogonal_init(
            nn.Linear(total_input_dim, action_size * self.num_atoms), gain=0.01
        )
        self.value_stream = orthogonal_init(
            nn.Linear(total_input_dim, self.num_atoms), gain=1.0
        )

        self.target_network = copy.deepcopy(self)
        self.learning_rate = config.learning_rate
        self.optimizer = optim.Adam(self.parameters(), lr=self.learning_rate, eps=1e-5)

    def forward(self, x, pi_w_hidden):
        if self.type == "T3":
            hidden = pi_w_hidden
        else:
            hidden = self.embedder(x)
            if self.type == "T2":
                hidden = torch.cat([hidden, pi_w_hidden], dim=1)

        advantage = self.advantage_stream(hidden).view(
            -1, self.action_size, self.num_atoms
        )
        value = self.value_stream(hidden).view(-1, 1, self.num_atoms)
        q_dist = value + advantage - advantage.mean(dim=1, keepdim=True)
        return F.softmax(q_dist, dim=-1)

    def compute_loss(self, current_q, target_q):
        # Add a small epsilon to avoid taking log of zero
        epsilon = 1e-8
        current_q_clamped = torch.clamp(current_q, epsilon, 1.0)
        # Compute cross-entropy loss
        loss = -(target_q * torch.log(current_q_clamped)).sum(1)
        return loss.mean()

    def update_target_network(self):
        for target_param, param in zip(
            self.target_network.parameters(), self.parameters()
        ):
            target_param.data.copy_(param.data)

    @torch.no_grad()
    def predict(self, obs, pi_w_hidden):
        q_dist = self(obs, pi_w_hidden)
        q_values = (q_dist * self.support).sum(dim=-1)
        action = q_values.argmax(dim=-1)
        return action.cpu().numpy(), None, q_values.cpu().numpy()

    @torch.no_grad()
    def act(self, obs, pi_w_hidden):
        q_dist = self(obs, pi_w_hidden)
        q_values = (q_dist * self.support).sum(dim=-1)
        action = q_values.argmax(dim=-1)
        return action.cpu().numpy()

    def save_model(self, save_path, model):
        print(":::::::::::::SAVING MODEL:::::::::::::")
        torch.save(
            {
                "model_state_dict": self.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
            },
            save_path,
        )


class ProcgenDQN(RainbowDQN):
    def predict(self, obs, pi_w_hidden):
        obs = torch.Tensor(obs).to(device=get_global_variable("device"))
        return super().predict(obs, pi_w_hidden)

    def act(self, obs, pi_w_hidden):
        obs = torch.Tensor(obs).to(device=get_global_variable("device"))
        return super().act(obs, pi_w_hidden)


class CliportDQN(RainbowDQN):
    def predict(self, obs, pi_w_hidden):
        obs = cliport_utils.get_image(obs)
        obs = torch.Tensor(obs).to(device=get_global_variable("device"))
        obs = obs.permute(2, 0, 1).unsqueeze(0)
        return super().predict(obs, pi_w_hidden)

    def act(self, obs, pi_w_hidden):
        obs = cliport_utils.get_image(obs)
        obs = torch.Tensor(obs).to(device=get_global_variable("device"))
        obs = obs.permute(2, 0, 1).unsqueeze(0)
        return super().act(obs, pi_w_hidden)
