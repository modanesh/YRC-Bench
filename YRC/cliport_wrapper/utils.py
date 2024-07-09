import json
import os
import torch
import numpy as np
from cliport.environments.environment import HelpEnvWrapper, Environment
from cliport import tasks, agents
import torch.optim as optim
from .models import CategoricalPolicyT3, ImpalaModel, PPO
from YRC.core.utils import to_dict
from cliport.utils import utils
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler


class ReplayBuffer:
    def __init__(self, state_dim, action_dim, buffer_size, num_envs, device='cpu'):
        self._buffer_size = buffer_size
        self._pointer = 0
        self._size = 0
        self._device = device
        self.num_envs = num_envs
        self._states = torch.zeros((buffer_size,) + state_dim, dtype=torch.float32, device=device)
        self._actions = torch.zeros((buffer_size, self.num_envs), dtype=torch.float32, device=device)
        self._rewards = torch.zeros((buffer_size, self.num_envs), dtype=torch.float32, device=device)
        self._dones = torch.zeros((buffer_size, self.num_envs), dtype=torch.float32, device=device)
        self._logprobs = torch.zeros((buffer_size, self.num_envs), dtype=torch.float32, device=device)
        self._values = torch.zeros((buffer_size, self.num_envs), dtype=torch.float32, device=device)
        self._next_states = torch.zeros((buffer_size,) + state_dim, dtype=torch.float32, device=device)
        self._adv = torch.zeros((buffer_size, self.num_envs), dtype=torch.float32, device=device)
        self._returns = torch.zeros((buffer_size, self.num_envs), dtype=torch.float32, device=device)
        self._info = [None] * buffer_size

    def _to_numpy(self, data):
        if isinstance(data, torch.Tensor):
            data = data.cpu()
        return np.array(data, dtype=np.float32)

    def _to_torch(self, data):
        return torch.tensor(data, dtype=torch.float32, device=self._device)

    def sample(self, batch_size):
        indices = np.random.randint(0, self._size, size=batch_size)
        return (
            self._states[indices],
            self._actions[indices],
            self._logprobs[indices],
            self._rewards[indices],
            self._next_states[indices],
            self._dones[indices],
            self._values[indices],
            self._adv[indices],
            self._returns[indices],
            np.array(self._info)[indices]
        )

    def add_transition(self, state, action, logprob, reward, next_state, done, value, info):
        self._states[self._pointer] = self._to_torch(state)
        self._actions[self._pointer] = self._to_torch(action)
        self._logprobs[self._pointer] = self._to_torch(logprob)
        self._rewards[self._pointer] = self._to_torch(reward)
        self._next_states[self._pointer] = self._to_torch(next_state)
        self._dones[self._pointer] = self._to_torch(done)
        self._values[self._pointer] = self._to_torch(value)
        self._info[self._pointer] = info

        self._pointer = (self._pointer + self.num_envs) % self._buffer_size
        self._size = min(self._size + 1, self._buffer_size)

    def get_info(self):
        return self._info[:self._size]

    def fetch_train_generator(self, mini_batch_size=None):
        batch_size = self._size
        if mini_batch_size is None:
            mini_batch_size = batch_size

        sampler = BatchSampler(SubsetRandomSampler(range(batch_size)),
                               mini_batch_size,
                               drop_last=True)

        for indices in sampler:
            yield (
                self._states[indices],
                self._actions[indices].squeeze(1),
                self._dones[indices],
                self._logprobs[indices].squeeze(1),
                self._values[indices].squeeze(1),
                self._returns[indices].squeeze(1),
                self._adv[indices].squeeze(1),
                [self._info[i] for i in indices]
            )

    def compute_estimates(self, gamma=0.99, lmbda=0.95, use_gae=True, normalize_adv=True):
        with torch.no_grad():
            last_value = self._values[(self._pointer - 1) % self._buffer_size]
            advantages = torch.zeros_like(self._rewards)
            returns = torch.zeros_like(self._rewards)

            if use_gae:
                gae = 0
                for step in reversed(range(self._size)):
                    if step == self._size - 1:
                        next_value = last_value
                    else:
                        next_value = self._values[(step + 1) % self._buffer_size]

                    delta = self._rewards[step] + gamma * next_value * (1 - self._dones[step]) - self._values[step]
                    gae = delta + gamma * lmbda * (1 - self._dones[step]) * gae
                    advantages[step] = gae
                    returns[step] = advantages[step] + self._values[step]
            else:
                running_return = last_value
                for step in reversed(range(self._size)):
                    running_return = self._rewards[step] + gamma * running_return * (1 - self._dones[step])
                    returns[step] = running_return
                advantages = returns - self._values[:self._size]

            if normalize_adv:
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            # Update the stored advantages and returns
            self._adv[:self._size] = advantages[:self._size]
            self._returns[:self._size] = returns[:self._size]

    def fetch_log_data(self):
        rew_batch = self._rewards.cpu().numpy()[:self._size]
        done_batch = self._dones.cpu().numpy()[:self._size]
        return rew_batch, done_batch




def load_ckpts(results_path, model_task):
    result_jsons = [c for c in os.listdir(results_path) if "results-val" in c]
    if 'multi' in model_task:
        result_jsons = [r for r in result_jsons if "multi" in r]
    else:
        result_jsons = [r for r in result_jsons if "multi" not in r]

    if len(result_jsons) > 0:
        result_json = result_jsons[0]
        with open(os.path.join(results_path, result_json), 'r') as f:
            eval_res = json.load(f)
        best_success = -1.0
        for ckpt, res in eval_res.items():
            if res['mean_reward'] > best_success:
                best_checkpoint = ckpt
                best_success = res['mean_reward']
        ckpt = best_checkpoint
    else:
        raise ValueError(f"No best val ckpt found!")
    return ckpt


def environment_setup(assets_root, weak_policy, reward_max, timeout, strong_query_cost, switching_agent_cost, disp, shared_memory, task):
    tsk = tasks.names[task]()
    env = HelpEnvWrapper(assets_root, weak_policy, None, reward_max, timeout, strong_query_cost, switching_agent_cost, disp=disp,
                         shared_memory=shared_memory, hz=480)
    env.set_task(tsk)
    strong_policy = tsk.oracle(env)
    env.set_strong_policy(strong_policy)
    return env, tsk


def load_weak_policy(cfgs):
    ckpt = load_ckpts(cfgs.results_path, cfgs.model_task)
    model_file = os.path.join(cfgs.model_path, ckpt)
    name = f'{cfgs.task}-{cfgs.agent}-n{cfgs.n_demos}'
    pi_w = agents.names[cfgs.agent](name, to_dict(cfgs))
    pi_w.load(model_file)
    pi_w.eval()
    return pi_w


def define_help_policy(env, weak_agent, in_feature_shape, device):
    obs = env.reset()
    img = [utils.get_image(obs)]
    info = [env.info]
    pick_features, place_features = weak_agent.extract_features(img, info)
    hidden_size = pick_features[0].shape[0] + place_features[0].shape[0]
    # model = model_setup(hidden_size)
    action_size = 2

    # hidden_size = weak_agent.policy.embedder.output_dim
    # softmax_size = weak_agent.policy.fc_policy.out_features
    # if help_policy_type == "T1":
    #     policy = CategoricalPolicy(model, action_size)
    # elif help_policy_type == "T2":
    #     policy = CategoricalPolicyT2(model, action_size, hidden_size, softmax_size)
    # elif help_policy_type == "T3":
    #     policy = CategoricalPolicyT3(action_size, hidden_size)
    # else:
    #     raise ValueError("Invalid help policy type.")
    policy = CategoricalPolicyT3(action_size, hidden_size)  # TODO: implement other types of help policies. Currently only type 3 is implemented
    policy.to(device)
    # return model, policy
    return None, policy


def model_setup(in_feature_shape):
    model = ImpalaModel(in_feature_shape)
    return model


def algorithm_setup(env, tsk, policy, logger, storage, storage_valid, device, num_checkpoints, model_file, hyperparameters, pi_w=None, pi_o=None,
                    help_policy_type=None):
    print('::[LOGGING]::INTIALIZING AGENT...')
    agent = PPO(env, policy, logger, storage, device,
                num_checkpoints,
                tsk=tsk,
                storage_valid=storage_valid,
                pi_w=pi_w,
                pi_o=pi_o,
                help_policy_type=help_policy_type,
                **hyperparameters)
    return agent