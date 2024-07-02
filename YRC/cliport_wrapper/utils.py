import json
import os
import torch
import numpy as np
from cliport.environments.environment import Environment
from cliport import tasks, agents
import torch.optim as optim
from models import Agent, ImpalaModel


class ReplayBuffer:
    def __init__(self, state_dim, action_dim, buffer_size, use_torch=False, device='cpu'):
        self._buffer_size = buffer_size
        self._pointer = 0
        self._size = 0
        self._device = device
        self._use_torch = use_torch
        if not use_torch:
            self._states = np.zeros((buffer_size,) + state_dim, dtype=np.float32)
            self._actions = np.zeros((buffer_size, action_dim), dtype=np.float32)
            self._logprobs = np.zeros((buffer_size, 1), dtype=np.float32)
            self._rewards = np.zeros((buffer_size, 1), dtype=np.float32)
            self._next_states = np.zeros((buffer_size,) + state_dim, dtype=np.float32)
            self._dones = np.zeros((buffer_size, 1), dtype=np.float32)
            self._values = np.zeros((buffer_size, 1), dtype=np.float32)
        else:
            self._states = torch.zeros((buffer_size,) + state_dim, dtype=torch.float32, device=device)
            self._actions = torch.zeros((buffer_size, action_dim), dtype=torch.float32, device=device)
            self._logprobs = torch.zeros((buffer_size, 1), dtype=torch.float32, device=device)
            self._rewards = torch.zeros((buffer_size, 1), dtype=torch.float32, device=device)
            self._next_states = torch.zeros((buffer_size,) + state_dim, dtype=torch.float32, device=device)
            self._dones = torch.zeros((buffer_size, 1), dtype=torch.float32, device=device)
            self._values = torch.zeros((buffer_size, 1), dtype=torch.float32, device=device)
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
            np.array(self._info)[indices]
        )

    def add_transition(self, state, action, logprob, reward, next_state, done, value, info):
        if not self._use_torch:
            self._states[self._pointer] = self._to_numpy(state)
            self._actions[self._pointer] = self._to_numpy(action)
            self._logprobs[self._pointer] = self._to_numpy(logprob)
            self._rewards[self._pointer] = self._to_numpy(reward)
            self._next_states[self._pointer] = self._to_numpy(next_state)
            self._dones[self._pointer] = self._to_numpy(done)
            self._values[self._pointer] = self._to_numpy(value)
        else:
            self._states[self._pointer] = self._to_torch(state)
            self._actions[self._pointer] = self._to_torch(action)
            self._logprobs[self._pointer] = self._to_torch(logprob)
            self._rewards[self._pointer] = self._to_torch(reward)
            self._next_states[self._pointer] = self._to_torch(next_state)
            self._dones[self._pointer] = self._to_torch(done)
            self._values[self._pointer] = self._to_torch(value)
        self._info[self._pointer] = info

        self._pointer = (self._pointer + 1) % self._buffer_size
        self._size = min(self._size + 1, self._buffer_size)

    def get_info(self):
        return self._info[:self._size]


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


def environment_setup(assets_root, disp, shared_memory, task):
    env = Environment(assets_root, disp=disp, shared_memory=shared_memory, hz=480)
    tsk = tasks.names[task]()
    env.set_task(tsk)
    return env, tsk


def load_agents(cfgs, env, tsk, train_d, val_d, in_feature_shape, out_shape, lr):
    pi_o = tsk.oracle(env)
    ckpt = load_ckpts(cfgs.results_path, cfgs.model_task)
    model_file = os.path.join(cfgs.model_path, ckpt)
    name = f'{cfgs.task}-{cfgs.agent}-n{cfgs.n_demos}'
    pi_w = agents.names[cfgs.agent](name, vars(cfgs), train_d, val_d)
    pi_w.load(model_file)
    pi_w.eval()

    embedder = ImpalaModel(in_feature_shape)
    embedder.to(cfgs.device)
    pi_h = Agent(embedder, out_shape)
    pi_h.to(cfgs.device)
    opt = optim.Adam(pi_h.parameters(), lr=lr, eps=1e-5)
    return pi_w, pi_o, pi_h, opt