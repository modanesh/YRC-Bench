import torch
import numpy as np
from .configs.global_configs import get_global_variable


class Evaluator:
    def __init__(self, config, logger, val_env, test_env):
        self.validation_steps = config.validation_steps
        self.logger = logger
        self.val_env = val_env
        self.test_env = test_env
        self.best_reward = float('-inf')
        self.model_improved = False
        self.best_index = -1

    def evaluate(self, policy):
        reward_batch, done_batch = self._run_validation_episodes(policy)
        val_reward = self.logger.feed(reward_batch, done_batch, is_val=True)
        self.logger.dump(is_val=True)
        self._update_best_reward(val_reward)

    def _run_validation_episodes(self, policy):
        num_envs = self.val_env.base_env.num_envs
        reward_batch = np.zeros((self.validation_steps, num_envs))
        done_batch = np.zeros((self.validation_steps, num_envs))

        policy.eval()
        with torch.no_grad():
            obs, pi_w_hidden = self.val_env.reset()
            ep_steps = 0
            for i in range(self.validation_steps):
                action, _, _ = policy.predict(obs, pi_w_hidden)
                obs, reward, done, info, pi_w_hidden = self.val_env.step(action)
                reward_batch[i] = reward
                done_batch[i] = done
                ep_steps += 1
                if self._should_reset_environment(done, ep_steps):
                    done_batch[i] = True
                    obs, pi_w_hidden = self._reset_environment(i)
                    ep_steps = 0

        return reward_batch, done_batch

    def _should_reset_environment(self, done, ep_steps):
        return (get_global_variable('benchmark') == 'cliport' and
                (done or ep_steps == self.val_env.base_env.task.max_steps))

    def _reset_environment(self, step):
        self.val_env.base_env.seed(self.val_env.base_env._seed + step)
        return self.val_env.reset()

    def _update_best_reward(self, val_reward):
        if val_reward > self.best_reward:
            self.best_reward = val_reward
            self.model_improved = True
        else:
            self.model_improved = False
