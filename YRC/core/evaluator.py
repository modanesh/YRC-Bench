import numpy as np
import torch
from pyod.utils.data import evaluate_print
from .configs.global_configs import get_global_variable


class Evaluator:
    def __init__(self, config, logger, eval_id_env, eval_ood_env):
        self.validation_steps = config.validation_steps
        self.logger = logger
        self.eval_id_env = eval_id_env
        self.eval_ood_env = eval_ood_env
        self.best_id_index = -1
        self.best_ood_index = -1
        self.best_rewards = {'id': float('-inf'), 'ood': float('-inf')}
        self.model_improved = {'id': False, 'ood': False}

    def evaluate_policy(self, policy):
        rewards = {
            'id': self._run_validation_episodes(policy, self.eval_id_env),
            'ood': self._run_validation_episodes(policy, self.eval_ood_env)
        }

        for env_type, (reward_batch, done_batch) in rewards.items():
            val_reward = self.logger.feed(reward_batch, done_batch, is_val=True, is_id=(env_type == 'id'))
            self.logger.dump(is_val=True, is_id=(env_type == 'id'))
            self._update_best_reward(env_type, val_reward)

        self._close_environments()

    def _run_validation_episodes(self, policy, eval_env):
        num_envs = eval_env.base_env.num_envs
        reward_batch = np.zeros((self.validation_steps, num_envs))
        done_batch = np.zeros((self.validation_steps, num_envs))

        with torch.no_grad():
            obs, pi_w_hidden = eval_env.reset()
            ep_steps = 0
            for i in range(self.validation_steps):
                action = policy.act(obs, pi_w_hidden)
                obs, reward, done, info, pi_w_hidden = eval_env.step(action)
                reward_batch[i], done_batch[i] = reward, done
                ep_steps += 1
                if self._should_reset_environment(done, ep_steps, eval_env):
                    done_batch[i] = True
                    obs, pi_w_hidden = self._reset_environment(eval_env)
                    ep_steps = 0

        return reward_batch, done_batch

    def _should_reset_environment(self, done, ep_steps, eval_env):
        return (get_global_variable('benchmark') == 'cliport' and
                (done or ep_steps == eval_env.base_env.task.max_steps))

    def _reset_environment(self, eval_env):
        eval_env.base_env.seed(eval_env.base_env._seed + 1)  # todo: check if the seed actually changes for the eval_env (id and ood)
        return eval_env.reset()

    def _update_best_reward(self, env_type, val_reward):
        self.model_improved[env_type] = val_reward >= self.best_rewards[env_type]
        self.best_rewards[env_type] = max(self.best_rewards[env_type], val_reward)

    def _close_environments(self):
        self.eval_id_env.close()
        self.eval_ood_env.close()

    def evaluate_detector(self, policy, classifier, train_env):
        eval_rollout_len = self.validation_steps // 10
        data = {
            'train': (policy.gather_rollouts(self.validation_steps, train_env), np.zeros(self.validation_steps)),
            'id': (policy.gather_rollouts(eval_rollout_len, self.eval_id_env), np.ones(eval_rollout_len)),
            'ood': (policy.gather_rollouts(eval_rollout_len, self.eval_ood_env), np.ones(eval_rollout_len))
        }

        x_id, y_id = np.concatenate([data['train'][0], data['id'][0]]), np.concatenate([data['train'][1], data['id'][1]])
        x_ood, y_ood = np.concatenate([data['train'][0], data['ood'][0]]), np.concatenate([data['train'][1], data['ood'][1]])

        for name, (x, y) in zip(["ID", "OOD"], [(x_id, y_id), (x_ood, y_ood)]):
            y_pred, y_scores = self.get_predictions(classifier, x)
            print(f"\nOn Full {name} Data:")
            evaluate_print(policy.policy.clf_name, y, y_scores)

        self._close_environments()

    def get_predictions(self, clf, x):
        return clf.predict(x), clf.decision_function(x)

    def evaluate_nonparam(self, policy):
        rewards = {
            'id': self._run_validation_episodes_nonparam(policy, self.eval_id_env),
            'ood': self._run_validation_episodes_nonparam(policy, self.eval_ood_env)
        }

        for env_type, (reward_batch, done_batch) in rewards.items():
            val_reward = self.logger.feed(reward_batch, done_batch, is_val=True, is_id=(env_type == 'id'))
            self.logger.dump(is_val=True, is_id=(env_type == 'id'))
            self._update_best_reward(env_type, val_reward)

        self._close_environments()

    def _run_validation_episodes_nonparam(self, policy, eval_env):
        num_envs = eval_env.base_env.num_envs
        reward_batch = np.zeros((self.validation_steps, num_envs))
        done_batch = np.zeros((self.validation_steps, num_envs))

        with torch.no_grad():
            obs, pi_w_hidden = eval_env.reset()
            ep_steps = 0
            for i in range(self.validation_steps):
                sampled_logits, max_logits, sampled_probs, max_probs, entropy = eval_env.weak_policy.get_logits_probs(obs)
                actions = self._get_actions(policy, sampled_logits, max_logits, sampled_probs, max_probs, entropy)
                obs, reward, done, info, pi_w_hidden = eval_env.step(actions)
                reward_batch[i], done_batch[i] = reward, done
                ep_steps += 1
                if self._should_reset_environment(done, ep_steps, eval_env):
                    done_batch[i] = True
                    obs, pi_w_hidden = self._reset_environment(eval_env)
                    ep_steps = 0

        return reward_batch, done_batch

    def _get_actions(self, policy, sampled_logits, max_logits, sampled_probs, max_probs, entropy):
        policy_type = policy.policy.type
        threshold = policy.policy.threshold
        if policy_type == "sampled_logit":
            return np.where(sampled_logits < threshold, 0, 1)
        elif policy_type == "max_logit":
            return np.where(max_logits < threshold, 0, 1)
        elif policy_type == "sampled_prob":
            return np.where(sampled_probs < threshold, 0, 1)
        elif policy_type == "max_prob":
            return np.where(max_probs < threshold, 0, 1)
        elif policy_type == "entropy":
            return np.where(entropy < threshold, 0, 1)
        else:
            raise ValueError(f"Unknown policy type: {policy_type}")
