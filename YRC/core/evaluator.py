import logging
import numpy as np


class Evaluator:
    LOGGED_ACTION = 1

    def __init__(self, config):
        self.args = config

    def eval(self, policy, envs, eval_splits, num_episodes=None):
        args = self.args
        policy.eval()

        summary = {}
        for split in eval_splits:
            if num_episodes is None:
                if "val" in split:
                    num_episodes = args.validation_episodes
                else:
                    assert "test" in split
                    num_episodes = args.test_episodes
                assert num_episodes % envs[split].num_envs == 0

            logging.info(f"Evaluation on {split} for {num_episodes} episodes")

            num_iterations = num_episodes // envs[split].num_envs
            log = {}
            for _ in range(num_iterations):
                this_log = self._eval_one_iteration(policy, envs[split])
                self._update_log(log, this_log)

            summary[split] = self.summarize(log)
            self.write_summary(split, summary[split])

            envs[split].close()

        return summary

    def _update_log(self, log, this_log):
        if not log:
            log.update(this_log)
        for k, v in this_log.items():
            if isinstance(v, list):
                log[k].extend(v)
            else:
                log[k] += v

    def _eval_one_iteration(self, policy, env):
        args = self.args
        log = {
            "reward": [0] * env.num_envs,
            "env_reward": [0] * env.num_envs,
            "episode_length": [0] * env.num_envs,
            f"action_{self.LOGGED_ACTION}": 0,
        }

        obs = env.reset()
        has_done = np.array([False] * env.num_envs)
        step = 0
        while not has_done.all():
            action = policy.act(obs, greedy=args.act_greedy)

            obs, reward, done, info = env.step(action)

            """
            if action.ndim == 0:
                action = action.reshape(1)
                reward = reward.reshape(1)
            """

            for i in range(env.num_envs):

                if "env_reward" in info[i]:
                    log["env_reward"][i] += info[i]["env_reward"] * (1 - has_done[i])

                log["reward"][i] += reward[i] * (1 - has_done[i])
                log["episode_length"][i] += 1 - has_done[i]
                if not has_done[i]:
                    log[f"action_{self.LOGGED_ACTION}"] += (action[i] == self.LOGGED_ACTION).sum()

            has_done |= done
            step += 1

        return log

    def summarize(self, log):
        return {
            "steps": int(sum(log["episode_length"])),
            "episode_length_mean": float(np.mean(log["episode_length"])),
            "episode_length_min": int(np.min(log["episode_length"])),
            "episode_length_max": int(np.max(log["episode_length"])),
            "reward_mean": float(np.mean(log["reward"])),
            "raw_reward": log["reward"],
            "reward_std": float(np.std(log["reward"])),
            "env_reward_mean": float(np.mean(log["env_reward"])),
            "env_reward_std": float(np.std(log["env_reward"])),
            f"action_{self.LOGGED_ACTION}_frac": float(
                log[f"action_{self.LOGGED_ACTION}"] / sum(log["episode_length"])
            ),
        }

    def write_summary(self, split, summary):
        log_str = f"   Steps:       {summary['steps']}\n"
        log_str += "   Episode:    "
        log_str += f"mean {summary['episode_length_mean']:7.2f}  "
        log_str += f"min {summary['episode_length_min']:7.2f}  "
        log_str += f"max {summary['episode_length_max']:7.2f}\n"
        log_str += "   Reward:     "
        log_str += f"mean {summary['reward_mean']:.2f} "
        log_str += f"± {(1.96 * summary['reward_std']) / (len(summary['raw_reward']) ** 0.5):.2f}\n"
        log_str += "   Env Reward: "
        log_str += f"mean {summary['env_reward_mean']:.2f} "
        log_str += f"± {(1.96 * summary['env_reward_std']) / (len(summary['raw_reward']) ** 0.5):.2f}\n"
        log_str += f"   Action {self.LOGGED_ACTION} fraction: {summary[f'action_{self.LOGGED_ACTION}_frac']:7.2f}\n"
        log_str += "   Raw Rewards: "
        for r in summary["raw_reward"]:
            log_str += f"{r:.2f},"
        logging.info(log_str)

        return summary