import numpy as np
import logging
from YRC.core import Algorithm
from YRC.core.configs.global_configs import get_global_variable



class OODAlgorithm(Algorithm):
    def __init__(self, config, env):
        super().__init__()
        self.args = config
        self.env = env
        self.save_dir = get_global_variable("experiment_dir")

    def train(
        self,
        policy,
        envs,
        evaluator=None,
        train_split=None,
        eval_splits=None,
        dataset=None,
    ):
        args = self.args
        best_summary = {split: {"reward_mean": -1e9} for split in eval_splits}
        best_params = {}

        # Initialize OOD detector
        policy.initialize_ood_detector(args, envs["train"])

        # Generate rollouts for training OOD detector
        rollout_obs = policy.gather_rollouts(envs["train"], args.num_rollouts)
        rollout_obs_threshold = policy.gather_rollouts(envs["train"], args.num_rollouts)

        # Train OOD detector
        policy.clf.fit(rollout_obs, rollout_obs_threshold)

        # Threshold search
        thresholds_min, thresholds_max = policy.clf.decision_scores_.min(), policy.clf.decision_scores_.max()
        if thresholds_min == thresholds_max:
            cand_thresholds = [thresholds_min]
        else:
            cand_thresholds = np.linspace(thresholds_min, thresholds_max, args.num_thresholds)
        for threshold in cand_thresholds:
            params = {"threshold": threshold}
            logging.info(f"Evaluating threshold: {threshold}")

            policy.update_params(params)
            split_summary = evaluator.eval(policy, envs, eval_splits)

            for split in eval_splits:
                if split_summary[split]["reward_mean"] > best_summary[split]["reward_mean"]:
                    best_params[split] = params
                    best_summary[split] = split_summary[split]
                    policy.save_model(f"best_{split}", self.save_dir)

                # Log best result so far
                logging.info(f"Best {split} so far")
                logging.info(f"Parameters: {best_params[split]}")
                evaluator.write_summary(f"best_{split}", best_summary[split], envs[split].num_envs)

        policy.update_params(best_params[eval_splits[0]])  # Update with best params from first eval split

