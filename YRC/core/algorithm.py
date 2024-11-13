import logging
import importlib
import wandb

from YRC.core.configs import get_global_variable


def make(config, env):
    config.algorithm.use_wandb = config.use_wandb
    algorithm = getattr(
        importlib.import_module("YRC.algorithms"), config.algorithm.cls
    )(config.algorithm, env)
    return algorithm


class Algorithm:
    def train(
        self,
        policy,
        envs,
        evaluator=None,
        train_split=None,
        eval_splits=None,
        dataset=None,
    ):
        self.init(policy, envs)

        args = self.args
        save_dir = get_global_variable("experiment_dir")

        best_summary = {}
        for split in eval_splits:
            best_summary[split] = {"reward_mean": -1e9}

        train_log = {}

        for iteration in range(args.num_iterations):
            # evaluate the first model as well
            if iteration % args.log_freq == 0:
                wandb_log = {}
                wandb_log["step"] = self.global_step

                logging.info(f"Iteration {iteration}")
                if iteration > 0:
                    train_summary = self.summarize(train_log)
                    logging.info(f"Train {self.global_step} steps:")
                    self.write_summary(train_summary)
                    self.update_wandb_log(wandb_log, "train", train_summary)

                if not args.no_eval:
                    split_summary = evaluator.eval(policy, envs, eval_splits)
                    for split in eval_splits:
                        if (
                            split_summary[split]["reward_mean"]
                            > best_summary[split]["reward_mean"]
                        ):
                            best_summary[split] = split_summary[split]
                            policy.save_model(f"best_{split}", save_dir)
                            # policy.save_model(f"best_{iteration}", save_dir)

                        logging.info(f"Best {split} so far")
                        evaluator.write_summary(f"best_{split}", best_summary[split])

                        self.update_wandb_log(wandb_log, split, split_summary[split])
                        self.update_wandb_log(
                            wandb_log, f"best_{split}", best_summary[split]
                        )

                policy.save_model("last", save_dir)

                wandb.log(wandb_log)

            this_train_log = self.train_one_iteration(
                iteration, policy, train_env=envs[train_split], dataset=dataset
            )
            self.aggregate_log(train_log, this_train_log)

        # close env after training
        envs[train_split].close()

    def init(self, policy, envs):
        pass

    def update_wandb_log(self, wandb_log, split, summary):
        for k, v in summary.items():
            wandb_log[f"{split}/{k}"] = v