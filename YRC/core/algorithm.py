import logging
import importlib


def make(config, env):
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

        best_summary = {}
        for split in eval_splits:
            best_summary[split] = {"reward_mean": -1e9}

        for iteration in trange(args.num_iterations):
            logging.info(f"Iteration {iteration}")
            # evaluate the first model as well
            if iteration % args.log_freq == 0:
                split_summary = evaluator.eval(policy, envs, eval_splits)

                for split in eval_splits:
                    if (
                        split_summary[split]["reward_mean"]
                        > best_summary[split]["reward_mean"]
                    ):
                        best_summary[split] = split_summary[split]
                        policy.save_model(f"best_{split}", self.save_dir)
                        policy.save_model(f"best_{iteration}", self.save_dir)

                policy.save_model("last", args.save_dir)

                for split in eval_splits:
                    evaluator.write_summary(f"best_{split}", best_summary[split])

            train_summary = self.train_one_iteration(
                iteration, policy, train_env=envs[train_split], dataset=dataset
            )
            self.write_summary(train_summary)

        # close env after training
        envs[train_split].close()

    def init(self, policy, envs):
        pass
