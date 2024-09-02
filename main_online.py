from YRC.core import Evaluator
from YRC.core import HelpPolicy
from YRC.core import (
    PPOAlgorithm,
    DQNAlgorithm,
    SVDDAlgorithm,
    KDEAlgorithm,
    NonParametricAlgorithm,
    RandomAlgorithm,
)

# from YRC.core.configs import config_utils
from YRC.core.utils import logger_setup
import YRC.core.environment as env_factory
import YRC.core.policy as policy_factory
import YRC.core.algorithm as algo_factory
import YRC.core.configs.utils as config_utils
from YRC.policies import *

import flags


if __name__ == "__main__":

    args = flags.make()
    config = config_utils.load(args.config, flags=args)

    envs = env_factory.make(config)
    policy = policy_factory.make(config, envs["train"])
    algorithm = algo_factory.make(config, envs["train"])
    evaluator = Evaluator(config)

    #config.coord_policy.which = "strong"
    #policy = AlwaysPolicy(config, envs["train"])

    algorithm.train(
        policy,
        envs,
        evaluator,
        eval_splits=["val_id", "val_ood", "test"],
        train_split="train",
    )

    # help_policy = HelpPolicy(config.help_policy, alg_class, train_env)

    # algorithms = {"PPO": PPOAlgorithm, "DQN": DQNAlgorithm, "SVDD": SVDDAlgorithm, "KDE": KDEAlgorithm, "NonParam": NonParametricAlgorithm, "Random": RandomAlgorithm}.get(alg_class)
    # if not algorithms:
    #    raise ValueError(f"Unsupported algorithm: {alg_class}")

    # algorithm = algorithms(getattr(config.algorithm, alg_class), logger, train_env)

    """
    algorithm = YRC.core.algorithm.make(config)

    evaluator = Evaluator(config, envs=(eval_id_env, eval_ood_env))
    algorithm.train(coord_policy, evaluator, train_env)

    test_policy = HelpPolicy(config.help_policy, alg_class, test_env)
    print("Testing on the best model from the ID eval env")
    algorithm.test(test_policy, test_env, evaluator.best_id_index, id_evaluated=True)
    print("Testing on the best model from the OOD eval env")
    algorithm.test(test_policy, test_env, evaluator.best_ood_index, id_evaluated=False)
    """
