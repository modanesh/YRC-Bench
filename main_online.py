from YRC.core import Evaluator
from YRC.core import HelpPolicy
from YRC.core import PPOAlgorithm, DQNAlgorithm, SVDDAlgorithm, KDEAlgorithm, NonParametricAlgorithm, RandomAlgorithm
from YRC.core import environment
from YRC.core.configs import config_utils
from YRC.core.utils import logger_setup

if __name__ == '__main__':
    config = config_utils.parse_args()
    config = config_utils.merge(config, vars(config))
    logger = logger_setup(config)
    alg_class = config.algorithm.cls

    train_env, eval_id_env, eval_ood_env, test_env = environment.make_help_envs(config)
    help_policy = HelpPolicy(config.help_policy, alg_class, train_env)

    algorithms = {"PPO": PPOAlgorithm, "DQN": DQNAlgorithm, "SVDD": SVDDAlgorithm, "KDE": KDEAlgorithm, "NonParam": NonParametricAlgorithm, "Random": RandomAlgorithm}.get(alg_class)
    if not algorithms:
        raise ValueError(f"Unsupported algorithm: {alg_class}")

    algorithm = algorithms(getattr(config.algorithm, alg_class), logger, train_env)

    evaluator = Evaluator(config.evaluation, logger, eval_id_env, eval_ood_env)
    algorithm.train(help_policy, evaluator, train_env)

    test_policy = HelpPolicy(config.help_policy, alg_class, test_env)
    print("Testing on the best model from the ID eval env")
    algorithm.test(test_policy, test_env, evaluator.best_id_index, id_evaluated=True)
    print("Testing on the best model from the OOD eval env")
    algorithm.test(test_policy, test_env, evaluator.best_ood_index, id_evaluated=False)
