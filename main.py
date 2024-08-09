from YRC.core import HelpPolicy
from YRC.core.utils import logger_setup
from YRC.core import environment
from YRC.core import PPOAlgorithm
from YRC.core import Evaluator
from YRC.core.configs import config_utils

if __name__ == '__main__':
    config = config_utils.parse_args()
    config = config_utils.merge(config, vars(config))
    logger = logger_setup(config)

    train_env, eval_env, test_env = environment.make_help_envs(config)
    help_policy = HelpPolicy.create_policy(config.help_policy, train_env)
    algorithm = PPOAlgorithm(config.algorithm, logger, train_env)

    evaluator = Evaluator(config.evaluation, logger, eval_env, test_env)
    algorithm.train(help_policy, evaluator, train_env=train_env)

    test_policy = HelpPolicy.create_policy(config.help_policy, test_env)
    algorithm.test(test_policy, test_env, evaluator.best_index)
