import flags
import YRC.core.algorithm as algo_factory
import YRC.core.configs.utils as config_utils
import YRC.core.environment as env_factory
import YRC.core.policy as policy_factory
from YRC.core import Evaluator
from YRC.policies import *

if __name__ == "__main__":
    args = flags.make()
    args.eval_mode = True
    config = config_utils.load(args.config, flags=args)

    envs = env_factory.make(config)
    policy = policy_factory.make(config, envs["train"])
    if config.general.algorithm != "always":
        policy.load_model(os.join.path(config.experiment_dir, "best_val_id.ckpt"))
    evaluator = Evaluator(config.evaluation)

    evaluator.eval(policy, envs, ["test"])


