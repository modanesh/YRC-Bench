import os
import sys
import logging
import time
import yaml
from datetime import datetime

import torch
import numpy as np


from YRC.core.configs import ConfigDict
from YRC.core.configs.global_configs import set_global_variable


def load(yaml_file_or_str, flags=None):
    if yaml_file_or_str.endswith(".yaml"):
        with open(yaml_file_or_str) as f:
            config_dict = yaml.safe_load(f)
    else:
        config_dict = yaml.safe_load(yaml_file_or_str)

    with open("configs/common.yaml") as f:
        common_config = yaml.safe_load(f)
        common_config = ConfigDict(**common_config)

    config = ConfigDict(**config_dict)
    algorithm = config.general.algorithm
    benchmark = config.general.benchmark
    config.coord_policy = getattr(common_config.coord_policy, algorithm)
    config.coord_env = getattr(common_config.coord_env, benchmark)
    config.evaluation = getattr(common_config.evaluation, benchmark)
    config.algorithm = getattr(common_config.algorithm, algorithm)
    config.environment = getattr(common_config.environment, benchmark)

    if flags is not None:
        config_dict = config.as_dict()
        update_config(flags.as_dict(), config_dict)
        config = ConfigDict(**config_dict)

    config.data_dir = os.getenv("SM_DATA_DIR", config.data_dir)
    output_dir = os.getenv("SM_OUTPUT_DIR", "experiments")
    config.experiment_dir = "%s/%s" % (output_dir, config.name)

    if not config.eval_mode and (config.overwrite is None or not config.overwrite):
        try:
            os.makedirs(config.experiment_dir)
        except:
            raise FileExistsError(
                "Experiment directory %s probably exists!" % config.experiment_dir
            )

    seed = config.general.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    # config.random = random.Random(seed)

    config.general.device = torch.device("cuda", config.general.device)
    set_global_variable("device", config.general.device)
    set_global_variable("benchmark", config.general.benchmark)
    set_global_variable("experiment_dir", config.experiment_dir)
    set_global_variable("seed", config.general.seed)

    config.start_time = time.time()
    if config.eval_mode:
        log_file = os.path.join(config.experiment_dir, "eval.log")
    else:
        log_file = os.path.join(config.experiment_dir, "run.log")
    if os.path.isfile(log_file):
        os.remove(log_file)
    _config_logging(log_file)
    logging.info(str(datetime.now()))
    logging.info("python -u " + " ".join(sys.argv[1:]))
    logging.info("Write log to %s" % log_file)
    logging.info(str(config))

    return config


def update_config(source, target):
    for k in source.keys():
        if isinstance(source[k], dict):
            if k not in target:
                target[k] = {}
            update_config(source[k], target[k])
        elif source[k] is not None:
            target[k] = source[k]


def _config_logging(log_file):
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(ElapsedFormatter())

    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(ElapsedFormatter())

    logging.basicConfig(
        level=logging.INFO, handlers=[stream_handler, file_handler], force=True
    )

    def handler(type, value, tb):
        logging.exception("Uncaught exception: %s", str(value))
        logging.exception("\n".join(traceback.format_exception(type, value, tb)))

    sys.excepthook = handler


class ElapsedFormatter:
    def __init__(self):
        self.start_time = datetime.now()

    def format_time(self, t):
        return str(t)[:-7]

    def format(self, record):
        elapsed_time = self.format_time(datetime.now() - self.start_time)
        log_str = "[%s %s]: %s" % (elapsed_time, record.levelname, record.getMessage())
        return log_str
