import os
import sys
import logging
import time
import shutil
import random
import yaml
import errno
from datetime import datetime

import torch
import numpy as np


from YRC.core.configs import ConfigDict
from YRC.core.configs.global_configs import set_global_variable


def load(yaml_file_or_str, flags=None):

    if yaml_file_or_str.endswith('.yaml'):
        with open(yaml_file_or_str) as f:
            config_dict = yaml.safe_load(f)
    else:
        config_dict = yaml.safe_load(yaml_file_or_str)

    if flags is not None:
        update_config(flags, config_dict)

    config = ConfigDict(**config_dict)

    config.data_dir = os.getenv('SM_DATA_DIR', config.data_dir)
    output_dir = os.getenv('SM_OUTPUT_DIR', 'experiments')
    config.experiment_dir = "%s/%s" % (output_dir, config.name)

    try:
        os.makedirs(config.experiment_dir)
    except:
        #raise FileExistsError('Experiment directory %s probably exists!' % config.experiment_dir)
        pass

    seed = config.general.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    config.random = random.Random(seed)

    config.general.device = torch.device('cuda', config.general.device)

    set_global_variable("device", config.general.device)
    set_global_variable("benchmark", config.general.benchmark)

    config.start_time = time.time()
    log_file = os.path.join(config.experiment_dir, 'run.log')
    _config_logging(log_file)
    logging.info(str(datetime.now()))
    logging.info('Write log to %s' % log_file)
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

    logging.basicConfig(level=logging.INFO,
                        handlers=[stream_handler, file_handler],
                        force=True)

    def handler(type, value, tb):
        logging.exception("Uncaught exception: %s", str(value))
        logging.exception("\n".join(traceback.format_exception(type, value, tb)))

    sys.excepthook = handler


class ElapsedFormatter():

    def __init__(self):
        self.start_time = datetime.now()

    def format_time(self, t):
        return str(t)[:-7]

    def format(self, record):
        elapsed_time = self.format_time(datetime.now() - self.start_time)
        log_str = "[%s %s]: %s" % (elapsed_time,
                                record.levelname,
                                record.getMessage())
        return log_str
