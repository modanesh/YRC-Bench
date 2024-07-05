# YRC/core/__init__.py

from .environment import Environment
from .policy import Policy
from .utils import Logger
# from .evaluator import Evaluator
from .env_registry import env_registry
from .procgen_config import ProcgenCfg


__all__ = ['Environment', 'Policy', 'Logger']

env_registry.register(ProcgenCfg())
