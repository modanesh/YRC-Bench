# YRC/core/__init__.py

from .environment import Environment
from .policy import Policy
from .utils import Logger
from .env_registry import env_registry
from .configs import ProcgenCfg, CliportCfg


__all__ = ['Environment', 'Policy', 'Logger', 'env_registry']
env_registry.register("procgen", ProcgenCfg())
env_registry.register("cliport", CliportCfg())

