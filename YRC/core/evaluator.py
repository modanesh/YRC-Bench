from YRC.cliport_wrapper.evaluators import Evaluator as CliportEvaluator
from YRC.procgen_wrapper.evaluators import Evaluator as ProcgenEvaluator

class Evaluator:
    @staticmethod
    def make(wrapper_type, *args, **kwargs):
        if wrapper_type == 'cliport':
            return CliportEvaluator(*args, **kwargs)
        elif wrapper_type == 'procgen':
            return ProcgenEvaluator(*args, **kwargs)
        else:
            raise ValueError(f"Unknown wrapper type: {wrapper_type}")
