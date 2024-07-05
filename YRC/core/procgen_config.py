from .base.base_config import BaseConfig


class ProcgenCfg(BaseConfig):
    exp_name = None
    env_name = 'coinrun'
    val_env_name = None
    start_level = 0
    num_levels = 0
    distribution_mode = 'easy'
    param_name = 'easy-200'
    device = 'cuda'
    num_timesteps = 25_000_000
    seed = 0
    log_level = 40
    num_checkpoints = 1
    weak_model_file = None
    strong_model_file = None
    random_percent = 0  # COINRUN: percent of environments in which coin is randomized (only for coinrun)
    key_penalty = 0  # HEIST_AISC: Penalty for picking up keys (divided by 10)
    step_penalty = 0  # HEIST_AISC: Time penalty per step (divided by 1000)
    rand_region = 0  # MAZE: size of region (in upper left corner) in which goal is sampled.
    config_path = './YRC/procgen_wrapper/configs'
    num_threads = 8
    switching_cost = 0.2
    strong_query_cost = 0.8

    class reward_range:
        class coinrun:
            easy = dict(min=5.0, max=10.0, timeout=1000.0)
            hard = dict(min=5.0, max=10.0, timeout=1000.0)

        class starpilot:
            easy = dict(min=1.5, max=35.0, timeout=1000.0)
            hard = dict(min=2.5, max=64.0, timeout=1000.0)

        class chaser:
            easy = dict(min=0.5, max=14.2, timeout=1000.0)
            hard = dict(min=0.5, max=13.0, timeout=1000.0)

    class policy:
        n_steps = 128
        n_envs = 8
        epoch = 3
        mini_batch_per_epoch = 8
        mini_batch_size = 32 * 8
        gamma = 0.99
        lmbda = 0.95
        learning_rate = 2.5e-4
        grad_clip_norm = 0.5
        eps_clip = 0.2
        value_coef = 0.5
        entropy_coef = 0.01
        normalize_adv = True
        normalize_rew = True
        use_gae = True
