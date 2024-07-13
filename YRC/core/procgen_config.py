from .base.base_config import BaseConfig


class ProcgenCfg(BaseConfig):
    exp_name = None
    env_name = 'coinrun'
    val_env_name = None
    start_level = 0
    num_levels = 0
    distribution_mode = 'easy'
    param_name = 'easy_200'
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

    class debug:
        algo = 'ppo'
        n_envs = 2
        n_steps = 64
        epoch = 1
        mini_batch_per_epoch = 4
        mini_batch_size = 512
        gamma = 0.999
        lmbda = 0.95
        learning_rate = 0.0005
        grad_clip_norm = 0.5
        eps_clip = 0.2
        value_coef = 0.5
        entropy_coef = 0.01
        normalize_adv = True
        normalize_rew = False
        use_gae = True
        architecture = 'impala'

    class easy:
        algo = 'ppo'
        n_envs = 64
        n_steps = 256
        epoch = 3
        mini_batch_per_epoch = 8
        mini_batch_size = 2048
        gamma = 0.999
        lmbda = 0.95
        learning_rate = 0.0005
        grad_clip_norm = 0.5
        eps_clip = 0.2
        value_coef = 0.5
        entropy_coef = 0.01
        normalize_adv = True
        normalize_rew = False
        use_gae = True
        architecture = 'impala'
    
    class easy_200:
        algo = 'ppo'
        n_envs = 256
        n_steps = 256
        epoch = 3
        mini_batch_per_epoch = 8
        mini_batch_size = 2048
        gamma = 0.999
        lmbda = 0.95
        learning_rate = 0.0005
        grad_clip_norm = 0.5
        eps_clip = 0.2
        value_coef = 0.5
        entropy_coef = 0.01
        normalize_adv = True
        normalize_rew = False
        use_gae = True
        architecture = 'impala'
    
    class hard:
        algo = 'ppo'
        n_envs = 128
        n_steps = 256
        epoch = 3
        mini_batch_per_epoch = 8
        mini_batch_size = 4096
        gamma = 0.999
        lmbda = 0.95
        learning_rate = 0.0005
        grad_clip_norm = 0.5
        eps_clip = 0.2
        value_coef = 0.5
        entropy_coef = 0.01
        normalize_adv = True
        normalize_rew = False
        use_gae = True
        architecture = 'impala'
    
    class hard_500:
        algo = 'ppo'
        n_envs = 256
        n_steps = 256
        epoch = 3
        mini_batch_per_epoch = 8
        mini_batch_size = 8192
        gamma = 0.999
        lmbda = 0.95
        learning_rate = 0.0005
        grad_clip_norm = 0.5
        eps_clip = 0.2
        value_coef = 0.5
        entropy_coef = 0.01
        normalize_adv = True
        normalize_rew = False
        use_gae = True
        architecture = 'impala'
    
    class hard_500_mem:
        algo = 'ppo'
        n_envs = 256
        n_steps = 256
        epoch = 3
        mini_batch_per_epoch = 8
        mini_batch_size = 8192
        gamma = 0.999
        lmbda = 0.95
        learning_rate = 0.0005
        grad_clip_norm = 0.5
        eps_clip = 0.2
        value_coef = 0.5
        entropy_coef = 0.01
        normalize_adv = True
        normalize_rew = False
        use_gae = True
        architecture = 'impala'
    
    class hard_rec:
        algo = 'ppo'
        n_envs = 256
        n_steps = 256
        epoch = 3
        mini_batch_per_epoch = 8
        mini_batch_size = 8192
        gamma = 0.999
        lmbda = 0.95
        learning_rate = 0.0005
        grad_clip_norm = 0.5
        eps_clip = 0.2
        value_coef = 0.5
        entropy_coef = 0.01
        normalize_adv = True
        normalize_rew = False
        use_gae = True
        architecture = 'impala'
    
    class hard_local_dev:
        algo = 'ppo'
        n_envs = 16
        n_steps = 256
        epoch = 3
        mini_batch_per_epoch = 8
        mini_batch_size = 8192
        gamma = 0.999
        lmbda = 0.95
        learning_rate = 0.0005
        grad_clip_norm = 0.5
        eps_clip = 0.2
        value_coef = 0.5
        entropy_coef = 0.01
        normalize_adv = True
        normalize_rew = False
        use_gae = True
        architecture = 'impala'
    
    class hard_local_dev_rec:
        algo = 'ppo'
        n_envs = 16
        n_steps = 256
        epoch = 3
        mini_batch_per_epoch = 8
        mini_batch_size = 8192
        gamma = 0.999
        lmbda = 0.95
        learning_rate = 0.0005
        grad_clip_norm = 0.5
        eps_clip = 0.2
        value_coef = 0.5
        entropy_coef = 0.01
        normalize_adv = True
        normalize_rew = False
        use_gae = True
        architecture = 'impala'
    
    class A100:
        algo = 'ppo'
        n_envs = 512
        n_steps = 256
        epoch = 3
        mini_batch_per_epoch = 16
        mini_batch_size = 32768  # 32768  # this is just a maximum
        gamma = 0.999
        lmbda = 0.95
        learning_rate = 0.0005  # should make larger?
        grad_clip_norm = 0.5
        eps_clip = 0.2
        value_coef = 0.5
        entropy_coef = 0.01
        normalize_adv = True
        normalize_rew = False
        use_gae = True
        architecture = 'impala'
    
    class A100_large:  # for larger model (16x params)
        algo = 'ppo'
        n_envs = 512
        n_steps = 256
        epoch = 3
        mini_batch_per_epoch = 16
        mini_batch_size = 2048  # vary this param to adjust for memory
        gamma = 0.999
        lmbda = 0.95
        learning_rate = 0.0005  # scale by 1 / sqrt(channel_scale)
        grad_clip_norm = 0.5
        eps_clip = 0.2
        value_coef = 0.5
        entropy_coef = 0.01
        normalize_adv = True
        normalize_rew = False
        use_gae = True
        architecture = 'impala'
