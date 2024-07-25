import os

from .base.base_config import BaseConfig


class CliportCfg(BaseConfig):
    cliport_root = os.getenv("CLIPORT_ROOT")
    if cliport_root is None or not os.path.exists(cliport_root) or cliport_root == "":
        raise ValueError("Please set the environment variable CLIPORT_ROOT to the root directory of the Cliport repository.")
    assets_root = f'{cliport_root}/cliport/environments/assets'
    disp = False
    shared_memory = False
    task = None
    weak_n_demos = 100  # only to load weak agent
    seed = 11
    agent = 'cliport'
    buffer_size = 1_000
    device = 'cuda'
    num_checkpoints = 1
    num_timesteps = 200_000
    num_test_steps = 10
    n_rotations = 36
    update_epochs = 10
    val_repeats = 1
    save_steps = [2000, 4000, 10000, 40000, 120000, 200000, 400000, 800000, 1200000]
    weak_agent_lr = 1e-4  # only to load weak agent
    attn_stream_fusion_type = 'add'
    trans_stream_fusion_type = 'conv'
    lang_fusion_type = 'mult'
    batchnorm = False
    batch_size = 16
    switching_cost = 0.2
    strong_query_cost = 0.8
    weak_model_file = None

    class policy:
        n_steps = 128
        n_envs = 1
        epoch = 3
        mini_batch_per_epoch = 4
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
        gae_lambda = 0.95
        n_train_episodes = 10
        seed = 0


class ProcgenCfg(BaseConfig):
    exp_name = None
    env_name = None
    val_env_name = None
    start_level = 0
    num_levels = 0
    distribution_mode = 'easy'
    device = 'cuda'
    num_timesteps = 25_000_000
    num_test_steps = 500_000
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

        class coinrun_aisc:
            easy = dict(min=5.0, max=10.0, timeout=1000.0)
            hard = dict(min=5.0, max=10.0, timeout=1000.0)

        class starpilot:
            easy = dict(min=2.5, max=64.0, timeout=1000.0)
            hard = dict(min=1.5, max=35.0, timeout=1000.0)

        class chaser:
            easy = dict(min=0.5, max=13., timeout=1000.0)
            hard = dict(min=0.5, max=14.2, timeout=1000.0)

        class caveflyer:
            easy = dict(min=3.5, max=12., timeout=1000.0)
            hard = dict(min=2.0, max=13.4, timeout=1000.0)

        class dodgeball:
            easy = dict(min=1.5, max=19., timeout=1000.0)
            hard = dict(min=1.5, max=19., timeout=1000.0)

        class fruitbot:
            easy = dict(min=-1.5, max=32.4, timeout=1000.0)
            hard = dict(min=-0.5, max=27.2, timeout=1000.0)

        class miner:
            easy = dict(min=1.5, max=13., timeout=1000.0)
            hard = dict(min=1.5, max=20., timeout=1000.0)

        class jumper:
            easy = dict(min=3., max=10., timeout=1000.0)
            hard = dict(min=1., max=10., timeout=1000.0)

        class leaper:
            easy = dict(min=3., max=10., timeout=1000.0)
            hard = dict(min=1.5, max=10., timeout=1000.0)

        class maze:
            easy = dict(min=5., max=10., timeout=1000.0)
            hard = dict(min=4., max=10., timeout=1000.0)

        class bigfish:
            easy = dict(min=1., max=40., timeout=1000.0)
            hard = dict(min=0., max=40., timeout=1000.0)

        class heist:
            easy = dict(min=3.5, max=10., timeout=1000.0)
            hard = dict(min=2., max=10., timeout=1000.0)

        class climber:
            easy = dict(min=2., max=12.6, timeout=1000.0)
            hard = dict(min=1., max=12.6, timeout=1000.0)

        class plunder:
            easy = dict(min=4.5, max=30., timeout=1000.0)
            hard = dict(min=3., max=30., timeout=1000.0)

        class ninja:
            easy = dict(min=3.5, max=10., timeout=1000.0)
            hard = dict(min=2., max=10., timeout=1000.0)

        class bossfight:
            easy = dict(min=.5, max=13., timeout=1000.0)
            hard = dict(min=.5, max=13., timeout=1000.0)

    class policy:
        algo = None
        n_envs = None
        n_steps = None
        epoch = None
        mini_batch_per_epoch = None
        mini_batch_size = None
        gamma = None
        lmbda = None
        learning_rate = None
        grad_clip_norm = None
        eps_clip = None
        value_coef = None
        entropy_coef = None
        normalize_adv = None
        normalize_rew = None
        use_gae = None
        architecture = None

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

    @classmethod
    def load_subclass_attributes(cls, subclass_name):
        subclass = getattr(cls, subclass_name, None)
        if subclass is None:
            raise ValueError(f"No subclass with name {subclass_name}")

        return {k: v for k, v in subclass.__dict__.items() if not k.startswith('__') and not callable(v)}