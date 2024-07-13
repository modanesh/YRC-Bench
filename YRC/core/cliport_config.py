import os

from .base.base_config import BaseConfig


class CliportCfg(BaseConfig):
    cliport_root = os.getenv("CLIPORT_ROOT")
    if cliport_root is None or not os.path.exists(cliport_root) or cliport_root == "":
        raise ValueError("Please set the environment variable CLIPORT_ROOT to the root directory of the Cliport repository.")
    assets_root = f'{cliport_root}/cliport/environments/assets'
    disp = False
    shared_memory = False
    task = 'stack-block-pyramid-seq-seen-colors'
    dataset_type = 'multi'
    data_dir = f'{cliport_root}/data'
    n_demos = 100
    n_val = 100
    results_path = f'{cliport_root}/exps'
    model_task = 'multi-language-conditioned'
    seed = 11
    model_path = f'{cliport_root}/exps'
    agent = 'cliport'
    buffer_size = 1_000
    device = 'cuda'
    learning_rate = 3e-4
    num_checkpoints = 1
    num_timesteps = 25_000_000
    n_iter = 1_000_000
    n_rotations = 36
    gamma = 0.99
    minibatch_size = 512
    update_epochs = 10
    norm_adv = True
    clip_coef = 0.2
    ent_coef = 0.0
    vf_coef = 0.5
    max_grad_norm = 0.5
    clip_vloss = True
    val_repeats = 1
    save_steps = [2000, 4000, 10000, 40000, 120000, 200000, 400000, 800000, 1200000]
    weak_agent_lr = 1e-4  # only to load weak agent
    attn_stream_fusion_type = 'add'
    trans_stream_fusion_type = 'conv'
    lang_fusion_type = 'mult'
    batchnorm = False
    log = False
    dataset_images = True
    dataset_cache = True
    batch_size = 16
    switching_cost = 0.2
    strong_query_cost = 0.8

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
        n_train_episodes = 2
        seed = 0

