ENVS = {
    "minigrid": ["DistShift", "DoorKey", "LavaGap"],
    "procgen": ["bossfight", "caveflyer", "chaser", "climber", "coinrun", "dodgeball", "heist", "jumper", "maze", "ninja", "plunder"],
    "cliport": ["assembling-kits-seq", "packing-boxes-pairs", "put-block-in-bowl", "stack-block-pyramid-seq", "separating-piles"]
}

METHODS = ["always_strong",
        "always_weak",
        "random05",
        "always_random",
        "threshold_max_prob",
        "threshold_max_logit",
        "threshold_margin",
        "threshold_neg_entropy",
        "threshold_neg_energy",
        "ood_obs",
        "ood_hidden",
        "ood_hidden_obs",
        "ood_dist",
        "ood_obs_dist",
        "ood_hidden_dist",
        "ood_obs_hidden_dist",
        "rl_obs",
        "rl_hidden",
        "rl_hidden_obs",
        "rl_dist",
        "rl_obs_dist",
        "rl_hidden_dist",
        "rl_obs_hidden_dist",
        ]

METHOD_NAME_MAP = {
    'always_random': 'random',
    'always_strong': 'always_strong',
    'always_weak': 'always_weak',
    'ood_dist': 'ood_dist',
    'ood_hidden': 'ood_hidden',
    'ood_hidden_dist': 'ood_hidden_dist',
    'ood_hidden_obs': 'ood_hidden_obs',
    'ood_obs': 'ood_obs',
    'ood_obs_dist': 'ood_obs_dist',
    'ood_obs_hidden_dist': 'ood_obs_hidden_dist',
    'always_random_0.5': 'always_random_0.5',
    'threshold_margin': 'margin',
    'threshold_max_logit': 'max_logit',
    'threshold_max_prob': 'max_prob',
    'threshold_neg_energy': 'neg_energy',
    'threshold_neg_entropy': 'neg_entropy'
 }

