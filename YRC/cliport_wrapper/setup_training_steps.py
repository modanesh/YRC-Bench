from models import CategoricalPolicy, ImpalaModel, PPO


def agent_setup(env, policy, logger, storage, storage_valid, device, num_checkpoints, hyperparameters=None, pi_w=None, pi_o=None):
    print('::[LOGGING]::INTIALIZING AGENT...')
    agent = PPO(env, policy, logger, storage, device,
                num_checkpoints,
                storage_valid=storage_valid,
                pi_w=pi_w,
                pi_o=pi_o,
                **hyperparameters)
    return agent


def model_setup(env, configs):
    observation_shape = env.observation_space.shape
    in_channels = observation_shape[0]

    model = ImpalaModel(in_channels=in_channels)

    # trick to make the action space for pi_h to be 1: 0 means weak agent, 1 means oracle agent
    action_size = 2
    policy = CategoricalPolicy(model, action_size)
    policy.to(configs.device)
    return model, policy