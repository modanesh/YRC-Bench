Added environments and benchmarks:
- Procgen
- Matterport
- Cliport
    - Useful link: https://medium.com/@limyoonaxi/common-bugs-you-may-encounter-while-installing-cliport-ef1790e1cc0a

  **IMPORTANT**: Cliport is heavily based on Ravens (link: https://github.com/google-research/ravens). Cliport contains additional tasks that incorporate human language instructions as additional inputs to the agent. 


### How to
To run the code, configurations for each run can be modified. The set of available configs are located at: `YRC/core/configs/config.yaml`. Any parameter within that file can be modified by passing the desired value as an argument to the script. For example, to change the benchmark, the following command can be used:
```shell
python main_online.py --general.benchmark BENCHMARK_NAME
```

For the following, it is assumed that the weak and strong agents are available for the benchmark and the environment. To run the training, the file to the weak and strong agents should be passed as arguments. The following commands can be used to run the code for each benchmark:

- Procgen:
```shell
python main_online.py --general.benchmark procgen --acting_policy.weak.env_name coinrun --acting_policy.strong.env_name coinrun --acting_policy.weak.file PPO-procgen-coinrun-easy-200-2fc595d8/model_80019456.pth --acting_policy.strong.file PPO-procgen-coinrun-hard-500-bde01d68/model_80019456.pth --help_env.policy_type T3 --environments.procgen.train.env_name coinrun --environments.procgen.val.env_name starpilot --help_env.switching_cost 0.1 --help_env.strong_query_cost 0.1
```

- Cliport:
```shell
--general.benchmark cliport --acting_policy.weak.env_name multi-language-conditioned --acting_policy.weak.file multi-language-conditioned-cliport-n100-train/checkpoints/steps=300000-val_loss=0.00017400.ckpt --help_env.policy_type T1 --environments.cliport.train.env_name stack-block-pyramid-seq-seen-colors --environments.cliport.val.env_name stack-block-pyramid-seq-unseen-colors --help_env.switching_cost 0.1 --help_env.strong_query_cost 0.1 --algorithm.rollout_length 50 --acting_policy.weak.architecture cliport --evaluation.validation_steps 30
```