Added environments and benchmarks:
- Procgen
- Matterport
- Cliport
    - Useful link: https://medium.com/@limyoonaxi/common-bugs-you-may-encounter-while-installing-cliport-ef1790e1cc0a

IMPORTANT: Cliport is heavily based on Ravens (link: https://github.com/google-research/ravens). Cliport contains additional tasks that incorporate human language instructions as additional inputs to the agent. 



## Setup LIBERO on CHAI cluster
```shell
export MUJOCO_GL="osmesa"
export MJLIB_PATH=/nas/ucb/$(whoami)/.mujoco/mujoco200/bin/libmujoco200.so
export LD_LIBRARY_PATH=/nas/ucb/$(whoami)/.mujoco/mujoco200/bin:$LD_LIBRARY_PATH
export MUJOCO_PY_MJPRO_PATH=/nas/ucb/$(whoami)/.mujoco/mujoco200/
```

## Setup RLBench on CHAI cluster
```shell
export COPPELIASIM_ROOT=/nas/ucb/mohamad/CoppeliaSim_Edu_V4_1_0_Ubuntu20_04
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$COPPELIASIM_ROOT
export QT_QPA_PLATFORM_PLUGIN_PATH=$COPPELIASIM_ROOT
export QT_QPA_PLATFORM=offscreen
```
