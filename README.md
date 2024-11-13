#### Supported Environments
- Procgen
- Cliport
    - Useful link: https://medium.com/@limyoonaxi/common-bugs-you-may-encounter-while-installing-cliport-ef1790e1cc0a

  **IMPORTANT**: Cliport is heavily based on Ravens (link: https://github.com/google-research/ravens). Cliport contains additional tasks that incorporate human language instructions as additional inputs to the agent. 


#### Supported Algorithms
- Random
- Always (always querying the same agent)
- Threshold-based (maximum logit, maximum probability, margin, entropy, energy)
- RL-based (PPO)
- OOD detection (DeepSVDD)