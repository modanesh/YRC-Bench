#### Supported Environments
- Procgen
- Cliport
    - Useful link: https://medium.com/@limyoonaxi/common-bugs-you-may-encounter-while-installing-cliport-ef1790e1cc0a

  **IMPORTANT**: Cliport is heavily based on Ravens (link: https://github.com/google-research/ravens). Cliport contains additional tasks that incorporate human language instructions as additional inputs to the agent. 
- MiniGrid

#### Supported Algorithms
- Random
- Always (always querying the same agent)
- Threshold-based (maximum logit, maximum probability, margin, entropy, energy)
- RL-based (PPO)
- OOD detection (DeepSVDD)


#### Supported Feature Types
- Raw image observations
- Weak agent's hidden features
- Weak agent's action distributions
- Raw image observations + Weak agent's hidden features
- Raw image observations + Weak agent's action distributions
- Weak agent's hidden features + Weak agent's action distributions
- Raw image observations + Weak agent's hidden features + Weak agent's action distributions



#### Experiments
- procgen:
  - DONE: train always, threshold, ood
  - TO DO: 
    - train rl
    - eval sim with correct command all algs
    - eval sim and true for multiple seeds

- minigrid:
  - DONE: train always, threshold, ood
  - TO DO: 
    - train rl
    - eval sim with correct command all algs
    - eval sim and true for multiple seeds
  
- cliport:
  - TO DO: 
    - train always
    - train threshold
    - train ood
    - train rl
