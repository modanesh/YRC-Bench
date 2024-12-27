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


#### Supported Feature Types
- Raw image observations
- Weak agent's hidden features
- Weak agent's action distributions
- Raw image observations + Weak agent's hidden features
- Raw image observations + Weak agent's action distributions
- Weak agent's hidden features + Weak agent's action distributions
- Raw image observations + Weak agent's hidden features + Weak agent's action distributions
- Weighted feature inputs: Raw image observations + Weak agent's hidden features + Weak agent's action distributions 
  - Instead of using combinations directly, introduce trainable weights for each feature type (obs, hidden, dist) and optimize them during training. This approach can dynamically adjust the importance of each feature type.


#### To-Dos
- Ensemble Coordination Policies: 
  - Train multiple coordination policies, each specialized in a different feature subset, and ensemble their decisions using a meta-policy.