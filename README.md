Added environments and benchmarks:
- Procgen
- Matterport
- Cliport
    - Useful link: https://medium.com/@limyoonaxi/common-bugs-you-may-encounter-while-installing-cliport-ef1790e1cc0a

  **IMPORTANT**: Cliport is heavily based on Ravens (link: https://github.com/google-research/ravens). Cliport contains additional tasks that incorporate human language instructions as additional inputs to the agent. 


### 2024-10-22
Concerns for OOD approach:
- Features are extracted from observations using a CNN. Since the CNN is fixed (not trained), we may lose some information that is useful for OOD detection.
- In case of loading an OOD model for future runs, we need to ensure that the feature extractor is the same as the one used to train the OOD model. This is because the feature extractor is used to extract features from observations, which are then used to calculate the OOD score. If the feature extractor is different (which it is since its parameters are initialized randomly), the OOD score will be calculated using different features, which may not be useful for OOD detection.
