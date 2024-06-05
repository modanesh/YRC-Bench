Added environments and benchmarks:
- Procgen
- Matterport
- Cliport
    - Useful link: https://medium.com/@limyoonaxi/common-bugs-you-may-encounter-while-installing-cliport-ef1790e1cc0a

  **IMPORTANT**: Cliport is heavily based on Ravens (link: https://github.com/google-research/ravens). Cliport contains additional tasks that incorporate human language instructions as additional inputs to the agent. 


Performance of CLIPort agent trained on seen splits of all 10 tasks with 1000 demonstrations from Oracle. The task is: `stack-block-pyramid-seq`. 
Task description: Build a pyramid of colored blocks in a color sequence specified through the step-by-step language instructions. Each task contains 6 blocks with randomized colors and 1 rectangular base, all initially placed at random poses

| Conditions    | Obtained pre-trained performance | Reported performance |
|---------------|----------------------------------|----------------------|
| Seen colors   | 97.3%                            | 96.8%                |
| Unseen colors | 34.5%                            | 31.7%                |