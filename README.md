Added environments and benchmarks:
- Procgen
- Matterport
- Cliport
    - Useful link: https://medium.com/@limyoonaxi/common-bugs-you-may-encounter-while-installing-cliport-ef1790e1cc0a

  **IMPORTANT**: Cliport is heavily based on Ravens (link: https://github.com/google-research/ravens). Cliport contains additional tasks that incorporate human language instructions as additional inputs to the agent. 

### Most recent updates:
- procgen: changing/simplifying how configs are loaded
- cliport: changing/simplifying how configs are loaded
- cliport: getting max_reward and timeout from task
- cliport: getting seed as config parameter
- MAJOR: code refactoring and cleaning
- MAJOR - cliport: adding T1 and T2 types for help policy
- merging configs of cliport and procgen
- cliport: adding validation for help policy
- merging utils of cliport and procgen