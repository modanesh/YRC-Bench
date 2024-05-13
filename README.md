Added environments and benchmarks:
- Procgen
- Matterport
- Cliport
    - Useful link: https://medium.com/@limyoonaxi/common-bugs-you-may-encounter-while-installing-cliport-ef1790e1cc0a


### Environment Structure
#### Procgen

```python
import gym3
from procgen.procgen import ProcgenGym3Env
env = ProcgenGym3Env(num=2, env_name="coinrun")
step = 0
while True:
    # take a random action
    env.act(gym3.types_np.sample(env.ac_space, bshape=(env.num,)))
    
    rew, obs, first = env.observe()
    print(f"step {step} reward {rew} first {first}")
    step += 1
```