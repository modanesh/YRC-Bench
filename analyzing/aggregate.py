import os
import json
import numpy as np
from collections import defaultdict
from metric import compute_metric
from constants import ENVS, METHODS


with open("./raw_results.json") as f:
    data = json.load(f)

result = defaultdict(list)
for suite in ENVS.keys():
    for env in ENVS[suite]:
        for method in METHODS:
            for eval_mode in ["eval", "eval_sim", "eval_true"]:
                qc_results = []
                for qc in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]:
                    qc_str = "qc" + str(qc).replace(".", "")
                    key = "/".join((suite, env, method, str(qc), eval_mode))
                    if key in data:
                        qc_results.append((qc, data[key]))
                if len(qc_results) == 0:
                    print(key)
                    res = (-1, 0)
                else:
                    res = compute_metric(qc_results)
                new_key ="/".join((suite, env, method, eval_mode))
                result[new_key] = res
                print(new_key, res)

with open("./aggregated_results.json", "w") as f:
    json.dump(result, f, indent=2)


