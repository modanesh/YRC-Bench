import os
import re
import json
import numpy as np
from collections import defaultdict
from constants import ENVS, METHODS

def parse_file(key, file_path):
    # Open the file and loop through all lines
    with open(file_path, "r") as file:
        for line in file:
            if "Raw Rewards" not in line:
                continue
            rewards = [float(value) for value in line.replace("Raw Rewards:", "").split(",") if value.strip()]
            result[key].extend(rewards)

result = defaultdict(list)
for suite in ENVS.keys():
    for env in ENVS[suite]:
        for method in METHODS:
            for qc in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]:
                qc_str = "qc" + str(qc).replace(".", "")
                for eval_mode in ["eval", "eval_sim", "eval_true"]:
                    dir_path = "../experiments/" + "_".join((env, method, qc_str))
                    if os.path.isdir(dir_path):
                        for root, dirs, files in os.walk(dir_path):
                            for file in files:
                                file_path = os.path.join(dir_path, file)  # Full path of the file
                                if eval_mode + "_seed" in file_path:
                                    key = "/".join((suite, env, method, str(qc), eval_mode))
                                    parse_file(key, file_path)

with open("./raw_results.json", "w") as f:
    json.dump(result, f, indent=2)


