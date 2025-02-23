import os
import json
from collections import defaultdict
from constants import METHODS, ENVS, METHOD_NAME_MAP
from pprint import pprint

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.ticker as mticker


def create_bar_chart(data, output_filename):
    """
    Creates a bar chart illustrating the number of wins for each method.

    Parameters:
    data (list of tuple): A list where each element is a tuple (method, win_count).
    output_filename (str): The filename for saving the chart as a high-resolution PDF.
    """
    # Unpack methods and win counts
    methods, win_counts = zip(*data)

    bar_color = "steelblue"

    # Define bar colors based on method groups
    special_methods = {"always_strong", "always_weak", "always_random_0.5"}
    #colors = ["steelblue" if method in special_methods else "pink" for method in methods]
    hatches = ["//" if method in special_methods else "" for method in methods]

    # Create the bar chart
    plt.figure(figsize=(10, 6))  # Adjust the figure size for better readability
    #plt.bar(methods, win_counts, color='skyblue', edgecolor='black')
    new_methods = []
    for m in methods:
        if m == "always_strong":
            new_methods.append("always expert")
        elif m == "always_weak":
            new_methods.append("always novice")
        else:
            new_methods.append(m.replace("_", " "))
    bars = plt.bar(new_methods, win_counts, hatch=hatches, color=bar_color, edgecolor='white', width=0.6)  # Reduced width for more space

    # Add chart title and labels
    plt.ylabel('Wins  (# of environments)', fontsize=14, labelpad=15)

    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45, ha='right', fontsize=14)

    plt.yticks(fontsize=16)

    # Ensure y-axis tick labels are integers
    plt.gca().yaxis.set_major_locator(mticker.MaxNLocator(integer=True))

    # Add gridlines for clarity
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # Remove top and right frame borders
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['right'].set_visible(False)

    special_patch = plt.Rectangle((0, 0), 1, 1, facecolor=bar_color, edgecolor='white', hatch='//', label='No simulated validator')
    regular_patch = plt.Rectangle((0, 0), 1, 1, facecolor=bar_color, edgecolor='white', label='Use simulated validator')
    plt.legend(handles=[special_patch, regular_patch], fontsize=14)

    # Save the chart as a high-resolution PDF
    plt.tight_layout()  # Adjust layout to prevent clipping
    plt.savefig(output_filename, format='pdf', dpi=300)

    # Display a confirmation message
    print(f"Bar chart saved as {output_filename}")


with open("./aggregated_results.json") as f:
    data = json.load(f)


cnt = defaultdict(int)
for suite in ENVS.keys():
    for env in ENVS[suite]:

        result = []
        for method in METHODS:
            if method.startswith("rl"):
                continue

            if method in ["always_strong", "always_weak", "random05"]:
                eval_mode = "eval"
            else:
                eval_mode = "eval_sim"

            if method == "random05":
                method = "always_random"
            key = "/".join((suite, env, method, eval_mode))
            if data[key][0] == -1:
                print(key)
            result.append((data[key], "always_random_0.5" if eval_mode == "eval" and method == "always_random" else method))


        for i in range(len(result)):
            is_best = 1
            for j in range(len(result)):
                if j != i and result[j][0] >= result[i][0]:
                    is_best = 0
                    break
            cnt[result[i][1]] += is_best


plot_data = list((METHOD_NAME_MAP[k], v) for k, v in cnt.items())

print(plot_data)

create_bar_chart(plot_data, "./final_plots/wins.pdf")
