import os
import json
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.ticker as mticker


from constants import METHODS, ENVS


def create_grouped_bar_chart_with_legends_side_by_side(data, output_filename):
    """
    Creates a grouped bar chart where hatch patterns distinguish results within each environment,
    and color distinguishes Env1's bars from the rest, with legends placed side by side at the top.

    Parameters:
    data (list of tuple): A list where each element is a tuple (env, result1, result2, result3).
    output_filename (str): The filename for saving the chart as a high-resolution PDF.
    """
    # Unpack data
    envs = [item[0].replace("-", "\n") for item in data]
    results = np.array([item[1:] for item in data])  # Convert results to a NumPy array for easier manipulation

    # Define bar properties
    n_groups = len(results[0]) // 2  # Number of results (result1, result2, result3)
    x = np.arange(len(envs))  # Group positions
    bar_width = 0.2 # Width of each bar
    colors = ['lightcoral', 'steelblue', 'seagreen']
    #base_color = 'steelblue'  # Base color for bars
    #alt_color = 'lightcoral'  # Alternate color for Env1
    hatches = ['', '.', '/', '-']  # Hatch patterns for results

    # Create the plot
    plt.figure(figsize=(20, 5))  # Larger figure to fit legends and chart
    for i in range(n_groups):
        bar_color = []
        for env in envs:
            if env in ENVS["minigrid"]:
                bar_color.append(colors[0])
            elif env in ENVS["procgen"]:
                bar_color.append(colors[1])
            else:
                bar_color.append(colors[2])

        #bar_color = [alt_color if env in ["DistShift", "DoorKey", "LavaGap"] else base_color for env in envs]
        plt.bar(
            x + i * bar_width,
            results[:, 2 * i],
            yerr=results[:, 2 * i + 1] * 1.96,
            #capsize=2,
            width=bar_width,
            color=bar_color,
            hatch=hatches[i],
            edgecolor='white',
            label=None  # Avoid repeating labels for every bar
        )

    # Add chart labels
    # plt.ylabel('Fractio of RLOracle performance', fontsize=14, labelpad=15)
    #plt.xlabel('Environment', fontsize=14)

    # Start x-axis at 0.5
    ax = plt.gca()
    ax.set_ylim(bottom=0.5, top=1.05)  # Adjust x-axis to fit the data

    # Set x-ticks and labels
    plt.xticks(x + bar_width * (n_groups - 1) / 2, envs, fontsize=13, rotation=0, ha='center')
    plt.yticks(fontsize=12)

    # Add gridlines
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # Remove top and right borders
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)

    # Add legend for hatch patterns
    hatch_handles = [plt.Rectangle((0, 0), 1, 1, facecolor='white', hatch=h, edgecolor='black') for h in hatches]
    hatch_labels = ["Best method", "+oracle validator", "+oracle policy proposer"]
    #hatch_labels = [f"Result {i+1}" for i in range(n_groups)]

    # Add legend for colors
    color_handles = [
        plt.Rectangle((0, 0), 1, 1, facecolor=colors[0], edgecolor='white', label='minigrid'),
        plt.Rectangle((0, 0), 1, 1, facecolor=colors[1], edgecolor='white', label='procgen'),
        plt.Rectangle((0, 0), 1, 1, facecolor=colors[2], edgecolor='white', label='cliport')
    ]

    # Combine legends side by side at the top
    hatch_legend = plt.legend(hatch_handles, hatch_labels, fontsize=16, title=None, loc="upper center", bbox_to_anchor=(0.3, 1.2), ncol=3)
    color_legend = plt.legend(color_handles, [h.get_label() for h in color_handles], fontsize=16, title=None, loc="upper center", bbox_to_anchor=(0.7, 1.2), ncol=3)
    plt.gca().add_artist(hatch_legend)  # Add the hatch legend to the plot

    # Adjust layout to fit legends and chart
    plt.tight_layout(rect=[0, 0, 1, 0.99])  # Reserve space for legends at the top

    # Save the chart as a high-resolution PDF
    plt.savefig(output_filename, format='pdf', dpi=300)

    # Display a confirmation message
    print(f"Grouped bar chart saved as {output_filename}")



with open("aggregated_results.json") as f:
    data = json.load(f)


plot_data = []
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

        result = sorted(result, key=lambda x: x[0][0])

        best_method = result[-1]
        eval_true_best_method = "eval" if best_method[-1] in  ["always_strong", "always_weak", "always_random_0.5"] else "eval_true"

        best_method = list(best_method)  # Convert tuple to list
        best_method[1] = "always_random" if best_method[1] == "always_random_0.5" else best_method[1]
        best_method = tuple(best_method)

        result = []
        for method in METHODS:
            if not method.startswith("rl"):
                continue

            eval_mode = "eval_true"

            key = "/".join((suite, env, method, eval_mode))

            result.append((data[key], method))

        result = sorted(result, key=lambda x: x[0][0])

        best_rl_method = result[-1]


        result = []
        for method in METHODS:
            if not method.startswith("rl"):
                continue

            eval_mode = "eval_sim"

            key = "/".join((suite, env, method, eval_mode))

            result.append((data[key], method))

        result = sorted(result, key=lambda x: x[0][0])

        best_rl_sim_method = result[-1]


        plot_data.append((
            env,
            best_method[0][0] / best_rl_method[0][0],
            best_method[0][1] / best_rl_method[0][0],
            data["/".join((suite, env, best_method[-1], eval_true_best_method))][0] / best_rl_method[0][0],
            data["/".join((suite, env, best_method[-1], eval_true_best_method))][1] / best_rl_method[0][0],
            best_rl_sim_method[0][0] / best_rl_method[0][0],
            best_rl_sim_method[0][1] / best_rl_method[0][0],
        ))

        #print(env, best_method[-1], "/".join((suite, env, best_method[-1], eval_true_best_method)))

create_grouped_bar_chart_with_legends_side_by_side(plot_data, "./final_plots/ratio_to_rl.pdf")

#create_bar_chart(plot_data, "final_plots/ratio_to_rl.pdf")
