import json
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import matplotlib.gridspec as gridspec



# Load JSON data
with open('./aggregated_results.json', 'r') as f:
    data = json.load(f)

# Organize data by environment, but merge different versions of threshold and ood
merged_data = {}
for key in data:
    parts = key.split('/')
    if len(parts) < 4:
        continue

    env = parts[1]
    method_type = parts[2]
    eval_type = parts[3]

    # Skip filtered environments
    if env in ["starpilot", "bigfish", "fruitbot", "leaper", "miner", "Dynamic-Obstacles", "GoToDoor"]:
        continue

    # Only process threshold and ood methods with eval_sim
    if (method_type.startswith('threshold') or method_type.startswith('ood')) and eval_type == 'eval_sim':
        if env not in merged_data:
            merged_data[env] = {'threshold': [], 'ood': []}

        # Store the performance value based on method type
        if method_type.startswith('threshold'):
            merged_data[env]['threshold'].append(data[key][0])
        elif method_type.startswith('ood'):
            merged_data[env]['ood'].append(data[key][0])

n_envs = len(merged_data)
n_cols = 12
last_row_items = n_envs - n_cols  # e.g. 19 envs → 9 in bottom row

# Create figure
fig = plt.figure(figsize=(16, 6))

# Outer GridSpec: 2 rows, 1 column
outer = gridspec.GridSpec(
    nrows=2, ncols=1,
    height_ratios=[1, 1],
    hspace=0.3
)

# Top row: 1×10 sub-GridSpec
top_gs = outer[0].subgridspec(
    nrows=1, ncols=n_cols,
    wspace=0.4
)

# Bottom row: 1×9 sub-GridSpec
bottom_gs = outer[1].subgridspec(
    nrows=1, ncols=last_row_items,
    wspace=0.5
)

# Colors for bars
colors = {'threshold': '#d62728', 'ood': '#9467bd'}

# Loop through environments and plot
for idx, (env, methods) in enumerate(merged_data.items()):
    # Choose which GridSpec to use
    if idx < n_cols:
        ax = fig.add_subplot(top_gs[0, idx])
    else:
        ax = fig.add_subplot(bottom_gs[0, idx - n_cols])

    # Prepare data for bars
    labels, means, stds, color_list = [], [], [], []
    if methods.get('threshold'):
        labels.append('threshold')
        means.append(np.mean(methods['threshold']))
        stds.append(np.std(methods['threshold']))
        color_list.append(colors['threshold'])
    if methods.get('ood'):
        labels.append('ood')
        means.append(np.mean(methods['ood']))
        stds.append(np.std(methods['ood']))
        color_list.append(colors['ood'])

    # Bar positioning
    bar_width = 0.25
    total_width = bar_width * len(labels)
    start_x = (1 - total_width) / 2
    x_positions = [start_x + i * bar_width for i in range(len(labels))]

    # Plot bars with error bars
    ax.bar(
        x_positions, means,
        yerr=stds,
        color=color_list,
        alpha=0.7,
        capsize=4,
        width=bar_width,
        linewidth=0.5
    )

    # Title and styling
    env_display = env.replace('-', '\n', 1)
    ax.set_title(env_display, fontsize=13, pad=8)
    ax.grid(True, axis='y', linestyle=':', alpha=0.6)
    ax.set_xticks([])
    ax.tick_params(axis='y', labelsize=8)

    # Add y-label to first column of each row
    row = 0 if idx < n_cols else 1
    col_in_row = idx if row == 0 else idx - n_cols
    if col_in_row == 0:
        ax.set_ylabel('Performance', fontsize=14, labelpad=5)

# Global legend
legend_elements = [
    plt.Rectangle((0, 0), 1, 1, color=colors['threshold'], alpha=0.7, label='Logit'),
    plt.Rectangle((0, 0), 1, 1, color=colors['ood'],       alpha=0.7, label='OOD')
]
fig.legend(
    handles=legend_elements,
    loc='upper center',
    bbox_to_anchor=(0.5, 0.99),
    ncol=2,
    fontsize=16
)

# Save and close
plt.savefig(
    "./final_plots/logit_ood_summary.pdf",
    bbox_inches='tight',
    format="pdf"
)
plt.close()






