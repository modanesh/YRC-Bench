import json
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

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

# Calculate number of rows and columns needed
n_envs = len(merged_data)
n_cols = 7
n_rows = (n_envs + n_cols - 1) // n_cols  # Ceiling division

# Create figure with adjusted layout
fig = plt.figure(figsize=(14, 12))  # Slightly wider figure
gs = fig.add_gridspec(n_rows, n_cols, wspace=0.4, hspace=0.3)  # Increased spacing

# Centering for last row
last_row_items = n_envs - (n_rows - 1) * n_cols
start_col = (n_cols - last_row_items) // 2 if last_row_items < n_cols else 0

colors = {'threshold': '#d62728', 'ood': '#9467bd'}

# Plot each environment with thinner bars
for idx, (env, methods) in enumerate(merged_data.items()):
    row = idx // n_cols
    col = idx % n_cols

    if row == n_rows - 1:
        col = start_col + (idx - (n_rows - 1) * n_cols)

    ax = fig.add_subplot(gs[row, col])

    labels = []
    means = []
    stds = []
    color_list = []

    # Process threshold data
    if methods['threshold']:
        labels.append('threshold')
        means.append(np.mean(methods['threshold']))
        stds.append(np.std(methods['threshold']))
        color_list.append(colors['threshold'])

    # Process OOD data
    if methods['ood']:
        labels.append('ood')
        means.append(np.mean(methods['ood']))
        stds.append(np.std(methods['ood']))
        color_list.append(colors['ood'])

    # Adjust bar parameters
    bar_width = 0.25  # Reduced bar width
    num_bars = len(labels)
    total_width = bar_width * num_bars
    start_x = (1 - total_width) / 2
    x_positions = [start_x + i * bar_width for i in range(num_bars)]

    # Create bars with error caps
    ax.bar(x_positions, means, yerr=stds, color=color_list, alpha=0.7,
           capsize=3, width=bar_width, linewidth=0.5)

    # Formatting
    env = env.replace('-', '\n', 1)
    ax.set_title(f'{env}', fontsize=14, pad=8)
    ax.grid(True, axis='y', linestyle=':', alpha=0.6)
    ax.set_xticks([])
    ax.tick_params(axis='y', labelsize=8)

    # Add y-label to leftmost plots
    if col == (start_col if row == n_rows - 1 else 0):
        ax.set_ylabel('Performance', fontsize=14, labelpad=5)

# Add legend
legend_elements = [
    plt.Rectangle((0, 0), 1, 1, color=colors['threshold'], alpha=0.7, label='Logit'),
    plt.Rectangle((0, 0), 1, 1, color=colors['ood'], alpha=0.7, label='OOD')
]
fig.legend(handles=legend_elements, loc='upper center',
           bbox_to_anchor=(0.5, 0.95), ncol=2, fontsize=16)

plt.tight_layout()
plt.savefig("./final_plots/logit_ood_summary.pdf", bbox_inches='tight', format="pdf")
plt.close()
