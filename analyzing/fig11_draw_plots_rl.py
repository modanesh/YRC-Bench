import json
import matplotlib.pyplot as plt

# Load JSON data
with open('./aggregated_results.json', 'r') as f:
    data = json.load(f)

# Organize data by environment (only RL methods with eval_true)
env_data = {}
for key in data:
    parts = key.split('/')
    if len(parts) < 4:
        continue

    env = parts[1]
    method_type = parts[2]
    eval_type = parts[3]

    if method_type.startswith('rl') and eval_type == 'eval_true':
        if env not in env_data:
            env_data[env] = {}
        env_data[env][method_type] = data[key]

# Calculate number of rows and columns needed
n_envs = len(env_data)
n_cols = 5
n_rows = (n_envs + n_cols - 1) // n_cols  # Ceiling division to get number of rows

# Create figure with exact number of subplots needed
fig = plt.figure(figsize=(22, 20))
gs = fig.add_gridspec(n_rows, n_cols)


# Define colors based on whether method contains 'obs'
def get_color(method_name):
    return 'steelblue' if 'obs' in method_name else 'lightcoral'


# Create legend elements
legend_elements = [
    plt.Rectangle((0, 0), 1, 1, color='steelblue', alpha=0.7, label='obs-based'),
    plt.Rectangle((0, 0), 1, 1, color='lightcoral', alpha=0.7, label='non-obs')
]

# Calculate the starting column for centering the last row
last_row_items = n_envs - (n_rows - 1) * n_cols
start_col = (n_cols - last_row_items) // 2 if last_row_items < n_cols else 0

# Plot each environment
for idx, (env, methods) in enumerate(env_data.items()):
    row = idx // n_cols
    col = idx % n_cols

    # Adjust column position for the last row
    if row == n_rows - 1:
        col = start_col + (idx - (n_rows - 1) * n_cols)

    ax = fig.add_subplot(gs[row, col])

    # Prepare data in order of appearance in the JSON
    labels = [label.replace('rl_', '').replace('_', '\n') for label in methods.keys()]
    means = [methods[method][0] for method in methods]
    errors = [methods[method][1] for method in methods]
    colors = [get_color(method) for method in methods]

    # Create plot with thinner bars (width=0.6)
    x = range(len(labels))
    bars = ax.bar(x, means, yerr=errors, color=colors, alpha=0.7, capsize=5, width=0.6)

    ax.set_title(f'Environment: {env}', fontsize=16)

    ax.set_xticks([])
    ax.set_xticks([], minor=True)

    ax.grid(True, axis='y', linestyle='--', alpha=0.7)

    # Add x-axis labels only to bottom row
    if row == n_rows - 1:
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=0, ha='center', fontsize=12)

    # Add y-axis label to leftmost plots
    if col == (start_col if row == n_rows - 1 else 0):
        ax.set_ylabel('Performance', fontsize=16)

# Add legend above the plots
fig.legend(
    handles=legend_elements,
    loc='upper center',
    bbox_to_anchor=(0.51, 1.03),
    ncol=2,
    fontsize=18
)

# Adjust layout
plt.tight_layout()
plt.savefig("./final_plots/rl_performance.pdf", bbox_inches='tight', pad_inches=0.1, format="pdf")
plt.close()