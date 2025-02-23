import json
import matplotlib.pyplot as plt

# Load JSON data
with open('./aggregated_results.json', 'r') as f:
    data = json.load(f)

# Organize data by environment
env_data = {}
for key in data:
    parts = key.split('/')
    env = parts[1]
    method_type = parts[2]
    eval_type = parts[3]

    if env not in env_data:
        env_data[env] = {}

    # Filter based on evaluation type
    if method_type == 'always_strong' and eval_type == 'eval':
        env_data[env]['always_strong'] = data[key]
    elif method_type == 'always_weak' and eval_type == 'eval':
        env_data[env]['always_weak'] = data[key]
    elif method_type == 'always_random' and eval_type == 'eval_sim':
        env_data[env]['always_random'] = data[key]
    elif method_type == 'always_random' and eval_type == 'eval':
        env_data[env]['fixed 0.5 random'] = data[key]
    elif method_type.startswith('threshold') and eval_type == 'eval_sim':
        env_data[env][method_type] = data[key]
    elif method_type.startswith('ood') and eval_type == 'eval_sim':
        env_data[env][method_type] = data[key]
    elif method_type.startswith('rl') and eval_type == 'eval_true':
        env_data[env][method_type] = data[key]

# Calculate number of rows and columns needed
n_envs = len(env_data)
n_cols = 5
n_rows = (n_envs + n_cols - 1) // n_cols  # Ceiling division to get number of rows

# Create figure with exact number of subplots needed
fig = plt.figure(figsize=(35, 22))
# gs = fig.add_gridspec(n_rows, n_cols)
gs = fig.add_gridspec(n_rows, n_cols, wspace=0.1, hspace=0.1)
# Color coding for different methods
colors = {
    'always_strong': '#1f77b4',
    'always_weak': '#ff7f0e',
    'always_random': '#2ca02c',
    'fixed 0.5 random': '#085c08',
    'threshold': '#d62728',
    'ood': '#9467bd',
    'rl': '#8c564b'
}

# Create legend elements once
legend_elements = []
for label, color in colors.items():
    if label == "threshold":
        label = "logit"
    elif label == "always_strong":
        label = "always expert"
    elif label == "always_weak":
        label = "always novice"
    label = label.replace('_', ' ')
    legend_elements.append(plt.Rectangle((0, 0), 1, 1, color=color, alpha=0.7, label=label))

# Calculate the starting column for centering the last row
last_row_items = n_envs - (n_rows - 1) * n_cols
start_col = (n_cols - last_row_items) // 2 if last_row_items < n_cols else 0

for idx, (env, methods) in enumerate(env_data.items()):
    row = idx // n_cols
    col = idx % n_cols

    # Adjust column position for the last row
    if row == n_rows - 1:
        col = start_col + (idx - (n_rows - 1) * n_cols)

    ax = fig.add_subplot(gs[row, col])

    # Prepare data
    labels = []
    means = []
    errors = []
    color_list = []

    # Order: always_strong, always_weak, always_random, always_random_baseline, thresholds, ood, rl
    categories = [
        ('always_strong', 'always_strong'),
        ('always_weak', 'always_weak'),
        ('always_random', 'always_random'),
        ('fixed 0.5 random', 'fixed 0.5 random'),
        (lambda x: x.startswith('threshold'), 'threshold'),
        (lambda x: x.startswith('ood'), 'ood'),
        (lambda x: x.startswith('rl'), 'rl')
    ]

    for category in categories:
        for method in methods:
            if (callable(category[0]) and category[0](method)) or method == category[0]:
                m = method.replace('rl_', '').replace('ood_', '').replace('threshold_', '').replace('strong', 'expert').replace('_', ' ').replace("weak", "novice")
                labels.append(m)
                means.append(methods[method][0])
                errors.append(methods[method][1])
                color_list.append(colors[category[1]])

    # Create plot
    x = range(len(labels))
    ax.bar(x, means, yerr=errors, color=color_list, alpha=0.7, capsize=5)

    ax.set_xticks([])
    ax.set_xticks([], minor=True)

    # Customize plot
    ax.set_title(f'Environment: {env}', fontsize=14)
    ax.grid(True, axis='y', linestyle='--', alpha=0.7)

    # Add x-axis labels only to bottom row
    if row == n_rows - 1:
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=90, ha='center', fontsize=12)

    # Add y-axis label to leftmost plots
    if col == (start_col if row == n_rows - 1 else 0):
        ax.set_ylabel('Performance', fontsize=16)

# Remove all manual subplots_adjust calls
# Add legend first with proper positioning
fig.legend(
    handles=legend_elements,
    loc='upper center',
    bbox_to_anchor=(0.5, 0.92),  # Position above the figure
    ncol=7,
    fontsize=14
)

# Then apply tight_layout
plt.tight_layout()

# Finally save with bbox_inches='tight' to auto-crop
plt.savefig("./final_plots/env_performance.pdf", bbox_inches='tight', pad_inches=0.1, format="pdf")