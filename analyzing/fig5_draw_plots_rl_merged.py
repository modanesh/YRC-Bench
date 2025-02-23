import json
import matplotlib.pyplot as plt
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

    # Define hatches for bars based on "obs" in method name
    hatches = ["//" if "obs" in method else "" for method in methods]

    # Convert method names to more readable format
    new_methods = []
    for m in methods:
        if m == "always_strong":
            new_methods.append("always expert")
        elif m == "always_weak":
            new_methods.append("always novice")
        else:
            new_methods.append(m.replace("_", "\n").replace("rl", ""))
    # Set bar positions
    x_positions = range(len(new_methods))

    # Create the bar chart
    plt.figure(figsize=(10, 6))
    bars = plt.bar(x_positions, win_counts, hatch=hatches, color=bar_color, edgecolor='white', width=0.5)

    # Set labels and title
    plt.ylabel('Wins  (# of environments)', fontsize=16, labelpad=15)

    # Set x-ticks in the center of bars
    plt.xticks(x_positions, new_methods, rotation=0, ha='center', fontsize=16)

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

    special_patch = plt.Rectangle((0, 0), 1, 1, facecolor=bar_color, edgecolor='white', hatch='//', label='obs-based')
    regular_patch = plt.Rectangle((0, 0), 1, 1, facecolor=bar_color, edgecolor='white', label='non-obs')
    plt.legend(handles=[special_patch, regular_patch], fontsize=16)

    # Save the chart as a high-resolution PDF
    plt.tight_layout()  # Adjust layout to prevent clipping
    plt.savefig(output_filename, format='pdf', dpi=300)

    # Display a confirmation message
    print(f"Bar chart saved as {output_filename}")


# Load JSON data
with open('./aggregated_results.json', 'r') as f:
    data = json.load(f)

# Organize data by RL method across all environments
rl_methods_data = {}
for key in data:
    parts = key.split('/')
    if len(parts) < 4:
        continue
        
    env = parts[1]
    method_type = parts[2]
    eval_type = parts[3]
    
    # Filter only RL methods with eval_true
    if method_type.startswith('rl') and eval_type == 'eval_true':
        if method_type not in rl_methods_data:
            rl_methods_data[method_type] = []
        rl_methods_data[method_type].append(data[key][0])

max_count = {key: 0 for key in rl_methods_data.keys()}

# Iterate over each index in the lists
for i in range(len(next(iter(rl_methods_data.values())))):  # Assume all lists are of the same length
    # Find the maximum value at this index
    max_value = max(rl_methods_data[key][i] for key in rl_methods_data)

    # Find which keys have this max value and increment their count
    for key in rl_methods_data:
        if rl_methods_data[key][i] == max_value:
            max_count[key] += 1

# Convert dictionary to a sorted list of tuples
result = [(key, count) for key, count in max_count.items()]

create_bar_chart(result, "./final_plots/rl_summary_across_envs.pdf")















