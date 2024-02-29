import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
import seaborn as sns

# Use LaTeX fonts
rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
rc('text', usetex=True)

# Define LaTeX color names (replace with desired defaults if needed)
colors = ['black', 'red', 'blue', 'green', 'purple']

# Data
model_types = ['Float Base Model', 'Float Model with Adapter', 'Float Merged Model', 'Int4 LLM.int8() Merged', 'Int4 GPTQ Merged']
clusters = ['c59', 'c60']
run_labels = ['Run 1', 'Run 2', 'Run 3', 'Run 4']
average_times = {
    'c59': [[68.69, 69.68, 63.39, 62.82], [65.91, 65.64, 78.46, 83.64], [71.25, 73.44, 72.3, 69.19], [9.91, 9.58, 10.14, 9.77], [4.39, 5.9, 5.46, 6.26]],
    'c60': [[45.3, 40.09, 35.4, 40.7], [42.06, 44.26, 48.83, 44.11], [38.71, 46.74, 43.81, 41.22], [5.45, 4.88, 4.96, 5.02], [2.53, 2.63, 2.8, 2.63]]
}

# Plot bar graphs for each cluster
for cluster in clusters:
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(model_types))
    width = 0.15

    # Loop through runs and plot bars with defined colors
    for i, run_label in enumerate(run_labels):
        ax.bar(x + i*width - (width*len(run_labels))/2, [times[i] for times in average_times[cluster]], width=width, label=run_label, color=colors[i], alpha=0.8)

    ax.set_xlabel('Model Type', fontsize=14)
    ax.set_ylabel('Inference Time (seconds)', fontsize=14)
    ax.set_title(f'Inference Times for Different Models - {cluster}', fontsize=16)
    ax.set_xticks(x)
    ax.set_xticklabels(model_types, rotation=45, ha='right', fontsize=12)
    ax.set_yticklabels(ax.get_yticks(), fontsize=12)
    ax.legend(fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()

# Plot line graph for average inference times across all runs
fig, ax = plt.subplots(figsize=(10, 6))

# Loop through clusters and plot lines with defined colors
for i, cluster in enumerate(clusters):
    avg_times = np.mean(average_times[cluster], axis=1)
    ax.plot(model_types, avg_times, marker='o', label=cluster, color=colors[i])

ax.set_xlabel('Model Type', fontsize=14)
ax.set_ylabel('Average Inference Time (seconds)', fontsize=14)
ax.set_title('Average Inference Times for Different Models', fontsize=16)
ax.set_xticklabels(model_types, rotation=45, ha='right', fontsize=12)
ax.set_yticklabels(ax.get_yticks(), fontsize=12)
ax.legend(fontsize=12)
ax.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()