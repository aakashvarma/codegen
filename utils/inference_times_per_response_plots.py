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
run_labels = ['Inference 1', 'Inference 2', 'Inference 3', 'Inference 4']
average_times = {
    'c59': [[619.19, 667.11, 709.35, 705.12], [302.64, 301.61, 220.38, 279.33], [271.05, 288.68, 269.05, 247.32], [15.22, 14.84, 14.6, 14.84], [7.07, 4.88, 6.62, 7.32]],
    'c60': [[323.07, 324.57, 334.03, 320.77], [144.75, 141.17, 148.92, 147.31], [138.42, 140.44, 139.6, 145.02], [10.42, 10.38, 10.22, 10.65], [4.63, 4.5, 3.5, 3.21]]
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
    ax.set_ylabel('Prompt Inference Time (seconds)', fontsize=14)
    ax.set_title(f'SQL Prompt Inference Times for Different Models - {cluster}', fontsize=16)
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