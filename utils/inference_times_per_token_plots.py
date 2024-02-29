import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
import seaborn as sns

# Use LaTeX fonts
rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
rc('text', usetex=True)

# Define LaTeX color names (replace with desired defaults if needed)
colors = ['black', 'red', 'blue', 'green', 'purple']

# Data from the table
model_types = ['Float Base Model', 'Float Model with Adapter', 'Float Merged Model', 'Int4 LLM.int8() Merged', 'Int4 GPTQ Merged']
average_times_c59 = [66.145, 73.4125, 71.545, 9.85, 5.5025]
tokens_per_second_c59 = [0.151183007, 0.136216584, 0.139772171, 1.015228426, 1.817355747]

average_times_c60 = [40.3725, 44.815, 42.62, 5.0775, 2.6475]
tokens_per_second_c60 = [0.247693356, 0.223139574, 0.234631628, 1.969473166, 3.777148253]

# Plot for c59
fig, ax1 = plt.subplots(figsize=(10, 6))

# Bar plot for average times
ax1.bar(model_types, average_times_c59, color=colors[0], alpha=0.8)
ax1.set_xlabel('Model Type', fontsize=14)
ax1.set_ylabel('Average Time (seconds)', fontsize=14, color=colors[0])
ax1.tick_params(axis='y', labelcolor=colors[0])

# Line plot for tokens per second
ax2 = ax1.twinx()
ax2.plot(model_types, tokens_per_second_c59, marker='o', color=colors[1])
ax2.set_ylabel('Tokens per second', fontsize=14, color=colors[1])
ax2.tick_params(axis='y', labelcolor=colors[1])

plt.title('Average Time and Tokens per Second for Different Models - c59', fontsize=16)
plt.xticks(rotation=45, ha='right', fontsize=12)
plt.yticks(fontsize=12)
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()

# Plot for c60
fig, ax1 = plt.subplots(figsize=(10, 6))

# Bar plot for average times
ax1.bar(model_types, average_times_c60, color=colors[0], alpha=0.8)
ax1.set_xlabel('Model Type', fontsize=14)
ax1.set_ylabel('Average Time (seconds)', fontsize=14, color=colors[0])
ax1.tick_params(axis='y', labelcolor=colors[0])

# Line plot for tokens per second
ax2 = ax1.twinx()
ax2.plot(model_types, tokens_per_second_c60, marker='o', color=colors[1])
ax2.set_ylabel('Tokens per second', fontsize=14, color=colors[1])
ax2.tick_params(axis='y', labelcolor=colors[1])

plt.title('Average Time and Tokens per Second for Different Models - c60', fontsize=16)
plt.xticks(rotation=45, ha='right', fontsize=12)
plt.yticks(fontsize=12)
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()

