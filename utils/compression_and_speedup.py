# import numpy as np
# import matplotlib.pyplot as plt
# from matplotlib import rc
# import seaborn as sns

# # Use LaTeX fonts
# rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
# rc('text', usetex=True)

# # Define LaTeX color names (replace with desired defaults if needed)
# colors = ['blue', 'red', 'black', 'green', 'purple']

# # Data from the table
# model_types = ['Float Base Model', 'Float Model with Adapter', 'Float Merged Model', 'Int4 LLM.int8() Merged', 'Int4 GPTQ Merged']

# model_sizes_c60 = [13, 13.19, 13, 3.7, 3.7]
# tokens_per_second_c60 = [0.247693356, 0.223139574, 0.234631628, 1.969473166, 3.777148253]
# accuracy = [4.2, 86.5, 86.5, 86.4, 86.2]

# fig, ax1 = plt.subplots(figsize=(10, 6))

# # Bar plot for model sizes
# ax1.bar(model_types, model_sizes_c60, color=colors[0], alpha=0.8)
# ax1.set_xlabel('Model Type', fontsize=14)
# ax1.set_ylabel('Model size (GB)', fontsize=14, color=colors[0])
# ax1.tick_params(axis='y', labelcolor=colors[0])

# # Line plot for tokens per second
# ax2 = ax1.twinx()
# ax2.plot(model_types, tokens_per_second_c60, marker='o', color=colors[1])
# ax2.set_ylabel('Tokens per second', fontsize=14, color=colors[1])
# ax2.tick_params(axis='y', labelcolor=colors[1])

# plt.title('Average Time and Tokens per Second for Different Models - c60', fontsize=16)
# plt.xticks(rotation=45, ha='right', fontsize=12)
# plt.yticks(fontsize=12)
# plt.grid(True, linestyle='--', alpha=0.5)
# plt.tight_layout()
# plt.show()

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
import seaborn as sns

# Use LaTeX fonts
rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
rc('text', usetex=True)

# Define LaTeX color names (replace with desired defaults if needed)
colors = ['blue', 'red', 'green', 'purple']

# Data from the table
model_types = ['Float Base Model', 'Float Model with Adapter', 'Float Merged Model', 'Int4 LLM.int8() Merged', 'Int4 GPTQ Merged']

model_sizes_c60 = [13, 13.19, 13, 3.7, 3.7]
tokens_per_second_c60 = [0.247693356, 0.223139574, 0.234631628, 1.969473166, 3.777148253]
# accuracy = [4.2, 86.5, 86.5, 86.4, 86.2]

fig, ax1 = plt.subplots(figsize=(10, 6))

# Line plot for tokens per second
ax2 = ax1.twinx()
ax2.plot(model_types, tokens_per_second_c60, marker='o', color=colors[1], label='Tokens per Second')
ax2.set_ylabel('Tokens per second', fontsize=14, color=colors[1])
ax2.tick_params(axis='y', labelcolor=colors[1])

# Right hand y-axis for accuracy
# ax3 = ax1.twinx()
# ax3.plot(model_types, accuracy, marker='s', color=colors[2], label='Accuracy (%)')
# ax3.set_ylabel('Accuracy (%)', fontsize=14, color=colors[2])
# ax3.tick_params(axis='y', labelcolor=colors[2])
# ax1.bar(model_types, accuracy, color=colors[2], alpha=0.8, label='Accuracy (%)')
# ax1.set_xlabel('Model Type', fontsize=14)
# ax1.set_ylabel('Accuracy (%)', fontsize=14, color=colors[0])
# ax1.tick_params(axis='y', labelcolor=colors[0])

# Bar plot for model sizes
ax1.bar(model_types, model_sizes_c60, color=colors[0], alpha=0.8, label='Model Size (GB)')
ax1.set_xlabel('Model Type', fontsize=14)
ax1.set_ylabel('Model size (GB)', fontsize=14, color=colors[0])
ax1.tick_params(axis='y', labelcolor=colors[0])

plt.title('Model size and Tokens per Second for Different Models', fontsize=16)
plt.xticks(rotation=45, ha='right', fontsize=12)
plt.yticks(fontsize=12)
plt.grid(True, linestyle='--', alpha=0.5)

# Add legends for all three plots
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
lines3, labels3 = ax1.get_legend_handles_labels()
plt.legend(lines1 + lines2, labels1 + labels2, loc='lower left', bbox_to_anchor=(1, 1))

plt.tight_layout()
plt.show()
