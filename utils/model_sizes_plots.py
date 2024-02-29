import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
import seaborn as sns

rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
rc('text', usetex=True)

colors = ['blue', 'red', 'black', 'green', 'purple']

model_types = ['Float Base Model', 'Float Model with Adapter', 'Float Merged Model', 'Int4 LLM.int8() Merged', 'Int4 GPTQ Merged']
model_sizes = [13, 13.18848, 13, 3.7, 3.7]

fig, ax1 = plt.subplots(figsize=(10, 6))

ax1.bar(model_types, model_sizes, color=colors[0], alpha=0.8)
ax1.set_xlabel('Model Type', fontsize=14)
ax1.set_ylabel('Model Sizes in GB', fontsize=14, color=colors[0])
ax1.tick_params(axis='y', labelcolor=colors[0])

plt.title('Average Time and Tokens per Second for Different Models - c59', fontsize=16)
plt.xticks(rotation=45, ha='right', fontsize=12)
plt.yticks(fontsize=12)
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()