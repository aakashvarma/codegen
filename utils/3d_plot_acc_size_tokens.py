import matplotlib.pyplot as plt
from matplotlib import rc
from mpl_toolkits.mplot3d import Axes3D

rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
rc('text', usetex=True)

# Sample data
model_names = ["Float Base model", "Float model with Adapter", "Float merged model",
               "Int4 LLM.int8() model", "Int4 GPTQ model"]
model_sizes = [13, 13.19, 13, 3.7, 3.7]  # Size in GB
tokens_per_second = [0.247693356, 0.223139574, 0.234631628, 1.969473166, 3.777148253]
accuracies = [4.2, 86.5, 86.5, 86.4, 86.2]

# Create a color list for different models
colors = ['red', 'green', 'blue', 'purple', 'orange']

# Create the figure and 3D axes
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection='3d')

# Plot the data points with different colors
for i, (model_name, x, y, z, color) in enumerate(zip(model_names, model_sizes, tokens_per_second, accuracies, colors)):
    ax.scatter3D(x, y, z, c=color, marker='o', label=model_name)

# Print data point coordinates
for i, (model_name, x, y, z) in enumerate(zip(model_names, model_sizes, tokens_per_second, accuracies)):
    print(f"{model_name}: (Model Size: {x:.2f} GB, Tokens per Second: {y:.6f}, Accuracy: {z:.2f})")

# Set labels for axes
ax.set_xlabel('Model Size (GB)')
ax.set_ylabel('Tokens per Second')
ax.set_zlabel('Accuracy')

# Set title for the plot
ax.set_title("Model Size, Tokens per Second, and Accuracy")

# Add legend with a bit more space away from the plot
ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')  # Adjust these values for placement

# Rotate the plot for better viewing angle
ax.view_init(elev=30, azim=60)  # Adjust these values for desired view

# Code for adding orthogonal lines (same as before)
for i, (x, y, z) in enumerate(zip(model_sizes, tokens_per_second, accuracies)):
    ax.plot3D([x, x], [0, y], [z, z], color='gray', linestyle='--')
    ax.plot3D([x, x], [y, y], [0, z], color='gray', linestyle='--')

plt.show()