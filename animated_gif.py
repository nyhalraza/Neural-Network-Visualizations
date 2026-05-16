import numpy as np
import matplotlib.pyplot as plt
import imageio.v3 as iio
import io

layers = [5, 8, 8, 3]  # Defines the number of neurons in each layer
max_neurons = max(layers)
layer_dist = 2
neuron_dist = 1
plt.style.use('dark_background')

fig, ax = plt.subplots(figsize=(10, 4)) 
frames = []

# Pre-calculate neuron positions
neuron_positions = {}
all_positions = []
for i, l in enumerate(layers):
    x = i * layer_dist
    y_start = (max_neurons - l) / 2.0 * neuron_dist
    for j in range(l):
        pos = (x, y_start + j * neuron_dist)
        neuron_positions[(i, j)] = pos
        all_positions.append(pos)

print("Generating neural network frames...")

total_steps = len(layers) * 2 
num_frames = 20 

for frame_idx in range(num_frames):
    ax.clear()

    progress = frame_idx / (num_frames - 1) * (len(layers) - 1)

    for i in range(len(layers) - 1):
        for j1 in range(layers[i]):
            for j2 in range(layers[i + 1]):
                pos1 = neuron_positions[(i, j1)]
                pos2 = neuron_positions[(i + 1), j2]

                # Dynamic alpha based on pulse progress
                dist_to_center = abs(i + 0.5 - progress)
                # Connections light up as the pulse passes between layers
                alpha = max(0.05, 0.6 * np.exp(-4.0 * dist_to_center ** 2))

                # Active colors vs background color
                if alpha > 0.15:
                    color = '#58a6ff' 
                else:
                    color = '#8b949e'
                    alpha = 0.05

                ax.plot([pos1[0], pos2[0]], [pos1[1], pos2[1]],
                        color=color, lw=0.8, alpha=alpha, zorder=1)

    # --- Draw Neurons (Nodes) ---
    all_positions_np = np.array(all_positions)
    x_coords = all_positions_np[:, 0]
    y_coords = all_positions_np[:, 1]

    # Dynamic coloring for nodes
    node_alphas = []
    for pos in all_positions:
        dist_to_progress = abs(pos[0] / layer_dist - progress)
        node_alpha = max(0.1, 1.0 * np.exp(-6.0 * dist_to_progress ** 2))
        node_alphas.append(node_alpha)

    # Scatters nodes (all positions)
    ax.scatter(x_coords, y_coords, color='#f0f6fc',  
               s=150, edgecolors='#8b949e', zorder=5, alpha=node_alphas)

    ax.text(0, -0.5, "Input", ha='center', color='white', fontsize=10, alpha=node_alphas[0])
    ax.text((len(layers) - 1) * layer_dist, -0.5, "Output", ha='center', color='white', fontsize=10,
            alpha=node_alphas[-1])
    ax.set_title("FORWARD PROPAGATION ENGINE", color='#a2d9ff', pad=15, loc='left', fontfamily='monospace')

    # Cleanup visual clutter
    ax.set_xlim(-1, (len(layers) - 1) * layer_dist + 1)
    ax.set_ylim(-1, max_neurons * neuron_dist + 1)
    ax.axis('off')
    plt.tight_layout()

    # Buffer to Frame
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100, bbox_inches='tight', pad_inches=0)
    buf.seek(0)
    frames.append(iio.imread(buf))

# 3. Save Output
print("Saving NN GIF...")
iio.imwrite('nn_propagation.gif', frames, duration=100, loop=0)
plt.close()
print("Done! Check 'nn_propagation.gif'")

