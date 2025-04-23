import torch
from torch_geometric.data import Data

# Tell PyTorch you trust loading Data objects
torch.serialization.add_safe_globals([Data])

# Now load the full graph object
graph = torch.load("graph_127.pt", weights_only=False)

print(graph)

# Individual components
print("Node feature shape:", graph.x.shape)      # [num_nodes, num_features]
print("Edge index shape:", graph.edge_index.shape)  # [2, num_edges]
print("Target temperature shape:", graph.y.shape)   # [num_nodes]

# Optional: visualize node positions colored by temperature
import matplotlib.pyplot as plt

# Extract node features
x_pos = graph.x[:, 0].numpy()
y_pos = graph.x[:, 1].numpy()
k_vals = graph.x[:, 2].numpy()
bc_labels = graph.x[:, 3].numpy()
temps = graph.y.numpy()

print(graph.edge_index)
print(graph.y)

# Plot temperature at nodes
plt.figure(figsize=(6, 5))
plt.scatter(y_pos, x_pos, c=temps, cmap='coolwarm', s=60)
plt.colorbar(label="Temperature (°C)")
plt.title("Node Temperatures")
plt.xlabel("x")
plt.ylabel("y")
plt.axis('equal')
plt.grid(True)
plt.tight_layout()
plt.show()
