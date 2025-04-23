# --- GNN Model for Temperature Prediction ---
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import DataLoader
from torch_geometric.data import Data
import matplotlib.pyplot as plt
import time
import numpy as np

# --- Model ---
class TemperatureGNN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.lin1 = nn.Linear(hidden_channels, hidden_channels)
        self.lin2 = nn.Linear(hidden_channels, out_channels)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.lin1(x))
        return self.lin2(x)

# --- Data loading ---
def load_graph_data(num_samples):
    data_list = []
    for i in range(num_samples):
        data = torch.load(f"graph_{i:03d}.pt", weights_only=False)
        data_list.append(data)
    return data_list

# --- Training step ---
def train(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0
    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data)
        loss = criterion(out, data.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

# --- Evaluation ---
def evaluate(model, loader, criterion):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            out = model(data)
            loss = criterion(out, data.y)
            total_loss += loss.item()
    return total_loss / len(loader)

# --- Setup ---
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

all_graphs = load_graph_data(1000)
train_graphs = all_graphs[:900]
test_graphs = all_graphs[900:]

train_loader = DataLoader(train_graphs, batch_size=32, shuffle=True)
test_loader = DataLoader(test_graphs, batch_size=32, shuffle=False)

model = TemperatureGNN(in_channels=4, hidden_channels=64, out_channels=1).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
criterion = nn.L1Loss()  # nn.MSELoss()

# --- Training loop ---
train_losses, test_losses = [], []
best_loss = float('inf')
patience, wait = 75, 0
max_epochs = 12500

#Time the training
start_time = time.time()
for epoch in range(1, max_epochs + 1):
    train_loss = train(model, train_loader, optimizer, criterion)
    test_loss = evaluate(model, test_loader, criterion)

    train_losses.append(train_loss)
    test_losses.append(test_loss)

    print(f"Epoch {epoch} | Train Loss: {train_loss:.4f} | Test Loss: {test_loss:.4f}")

    if test_loss < best_loss:
        best_loss = test_loss
        wait = 0
        torch.save(model.state_dict(), "gnn_best_model.pth")
    else:
        wait += 1
        if wait >= patience:
            print("Early stopping triggered.")
            break

end_time = time.time()
elapsed_time = end_time - start_time
print(f"Best lost: {best_loss:.4f}")
print(f"Total epochs: {epoch}")
print(f"Training completed in {elapsed_time:.2f} seconds.")
# --- Plot losses ---
plt.figure()
plt.plot(train_losses, label='Train Loss')
plt.plot(test_losses, label='Test Loss')
plt.yscale('log')
plt.xlabel('Epoch')
plt.ylabel('Loss (log scale)')
plt.legend()
plt.title("Loss vs Epochs")
plt.savefig("gnn_loss_plot.png")

# --- Inference visualization on 1 sample ---
model.load_state_dict(torch.load("gnn_best_model.pth"))
model.eval()

sample = test_graphs[0]
with torch.no_grad():
    pred = model(sample.to(device)).cpu().numpy()
    true = sample.y.cpu().numpy()
    xy = sample.x[:, :2].cpu().numpy()

# --- 1D Node vs Temp plot ---
plt.figure(figsize=(8, 4))
plt.plot(true, label="True Temp", marker='o')
plt.plot(pred, label="Predicted Temp", marker='x')
plt.xlabel("Node Index")
plt.ylabel("Temperature")
plt.title("Node-wise Temperature")
plt.legend()
plt.savefig("gnn_nodewise_comparison.png")

# --- 2D scatter ---
fig, axs = plt.subplots(1, 2, figsize=(10, 4))
sc0 = axs[0].scatter(xy[:, 1], xy[:, 0], c=true.squeeze(), cmap='coolwarm', s=80)
axs[0].set_title("Ground Truth Temp")
fig.colorbar(sc0, ax=axs[0])

sc1 = axs[1].scatter(xy[:, 1], xy[:, 0], c=pred.squeeze(), cmap='coolwarm', s=80)
axs[1].set_title("Predicted Temp")
fig.colorbar(sc1, ax=axs[1])

plt.suptitle("2D Spatial Comparison")
plt.savefig("gnn_spatial_comparison.png")

# --- Interpolate GNN 5x5 predictions to 50x50 grid ---
from scipy.interpolate import griddata

# Generate 50x50 regular grid in [0, 1] × [0, 1]
grid_x, grid_y = np.meshgrid(
    np.linspace(0, 1, 50),
    np.linspace(0, 1, 50),
    indexing='ij'
)

# Interpolate predicted temperatures
interp_pred = griddata(xy, pred.squeeze(), (grid_x, grid_y), method='cubic')
interp_true = griddata(xy, true.squeeze(), (grid_x, grid_y), method='cubic')

# Plot interpolated predictions vs ground truth
fig, axs = plt.subplots(1, 2, figsize=(12, 5))

# Compute shared vmin and vmax from both arrays
vmin = min(np.nanmin(interp_true), np.nanmin(interp_pred))
vmax = max(np.nanmax(interp_true), np.nanmax(interp_pred))

im0 = axs[0].imshow(interp_true, origin='lower', cmap='coolwarm', extent=[0, 1, 0, 1], vmin=vmin, vmax=vmax)
axs[0].set_title("True Temp (Interpolated)")

im1 = axs[1].imshow(interp_pred, origin='lower', cmap='coolwarm', extent=[0, 1, 0, 1], vmin=vmin, vmax=vmax)
axs[1].set_title("Predicted Temp (Interpolated)")

# Add a single shared colorbar on the right
cbar = fig.colorbar(im1, ax=axs, orientation='vertical', fraction=0.02, pad=0.04)
cbar.set_label("Temperature (°C)")


plt.suptitle("GNN Interpolated to 50x50 Grid")
plt.savefig("gnn_interpolated_50x50.png")
