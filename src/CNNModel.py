import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, random_split
import numpy as np
import matplotlib.pyplot as plt
import time

class TemperatureCNN(nn.Module):
    def __init__(self):
        super(TemperatureCNN, self).__init__()
        
        self.encoder = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128)
        )

        self.decoder = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),

            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),

            nn.Conv2d(32, 1, kernel_size=1)  # Output: temperature
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

def train_step(model, train_loader, loss_fn, optimizer):
    model.train()
    total_loss = 0.0

    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_fn(outputs, targets)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(train_loader)

def test_step(model, test_loader, loss_fn):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = loss_fn(outputs, targets)
            total_loss += loss.item()
        
        return total_loss / len(test_loader)
    
if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # === Load Data ===
    X = torch.tensor(np.load("cnn_inputs.npy"), dtype=torch.float32).permute(0, 3, 1, 2)  # (1000, 4, 50, 50)
    Y = torch.tensor(np.load("cnn_targets.npy"), dtype=torch.float32).unsqueeze(1)  # (1000, 1, 50, 50)

    # === Dataset and Dataloaders ===
    full_dataset = TensorDataset(X, Y)
    train_dataset, test_dataset = random_split(full_dataset, [800, 200])

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    # === Model, Loss, Optimizer ===
    model = TemperatureCNN().to(device)
    criterion = nn.MSELoss()#nn.MSELoss() nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    epochs = 200
    train_losses = []
    test_losses = []

    #Time the training
    start_time = time.time()
    for epoch in range(epochs):
        train_loss = train_step(model, train_loader, criterion, optimizer)
        test_loss = test_step(model, test_loader, criterion)

        train_losses.append(train_loss)
        test_losses.append(test_loss)

        print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f} - Test Loss: {test_loss:.4f}")

    end_time = time.time()
    print(f"Training Time: {end_time - start_time:.2f} seconds")

    # === Save Model ===
    torch.save(model.state_dict(), "temperature_cnn.pth")
    print("Model saved as temperature_cnn.pth")

    plt.plot(train_losses, label='Train Loss')
    plt.plot(test_losses, label='Test Loss')
    plt.xlabel("Epoch")
    plt.ylabel("Loss (MSE)")
    plt.yscale("log")
    plt.legend()
    plt.title("Loss over Epochs")
    plt.show()

    # === Visualize predictions on a few test samples ===
    model.eval()
    with torch.no_grad():
        sample_inputs, sample_targets = next(iter(test_loader))
        sample_inputs = sample_inputs.to(device)
        preds = model(sample_inputs).cpu().squeeze().numpy()
        sample_targets = sample_targets.squeeze().numpy()
        sample_inputs = sample_inputs.cpu()

    # Pick a few examples to plot
    for i in range(3):
        gt = sample_targets[i].T
        pred = preds[i].T
        vmin = min(gt.min(), pred.min())
        vmax = max(gt.max(), pred.max())

        fig, axs = plt.subplots(1, 2, figsize=(12, 4))
        fig.subplots_adjust(right=0.750)  # make room for colorbar

        im0 = axs[0].imshow(gt, cmap='coolwarm', origin='lower', vmin=vmin, vmax=vmax)
        axs[0].set_title("Ground Truth")

        im1 = axs[1].imshow(pred, cmap='coolwarm', origin='lower', vmin=vmin, vmax=vmax)
        axs[1].set_title("Prediction")

        # Add colorbar outside the plots
        cbar_ax = fig.add_axes([0.48, 0.10, 0.02, 0.7])  # [left, bottom, width, height]
        fig.colorbar(im1, cax=cbar_ax, label="Temperature (°C)")

        plt.suptitle(f"Test Sample {i+1}")
        plt.tight_layout()
        plt.show()






