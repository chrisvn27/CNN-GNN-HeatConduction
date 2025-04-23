import numpy as np
import matplotlib.pyplot as plt

# Load files
inputs = np.load("cnn_inputs.npy")
targets = np.load("cnn_targets.npy")

# Show basic shapes
print("Inputs shape:", inputs.shape)     # Should be (1, 50, 50, 4) for 1 sample
print("Targets shape:", targets.shape)   # Should be (1, 50, 50)

# Inspect channels of first sample
x_pos = inputs[900, :, :, 0]
y_pos = inputs[900, :, :, 1]
k_values = inputs[900, :, :, 2]
bc_labels = inputs[900, :, :, 3]
temperature = targets[900]

# Plot each channel
fig, axs = plt.subplots(1, 4, figsize=(18, 4))
axs[0].imshow(x_pos, origin="lower",cmap='coolwarm'); axs[0].set_title("x positions")
axs[1].imshow(y_pos, origin="lower",cmap='coolwarm'); axs[1].set_title("y positions")
axs[2].imshow(k_values, origin="lower",cmap='coolwarm'); axs[2].set_title("k values (normalized)")
axs[3].imshow(bc_labels, origin="lower",cmap='coolwarm'); axs[3].set_title("BC labels")
plt.tight_layout()
plt.show()

# Plot the temperature output
plt.figure()
plt.imshow(temperature.T, cmap='coolwarm', origin='lower')
plt.colorbar(label="Temperature (°C)")
plt.title("Target Temperature Field")
plt.show()