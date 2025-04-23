import numpy as np
import matplotlib.pyplot as plt
import torch
from torch_geometric.data import Data


# Reproducibility
np.random.seed(0)

# Grid setup
N = 50
x = np.linspace(0, 1, N)
y = np.linspace(0, 1, N)
X, Y = np.meshgrid(x, y)  # coordinate matrices of shape (64,64)


def solve_temperature_field(k):
    # Grid setup
    nx, ny = 50, 50
    Lx, Ly = 1.0, 1.0
    dx, dy = Lx / (nx - 1), Ly / (ny - 1)

    # Coordinates (for plotting)
    x = np.linspace(0, Lx, nx)
    y = np.linspace(0, Ly, ny)
    X, Y = np.meshgrid(x, y, indexing='ij')

    # Initialize temperature and thermal conductivity arrays
    T = np.zeros((nx, ny)) + 400

    # Boundary conditions
    T[:, -1] = 800.0   # Top: Dirichlet
    q_flux = 6000.0     # Bottom: Neumann

    # Iterative solver (Gauss-Seidel)
    max_iter = 100000
    tolerance = 1e-4

    for it in range(max_iter):
        T_old = T.copy()

        for i in range(1, nx - 1):
            for j in range(1, ny - 1):
                kx_plus = 0.5 * (k[i, j] + k[i+1, j])
                kx_minus = 0.5 * (k[i, j] + k[i-1, j])
                ky_plus = 0.5 * (k[i, j] + k[i, j+1])
                ky_minus = 0.5 * (k[i, j] + k[i, j-1])

                Ax = kx_plus + kx_minus
                Ay = ky_plus + ky_minus

                T[i, j] = (
                    kx_plus * T[i+1, j] + kx_minus * T[i-1, j] +
                    ky_plus * T[i, j+1] + ky_minus * T[i, j-1]
                ) / (Ax + Ay)

        # Neumann BC (Bottom): -k * dT/dy = q  => T[i, 0] = T[i,1] + q * dy / k
        for i in range(nx):
            T[i, 0] = T[i, 1] + q_flux * dy / k[i, 0]

        # Adiabatic BC (Left/Right): dT/dx = 0 => T[0,j] = T[1,j], T[-1,j] = T[-2,j]
        for j in range(ny):
            T[0, j] = T[1, j]
            T[-1, j] = T[-2, j]

        # Check convergence
        error = np.linalg.norm(T - T_old, ord=np.inf)
        if error < tolerance:
            print(f"✅ Converged after {it} iterations with max error {error:.2e}")
            break
    else:
        print("⚠️ Did not converge.")

    return T




num_samples = 1000  # Number of samples to generate
N=50
inputs = np.zeros((num_samples, 50, 50, 4), dtype=np.float32)
targets = np.zeros((num_samples, 50, 50), dtype=np.float32)

for n in range(num_samples):
    # Randomize conductivity parameters
    a = np.random.uniform(400, 600)
    b = np.random.uniform(200, 400)
    k_field = 600 + a * X + b * Y
    k_field += np.random.normal(scale=5.0, size=k_field.shape)  # small noise
    
    # Solve for temperature field
    T_field = solve_temperature_field(k_field)
    
    # Prepare input channels
    bc_type = np.zeros_like(T_field, dtype=np.int64)
    bc_type[49,:] = 1;  bc_type[0,:] = 2;  bc_type[:,0] = 3;  bc_type[:,49] = 3
    bc_type[49,0] = 1;  bc_type[49,49] = 1;  bc_type[0,0] = 2;  bc_type[0,49] = 2
    x_norm = X;  y_norm = Y
    k_norm = k_field / 1600.0
    
    inputs[n] = np.stack([x_norm, y_norm, k_norm, bc_type], axis=2)
    targets[n] = T_field
    
    # Now, let's create the graph dataset for the GNN
    # Use the same conductivity and temperature fields from the CNN sample n
    k_field = inputs[n, :, :, 2] * 1600.0   # (since we stored k_norm = k/1000)
    T_field = targets[n]
    bc_type_map = inputs[n, :, :, 3]       # BC type channel


    # Fixed 5x5 grid sampling over 50x50 domain
    grid_size = 5
    step = (N - 1) // (grid_size - 1)  
    node_is, node_js = [], []
    for i in range(grid_size):
        for j in range(grid_size):
            node_is.append(i * step)
            node_js.append(j * step)

    # Prepare node feature matrix
    node_feats = []
    node_temps = []
    for (i, j) in zip(node_is, node_js):
        x_norm = i / (N-1)
        y_norm = j / (N-1)
        k_val = k_field[j, i] 
        bc_val = bc_type_map[j, i]
        node_feats.append([x_norm, y_norm, k_val, bc_val])
        node_temps.append(T_field[j, i])
    node_feats = torch.tensor(node_feats, dtype=torch.float)          # shape [25, 4]
    node_temps = torch.tensor(node_temps, dtype=torch.float).view(-1,1)  # shape [25, 1]
    
    # Build edge index
    edges = []
    for a in range(25):
        ia, ja = divmod(a, 5)  # 5×5 grid: row = a//5, col = a%5
        for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:  # 4-neighbors
            ni, nj = ia + di, ja + dj
            if 0 <= ni < 5 and 0 <= nj < 5:
                b = ni * 5 + nj
                edges.append((a, b))
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()  # shape [2, E]
    
    # Create Data object for the graph
    data = Data(x=node_feats, edge_index=edge_index, y=node_temps)
    torch.save(data, f"graph_{n:03d}.pt")

    print(f"Sample {n+1}/{num_samples} done.")

# Save the CNN dataset to disk
np.save("cnn_inputs.npy", inputs)
np.save("cnn_targets.npy", targets)
print("Saved CNN dataset: cnn_inputs.npy (shape {}), cnn_targets.npy (shape {})"
    .format(inputs.shape, targets.shape))