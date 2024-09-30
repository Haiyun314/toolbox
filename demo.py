import numpy as np
import matplotlib.pyplot as plt

# Parameters
L = 1  # Length of the rod
N = 5  # Number of interior points
h = L / (N + 1)  # Grid spacing
nu = 0.1  # Regularization parameter

# Discretized grid points (without boundary)
x = np.linspace(h, L-h, N)

# Desired state (target temperature profile)
y_d = np.sin(np.pi * x)

# Initial guess for control (zero control)
u = np.zeros(N)

# Number of iterations for convergence
num_iterations = 500

# Initialize arrays for state (y) and adjoint (p)
y = np.zeros(N)
p = np.zeros(N)

# Finite difference matrix for Laplacian (1D with Dirichlet boundary conditions)
def laplacian_matrix(N, h):
    A = np.zeros((N, N))
    for i in range(1, N-1):
        A[i, i-1] = 1
        A[i, i] = -2
        A[i, i+1] = 1
    A[0, 0] = -2
    A[0, 1] = 1
    A[N-1, N-2] = 1
    A[N-1, N-1] = -2
    return A / (h**2)

A = laplacian_matrix(N, h)

# Iterate to solve the state, adjoint, and update control
for iteration in range(num_iterations):
    # Solve the state equation: A @ y = u (discretized state equation)
    y = np.linalg.solve(A, -u)

    # Solve the adjoint equation: A @ p = y - y_d (discretized adjoint equation)
    p = np.linalg.solve(A, y - y_d)

    # Update control: u = -(1/nu) * p
    u = -(1 / nu) * p

# Plot the results
plt.figure(figsize=(10, 6))
plt.plot(x, y_d, label="Desired state (y_d)", linestyle="--", color="blue")
plt.plot(x, y, label="State after optimization (y)", color="red")
plt.plot(x, u, label="Control (u)", color="green")
plt.xlabel("Position along the rod")
plt.ylabel("Value")
plt.title("Optimal Control of Temperature in a 1D Rod")
plt.legend()
plt.grid(True)
plt.show()
