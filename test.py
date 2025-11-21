import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Parameters
R1 = 0.5      # Longitudinal relaxation rate
R2 = 0.1      # Transverse relaxation rate
M0 = 1.0      # Equilibrium magnetization along z
dt = 0.01     # Time step
t = np.arange(0, 50, dt)

# Homogeneous (constant) magnetic field
Bx, By, Bz = 0.0, 0.0, 1.0

# Initialize magnetization
Mx = np.zeros_like(t)
My = np.zeros_like(t)
Mz = np.zeros_like(t)
Mz[0] = 0.0  # Start misaligned (e.g., tipped magnetization)

# Time integration (Euler method)
for i in range(1, len(t)):
    dMx = (My[i-1]*Bz - Mz[i-1]*By) - R2*Mx[i-1]
    dMy = (Mz[i-1]*Bx - Mx[i-1]*Bz) - R2*My[i-1]
    dMz = (Mx[i-1]*By - My[i-1]*Bx) - R1*(Mz[i-1] - M0)
    Mx[i] = Mx[i-1] + dt*dMx
    My[i] = My[i-1] + dt*dMy
    Mz[i] = Mz[i-1] + dt*dMz

# 3D Plot
fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot(111, projection='3d')
ax.plot(Mx, My, Mz, lw=2)
ax.set_xlabel("Mx")
ax.set_ylabel("My")
ax.set_zlabel("Mz")
ax.set_title("Magnetization Precession in a Homogeneous Field")
ax.grid(True)

# Add reference vector for B field
ax.quiver(0, 0, 0, Bx, By, Bz, color='red', length=1.2, normalize=True)
ax.text(0.1, 0.1, 1.1, "B", color='red')

plt.show()
