import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation

M = np.array([1.,0.,0.]) # Net Magnitization Vector
M0 = np.sqrt(np.dot(M,M))
B = np.array([0.,0.,100.]) # Magnetic Field Vector (points in the positive z direction)

T2 = 5 # Transverse Relaxation Time Constant
T1 = 10 # Longitudinal Relaxation Time Constant
R1 = 10
R2 = 10
gamma = 10
dt = 0.00001


def calculate_bloch_differentials(M, M0, B, R1, R2, gamma) -> np.array:
    dmxdt = gamma*np.cross(M,B)[0] - R2*M[0]
    dmydt = gamma*np.cross(M,B)[1] - R2*M[1]
    dmzdt = gamma*np.cross(M,B)[2] - R1*(M[2]-M0)
    
    dM = np.array([dmxdt, dmydt, dmzdt])

    # print(f"{dM} : magnitude: {np.dot(dM,dM)}")
    return dM

def compute_next_state(M, M0, B, R1, R2, gamma) -> np.array:
    M += calculate_bloch_differentials(M,M0, B,R1,R2,gamma) * dt
    return M


# Attaching 3D axis to the figure
fig = plt.figure()
ax = fig.add_subplot(projection="3d")
ax.set_aspect('equal')
# Setting the Axes properties
ax.set(xlim3d=(0, 1), xlabel='X')
ax.set(ylim3d=(0, 1), ylabel='Y')
ax.set(zlim3d=(0, 1), zlabel='Z')

def generate_arrow(M):
    x = 0
    y = 0
    z = 0
    u = M[0]
    v = M[1]
    w = M[2]
    return x,y,z,u,v,w

quiver = ax.quiver(*generate_arrow(M))

ax.set_xlim(-1, 1)
ax.set_ylim(-1, 1)
ax.set_zlim(-0.2, 1)

def simulation_frame(empty):
    global quiver
    global M
    global B
    global ax

    M = compute_next_state(M,M0, B,R1,R2, gamma)
    
    ax.scatter(M[0], M[1], M[2],color="r")
    quiver.remove()

    quiver = ax.quiver(*generate_arrow(M))

    
# Creating the Animation object
ani = animation.FuncAnimation(fig, simulation_frame, frames=np.linspace(0,2*np.pi,200), interval=50)

plt.show()