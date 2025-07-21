import sys
import numpy as np
import h5py
import matplotlib.animation as animation
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats
import os

# Add the parent directory to the python path
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(script_dir)

# Get the file path for structure_formation_data
structure_formation_data = os.path.join(script_dir, "structure_formation_data/", "pycosmo_data/")

n_i = 0
n_max = 9 # Number of frames
num_processors = 1 # number of processors
dt = (64)**(1/n_max)
folder = structure_formation_data
filetype = "csv"
saving_name = "dens_animation"

stack = []
time = np.zeros(n_max)

if(filetype == "hdf5"):
    time[0] = 1./64
    for i in range(0,n_max):
        f = h5py.File(folder + f"snapshot_{i:03d}.hdf5", 'r')
        data_all = f['PartType1']
        data = np.asarray(data_all['Coordinates'])
        if (i>0):
            time[i] = np.sqrt(2) * time[i-1]
        f.close()
        
        #Data = np.hstack((Data, data))
        stack.append(data)
    Data = np.stack(stack, axis = 0)

elif(filetype == "csv"):
    for i in range(n_max): # go trough all timesteps
        
        if (num_processors == 1):
            filename = folder + f"snapshot_0{i}0.csv"
            data = np.loadtxt(filename, delimiter=',', dtype=float, usecols = (0,1,2))
            stack.append(data)
        elif(num_processors > 1):
            filename = folder + f"snapshot_0_{i:03d}.csv"
            data = np.loadtxt(filename, delimiter=',', dtype=float, usecols = (0,1,2))
            for j in range(1,num_processors):
                filename = folder + f"snapshot_{j}_{i:03d}.csv"
                data_j = np.loadtxt(filename, delimiter=',', dtype=float, usecols = (0,1,2))
                data = np.concatenate((data, data_j), axis = 0)
            stack.append(data)
        else:
            print("non valid number of processors")
            break

    Data = np.stack(stack, axis = 0)
else:
    print("no known datatype")

 
num_particles = Data.shape[1]

class Plotting:

    def __init__(self, fig, ax, data):
        self.fig = fig
        self.ax = ax
        self.data = data/1000

    def animate2d(self, n):
        self.ax.clear()
        self.ax.hist2d(self.data[n, :, 0], self.data[n, :, 1], bins = 300, cmap = "magma", edgecolor='none')
        # norm=mpl.colors.LogNorm()
        #self.ax.set_title(f"a = {round(time[n], 4)}, z = {round(1/time[n],4)}")
        self.ax.set_xlabel("Mpc", fontsize=16)
        self.ax.set_ylabel("Mpc", fontsize=16)
        self.ax.tick_params(axis='both', which='major', labelsize=14)
        self.ax.grid(False)


    def animate3d(self, n):
        self.ax.clear()
        ax.set_facecolor('black') 
        ax.grid(False) 
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        size = 32 / (num_particles**(1./3))  

        self.ax.scatter(self.data[n, :, 0], self.data[n, :, 1], self.data[n, :, 2], s=size, alpha = 0.1, color = "white")
        self.ax.set_xlim3d(0, 50000)
        self.ax.set_ylim3d(0, 50000)
        self.ax.set_zlim3d(0, 50000)
        print("Frame ", n, " was done.")


mpl.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",  # Or "sans-serif", or specific LaTeX font
    "font.serif": ["Computer Modern Roman"],  # Default LaTeX font
})

fig, ax = plt.subplots(figsize=(7,7))
picture = Plotting(fig, ax, Data)

ani = animation.FuncAnimation(fig, picture.animate2d, frames = n_max - n_i, interval = 500)
writer = animation.PillowWriter(fps = 4)
ani.save(folder + saving_name + "2d.gif", writer = writer)
plt.savefig(folder + saving_name + "2d.png")
print("saved " + folder + saving_name + "2d.gif")


# 3d
'''
fig = plt.figure(figsize=(10,10))
ax = plt.axes(projection='3d')
fig.set_facecolor('black')
picture = Plotting(fig, ax, Data)

ani = animation.FuncAnimation(fig, picture.animate3d, frames = n_max, interval = 500)
writer = animation.PillowWriter(fps = 4)
ani.save(folder + saving_name + "3d.gif", writer = writer)
plt.savefig(folder + saving_name + "3d.png")
print("saved " + folder + saving_name + "3d.gif")
'''

