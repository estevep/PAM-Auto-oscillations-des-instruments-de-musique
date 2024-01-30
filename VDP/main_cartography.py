# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from sklearn import svm
from sklearn.inspection import DecisionBoundaryDisplay

from VDP import *
from impedance import *
from save import *


#%%

def is_sound(P):
    return np.any(P>1e-7)


def plot_training_data_with_decision_boundary(X, y):
    # Train the SVC
    clf = svm.SVC(gamma=10).fit(X, y)

    # Settings for plotting
    _, ax = plt.subplots(figsize=(7, 7))
    x_min, x_max, y_min, y_max = 0, 1, 0, 1
    ax.set(xlim=(x_min, x_max), ylim=(y_min, y_max))

    # Plot decision boundary and margins
    common_params = {"estimator": clf, "X": X, "ax": ax}
    DecisionBoundaryDisplay.from_estimator(
        **common_params,
        grid_resolution=1000,
        response_method="predict",
        plot_method="pcolormesh",
        alpha=0.3,
    )
    DecisionBoundaryDisplay.from_estimator(
        **common_params,
        response_method="decision_function",
        plot_method="contour",
        levels=[-1, 0, 1],
        colors=["k", "k", "k"],
        linestyles=["--", "-", "--"],
    )

    # Plot bigger circles around samples that serve as support vectors
    ax.scatter(
        clf.support_vectors_[:, 0],
        clf.support_vectors_[:, 1],
        s=50,
        facecolors="none",
        edgecolors="k",
    )
    # Plot samples by color and add legend
    scatter = ax.scatter(X[:, 0], X[:, 1], c=y, s=50, label=y, edgecolors="k")
    ax.legend(*scatter.legend_elements(), loc="upper right", title="Classes")
    ax.set_title(f" Decision boundaries in SVC")

    _ = plt.show()

# %% ------------------- USER INPUT -------------------

plot_bool   = False
save_bool   = False
save_dir    = "C:/Users/vppit/Desktop/Sorbonne/PAM/WAV_VDP"
Fs          = 44100                         # Sampling frequency [Hz]
T           = 0.3                             # Total time of the signal [s]
N           = 1                             # Number of cavity modes 
P0          = 1e-8*np.random.randn(1)       # Initial (normalized) pressure value [Pa]
Pdot0       = 0.                            # Initial time derivative of the (normalized) pressure value [Pa/s]
gamma_func  = give_gamma_func("constant")   # Gamma function with respect to time
zeta_func   = give_zeta_func("constant")    # Zeta function with respect to time

# Gamma/Zeta parameters grid

dim_grid = (10,10)

g_grid          = np.linspace(1e-8, 1, dim_grid[0])
z_grid          = np.linspace(1e-8, 1, dim_grid[0])
g_grid, z_grid  = np.meshgrid(g_grid, z_grid)
g_grid, z_grid  = g_grid.reshape((-1,1)), z_grid.reshape((-1,1))  
crit_grid       = np.full(g_grid.size, False)
# %% ------------------- DO NOT CHANGE -------------------

# Impedance   

w               = give_w_axis()
Z               = Z_cylinder()
wn, Qn, Fn, Ymn = find_Z_model(N, Z)
Z_model         = create_Z_model(wn, Qn, Fn)

# Solve the VDP equation

Pn0 = [P0, Pdot0]
t = np.arange(0,T,1/Fs)

for i in range(g_grid.size):
    print('------ Point (',i+1,' over ',g_grid.size,') ------')
    g0, z0 = g_grid[i], z_grid[i]
    in_params = {
        "T"     : T,
        "g0"    : g0,
        "z0"    : z0
        }
    P, Pdot = np.zeros(t.size), np.zeros(t.size)
    for k in range(N):
        print('Mode ',k+1,'over ',N)
        args = (gamma_func, zeta_func, in_params, Fn[k], Ymn[k], wn[k])
        
        temp    = odeint(VDP, Pn0, t, args)
        P       += temp[:,0]
        Pdot    += temp[:,1]
    crit_grid[i] = is_sound(P)


#%%
X = np.concatenate((g_grid, z_grid),axis=1)

plot_training_data_with_decision_boundary(X, crit_grid)