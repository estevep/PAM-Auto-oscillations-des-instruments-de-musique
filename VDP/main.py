# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

from VDP import *
from impedance import *
from save import *


# %% ------------------- USER INPUT -------------------

plot_bool   = True
save_bool   = False
save_dir    = "C:/Users/vppit/Desktop/Sorbonne/PAM/WAV_VDP"
Fs          = 44100                         # Sampling frequency [Hz]
T           = 2                             # Total time of the signal [s]
N           = 3                             # Number of cavity modes 
P0          = 1e-8*np.random.randn(1)       # Initial (normalized) pressure value [Pa]
Pdot0       = 0.                            # Initial time derivative of the (normalized) pressure value [Pa/s]
gamma_func  = give_gamma_func("constant")   # Gamma function with respect to time
zeta_func   = give_zeta_func("constant")    # Zeta function with respect to time

# Gamma/Zeta parameters
g0          = 0.5
z0          = 0.5

in_params = {
    "T"     : T,
    "g0"    : g0,
    "z0"    : z0
    }

# %% ------------------- DO NOT CHANGE -------------------

# Impedance   

w               = give_w_axis()
Z               = Z_cylinder()
wn, Qn, Fn, Ymn = find_Z_model(N, Z)
Z_model         = create_Z_model(wn, Qn, Fn)

# Solve the VDP equation

Pn0 = [P0, Pdot0]
t = np.arange(0,T,1/Fs)

P, Pdot = np.zeros(t.size), np.zeros(t.size)

for i in range(N):
    print('Mode ',i+1,'over ',N)
    args = (gamma_func, zeta_func, in_params, Fn[i], Ymn[i], wn[i])
    
    temp    = odeint(VDP, Pn0, t, args)
    P       += temp[:,0]
    Pdot    += temp[:,1]

# Plots

if plot_bool :
    plt.figure(figsize = (15,10))
    plt.subplot(2, 1, 1)
    plt.plot(t, P)
    plt.title('Pressure at resonator input')
    plt.xlabel('t (s)')
    plt.ylabel(r'$P$ (Pa)')
    plt.subplot(2, 1, 2)
    plt.plot(t, Pdot)
    plt.title('Pressure time derivative at resonator input')
    plt.xlabel('t (s)')
    plt.ylabel(r'$\partial_t P$ (Pa/s)')

# Saving P

if save_bool:
    save_P(save_dir, Fs, P)