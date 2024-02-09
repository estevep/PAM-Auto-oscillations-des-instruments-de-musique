# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import os

from VDP import *
from impedance import *
from save import *


# %% ------------------- USER INPUT -------------------

plot_bool       = True
save_bool       = True
save_dir        = f"{os.getcwd()}\\..\\Descripteurs\\samples"
instrument_type = 'clarinet'
solver_type     = "coupled"
Fs              = 44100                         # Sampling frequency [Hz]
T               = 5                             # Total time of the signal [s]
N               = 3                             # Number of cavity modes 
P0              = 1e-8*np.random.randn()        # Initial (normalized) pressure value [Pa]
Pdot0           = 0.                            # Initial time derivative of the (normalized) pressure value [Pa/s]
gamma_func      = give_gamma_func("constant")   # Gamma function with respect to time
zeta_func       = give_zeta_func("constant")    # Zeta function with respect to time

# Gamma/Zeta parameters
g0          = 0.99
z0          = 0.5

in_params = {
    "T"     : T,
    "g0"    : g0,
    "z0"    : z0
    }

# %% ------------------- DO NOT CHANGE -------------------

# ODE solver 

def RK4_step(ode_fun, t, y, h, args=()):
    K1 = h*np.array(ode_fun(y, t, *args))
    K2 = h*np.array(ode_fun(y+1/2*K1, t+1/2*h, *args))
    K3 = h*np.array(ode_fun(y+1/2*K2, t+1/2*h, *args))
    K4 = h*np.array(ode_fun(y+K3, t+h, *args))
    
    return y + 1/6*(K1+2*K2+2*K3+K4)


def RK_solver(ode_fun, y0, t, args=()): 
    y       = np.zeros((t.size, len(y0)), dtype = complex)
    y[0]    = np.array(y0, dtype=complex)
    
    for i in range(1, t.size):
        h       = t[i] - t[i-1] 
        y[i]    = RK4_step(ode_fun, t[i-1], y[i-1], h, args)
    
    return np.real(y)


def uncoupled_solver(N, t, gamma_func, zeta_func, in_params, Fn, Ymn, wn, Pn0):
    P, Pdot = np.zeros(t.size), np.zeros(t.size)

    for i in range(N):
        print('Mode ',i+1,'over ',N)
        args = (gamma_func, zeta_func, in_params, Fn[i], Ymn[i], wn[i])
        
        temp    = odeint(VDP, Pn0, t, args)
        #temp    = RK_solver(VDP, Pn0, t, args)
        P       += temp[:,0]/N
        Pdot    += temp[:,1]/N    
        
    return P, Pdot


def coupled_solver(N, t, gamma_func, zeta_func, in_params, Fn, Ymn, wn, Pn0):
    X = np.zeros((t.size, 2*N))
    X[0] = np.repeat(np.array(Pn0), N)
    
    for i in range(1, t.size):
        if i%5000==0:
            print("Step " + str(i) + " over "+ str(t.size) + "(" + str(np.round(i/t.size*100)) + " %)")
        h           = t[i] - t[i-1] 
        
        args    = (gamma_func, zeta_func, in_params, Fn, Ymn, wn)
        X[i]    =  RK4_step(VDP_coupled, t[i-1], X[i-1], h, args)
    
    P       = np.sum(X[:,:N]/N, axis=1)
    Pdot    = np.sum(X[:,N:]/N, axis=1)
        
    return P, Pdot


# Impedance   

w               = give_w_axis(1, 1e3)
#Z               = Z_cylinder()
Z               = import_Z(w, model_type = instrument_type, 
                           model_plot=True)
wn, Qn, Fn, Ymn = find_Z_model(N, Z)
Z_model         = create_Z_model(wn, Qn, Fn)

# Solve the VDP equation

if solver_type == 'uncoupled':
        solver =  uncoupled_solver
elif solver_type == 'coupled':
    solver =  coupled_solver
else:
    solver =  coupled_solver
        
    
Pn0 = [P0, Pdot0]
t = np.arange(0,T,1/Fs)

P, Pdot = solver(N, t, gamma_func, zeta_func, in_params, Fn, Ymn, wn, Pn0)


# Compute the FFT

P_fft = np.fft.rfft(P)
f_plot = np.fft.rfftfreq(P.size, 1/Fs)  

# Plots

if plot_bool :
    plt.figure(figsize = (15,10))
    plt.subplot(2, 2, 1)
    plt.plot(t, P)
    plt.title('Pressure at resonator input')
    plt.xlabel('t (s)')
    plt.ylabel(r'$P$ (Pa)')
    plt.subplot(2, 2, 2)
    plt.plot(t, Pdot)
    plt.title('Pressure time derivative at resonator input')
    plt.xlabel('t (s)')
    plt.ylabel(r'$\partial_t P$ (Pa/s)')
    plt.subplot(2, 2, 3)
    plt.plot(w/(2*np.pi), np.real(Z), label='Real impedance')
    plt.plot(w/(2*np.pi), np.real(Z_model), label='Model impedance')
    plt.title('Modal and real normalized impedances w.r.t frequency')
    plt.xlabel('f (Hz)')
    plt.ylabel(r'$\frac{Re(Z)}{Z_c}$ (-)')
    plt.legend()
    plt.subplot(2, 2, 4)
    plt.plot(f_plot[f_plot<1000], np.abs(P_fft[f_plot<1000]))
    plt.title('Pressure FFT w.r.t frequency')
    plt.xlabel('f (Hz)')
    plt.ylabel(r'FFT($P$) (Pa)')
    plt.show()

# Saving P

if save_bool:
    save_P(save_dir, Fs, P)
    
