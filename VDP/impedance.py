# -*- coding: utf-8 -*-

import numpy as np
from scipy.signal import find_peaks
from scipy.special import jv


# Description : gives the angular velocity axis w [rad/s]
def give_w_axis(n, Fs):
    f = np.fft.rfftfreq(n, 1/Fs)
    f[0] = 1e-8 # to avoid zero-divisions 
    return 2*np.pi*f


# Description : gives the clarinet's geometric and thermodynamic parameters
def give_clarinet_params(w):
    l = 0.6                       # Length of the resonator [m]
    R = 13e-3                     # Radius of the resonator [m]
    S = np.pi*R**2                # Section of the resonator [m2]
    c = 340                       # Sound speed in the medium [m/s]
    rho = 1.2                     # Density of the medium [kg/m3]
    mu = 1.8e-5                   # Dynamic viscosity of the medium [kg/m.s]
    kappa = 0.025                 # Thermal conductivity [kg.m/s3.K]
    Cp = 1004                     # Heat capacity at constant pressure [m2/s2.K]
    Xs = 1/(rho*c**2)             # Adiabatic compressibility [m/kg.s2]
    gamma = 1.670                 # Adiabatic index [-]
    Iv = mu/(rho*c)               
    It = kappa/(rho*c*Cp)                   
    kv = np.sqrt(-1j*w/(c*Iv))    # Viscous diffusion wavenumber [m-1]
    kt = np.sqrt(-1j*w/(c*It))    # Thermal diffusion wavenumber [m-1]
    rv = np.abs(kv*R)             # Stokes number [-]
    a1 = 1.044                                        
    a2 = 1.08                                        
    a = w/c*(a1/rv + a2/rv**2)    # Real part of propagation constant tau [m-1]  
    k = w/c*(1+a1/rv)             # Wavelength [m-1]
    dl = 0.6*R                    # Length correction [m] 
    
    params = {
        "l"     : l,
        "R"     : R,
        "S"     : S,
        "c"     : c,
        "rho"   : rho,
        "mu"    : mu,
        "kappa" : kappa,
        "Cp"    : Cp,
        "Xs"    : Xs,
        "gamma" : gamma,
        "Iv"    : Iv,      
        "It"    : It,                   
        "kv"    : kv,
        "kt"    : kt, 
        "rv"    : rv, 
        "a1"    : a1,                                      
        "a2"    : a2,                                    
        "a"     : a,
        "k"     : k,
        "dl"    : dl 
        }
    
    return params


# Description : gives the impedance for a cylindrical cavity (with losses)
def Z_cylinder(w):
    params  = give_clarinet_params(w)    # Clarinet's geometric and thermodynamic parameters
    k       = params["k"]
    l       = params["l"]
    dl      = params["dl"]
    a       = params["a"]
    R       = params["R"]
    
    Z = np.tanh(1j*k*(l+dl) + a*l + 1/4*(k*R)**2)   # Input impedance [kg/m2s]
    
    return Z


# Description : finds the Z modal parameters (wn, Qn, Fn) for the given Z function
def find_Z_model(N, Z, w):
    dw          = w[1] - w[0]
    Zreal       = np.real(Z)
    peaks, _    = find_peaks(Zreal)
    peaks       = peaks[:N]
    wn          = w[peaks]
    
    Qn = np.zeros(N)
    for i,peak in enumerate(peaks):
        idx         = peak
        idx_max     = int(peak + 50*2*np.pi//dw)
        idx_half    = np.argmin(np.abs(Zreal[idx:idx_max] - Zreal[idx]/2))
        d_idx       = 2*idx_half
        Qn[i]       = w[idx]/(dw*d_idx)
        
    Fn  = Zreal[peaks]*wn/Qn
    Ymn = 1/Zreal[peaks]
    
    return wn, Qn, Fn, Ymn


# Description : creates the Z function from the modal parameters (wn, Qn, Fn)
def create_Z_model(wn, Qn, Fn, w):
    N       = wn.size 
    Z_model = np.zeros(w.size, dtype=complex)
    
    for i in range(N):
        Z_model += 1j*w*Fn[i]/(wn[i]**2 - w**2 + 1j*w*wn[i]/(Qn[i]))
    
    return Z_model


def give_P_u_output(P, u, w):
    params  = give_clarinet_params(w)    # Clarinet's geometric and thermodynamic parameters
    rho     = params["rho"]
    R       = params["R"]
    S       = params["S"]
    l       = params["l"]
    Xs      = params["Xs"]
    gamma   = params["gamma"]
    kv      = params["kv"]
    kt      = params["kt"]
    
    import matplotlib.pyplot as plt
    plt.figure()
    plt.plot(w/(2*np.pi), np.real(kt*R))
    plt.plot(w/(2*np.pi), np.imag(kt*R))
    idx         = np.real(kv*R)<600 # BUGGED !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    nidx        = np.logical_not(idx)
    Zv          = np.zeros(w.size, dtype=complex)
    Zv[idx]     = 1j*w[idx]*rho/S*(1 - 2/(kv[idx]*R)*jv(1,kv[idx]*R)/jv(0,kv[idx]*R))**-1
    Zv[nidx]    = 1j*w[nidx]*rho/S*(1 - 2/(kv[nidx]*R)*1j)**-1
    idx         =  np.real(kt*R)<600
    nidx        = np.logical_not(idx)
    Yt          = np.zeros(w.size, dtype=complex)
    Yt[idx]     = 1j*w[idx]*Xs*S*(1 + (gamma - 1)*2/(kt[idx]*R)*jv(1,kt[idx]*R)/jv(0,kt[idx]*R))
    Yt[nidx]    = 1j*w[nidx]*Xs*S*(1 + (gamma - 1)*2/(kt[nidx]*R)*1j)
    tau = np.sqrt(Zv*Yt)
    Zc = np.sqrt(Zv/Yt)
    
    M   = np.array([[np.cosh(tau*l)         , Zc*np.sinh(tau*l)],
                    [Zc**-1*np.sinh(tau*l)  , np.cosh(tau*l)]])
    M   = np.moveaxis(M,-1,0)
    M   = np.linalg.inv(M)
    P_w = np.fft.rfft(P)
    U_w = np.fft.rfft(u)
    
    PU_w = np.concatenate((P_w[np.newaxis,:],U_w[np.newaxis,:]), axis=0)
    
    for i in range(w.size):
        PU_w[:,i] = M[i]@PU_w[:,i]
    
    P_out = np.fft.irfft(PU_w[0])
    u_out = np.fft.irfft(PU_w[1])
    
    print(P_out.dtype, u_out.dtype)
    
    return P_out, u_out
    

