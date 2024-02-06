# -*- coding: utf-8 -*-

import numpy as np
from scipy.signal import find_peaks
import openwind

# Description : gives the angular velocity axis w [rad/s]
def give_w_axis():
    f = np.linspace(1, 1e3, 10000)
    return 2*np.pi*f


# Description : gives the impedance for a cylindrical cavity (with losses)
def Z_cylinder():
    w   = give_w_axis()                             # Angular velocity axis w [rad/s]
    l   = 0.6                                       # Length of the resonator [m]
    R   = 13e-3                                     # Radius of the resonator [m]
    S   = np.pi*R**2                                # Section of the resonator [m2]
    
    f   = w/2*np.pi
    
    c   = 340                                       # Sound speed in the medium [m/s]
    rho = 1.2                                       # Density of the medium [kg/m3]
    mu  = 1.8e-5                                    # Dynamic viscosity of the medium [kg/m.s]
    Iv  = mu/(rho*c)                                  
    kv  = np.sqrt(-1j*w/(c*Iv))                     # Viscous diffusion wavenumber [m-1]
    rv  = np.abs(kv*R)                              # Stokes number [-]
    a1  = 1.044                                        
    a2  = 1.08                                        
    a   = w/c*(a1/rv + a2/rv**2)                    # Real part of propagation constant tau [m-1]  
    k   = w/c*(1+a1/rv)                             # Wavelength [m-1]
    dl  = 0.6*R                                     # Length correction [m] 
    Zc  = rho*c/S                                   # Caracteristic impedance of the medium
    
    Z = Zc*np.tanh(1j*k*(l+dl) + a*l + 1/4*(k*R)**2)   # Input impedance [kg/m2s]
    
    return Z


# Description : finds the Z modal parameters (wn, Qn, Fn) for the given Z function
def find_Z_model(N, Z):
    w           = give_w_axis()
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
        
    Fn  = Z[peaks]*wn/Qn
    Ymn = 1/Zreal[peaks]
    
    return wn, Qn, Fn, Ymn


# Description : creates the Z function from the modal parameters (wn, Qn, Fn)
def create_Z_model(wn, Qn, Fn):
    w       = give_w_axis()
    N       = wn.size 
    Z_model = np.zeros(w.size, dtype=complex)
    
    for i in range(N):
        Z_model += 1j*w*Fn[i]/(wn[i]**2 - w**2 + 1j*w*wn[i]/(Qn[i]))
    
    return Z_model
    

# Description : imports the Z function from the OpenWind library
def import_Z(w, model_type='clarinet', model_plot = False):
    f = w/(2*np.pi)
    
    if model_type=='clarinet':
        file = "Impedance_models/clarinet.csv"
    elif model_type=='trumpet':
        file = "Impedance_models/simplified_trumpet.csv"
    else:
        raise Exception("Model type '"+model_type+"' does not exist.")
    
    instru_geom = openwind.InstrumentGeometry(file)
    
    if model_plot:
        instru_geom.plot_InstrumentGeometry()
    
    Z = openwind.ImpedanceComputation(f, file, temperature=25)
    
    Z.plot_impedance()
    
    return Z.impedance