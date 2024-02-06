import numpy as np
import openwind
from scipy.signal import find_peaks
from impedance import give_w_axis, find_Z_model

target_pulsations = 2 * np.pi * np.array([130.81,
                     138.59,
                     146.83,
                     155.56,
                     164.81,
                     174.61,
                     185.00,
                     196.00,
                     207.65,
                     220.00,
                     233.08,
                     246.94,
                     261.63])


def Z_cylinder_mod(l):
    w   = give_w_axis()                             # Angular velocity axis w [rad/s]
    # l   = 0.6                                       # Length of the resonator [m]
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

l_vec = np.linspace(0.1, 2.0, 150)
l_result = []
wn_vec = []
Ymn_vec = []
Fn_vec = []

for i in range(l_vec.shape[0]):
    
    Z = Z_cylinder_mod(l_vec[i])
    wn, _, Fn, Ymn = find_Z_model(1, Z)

    if np.isclose(wn, target_pulsations, rtol= 0.01).sum():
        l_result.append(l_vec[i])
        wn_vec.append(wn[0])
        Ymn_vec.append(Ymn[0])
        Fn_vec.append(Fn[0])

# print(np.array(wn_vec) / (2* np.pi))


file = open("real time/params.txt", "w")
for i, value in enumerate(l_result):
    file.writelines([f"{wn_vec[i]} {Ymn_vec[i]} {np.abs(Fn_vec[i])}\n"])

file.close()