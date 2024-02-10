import numpy as np
# import openwind
from scipy.signal import find_peaks
# from impedance import give_w_axis, find_Z_model

import matplotlib.pyplot as plt

target_pulsations = 2 * np.pi * np.array([261.63, 277.18, 293.66, 311.13,
                                          329.63, 349.23, 369.99, 392.00,
                                          415.30, 440.00, 466.16, 493.88,
                                          523.25, 554.37, 587.33, 622.25,
                                          659.25, 698.46, 739.99, 783.99,
                                          830.61, 880.00, 932.33, 987.77])


def give_w_axis_mod():
    f = np.arange(1, 1e4, 0.2)
    return 2*np.pi*f


def compute_Fn_Ymn(l, N):
    w   = give_w_axis_mod()                         # Angular velocity axis w [rad/s]
    R   = 13e-3                                     # Radius of the resonator [m]
    # N   = 5                                         # Number of holes    
    S   = np.pi*R**2                                # Section of the resonator [m2]
    
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
    
    Z = np.tanh(1j*k*(l+dl) + a*l + 1/4*(k*R)**2)   # Input impedance [kg/m2s]
    
    plt.figure()
    plt.plot(w/(2 * np.pi), np.abs((Z - 1)/(Z + 1)))
    plt.show(block=True)


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


    return wn, Fn, Ymn


if __name__ == "__main__":

    c = 343.0
    number_of_modes = 5
    pulsations = target_pulsations.copy()
    
    l_vec = np.linspace(0.08, 0.4, 2000)
    l_vec = np.flip(l_vec)

    file = None

    write = False

    if write:
        file = open("real time/params.txt", "w")

    compute_Fn_Ymn(0.6, number_of_modes)

    if write:
        w_index = 0
        for i in range(l_vec.shape[0]):
            
            wn, Fn, Ymn = compute_Fn_Ymn(l_vec[i], number_of_modes)

            if np.isclose(wn[0], pulsations[w_index], rtol=1e-3):

                w_index += 1

                for mode in range(number_of_modes):
                    file.write(f"{wn[mode]/(2*np.pi)} ")
                
                for mode in range(number_of_modes):
                    file.write(f"{Fn[mode]} ")
                
                for mode in range(number_of_modes):
                    file.write(f"{Ymn[mode]} ")

                file.write("\n")


        assert(w_index == pulsations.shape[0])

    file.close()