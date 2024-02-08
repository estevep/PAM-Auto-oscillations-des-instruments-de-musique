import numpy as np
import openwind
from scipy.signal import find_peaks
from impedance import give_w_axis, find_Z_model

target_pulsations = 2 * np.pi * np.array([261.63, 277.18, 293.66, 311.13,
                                          329.63, 349.23, 369.99, 392.00,
                                          415.30, 440.00, 466.16, 493.88,
                                          523.25, 554.37, 587.33, 622.25,
                                          659.25, 698.46, 739.99, 783.99,
                                          830.61, 880.00, 932.33, 987.77])


def compute_Fn_Ymn(l):
    w   = give_w_axis()                             # Angular velocity axis w [rad/s]
    R   = 13e-3                                     # Radius of the resonator [m]
    N   = 1                                         # Number of holes    
    
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
    
    Z = np.tanh(1j*k*(l+dl) + a*l + 1/4*(k*R)**2)   # Input impedance [kg/m2s]
    

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


    return wn[0], Fn[0], Ymn[0]


if __name__ == "__main__":

    c = 343.0
    pulsations = target_pulsations.copy()
    
    # l_vec = c*2*np.pi/(4 * pulsations)
    l_vec = np.linspace(0.08, 0.35, 2000)
    l_vec = np.flip(l_vec)

    file = open("real time/params.txt", "w")
    
    w_index = 0
    for i in range(l_vec.shape[0]):
        
        wn, Fn, Ymn = compute_Fn_Ymn(l_vec[i])

        if np.isclose(wn, pulsations[w_index], rtol=1e-3):


            w_index += 1
            file.writelines([f"{wn/(2*np.pi)} {Fn} {Ymn}\n"])

    assert(w_index == pulsations.shape[0])

    file.close()