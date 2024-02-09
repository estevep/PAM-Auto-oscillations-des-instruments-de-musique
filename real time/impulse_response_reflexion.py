import matplotlib.pyplot as plt
import numpy as np
from scipy.io.wavfile import write

    
target_frequencies = np.array([261.63, 277.18, 293.66, 311.13,
                               329.63, 349.23, 369.99, 392.00,
                               415.30, 440.00, 466.16, 493.88,
                               523.25, 554.37, 587.33, 622.25,
                               659.25, 698.46, 739.99, 783.99,
                               830.61, 880.00, 932.33, 987.77])


# Description : gives the impedance for a cylindrical cavity (with losses)
def main():
    
    samplerate = 44100.0
    nfft = 1 << 11

    impulse_response = np.zeros((target_frequencies.shape[0], nfft))


    fft_freqs = np.fft.rfftfreq(nfft, 1/samplerate) + 0.001

    c   = 340                                       # Sound speed in the medium [m/s]
    l_vec = np.flip(c/(4 * target_frequencies))

    # l   = 0.6                                       # Length of the resonator [m]
    R   = 13e-3                                     # Radius of the resonator [m]
    #S   = np.pi*R**2                                # Section of the resonator [m2]
    
    w = 2 * np.pi * fft_freqs
    
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
    #Zc  = rho*c/S                                   # Caracteristic impedance of the medium
    
    for i in range(l_vec.shape[0]):
        l = l_vec[i]
        Z = np.tanh(1j*k*(l+dl) + a*l + 1/4*(k*R)**2)   # Input impedance [kg/m2s]

        time_response = np.fft.irfft(Z)

        impulse_response[i, :] = time_response/np.max(time_response)

    # plt.figure()
    # plt.plot(fft_freqs, impulse_response.T)
    # plt.show()
    

    amplitude = np.iinfo(np.int32).max


    write("real time/reflexion_impulse_response.wav", int(samplerate), (amplitude * impulse_response).T.astype(np.int32))




if __name__ == "__main__":
    main()