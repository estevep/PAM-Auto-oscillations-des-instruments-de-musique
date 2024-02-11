import numpy as np
import scipy.io.wavfile as wavfile

audio, fs = wavfile.read(r"C:\Users\audas\OneDrive\Bureau\ATIAM\PAM\Code Repo\PAM-Auto-oscillations-des-instruments-de-musique\Descripteurs\samples\real\Changement de registre 2.wav")



def spectral_centroid(fft, fe):

    freq = np.arange(0, len(fft))/len(fft)*fe
    amp = 20*np.log(np.abs(fft))
    amp -= np.min(amp)

    return np.sum(freq*amp)/np.sum(amp)

print(fs)
