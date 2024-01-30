import numpy as np
from numpy.typing import ArrayLike

import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

from scipy.io.wavfile import write, read
from scipy.interpolate import CubicSpline
from scipy.signal import welch

import os 

samples_dir = f"{os.getcwd()}/Descripteurs/samples"
filenames = os.listdir(samples_dir)


def compute_offset_signal(signal: ArrayLike, deltaT: float, samplerate: float) -> ArrayLike:
    
    T = np.linspace(0, signal.size/samplerate, signal.size)
    interpolator = CubicSpline(T, signal)
    interpolated_signal = interpolator(T + deltaT)

    return interpolated_signal


def import_signal(filepath: str, time_begin=0, time_end=None):
    samplerate, signal = read(filepath)

    if signal.ndim > 1:
        signal = signal[:, 0]
    
    signal = signal[int(time_begin):int(time_end)]
    signal = signal.astype("float") / np.max(signal)

    return samplerate, signal


def main() -> None:

    print(filenames)

    samplerate, signal1 = import_signal(f"{samples_dir}/{filenames[1]}", 44100 * 3, 44100 * 3 + 1024)
    _,          signal2 = import_signal(f"{samples_dir}/{filenames[2]}", 0, 1024)
    

    deltaT = 4e-4
    signal1_interpolated = compute_offset_signal(signal1, deltaT, samplerate)
    signal2_interpolated = compute_offset_signal(signal2, deltaT, samplerate)


    fft_size = 1 << 15

    signal1_dft = 20 * np.log10(np.abs(np.fft.rfft(signal1, fft_size)))
    signal2_dft = 20 * np.log10(np.abs(np.fft.rfft(signal2, fft_size)))

    freqs = np.fft.rfftfreq(fft_size, 1/samplerate)

    fig, (ax1, ax2) = plt.subplots(1, 2)

    ax1.plot(freqs, signal1_dft)
    ax1.set_xlim((0, 5000))
    ax2.plot(freqs, signal2_dft)
    ax2.set_xlim((0, 5000))


    fig, (ax1, ax2) = plt.subplots(1, 2)
    
    line1, = ax1.plot(signal1, signal1_interpolated)
    ax1.spines[['left', 'bottom']].set_position('center')
    ax1.spines[['top', 'right']].set_visible(False)
    # ax1.set_title("une note")

    line2, = ax2.plot(signal2, signal2_interpolated)
    ax2.spines[['left', 'bottom']].set_position('center')
    ax2.spines[['top', 'right']].set_visible(False)
    # ax2.set_title("deux notes")

    axtau = fig.add_axes([0.25, 0.1, 0.65, 0.03])
    tau_slider = Slider(
        ax=axtau,
        label = "tau", 
        valmin=1e-4,
        valmax=1e-3,
        valinit=deltaT        
    )
    
    def slider_update(val): 
        line1.set_ydata(compute_offset_signal(signal1, tau_slider.val, samplerate))
        line2.set_ydata(compute_offset_signal(signal2, tau_slider.val, samplerate))

    tau_slider.on_changed(slider_update)   


    plt.show()
    
    return 


if __name__ == "__main__":
    main()