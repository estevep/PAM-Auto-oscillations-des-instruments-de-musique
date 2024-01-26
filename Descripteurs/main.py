import numpy as np
from numpy.typing import ArrayLike

import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

from scipy.io.wavfile import write, read
from scipy.interpolate import CubicSpline

import os 

samples_dir = f"{os.getcwd()}/Descripteurs/samples"
filenames = os.listdir(samples_dir)


def compute_offset_signal(signal: ArrayLike, deltaT: float, samplerate: float) -> ArrayLike:
    
    T = np.linspace(0, signal.size/samplerate, signal.size)
    interpolator = CubicSpline(T, signal)
    interpolated_signal = interpolator(T + deltaT)

    return interpolated_signal


def import_signal(filepath: str, idx_begin=0, idx_end=None):
    samplerate, signal = read(filepath)

    if signal.ndim > 1:
        signal = signal[idx_begin:idx_end, 0]

    return samplerate, signal


def main() -> None:

    print(filenames)

    samplerate, single_note_signal = import_signal(f"{samples_dir}/{filenames[1]}", 0, 1000)
    _,          double_note_signal = import_signal(f"{samples_dir}/{filenames[2]}", 0, 1000)
    

    deltaT = 4e-4
    single_note_interpolated_signal = compute_offset_signal(single_note_signal, deltaT, samplerate)
    double_note_interpolated_signal = compute_offset_signal(double_note_signal, deltaT, samplerate)


    fig, (ax1, ax2) = plt.subplots(1, 2)
    
    line1, = ax1.plot(single_note_signal, single_note_interpolated_signal)
    ax1.spines[['left', 'bottom']].set_position('center')
    ax1.spines[['top', 'right']].set_visible(False)
    ax1.set_title("une note")

    line2, = ax2.plot(double_note_signal, double_note_interpolated_signal)
    ax2.spines[['left', 'bottom']].set_position('center')
    ax2.spines[['top', 'right']].set_visible(False)
    ax2.set_title("deux notes")

    axtau = fig.add_axes([0.25, 0.1, 0.65, 0.03])
    tau_slider = Slider(
        ax=axtau,
        label = "tau", 
        valmin=1e-4,
        valmax=1e-3,
        valinit=deltaT        
    )
    
    def slider_update(val): 
        line1.set_ydata(compute_offset_signal(single_note_signal, tau_slider.val, samplerate))
        line2.set_ydata(compute_offset_signal(double_note_signal, tau_slider.val, samplerate))

    tau_slider.on_changed(slider_update)   


    plt.show()
    
    return 


if __name__ == "__main__":
    main()