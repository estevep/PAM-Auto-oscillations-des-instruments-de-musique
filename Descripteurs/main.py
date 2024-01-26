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

    

def main() -> None:

    print(filenames)

    samplerate, signal = read(f"{samples_dir}/{filenames[2]}")
    if signal.ndim > 1:
        signal = signal[0:1000, 0]
    
    deltaT = 4e-4
    interpolated_signal = compute_offset_signal(signal, deltaT, samplerate)


    fig, ax = plt.subplots()
    
    line, = ax.plot(signal, interpolated_signal)
    ax.spines[['left', 'bottom']].set_position('center')
    ax.spines[['top', 'right']].set_visible(False)
    

    axtau = fig.add_axes([0.25, 0.1, 0.65, 0.03])
    tau_slider = Slider(
        ax=axtau,
        label = "tau", 
        valmin=1e-4,
        valmax=1e-3,
        valinit=deltaT        
    )
    
    def slider_update(val):
        line.set_ydata(compute_offset_signal(signal, tau_slider.val, samplerate))

    tau_slider.on_changed(slider_update)   


    plt.show()
    
    return 


if __name__ == "__main__":
    main()