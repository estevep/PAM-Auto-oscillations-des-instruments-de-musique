import numpy as np
from numpy.typing import ArrayLike

import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from matplotlib.animation import FuncAnimation

from scipy.io.wavfile import write, read
from scipy.interpolate import CubicSpline
from scipy.signal import welch

import os 
from os.path import isfile

from time import sleep

samples_dir = f"{os.getcwd()}\\Descripteurs\\samples\\real"
filenames = os.listdir(samples_dir)
filenames = [file for file in filenames if isfile(f"{samples_dir}/{file}")]


def compute_offset_signal(signal: ArrayLike, deltaT: float, samplerate: float) -> ArrayLike:
    
    T = np.linspace(0, signal.size/samplerate, signal.size)
    interpolator = CubicSpline(T, signal)
    interpolated_signal = interpolator(T + deltaT)

    return interpolated_signal


def import_signal(filepath: str):
    samplerate, signal = read(filepath)

    if signal.ndim > 1:
        signal = signal[:, 0]
    
    signal = signal.astype("float") / np.max(signal)

    return samplerate, signal


def display2D(signal, deltaT, samplerate):

    signal_interpolated = compute_offset_signal(signal, deltaT, samplerate)


    fft_size = 1 << 16

    # signal1_dft = 20 * np.log10(np.abs(np.fft.rfft(signal, fft_size)))    
    # freqs = np.fft.rfftfreq(fft_size, 1/samplerate)

    freqs, signal_pxx = welch(signal, samplerate, nfft=fft_size) 
    signal_pxx = 20*np.log10(np.abs(signal_pxx))

    # fig, ax = plt.subplots()
    # ax.plot(freqs, signal_pxx)
    # ax.set_xlim((0, 10000))

    init_position = 0
    init_window_size = 512
    init_tau = deltaT


    fig, line_ax = plt.subplots()
    
    # line_ax = fig.add_axes()

    line1, = line_ax.plot(signal, signal_interpolated)
    line_ax.spines[['left', 'bottom']].set_position('center')
    line_ax.spines[['top', 'right']].set_visible(False)

    animate = True
    index = 0

    def animation(i):

        if (i+1) * init_window_size > signal.shape[0]:
            return line1
        
        block = signal[i*init_window_size:((i+1) * init_window_size)]

        line1.set_ydata(compute_offset_signal(block, init_tau, samplerate))
        line1.set_xdata(block)

        return line1

    ani = None
    if animate:
        ani = FuncAnimation(fig, animation, frames=int((signal.shape[0]/samplerate)*10), interval=100)
        
    plt.show(block = True)


    ax_tau = fig.add_axes([0.25, 0.1, 0.65, 0.03])
    ax_window = fig.add_axes([0.25, 0.15, 0.65, 0.03])
    ax_position = fig.add_axes([0.25, 0.2, 0.65, 0.03])

    window_slider = Slider(
        ax=ax_window, 
        label="window length", 
        valmin=0,
        valmax=1024,
        valinit=init_window_size
    )

    position_slider = Slider(
        ax=ax_position, 
        label="window position", 
        valmin=0,
        valmax=signal.shape[0] - 1024,
        valinit=init_position
    )

    tau_slider = Slider(
        ax=ax_tau,
        label="tau", 
        valmin=1e-4,
        valmax=1e-3,
        valinit=deltaT
    )

    def update(val):
        
        position = position_slider.val
        window_length = window_slider.val
        tau = tau_slider.val
        
        line1.set_ydata(compute_offset_signal(signal[int(position):int(position + window_length)], tau, samplerate))
        line1.set_xdata(signal[int(position):int(position + window_length)])

    tau_slider.on_changed(update)   
    window_slider.on_changed(update)   
    position_slider.on_changed(update)   




def main() -> None:

    print(filenames)

    samplerate, signal = import_signal(f"{samples_dir}/{filenames[1]}")
    

    deltaT = 4e-4
    # signal_interpolated2 = compute_offset_signal(signal1_interpolated, deltaT, samplerate)

    display2D(signal, deltaT, samplerate)

    return 


if __name__ == "__main__":
    main()