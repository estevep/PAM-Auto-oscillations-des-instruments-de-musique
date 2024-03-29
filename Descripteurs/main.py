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

samples_dir = f"{os.getcwd()}\\Descripteurs\\samples"
filenames = os.listdir(samples_dir)
filenames = [file for file in filenames if isfile(f"{samples_dir}/{file}")]

class Biquad:
    
    def __init__(self, freq: float, Q: float, samplerate: float) -> None:
        
        self.x1: float = 0.0
        self.x2: float = 0.0
        self.y1: float = 0.0
        self.y2: float = 0.0

        w0 = 2 * np.pi * freq/samplerate;

        alpha = np.sin(w0)/(2*Q)

        a0 = 1 + alpha

        self.b0 = (1 - np.cos(w0))/(2*a0)
        self.b1 = 2*self.b0
        self.b2 = self.b0
        
        self.a1 = -2*np.cos(w0)/a0
        self.a2 = (1 - alpha)/a0

    def process(self, signal: ArrayLike) -> ArrayLike:
        
        for (i,), _ in np.ndenumerate(signal):

            output = (self.b0*signal[i]
                    + self.b1*self.x1
                    + self.b2*self.x2
                    - self.a1*self.y1
                    - self.a2*self.y2)
            
            self.x2 = self.x1
            self.x1 = signal[i]
            self.y2 = self.y1
            self.y1 = output
            signal[i] = output

        return signal
    

def fft_display(signal, samplerate) -> None:
    
    factor = 4
    filters = [Biquad(4500, 0.7, samplerate), Biquad(4500, 0.7, samplerate)]

    signal = filters[0].process(signal)
    signal = filters[1].process(signal)
    under_sampled_signal = signal[::factor]


    fft_size = 1 << 15
    window = None
    signal_size = under_sampled_signal.shape[0]

    if signal_size > fft_size:
        window = under_sampled_signal[:fft_size] * np.hamming(fft_size)
    else:
        window = np.pad(under_sampled_signal, (0, fft_size - signal_size), "constant", constant_values=(0, 0))
        window *= np.hamming(fft_size)

    spectrum = 20*np.log10(np.abs(np.fft.rfft(window, fft_size)))
    freqs = np.fft.rfftfreq(fft_size, factor/samplerate)

    plt.figure()
    plt.semilogx(freqs, spectrum)
    plt.grid(which="both")
    plt.xlim([100, 5000])
    plt.show()
    return 

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
    init_window_size = 256
    init_tau = deltaT


    # fig, line_ax = plt.subplots()
    fig = plt.figure(figsize=(8,6))
    plt.tight_layout()
    line_ax = fig.add_axes([0.15, 0.3, 0.7, 0.6])


    line1, = line_ax.plot(signal, signal_interpolated)
    # line_ax.spines[['left', 'bottom']].set_position('center')
    # line_ax.spines[['top', 'right']].set_visible(False)
    line_ax.set_xlabel("s(t)")
    line_ax.set_ylabel("$s(t + \\tau)$")
    line_ax.set_xlim([-1.5, 1.5])
    line_ax.set_ylim([-1.5, 1.5])
    line_ax.grid(True, which = "major")

    animate = False
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
        # int((signal.shape[0]/samplerate)*10)
        ani = FuncAnimation(fig, animation, frames=1000, interval=100)
        
    plt.show(block = False)

    ax_tau = fig.add_axes([0.25, 0.05, 0.65, 0.03])
    ax_window = fig.add_axes([0.25, 0.1, 0.65, 0.03])
    ax_position = fig.add_axes([0.25, 0.15, 0.65, 0.03])

    window_slider = Slider(
        ax=ax_window, 
        label="largeur de fenêtre", 
        valmin=0,
        valmax=1024,
        valinit=init_window_size
    )

    position_slider = Slider(
        ax=ax_position, 
        label="position de la fenêtre", 
        valmin=0,
        valmax=signal.shape[0] - 1024,
        valinit=init_position
    )

    tau_slider = Slider(
        ax=ax_tau,
        label=r"$\tau \; \; (ms)$", 
        valmin=1e-1,
        valmax=1.,
        valinit=deltaT * 1000,
    )

    def update(val):
        
        position = position_slider.val
        window_length = window_slider.val
        tau = tau_slider.val * 1e-3
        
        line1.set_ydata(compute_offset_signal(signal[int(position):int(position + window_length)], tau, samplerate))
        line1.set_xdata(signal[int(position):int(position + window_length)])

    tau_slider.on_changed(update)   
    window_slider.on_changed(update)   
    position_slider.on_changed(update)   

    plt.show()


def main() -> None:

    # accumuler les passages à zeros et calculer la variance des écarts

    print(filenames)

    # samplerate, signal = import_signal(f"{samples_dir}/{filenames[-3]}")
    samplerate, signal = import_signal(f"Descripteurs\\samples\\real\\Changement de registre 2.wav")
    

    deltaT = 4e-4
    # signal_interpolated2 = compute_offset_signal(signal1_interpolated, deltaT, samplerate)

    display2D(signal, deltaT, samplerate)
    # fft_display(signal, samplerate)

    return 


if __name__ == "__main__":
    main()