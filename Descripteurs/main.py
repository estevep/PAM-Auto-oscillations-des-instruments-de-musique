import numpy as np
from numpy.typing import ArrayLike
import matplotlib.pyplot as plt
from scipy.io.wavfile import write, read
from scipy.interpolate import CubicSpline

import os 

samples_dir = f"{os.getcwd()}/Descripteurs/samples"
filenames = os.listdir(samples_dir)


def compute_phase_space(signal: ArrayLike, deltaT: float, samplerate: float) -> ArrayLike:
    
    T = np.linspace(0, signal.size/samplerate, signal.size)
    interpolator = CubicSpline(T, signal)
    interpolated_signal = interpolator(T + deltaT)

    return np.stack([signal, interpolated_signal], axis = 0)


def main() -> None:

    print(filenames)

    samplerate, signal = read(f"{samples_dir}/{filenames[2]}")
    if signal.ndim > 1:
        signal = signal[:, 0]
    
    deltaT = 4e-4
    phase_space = compute_phase_space(signal[:1000], deltaT, samplerate)


    fig, ax = plt.subplots()

    ax.plot(phase_space[0, :], phase_space[1, :])
    ax.spines[['left', 'bottom']].set_position('center')
    ax.spines[['top', 'right']].set_visible(False)
    # ax.set_xlabel(r"t", loc="right")
    # ax.set_ylabel(r"$t + \tau$", loc="top")
    
    plt.show()
    
    return 


if __name__ == "__main__":
    main()