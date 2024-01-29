# -*- coding: utf-8 -*-


import numpy as np
from scipy.signal import find_peaks
from scipy.special import jv
#from tensorflow-probability.math import bessel_iv_ratio
import tensorflow_probability as tfp
import matplotlib.pyplot as plt


alpha = np.linspace(0, 800,100000)

val = alpha + 1j*alpha
fun = 1j*np.ones(alpha.size, dtype=complex)
fun[alpha<600]  = jv(1, val[alpha<600])/jv(0, val[alpha<600]) 

#fun = tfp.math.bessel_iv_ratio(1, alpha)

plt.figure()
plt.plot(alpha,np.real(fun), label='Real')
plt.plot(alpha,np.imag(fun), label='Imag')
plt.legend()
plt.figure()
plt.plot(alpha,np.imag(fun))

#plt.figure()
#plt.plot(alpha, )
