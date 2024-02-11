# -*- coding: utf-8 -*-
"""
Created on Wed Feb  7 16:25:44 2024

@author: charl
"""
# descipteurs : 

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import scipy.fft
rng = np.random.default_rng()
from scipy.signal import hilbert, chirp

from scipy.integrate import odeint
from sklearn import svm
from sklearn.inspection import DecisionBoundaryDisplay

import scipy.io.wavfile
from scipy.signal import convolve2d
import matplotlib.pyplot as plt
import sys

from scipy.signal.windows import hamming

import os

from VDP import *
from impedance import *
from save import *
from main import *

from scipy.fft import fft, fftfreq

#%%% fonction pour detécttion de pitch (d'après tp JP)



# definition des fonctions : 
def nextpow2(i):
    n = 1
    while n < i:
        n *= 2
    return n

def F_getSpectreFromAudio(x, L_n, Nfft, sr_hz, print=True, plot=True):
    """
    inputs:
        - x: signal
        - L_n: window duration in samples
        - Nfft: fft size
        - sr_hz: sampling rate

    outputs:
        - fftFreq_hz_v (N/2+1,): vector containing the DFT frequencies in Hz
        - fftAmpl_v (N/2+1,): vector containing the DFT amplitudes
    """   
    window = np.pad(hamming(L_n), (0, x.shape[0] - L_n), mode="constant", constant_values = (0, 0)) 
    fftAmpl_v = np.abs(np.fft.rfft(x * window, Nfft))
    fftFreq_hz_v = np.fft.rfftfreq(Nfft, 1/sr_hz)
    return fftFreq_hz_v, fftAmpl_v



def R_max(H, Nfft) -> int:
    return int(Nfft / (H * 2) + 1)

def f_max(H, R, Nfft, sr):
    return H * (R - 1)/Nfft * sr


def F_getF0FromSpectre(fftFreq_hz_v, fftAmpl_v, H, fmin_hz, fmax_hz, Nfft, sr_hz, printing=True, plot=True):
    """
    inputs:
        - fftFreq_hz_v (N/2+1,): vector containing the FFT frequencies in Hz
        - fftAmpl_v (N/2+1,): vector containing the FFT amplitude 
        - H: number of times the spectrum is decimated
        - fmin_hz: minimum frequency in Hz to look for F0
        - fmax_hz: maximum frequency in Hz to look for F0
        - Nfft: fft size
        - sr_hz: sampling rate
        
    outputs:
        - spFreq_hz_v: vector containing the SpectralProduct frequencies in Hz
        - spAmpl_v: vector containing the SpectralProduct amplitudes 
        - f0_hz: estimated F0 in Hz
    """
         
        
    R = R_max(H, Nfft)
    P = np.ones(R, dtype = float)
    
    for h in range(1, H + 1):
        P *= fftAmpl_v[0::h][0:R]
            
            
    spAmpl_v = P
    spFreq_hz_v = fftFreq_hz_v[0:R]
    f0_hz = 0
    
    Nmin = int(fmin_hz/sr_hz * 2 * Nfft)
    Nmax = int(fmax_hz/sr_hz * 2 * Nfft)         
    
    if Nmax >= R : return 0, 0, 0
    f0_hz = spFreq_hz_v[np.argmax(spAmpl_v[Nmin:Nmax]) + Nmin]
   
    return spFreq_hz_v, spAmpl_v, f0_hz



def detect_pich_audioFile(audioFile): #ajouter en option les bornes fmin et fmax
    sr_hz, data_v = scipy.io.wavfile.read(audioFile)
    data_v = data_v[:,1]
    fmin_hz = 50
    fmax_hz = 900
    H = 4
    L_sec = 0.1
    L_n = int(L_sec*sr_hz)
    Nfft = 4*nextpow2(L_n)
    fftFreq_hz_v, fftAmpl_v = F_getSpectreFromAudio(data_v, L_n, Nfft, sr_hz)
    spFreq_hz_v, spAmpl_v, f0_hz = F_getF0FromSpectre(fftFreq_hz_v, fftAmpl_v, H, fmin_hz, fmax_hz, Nfft, sr_hz)
    return f0_hz

def detect_pich_pression(P, Fs): #ajouter en option les bornes fmin et fmax
    sr_hz=Fs
    data_v = P
    fmin_hz = 50
    fmax_hz = 900
    H = 4
    L_sec = 0.1
    L_n = int(L_sec*sr_hz)
    Nfft = 4*nextpow2(L_n)
    fftFreq_hz_v, fftAmpl_v = F_getSpectreFromAudio(data_v, L_n, Nfft, sr_hz)
    spFreq_hz_v, spAmpl_v, f0_hz = F_getF0FromSpectre(fftFreq_hz_v, fftAmpl_v, H, fmin_hz, fmax_hz, Nfft, sr_hz)
    return f0_hz


def inharmonic(f0, h, beta):
    return h*f0*np.sqrt(1 + (h**2 - 1)*beta)



def F_getHarmonicsFromF0AndSpectre_audioFile(audioFile):
    
    sr_hz, data_v = scipy.io.wavfile.read(audioFile)
    data_v = data_v#[:,1]
    fmin_hz = 50
    fmax_hz = 900
    H = 4
    L_sec = 0.1
    L_n = int(L_sec*sr_hz)
    Nfft = 4*nextpow2(L_n)
    fftFreq_hz_v, fftAmpl_v = F_getSpectreFromAudio(data_v, L_n, Nfft, sr_hz)
    spFreq_hz_v, spAmpl_v, f0_hz = F_getF0FromSpectre(fftFreq_hz_v, fftAmpl_v, H, fmin_hz, fmax_hz, Nfft, sr_hz)
    
    alpha = 0.01
    beta = 2e-4

    n_harmo = 20
    
    harmoFreq_k_v = np.zeros(n_harmo, dtype = int)
    harmoAmpl_v = np.zeros(n_harmo)

    for index in range(n_harmo):
        h = index + 1
        fInharmo = inharmonic(f0_hz, h, beta)

        fMin = (1 - alpha) * fInharmo
        fMax = (1 + alpha) * fInharmo

        Nmin = int(fMin/sr_hz * Nfft)
        Nmax = int(fMax/sr_hz * Nfft)            
        nfInharmo = int(fInharmo/sr_hz * Nfft)

        if Nmax > fftFreq_hz_v.shape[0]:
            break

        k = np.argmax(fftAmpl_v[Nmin:Nmax])

        harmoFreq_k_v[index] = Nmin + k
        harmoAmpl_v[index] = fftAmpl_v[Nmin + k]

    return harmoFreq_k_v, harmoAmpl_v


def F_getHarmonicsFromF0AndSpectre_pression(P,Fs):
    
    sr_hz=Fs
    data_v = P
    fmin_hz = 50
    fmax_hz = 900
    H = 4
    L_sec = 0.1
    L_n = int(L_sec*sr_hz)
    Nfft = 4*nextpow2(L_n)
    fftFreq_hz_v, fftAmpl_v = F_getSpectreFromAudio(data_v, L_n, Nfft, sr_hz)
    spFreq_hz_v, spAmpl_v, f0_hz = F_getF0FromSpectre(fftFreq_hz_v, fftAmpl_v, H, fmin_hz, fmax_hz, Nfft, sr_hz)
    
    alpha = 0.01
    beta = 2e-4

    n_harmo = 20
    
    harmoFreq_k_v = np.zeros(n_harmo, dtype = int)
    harmoAmpl_v = np.zeros(n_harmo)

    for index in range(n_harmo):
        h = index + 1
        fInharmo = inharmonic(f0_hz, h, beta)

        fMin = (1 - alpha) * fInharmo
        fMax = (1 + alpha) * fInharmo

        Nmin = int(fMin/sr_hz * Nfft)
        Nmax = int(fMax/sr_hz * Nfft)            
        nfInharmo = int(fInharmo/sr_hz * Nfft)

        if Nmax > fftFreq_hz_v.shape[0]:
            break

        k = np.argmax(fftAmpl_v[Nmin:Nmax])

        harmoFreq_k_v[index] = Nmin + k
        harmoAmpl_v[index] = fftAmpl_v[Nmin + k]

    return harmoFreq_k_v, harmoAmpl_v






#%%%
def is_close(f1, f2, nb_cent_max):
    i = 1200*np.log(np.max([f1,f2])/np.min([f1,f2]))/np.log(2)
    return i< nb_cent_max

#%%%%%% définition des descripteurs 
'''
def is_sound_audioFile(audioFile): #le truc de victor 
    sr_hz, data_v = scipy.io.wavfile.read(audioFile)
    data_v = data_v[:,1]
    P = data_v
    return np.any(P>1e-7)

def est_just_audioFile(audioFile , f_attendu): #f0 est la frequences fondamentale de jeu attendu
    if is_sound_audioFile(audioFile) : 
        f0 = detect_pich_audioFile(audioFile)
        
        return is_close(f0, f_attendu,20 )
    return False


#%%%%%%%
audioFile = 'C:/Users/charl/Documents/PAM-Auto-oscillations-des-instruments-de-musique/Descripteurs/samples/real/Une note 2.wav' 

sr_hz, data_v = scipy.io.wavfile.read(audioFile)
data_v = data_v[:,1]
P = data_v

rugosite(P)
est_rugueux(P)

is_close(135, wn[0]/(2*np.pi) , 100)


'''
#%%%%%


def is_sound_pression(P,f_attendu,tolerence):
    #return np.any(P>1e-7)
    return ((1/len(P))*np.sum(np.abs(P[100:]))>tolerence)
 #tolerence : 0.1
#permet de definir une zonne de jeu juste autour de la frequence fondamentale du resonateur 
#ajouter calcul de f_attendu en fonction de la longueur du resonnateur? 

def est_just_pression(P , f_attendu,tolerence): #f0 est la frequences fondamentale de jeu attendu
    if is_sound_pression(P,f_attendu,tolerence) : 
        f0 = detect_pich_pression(P,Fs)
        return is_close(f0, f_attendu, tolerence)
    else : return False


def rugosite(P):
    harmoFreq_k_v, harmoAmpl_v = F_getHarmonicsFromF0AndSpectre_pression(P,Fs)
    freq = harmoFreq_k_v
    amp = harmoAmpl_v
    denom = 0
    num = 0 

    for i in range(freq.size):
        denom+= amp[i]**2
        for j in range(freq.size):
            cb = 1.72*(freq[i]+freq[j])**0.65
            delta = (freq[i]-freq[j])/(cb*0.25)
            r = (2.7183*delta*np.exp(-delta))**2

            num += amp[i]*amp[j]*r
            
    rug = num/denom     

    return(np.log(rug))

def est_rugueux(P,f_attendu,tolerence):
    if is_sound_pression(P, f_attendu, 0.1):
        return rugosite(P)<tolerence
    else :
        return False 
    

from scipy.signal import hilbert
import signal_envelope as se

def quasi_periodic (P): 
    P = P[100:]
    analytic_signal =( hilbert(P)) #amplitude_envelope 
    Pe = np.abs(analytic_signal)
    
    indice = np.var(Pe)/np.mean(Pe)
    #plt.plot(Pe[2000:8000])
    #plt.plot(P[2000:8000])
    return np.log(indice)

def is_quasi_periodic(P, f_attendu,tolerence): 
    return  quasi_periodic (P) > tolerence

def is_periodic(P, f_attendu,tolerence): 
    if is_sound_pression(P, f_attendu, 0.1):
        return  quasi_periodic (P) < tolerence
    else :
        return False 

#%%%
"""
is_sound_pression(P_matrice[100])

for k in range (g_grid.size) : 
    if is_sound_pression(P_matrice[k]):
        if is_periodic(P_matrice[k], -4):
            print (k)
        
k = 108
print(k,quasi_periodic(P_matrice[k]))
detect_pich_pression(P_matrice[k],Fs)

"""
#%%% Code de Victor pour SVM 

"""
def plot_training_data_with_decision_boundary(X, y):
    # Train the SVC
    clf = svm.SVC(gamma=10).fit(X, y)

    # Settings for plotting
    _, ax = plt.subplots(figsize=(7, 7))
    x_min, x_max, y_min, y_max = 0, 2, 0, 2
    ax.set(xlim=(x_min, x_max), ylim=(y_min, y_max))

    # Plot decision boundary and margins
    common_params = {"estimator": clf, "X": X, "ax": ax}
    DecisionBoundaryDisplay.from_estimator(
        **common_params,
        grid_resolution=1000,
        response_method="predict",
        plot_method="pcolormesh",
        alpha=0.3,
    )
    DecisionBoundaryDisplay.from_estimator(
        **common_params,
        response_method="decision_function",
        plot_method="contour",
        levels=[-1, 0, 1],
        colors=["k", "k", "k"],
        linestyles=["--", "-", "--"],
    )

    # Plot bigger circles around samples that serve as support vectors
    ax.scatter(
        clf.support_vectors_[:, 0],
        clf.support_vectors_[:, 1],
        s=50,
        facecolors="none",
        edgecolors="k",
    )
    # Plot samples by color and add legend
    scatter = ax.scatter(X[:, 0], X[:, 1], c=y, s=50, label=y, edgecolors="k")
    ax.legend(*scatter.legend_elements(), loc="upper right", title="Classes")
    ax.set_title(f" Decision boundaries in SVC")

    _ = plt.show()
"""

#%%%% Essay carto d'après code de victor
# %% ------------------- USER INPUT -------------------
"""
plot_bool   = True
save_bool   = True
save_dir    = f"{os.getcwd()}\\..\\Descripteurs\\samples"
Fs          = 44100                         # Sampling frequency [Hz]
T           = 5                             # Total time of the signal [s]
N           = 3                             # Number of cavity modes 
P0          = 1e-8*np.random.randn()        # Initial (normalized) pressure value [Pa]
Pdot0       = 0.                            # Initial time derivative of the (normalized) pressure value [Pa/s]
gamma_func  = give_gamma_func("constant")   # Gamma function with respect to time
zeta_func   = give_zeta_func("constant")    # Zeta function with respect to time

# Gamma/Zeta parameters
g0          = 0.4
z0          = 0.5

in_params = {
    "T"     : T,
    "g0"    : g0,
    "z0"    : z0
    }

# Impedance   

w               = give_w_axis()
#Z               = Z_cylinder()
Z               = import_Z(give_w_axis())
wn, Qn, Fn, Ymn = find_Z_model(N, Z)
Z_model         = create_Z_model(wn, Qn, Fn)

# Solve the VDP equation

Pn0 = [P0, Pdot0]
t = np.arange(0,T,1/Fs)

# Gamma/Zeta parameters grid
taille = 15
dim_grid = (taille,taille)

g_grid          = np.linspace(1e-8, 2, dim_grid[0])
z_grid          = np.linspace(1e-8,2, dim_grid[0])
g_grid, z_grid  = np.meshgrid(g_grid, z_grid)
g_grid, z_grid  = g_grid.reshape((-1,1)), z_grid.reshape((-1,1))  


P_matrice = np.zeros((taille*taille,len(t)))

for k in range(g_grid.size):
    print('------ Point (',k+1,' over ',g_grid.size,') ------')
    g0, z0 = float(g_grid[k]), float(z_grid[k])
    in_params = {
        "T"     : T,
        "g0"    : g0,
        "z0"    : z0
        }
    P, Pdot = np.zeros(t.size), np.zeros(t.size)
    for i in range(N):
        print('Mode ',i+1,'over ',N)
        args = (gamma_func, zeta_func, in_params, Fn[i], Ymn[i], wn[i])
    
        temp    = odeint(VDP, Pn0, t, args)
        #temp    = RK_solver(VDP, Pn0, t, args)
        P       += temp[:,0]/N
        Pdot    += temp[:,1]/N


    P_matrice[k] = P
"""
#%%%%%
#P_matrice2 = P_matrice 
#P_matrice = P_matrice1 

#%%%%%%%%
"""
crit_grid_is_sound = np.full(g_grid.size, False)

crit_grid_is_juste = np.full(g_grid.size, False)
crit_grid_est_rugueux = np.full(g_grid.size, False)
crit_grid_is_second = np.full(g_grid.size, False)
crit_grid_is_quasi_periodic = np.full(g_grid.size, False)
crit_grid_is_periodic = np.full(g_grid.size, False)


for k in range(g_grid.size):
    
    if is_sound_pression(P_matrice[k]) : 
        crit_grid_is_sound[k] = is_sound_pression(P_matrice[k])
        crit_grid_est_rugueux[k]=est_rugueux(P_matrice[k],np.log(1e+28))
        crit_grid_is_second[k]= est_just_pression(P_matrice[k],wn[1]/(2*(np.pi)),200)
        crit_grid_is_quasi_periodic[k] = is_quasi_periodic(P_matrice[k], -2)
        crit_grid_is_periodic[k] = is_periodic(P_matrice[k], -2)
        crit_grid_is_juste[k]= est_just_pression(P_matrice[k], wn[0]/(2*np.pi),20)

    else : 
        crit_grid_is_juste[k]= False
        crit_grid_est_rugueux[k]=False
        crit_grid_is_sound[k] = False
        crit_grid_is_second[k]=False
        crit_grid_is_quasi_periodic[k] = False
        crit_grid_is_periodic[k] = False

    
#%%%%
i = 1200*np.log(np.max([180,wn[0]/(2*np.pi)])/np.min([150,wn[0]/(2*np.pi)]))/np.log(2)

    

#%%%%
X = np.concatenate((g_grid, z_grid),axis=1)

plot_training_data_with_decision_boundary(X, crit_grid_is_sound)

plot_training_data_with_decision_boundary(X, crit_grid_is_juste)

plot_training_data_with_decision_boundary(X,crit_grid_est_rugueux)

#plot_training_data_with_decision_boundary(X,crit_grid_is_second)

plot_training_data_with_decision_boundary(X,crit_grid_is_quasi_periodic)

plot_training_data_with_decision_boundary(X,crit_grid_is_periodic)

"""

