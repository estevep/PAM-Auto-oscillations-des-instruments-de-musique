# -*- coding: utf-8 -*-
"""
Created on Fri Feb  9 15:11:07 2024

@author: charl
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy 
from sklearn import svm
from sklearn.inspection import DecisionBoundaryDisplay


from scipy.signal import find_peaks
import scipy.fft
rng = np.random.default_rng()
from scipy.signal import hilbert, chirp

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
from bruillon_descripteurs import *

from scipy.fft import fft, fftfreq

    
#%%%%%%
def classification (X, y):
    # Train the SVC
    clf = svm.SVC(gamma=10,probability=True).fit(X, y) #,probability=True
    
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
    return clf
#%%%%%
#definition zonne faible proba : 
def front (clf,X_objectif) : 
    l_obj = X_objectif.shape[0]
    proba_max = 1
    index = -1
    val = clf.predict(X_objectif)
    proba = clf.predict_proba(X_objectif)
    
    #front = np.zeros_like(X_objectif[0,:])
    front = [0,0]
    fontiere_new = front
    l = 0
    tat_val_true =0
    n=0
    
    for i in range(l_obj): 
        if (X_objectif[i,1]!=2 and X_objectif[i,1]!=1e-8 and X_objectif[i,0]!=2 and X_objectif[i,0]!=1e-8):
            if val[i]: 
                
                tat_val_true += proba[i][0]
                n+=1
    val_true = np.zeros((n,1))
    n=0  
    for i in range (l_obj): 
        if (X_objectif[i,1]!=2 and X_objectif[i,1]!=1e-8 and X_objectif[i,0]!=2 and X_objectif[i,0]!=1e-8):
            if val[i]:
                val_true[n,:] = proba[i][0]
                n+=1 
    lim = np.percentile(val_true, [90])
    for i in range (l_obj): 
         if (X_objectif[i,1]!=2 and X_objectif[i,1]!=1e-8 and X_objectif[i,0]!=2 and X_objectif[i,0]!=1e-8):    
            if val[i]: 
                if (proba[i][0]>lim): #>0.4999 and proba[i][0]<0.5001
                    front = fontiere_new
                    if l ==0 : 
                        front[l] = X_objectif[i,:]
                        fontiere_new = np.zeros((l+2,2))
                        fontiere_new[l,:]= front[l]
                        l=l+1
                    else : 
                        front[l,:] = X_objectif[i,:]
                        fontiere_new = np.zeros((l+2,2))
                        fontiere_new[:l+1,:]= front
                        l=l+1
            
                
    return front


#front1 = front (clf,X_objectif) 
plt.plot(front1[:,0],front1[:,1], "x")


#%%%%

front1 = front (clf,X_objectif)
def new_point (X, front1): 
    
    l = X.shape[0]
    max_dist = -1
    
    for i in range (front1.shape[0]): 
        point= front1[i,:]
        point=[[point[0],point[1]]]
        #val = clf.predict(point)
        
        dist_nearst_i = 5 
        for k in range (l) :
            
            
            #val_point = clf.predict([[X[k,0],X[k,0]]])
            #if val!=val_point : 
            dist = np.sqrt((X[k,0]-point[0][0])**2+(X[k,1]-point[0][1])**2)
            if dist <dist_nearst_i : 
                dist_nearst_i = dist 
                nearst_i = X[k,:]
            #print(dist_nearst_i)
        if dist_nearst_i>max_dist :
            max_dist=dist_nearst_i
            point_sortie = point
            #print(max_dist)
        
    return point_sortie[0]
    
#point = new_point(X, front1) 


#%%%%
"""
dim_obj = 100 #taille de la grille objectif 
N = 1 #nombre d'itération
n_dim_init=4  # dimension grille de départ 

dim_grid = (n_dim_init,n_dim_init)

g_grid          = np.linspace(1e-8, 2, dim_grid[0])
z_grid          = np.linspace(1e-8,2, dim_grid[0])
g_grid, z_grid  = np.meshgrid(g_grid, z_grid)
g_grid, z_grid  = g_grid.reshape((-1,1)), z_grid.reshape((-1,1))  
X = np.concatenate((g_grid, z_grid),axis=1)

dim_grid = (dim_obj,dim_obj)
g_grid          = np.linspace(1e-8, 2, dim_grid[0])
z_grid          = np.linspace(1e-8,2, dim_grid[0])
g_grid, z_grid  = np.meshgrid(g_grid, z_grid)
g_grid, z_grid  = g_grid.reshape((-1,1)), z_grid.reshape((-1,1))  
X_objectif = np.concatenate((g_grid, z_grid),axis=1)



y = np.full(n_dim_init*n_dim_init, False)
#X_objectif = X

#y : les valeurs à remplacer par calcul dans descripteurs 
for k in range(n_dim_init*n_dim_init):
    print('------ Point (',k+1,' over ',n_dim_init*n_dim_init ,') ------')
    g0, z0 = float(X[k,0]), float(X[k,1]) 
    in_params = {
        "T"     : T,
        "g0"    : g0,
        "z0"    : z0
        }
    P, Pdot = np.zeros(t.size), np.zeros(t.size)
    for i in range(1):
        print('Mode ',i+1,'over ',1)
        args = (gamma_func, zeta_func, in_params, Fn[i], Ymn[i], wn[i])
    
        #temp    = odeint(VDP, Pn0, t, args)
        temp    = RK_solver(VDP, Pn0, t, args)
        P       += temp[:,0]/1
        Pdot    += temp[:,1]/1 
        
    y[k] = is_sound_pression(P)


#y[6] = True

clf = classification (X, y)


#y[3] =True
#taille = 100

#X_objectif = X
X_new,ynew=X,y

for _ in range (7) : 
    X,y=X_new,ynew
    front1 = front (clf,X_objectif)
    coord_new = new_point (X, front1)
    
    plt.plot(front1[:,0],front1[:,1], "x")
    print(coord_new)
    length =  X.shape[0]
    X_new = np.zeros((length+1,2))
    X_new[:length,:] = X
    #coord_new = X_objectif[index,:]
    X_new[length,:] = coord_new
    
    k = length
    print('------ Point (',k+1,' over ',n_dim_init*n_dim_init ,') ------')
    g0, z0 = float(X_new[k,0]), float(X_new[k,1]) 
    in_params = {
        "T"     : T,
        "g0"    : g0,
        "z0"    : z0
        }
    P, Pdot = np.zeros(t.size), np.zeros(t.size)
    for i in range(1):
        print('Mode ',i+1,'over ',1)
        args = (gamma_func, zeta_func, in_params, Fn[i], Ymn[i], wn[i])
    
        #temp    = odeint(VDP, Pn0, t, args)
        temp    = RK_solver(VDP, Pn0, t, args)
        P       += temp[:,0]/1
        Pdot    += temp[:,1]/1 
        
    
    
    ynew = np.full(length+1, False)
    ynew[:length] = y
    ynew[k] = is_sound_pression(P)
    #ynew[length] = valeur_descipteur(coord_new)#[0]
    clf = classification (X_new, ynew)
"""

#%%%

#%%%%
dim_obj = 100 #taille de la grille objectif 
N_iter = 1 #nombre d'itération
n_dim_init=4  # dimension grille de départ 

dim_grid = (n_dim_init,n_dim_init)

g_grid          = np.linspace(1e-8, 2, dim_grid[0])
z_grid          = np.linspace(1e-8,2, dim_grid[0])
g_grid, z_grid  = np.meshgrid(g_grid, z_grid)
g_grid, z_grid  = g_grid.reshape((-1,1)), z_grid.reshape((-1,1))  
X = np.concatenate((g_grid, z_grid),axis=1)

dim_grid = (dim_obj,dim_obj)
g_grid          = np.linspace(1e-8, 2, dim_grid[0])
z_grid          = np.linspace(1e-8,2, dim_grid[0])
g_grid, z_grid  = np.meshgrid(g_grid, z_grid)
g_grid, z_grid  = g_grid.reshape((-1,1)), z_grid.reshape((-1,1))  
X_objectif = np.concatenate((g_grid, z_grid),axis=1)

y = np.full(n_dim_init*n_dim_init, False)
#X_objectif = X

#y : les valeurs à remplacer par calcul dans descripteurs 
for k in range(n_dim_init*n_dim_init):
    print('------ Point (',k+1,' over ',n_dim_init*n_dim_init ,') ------')
    g0, z0 = float(X[k,0]), float(X[k,1]) 
    in_params = {
        "T"     : T,
        "g0"    : g0,
        "z0"    : z0
        }
        
   
    w               = give_w_axis(1, 1e3)
    #Z               = Z_cylinder()
    Z               = import_Z(w, model_type = instrument_type, 
                               model_plot=True)
    wn, Qn, Fn, Ymn = find_Z_model(3, Z) #3 : N nombre de modes
    Z_model         = create_Z_model(wn, Qn, Fn)
    
    P0              = 1e-8*np.random.randn()        # Initial (normalized) pressure value [Pa]
    Pdot0           = 0. 


    solver_type = 'coupled'
    if solver_type == 'uncoupled':
            solver =  uncoupled_solver
    elif solver_type == 'coupled':
        solver =  coupled_solver
    else:
        solver =  coupled_solver
            
        
    Pn0 = [P0, Pdot0]
    t = np.arange(0,T,1/Fs)

    P, Pdot = solver(N, t, gamma_func, zeta_func, in_params, Fn, Ymn, wn, Pn0)
    
        
    y[k] = is_sound_pression(P)


#y[6] = True

clf = classification (X, y)


#y[3] =True
#taille = 100

#X_objectif = X
X_new,ynew=X,y

for _ in range (N_iter) : 
    X,y=X_new,ynew
    front1 = front (clf,X_objectif)
    coord_new = new_point (X, front1)
    
    plt.plot(front1[:,0],front1[:,1], "x")
    print(coord_new)
    length =  X.shape[0]
    X_new = np.zeros((length+1,2))
    X_new[:length,:] = X
    #coord_new = X_objectif[index,:]
    X_new[length,:] = coord_new
    
    k = length

    print('------ Point (',k+1,' over ',n_dim_init*n_dim_init ,') ------')
    g0, z0 = float(X_new[k,0]), float(X_new[k,1]) 
    in_params = {
        "T"     : T,
        "g0"    : g0,
        "z0"    : z0
        }
        
   
    w               = give_w_axis(1, 1e3)
    #Z               = Z_cylinder()
    Z               = import_Z(w, model_type = instrument_type, 
                               model_plot=True)
    wn, Qn, Fn, Ymn = find_Z_model(3, Z) #3 : N nombre de modes
    Z_model         = create_Z_model(wn, Qn, Fn)
    
    P0              = 1e-8*np.random.randn()        # Initial (normalized) pressure value [Pa]
    Pdot0           = 0. 


    solver_type = 'coupled'
    if solver_type == 'uncoupled':
            solver =  uncoupled_solver
    elif solver_type == 'coupled':
        solver =  coupled_solver
    else:
        solver =  coupled_solver
            
        
    Pn0 = [P0, Pdot0]
    t = np.arange(0,T,1/Fs)

    P, Pdot = solver(N, t, gamma_func, zeta_func, in_params, Fn, Ymn, wn, Pn0)
    
        
    #ynew[k] = is_sound_pression(P)
    


    ynew = np.full(length+1, False)
    ynew[:length] = y
    ynew[k] = is_sound_pression(P)
    #ynew[length] = valeur_descipteur(coord_new)#[0]
    clf = classification (X_new, ynew)























#%%%%
mat = clf.predict(X_objectif).reshape((100,100))
mat.shape
Out[104]: (100, 100)

plt.imshow(mat)
Out[105]: <matplotlib.image.AxesImage at 0x10e14482d40>

from matplotlib.image import imsave
imsave("carto_just_test.png", mat)

#%%%%

