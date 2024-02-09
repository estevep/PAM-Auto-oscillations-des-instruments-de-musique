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

#X la grille




#%%%%
# X_objectif : grille de 20*20, 400*2
point=[[1.85,0.01]]
print(clf.predict(point))
proba = clf.predict_proba(point)
proba = proba[0][0]
print(proba)

#%%% 
def valeur_descipteur(coord_new):
    return True 
    #return np.random.randn(1)<0

d = valeur_descipteur(coord_new)[0]
#%%%%
taille = 100 #taille de la grille objectif 
N = 10 #nombre d'itération
n_dim_init=4
def classification_itere(taille,N,n_dim_init): 
        
        
    
    
    dim_grid = (n_dim_init,n_dim_init)
    
    g_grid          = np.linspace(1e-8, 2, dim_grid[0])
    z_grid          = np.linspace(1e-8,2, dim_grid[0])
    g_grid, z_grid  = np.meshgrid(g_grid, z_grid)
    g_grid, z_grid  = g_grid.reshape((-1,1)), z_grid.reshape((-1,1))  
    X = np.concatenate((g_grid, z_grid),axis=1)
    
    #X_objectif = X
    
    #y : les valeurs à remplacer par calcul dans descripteurs 
    
    y = np.full(g_grid.size, False)
    y[6] = True
    
    #y[3] =True
    #taille = 100
    dim_grid = (taille,taille)
    g_grid          = np.linspace(1e-8, 2, dim_grid[0])
    z_grid          = np.linspace(1e-8,2, dim_grid[0])
    g_grid, z_grid  = np.meshgrid(g_grid, z_grid)
    g_grid, z_grid  = g_grid.reshape((-1,1)), z_grid.reshape((-1,1))  
    X_objectif = np.concatenate((g_grid, z_grid),axis=1)
    
    
    #X_objectif = X
    X_new,ynew = X,y
    for _ in range (N) : 
        X,y=X_new,ynew
        clf = classification (X, y)
        l = X_objectif.shape[0]
        proba_max = 1
        index = -1
        for i in range(l): 
            point= X_objectif[i,:]
            point=[[point[0],point[1]]]
            if not(point in X) :
                val = clf.predict(point)
                if val : 
                    proba = clf.predict_proba(point)
                    proba = proba[0][1]
                    if proba < proba_max : 
                        index = i
                        proba_max = proba
    
        
        
        length =  X.shape[0]
        X_new = np.zeros((length+1,2))
        X_new[:length,:] = X
        coord_new = X_objectif[index,:]
        X_new[length,:] = coord_new
        
        ynew = np.full(length+1, False)
        ynew[:length] = y
        ynew[length] = valeur_descipteur(coord_new)#[0]
    classification (X_new, ynew)
        
#%%%%
classification_itere(taille,3,n_dim_init)

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



#%%%%%%%

