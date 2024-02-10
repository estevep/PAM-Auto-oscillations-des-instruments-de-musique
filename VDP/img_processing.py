# -*- coding: utf-8 -*-

import numpy as np
import scipy.signal as sig
import matplotlib.pyplot as plt


# alpha: selectivity [0,1]
def find_frontier(img, alpha=0., plot_frontier=False):
    sobel_x = np.array([[-1,0,1],[-2,0,2],[-1,0,1]])
    sobel_y = np.array([[-1,-2,-1],[0,0,0],[1,2,1]])
    
    der_x = sig.convolve(img, sobel_x,'same')
    der_y = sig.convolve(img, sobel_y,'same')
    
    der_img = der_x**2 + der_y**2
    
    der_img[-1,:]   = 0
    der_img[0,:]    = 0
    der_img[:,-1]   = 0
    der_img[:,0]    = 0
    
    if plot_frontier:
        plt.imshow(der_img)
    
    return np.where(der_img>=((alpha+1e-8)*np.max(der_img)))


# Test function
def edge_of_circle():
    x = np.linspace(0, 10, 1000)
    y = np.linspace(0, 10, 1000)
    X,Y= np.meshgrid(x, y)
    
    img = ((X-5)**2 + (Y-5)**2)<1
    
    plt.imshow(img)
    
    idx = find_frontier(img,0.7, True)
    
    x_front, y_front = X[idx], Y[idx]
    
    plt.figure()
    plt.scatter(x_front,y_front, s=2)
    plt.axis("equal")
    
    return