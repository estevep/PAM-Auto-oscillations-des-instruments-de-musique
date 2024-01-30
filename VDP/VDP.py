# -*- coding: utf-8 -*-

import numpy as np


# Description : Gives the dynamic model for a simplified reed (prop. to p and no pressing) : VDP equation
def VDP(X, t, gamma_func, zeta_func, in_params, Fn, Ymn, wn):
    gamma = gamma_func(t, in_params)
    zeta = zeta_func(t, in_params)
    A = zeta*(3*gamma-1)/(2*np.sqrt(gamma))
    B = -zeta*(3*gamma+1)/(8*gamma**(3/2))
    C = -zeta*(gamma+1)/(16*gamma**(5/2))
    return [X[1], 
            -wn**2*X[0] - Fn*(Ymn-A)*X[1] + 2*Fn*B*X[0]*X[1] + 3*Fn*C*X[0]**2*X[1]]


# Description : Gives the dynamic model for a simplified reed (prop. to p with pressing) NOT FUNCTIONNAL DUE TO DISCONTINUITIES
def VDP_press(X, t, gamma_func, zeta_func, in_params, Fn, Ymn, wn):
    gamma       = gamma_func(t, in_params)
    zeta        = zeta_func(t, in_params)
    Qn          = wn/(Fn*Ymn)
    pdot        = -wn**2*X[0] - wn/Qn*X[1]
    if (gamma - X[0]) < 1:
        pdot    += X[1]*Fn*zeta*(np.sqrt(gamma-X[0]) - (1+X[0]-gamma)/(2*np.sqrt(gamma-X[0])))
    return [X[1], pdot]


# Description : constant gamma function w.r.t. time
def gamma_func_const(t, in_params):
    g0 = in_params["g0"]
    return g0


# Description : constant zeta function w.r.t. time
def zeta_func_const(t, in_params):
    z0 = in_params["z0"]
    return z0


# Description : provides a gamma function for the user 
def give_gamma_func(func_type = "constant"):
    if func_type == "constant":
        return gamma_func_const
    elif func_type == "multilinear":
        return 
    else:
        return gamma_func_const
    
    
# Description : provides a zeta function for the user
def give_zeta_func(func_type = "constant"):
    if func_type == "constant":
        return zeta_func_const
    elif func_type == "multilinear":
        return 
    else:
        return zeta_func_const