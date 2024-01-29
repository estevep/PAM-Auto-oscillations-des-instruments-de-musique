# -*- coding: utf-8 -*-

import numpy as np

def give_F_series(gamma, zeta):
    F0  = zeta*(1-gamma)*np.sqrt(gamma)
    A   = zeta*(3*gamma-1)/(2*np.sqrt(gamma))
    B   = -zeta*(3*gamma+1)/(8*gamma**(3/2))
    C   = -zeta*(gamma+1)/(16*gamma**(5/2))
    
    return F0, A, B, C 


# Description : Gives the dynamic model for a simplified reed (spring only) : VDP equation
def VDP(X, t, gamma_func, zeta_func, in_params, Fn, Ymn, wn):
    gamma = gamma_func(t, in_params)
    zeta = zeta_func(t, in_params)
    _, A, B, C = give_F_series(gamma, zeta)
    return [X[1], 
            -wn**2*X[0] - Fn*(Ymn-A)*X[1] + 2*Fn*B*X[0]*X[1] + 3*Fn*C*X[0]**2*X[1]]


# Description : gives the particle velocity at the resonator's input
def give_u(t, P, gamma_func, zeta_func, in_params):
    gamma = gamma_func(t, in_params)
    zeta = zeta_func(t, in_params)
    F0, A, B, C = give_F_series(gamma, zeta)
    u = F0 + A*P + B*P**2 + C*P**3
    return u


# Description : constant gamma function w.r.t. time
def gamma_func_const(t, in_params):
    g0 = in_params["g0"]
    return g0


# Description : constant zeta function w.r.t. time
def zeta_func_const(t, in_params):
    z0 = in_params["z0"]
    return z0


# Description : constant gamma multilinear function w.r.t. time
def gamma_func_linear(t, in_params):
    gn = np.array(in_params["gn"])
    tn = np.array(in_params["tgn"])
    T  = in_params["T"] 
        
    idx = np.sum(tn<t)
    if idx == tn.size:
        t0 = tn[idx-1]
        t1 = T
        g0 = gn[idx-1]
        g1 = 0
    else:
        t0 = tn[idx-1]
        t1 = tn[idx]
        g0 = gn[idx-1]
        g1 = gn[idx]
    
    return g1 + (t-t0)/(t1-t0)*g1


# Description : constant zeta multilinear function w.r.t. time
def zeta_func_linear(t, in_params):
    zn = in_params["zn"]
    tn = in_params["tzn"]
    T  = in_params["T"] 
        
    idx = np.sum(tn<t)
    if idx == tn.size:
        t0 = zn[idx-1]
        t1 = T
        z0 = zn[idx-1]
        z1 = 0
    else:
        t0 = tn[idx-1]
        t1 = tn[idx]
        z0 = zn[idx-1]
        z1 = zn[idx]
    
    return z1 + (t-t0)/(t1-t0)*z1


# Description : provides a gamma function for the user 
def give_gamma_func(func_type = "constant"):
    if func_type == "constant":
        return gamma_func_const
    elif func_type == "multilinear":
        return gamma_func_linear
    else:
        return gamma_func_const
    
    
# Description : provides a zeta function for the user
def give_zeta_func(func_type = "constant"):
    if func_type == "constant":
        return zeta_func_const
    elif func_type == "multilinear":
        return zeta_func_linear
    else:
        return zeta_func_const