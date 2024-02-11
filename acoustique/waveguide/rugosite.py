import numpy as np

def rugosite(amp, freq):
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

    return(rug)


#print(rugosite(np.abs(X[peaks]), f[peaks]))









