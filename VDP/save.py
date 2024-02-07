# -*- coding: utf-8 -*-

import os
from os.path import isdir 
from scipy.io.wavfile import write



# Description : saves the pressure as a .wav file in the given directory
def save_P(save_dir, Fs, P):
    
    if not isdir(save_dir): 
        os.mkdir(save_dir)
    
    files = [f for f in os.listdir(save_dir)]
    n = -1
    for f in files:
        if f[:19] == "clarinet_synth_VDP_": 
            temp = int(f[19])
            if n < temp:
                n = temp
    n += 1
    
    full_name = save_dir+"/clarinet_synth_VDP_"+str(n)+".wav"
    write(full_name, Fs, P)
    print("Saved as "+full_name)