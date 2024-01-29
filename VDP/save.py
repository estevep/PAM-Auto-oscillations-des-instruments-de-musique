# -*- coding: utf-8 -*-

import os
from scipy.io.wavfile import write


# Description : saves the pressure as a .wav file in the given directory
def save_P(save_dir, Fs, P):
    files = [f for f in os.listdir(save_dir)]
    n = -1
    for f in files:
        if f[:19] == "clarinet_synth_VDP_": 
            temp = int(f[19])
            if n < temp:
                n = temp
    n += 1
    write(save_dir+"/clarinet_synth_VDP_"+str(n)+".wav", Fs, P)
    print("Saved "+save_dir+"/clarinet_synth_VDP_"+str(n)+".wav")