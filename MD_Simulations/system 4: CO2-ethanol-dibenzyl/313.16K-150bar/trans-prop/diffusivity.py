
"""

SELF-DIFFUSIVITY


"""
import numpy as np
from numba import njit
import mmap


""" INPUTS """
freq = 200 # data output frequency
ts = 1 # fs - timestep
nsteps_total = 10000000 # total number of steps
nsteps_amostra = 100000 
norig = 100 
N1 = 10 #number of dibenzyl ether molecules

""" INITIAL CALCULATIONS """
dt = freq*ts 
nframes = int(nsteps_total/freq) 
nlen = int(nsteps_amostra/freq + 1)  
n_init = int(np.floor((nsteps_total-nsteps_amostra)/(norig*freq)) + 1)  

""" STARTING MATRICES AND VECTORS """
r1_0 = np.zeros((N1, 3), dtype=np.float64) 
r1 = np.zeros((nframes, N1, 3), dtype=np.float64) 
BSD1 = np.zeros((nlen), dtype=np.float64) 

""" READ CENTER OF MASS COORDINATES """
with open("1.out", "r") as f:
    mmapped_file = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
    linha_inicio = 0
    for _ in range(3):
        linha_inicio = mmapped_file.find(b'\n', linha_inicio) + 1
    for i in range(nframes):
        linha_inicio = mmapped_file.find(b'\n', linha_inicio) + 1
        for j in range(N1):
            linha_fim = mmapped_file.find(b'\n', linha_inicio)
            linha = mmapped_file[linha_inicio:linha_fim].decode('utf-8').split()
            n, x, y, z = map(float, linha) 
            r1[i,j,:] = x, y, z          
            linha_inicio = linha_fim + 1
        
""" MAIN """
@njit
def calculos(r1, BSD1, nframes, norig, nlen, n_init, N1):
    nfun = 0
    for i in range(n_init): 
        run = int(np.floor( (nframes - i*norig)/nlen  )) 
        for j in range(run):
            nfun = nfun + 1
            frame0 = int(i*norig + j*nlen)
            r1_0 = np.copy(r1[frame0,:,:])
            for m in range(nlen):
                frame = int(frame0 + m)
                BSD1[m] = BSD1[m] + np.sum( (r1[frame,:,:]-r1_0)**2 )/N1

    return BSD1, nfun
    
BSD1, nfun = calculos(r1, BSD1, nframes, norig, nlen, n_init, N1)

BSD1 = BSD1/nfun

np.savetxt('data_MSD_1.txt', BSD1, fmt='%e')


