
"""

DIFFUSIVITY


"""

import numpy as np
from numba import njit
import mmap


""" INPUTS """
freq = 200 #output frequency
ts = 1 # fs - timestep
nsteps_total = 10000000 #steps
nsteps_amostra = 100000 
norig = 100
nc = 2   # components
N = 2000 # total
N1 = 1900 # hexane
N2 = 100 #toluene

dt = freq*ts 
nframes = int(nsteps_total/freq) 
nlen = int(nsteps_amostra/freq + 1)  
n_init = int(np.floor((nsteps_total-nsteps_amostra)/(norig*freq)) + 1)  

r1_0 = np.zeros((N1, 3), dtype=np.float64) 
r1 = np.zeros((nframes, N1, 3), dtype=np.float64) 
BSD1 = np.zeros((nlen), dtype=np.float64) 
r2_0 = np.zeros((N2, 3), dtype=np.float64) 
r2 = np.zeros((nframes, N2, 3), dtype=np.float64) 
BSD2 = np.zeros((nlen), dtype=np.float64) 
O = np.zeros((nlen, 3), dtype=np.float64)  

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
with open("2.out", "r") as f:
    mmapped_file = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
    linha_inicio = 0
    for _ in range(3):
        linha_inicio = mmapped_file.find(b'\n', linha_inicio) + 1
    for i in range(nframes):
        linha_inicio = mmapped_file.find(b'\n', linha_inicio) + 1
        for j in range(N2):
            linha_fim = mmapped_file.find(b'\n', linha_inicio)
            linha = mmapped_file[linha_inicio:linha_fim].decode('utf-8').split()
            n, x, y, z = map(float, linha) 
            r2[i,j,:] = x, y, z          
            linha_inicio = linha_fim + 1
        
""" ONSAGER """
@njit
def calculos(r1, r2, O, BSD1, BSD2, nframes, norig, nlen, n_init, N1, N2, N):
    nfun = 0
    for i in range(n_init): 
        run = int(np.floor( (nframes - i*norig)/nlen  )) 
        for j in range(run):
            nfun = nfun + 1
            frame0 = int(i*norig + j*nlen)
            r1_0 = np.copy(r1[frame0,:,:])
            r2_0 = np.copy(r2[frame0,:,:])
            for m in range(nlen):
                frame = int(frame0 + m)
                BSD1[m] = BSD1[m] + np.sum( (r1[frame,:,:]-r1_0)**2 )/N1
                BSD2[m] = BSD2[m] + np.sum( (r2[frame,:,:]-r2_0)**2 )/N2
                sum1 = np.sum(r1[frame,:,:]-r1_0, axis=0)
                sum2 = np.sum(r2[frame,:,:]-r2_0, axis=0)
                O[m,0] = O[m,0] + np.sum(sum1*sum2)/N
                O[m,1] = O[m,1] + np.sum(sum1*sum1)/N
                O[m,2] = O[m,2] + np.sum(sum2*sum2)/N 
    return BSD1, BSD2, O, nfun
    
BSD1, BSD2, O, nfun = calculos(r1, r2, O, BSD1, BSD2, nframes, norig, nlen, n_init, N1, N2, N)

BSD1 = BSD1/nfun
BSD2 = BSD2/nfun
O = O/nfun

np.savetxt('data_MSD_1.txt', BSD1, fmt='%e')
np.savetxt('data_MSD_2.txt', BSD2, fmt='%e')
np.savetxt('data_O_12.txt', O[:,0], fmt='%e')
np.savetxt('data_O_11.txt', O[:,1], fmt='%e')
np.savetxt('data_O_22.txt', O[:,2], fmt='%e')


