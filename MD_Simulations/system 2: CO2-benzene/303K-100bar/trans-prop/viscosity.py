
"""

VISCOSITY (GREEN-KUBO)  


"""
import pandas as pd
import numpy as np
import scipy.integrate
from multiprocessing import Pool

""" INPUTS """
freq = 10 # data output frequency 
ts = 1 # fs - timestep
nsteps_total = 10000000 # total number of steps  
volume = 95.896512086 #[nm^3]
temperature = 303.0 #[K]
kb = 1.38064852e-23 #[J/K]
conv = 1e-32 * (0.986923)**2 #[nm^3/J * bar^2 * fs] = [nm^3/(Pa m^3)  * bar^2 * fs] -->  1e-27 * 1e10 * 1e-15 Pa s = 1e-32 Pa s

""" INITIAL CALCULATIONS """
dt = freq*ts # interval between outputs in unit of time
wline = int(nsteps_total/freq) # total number of frames
time = np.arange(0,nsteps_total,freq)*ts # vector of time 
                                                                        
""" STARTING MATRICES AND VECTORS """
P = np.zeros((wline, 6), dtype=np.float64) # pressure

""" READING PRESSURE TENSOR """
data = pd.read_csv('./output.out', sep=" ")
P[:,0] = data['Pxy'].values[1:] #Pxy
P[:,1] = data['Pxz'].values[1:] #Pxz
P[:,2] = data['Pyz'].values[1:] #Pyz
P[:,3] = data['Pxx'].values[1:] #Pxx
P[:,4] = data['Pyy'].values[1:] #Pyy
P[:,5] = data['Pzz'].values[1:] #Pzz

""" MAIN """
def autocorrelate (a):
    b=np.concatenate((a,np.zeros(len(a))),axis=0)
    c= np.fft.ifft(np.fft.fft(b)*np.conjugate(np.fft.fft(b))).real
    d=c[:len(c)//2]
    d=d/(np.array(range(len(a)))+1)[::-1]
    return d[:wline]

if __name__=='__main__':
    p = Pool(8)
    a1 = P[:,0] 
    a2 = P[:,1]
    a3 = P[:,2]
    a4 = P[:,3] - P[:,4]
    a5 = P[:,4] - P[:,5]
    a6 = P[:,3] - P[:,5]
    array_array=[a1,a2,a3,a4,a5,a6]
    pv=p.map(autocorrelate,array_array)
    pcorr = (pv[0]+pv[1]+pv[2])/6+(pv[3]+pv[4]+pv[5])/24
    visco = ( scipy.integrate.cumtrapz(pcorr, time[:len(pcorr)]) ) * volume/(kb*temperature) * conv          

    output=open('visc_time2.out','w')
    output.write('#Time (fs), Viscosity (Pa s), Average Pressure Correlation (bar^2)\n')
    for line in zip(time[:len(pcorr)-1],visco,pcorr):
        output.write(' '.join(str(x) for x in line)+'\n')
    output.close()

