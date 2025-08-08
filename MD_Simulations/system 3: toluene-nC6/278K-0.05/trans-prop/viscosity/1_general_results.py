
import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit
import figstyle

""" INPUTS """
Nt = 50000 #number of lines in data files
Ni = 200 # deleting initial 2ps of analysis
Nr = 10 #number of replicates

""" STARTING MATRICES AND VECTORS """
viscosity = np.zeros((Nt-Ni, Nr), dtype=np.float_)  
convergence = np.zeros((Nr), dtype=np.float_)  
value = np.zeros((Nr), dtype=np.float_)  

""" FUNCTIONS """
def biexponential(x, eta_inf, alpha, beta1, beta2):
    return ( alpha * beta1 * (1 - np.exp(-x / beta1)) + (1 - alpha) * beta2 * (1 - np.exp(-x / beta2)) ) / ( alpha * beta1 + (1-alpha)*beta2 ) * eta_inf

""" READING VISCOSITIES OVER TIME """
for i in range(Nr):
    data = np.loadtxt(f"../{i+1}/visc_time2.out", delimiter = " ", skiprows=1)
    viscosity[:,i] = data[Ni:Nt,1] #Pa.s
time = data[Ni:Nt,0] #fs

""" AVERAGE AND STANDARD DEVIATION """
viscosity_mean = np.mean(viscosity, axis=1)
viscosity_deviation = np.sqrt( 1/(Nr-1) * np.sum((viscosity - viscosity_mean[:, None])**2, axis=1) )

""" t_cut """
mask = viscosity_deviation <= 0.4 * viscosity_mean
tcut_index = np.argmax(~mask)  
if mask.all(): # Handle the case where the condition is true for all elements
    tcut_index = len(viscosity_mean)-1
print(f"t_cut: {time[tcut_index]}")    

""" CONVERGENCE """
index=0
for i in range(1,Nr): 
    selected_indices = np.arange(0,i)
    sample = viscosity[:,selected_indices]
    mean_sample = np.mean(sample, axis=1)
    params, covariance = curve_fit(biexponential, time/1000, mean_sample, method='trf', maxfev=1000000, p0=[4e-04, 9e-01, 6e-01, 2e+01], bounds=(0, 40))
    eta_inf, alpha, beta1, beta2 = params
    convergence[index] = eta_inf
    index+=1

""" FINAL MEAN AND ERROR """
for i in range(Nr):
    params, covariance = curve_fit(biexponential, time/1000, viscosity[:,i], method='trf', maxfev=1000000, p0=[4e-04, 9e-01, 6e-01, 2e+01], bounds=(0, 40))
    eta_inf, alpha, beta1, beta2 = params
    value[i] = eta_inf
convergence[index] = np.mean(value)
value_mean = np.mean(value)
value_error = np.std(value)/np.sqrt(Nr)
print(f"mean, error: {value_mean}, {value_error}")  


""" PLOTS """
fig = plt.figure(figsize=(3.0, 2.5),  dpi=300)
for i in range(Nr):
    plt.plot(time/1000,viscosity[:,i]*1e4, color='lightsteelblue', linewidth='0.4')
plt.plot(time/1000,viscosity_mean*1e4, 'k-')
plt.ylabel(r"$\eta \times 10^{-4}$ (Pa s)")
plt.xlabel("t (ps)")
plt.xticks([0,250,500,750,1000])
plt.savefig("ViscTime.png")

fig = plt.figure(figsize=(3.0, 2.5),  dpi=300)
plt.plot(time/1000,viscosity_deviation*1e4, 'k-')
plt.ylabel(r"$\sigma \times 10^{-4}$ (Pa s)")
plt.xlabel("t (ps)")
plt.xticks([0,250,500,750,1000])
plt.savefig("DeviationTime.png")

fig = plt.figure(figsize=(3.0, 2.5),  dpi=300)
plt.plot(np.arange(1, Nr + 1), convergence*1e4, 'ko')
plt.xticks([0,5,10,15,20,25,30])
plt.ylabel(r"$\eta \times 10^{-4}$ (Pa s)")
plt.xlabel(r'$N_R$')
plt.savefig('convergence.png') 
plt.show()



