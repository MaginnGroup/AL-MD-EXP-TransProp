
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import figstyle

""" INPUTS """
input_time = 200 # fs
Nt = 501 #number of lines in data files
x1 = 0.95 # mole fraction of compound 1
x2 = 1-x1 # mole fraction of compound 2
eta =  3.35e-4 # viscosity [Pa s]
L = 74.2157205376219e-10 #box size [m]
T = 278.0 # temperature [K]
Nr = 30 #number of replicates
TF = 0.976 #thermodynamic factor

""" STARTING MATRICES AND VECTORS """
bm = np.zeros((Nr), dtype=np.float_)
MSDa = np.zeros((Nr,Nt), dtype=np.float_)
convergence = np.zeros((Nr), dtype=np.float_)

""" FUNCTION"""
def MSD_calc(file_name):
    for i in range(Nr):

        df = pd.read_csv(f'../{i+1}/{file_name}.txt', skiprows=0, names=['MSD'])
        MSD = df[['MSD']].values
        t = np.arange(len(MSD))*input_time 
        df = pd.DataFrame()
        df['x'] = np.copy(t)
        df['y'] = np.copy(MSD)    
        x_v = df[['x']]
        y_v = df[['y']]    
        model = LinearRegression()
        model.fit(x_v, y_v)
        b = model.coef_
        a = model.intercept_
        R2 = model.score(x_v,y_v)   
        MSDa[i,:] = np.squeeze(MSD)
        bm[i] = b[0,0]
    MSD_mean = np.mean(MSDa, axis=0)
    
    fig = plt.figure(figsize=(3.0, 2.5),  dpi=300)
    for i in range(Nr):
        plt.plot(t, MSDa[i,:], color='lightsteelblue', linewidth='0.4')
    plt.plot(t, MSD_mean, 'k-')
    plt.ylabel(r'$y$ (\AA$^2$)')
    plt.xlabel(r'$t$ (fs)')
    plt.savefig(f'{file_name}.png') 
    plt.show()
    
    return bm 
    
""" MSD AND ONSAGER COEFFICIENTS"""

b_msd1 = np.copy(MSD_calc('data_MSD_1'))
b_msd2 = np.copy(MSD_calc('data_MSD_2'))
b_o11 = np.copy(MSD_calc('data_O_11'))
b_o22 = np.copy(MSD_calc('data_O_22'))
b_o12 = np.copy(MSD_calc('data_O_12'))

""" MAXWELL-STEFAN DIFFUSION """
D = (x2/x1*b_o11/6 + x1/x2*b_o22/6 - 2*b_o12/6)*1e-5 
D_mean = np.mean(D)
D_dev = np.std(D)/np.sqrt(Nr)
D_corr = D + 2.837297*1.380649e-23*T/(6*np.pi*eta*L*TF)
D_mean_corr = D_mean + 2.837297*1.380649e-23*T/(6*np.pi*eta*L*TF)

""" FICK DIFFUSION """ 
D_fick = D_corr*TF
D_fick_mean = D_mean_corr*TF
print("Fick:")
print(f"mean, error: {D_fick_mean}, {D_dev}") 

""" CONVERGENCE """
for i in range(Nr):
    convergence[i] = np.mean(D_fick[:i+1])
fig = plt.figure(figsize=(3.0, 2.5),  dpi=300)
plt.plot(np.arange(1, Nr + 1), convergence*1e9, 'ko')
plt.ylabel(r'$D_{11} \times 10^{-9}$ (m$^2$/s)')
plt.xlabel(r'$N_R$')
plt.savefig('convergence.png') 
plt.show()









