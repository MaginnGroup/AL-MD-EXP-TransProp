
#----------------------------------------------------------------------------------

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import figstyle

Nr = 30
b_msd1 = np.zeros((Nr), dtype=np.float_)
b_msd2 = np.zeros((Nr), dtype=np.float_)
MSDa = np.zeros((Nr,501), dtype=np.float_)
#----------------------------------------------------------------------------------
#MSD2

for i in range(Nr):

    df = pd.read_csv(f'../{i+1}/data_MSD_2.txt', skiprows=0, names=['MSD'])
    MSD = df[['MSD']].values
    t = np.arange(len(MSD))*200
    
    df = pd.DataFrame()
    df['x'] = np.copy(t)
    df['y'] = np.copy(MSD)    
    x_v = df[['x']]
    y_v = df[['y']]    
    model = LinearRegression()
    model.fit(x_v, y_v)
    b = model.coef_
    a = model.intercept_
    R2 = model.score(x_v,y_v)# para visualizar o R²
   
    MSDa[i,:] = np.squeeze(MSD)
    b_msd2[i] = b[0,0]

MSD_mean = np.mean(MSDa, axis=0)

fig = plt.figure(figsize=(3.0, 2.5),  dpi=300)
for i in range(Nr):
    plt.plot(t, MSDa[i,:], color='lightsteelblue', linewidth='0.4')
plt.plot(t, MSD_mean, 'k-')
plt.ylabel(r'$y$ (\AA$^2$)')
plt.xlabel(r'$t$ (fs)')
plt.savefig('figure_msd2.png') 
plt.show()

#----------------------------------------------------------------------------------
D = (b_msd2/6)*1e-5 
D_mean = np.mean(D)
D_dev = np.std(D)/np.sqrt(Nr) 
print(D_mean, D_dev)

#----------------------------------------------------------------------------------

#convergencia
Dv = np.zeros((Nr), dtype=np.float_)
for i in range(Nr):
    Dv[i] = np.mean(D[:i+1])
print(Dv)
fig = plt.figure(figsize=(3.0, 2.5),  dpi=300)
plt.plot(np.arange(1, Nr + 1), Dv, 'ko')
plt.ylabel(r'$D$ (m$^2$/s)')
plt.xlabel(r'$N_R$')
plt.savefig('figure.png') 
plt.show()




