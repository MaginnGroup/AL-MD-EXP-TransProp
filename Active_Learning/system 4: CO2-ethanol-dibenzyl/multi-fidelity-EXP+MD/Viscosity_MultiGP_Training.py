
# =============================================================================
# Imports
# =============================================================================

import os
import warnings
import numpy
import pandas
from sklearn import preprocessing
from matplotlib import pyplot as plt

import torch
from botorch.models import SingleTaskMultiFidelityGP
from botorch.fit import fit_gpytorch_mll
from gpytorch.likelihoods import FixedNoiseGaussianLikelihood
from gpytorch.mlls import ExactMarginalLogLikelihood

import figstyle

# =============================================================================
# Configuration
# =============================================================================

# Path to database folder
dbPath=r'../Databases'
# Database code
code_high='CO2_ethanol_solute_EXP'
code_low='CO2_ethanol_solute_MD_30_replicates'

# =============================================================================
# Auxiliary Functions
# =============================================================================

def normalize(inputArray,skScaler=None,method='Standardization',reverse=False):

    # If inputArray is a labels vector of size (N,), reshape to (N,1)
    if inputArray.ndim==1:
        inputArray=inputArray.reshape((-1,1))
        warnings.warn('Input to normalize() was of shape (N,). It was assumed'\
                      +' to be a column array and converted to a (N,1) shape.')
    # If skScaler is None, train for the first time
    if skScaler is None:
        # Check method
        if method=='Standardization' or method=='MinMax': aux=inputArray
        elif method=='LogStand' or method=='Log': aux=numpy.log(inputArray)
        else: raise ValueError('Could not recognize method in normalize().')
        if method=='MinMax':
            skScaler=preprocessing.MinMaxScaler().fit(aux)
        elif method=='Log':
            skScaler='NA'
        else:
            skScaler=preprocessing.StandardScaler().fit(aux) 
    # Do main operation (normalize or unnormalize)
    if reverse:
        # Rescale the data back to its original distribution
        if method!='Log':
            inputArray=skScaler.inverse_transform(inputArray)
        # Check method
        if method=='LogStand' or method=='Log':
            inputArray=numpy.exp(inputArray)
    elif not reverse:
        # Check method
        if method=='Standardization' or method=='MinMax': aux=inputArray
        elif method=='LogStand' or method=='Log': aux=numpy.log(inputArray)
        else: raise ValueError('Could not recognize method in normalize().')
        if method!='Log':
            inputArray=skScaler.transform(aux)
        else:
            inputArray=aux
    # Return
    return inputArray,skScaler

# =============================================================================
# Load Data
# =============================================================================

trainDB_Path_high=os.path.join(dbPath,code_high+'.csv')
trainDB_high=pandas.read_csv(trainDB_Path_high)

trainDB_Path_low=os.path.join(dbPath,code_low+'.csv')
trainDB_low=pandas.read_csv(trainDB_Path_low)

# =============================================================================
# Main Script
# =============================================================================

# Defining the input/output vectors
X_high=numpy.array([trainDB_high['P/bar'].values,trainDB_high['T/K'].values,trainDB_high['x_ethanol'].values]).T
Y_high1=numpy.array([trainDB_high['Visc/Pas'].values]).T *1e4
errY_high1=0.02*Y_high1

X_low=numpy.array([trainDB_low['P/bar'].values,trainDB_low['T/K'].values,trainDB_low['x_ethanol'].values]).T
Y_low1=numpy.array([trainDB_low['Visc/Pas'].values]).T *1e4
errY_low1=numpy.array([trainDB_low['errVisc'].values]).T *1e4

# Creating a vector for scanning P,T,x1 input conditions
P_range=numpy.linspace(X_low[:,0].min()*0.999,X_low[:,0].max()*1.001,50)
T_range=numpy.linspace(X_low[:,1].min()*0.999,X_low[:,1].max()*1.001,50)
x1_range=numpy.linspace(X_low[:,2].min()*0.999,X_low[:,2].max()*1.001,50)
X_Test=numpy.array([]).reshape(0,3)
for n in range(len(P_range)):
    P_aux=P_range[n]
    for n1 in range(len(T_range)):
        T_aux=T_range[n1]
        aux1=P_aux*numpy.ones(len(P_range)).reshape(-1,1)
        aux2=T_aux*numpy.ones(len(T_range)).reshape(-1,1)
        aux3=x1_range.reshape(-1,1)
        aux=numpy.concatenate((aux1,aux2,aux3),axis=1)
        X_Test=numpy.concatenate((X_Test,aux),axis=0)
X_Test_N,X_Scaler=normalize(numpy.vstack([X_Test,X_Test]),method='MinMax')
X_Test_N = numpy.hstack([X_Test_N,numpy.ones_like(X_Test_N)])[:,:-2]
X_Test_N[:len(X_Test),3] *= 0
X_Test_N[len(X_Test):,3] *= 1
X_Test_low_N = X_Test_N[:len(X_Test),:]
X_Test_high_N = X_Test_N[len(X_Test):,:]
X_test_low_tensor=torch.tensor(X_Test_low_N, dtype=torch.float64)
X_test_high_tensor=torch.tensor(X_Test_high_N, dtype=torch.float64)

# Normalizing input vector
X_Train = numpy.vstack([X_low,X_high])
X_Train_N,__=normalize(X_Train,method='MinMax',skScaler=X_Scaler)
X_Train_N = numpy.hstack([X_Train_N,numpy.ones_like(X_Train_N)])[:,:-2]
X_Train_N[:len(X_low),3] *= 0
X_Train_N[len(X_low):,3] *= 1
train_X = torch.tensor(X_Train_N, dtype=torch.float64)   
X_high_N,__=normalize(X_high,method='MinMax',skScaler=X_Scaler)
X_low_N,__=normalize(X_low,method='MinMax',skScaler=X_Scaler)

# Normalizing output vector
Y_Train1 = numpy.vstack([Y_low1,Y_high1])
Y_Train1_N,Y_Scaler1=normalize(Y_Train1,method='LogStand') 

# Normalizing output vector uncertainty
errY_Train1 = numpy.vstack([errY_low1,errY_high1])
errY_Train1_N = errY_Train1 / Y_Train1 / Y_Scaler1.scale_ 

# Creating torch tensors
train_Y1 = torch.tensor(Y_Train1_N, dtype=torch.float64)
train_errY1 = torch.tensor(errY_Train1_N, dtype=torch.float64)
train_Yvar1 = (train_errY1**2).flatten().to(train_Y1.dtype)

# Using fixed noise of output vector uncertainty
likelihood1 = FixedNoiseGaussianLikelihood(noise=train_Yvar1, learn_additional_noise=False)

# Creating GP model
model1 = SingleTaskMultiFidelityGP(train_X, train_Y1, 
                                  linear_truncated=False, # RBF for features and Downsampling for Fidelities
                                  data_fidelities=[3],
                                  likelihood=likelihood1)
mll1 = ExactMarginalLogLikelihood(model1.likelihood, model1)
fit_gpytorch_mll(mll1)

# High fidelity predictions for AL calculations
with torch.no_grad():
    pred1 = model1.posterior(X_test_high_tensor)
    mean1 = pred1.mean.numpy()
    std_high1 = numpy.sqrt(pred1.variance.numpy())
Y_Pred_high1,__=normalize(mean1,skScaler=Y_Scaler1,method='LogStand',reverse=True)
STD_high1 = std_high1 * Y_Scaler1.scale_ * Y_Pred_high1

# Low fidelity predictions for AL calculations
with torch.no_grad():
    pred3 = model1.posterior(X_test_low_tensor)
    mean3 = pred3.mean.numpy()
    std_low1 = numpy.sqrt(pred3.variance.numpy())
Y_Pred_low1,__=normalize(mean3,skScaler=Y_Scaler1,method='LogStand',reverse=True)
STD_low1 = std_low1 * Y_Scaler1.scale_ * Y_Pred_low1  
    
# High fidelity MRE calculations
with torch.no_grad():
    pred = model1.posterior(torch.tensor(numpy.hstack([X_high_N,1*numpy.ones_like(X_high_N)])[:,:-2], dtype=torch.float64))
    mean = pred.mean.numpy()
Y_p1,__=normalize(mean,method='LogStand',skScaler=Y_Scaler1,reverse=True)
MRE_high = (100*numpy.abs(Y_p1-Y_high1)/Y_high1).mean()    
print(f"Mean Relative Error (MRE) high fidelity: {MRE_high:.2f}%")  
         
# Low fidelity MRE calculations
with torch.no_grad():
    pred = model1.posterior(torch.tensor(numpy.hstack([X_low_N,0*numpy.ones_like(X_low_N)])[:,:-2], dtype=torch.float64))
    mean = pred.mean.numpy()
Y_p1,__=normalize(mean,method='LogStand',skScaler=Y_Scaler1,reverse=True)
MRE_low = (100*numpy.abs(Y_p1-Y_low1)/Y_low1).mean() 
print(f"Mean Relative Error (MRE) low fidelity: {MRE_low:.2f}%")    
                
# =============================================================================
# Plots
# =============================================================================

# Profiles
fig, axs = plt.subplots(1, 3, figsize=(6, 2), dpi=300)

temp1 = numpy.unique(X_high[:, 1])
comp1 = numpy.unique(X_high[:, 2])

X_s = []
Y_s1 = []
errY_s1 = []
for temp in temp1:
    indices = numpy.where(X_high[:, 1] == temp)[0]
    X_sub = X_high[indices]
    Y_sub1 = Y_high1[indices] 
    errY_sub1 = errY_high1[indices]
    X_s.append(X_sub)
    Y_s1.append(Y_sub1)
    errY_s1.append(errY_sub1)

X_s_ = []
Y_s1_ = []
errY_s1_ = []
for comp in comp1:
    indices = numpy.where(X_s[0][:, 2] == comp)[0]
    X_sub = X_s[0][indices]
    Y_sub1 = Y_s1[0][indices]   
    errY_sub1 = errY_s1[0][indices]
    X_s_.append(X_sub)
    Y_s1_.append(Y_sub1)
    errY_s1_.append(errY_sub1)     
axs[0].errorbar(X_s_[0][:,0], numpy.squeeze(Y_s1_[0]), numpy.squeeze(errY_s1_[0]), fmt='o', color='black')
axs[0].errorbar(X_s_[1][:,0], numpy.squeeze(Y_s1_[1]), numpy.squeeze(errY_s1_[1]), fmt='>', color='blue')
axs[0].errorbar(X_s_[2][:,0], numpy.squeeze(Y_s1_[2]), numpy.squeeze(errY_s1_[2]), fmt='*', color='red')
axs[0].errorbar(X_s_[3][:,0], numpy.squeeze(Y_s1_[3]), numpy.squeeze(errY_s1_[3]), fmt='D', color='pink')
axs[0].errorbar(X_s_[4][:,0], numpy.squeeze(Y_s1_[4]), numpy.squeeze(errY_s1_[4]), fmt='<', color='green')

X_s_ = []
Y_s1_ = []
errY_s1_ = []
for comp in comp1:
    indices = numpy.where(X_s[1][:, 2] == comp)[0]
    X_sub = X_s[1][indices]
    Y_sub1 = Y_s1[1][indices]   
    errY_sub1 = errY_s1[1][indices]
    X_s_.append(X_sub)
    Y_s1_.append(Y_sub1)
    errY_s1_.append(errY_sub1)   
axs[1].errorbar(X_s_[0][:,0], numpy.squeeze(Y_s1_[0]), numpy.squeeze(errY_s1_[0]), fmt='o', color='black')
axs[1].errorbar(X_s_[1][:,0], numpy.squeeze(Y_s1_[1]), numpy.squeeze(errY_s1_[1]), fmt='>', color='blue')
axs[1].errorbar(X_s_[2][:,0], numpy.squeeze(Y_s1_[2]), numpy.squeeze(errY_s1_[2]), fmt='*', color='red')
axs[1].errorbar(X_s_[3][:,0], numpy.squeeze(Y_s1_[3]), numpy.squeeze(errY_s1_[3]), fmt='D', color='pink')
axs[1].errorbar(X_s_[4][:,0], numpy.squeeze(Y_s1_[4]), numpy.squeeze(errY_s1_[4]), fmt='<', color='green')

X_s_ = []
Y_s1_ = []
errY_s1_ = []
for comp in comp1:
    indices = numpy.where(X_s[2][:, 2] == comp)[0]
    X_sub = X_s[2][indices]
    Y_sub1 = Y_s1[2][indices]  
    errY_sub1 = errY_s1[2][indices]
    X_s_.append(X_sub)
    Y_s1_.append(Y_sub1)
    errY_s1_.append(errY_sub1)     
axs[2].errorbar(X_s_[0][:,0], numpy.squeeze(Y_s1_[0]), numpy.squeeze(errY_s1_[0]), fmt='o', color='black')
axs[2].errorbar(X_s_[1][:,0], numpy.squeeze(Y_s1_[1]), numpy.squeeze(errY_s1_[1]), fmt='>', color='blue')
axs[2].errorbar(X_s_[2][:,0], numpy.squeeze(Y_s1_[2]), numpy.squeeze(errY_s1_[2]), fmt='*', color='red')
axs[2].errorbar(X_s_[3][:,0], numpy.squeeze(Y_s1_[3]), numpy.squeeze(errY_s1_[3]), fmt='D', color='pink')
axs[2].errorbar(X_s_[4][:,0], numpy.squeeze(Y_s1_[4]), numpy.squeeze(errY_s1_[4]), fmt='<', color='green')


temp1 = numpy.unique(X_Test[:, 1])
comp1 = numpy.unique(X_Test[:, 2])

X_s = []
Y_s1 = []
for temp in temp1:
    indices = numpy.where(X_Test[:, 1] == temp)[0]
    X_sub = X_Test[indices]
    Y_sub1 = Y_Pred_high1[indices] 
    X_s.append(X_sub)
    Y_s1.append(Y_sub1)

X_s_ = []
Y_s1_ = []
for comp in comp1:
    indices = numpy.where(X_s[1][:, 2] == comp)[0]
    X_sub = X_s[1][indices]
    Y_sub1 = Y_s1[1][indices]   
    X_s_.append(X_sub)
    Y_s1_.append(Y_sub1)
axs[0].plot(X_s_[0][:,0], Y_s1_[0], color='black')
axs[0].plot(X_s_[5][:,0], Y_s1_[5], color='blue')
axs[0].plot(X_s_[11][:,0], Y_s1_[11], color='red')
axs[0].plot(X_s_[22][:,0], Y_s1_[22], color='pink')
axs[0].plot(X_s_[49][:,0], Y_s1_[49], color='green')


X_s_ = []
Y_s1_ = []
for comp in comp1:
    indices = numpy.where(X_s[25][:, 2] == comp)[0]
    X_sub = X_s[25][indices]
    Y_sub1 = Y_s1[25][indices]  
    X_s_.append(X_sub)
    Y_s1_.append(Y_sub1)
axs[1].plot(X_s_[0][:,0], Y_s1_[0], color='black')
axs[1].plot(X_s_[5][:,0], Y_s1_[5], color='blue')
axs[1].plot(X_s_[11][:,0], Y_s1_[11], color='red')
axs[1].plot(X_s_[22][:,0], Y_s1_[22], color='pink')
axs[1].plot(X_s_[49][:,0], Y_s1_[49], color='green')

X_s_ = []
Y_s1_ = []
for comp in comp1:
    indices = numpy.where(X_s[48][:, 2] == comp)[0]
    X_sub = X_s[48][indices]
    Y_sub1 = Y_s1[48][indices]   
    X_s_.append(X_sub)
    Y_s1_.append(Y_sub1)
axs[2].plot(X_s_[0][:,0], Y_s1_[0], color='black')
axs[2].plot(X_s_[5][:,0], Y_s1_[5], color='blue')
axs[2].plot(X_s_[11][:,0], Y_s1_[11], color='red')
axs[2].plot(X_s_[22][:,0], Y_s1_[22], color='pink')
axs[2].plot(X_s_[49][:,0], Y_s1_[49], color='green')

axs[0].set_ylabel(r'$\eta \times 10^{-4}$ Pa s')
axs[0].set_xlabel(r'$P$ (bar)')
axs[1].set_xlabel(r'$P$ (bar)')
axs[2].set_xlabel(r'$P$ (bar)')
axs[1].set_yticklabels([])
axs[2].set_yticklabels([])
axs[0].set_xticks([150, 250, 350])
axs[1].set_xticks([150, 250, 350])
axs[2].set_xticks([150, 250, 350])
axs[0].set_ylim(0,9)
axs[1].set_ylim(0,9)
axs[2].set_ylim(0,9)
plt.tight_layout()
#plt.savefig('Training_Multi_Viscosity.png', dpi=600) 



