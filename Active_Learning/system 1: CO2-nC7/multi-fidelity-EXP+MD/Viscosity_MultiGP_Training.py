
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
code_high='CO2_nC7_EXP'
code_low='CO2_nC7_MD_30_replicates'

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
X_high=numpy.array([trainDB_high['P/bar'].values,trainDB_high['T/K'].values]).T
Y_high1=numpy.array([trainDB_high['Visc/Pas'].values]).T *1e4
errY_high1 = 0.02 * Y_high1

X_low=numpy.array([trainDB_low['P/bar'].values,trainDB_low['T/K'].values]).T
Y_low1=numpy.array([trainDB_low['Visc/Pas'].values]).T *1e4
errY_low1=numpy.array([trainDB_low['errVisc'].values]).T *1e4

# Creating a vector for scanning P,T input conditions
P_range=numpy.linspace(X_low[:,0].min()*0.999,X_low[:,0].max()*1.001,100)
T_range=numpy.linspace(X_low[:,1].min()*0.999,X_low[:,1].max()*1.001,100)
X_Test=numpy.array([]).reshape(0,2)
for n in range(len(P_range)):
    T_aux=T_range[n]
    P_aux=P_range[n]
    aux1=P_aux*numpy.ones(len(P_range)).reshape(-1,1)
    aux2=T_range.reshape(-1,1)
    aux=numpy.concatenate((aux1,aux2),axis=1)
    X_Test=numpy.concatenate((X_Test,aux),axis=0)
X_Test_N,X_Scaler=normalize(numpy.vstack([X_Test,X_Test]),method='MinMax')
X_Test_N = numpy.hstack([X_Test_N,numpy.ones_like(X_Test_N)])[:,:-1]
X_Test_N[:len(X_Test),2] *= 0
X_Test_N[len(X_Test):,2] *= 1
X_Test_low_N = X_Test_N[:len(X_Test),:]
X_Test_high_N = X_Test_N[len(X_Test):,:]
X_test_low_tensor=torch.tensor(X_Test_low_N, dtype=torch.float64)
X_test_high_tensor=torch.tensor(X_Test_high_N, dtype=torch.float64)

# Normalizing input vector
X_Train = numpy.vstack([X_low,X_high])
X_Train_N,__=normalize(X_Train,method='MinMax',skScaler=X_Scaler)
X_Train_N = numpy.hstack([X_Train_N,numpy.ones_like(X_Train_N)])[:,:-1]
X_Train_N[:len(X_low),2] *= 0
X_Train_N[len(X_low):,2] *= 1
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
                                  data_fidelities=[2],
                                  likelihood=likelihood1)
mll1 = ExactMarginalLogLikelihood(model1.likelihood, model1)
fit_gpytorch_mll(mll1)

# High fidelity predictions
with torch.no_grad():
    pred1 = model1.posterior(X_test_high_tensor)
    mean1 = pred1.mean.numpy()
    std_high1 = numpy.sqrt(pred1.variance.numpy())
Y_Pred_high1,__=normalize(mean1,skScaler=Y_Scaler1,method='LogStand',reverse=True)
STD_high1 = std_high1 * Y_Scaler1.scale_ * Y_Pred_high1

# Low fidelity predictions 
with torch.no_grad():
    pred3 = model1.posterior(X_test_low_tensor)
    mean3 = pred3.mean.numpy()
    std_low1 = numpy.sqrt(pred3.variance.numpy())
Y_Pred_low1,__=normalize(mean3,skScaler=Y_Scaler1,method='LogStand',reverse=True)
STD_low1 = std_low1 * Y_Scaler1.scale_ * Y_Pred_low1

# High fidelity MRE calculations
with torch.no_grad():
    pred = model1.posterior(torch.tensor(numpy.hstack([X_high_N,1*numpy.ones_like(X_high_N)])[:,:-1], dtype=torch.float64))
    mean = pred.mean.numpy()
Y_p1,__=normalize(mean,method='LogStand',skScaler=Y_Scaler1,reverse=True)
MRE_high = (100*numpy.abs(Y_p1-Y_high1)/Y_high1).mean()    
print(f"Mean Relative Error (MRE) high fidelity: {MRE_high:.2f}%")  
         
# Low fidelity MRE calculations
with torch.no_grad():
    pred = model1.posterior(torch.tensor(numpy.hstack([X_low_N,0*numpy.ones_like(X_low_N)])[:,:-1], dtype=torch.float64))
    mean = pred.mean.numpy()
Y_p1,__=normalize(mean,method='LogStand',skScaler=Y_Scaler1,reverse=True)
MRE_low = (100*numpy.abs(Y_p1-Y_low1)/Y_low1).mean() 
print(f"Mean Relative Error (MRE) low fidelity: {MRE_low:.2f}%")    

# =============================================================================
# Plots
# =============================================================================

fig, axs = plt.subplots(1, 2, figsize=(5, 2.5), dpi=300)

temps = numpy.unique(X_high[:, 1])
X_s = []
Y_s1 = []
errY_s1 = []
for temp in temps:
    indices = numpy.where(X_high[:, 1] == temp)[0]
    X_s.append(X_high[indices])
    Y_s1.append(Y_high1[indices])
    errY_s1.append(errY_high1[indices])
    
axs[0].errorbar(X_s[0][:,0], numpy.squeeze(Y_s1[0]), numpy.squeeze(errY_s1[0]), fmt='o', color='black')
axs[0].errorbar(X_s[1][:,0], numpy.squeeze(Y_s1[1]), numpy.squeeze(errY_s1[1]), fmt='>', color='blue')
axs[0].errorbar(X_s[2][:,0], numpy.squeeze(Y_s1[2]), numpy.squeeze(errY_s1[2]), fmt='*', color='red')
axs[0].errorbar(X_s[3][:,0], numpy.squeeze(Y_s1[3]), numpy.squeeze(errY_s1[3]), fmt='D', color='purple')
axs[0].errorbar(X_s[4][:,0], numpy.squeeze(Y_s1[4]), numpy.squeeze(errY_s1[4]), fmt='^', color='green')
axs[0].errorbar(X_s[5][:,0], numpy.squeeze(Y_s1[5]), numpy.squeeze(errY_s1[5]), fmt='o', color='m')

temps = numpy.unique(X_low[:, 1])
X_s = []
Y_s1 = []
errY_s1 = []
for temp in temps:
    indices = numpy.where(X_low[:, 1] == temp)[0]
    X_s.append(X_low[indices])
    Y_s1.append(Y_low1[indices])
    errY_s1.append(errY_low1[indices])
    
axs[1].errorbar(X_s[0][:,0], numpy.squeeze(Y_s1[0]), numpy.squeeze(errY_s1[0]), fmt='o', color='black')
axs[1].errorbar(X_s[1][:,0], numpy.squeeze(Y_s1[1]), numpy.squeeze(errY_s1[1]), fmt='>', color='blue')
axs[1].errorbar(X_s[2][:,0], numpy.squeeze(Y_s1[2]), numpy.squeeze(errY_s1[2]), fmt='*', color='red')
axs[1].errorbar(X_s[3][:,0], numpy.squeeze(Y_s1[3]), numpy.squeeze(errY_s1[3]), fmt='D', color='purple')
axs[1].errorbar(X_s[4][:,0], numpy.squeeze(Y_s1[4]), numpy.squeeze(errY_s1[4]), fmt='^', color='green')
axs[1].errorbar(X_s[5][:,0], numpy.squeeze(Y_s1[5]), numpy.squeeze(errY_s1[5]), fmt='o', color='m')

temps = numpy.unique(X_Test[:, 1])
X_s = []
Y_s1 = []
STD_s1 = []
for temp in temps:
    indices = numpy.where(X_Test[:, 1] == temp)[0]
    X_s.append(X_Test[indices])
    Y_s1.append(Y_Pred_high1[indices])
    STD_s1.append(STD_high1[indices])
    
axs[0].plot(X_s[0][:,0], Y_s1[0], color='black')
axs[0].plot(X_s[20][:,0], Y_s1[20], color='blue')
axs[0].plot(X_s[40][:,0], Y_s1[40], color='red')
axs[0].plot(X_s[59][:,0], Y_s1[59], color='purple')
axs[0].plot(X_s[79][:,0], Y_s1[79], color='green')
axs[0].plot(X_s[99][:,0], Y_s1[99], color='m')


temps = numpy.unique(X_Test[:, 1])
X_s = []
Y_s1 = []
Y_s2 = []
STD_s1 = []
STD_s2 = []
for temp in temps:
    indices = numpy.where(X_Test[:, 1] == temp)[0]
    X_s.append(X_Test[indices])
    Y_s1.append(Y_Pred_low1[indices])
    STD_s1.append(STD_low1[indices])
    
axs[1].plot(X_s[0][:,0], Y_s1[0], color='black')
axs[1].plot(X_s[20][:,0], Y_s1[20], color='blue')
axs[1].plot(X_s[40][:,0], Y_s1[40], color='red')
axs[1].plot(X_s[59][:,0], Y_s1[59], color='purple')
axs[1].plot(X_s[79][:,0], Y_s1[79], color='green')
axs[1].plot(X_s[99][:,0], Y_s1[99], color='m')

axs[0].set_title("High Fidelity Data")
axs[1].set_title("Low Fidelity Data")  
axs[0].set_ylabel(r'$\eta \times 10^{-4}$ Pa s')
axs[0].set_xlabel(r'$P$ (bar)')
axs[1].set_xlabel(r'$P$ (bar)')
axs[1].set_yticklabels([])
axs[0].set_xticks([0, 150, 300, 450, 600, 750])
axs[1].set_xticks([0, 150, 300, 450, 600, 750])
axs[0].set_ylim(1,8)
axs[1].set_ylim(1,8)
plt.tight_layout()
#plt.savefig('Training_Multi_Viscosity.png', dpi=600) 
#plt.show()


# for name, param in model1.named_parameters():
#     if param.requires_grad:
#         print(f"{name}: {param.data}")


