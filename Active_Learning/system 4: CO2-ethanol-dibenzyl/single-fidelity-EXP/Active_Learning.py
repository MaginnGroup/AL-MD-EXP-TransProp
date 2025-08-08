
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
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from gpytorch.likelihoods import FixedNoiseGaussianLikelihood
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.kernels import RBFKernel

import figstyle

# =============================================================================
# Configuration
# =============================================================================

# Path to database folder
dbPath=r'../Databases'
# Database code
code='CO2_ethanol_solute_EXP'
# Define minimum AF
minAF=0.05
# Define maximum number of iteractions for AL
maxIter=30

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

trainDB_Path=os.path.join(dbPath,code+'.csv')
trainDB=pandas.read_csv(trainDB_Path)

# =============================================================================
# Main Script
# =============================================================================

# Defining the input/output vectors
X_Exp=numpy.array([trainDB['P/bar'].values,trainDB['T/K'].values,trainDB['x_ethanol'].values]).T
Y_Exp1=numpy.array([trainDB['D/m2s-1'].values]).T *1e9
Y_Exp2=numpy.array([trainDB['Visc/Pas'].values]).T *1e4
errY_Exp1=numpy.array([trainDB['errD'].values]).T *1e9
errY_Exp2=0.02*Y_Exp2

# Creating a copy of the input/output vectors
X_Exp_=numpy.copy(X_Exp)
Y_Exp1_=numpy.copy(Y_Exp1)
Y_Exp2_=numpy.copy(Y_Exp2)
errY_Exp1_=numpy.copy(errY_Exp1)
errY_Exp2_=numpy.copy(errY_Exp2)

# Creating a vector for scanning P,T,x1 input conditions
P_range=numpy.linspace(X_Exp[:,0].min()*0.999,X_Exp[:,0].max()*1.001,50)
T_range=numpy.linspace(X_Exp[:,1].min()*0.999,X_Exp[:,1].max()*1.001,50)
x1_range=numpy.linspace(X_Exp[:,2].min()*0.999,X_Exp[:,2].max()*1.001,50)
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
X_Test_N,X_Scaler=normalize(X_Test,method='MinMax')
X_Exp_N,__=normalize(X_Exp,method='MinMax')

# AL training starting points: database endpoints
X_Train = numpy.array([X_Exp[0],X_Exp[-1]])
Y_Train1 = numpy.array([Y_Exp1[0],Y_Exp1[-1]]) 
Y_Train2 = numpy.array([Y_Exp2[0],Y_Exp2[-1]]) 
errY_Train1 = numpy.array([errY_Exp1[0],errY_Exp1[-1]]) 
errY_Train2 = numpy.array([errY_Exp2[0],errY_Exp2[-1]]) 

# Deleting training points from AL available values
X_Exp_=numpy.delete(X_Exp_,(0,-1),axis=0)
Y_Exp1_=numpy.delete(Y_Exp1_,(0,-1),axis=0)
Y_Exp2_=numpy.delete(Y_Exp2_,(0,-1),axis=0)
errY_Exp1_=numpy.delete(errY_Exp1_,(0,-1),axis=0)
errY_Exp2_=numpy.delete(errY_Exp2_,(0,-1),axis=0)

# Saving data during AL
MRE1 = []
MRE2 = []
AF_Hist=[]
AF_Hist1=[]
AF_Hist2=[]

for it in range(maxIter):

    
    # Normalizing input vector
    X_Train_N,__=normalize(X_Train,method='MinMax',skScaler=X_Scaler)
    train_X = torch.tensor(X_Train_N, dtype=torch.float64)

    
    # -----------------------FICK DIFFUSIVITY-------------------------------
    # Normalizing output vector
    Y_Train1_N,Y_Scaler1=normalize(Y_Train1,method='LogStand')
    # Normalizing output vector uncertainty
    errY_Train1_N = errY_Train1 / Y_Train1 / Y_Scaler1.scale_ 
    # Creating torch tensors
    train_Y1 = torch.tensor(Y_Train1_N, dtype=torch.float64)    
    train_errY1 = torch.tensor(errY_Train1_N, dtype=torch.float64)
    train_Yvar1 = (train_errY1**2).flatten().to(train_Y1.dtype)
    # Using fixed noise of output vector uncertainty
    likelihood1 = FixedNoiseGaussianLikelihood(noise=train_Yvar1, learn_additional_noise=False)
    # Creating GP model
    model1 = SingleTaskGP(train_X, train_Y1, likelihood=likelihood1, covar_module = RBFKernel())
    mll1 = ExactMarginalLogLikelihood(model1.likelihood, model1)
    fit_gpytorch_mll(mll1)
    model1.eval()
    # Predictions for AL calculations
    with torch.no_grad():
        pred1 = model1.posterior(torch.tensor(X_Test_N, dtype=torch.float64))
        mean1 = pred1.mean.numpy()
        var1 = pred1.variance.numpy()
        std1 = numpy.sqrt(var1)
    Y_Pred1,__=normalize(mean1,method='LogStand',skScaler=Y_Scaler1,reverse='True')
    STD1 = std1 * Y_Scaler1.scale_ * Y_Pred1
    # Acquisition function    
    AF1=STD1/Y_Pred1
    avgAF1= AF1.max()
    # Predictions for MRE calculations
    with torch.no_grad():
        pred = model1.posterior(torch.tensor(X_Exp_N, dtype=torch.float64))
    Y_p1,__=normalize(pred.mean.numpy(),method='LogStand',skScaler=Y_Scaler1,reverse='True')


    # -----------------------VISCOSITY-------------------------------
    # Normalizing output vector
    Y_Train2_N,Y_Scaler2=normalize(Y_Train2,method='LogStand')
    # Normalizing output vector uncertainty
    errY_Train2_N = errY_Train2 / Y_Train2 / Y_Scaler2.scale_  
    # Creating torch tensors
    train_Y2 = torch.tensor(Y_Train2_N, dtype=torch.float64) 
    train_errY2 = torch.tensor(errY_Train2_N, dtype=torch.float64)
    train_Yvar2 = (train_errY2**2).flatten().to(train_Y2.dtype)
    # Using fixed noise of output vector uncertainty
    likelihood2 = FixedNoiseGaussianLikelihood(noise=train_Yvar2, learn_additional_noise=False)
    # Creating GP model
    model2 = SingleTaskGP(train_X, train_Y2, likelihood=likelihood2, covar_module = RBFKernel())
    mll2 = ExactMarginalLogLikelihood(model2.likelihood, model2)
    fit_gpytorch_mll(mll2)
    model2.eval()
    # Predictions for AL calculations
    with torch.no_grad():
        pred2 = model2.posterior(torch.tensor(X_Test_N, dtype=torch.float64))
        mean2 = pred2.mean.numpy()
        var2 = pred2.variance.numpy()
        std2 = numpy.sqrt(var2)
    Y_Pred2,__=normalize(mean2,method='LogStand',skScaler=Y_Scaler2,reverse='True')
    STD2 = std2 * Y_Scaler2.scale_ * Y_Pred2
    # Acquisition function    
    AF2=STD2/Y_Pred2
    avgAF2= AF2.max()
    # Predictions for MRE calculations
    with torch.no_grad():
        pred = model2.posterior(torch.tensor(X_Exp_N, dtype=torch.float64))
    Y_p2,__=normalize(pred.mean.numpy(),method='LogStand',skScaler=Y_Scaler2,reverse='True')


    # Choosing which transport property has the largest acquisition function
    avgAF, AF = (avgAF1, AF1) if avgAF1 > avgAF2 else (avgAF2, AF2)
    prop = 'Dif' if avgAF1 > avgAF2 else 'Visc'
    
    # Saving data during AL
    if it>0:
        MRE1.append((100*numpy.abs(Y_p1-Y_Exp1)/Y_Exp1).mean())
        MRE2.append((100*numpy.abs(Y_p2-Y_Exp2)/Y_Exp2).mean())
        AF_Hist.append(avgAF)
        AF_Hist1.append(avgAF1)
        AF_Hist2.append(avgAF2)
        # Stopping criterion
        if avgAF<minAF: break
    
    # Selecting the highest AF point
    index=AF.argmax()
    newTrain=X_Test[index].reshape(1,-1)  
    
    # Selecting the closest point in the database
    diff=numpy.abs(X_Exp_-newTrain)
    indexExp=numpy.abs(diff[:,:].mean(axis=1)).argmin()
    print(f"New point: {newTrain}; Closest point: {X_Exp_[indexExp,:]}; {prop}")
    
    # Adding new AL training point 
    X_Train=numpy.concatenate((X_Train,X_Exp_[indexExp,:].reshape(-1,3)), axis=0)
    Y_Train1=numpy.concatenate((Y_Train1,Y_Exp1_[indexExp,:].reshape(-1,1)), axis=0)
    Y_Train2=numpy.concatenate((Y_Train2,Y_Exp2_[indexExp,:].reshape(-1,1)), axis=0)
    errY_Train1=numpy.concatenate((errY_Train1,errY_Exp1_[indexExp,:].reshape(-1,1)),axis=0)
    errY_Train2=numpy.concatenate((errY_Train2,errY_Exp2_[indexExp,:].reshape(-1,1)),axis=0)
    
    # Deleting new training point from AL available values
    X_Exp_=numpy.delete(X_Exp_,indexExp,axis=0)
    Y_Exp1_=numpy.delete(Y_Exp1_,indexExp,axis=0)
    Y_Exp2_=numpy.delete(Y_Exp2_,indexExp,axis=0)
    errY_Exp1_=numpy.delete(errY_Exp1_,indexExp,axis=0)
    errY_Exp2_=numpy.delete(errY_Exp2_,indexExp,axis=0)

# =============================================================================
# Plots
# =============================================================================

# MRE
plt.figure(figsize=(3.0,2.0), dpi=300)
plt.plot(numpy.linspace(1,len(MRE1),len(MRE1)),MRE1,'r--*', label=r'$D$')
plt.plot(numpy.linspace(1,len(MRE2),len(MRE2)),MRE2,'b--*', label=r'$\eta$')
plt.legend()
plt.xticks([2,4,6,8,10,12,14,16,18])
plt.xlabel('Iteration')
plt.ylabel('MRE (\%)')
plt.savefig('AL_MRE.png', dpi=600) 
print(f"Final MRE: Fick diffusivity {MRE1[-1:][0]:.2f}%, viscosity {MRE2[-1:][0]:.2f}%")

# Acquisition Function
plt.figure(figsize=(3.0,2.0), dpi=300)
plt.plot(numpy.linspace(1,len(AF_Hist),len(AF_Hist)), minAF*100*numpy.ones((len(AF_Hist),)), color='gray', alpha=0.5)
plt.plot(numpy.linspace(1,len(AF_Hist),len(AF_Hist)),numpy.array(AF_Hist)*100,'k--*')
plt.xticks([2,4,6,8,10,12,14,16,18])
plt.xlabel('Iteration')
plt.ylabel('Acquisition Function (\%)')
plt.savefig('AL_AF.png', dpi=600) 
print(f"Final AF: Fick diffusivity {100*AF_Hist1[-1:][0]:.2f}%, viscosity {100*AF_Hist2[-1:][0]:.2f}%")

# Profiles
fig, axs = plt.subplots(2, 3, figsize=(6.5, 4.0), dpi=300, constrained_layout=True)

temp1 = numpy.unique(X_Train[:, 1])
X_s = []
Y_s1 = []
Y_s2 = []
for temp in temp1:
    indices = numpy.where(X_Train[:, 1] == temp)[0]
    X_sub = X_Train[indices]
    Y_sub1 = Y_Train1[indices] 
    Y_sub2 = Y_Train2[indices] 
    X_s.append(X_sub)
    Y_s1.append(Y_sub1)
    Y_s2.append(Y_sub2)
axs[1,0].plot(X_s[0][:,0], Y_s1[0], 'o', color='black', markersize=10, markerfacecolor='none')
axs[0,0].plot(X_s[0][:,0], Y_s2[0], 'o', color='black', markersize=10, markerfacecolor='none')
axs[1,1].plot(X_s[1][:,0], Y_s1[1], 'o', color='black', markersize=10, markerfacecolor='none')
axs[0,1].plot(X_s[1][:,0], Y_s2[1], 'o', color='black', markersize=10, markerfacecolor='none')
axs[1,2].plot(X_s[2][:,0], Y_s1[2], 'o', color='black', markersize=10, markerfacecolor='none')
axs[0,2].plot(X_s[2][:,0], Y_s2[2], 'o', color='black', markersize=10, markerfacecolor='none')

axs[0,2].text(X_s[2][1,0], Y_s2[2][1]+1, f"$\mathbf{str(1)}$", color='black', fontsize=10, ha='center', va='center')    
axs[0,0].text(X_s[0][1,0], Y_s2[0][1]+1, f"$\mathbf{str(2)}$", color='black', fontsize=10, ha='center', va='center')    
axs[0,0].text(X_s[0][2,0], Y_s2[0][2]+1, f"$\mathbf{str(3)}$", color='black', fontsize=10, ha='center', va='center')    
axs[0,2].text(X_s[2][2,0]-15, Y_s2[2][2]-0.5, f"$\mathbf{str(4)}$", color='black', fontsize=10, ha='center', va='center')    
axs[0,0].text(X_s[0][3,0], Y_s2[0][3]-1, f"$\mathbf{str(5)}$", color='black', fontsize=10, ha='center', va='center')    
axs[0,2].text(X_s[2][3,0], Y_s2[2][3]+1, f"$\mathbf{str(6)}$", color='black', fontsize=10, ha='center', va='center')   
axs[0,1].text(X_s[1][0,0], Y_s2[1][0]+1, f"$\mathbf{str(7)}$", color='black', fontsize=10, ha='center', va='center')    
axs[0,2].text(X_s[2][4,0]+20, Y_s2[2][4]+1, f"$\mathbf{str(8)}$", color='black', fontsize=10, ha='center', va='center')   
axs[0,0].text(X_s[0][4,0], Y_s2[0][4]+1, f"$\mathbf{str(9)}$", color='black', fontsize=10, ha='center', va='center')   
axs[0,1].text(X_s[1][1,0], Y_s2[1][1]+1, f"$\mathbf{str(10)}$", color='black', fontsize=10, ha='center', va='center')   
axs[0,2].text(X_s[2][5,0], Y_s2[2][5]+1, f"$\mathbf{str(11)}$", color='black', fontsize=10, ha='center', va='center')   
axs[0,1].text(X_s[1][2,0], Y_s2[1][2]+1, f"$\mathbf{str(12)}$", color='black', fontsize=10, ha='center', va='center')   
axs[0,1].text(X_s[1][3,0]+20, Y_s2[1][3]-0.5, f"$\mathbf{str(13)}$", color='black', fontsize=10, ha='center', va='center')   
axs[0,1].text(X_s[1][4,0], Y_s2[1][4]+1, f"$\mathbf{str(14)}$", color='black', fontsize=10, ha='center', va='center')   
axs[0,0].text(X_s[0][5,0], Y_s2[0][5]+1, f"$\mathbf{str(15)}$", color='black', fontsize=10, ha='center', va='center')   
axs[0,1].text(X_s[1][5,0], Y_s2[1][5]+1, f"$\mathbf{str(16)}$", color='black', fontsize=10, ha='center', va='center')   
axs[0,2].text(X_s[2][6,0], Y_s2[2][6]+1, f"$\mathbf{str(17)}$", color='black', fontsize=10, ha='center', va='center')   
axs[0,1].text(X_s[1][6,0], Y_s2[1][6]+1, f"$\mathbf{str(18)}$", color='black', fontsize=10, ha='center', va='center')   

axs[1,2].text(X_s[2][1,0], Y_s1[2][1]+1, f"$\mathbf{str(1)}$", color='black', fontsize=10, ha='center', va='center')    
axs[1,0].text(X_s[0][1,0], Y_s1[0][1]+1, f"$\mathbf{str(2)}$", color='black', fontsize=10, ha='center', va='center')    
axs[1,0].text(X_s[0][2,0], Y_s1[0][2]+1, f"$\mathbf{str(3)}$", color='black', fontsize=10, ha='center', va='center')    
axs[1,2].text(X_s[2][2,0], Y_s1[2][2]+1, f"$\mathbf{str(4)}$", color='black', fontsize=10, ha='center', va='center')    
axs[1,0].text(X_s[0][3,0], Y_s1[0][3]+1, f"$\mathbf{str(5)}$", color='black', fontsize=10, ha='center', va='center')    
axs[1,2].text(X_s[2][3,0], Y_s1[2][3]+1, f"$\mathbf{str(6)}$", color='black', fontsize=10, ha='center', va='center')   
axs[1,1].text(X_s[1][0,0], Y_s1[1][0]+1, f"$\mathbf{str(7)}$", color='black', fontsize=10, ha='center', va='center')    
axs[1,2].text(X_s[2][4,0]+20, Y_s1[2][4], f"$\mathbf{str(8)}$", color='black', fontsize=10, ha='center', va='center')   
axs[1,0].text(X_s[0][4,0], Y_s1[0][4]+1, f"$\mathbf{str(9)}$", color='black', fontsize=10, ha='center', va='center')   
axs[1,1].text(X_s[1][1,0], Y_s1[1][1]+1, f"$\mathbf{str(10)}$", color='black', fontsize=10, ha='center', va='center')   
axs[1,2].text(X_s[2][5,0], Y_s1[2][5]+1, f"$\mathbf{str(11)}$", color='black', fontsize=10, ha='center', va='center')   
axs[1,1].text(X_s[1][2,0], Y_s1[1][2]+1, f"$\mathbf{str(12)}$", color='black', fontsize=10, ha='center', va='center')   
axs[1,1].text(X_s[1][3,0], Y_s1[1][3]+1, f"$\mathbf{str(13)}$", color='black', fontsize=10, ha='center', va='center')   
axs[1,1].text(X_s[1][4,0], Y_s1[1][4]+1, f"$\mathbf{str(14)}$", color='black', fontsize=10, ha='center', va='center')   
axs[1,0].text(X_s[0][5,0], Y_s1[0][5]+1, f"$\mathbf{str(15)}$", color='black', fontsize=10, ha='center', va='center')   
axs[1,1].text(X_s[1][5,0], Y_s1[1][5]+1, f"$\mathbf{str(16)}$", color='black', fontsize=10, ha='center', va='center')   
axs[1,2].text(X_s[2][6,0], Y_s1[2][6]+1, f"$\mathbf{str(17)}$", color='black', fontsize=10, ha='center', va='center')   
axs[1,1].text(X_s[1][6,0], Y_s1[1][6]+1, f"$\mathbf{str(18)}$", color='black', fontsize=10, ha='center', va='center')   

temp1 = numpy.unique(X_Exp[:, 1])
comp1 = numpy.unique(X_Exp[:, 2])

X_s = []
Y_s1 = []
Y_s2 = []
errY_s1 = []
errY_s2 = []
for temp in temp1:
    indices = numpy.where(X_Exp[:, 1] == temp)[0]
    X_sub = X_Exp[indices]
    Y_sub1 = Y_Exp1[indices] 
    Y_sub2 = Y_Exp2[indices] 
    errY_sub1 = errY_Exp1[indices]
    errY_sub2 = errY_Exp2[indices]
    X_s.append(X_sub)
    Y_s1.append(Y_sub1)
    Y_s2.append(Y_sub2)
    errY_s1.append(errY_sub1)
    errY_s2.append(errY_sub2)

X_s_ = []
Y_s1_ = []
Y_s2_ = []
errY_s1_ = []
errY_s2_ = []
for comp in comp1:
    indices = numpy.where(X_s[0][:, 2] == comp)[0]
    X_sub = X_s[0][indices]
    Y_sub1 = Y_s1[0][indices]  
    Y_sub2 = Y_s2[0][indices]  
    errY_sub1 = errY_s1[0][indices]
    errY_sub2 = errY_s2[0][indices]
    X_s_.append(X_sub)
    Y_s1_.append(Y_sub1)
    Y_s2_.append(Y_sub2)
    errY_s1_.append(errY_sub1)   
    errY_s2_.append(errY_sub2)  
axs[1,0].errorbar(X_s_[0][:,0], numpy.squeeze(Y_s1_[0]), numpy.squeeze(errY_s1_[0]), fmt='o', color='black')
axs[1,0].errorbar(X_s_[1][:,0], numpy.squeeze(Y_s1_[1]), numpy.squeeze(errY_s1_[1]), fmt='>', color='blue')
axs[1,0].errorbar(X_s_[2][:,0], numpy.squeeze(Y_s1_[2]), numpy.squeeze(errY_s1_[2]), fmt='*', color='red')
axs[1,0].errorbar(X_s_[3][:,0], numpy.squeeze(Y_s1_[3]), numpy.squeeze(errY_s1_[3]), fmt='D', color='purple')
axs[1,0].errorbar(X_s_[4][:,0], numpy.squeeze(Y_s1_[4]), numpy.squeeze(errY_s1_[4]), fmt='<', color='green')
axs[0,0].errorbar(X_s_[0][:,0], numpy.squeeze(Y_s2_[0]), numpy.squeeze(errY_s2_[0]), fmt='o', color='black')
axs[0,0].errorbar(X_s_[1][:,0], numpy.squeeze(Y_s2_[1]), numpy.squeeze(errY_s2_[1]), fmt='>', color='blue')
axs[0,0].errorbar(X_s_[2][:,0], numpy.squeeze(Y_s2_[2]), numpy.squeeze(errY_s2_[2]), fmt='*', color='red')
axs[0,0].errorbar(X_s_[3][:,0], numpy.squeeze(Y_s2_[3]), numpy.squeeze(errY_s2_[3]), fmt='D', color='purple')
axs[0,0].errorbar(X_s_[4][:,0], numpy.squeeze(Y_s2_[4]), numpy.squeeze(errY_s2_[4]), fmt='<', color='green')

X_s_ = []
Y_s1_ = []
Y_s2_ = []
errY_s1_ = []
errY_s2_ = []
for comp in comp1:
    indices = numpy.where(X_s[1][:, 2] == comp)[0]
    X_sub = X_s[1][indices]
    Y_sub1 = Y_s1[1][indices]  
    Y_sub2 = Y_s2[1][indices]  
    errY_sub1 = errY_s1[1][indices]
    errY_sub2 = errY_s2[1][indices]
    X_s_.append(X_sub)
    Y_s1_.append(Y_sub1)
    Y_s2_.append(Y_sub2)
    errY_s1_.append(errY_sub1)   
    errY_s2_.append(errY_sub2)  
axs[1,1].errorbar(X_s_[0][:,0], numpy.squeeze(Y_s1_[0]), numpy.squeeze(errY_s1_[0]), fmt='o', color='black')
axs[1,1].errorbar(X_s_[1][:,0], numpy.squeeze(Y_s1_[1]), numpy.squeeze(errY_s1_[1]), fmt='>', color='blue')
axs[1,1].errorbar(X_s_[2][:,0], numpy.squeeze(Y_s1_[2]), numpy.squeeze(errY_s1_[2]), fmt='*', color='red')
axs[1,1].errorbar(X_s_[3][:,0], numpy.squeeze(Y_s1_[3]), numpy.squeeze(errY_s1_[3]), fmt='D', color='purple')
axs[1,1].errorbar(X_s_[4][:,0], numpy.squeeze(Y_s1_[4]), numpy.squeeze(errY_s1_[4]), fmt='<', color='green')
axs[0,1].errorbar(X_s_[0][:,0], numpy.squeeze(Y_s2_[0]), numpy.squeeze(errY_s2_[0]), fmt='o', color='black')
axs[0,1].errorbar(X_s_[1][:,0], numpy.squeeze(Y_s2_[1]), numpy.squeeze(errY_s2_[1]), fmt='>', color='blue')
axs[0,1].errorbar(X_s_[2][:,0], numpy.squeeze(Y_s2_[2]), numpy.squeeze(errY_s2_[2]), fmt='*', color='red')
axs[0,1].errorbar(X_s_[3][:,0], numpy.squeeze(Y_s2_[3]), numpy.squeeze(errY_s2_[3]), fmt='D', color='purple')
axs[0,1].errorbar(X_s_[4][:,0], numpy.squeeze(Y_s2_[4]), numpy.squeeze(errY_s2_[4]), fmt='<', color='green')

X_s_ = []
Y_s1_ = []
Y_s2_ = []
errY_s1_ = []
errY_s2_ = []
for comp in comp1:
    indices = numpy.where(X_s[2][:, 2] == comp)[0]
    X_sub = X_s[2][indices]
    Y_sub1 = Y_s1[2][indices]  
    Y_sub2 = Y_s2[2][indices]  
    errY_sub1 = errY_s1[2][indices]
    errY_sub2 = errY_s2[2][indices]
    X_s_.append(X_sub)
    Y_s1_.append(Y_sub1)
    Y_s2_.append(Y_sub2)
    errY_s1_.append(errY_sub1)   
    errY_s2_.append(errY_sub2)  
axs[1,2].errorbar(X_s_[0][:,0], numpy.squeeze(Y_s1_[0]), numpy.squeeze(errY_s1_[0]), fmt='o', color='black')
axs[1,2].errorbar(X_s_[1][:,0], numpy.squeeze(Y_s1_[1]), numpy.squeeze(errY_s1_[1]), fmt='>', color='blue')
axs[1,2].errorbar(X_s_[2][:,0], numpy.squeeze(Y_s1_[2]), numpy.squeeze(errY_s1_[2]), fmt='*', color='red')
axs[1,2].errorbar(X_s_[3][:,0], numpy.squeeze(Y_s1_[3]), numpy.squeeze(errY_s1_[3]), fmt='D', color='purple')
axs[1,2].errorbar(X_s_[4][:,0], numpy.squeeze(Y_s1_[4]), numpy.squeeze(errY_s1_[4]), fmt='<', color='green')
axs[0,2].errorbar(X_s_[0][:,0], numpy.squeeze(Y_s2_[0]), numpy.squeeze(errY_s2_[0]), fmt='o', color='black')
axs[0,2].errorbar(X_s_[1][:,0], numpy.squeeze(Y_s2_[1]), numpy.squeeze(errY_s2_[1]), fmt='>', color='blue')
axs[0,2].errorbar(X_s_[2][:,0], numpy.squeeze(Y_s2_[2]), numpy.squeeze(errY_s2_[2]), fmt='*', color='red')
axs[0,2].errorbar(X_s_[3][:,0], numpy.squeeze(Y_s2_[3]), numpy.squeeze(errY_s2_[3]), fmt='D', color='purple')
axs[0,2].errorbar(X_s_[4][:,0], numpy.squeeze(Y_s2_[4]), numpy.squeeze(errY_s2_[4]), fmt='<', color='green')


temp1 = numpy.unique(X_Test[:, 1])
comp1 = numpy.unique(X_Test[:, 2])

X_s = []
Y_s1 = []
Y_s2 = []
for temp in temp1:
    indices = numpy.where(X_Test[:, 1] == temp)[0]
    X_sub = X_Test[indices]
    Y_sub1 = Y_Pred1[indices] 
    Y_sub2 = Y_Pred2[indices] 
    X_s.append(X_sub)
    Y_s1.append(Y_sub1)
    Y_s2.append(Y_sub2)

X_s_ = []
Y_s1_ = []
Y_s2_ = []
for comp in comp1:
    indices = numpy.where(X_s[1][:, 2] == comp)[0]
    X_sub = X_s[1][indices]
    Y_sub1 = Y_s1[1][indices]  
    Y_sub2 = Y_s2[1][indices]  
    X_s_.append(X_sub)
    Y_s1_.append(Y_sub1)
    Y_s2_.append(Y_sub2) 
axs[1,0].plot(X_s_[0][:,0], Y_s1_[0], color='black')
axs[1,0].plot(X_s_[5][:,0], Y_s1_[5], color='blue')
axs[1,0].plot(X_s_[11][:,0], Y_s1_[11], color='red')
axs[1,0].plot(X_s_[22][:,0], Y_s1_[22], color='purple')
axs[1,0].plot(X_s_[49][:,0], Y_s1_[49], color='green')
axs[0,0].plot(X_s_[0][:,0], Y_s2_[0], color='black')
axs[0,0].plot(X_s_[5][:,0], Y_s2_[5], color='blue')
axs[0,0].plot(X_s_[11][:,0], Y_s2_[11], color='red')
axs[0,0].plot(X_s_[22][:,0], Y_s2_[22], color='purple')
axs[0,0].plot(X_s_[49][:,0], Y_s2_[49], color='green')


X_s_ = []
Y_s1_ = []
Y_s2_ = []
for comp in comp1:
    indices = numpy.where(X_s[25][:, 2] == comp)[0]
    X_sub = X_s[25][indices]
    Y_sub1 = Y_s1[25][indices]  
    Y_sub2 = Y_s2[25][indices]  
    X_s_.append(X_sub)
    Y_s1_.append(Y_sub1)
    Y_s2_.append(Y_sub2) 
axs[1,1].plot(X_s_[0][:,0], Y_s1_[0], color='black')
axs[1,1].plot(X_s_[5][:,0], Y_s1_[5], color='blue')
axs[1,1].plot(X_s_[11][:,0], Y_s1_[11], color='red')
axs[1,1].plot(X_s_[22][:,0], Y_s1_[22], color='purple')
axs[1,1].plot(X_s_[49][:,0], Y_s1_[49], color='green')
axs[0,1].plot(X_s_[0][:,0], Y_s2_[0], color='black')
axs[0,1].plot(X_s_[5][:,0], Y_s2_[5], color='blue')
axs[0,1].plot(X_s_[11][:,0], Y_s2_[11], color='red')
axs[0,1].plot(X_s_[22][:,0], Y_s2_[22], color='purple')
axs[0,1].plot(X_s_[49][:,0], Y_s2_[49], color='green')

X_s_ = []
Y_s1_ = []
Y_s2_ = []
for comp in comp1:
    indices = numpy.where(X_s[48][:, 2] == comp)[0]
    X_sub = X_s[48][indices]
    Y_sub1 = Y_s1[48][indices]  
    Y_sub2 = Y_s2[48][indices]  
    X_s_.append(X_sub)
    Y_s1_.append(Y_sub1)
    Y_s2_.append(Y_sub2) 
axs[1,2].plot(X_s_[0][:,0], Y_s1_[0], color='black')
axs[1,2].plot(X_s_[5][:,0], Y_s1_[5], color='blue')
axs[1,2].plot(X_s_[11][:,0], Y_s1_[11], color='red')
axs[1,2].plot(X_s_[22][:,0], Y_s1_[22], color='purple')
axs[1,2].plot(X_s_[49][:,0], Y_s1_[49], color='green')
axs[0,2].plot(X_s_[0][:,0], Y_s2_[0], color='black')
axs[0,2].plot(X_s_[5][:,0], Y_s2_[5], color='blue')
axs[0,2].plot(X_s_[11][:,0], Y_s2_[11], color='red')
axs[0,2].plot(X_s_[22][:,0], Y_s2_[22], color='purple')
axs[0,2].plot(X_s_[49][:,0], Y_s2_[49], color='green')

from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0], [0], marker='o', color='black', label='0.000'),
    Line2D([0], [0], marker='>', color='blue', label='0.096'),
    Line2D([0], [0], marker='*', color='red', label='0.230'),
    Line2D([0], [0], marker='D', color='purple', label='0.443'),
    Line2D([0], [0], marker='^', color='green', label='1.000'),
]

axs[0,0].legend(handles=legend_elements, ncol=2, fontsize=7, handletextpad=0.4, labelspacing=0.3, handlelength=1.2) 

axs[1,0].set_ylabel(r'$D \times 10^{-9}$ m²/s')
axs[1,1].set_ylabel(r'$D \times 10^{-9}$ m²/s')
axs[1,2].set_ylabel(r'$D \times 10^{-9}$ m²/s')
axs[0,0].set_ylabel(r'$\eta \times 10^{-4}$ Pa s')
axs[0,1].set_ylabel(r'$\eta \times 10^{-4}$ Pa s')
axs[0,2].set_ylabel(r'$\eta \times 10^{-4}$ Pa s')
axs[1,0].set_xlabel(r'$P$ (bar)')
axs[1,1].set_xlabel(r'$P$ (bar)')
axs[1,2].set_xlabel(r'$P$ (bar)')
axs[0,0].set_xticklabels([])
axs[0,1].set_xticklabels([])
axs[0,2].set_xticklabels([])
axs[0,0].set_xticks([150, 250, 350])
axs[0,1].set_xticks([150, 250, 350])
axs[0,2].set_xticks([150, 250, 350])
axs[1,0].set_xticks([150, 250, 350])
axs[1,1].set_xticks([150, 250, 350])
axs[1,2].set_xticks([150, 250, 350])
axs[0,0].set_ylim(0,11)
axs[0,1].set_ylim(0,11)
axs[0,2].set_ylim(0,11)
axs[1,0].set_ylim(0,12)
axs[1,1].set_ylim(0,12)
axs[1,2].set_ylim(0,12)

fig.suptitle(r'\textbf{Experimental Data}', x=0.53, fontsize=10)
axs[0,0].set_title(r'$\bm{T = 313.16}$ \textbf{K}', fontsize=10)
axs[0,1].set_title(r'$\bm{T = 323.16}$ \textbf{K}', fontsize=10)
axs[0,2].set_title(r'$\bm{T = 333.16}$ \textbf{K}', fontsize=10)

#plt.tight_layout()
plt.savefig('AL_Profiles.png', dpi=600) 

