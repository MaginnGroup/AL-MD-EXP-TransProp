
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
# Define high fidelity cost relative to low fidelity cost
cost_high=1
cost_low=1
# Define minimum AF
minAF=0.05
# Define maximum number of iteractions for AL
maxIter=40

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
Y_high1=numpy.array([trainDB_high['D/m2s-1'].values]).T *1e9
Y_high2=numpy.array([trainDB_high['Visc/Pas'].values]).T *1e4
errY_high1=numpy.array([trainDB_high['errD'].values]).T *1e9
errY_high2 = 0.02 * Y_high2

X_low=numpy.array([trainDB_low['P/bar'].values,trainDB_low['T/K'].values,trainDB_low['x_ethanol'].values]).T
Y_low1=numpy.array([trainDB_low['D/m2s-1'].values]).T *1e9
Y_low2=numpy.array([trainDB_low['Visc/Pas'].values]).T *1e4
errY_low1=numpy.array([trainDB_low['errD'].values]).T *1e9
errY_low2=numpy.array([trainDB_low['errVisc'].values]).T *1e4

# Creating a copy of the input/output vectors
X_high_ = X_high.copy()
Y_high1_ = Y_high1.copy()
Y_high2_ = Y_high2.copy()
errY_high1_ = errY_high1.copy()
errY_high2_ = errY_high2.copy()

X_low_ = X_low.copy()
Y_low1_ = Y_low1.copy()
Y_low2_ = Y_low2.copy()
errY_low1_ = errY_low1.copy()
errY_low2_ = errY_low2.copy()

# Creating a vector for scanning P,T input conditions
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

# AL training starting points: database endpoints
xlow = numpy.array([X_low[0], X_low[-1]])
ylow1 = numpy.array([Y_low1[0],Y_low1[-1]]) 
ylow2 = numpy.array([Y_low2[0],Y_low2[-1]])
errylow1 = numpy.array([errY_low1[0],errY_low1[-1]])  
errylow2 = numpy.array([errY_low2[0],errY_low2[-1]])  

xhigh = numpy.array([X_high[0], X_high[-1]])
yhigh1 = numpy.array([Y_high1[0],Y_high1[-1]])
yhigh2 = numpy.array([Y_high2[0],Y_high2[-1]])
erryhigh1 = numpy.array([errY_high1[0],errY_high1[-1]])
erryhigh2 = numpy.array([errY_high2[0],errY_high2[-1]])

# Deleting training points from AL available values
X_low_=numpy.delete(X_low_,(0,-1),axis=0)
Y_low1_=numpy.delete(Y_low1_,(0,-1),axis=0)
Y_low2_=numpy.delete(Y_low2_,(0,-1),axis=0)
errY_low1_=numpy.delete(errY_low1_,(0,-1),axis=0)
errY_low2_=numpy.delete(errY_low2_,(0,-1),axis=0)

X_high_=numpy.delete(X_high_,(0,-1),axis=0)
Y_high1_=numpy.delete(Y_high1_,(0,-1),axis=0)
Y_high2_=numpy.delete(Y_high2_,(0,-1),axis=0)
errY_high1_=numpy.delete(errY_high1_,(0,-1),axis=0)
errY_high2_=numpy.delete(errY_high2_,(0,-1),axis=0)

# Saving data during AL
MRE1 = []
MRE2 = []
AF_Hist_high=[]
AF_Hist_low=[]

for it in range(maxIter):

    
    # Normalizing input vector
    X_Train = numpy.vstack([xlow,xhigh])
    X_Train_N,__=normalize(X_Train,method='MinMax',skScaler=X_Scaler)
    X_Train_N = numpy.hstack([X_Train_N,numpy.ones_like(X_Train_N)])[:,:-2]
    X_Train_N[:len(xlow),3] *= 0
    X_Train_N[len(xlow):,3] *= 1
    train_X = torch.tensor(X_Train_N, dtype=torch.float64)   
    X_high_N,__=normalize(X_high,method='MinMax',skScaler=X_Scaler)

    
    # -----------------------FICK DIFFUSIVITY-------------------------------
    Y_Train1 = numpy.vstack([ylow1,yhigh1])
    errY_Train1 = numpy.vstack([errylow1,erryhigh1])
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
    # High fidelity acquisition function 
    AF_high1=STD_high1/Y_Pred_high1
    avgAF_high1= AF_high1.max() 
    # Low fidelity acquisition function 
    AF_low1=STD_low1/Y_Pred_low1
    avgAF_low1= AF_low1.max()   
    # Predictions for MRE calculations
    with torch.no_grad():
        pred = model1.posterior(torch.tensor(numpy.hstack([X_high_N,1*numpy.ones_like(X_high_N)])[:,:-2], dtype=torch.float64))
        mean = pred.mean.numpy()
    Y_p1,__=normalize(mean,method='LogStand',skScaler=Y_Scaler1,reverse=True)
    
    
    # -----------------------VISCOSITY-------------------------------
    Y_Train2 = numpy.vstack([ylow2,yhigh2])
    errY_Train2 = numpy.vstack([errylow2,erryhigh2])
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
    model2 = SingleTaskMultiFidelityGP(train_X, train_Y2, 
                                      linear_truncated=False, # RBF for features and Downsampling for Fidelities
                                      data_fidelities=[3],
                                      likelihood=likelihood2)
    mll2 = ExactMarginalLogLikelihood(model2.likelihood, model2)
    fit_gpytorch_mll(mll2)
    # High fidelity predictions for AL calculations 
    with torch.no_grad():
        pred2 = model2.posterior(X_test_high_tensor)
        mean2 = pred2.mean.numpy()
        std_high2 = numpy.sqrt(pred2.variance.numpy())
    Y_Pred_high2,__=normalize(mean2,skScaler=Y_Scaler2,method='LogStand',reverse=True)
    STD_high2 = std_high2 * Y_Scaler2.scale_ * Y_Pred_high2
    # Low fidelity predictions for AL calculations
    with torch.no_grad():
        pred4 = model2.posterior(X_test_low_tensor)
        mean4 = pred4.mean.numpy()
        std_low2 = numpy.sqrt(pred4.variance.numpy())
    Y_Pred_low2,__=normalize(mean4,skScaler=Y_Scaler2,method='LogStand',reverse=True)
    STD_low2 = std_low2 * Y_Scaler2.scale_ * Y_Pred_low2
    # High fidelity acquisition function
    AF_high2=STD_high2/Y_Pred_high2
    avgAF_high2= AF_high2.max()
    # Low fidelity acquisition function
    AF_low2=STD_low2/Y_Pred_low2
    avgAF_low2= AF_low2.max()
    # Predictions for MRE calculations
    with torch.no_grad():
        pred = model2.posterior(torch.tensor(numpy.hstack([X_high_N,1*numpy.ones_like(X_high_N)])[:,:-2], dtype=torch.float64))
        mean = pred.mean.numpy()
    Y_p2,__=normalize(mean,method='LogStand',skScaler=Y_Scaler2,reverse=True)

    
    # Choosing which transport property has the largest acquisition function
    avgAF_high, AF_high = (avgAF_high1, AF_high1) if avgAF_high1 > avgAF_high2 else (avgAF_high2, AF_high2)
    avgAF_low, AF_low = (avgAF_low1, AF_low1) if avgAF_low1 > avgAF_low2 else (avgAF_low2, AF_low2)

    # Saving data during AL
    if it > 0:
        MRE1.append((100*numpy.abs(Y_p1-Y_high1)/Y_high1).mean())
        MRE2.append((100*numpy.abs(Y_p2-Y_high2)/Y_high2).mean())    
        AF_Hist_high.append(avgAF_high)
        AF_Hist_low.append(avgAF_low)
        # Stopping criterion
        if (avgAF_high)<minAF: break
    
    # Choosing whether to add a new high fidelity or low fidelity point
    if (avgAF_low/cost_low) >= (avgAF_high/cost_high):
        # Selecting the highest AF point
        index=AF_low.argmax()
        newTrain=X_Test[index].reshape(1,-1)
        # Selecting the closest point in the database
        diff=numpy.abs(X_low_-newTrain)
        indexExp=numpy.abs(diff[:,:].mean(axis=1)).argmin()
        print(f"LOW FIDELITY - New point: {newTrain}; Closest point: {X_low_[indexExp,:]}")
        # Adding new AL training point        
        xlow=numpy.concatenate((xlow,X_low_[indexExp,:].reshape(-1,3)),axis=0)
        ylow1=numpy.concatenate((ylow1,Y_low1_[indexExp,:].reshape(-1,1)),axis=0)
        ylow2=numpy.concatenate((ylow2,Y_low2_[indexExp,:].reshape(-1,1)),axis=0)
        errylow1=numpy.concatenate((errylow1,errY_low1_[indexExp,:].reshape(-1,1)),axis=0)
        errylow2=numpy.concatenate((errylow2,errY_low2_[indexExp,:].reshape(-1,1)),axis=0)
        # Deleting new training point from AL available values
        X_low_=numpy.delete(X_low_,indexExp,axis=0)
        Y_low1_=numpy.delete(Y_low1_,indexExp,axis=0)
        Y_low2_=numpy.delete(Y_low2_,indexExp,axis=0)
        errY_low1_=numpy.delete(errY_low1_,indexExp,axis=0)
        errY_low2_=numpy.delete(errY_low2_,indexExp,axis=0)
    else:
        # Selecting the highest AF point
        index=AF_high.argmax()   
        newTrain=X_Test[index].reshape(1,-1)
        # Selecting the closest point in the database
        diff=numpy.abs(X_high_-newTrain)
        indexExp=numpy.abs(diff[:,:].mean(axis=1)).argmin()
        print(f"HIGH FIDELITY - New point: {newTrain}; Closest point: {X_high_[indexExp,:]}")
        # Adding new AL training point 
        xhigh=numpy.concatenate((xhigh,X_high_[indexExp,:].reshape(-1,3)),axis=0)
        yhigh1=numpy.concatenate((yhigh1,Y_high1_[indexExp,:].reshape(-1,1)),axis=0)
        yhigh2=numpy.concatenate((yhigh2,Y_high2_[indexExp,:].reshape(-1,1)),axis=0)
        erryhigh1=numpy.concatenate((erryhigh1,errY_high1_[indexExp,:].reshape(-1,1)),axis=0)
        erryhigh2=numpy.concatenate((erryhigh2,errY_high2_[indexExp,:].reshape(-1,1)),axis=0)
        # Deleting new training point from AL available values
        X_high_=numpy.delete(X_high_,indexExp,axis=0)
        Y_high1_=numpy.delete(Y_high1_,indexExp,axis=0)    
        Y_high2_=numpy.delete(Y_high2_,indexExp,axis=0) 
        errY_high1_=numpy.delete(errY_high1_,indexExp,axis=0)    
        errY_high2_=numpy.delete(errY_high2_,indexExp,axis=0) 
                
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
plt.plot(numpy.linspace(1,len(AF_Hist_high),len(AF_Hist_high)), minAF*100*numpy.ones((len(AF_Hist_high),)), color='gray', alpha=0.5)
plt.plot(numpy.linspace(1,len(AF_Hist_low),len(AF_Hist_low)),numpy.array(AF_Hist_low)*100,'b--*', label='Low-Fidelity')
plt.plot(numpy.linspace(1,len(AF_Hist_high),len(AF_Hist_high)),numpy.array(AF_Hist_high)*100,'k--*', label='High-Fidelity')
plt.legend()
plt.xticks([2,4,6,8,10,12,14,16,18])
plt.xlabel('Iteration')
plt.ylabel('Acquisition Function (\%)')
plt.savefig('AL_AF.png', dpi=600) 
print(f"Final AF: High Fidelity {100*AF_Hist_high[-1:][0]:.2f}%, Low Fidelity {100*AF_Hist_low[-1:][0]:.2f}%")


# Profiles high-fidelity

fig, axs = plt.subplots(2, 3, figsize=(6.5, 4.0), dpi=300)

temp1 = numpy.unique(xhigh[:, 1])
X_s = []
Y_s1 = []
Y_s2 = []
for temp in temp1:
    indices = numpy.where(xhigh[:, 1] == temp)[0]
    X_sub = xhigh[indices]
    Y_sub1 = yhigh1[indices] 
    Y_sub2 = yhigh2[indices] 
    X_s.append(X_sub)
    Y_s1.append(Y_sub1)
    Y_s2.append(Y_sub2)
axs[1,0].plot(X_s[0][:,0], Y_s1[0], 'o', color='black', markersize=10, markerfacecolor='none')
axs[0,0].plot(X_s[0][:,0], Y_s2[0], 'o', color='black', markersize=10, markerfacecolor='none')
axs[1,2].plot(X_s[1][:,0], Y_s1[1], 'o', color='black', markersize=10, markerfacecolor='none')
axs[0,2].plot(X_s[1][:,0], Y_s2[1], 'o', color='black', markersize=10, markerfacecolor='none')

axs[0,0].text(X_s[0][1,0], Y_s2[0][1]+1, f"$\mathbf{str(5)}$", color='black', fontsize=10, ha='center', va='center') 
axs[0,2].text(X_s[1][1,0]-15, Y_s2[1][1]-0.5, f"$\mathbf{str(6)}$", color='black', fontsize=10, ha='center', va='center') 
axs[0,0].text(X_s[0][2,0], Y_s2[0][2]+1, f"$\mathbf{str(8)}$", color='black', fontsize=10, ha='center', va='center') 
axs[0,2].text(X_s[1][2,0], Y_s2[1][2]+1, f"$\mathbf{str(13)}$", color='black', fontsize=10, ha='center', va='center') 
axs[0,0].text(X_s[0][3,0], Y_s2[0][3]+1, f"$\mathbf{str(15)}$", color='black', fontsize=10, ha='center', va='center') 
axs[0,2].text(X_s[1][3,0], Y_s2[1][3]+1, f"$\mathbf{str(16)}$", color='black', fontsize=10, ha='center', va='center') 
axs[0,0].text(X_s[0][4,0], Y_s2[0][4]+1, f"$\mathbf{str(18)}$", color='black', fontsize=10, ha='center', va='center') 

axs[1,0].text(X_s[0][1,0], Y_s1[0][1]+1, f"$\mathbf{str(5)}$", color='black', fontsize=10, ha='center', va='center') 
axs[1,2].text(X_s[1][1,0], Y_s1[1][1]+1, f"$\mathbf{str(6)}$", color='black', fontsize=10, ha='center', va='center') 
axs[1,0].text(X_s[0][2,0], Y_s1[0][2]+1, f"$\mathbf{str(8)}$", color='black', fontsize=10, ha='center', va='center') 
axs[1,2].text(X_s[1][2,0], Y_s1[1][2]+1, f"$\mathbf{str(13)}$", color='black', fontsize=10, ha='center', va='center') 
axs[1,0].text(X_s[0][3,0], Y_s1[0][3]+1, f"$\mathbf{str(15)}$", color='black', fontsize=10, ha='center', va='center') 
axs[1,2].text(X_s[1][3,0], Y_s1[1][3]+1, f"$\mathbf{str(16)}$", color='black', fontsize=10, ha='center', va='center') 
axs[1,0].text(X_s[0][4,0], Y_s1[0][4]+1, f"$\mathbf{str(18)}$", color='black', fontsize=10, ha='center', va='center') 


temp1 = numpy.unique(X_high[:, 1])
comp1 = numpy.unique(X_high[:, 2])

X_s = []
Y_s1 = []
Y_s2 = []
errY_s1 = []
errY_s2 = []
for temp in temp1:
    indices = numpy.where(X_high[:, 1] == temp)[0]
    X_sub = X_high[indices]
    Y_sub1 = Y_high1[indices] 
    Y_sub2 = Y_high2[indices] 
    errY_sub1 = errY_high1[indices]
    errY_sub2 = errY_high2[indices]
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
axs[1,0].errorbar(X_s_[3][:,0], numpy.squeeze(Y_s1_[3]), numpy.squeeze(errY_s1_[3]), fmt='D', color='pink')
axs[1,0].errorbar(X_s_[4][:,0], numpy.squeeze(Y_s1_[4]), numpy.squeeze(errY_s1_[4]), fmt='<', color='green')
axs[0,0].errorbar(X_s_[0][:,0], numpy.squeeze(Y_s2_[0]), numpy.squeeze(errY_s2_[0]), fmt='o', color='black')
axs[0,0].errorbar(X_s_[1][:,0], numpy.squeeze(Y_s2_[1]), numpy.squeeze(errY_s2_[1]), fmt='>', color='blue')
axs[0,0].errorbar(X_s_[2][:,0], numpy.squeeze(Y_s2_[2]), numpy.squeeze(errY_s2_[2]), fmt='*', color='red')
axs[0,0].errorbar(X_s_[3][:,0], numpy.squeeze(Y_s2_[3]), numpy.squeeze(errY_s2_[3]), fmt='D', color='pink')
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
axs[1,1].errorbar(X_s_[3][:,0], numpy.squeeze(Y_s1_[3]), numpy.squeeze(errY_s1_[3]), fmt='D', color='pink')
axs[1,1].errorbar(X_s_[4][:,0], numpy.squeeze(Y_s1_[4]), numpy.squeeze(errY_s1_[4]), fmt='<', color='green')
axs[0,1].errorbar(X_s_[0][:,0], numpy.squeeze(Y_s2_[0]), numpy.squeeze(errY_s2_[0]), fmt='o', color='black')
axs[0,1].errorbar(X_s_[1][:,0], numpy.squeeze(Y_s2_[1]), numpy.squeeze(errY_s2_[1]), fmt='>', color='blue')
axs[0,1].errorbar(X_s_[2][:,0], numpy.squeeze(Y_s2_[2]), numpy.squeeze(errY_s2_[2]), fmt='*', color='red')
axs[0,1].errorbar(X_s_[3][:,0], numpy.squeeze(Y_s2_[3]), numpy.squeeze(errY_s2_[3]), fmt='D', color='pink')
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
axs[1,2].errorbar(X_s_[3][:,0], numpy.squeeze(Y_s1_[3]), numpy.squeeze(errY_s1_[3]), fmt='D', color='pink')
axs[1,2].errorbar(X_s_[4][:,0], numpy.squeeze(Y_s1_[4]), numpy.squeeze(errY_s1_[4]), fmt='<', color='green')
axs[0,2].errorbar(X_s_[0][:,0], numpy.squeeze(Y_s2_[0]), numpy.squeeze(errY_s2_[0]), fmt='o', color='black')
axs[0,2].errorbar(X_s_[1][:,0], numpy.squeeze(Y_s2_[1]), numpy.squeeze(errY_s2_[1]), fmt='>', color='blue')
axs[0,2].errorbar(X_s_[2][:,0], numpy.squeeze(Y_s2_[2]), numpy.squeeze(errY_s2_[2]), fmt='*', color='red')
axs[0,2].errorbar(X_s_[3][:,0], numpy.squeeze(Y_s2_[3]), numpy.squeeze(errY_s2_[3]), fmt='D', color='pink')
axs[0,2].errorbar(X_s_[4][:,0], numpy.squeeze(Y_s2_[4]), numpy.squeeze(errY_s2_[4]), fmt='<', color='green')


temp1 = numpy.unique(X_Test[:, 1])
comp1 = numpy.unique(X_Test[:, 2])

X_s = []
Y_s1 = []
Y_s2 = []
for temp in temp1:
    indices = numpy.where(X_Test[:, 1] == temp)[0]
    X_sub = X_Test[indices]
    Y_sub1 = Y_Pred_high1[indices] 
    Y_sub2 = Y_Pred_high2[indices] 
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
axs[1,0].plot(X_s_[22][:,0], Y_s1_[22], color='pink')
axs[1,0].plot(X_s_[49][:,0], Y_s1_[49], color='green')
axs[0,0].plot(X_s_[0][:,0], Y_s2_[0], color='black')
axs[0,0].plot(X_s_[5][:,0], Y_s2_[5], color='blue')
axs[0,0].plot(X_s_[11][:,0], Y_s2_[11], color='red')
axs[0,0].plot(X_s_[22][:,0], Y_s2_[22], color='pink')
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
axs[1,1].plot(X_s_[22][:,0], Y_s1_[22], color='pink')
axs[1,1].plot(X_s_[49][:,0], Y_s1_[49], color='green')
axs[0,1].plot(X_s_[0][:,0], Y_s2_[0], color='black')
axs[0,1].plot(X_s_[5][:,0], Y_s2_[5], color='blue')
axs[0,1].plot(X_s_[11][:,0], Y_s2_[11], color='red')
axs[0,1].plot(X_s_[22][:,0], Y_s2_[22], color='pink')
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
axs[1,2].plot(X_s_[22][:,0], Y_s1_[22], color='pink')
axs[1,2].plot(X_s_[49][:,0], Y_s1_[49], color='green')
axs[0,2].plot(X_s_[0][:,0], Y_s2_[0], color='black')
axs[0,2].plot(X_s_[5][:,0], Y_s2_[5], color='blue')
axs[0,2].plot(X_s_[11][:,0], Y_s2_[11], color='red')
axs[0,2].plot(X_s_[22][:,0], Y_s2_[22], color='pink')
axs[0,2].plot(X_s_[49][:,0], Y_s2_[49], color='green')

from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0], [0], marker='o', color='black', label='0'),
    Line2D([0], [0], marker='>', color='blue', label='0.096'),
    Line2D([0], [0], marker='*', color='red', label='0.230'),
    Line2D([0], [0], marker='D', color='purple', label='0.443'),
    Line2D([0], [0], marker='^', color='green', label='1'),
]

axs[0,0].legend(handles=legend_elements, ncol=3, fontsize=7, handletextpad=0.4, labelspacing=0.3, handlelength=1.2) 

fig.suptitle(r'\textbf{Experimental Data (High-Fidelity)}', x=0.53, fontsize=10)
axs[0,0].set_title(r'$\bm{T = 313.16}$ \textbf{K}', fontsize=10)
axs[0,1].set_title(r'$\bm{T = 323.16}$ \textbf{K}', fontsize=10)
axs[0,2].set_title(r'$\bm{T = 333.16}$ \textbf{K}', fontsize=10)

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
axs[0,0].set_ylim(0,12)
axs[0,1].set_ylim(0,12)
axs[0,2].set_ylim(0,12)
axs[1,0].set_ylim(0,12)
axs[1,1].set_ylim(0,12)
axs[1,2].set_ylim(0,12)
plt.tight_layout()
plt.savefig('AL_Profiles_high.png', dpi=600) 


# Profiles low-fidelity

fig, axs = plt.subplots(2, 3, figsize=(6.5, 4.0), dpi=300)

temp1 = numpy.unique(xlow[:, 1])
X_s = []
Y_s1 = []
Y_s2 = []
for temp in temp1:
    indices = numpy.where(xlow[:, 1] == temp)[0]
    X_sub = xlow[indices]
    Y_sub1 = ylow1[indices] 
    Y_sub2 = ylow2[indices] 
    X_s.append(X_sub)
    Y_s1.append(Y_sub1)
    Y_s2.append(Y_sub2)
axs[1,0].plot(X_s[0][:,0], Y_s1[0], 'o', color='black', markersize=10, markerfacecolor='none')
axs[0,0].plot(X_s[0][:,0], Y_s2[0], 'o', color='black', markersize=10, markerfacecolor='none')
axs[1,1].plot(X_s[1][:,0], Y_s1[1], 'o', color='black', markersize=10, markerfacecolor='none')
axs[0,1].plot(X_s[1][:,0], Y_s2[1], 'o', color='black', markersize=10, markerfacecolor='none')
axs[1,2].plot(X_s[2][:,0], Y_s1[2], 'o', color='black', markersize=10, markerfacecolor='none')
axs[0,2].plot(X_s[2][:,0], Y_s2[2], 'o', color='black', markersize=10, markerfacecolor='none')


axs[0,0].text(X_s[0][1,0], Y_s2[0][1]+1, f"$\mathbf{str(1)}$", color='black', fontsize=10, ha='center', va='center') 
axs[0,1].text(X_s[1][0,0], Y_s2[1][0]+1, f"$\mathbf{str(2)}$", color='black', fontsize=10, ha='center', va='center') 
axs[0,2].text(X_s[2][1,0], Y_s2[2][1]+1, f"$\mathbf{str(3)}$", color='black', fontsize=10, ha='center', va='center') 
axs[0,0].text(X_s[0][2,0], Y_s2[0][2]+1, f"$\mathbf{str(4)}$", color='black', fontsize=10, ha='center', va='center') 
axs[0,2].text(X_s[2][2,0], Y_s2[2][2]+1, f"$\mathbf{str(7)}$", color='black', fontsize=10, ha='center', va='center') 
axs[0,0].text(X_s[0][3,0], Y_s2[0][3]+1, f"$\mathbf{str(9)}$", color='black', fontsize=10, ha='center', va='center') 
axs[0,2].text(X_s[2][3,0], Y_s2[2][3]+1, f"$\mathbf{str(10)}$", color='black', fontsize=10, ha='center', va='center') 
axs[0,0].text(X_s[0][4,0], Y_s2[0][4]+1, f"$\mathbf{str(11)}$", color='black', fontsize=10, ha='center', va='center') 
axs[0,2].text(X_s[2][4,0]+30, Y_s2[2][4]+1, f"$\mathbf{str(12)}$", color='black', fontsize=10, ha='center', va='center') 
axs[0,2].text(X_s[2][5,0]+30, Y_s2[2][5]+0, f"$\mathbf{str(14)}$", color='black', fontsize=10, ha='center', va='center')
axs[0,0].text(X_s[0][5,0], Y_s2[0][5]+1, f"$\mathbf{str(17)}$", color='black', fontsize=10, ha='center', va='center')  

axs[1,0].text(X_s[0][1,0], Y_s1[0][1]+1, f"$\mathbf{str(1)}$", color='black', fontsize=10, ha='center', va='center') 
axs[1,1].text(X_s[1][0,0], Y_s1[1][0]+1, f"$\mathbf{str(2)}$", color='black', fontsize=10, ha='center', va='center') 
axs[1,2].text(X_s[2][1,0], Y_s1[2][1]+1, f"$\mathbf{str(3)}$", color='black', fontsize=10, ha='center', va='center') 
axs[1,0].text(X_s[0][2,0], Y_s1[0][2]+1, f"$\mathbf{str(4)}$", color='black', fontsize=10, ha='center', va='center') 
axs[1,2].text(X_s[2][2,0], Y_s1[2][2]+1, f"$\mathbf{str(7)}$", color='black', fontsize=10, ha='center', va='center') 
axs[1,0].text(X_s[0][3,0], Y_s1[0][3]+1, f"$\mathbf{str(9)}$", color='black', fontsize=10, ha='center', va='center') 
axs[1,2].text(X_s[2][3,0], Y_s1[2][3]+1, f"$\mathbf{str(10)}$", color='black', fontsize=10, ha='center', va='center') 
axs[1,0].text(X_s[0][4,0], Y_s1[0][4]+1, f"$\mathbf{str(11)}$", color='black', fontsize=10, ha='center', va='center') 
axs[1,2].text(X_s[2][4,0], Y_s1[2][4]+1, f"$\mathbf{str(12)}$", color='black', fontsize=10, ha='center', va='center') 
axs[1,2].text(X_s[2][5,0]+30, Y_s1[2][5]+0, f"$\mathbf{str(14)}$", color='black', fontsize=10, ha='center', va='center')
axs[1,0].text(X_s[0][5,0], Y_s1[0][5]+1, f"$\mathbf{str(17)}$", color='black', fontsize=10, ha='center', va='center')  


temp1 = numpy.unique(X_low[:, 1])
comp1 = numpy.unique(X_low[:, 2])

X_s = []
Y_s1 = []
Y_s2 = []
errY_s1 = []
errY_s2 = []
for temp in temp1:
    indices = numpy.where(X_low[:, 1] == temp)[0]
    X_sub = X_low[indices]
    Y_sub1 = Y_low1[indices] 
    Y_sub2 = Y_low2[indices] 
    errY_sub1 = errY_low1[indices]
    errY_sub2 = errY_low2[indices]
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
axs[1,0].errorbar(X_s_[3][:,0], numpy.squeeze(Y_s1_[3]), numpy.squeeze(errY_s1_[3]), fmt='D', color='pink')
axs[1,0].errorbar(X_s_[4][:,0], numpy.squeeze(Y_s1_[4]), numpy.squeeze(errY_s1_[4]), fmt='<', color='green')
axs[0,0].errorbar(X_s_[0][:,0], numpy.squeeze(Y_s2_[0]), numpy.squeeze(errY_s2_[0]), fmt='o', color='black')
axs[0,0].errorbar(X_s_[1][:,0], numpy.squeeze(Y_s2_[1]), numpy.squeeze(errY_s2_[1]), fmt='>', color='blue')
axs[0,0].errorbar(X_s_[2][:,0], numpy.squeeze(Y_s2_[2]), numpy.squeeze(errY_s2_[2]), fmt='*', color='red')
axs[0,0].errorbar(X_s_[3][:,0], numpy.squeeze(Y_s2_[3]), numpy.squeeze(errY_s2_[3]), fmt='D', color='pink')
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
axs[1,1].errorbar(X_s_[3][:,0], numpy.squeeze(Y_s1_[3]), numpy.squeeze(errY_s1_[3]), fmt='D', color='pink')
axs[1,1].errorbar(X_s_[4][:,0], numpy.squeeze(Y_s1_[4]), numpy.squeeze(errY_s1_[4]), fmt='<', color='green')
axs[0,1].errorbar(X_s_[0][:,0], numpy.squeeze(Y_s2_[0]), numpy.squeeze(errY_s2_[0]), fmt='o', color='black')
axs[0,1].errorbar(X_s_[1][:,0], numpy.squeeze(Y_s2_[1]), numpy.squeeze(errY_s2_[1]), fmt='>', color='blue')
axs[0,1].errorbar(X_s_[2][:,0], numpy.squeeze(Y_s2_[2]), numpy.squeeze(errY_s2_[2]), fmt='*', color='red')
axs[0,1].errorbar(X_s_[3][:,0], numpy.squeeze(Y_s2_[3]), numpy.squeeze(errY_s2_[3]), fmt='D', color='pink')
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
axs[1,2].errorbar(X_s_[3][:,0], numpy.squeeze(Y_s1_[3]), numpy.squeeze(errY_s1_[3]), fmt='D', color='pink')
axs[1,2].errorbar(X_s_[4][:,0], numpy.squeeze(Y_s1_[4]), numpy.squeeze(errY_s1_[4]), fmt='<', color='green')
axs[0,2].errorbar(X_s_[0][:,0], numpy.squeeze(Y_s2_[0]), numpy.squeeze(errY_s2_[0]), fmt='o', color='black')
axs[0,2].errorbar(X_s_[1][:,0], numpy.squeeze(Y_s2_[1]), numpy.squeeze(errY_s2_[1]), fmt='>', color='blue')
axs[0,2].errorbar(X_s_[2][:,0], numpy.squeeze(Y_s2_[2]), numpy.squeeze(errY_s2_[2]), fmt='*', color='red')
axs[0,2].errorbar(X_s_[3][:,0], numpy.squeeze(Y_s2_[3]), numpy.squeeze(errY_s2_[3]), fmt='D', color='pink')
axs[0,2].errorbar(X_s_[4][:,0], numpy.squeeze(Y_s2_[4]), numpy.squeeze(errY_s2_[4]), fmt='<', color='green')


temp1 = numpy.unique(X_Test[:, 1])
comp1 = numpy.unique(X_Test[:, 2])

X_s = []
Y_s1 = []
Y_s2 = []
for temp in temp1:
    indices = numpy.where(X_Test[:, 1] == temp)[0]
    X_sub = X_Test[indices]
    Y_sub1 = Y_Pred_low1[indices] 
    Y_sub2 = Y_Pred_low2[indices] 
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
axs[1,0].plot(X_s_[22][:,0], Y_s1_[22], color='pink')
axs[1,0].plot(X_s_[49][:,0], Y_s1_[49], color='green')
axs[0,0].plot(X_s_[0][:,0], Y_s2_[0], color='black')
axs[0,0].plot(X_s_[5][:,0], Y_s2_[5], color='blue')
axs[0,0].plot(X_s_[11][:,0], Y_s2_[11], color='red')
axs[0,0].plot(X_s_[22][:,0], Y_s2_[22], color='pink')
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
axs[1,1].plot(X_s_[22][:,0], Y_s1_[22], color='pink')
axs[1,1].plot(X_s_[49][:,0], Y_s1_[49], color='green')
axs[0,1].plot(X_s_[0][:,0], Y_s2_[0], color='black')
axs[0,1].plot(X_s_[5][:,0], Y_s2_[5], color='blue')
axs[0,1].plot(X_s_[11][:,0], Y_s2_[11], color='red')
axs[0,1].plot(X_s_[22][:,0], Y_s2_[22], color='pink')
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
axs[1,2].plot(X_s_[22][:,0], Y_s1_[22], color='pink')
axs[1,2].plot(X_s_[49][:,0], Y_s1_[49], color='green')
axs[0,2].plot(X_s_[0][:,0], Y_s2_[0], color='black')
axs[0,2].plot(X_s_[5][:,0], Y_s2_[5], color='blue')
axs[0,2].plot(X_s_[11][:,0], Y_s2_[11], color='red')
axs[0,2].plot(X_s_[22][:,0], Y_s2_[22], color='pink')
axs[0,2].plot(X_s_[49][:,0], Y_s2_[49], color='green')

from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0], [0], marker='o', color='black', label='0'),
    Line2D([0], [0], marker='>', color='blue', label='0.096'),
    Line2D([0], [0], marker='*', color='red', label='0.230'),
    Line2D([0], [0], marker='D', color='purple', label='0.443'),
    Line2D([0], [0], marker='^', color='green', label='1'),
]

axs[0,0].legend(handles=legend_elements, ncol=3, fontsize=7, handletextpad=0.4, labelspacing=0.3, handlelength=1.2) 

fig.suptitle(r'\textbf{MD Data (Low-Fidelity)}', x=0.53, fontsize=10)
axs[0,0].set_title(r'$\bm{T = 313.16}$ \textbf{K}', fontsize=10)
axs[0,1].set_title(r'$\bm{T = 323.16}$ \textbf{K}', fontsize=10)
axs[0,2].set_title(r'$\bm{T = 333.16}$ \textbf{K}', fontsize=10)

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
axs[0,0].set_ylim(0,12)
axs[0,1].set_ylim(0,12)
axs[0,2].set_ylim(0,12)
axs[1,0].set_ylim(0,12)
axs[1,1].set_ylim(0,12)
axs[1,2].set_ylim(0,12)
plt.tight_layout()
plt.savefig('AL_Profiles_low.png', dpi=600) 
