
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
code='TOL_NC6_EXP'
# Define normalization methods
featureNorm='MinMax' # Standardization,LogStand,MinMax
labelNorm='LogStand' # Standardization,LogStand,MinMax

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
# Training
# =============================================================================

# Defining the input/output vectors
X=numpy.array([trainDB['x1'].values,trainDB['T/K'].values]).T
Y=numpy.array([trainDB['Visc/Pas'].values]).T *1e4
errY = 0.02 * Y
X_Train = X.copy()
Y_Train = Y.copy()

# Normalize
X_Train_N,skScaler_X=normalize(X_Train,method=featureNorm)
Y_Train_N,skScaler_Y=normalize(Y_Train,method=labelNorm)
errY_Train_N = errY / Y / skScaler_Y.scale_

# Torch tensors
train_X = torch.tensor(X_Train_N, dtype=torch.float64)
train_Y = torch.tensor(Y_Train_N, dtype=torch.float64)
train_errY = torch.tensor(errY_Train_N, dtype=torch.float64)
train_Yvar = (train_errY**2).flatten().to(train_Y.dtype)

# Build GP model
likelihood = FixedNoiseGaussianLikelihood(noise=train_Yvar, learn_additional_noise=False)
model = SingleTaskGP(train_X, train_Y, likelihood=likelihood, covar_module = RBFKernel())
mll = ExactMarginalLogLikelihood(model.likelihood, model)
fit_gpytorch_mll(mll)

# Predictions
model.eval()
with torch.no_grad():
    pred = model.posterior(train_X)
    mean = pred.mean.numpy()
    var = pred.variance.numpy()
    std = numpy.sqrt(var)

# Unnormalize prediction
Y_pred, _ = normalize(mean, method=labelNorm, skScaler=skScaler_Y, reverse=True)

# Compute MRE
mre = 100 * numpy.abs(Y_pred - Y) / Y
print(f"Mean Relative Error (MRE): {mre.mean():.2f}%")

# =============================================================================
# Generate Model Data
# =============================================================================

x1 = numpy.arange(0.05, 0.96, 0.01)
temp = numpy.arange(278, 319, 10)
X_new = numpy.array([[f, T] for T in temp for f in x1])
X_new_N,skScaler_X=normalize(X_new,method=featureNorm)
X_new_tensor=torch.tensor(X_new_N, dtype=torch.float64)

with torch.no_grad():
    pred = model.posterior(X_new_tensor)
    mean = pred.mean.numpy()
    std = numpy.sqrt(pred.variance.numpy())

Y_new, _ = normalize(mean, method=labelNorm, skScaler=skScaler_Y, reverse=True)

# LogStand variance transform
#mu_log = mean * skScaler_Y.scale_ + skScaler_Y.mean_
#sigma_log = std * skScaler_Y.scale_
#STD_new = numpy.sqrt(   (numpy.exp(sigma_log**2) - 1) * numpy.exp(2 * mu_log + sigma_log**2) )
STD_new = std * skScaler_Y.scale_ * Y_new

AF = 100 * STD_new / Y_new
print(f"AF (max): {AF.max():.2f}%")

# =============================================================================
# Plots
# =============================================================================

fig = plt.figure(figsize=(3.0, 2.5), dpi=300)

temp1 = numpy.unique(X[:, 1])
X_s = []
Y_s = []
errY_s = []
for temp in temp1:
    indices = numpy.where(X[:, 1] == temp)[0]
    X_sub = X[indices]
    Y_sub = Y[indices]  
    errY_sub = errY[indices]
    X_s.append(X_sub)
    Y_s.append(Y_sub)
    errY_s.append(errY_sub)
plt.errorbar(X_s[0][:,0], numpy.squeeze(Y_s[0]), numpy.squeeze(errY_s[0]), fmt='o', color='black')
plt.errorbar(X_s[1][:,0], numpy.squeeze(Y_s[1]), numpy.squeeze(errY_s[1]), fmt='>', color='blue')
plt.errorbar(X_s[2][:,0], numpy.squeeze(Y_s[2]), numpy.squeeze(errY_s[2]), fmt='*', color='red')
plt.errorbar(X_s[3][:,0], numpy.squeeze(Y_s[3]), numpy.squeeze(errY_s[3]), fmt='D', color='pink')
plt.errorbar(X_s[4][:,0], numpy.squeeze(Y_s[4]), numpy.squeeze(errY_s[4]), fmt='<', color='green')

numpy.savez("data_Viscosity_EXP.npz", X_s=X_s, Y_s=Y_s, errY_s=errY_s)

temp1 = numpy.unique(X_new[:, 1])
X_s = []
Y_s = []
STD_s = []
for temp in temp1:
    indices = numpy.where(X_new[:, 1] == temp)[0]
    X_sub = X_new[indices]
    Y_sub = Y_new[indices]   
    STD_sub = STD_new[indices] 
    X_s.append(X_sub)
    Y_s.append(Y_sub)
    STD_s.append(STD_sub)
plt.plot(X_s[0][:,0], Y_s[0], color='black')
plt.plot(X_s[1][:,0], Y_s[1], color='blue')
plt.plot(X_s[2][:,0], Y_s[2], color='red')
plt.plot(X_s[3][:,0], Y_s[3], color='pink')
plt.plot(X_s[4][:,0], Y_s[4], color='green')
plt.fill_between(X_s[0][:,0].flatten(), (Y_s[0] - 2*STD_s[0]).flatten(), (Y_s[0] + 2*STD_s[0]).flatten(), color='black', alpha=0.2)
plt.fill_between(X_s[1][:,0].flatten(), (Y_s[1] - 2*STD_s[1]).flatten(), (Y_s[1] + 2*STD_s[1]).flatten(), color='blue', alpha=0.2)
plt.fill_between(X_s[2][:,0].flatten(), (Y_s[2] - 2*STD_s[2]).flatten(), (Y_s[2] + 2*STD_s[2]).flatten(), color='red', alpha=0.2)
plt.fill_between(X_s[3][:,0].flatten(), (Y_s[3] - 2*STD_s[3]).flatten(), (Y_s[3] + 2*STD_s[3]).flatten(), color='pink', alpha=0.2)
plt.fill_between(X_s[4][:,0].flatten(), (Y_s[4] - 2*STD_s[4]).flatten(), (Y_s[4] + 2*STD_s[4]).flatten(), color='green', alpha=0.2)

numpy.savez("data_Viscosity_EXP_gp.npz", X_s=X_s, Y_s=Y_s, STD_s=STD_s)

plt.ylabel(r'$\eta \times 10^{-4}$ Pa s')
plt.xlabel(r'$x_{tol}$')
plt.xticks([0, 0.2, 0.4, 0.6, 0.8, 1])
plt.ylim(2,8)
plt.tight_layout()
#plt.savefig('Training_Viscosity.png', dpi=600) 


