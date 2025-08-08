import numpy as np
import matplotlib.pyplot as plt
import figstyle


fig, axs = plt.subplots(2, 2, figsize=(5, 4), dpi=300)

data = np.load("../single-fidelity-EXP/data_Diffusivity_EXP.npz")
X_s = data["X_s"]
Y_s = data["Y_s"]
errY_s = data["errY_s"]
axs[0,0].errorbar(X_s[0][:,0], np.squeeze(Y_s[0]), np.squeeze(errY_s[0]), fmt='o', color='black', label='298 K')
axs[0,0].errorbar(X_s[1][:,0], np.squeeze(Y_s[1]), np.squeeze(errY_s[1]), fmt='>', color='blue', label='323 K')
axs[0,0].errorbar(X_s[2][:,0], np.squeeze(Y_s[2]), np.squeeze(errY_s[2]), fmt='*', color='red', label='348 K')
axs[0,0].errorbar(X_s[3][:,0], np.squeeze(Y_s[3]), np.squeeze(errY_s[3]), fmt='D', color='purple', label='373 K')
axs[0,0].errorbar(X_s[4][:,0], np.squeeze(Y_s[4]), np.squeeze(errY_s[4]), fmt='^', color='green', label='398 K')
axs[0,0].errorbar(X_s[5][:,0], np.squeeze(Y_s[5]), np.squeeze(errY_s[5]), fmt='o', color='m', label='423 K')

data = np.load("../single-fidelity-EXP/data_Diffusivity_EXP_gp.npz")
X_s = data["X_s"]
Y_s = data["Y_s"]
STD_s = data["STD_s"]
axs[0,0].plot(X_s[0][:,0], Y_s[0], color='black')
axs[0,0].plot(X_s[1][:,0], Y_s[1], color='blue')
axs[0,0].plot(X_s[2][:,0], Y_s[2], color='red')
axs[0,0].plot(X_s[3][:,0], Y_s[3], color='purple')
axs[0,0].plot(X_s[4][:,0], Y_s[4], color='green')
axs[0,0].plot(X_s[5][:,0], Y_s[5], color='m')

Y_exp_D = data["Y_s"]


data = np.load("../single-fidelity-EXP/data_Viscosity_EXP.npz")
X_s = data["X_s"]
Y_s = data["Y_s"]
errY_s = data["errY_s"]
axs[1,0].errorbar(X_s[0][:,0], np.squeeze(Y_s[0]), np.squeeze(errY_s[0]), fmt='o', color='black', label='298 K')
axs[1,0].errorbar(X_s[1][:,0], np.squeeze(Y_s[1]), np.squeeze(errY_s[1]), fmt='>', color='blue', label='323 K')
axs[1,0].errorbar(X_s[2][:,0], np.squeeze(Y_s[2]), np.squeeze(errY_s[2]), fmt='*', color='red', label='348 K')
axs[1,0].errorbar(X_s[3][:,0], np.squeeze(Y_s[3]), np.squeeze(errY_s[3]), fmt='D', color='purple', label='373 K')
axs[1,0].errorbar(X_s[4][:,0], np.squeeze(Y_s[4]), np.squeeze(errY_s[4]), fmt='^', color='green', label='398 K')
axs[1,0].errorbar(X_s[5][:,0], np.squeeze(Y_s[5]), np.squeeze(errY_s[5]), fmt='o', color='m', label='423 K')


data = np.load("../single-fidelity-EXP/data_Viscosity_EXP_gp.npz")
X_s = data["X_s"]
Y_s = data["Y_s"]
STD_s = data["STD_s"]
axs[1,0].plot(X_s[0][:,0], Y_s[0], color='black')
axs[1,0].plot(X_s[1][:,0], Y_s[1], color='blue')
axs[1,0].plot(X_s[2][:,0], Y_s[2], color='red')
axs[1,0].plot(X_s[3][:,0], Y_s[3], color='purple')
axs[1,0].plot(X_s[4][:,0], Y_s[4], color='green')
axs[1,0].plot(X_s[5][:,0], Y_s[5], color='m')

Y_exp_V = data["Y_s"]


data = np.load("../single-fidelity-MD/data_Diffusivity_MD.npz")
X_s = data["X_s"]
Y_s = data["Y_s"]
errY_s = data["errY_s"]
axs[0,1].errorbar(X_s[0][:,0], np.squeeze(Y_s[0]), np.squeeze(errY_s[0]), fmt='o', color='black', label='298 K')
axs[0,1].errorbar(X_s[1][:,0], np.squeeze(Y_s[1]), np.squeeze(errY_s[1]), fmt='>', color='blue', label='323 K')
axs[0,1].errorbar(X_s[2][:,0], np.squeeze(Y_s[2]), np.squeeze(errY_s[2]), fmt='*', color='red', label='348 K')
axs[0,1].errorbar(X_s[3][:,0], np.squeeze(Y_s[3]), np.squeeze(errY_s[3]), fmt='D', color='purple', label='373 K')
axs[0,1].errorbar(X_s[4][:,0], np.squeeze(Y_s[4]), np.squeeze(errY_s[4]), fmt='^', color='green', label='398 K')
axs[0,1].errorbar(X_s[5][:,0], np.squeeze(Y_s[5]), np.squeeze(errY_s[5]), fmt='o', color='m', label='423 K')


data = np.load("../single-fidelity-MD/data_Diffusivity_MD_gp.npz")
X_s = data["X_s"]
Y_s = data["Y_s"]
STD_s = data["STD_s"]
axs[0,1].plot(X_s[0][:,0], Y_s[0], color='black')
axs[0,1].plot(X_s[1][:,0], Y_s[1], color='blue')
axs[0,1].plot(X_s[2][:,0], Y_s[2], color='red')
axs[0,1].plot(X_s[3][:,0], Y_s[3], color='purple')
axs[0,1].plot(X_s[4][:,0], Y_s[4], color='green')
axs[0,1].plot(X_s[5][:,0], Y_s[5], color='m')

Y_sim_D = data["Y_s"]


data = np.load("../single-fidelity-MD/data_Viscosity_MD.npz")
X_s = data["X_s"]
Y_s = data["Y_s"]
errY_s = data["errY_s"]
axs[1,1].errorbar(X_s[0][:,0], np.squeeze(Y_s[0]), np.squeeze(errY_s[0]), fmt='o', color='black', label='298 K')
axs[1,1].errorbar(X_s[1][:,0], np.squeeze(Y_s[1]), np.squeeze(errY_s[1]), fmt='>', color='blue', label='323 K')
axs[1,1].errorbar(X_s[2][:,0], np.squeeze(Y_s[2]), np.squeeze(errY_s[2]), fmt='*', color='red', label='348 K')
axs[1,1].errorbar(X_s[3][:,0], np.squeeze(Y_s[3]), np.squeeze(errY_s[3]), fmt='D', color='purple', label='373 K')
axs[1,1].errorbar(X_s[4][:,0], np.squeeze(Y_s[4]), np.squeeze(errY_s[4]), fmt='^', color='green', label='398 K')
axs[1,1].errorbar(X_s[5][:,0], np.squeeze(Y_s[5]), np.squeeze(errY_s[5]), fmt='o', color='m', label='423 K')


data = np.load("../single-fidelity-MD/data_Viscosity_MD_gp.npz")
X_s = data["X_s"]
Y_s = data["Y_s"]
STD_s = data["STD_s"]
axs[1,1].plot(X_s[0][:,0], Y_s[0], color='black')
axs[1,1].plot(X_s[1][:,0], Y_s[1], color='blue')
axs[1,1].plot(X_s[2][:,0], Y_s[2], color='red')
axs[1,1].plot(X_s[3][:,0], Y_s[3], color='purple')
axs[1,1].plot(X_s[4][:,0], Y_s[4], color='green')
axs[1,1].plot(X_s[5][:,0], Y_s[5], color='m')

Y_sim_V = data["Y_s"]

from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0], [0], marker='o', color='black', label='298 K'),
    Line2D([0], [0], marker='>', color='blue', label='323 K'),
    Line2D([0], [0], marker='*', color='red', label='348 K'),
    Line2D([0], [0], marker='D', color='purple', label='373 K'),
    Line2D([0], [0], marker='^', color='green', label='398 K'),
    Line2D([0], [0], marker='o', color='m', label='423 K')
]

axs[0,0].legend(handles=legend_elements, ncol=3, fontsize=7, handletextpad=0.4, labelspacing=0.3, handlelength=1.2) 

axs[0,0].set_title("Experimental Data")
axs[0,1].set_title("Simulated Data")
axs[0,0].set_ylabel(r'$D \times 10^{-9}$ m²/s')
axs[1,0].set_ylabel(r'$\eta \times 10^{-4}$ Pa s')
axs[1,0].set_xlabel(r'$P$ (bar)')
axs[1,1].set_xlabel(r'$P$ (bar)')
axs[0,0].set_xticks([0, 150, 300, 450, 600, 750])
axs[0,1].set_xticks([0, 150, 300, 450, 600, 750])
axs[1,0].set_xticks([0, 150, 300, 450, 600, 750])
axs[1,1].set_xticks([0, 150, 300, 450, 600, 750])
axs[0,1].set_yticklabels([])
axs[0,0].set_xticklabels([])
axs[0,1].set_xticklabels([])
axs[1,1].set_yticklabels([])
axs[0,0].set_ylim(3,21)
axs[0,1].set_ylim(3,21)
axs[1,0].set_ylim(1,8)
axs[1,1].set_ylim(1,8)
plt.tight_layout()
plt.savefig('figure1.png', dpi=600) 


MRE1 = (100*np.abs(Y_sim_D-Y_exp_D)/Y_exp_D)
MRE2 = (100*np.abs(Y_sim_V-Y_exp_V)/Y_exp_V)
print(f"MRE Diffusivity: min {MRE1.min():.2f}%, max {MRE1.max():.2f}%, mean {MRE1.mean():.2f}%")
print(f"MRE Viscosity: min {MRE2.min():.2f}%, max {MRE2.max():.2f}%, mean {MRE2.mean():.2f}%")




