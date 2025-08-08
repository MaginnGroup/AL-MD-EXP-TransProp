import numpy as np
import matplotlib.pyplot as plt
import figstyle


fig, axs = plt.subplots(2, 3, figsize=(7.2, 4), dpi=300, constrained_layout=True)

data = np.load("../system 1: CO2-nC7/single-fidelity-MD/data_Diffusivity_MD.npz")
X_s = data["X_s"]
Y_s = data["Y_s"]
errY_s = data["errY_s"]
axs[1,0].errorbar(X_s[0][:,0], np.squeeze(Y_s[0]), np.squeeze(errY_s[0]), fmt='o', color='black')
axs[1,0].errorbar(X_s[1][:,0], np.squeeze(Y_s[1]), np.squeeze(errY_s[1]), fmt='>', color='blue')
axs[1,0].errorbar(X_s[2][:,0], np.squeeze(Y_s[2]), np.squeeze(errY_s[2]), fmt='*', color='red')
axs[1,0].errorbar(X_s[3][:,0], np.squeeze(Y_s[3]), np.squeeze(errY_s[3]), fmt='D', color='purple')
axs[1,0].errorbar(X_s[4][:,0], np.squeeze(Y_s[4]), np.squeeze(errY_s[4]), fmt='^', color='green')
axs[1,0].errorbar(X_s[5][:,0], np.squeeze(Y_s[5]), np.squeeze(errY_s[5]), fmt='o', color='m')

data = np.load("../system 2: CO2-benzene/single-fidelity-MD/data_Diffusivity_MD.npz", allow_pickle=True)
X_s = data["X_s"]
Y_s = data["Y_s"]
errY_s = data["errY_s"]
axs[1,1].errorbar(X_s[0][:,1], np.squeeze(Y_s[0]), np.squeeze(errY_s[0]), fmt='o', color='black')
axs[1,1].errorbar(X_s[1][:,1], np.squeeze(Y_s[1]), np.squeeze(errY_s[1]), fmt='>', color='blue')
axs[1,1].errorbar(X_s[2][:,1], np.squeeze(Y_s[2]), np.squeeze(errY_s[2]), fmt='*', color='red')
axs[1,1].errorbar(X_s[3][:,1], np.squeeze(Y_s[3]), np.squeeze(errY_s[3]), fmt='D', color='purple')
axs[1,1].errorbar(X_s[4][:,1], np.squeeze(Y_s[4]), np.squeeze(errY_s[4]), fmt='^', color='green')
axs[1,1].errorbar(X_s[5][:,1], np.squeeze(Y_s[5]), np.squeeze(errY_s[5]), fmt='o', color='m')

data = np.load("../system 3: toluene-nC6/single-fidelity-MD/data_Diffusivity_MD.npz")
X_s = data["X_s"]
Y_s = data["Y_s"]
errY_s = data["errY_s"]
axs[1,2].errorbar(X_s[0][:,0], np.squeeze(Y_s[0]), np.squeeze(errY_s[0]), fmt='o', color='black')
axs[1,2].errorbar(X_s[1][:,0], np.squeeze(Y_s[1]), np.squeeze(errY_s[1]), fmt='>', color='blue')
axs[1,2].errorbar(X_s[2][:,0], np.squeeze(Y_s[2]), np.squeeze(errY_s[2]), fmt='*', color='red')
axs[1,2].errorbar(X_s[3][:,0], np.squeeze(Y_s[3]), np.squeeze(errY_s[3]), fmt='D', color='purple')
axs[1,2].errorbar(X_s[4][:,0], np.squeeze(Y_s[4]), np.squeeze(errY_s[4]), fmt='^', color='green')

data = np.load("../system 1: CO2-nC7/single-fidelity-MD/data_Diffusivity_MD_gp.npz")
X_s = data["X_s"]
Y_s = data["Y_s"]
axs[1,0].plot(X_s[0][:,0], Y_s[0], color='black')
axs[1,0].plot(X_s[1][:,0], Y_s[1], color='blue')
axs[1,0].plot(X_s[2][:,0], Y_s[2], color='red')
axs[1,0].plot(X_s[3][:,0], Y_s[3], color='purple')
axs[1,0].plot(X_s[4][:,0], Y_s[4], color='green')
axs[1,0].plot(X_s[5][:,0], Y_s[5], color='m')

data = np.load("../system 2: CO2-benzene/single-fidelity-MD/data_Diffusivity_MD_gp.npz", allow_pickle=True)
X_s = data["X_s"]
Y_s = data["Y_s"]
axs[1,1].plot(X_s[0][:,1], Y_s[0], color='black')
axs[1,1].plot(X_s[1][:,1], Y_s[1], color='blue')
axs[1,1].plot(X_s[2][:,1], Y_s[2], color='red')
axs[1,1].plot(X_s[3][:,1], Y_s[3], color='purple')
axs[1,1].plot(X_s[4][:,1], Y_s[4], color='green')
axs[1,1].plot(X_s[5][:,1], Y_s[5], color='m')

data = np.load("../system 3: toluene-nC6/single-fidelity-MD/data_Diffusivity_MD_gp.npz")
X_s = data["X_s"]
Y_s = data["Y_s"]
axs[1,2].plot(X_s[0][:,0], Y_s[0], color='black')
axs[1,2].plot(X_s[1][:,0], Y_s[1], color='blue')
axs[1,2].plot(X_s[2][:,0], Y_s[2], color='red')
axs[1,2].plot(X_s[3][:,0], Y_s[3], color='purple')
axs[1,2].plot(X_s[4][:,0], Y_s[4], color='green')


data = np.load("../system 1: CO2-nC7/single-fidelity-MD/data_Viscosity_MD.npz")
X_s = data["X_s"]
Y_s = data["Y_s"]
errY_s = data["errY_s"]
axs[0,0].errorbar(X_s[0][:,0], np.squeeze(Y_s[0]), np.squeeze(errY_s[0]), fmt='o', color='black')
axs[0,0].errorbar(X_s[1][:,0], np.squeeze(Y_s[1]), np.squeeze(errY_s[1]), fmt='>', color='blue')
axs[0,0].errorbar(X_s[2][:,0], np.squeeze(Y_s[2]), np.squeeze(errY_s[2]), fmt='*', color='red')
axs[0,0].errorbar(X_s[3][:,0], np.squeeze(Y_s[3]), np.squeeze(errY_s[3]), fmt='D', color='purple')
axs[0,0].errorbar(X_s[4][:,0], np.squeeze(Y_s[4]), np.squeeze(errY_s[4]), fmt='^', color='green')
axs[0,0].errorbar(X_s[5][:,0], np.squeeze(Y_s[5]), np.squeeze(errY_s[5]), fmt='o', color='m')


data = np.load("../system 2: CO2-benzene/single-fidelity-MD/data_Viscosity_MD.npz", allow_pickle=True)
X_s = data["X_s"]
Y_s = data["Y_s"]
errY_s = data["errY_s"]
axs[0,1].errorbar(X_s[0][:,1], np.squeeze(Y_s[0]), np.squeeze(errY_s[0]), fmt='o', color='black')
axs[0,1].errorbar(X_s[1][:,1], np.squeeze(Y_s[1]), np.squeeze(errY_s[1]), fmt='>', color='blue')
axs[0,1].errorbar(X_s[2][:,1], np.squeeze(Y_s[2]), np.squeeze(errY_s[2]), fmt='*', color='red')
axs[0,1].errorbar(X_s[3][:,1], np.squeeze(Y_s[3]), np.squeeze(errY_s[3]), fmt='D', color='purple')
axs[0,1].errorbar(X_s[4][:,1], np.squeeze(Y_s[4]), np.squeeze(errY_s[4]), fmt='^', color='green')
axs[0,1].errorbar(X_s[5][:,1], np.squeeze(Y_s[5]), np.squeeze(errY_s[5]), fmt='o', color='m')

data = np.load("../system 3: toluene-nC6/single-fidelity-MD/data_Viscosity_MD.npz")
X_s = data["X_s"]
Y_s = data["Y_s"]
errY_s = data["errY_s"]
axs[0,2].errorbar(X_s[0][:,0], np.squeeze(Y_s[0]), np.squeeze(errY_s[0]), fmt='o', color='black')
axs[0,2].errorbar(X_s[1][:,0], np.squeeze(Y_s[1]), np.squeeze(errY_s[1]), fmt='>', color='blue')
axs[0,2].errorbar(X_s[2][:,0], np.squeeze(Y_s[2]), np.squeeze(errY_s[2]), fmt='*', color='red')
axs[0,2].errorbar(X_s[3][:,0], np.squeeze(Y_s[3]), np.squeeze(errY_s[3]), fmt='D', color='purple')
axs[0,2].errorbar(X_s[4][:,0], np.squeeze(Y_s[4]), np.squeeze(errY_s[4]), fmt='^', color='green')

data = np.load("../system 1: CO2-nC7/single-fidelity-MD/data_Viscosity_MD_gp.npz")
X_s = data["X_s"]
Y_s = data["Y_s"]
axs[0,0].plot(X_s[0][:,0], Y_s[0], color='black')
axs[0,0].plot(X_s[1][:,0], Y_s[1], color='blue')
axs[0,0].plot(X_s[2][:,0], Y_s[2], color='red')
axs[0,0].plot(X_s[3][:,0], Y_s[3], color='purple')
axs[0,0].plot(X_s[4][:,0], Y_s[4], color='green')
axs[0,0].plot(X_s[5][:,0], Y_s[5], color='m')

data = np.load("../system 2: CO2-benzene/single-fidelity-MD/data_Viscosity_MD_gp.npz", allow_pickle=True)
X_s = data["X_s"]
Y_s = data["Y_s"]
axs[0,1].plot(X_s[0][:,1], Y_s[0], color='black')
axs[0,1].plot(X_s[1][:,1], Y_s[1], color='blue')
axs[0,1].plot(X_s[2][:,1], Y_s[2], color='red')
axs[0,1].plot(X_s[3][:,1], Y_s[3], color='purple')
axs[0,1].plot(X_s[4][:,1], Y_s[4], color='green')
axs[0,1].plot(X_s[5][:,1], Y_s[5], color='m')

data = np.load("../system 3: toluene-nC6/single-fidelity-MD/data_Viscosity_MD_gp.npz")
X_s = data["X_s"]
Y_s = data["Y_s"]
axs[0,2].plot(X_s[0][:,0], Y_s[0], color='black')
axs[0,2].plot(X_s[1][:,0], Y_s[1], color='blue')
axs[0,2].plot(X_s[2][:,0], Y_s[2], color='red')
axs[0,2].plot(X_s[3][:,0], Y_s[3], color='purple')
axs[0,2].plot(X_s[4][:,0], Y_s[4], color='green')


from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0], [0], marker='o', color='black', label='298 K'),
    Line2D([0], [0], marker='>', color='blue', label='323 K'),
    Line2D([0], [0], marker='*', color='red', label='348 K'),
    Line2D([0], [0], marker='D', color='purple', label='373 K'),
    Line2D([0], [0], marker='^', color='green', label='398 K'),
    Line2D([0], [0], marker='o', color='m', label='423 K')
]
axs[0,0].legend(handles=legend_elements, ncol=1, fontsize=8, handletextpad=0.3, labelspacing=0.2, handlelength=1.0) 

legend_elements = [
    Line2D([0], [0], marker='o', color='black', label='100 bar'),
    Line2D([0], [0], marker='>', color='blue', label='110 bar'),
    Line2D([0], [0], marker='*', color='red', label='130 bar'),
    Line2D([0], [0], marker='D', color='purple', label='150 bar'),
    Line2D([0], [0], marker='^', color='green', label='200 bar'),
    Line2D([0], [0], marker='o', color='m', label='300 bar')
]
axs[0,1].legend(handles=legend_elements, ncol=1, fontsize=8, handletextpad=0.3, labelspacing=0.2, handlelength=1.0) 

legend_elements = [
    Line2D([0], [0], marker='o', color='black', label='278 K'),
    Line2D([0], [0], marker='>', color='blue', label='288 K'),
    Line2D([0], [0], marker='*', color='red', label='298 K'),
    Line2D([0], [0], marker='D', color='purple', label='308 K'),
    Line2D([0], [0], marker='^', color='green', label='318 K')
]
axs[0,2].legend(handles=legend_elements, ncol=1, fontsize=8, handletextpad=0.3, labelspacing=0.2, handlelength=1.0) 

axs[1,0].set_ylim(3,21)
axs[0,0].set_ylim(1,9)
axs[1,1].set_ylim(5,80)
axs[0,1].set_ylim(1,15)
axs[1,2].set_ylim(1.5,6.5)
axs[0,2].set_ylim(1.5,6.5)

axs[0,0].set_title(r'\textbf{CO$_{\bm{2}}$/n-heptane}', fontsize=10)
axs[0,1].set_title(r'\textbf{CO$_{\bm{2}}$/benzene}', fontsize=10)
axs[0,2].set_title(r'\textbf{toluene/n-hexane}', fontsize=10)

axs[1,0].set_ylabel(r'$D \times 10^{-9}$ m²/s')
axs[1,1].set_ylabel(r'$D \times 10^{-9}$ m²/s')
axs[1,2].set_ylabel(r'$D \times 10^{-9}$ m²/s')
axs[0,0].set_ylabel(r'$\eta \times 10^{-4}$ Pa s')
axs[0,1].set_ylabel(r'$\eta \times 10^{-5}$ Pa s')
axs[0,2].set_ylabel(r'$\eta \times 10^{-4}$ Pa s')
axs[1,0].set_xlabel(r'$P$ (bar)')
axs[1,1].set_xlabel(r'$T$ (K)')
axs[1,2].set_xlabel(r'$x_1$')
axs[0,0].set_xticklabels([])
axs[0,1].set_xticklabels([])
axs[0,2].set_xticklabels([])
axs[0,0].set_xticks([0, 250, 500, 750])
axs[0,1].set_xticks([300,325,350,375])
axs[0,2].set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1])
axs[1,0].set_xticks([0, 250, 500, 750])
axs[1,1].set_xticks([300,325,350,375])
axs[1,2].set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1])

axs[1,1].set_yticks([15, 35, 55, 75])
axs[0,1].set_yticks([2, 6, 10, 14])
axs[0,2].set_yticks([2, 3, 4, 5, 6])
axs[1,2].set_yticks([2, 3, 4, 5, 6])
fig.align_ylabels(axs[:])
#fig.subplots_adjust(wspace=0.35)
#plt.tight_layout()
plt.savefig('plot_systems_MD.png', dpi=600) 



