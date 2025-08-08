import numpy as np
import matplotlib.pyplot as plt
import figstyle


Y_sim_D = []
Y_sim_V = []
Y_exp_D = []
Y_exp_V = []

fig, axs = plt.subplots(2, 3, figsize=(7.0, 4.1), dpi=300, constrained_layout=True)

data = np.load("../single-fidelity-MD/data_Diffusivity_T1_MD.npz")
X_s = data["X_s1"]
Y_s = data["Y_s1"]
errY_s = data["errY_s1"]
axs[1,0].errorbar(X_s[0][:,0], np.squeeze(Y_s[0]), np.squeeze(errY_s[0]), fmt='o', color='black')
axs[1,0].errorbar(X_s[1][:,0], np.squeeze(Y_s[1]), np.squeeze(errY_s[1]), fmt='>', color='blue')
axs[1,0].errorbar(X_s[2][:,0], np.squeeze(Y_s[2]), np.squeeze(errY_s[2]), fmt='*', color='red')
axs[1,0].errorbar(X_s[3][:,0], np.squeeze(Y_s[3]), np.squeeze(errY_s[3]), fmt='D', color='purple')
axs[1,0].errorbar(X_s[4][:,0], np.squeeze(Y_s[4]), np.squeeze(errY_s[4]), fmt='^', color='green')

Y_sim_D.append(data["Y_s1"])

data = np.load("../single-fidelity-MD/data_Diffusivity_T2_MD.npz")
X_s = data["X_s1"]
Y_s = data["Y_s1"]
errY_s = data["errY_s1"]
axs[1,1].errorbar(X_s[0][:,0], np.squeeze(Y_s[0]), np.squeeze(errY_s[0]), fmt='o', color='black')
axs[1,1].errorbar(X_s[1][:,0], np.squeeze(Y_s[1]), np.squeeze(errY_s[1]), fmt='>', color='blue')
axs[1,1].errorbar(X_s[2][:,0], np.squeeze(Y_s[2]), np.squeeze(errY_s[2]), fmt='*', color='red')
axs[1,1].errorbar(X_s[3][:,0], np.squeeze(Y_s[3]), np.squeeze(errY_s[3]), fmt='D', color='purple')
axs[1,1].errorbar(X_s[4][:,0], np.squeeze(Y_s[4]), np.squeeze(errY_s[4]), fmt='^', color='green')

Y_sim_D.append(data["Y_s1"])

data = np.load("../single-fidelity-MD/data_Diffusivity_T3_MD.npz")
X_s = data["X_s1"]
Y_s = data["Y_s1"]
errY_s = data["errY_s1"]
axs[1,2].errorbar(X_s[0][:,0], np.squeeze(Y_s[0]), np.squeeze(errY_s[0]), fmt='o', color='black')
axs[1,2].errorbar(X_s[1][:,0], np.squeeze(Y_s[1]), np.squeeze(errY_s[1]), fmt='>', color='blue')
axs[1,2].errorbar(X_s[2][:,0], np.squeeze(Y_s[2]), np.squeeze(errY_s[2]), fmt='*', color='red')
axs[1,2].errorbar(X_s[3][:,0], np.squeeze(Y_s[3]), np.squeeze(errY_s[3]), fmt='D', color='purple')
axs[1,2].errorbar(X_s[4][:,0], np.squeeze(Y_s[4]), np.squeeze(errY_s[4]), fmt='^', color='green')

Y_sim_D.append(data["Y_s1"])

data = np.load("../single-fidelity-MD/data_Diffusivity_T1_MD_gp.npz")
X_s = data["X_s1"]
Y_s = data["Y_s1"]
axs[1,0].plot(X_s[0][:,0], Y_s[0], color='black')
axs[1,0].plot(X_s[1][:,0], Y_s[1], color='blue')
axs[1,0].plot(X_s[2][:,0], Y_s[2], color='red')
axs[1,0].plot(X_s[3][:,0], Y_s[3], color='purple')
axs[1,0].plot(X_s[4][:,0], Y_s[4], color='green')

data = np.load("../single-fidelity-MD/data_Diffusivity_T2_MD_gp.npz")
X_s = data["X_s1"]
Y_s = data["Y_s1"]
axs[1,1].plot(X_s[0][:,0], Y_s[0], color='black')
axs[1,1].plot(X_s[1][:,0], Y_s[1], color='blue')
axs[1,1].plot(X_s[2][:,0], Y_s[2], color='red')
axs[1,1].plot(X_s[3][:,0], Y_s[3], color='purple')
axs[1,1].plot(X_s[4][:,0], Y_s[4], color='green')

data = np.load("../single-fidelity-MD/data_Diffusivity_T3_MD_gp.npz")
X_s = data["X_s1"]
Y_s = data["Y_s1"]
axs[1,2].plot(X_s[0][:,0], Y_s[0], color='black')
axs[1,2].plot(X_s[1][:,0], Y_s[1], color='blue')
axs[1,2].plot(X_s[2][:,0], Y_s[2], color='red')
axs[1,2].plot(X_s[3][:,0], Y_s[3], color='purple')
axs[1,2].plot(X_s[4][:,0], Y_s[4], color='green')


data = np.load("../single-fidelity-MD/data_Viscosity_T1_MD.npz")
X_s = data["X_s1"]
Y_s = data["Y_s1"]
errY_s = data["errY_s1"]
axs[0,0].errorbar(X_s[0][:,0], np.squeeze(Y_s[0]), np.squeeze(errY_s[0]), fmt='o', color='black')
axs[0,0].errorbar(X_s[1][:,0], np.squeeze(Y_s[1]), np.squeeze(errY_s[1]), fmt='>', color='blue')
axs[0,0].errorbar(X_s[2][:,0], np.squeeze(Y_s[2]), np.squeeze(errY_s[2]), fmt='*', color='red')
axs[0,0].errorbar(X_s[3][:,0], np.squeeze(Y_s[3]), np.squeeze(errY_s[3]), fmt='D', color='purple')
axs[0,0].errorbar(X_s[4][:,0], np.squeeze(Y_s[4]), np.squeeze(errY_s[4]), fmt='^', color='green')

Y_sim_V.append(data["Y_s1"])

data = np.load("../single-fidelity-MD/data_Viscosity_T2_MD.npz")
X_s = data["X_s1"]
Y_s = data["Y_s1"]
errY_s = data["errY_s1"]
axs[0,1].errorbar(X_s[0][:,0], np.squeeze(Y_s[0]), np.squeeze(errY_s[0]), fmt='o', color='black')
axs[0,1].errorbar(X_s[1][:,0], np.squeeze(Y_s[1]), np.squeeze(errY_s[1]), fmt='>', color='blue')
axs[0,1].errorbar(X_s[2][:,0], np.squeeze(Y_s[2]), np.squeeze(errY_s[2]), fmt='*', color='red')
axs[0,1].errorbar(X_s[3][:,0], np.squeeze(Y_s[3]), np.squeeze(errY_s[3]), fmt='D', color='purple')
axs[0,1].errorbar(X_s[4][:,0], np.squeeze(Y_s[4]), np.squeeze(errY_s[4]), fmt='^', color='green')

Y_sim_V.append(data["Y_s1"])

data = np.load("../single-fidelity-MD/data_Viscosity_T3_MD.npz")
X_s = data["X_s1"]
Y_s = data["Y_s1"]
errY_s = data["errY_s1"]
axs[0,2].errorbar(X_s[0][:,0], np.squeeze(Y_s[0]), np.squeeze(errY_s[0]), fmt='o', color='black')
axs[0,2].errorbar(X_s[1][:,0], np.squeeze(Y_s[1]), np.squeeze(errY_s[1]), fmt='>', color='blue')
axs[0,2].errorbar(X_s[2][:,0], np.squeeze(Y_s[2]), np.squeeze(errY_s[2]), fmt='*', color='red')
axs[0,2].errorbar(X_s[3][:,0], np.squeeze(Y_s[3]), np.squeeze(errY_s[3]), fmt='D', color='purple')
axs[0,2].errorbar(X_s[4][:,0], np.squeeze(Y_s[4]), np.squeeze(errY_s[4]), fmt='^', color='green')

Y_sim_V.append(data["Y_s1"])

data = np.load("../single-fidelity-MD/data_Viscosity_T1_MD_gp.npz")
X_s = data["X_s1"]
Y_s = data["Y_s1"]
axs[0,0].plot(X_s[0][:,0], Y_s[0], color='black')
axs[0,0].plot(X_s[1][:,0], Y_s[1], color='blue')
axs[0,0].plot(X_s[2][:,0], Y_s[2], color='red')
axs[0,0].plot(X_s[3][:,0], Y_s[3], color='purple')
axs[0,0].plot(X_s[4][:,0], Y_s[4], color='green')

data = np.load("../single-fidelity-MD/data_Viscosity_T2_MD_gp.npz")
X_s = data["X_s1"]
Y_s = data["Y_s1"]
axs[0,1].plot(X_s[0][:,0], Y_s[0], color='black')
axs[0,1].plot(X_s[1][:,0], Y_s[1], color='blue')
axs[0,1].plot(X_s[2][:,0], Y_s[2], color='red')
axs[0,1].plot(X_s[3][:,0], Y_s[3], color='purple')
axs[0,1].plot(X_s[4][:,0], Y_s[4], color='green')

data = np.load("../single-fidelity-MD/data_Viscosity_T3_MD_gp.npz")
X_s = data["X_s1"]
Y_s = data["Y_s1"]
axs[0,2].plot(X_s[0][:,0], Y_s[0], color='black')
axs[0,2].plot(X_s[1][:,0], Y_s[1], color='blue')
axs[0,2].plot(X_s[2][:,0], Y_s[2], color='red')
axs[0,2].plot(X_s[3][:,0], Y_s[3], color='purple')
axs[0,2].plot(X_s[4][:,0], Y_s[4], color='green')


from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0], [0], marker='o', color='black', label='0.000'),
    Line2D([0], [0], marker='>', color='blue', label='0.096'),
    Line2D([0], [0], marker='*', color='red', label='0.230'),
    Line2D([0], [0], marker='D', color='purple', label='0.443'),
    Line2D([0], [0], marker='^', color='green', label='1.000'),
]

axs[0,2].legend(handles=legend_elements, ncol=1, fontsize=8, handletextpad=0.3, labelspacing=0.2, handlelength=1.0) 


fig.suptitle(r'\textbf{CO$_{\bm{2}}$/ethanol/dibenzyl ether}', x=0.53, fontsize=10)

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
# axs[0,1].set_yticklabels([])
# axs[0,2].set_yticklabels([])
# axs[1,1].set_yticklabels([])
# axs[1,2].set_yticklabels([])
axs[0,0].set_xticks([150, 200, 250, 300, 350])
axs[0,1].set_xticks([150, 200, 250, 300, 350])
axs[0,2].set_xticks([150, 200, 250, 300, 350])
axs[1,0].set_xticks([150, 200, 250, 300, 350])
axs[1,1].set_xticks([150, 200, 250, 300, 350])
axs[1,2].set_xticks([150, 200, 250, 300, 350])
axs[1,0].set_yticks([1,5,9,13])
axs[1,1].set_yticks([1,5,9,13])
axs[1,2].set_yticks([1,5,9,13])
axs[0,0].set_yticks([2,5,8,11])
axs[0,1].set_yticks([2,5,8,11])
axs[0,2].set_yticks([2,5,8,11])
axs[1,0].set_ylim(0,14)
axs[1,1].set_ylim(0,14)
axs[1,2].set_ylim(0,14)
axs[0,0].set_ylim(0,12)
axs[0,1].set_ylim(0,12)
axs[0,2].set_ylim(0,12)

plt.savefig('plot_CO2_ethanol_dibenzylether_MD.png', dpi=600) 








fig, axs = plt.subplots(2, 3, figsize=(7.0, 4.1), dpi=300, constrained_layout=True)

data = np.load("../single-fidelity-EXP/data_Diffusivity_T1_EXP.npz")
X_s = data["X_s1"]
Y_s = data["Y_s1"]
errY_s = data["errY_s1"]
axs[1,0].errorbar(X_s[0][:,0], np.squeeze(Y_s[0]), np.squeeze(errY_s[0]), fmt='o', color='black')
axs[1,0].errorbar(X_s[1][:,0], np.squeeze(Y_s[1]), np.squeeze(errY_s[1]), fmt='>', color='blue')
axs[1,0].errorbar(X_s[2][:,0], np.squeeze(Y_s[2]), np.squeeze(errY_s[2]), fmt='*', color='red')
axs[1,0].errorbar(X_s[3][:,0], np.squeeze(Y_s[3]), np.squeeze(errY_s[3]), fmt='D', color='purple')
axs[1,0].errorbar(X_s[4][:,0], np.squeeze(Y_s[4]), np.squeeze(errY_s[4]), fmt='^', color='green')

Y_exp_D.append(data["Y_s1"])

data = np.load("../single-fidelity-EXP/data_Diffusivity_T2_EXP.npz")
X_s = data["X_s1"]
Y_s = data["Y_s1"]
errY_s = data["errY_s1"]
axs[1,1].errorbar(X_s[0][:,0], np.squeeze(Y_s[0]), np.squeeze(errY_s[0]), fmt='o', color='black')
axs[1,1].errorbar(X_s[1][:,0], np.squeeze(Y_s[1]), np.squeeze(errY_s[1]), fmt='>', color='blue')
axs[1,1].errorbar(X_s[2][:,0], np.squeeze(Y_s[2]), np.squeeze(errY_s[2]), fmt='*', color='red')
axs[1,1].errorbar(X_s[3][:,0], np.squeeze(Y_s[3]), np.squeeze(errY_s[3]), fmt='D', color='purple')
axs[1,1].errorbar(X_s[4][:,0], np.squeeze(Y_s[4]), np.squeeze(errY_s[4]), fmt='^', color='green')

Y_exp_D.append(data["Y_s1"])

data = np.load("../single-fidelity-EXP/data_Diffusivity_T3_EXP.npz")
X_s = data["X_s1"]
Y_s = data["Y_s1"]
errY_s = data["errY_s1"]
axs[1,2].errorbar(X_s[0][:,0], np.squeeze(Y_s[0]), np.squeeze(errY_s[0]), fmt='o', color='black')
axs[1,2].errorbar(X_s[1][:,0], np.squeeze(Y_s[1]), np.squeeze(errY_s[1]), fmt='>', color='blue')
axs[1,2].errorbar(X_s[2][:,0], np.squeeze(Y_s[2]), np.squeeze(errY_s[2]), fmt='*', color='red')
axs[1,2].errorbar(X_s[3][:,0], np.squeeze(Y_s[3]), np.squeeze(errY_s[3]), fmt='D', color='purple')
axs[1,2].errorbar(X_s[4][:,0], np.squeeze(Y_s[4]), np.squeeze(errY_s[4]), fmt='^', color='green')

Y_exp_D.append(data["Y_s1"])

data = np.load("../single-fidelity-EXP/data_Diffusivity_T1_EXP_gp.npz")
X_s = data["X_s1"]
Y_s = data["Y_s1"]
axs[1,0].plot(X_s[0][:,0], Y_s[0], color='black')
axs[1,0].plot(X_s[1][:,0], Y_s[1], color='blue')
axs[1,0].plot(X_s[2][:,0], Y_s[2], color='red')
axs[1,0].plot(X_s[3][:,0], Y_s[3], color='purple')
axs[1,0].plot(X_s[4][:,0], Y_s[4], color='green')

data = np.load("../single-fidelity-EXP/data_Diffusivity_T2_EXP_gp.npz")
X_s = data["X_s1"]
Y_s = data["Y_s1"]
axs[1,1].plot(X_s[0][:,0], Y_s[0], color='black')
axs[1,1].plot(X_s[1][:,0], Y_s[1], color='blue')
axs[1,1].plot(X_s[2][:,0], Y_s[2], color='red')
axs[1,1].plot(X_s[3][:,0], Y_s[3], color='purple')
axs[1,1].plot(X_s[4][:,0], Y_s[4], color='green')

data = np.load("../single-fidelity-EXP/data_Diffusivity_T3_EXP_gp.npz")
X_s = data["X_s1"]
Y_s = data["Y_s1"]
axs[1,2].plot(X_s[0][:,0], Y_s[0], color='black')
axs[1,2].plot(X_s[1][:,0], Y_s[1], color='blue')
axs[1,2].plot(X_s[2][:,0], Y_s[2], color='red')
axs[1,2].plot(X_s[3][:,0], Y_s[3], color='purple')
axs[1,2].plot(X_s[4][:,0], Y_s[4], color='green')


data = np.load("../single-fidelity-EXP/data_Viscosity_T1_EXP.npz")
X_s = data["X_s1"]
Y_s = data["Y_s1"]
errY_s = data["errY_s1"]
axs[0,0].errorbar(X_s[0][:,0], np.squeeze(Y_s[0]), np.squeeze(errY_s[0]), fmt='o', color='black')
axs[0,0].errorbar(X_s[1][:,0], np.squeeze(Y_s[1]), np.squeeze(errY_s[1]), fmt='>', color='blue')
axs[0,0].errorbar(X_s[2][:,0], np.squeeze(Y_s[2]), np.squeeze(errY_s[2]), fmt='*', color='red')
axs[0,0].errorbar(X_s[3][:,0], np.squeeze(Y_s[3]), np.squeeze(errY_s[3]), fmt='D', color='purple')
axs[0,0].errorbar(X_s[4][:,0], np.squeeze(Y_s[4]), np.squeeze(errY_s[4]), fmt='^', color='green')

Y_exp_V.append(data["Y_s1"])

data = np.load("../single-fidelity-EXP/data_Viscosity_T2_EXP.npz")
X_s = data["X_s1"]
Y_s = data["Y_s1"]
errY_s = data["errY_s1"]
axs[0,1].errorbar(X_s[0][:,0], np.squeeze(Y_s[0]), np.squeeze(errY_s[0]), fmt='o', color='black')
axs[0,1].errorbar(X_s[1][:,0], np.squeeze(Y_s[1]), np.squeeze(errY_s[1]), fmt='>', color='blue')
axs[0,1].errorbar(X_s[2][:,0], np.squeeze(Y_s[2]), np.squeeze(errY_s[2]), fmt='*', color='red')
axs[0,1].errorbar(X_s[3][:,0], np.squeeze(Y_s[3]), np.squeeze(errY_s[3]), fmt='D', color='purple')
axs[0,1].errorbar(X_s[4][:,0], np.squeeze(Y_s[4]), np.squeeze(errY_s[4]), fmt='^', color='green')

Y_exp_V.append(data["Y_s1"])

data = np.load("../single-fidelity-EXP/data_Viscosity_T3_EXP.npz")
X_s = data["X_s1"]
Y_s = data["Y_s1"]
errY_s = data["errY_s1"]
axs[0,2].errorbar(X_s[0][:,0], np.squeeze(Y_s[0]), np.squeeze(errY_s[0]), fmt='o', color='black')
axs[0,2].errorbar(X_s[1][:,0], np.squeeze(Y_s[1]), np.squeeze(errY_s[1]), fmt='>', color='blue')
axs[0,2].errorbar(X_s[2][:,0], np.squeeze(Y_s[2]), np.squeeze(errY_s[2]), fmt='*', color='red')
axs[0,2].errorbar(X_s[3][:,0], np.squeeze(Y_s[3]), np.squeeze(errY_s[3]), fmt='D', color='purple')
axs[0,2].errorbar(X_s[4][:,0], np.squeeze(Y_s[4]), np.squeeze(errY_s[4]), fmt='^', color='green')

Y_exp_V.append(data["Y_s1"])

data = np.load("../single-fidelity-EXP/data_Viscosity_T1_EXP_gp.npz")
X_s = data["X_s1"]
Y_s = data["Y_s1"]
axs[0,0].plot(X_s[0][:,0], Y_s[0], color='black')
axs[0,0].plot(X_s[1][:,0], Y_s[1], color='blue')
axs[0,0].plot(X_s[2][:,0], Y_s[2], color='red')
axs[0,0].plot(X_s[3][:,0], Y_s[3], color='purple')
axs[0,0].plot(X_s[4][:,0], Y_s[4], color='green')

data = np.load("../single-fidelity-EXP/data_Viscosity_T2_EXP_gp.npz")
X_s = data["X_s1"]
Y_s = data["Y_s1"]
axs[0,1].plot(X_s[0][:,0], Y_s[0], color='black')
axs[0,1].plot(X_s[1][:,0], Y_s[1], color='blue')
axs[0,1].plot(X_s[2][:,0], Y_s[2], color='red')
axs[0,1].plot(X_s[3][:,0], Y_s[3], color='purple')
axs[0,1].plot(X_s[4][:,0], Y_s[4], color='green')

data = np.load("../single-fidelity-EXP/data_Viscosity_T3_EXP_gp.npz")
X_s = data["X_s1"]
Y_s = data["Y_s1"]
axs[0,2].plot(X_s[0][:,0], Y_s[0], color='black')
axs[0,2].plot(X_s[1][:,0], Y_s[1], color='blue')
axs[0,2].plot(X_s[2][:,0], Y_s[2], color='red')
axs[0,2].plot(X_s[3][:,0], Y_s[3], color='purple')
axs[0,2].plot(X_s[4][:,0], Y_s[4], color='green')


from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0], [0], marker='o', color='black', label='0.000'),
    Line2D([0], [0], marker='>', color='blue', label='0.096'),
    Line2D([0], [0], marker='*', color='red', label='0.230'),
    Line2D([0], [0], marker='D', color='purple', label='0.443'),
    Line2D([0], [0], marker='^', color='green', label='1.000'),
]

axs[0,0].legend(handles=legend_elements, ncol=1, fontsize=8, handletextpad=0.3, labelspacing=0.2, handlelength=1.0) 


fig.suptitle(r'\textbf{CO$_{\bm{2}}$/ethanol/dibenzyl ether}', x=0.53, fontsize=10)

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
# axs[0,1].set_yticklabels([])
# axs[0,2].set_yticklabels([])
# axs[1,1].set_yticklabels([])
# axs[1,2].set_yticklabels([])
axs[0,0].set_xticks([150, 200, 250, 300, 350])
axs[0,1].set_xticks([150, 200, 250, 300, 350])
axs[0,2].set_xticks([150, 200, 250, 300, 350])
axs[1,0].set_xticks([150, 200, 250, 300, 350])
axs[1,1].set_xticks([150, 200, 250, 300, 350])
axs[1,2].set_xticks([150, 200, 250, 300, 350])
axs[1,0].set_yticks([1,5,9,13])
axs[1,1].set_yticks([1,5,9,13])
axs[1,2].set_yticks([1,5,9,13])
axs[0,0].set_yticks([2,5,8,11])
axs[0,1].set_yticks([2,5,8,11])
axs[0,2].set_yticks([2,5,8,11])
axs[1,0].set_ylim(0,14)
axs[1,1].set_ylim(0,14)
axs[1,2].set_ylim(0,14)
axs[0,0].set_ylim(0,12)
axs[0,1].set_ylim(0,12)
axs[0,2].set_ylim(0,12)

plt.savefig('plot_CO2_ethanol_dibenzylether_EXP.png', dpi=600) 

MRE1 = (100*np.abs(np.array(Y_sim_D)-np.array(Y_exp_D))/np.array(Y_exp_D))
MRE2 = (100*np.abs(np.array(Y_sim_V)-np.array(Y_exp_V))/np.array(Y_exp_V))
print(f"MRE Diffusivity: min {MRE1.min():.2f}%, max {MRE1.max():.2f}%, mean {MRE1.mean():.2f}%")
print(f"MRE Viscosity: min {MRE2.min():.2f}%, max {MRE2.max():.2f}%, mean {MRE2.mean():.2f}%")




