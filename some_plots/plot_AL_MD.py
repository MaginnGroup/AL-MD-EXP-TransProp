import numpy as np
import matplotlib.pyplot as plt
import figstyle


fig, axs = plt.subplots(3, 1, figsize=(3.2, 5.0), dpi=300, constrained_layout=True)


categorias = [
    r'\textbf{CO$_{\bm{2}}$/}' + '\n' + r'\textbf{n-heptane}',
    r'\textbf{CO$_{\bm{2}}$/}' + '\n' + r'\textbf{benzene}',
    r'\textbf{toluene/}' + '\n' + r'\textbf{n-hexane}',
    r'\textbf{CO$_{\bm{2}}$/}' + '\n' + r'\textbf{ethanol/}' + '\n' + r'\textbf{dibenzyl ether}'
]

x = np.arange(len(categorias)) *1.1 # [0, 1, 2, 3]
largura = 0.2  # Largura das barras
gap = 0.05  # Espaçamento entre as barras do mesmo grupo


g1 = [48, 48, 25, 75]
g2 = [7, 17, 7, 15]
axs[0].bar(x - largura - gap/2, g1, width=largura, hatch='//', edgecolor='black', color='#CCCCCC', label='Total MD')
axs[0].bar(x + gap/2, g2, width=largura, hatch='..', edgecolor='black', color='#969696', label='Single-fidelity AL MD')
for x_, val in zip(x - largura - gap/2, g1):
    axs[0].text(x_, val + 1, str(val), ha='center', va='bottom')
for x_, val in zip(x + gap/2, g2):
    axs[0].text(x_, val + 1, str(val), ha='center', va='bottom')
    

g1 = [1.4, 1.6, 1.5, 1.9]
g2 = [3.0, 3.2, 2.5, 5.1]
axs[1].bar(x - largura - gap/2, g1, width=largura, hatch='//', edgecolor='black', color='#CCCCCC')
axs[1].bar(x + gap/2, g2, width=largura, hatch='..', edgecolor='black', color='#969696')
for x_, val in zip(x - largura - gap/2, g1):
    axs[1].text(x_ -0.07, val + 0.1, f'{val}\%', ha='center', va='bottom')
for x_, val in zip(x + gap/2, g2):
    axs[1].text(x_, val + 0.1, f'{val}\%', ha='center', va='bottom')

g1 = [0.3, 0.7, 0.9, 0.2]
g2 = [2.2, 2.2, 1.5, 2.0]
axs[2].bar(x - largura - gap/2, g1, width=largura, hatch='//', edgecolor='black', color='#CCCCCC')
axs[2].bar(x + gap/2, g2, width=largura, hatch='..', edgecolor='black', color='#969696')
for x_, val in zip(x - largura - gap/2, g1):
    axs[2].text(x_ -0.07, val + 0.1, f'{val}\%'.replace('0.', '.'), ha='center', va='bottom')
for x_, val in zip(x + gap/2, g2):
    axs[2].text(x_, val + 0.1, f'{val}\%', ha='center', va='bottom')
    
axs[0].set_ylabel('Training Points')
axs[1].set_ylabel(r'MRE of $\eta$ (\%)')
axs[2].set_ylabel(r'MRE of $D$ (\%)')

axs[0].set_ylim(0,85)
axs[1].set_ylim(0,6.5)
axs[2].set_ylim(0,6.5)

axs[0].legend()

axs[0].set_xticks(x - largura / 2, categorias, ha='center')
axs[1].set_xticks(x - largura / 2, categorias, ha='center') 
axs[2].set_xticks(x - largura / 2, categorias, ha='center') 

axs[0].set_xticklabels([])
axs[1].set_xticklabels([])

fig.align_ylabels(axs[:])

plt.savefig('plot_AL_MD.png', dpi=600) 



