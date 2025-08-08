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
offset = 0.25  # Largura das barras


g1 = [30, 22, 25, 75]
g2 = [9, 10, 9, 20]
g3 = [5, 8, 5, 9]
g3b = [7, 10, 7, 13]    
axs[0].bar(x - offset, g1, width=largura, hatch='//', edgecolor='black', color='#CCCCCC', label='Total EXP')
axs[0].bar(x, g2, width=largura, hatch='..', edgecolor='black', color='#969696', label='Single-fidelity AL EXP')
axs[0].bar(x + offset, g3, width=largura, hatch='xx', edgecolor='black', color='white', label='Multi-fidelity AL EXP')
for x_, val in zip(x - offset, g1):
    axs[0].text(x_, val + 1, str(val), ha='center', va='bottom')
for x_, val in zip(x, g2):
    axs[0].text(x_, val + 1, str(val), ha='center', va='bottom')
for x_, a, b in zip(x + offset, g3, g3b):
    axs[0].text(x_, a + 1, str(a), ha='center', va='bottom')  
    axs[0].text(x_+0.1, a + 32, f'+{b} MD', ha='center', va='top', color='brown', rotation=60, fontsize=7)       


g1 = [0.6, 0.9, 0.6, 1.5]
g2 = [2.0, 5.1, 0.9, 3.4]
g3 = [2.7, 5.6, 0.8, 4.3]   
axs[1].bar(x - offset, g1, width=largura, hatch='//', edgecolor='black', color='#CCCCCC', label='without AL')
axs[1].bar(x, g2, width=largura, hatch='..', edgecolor='black', color='#969696', label='with AL')
axs[1].bar(x + offset, g3, width=largura, hatch='xx', edgecolor='black', color='white', label='with AL')
for x_, val in zip(x - offset, g1):
    axs[1].text(x_ -0.05, val + 0.1, f'{val}\%'.replace('0.', '.'), ha='center', va='bottom')
for x_, val in zip(x, g2):
    if x_== 2.2:
        axs[1].text(x_, val + 0.1, f'{val}\%'.replace('0.', '.'), ha='center', va='bottom')
    else:
        axs[1].text(x_-0.08, val + 0.1, f'{val}\%'.replace('0.', '.'), ha='center', va='bottom')
for x_, val in zip(x + offset, g3):
    axs[1].text(x_ + 0.05, val + 0.1, f'{val}\%'.replace('0.', '.'), ha='center', va='bottom')


g1 = [1.4, 0.9, 1.2, 1.0]
g2 = [1.7, 3.8, 1.6, 2.0]
g3 = [1.8, 4.8, 2.0, 2.3]   
axs[2].bar(x - offset, g1, width=largura, hatch='//', edgecolor='black', color='#CCCCCC', label='without AL')
axs[2].bar(x, g2, width=largura, hatch='..', edgecolor='black', color='#969696', label='with AL')
axs[2].bar(x + offset, g3, width=largura, hatch='xx', edgecolor='black', color='white', label='with AL')
for x_, val in zip(x - offset, g1):
    axs[2].text(x_-0.07, val + 0.1, f'{val}\%'.replace('0.', '.'), ha='center', va='bottom')
for x_, val in zip(x, g2):
    if x_== 0.0:
        axs[2].text(x_-0.04, val + 0.4, f'{val}\%'.replace('0.', '.'), ha='center', va='bottom')
    elif x_== 2.2:
        axs[2].text(x_-0.05, val + 0.3, f'{val}\%'.replace('0.', '.'), ha='center', va='bottom')
    else:
        axs[2].text(x_-0.08, val + 0.1, f'{val}\%'.replace('0.', '.'), ha='center', va='bottom')
for x_, val in zip(x + offset, g3):
    if x_== 0.25:
        axs[2].text(x_+0.15, val + 0.1, f'{val}\%'.replace('0.', '.'), ha='center', va='bottom')
    elif x_== 2.45:
        axs[2].text(x_+0.15, val + 0.1, f'{val}\%'.replace('0.', '.'), ha='center', va='bottom')
    else:
        axs[2].text(x_+0.06, val + 0.14, f'{val}\%'.replace('0.', '.'), ha='center', va='bottom')
    
axs[0].set_ylabel('Training Points')
axs[1].set_ylabel(r'MRE of $\eta$ (\%)')
axs[2].set_ylabel(r'MRE of $D$ (\%)')


axs[0].set_ylim(0,85)
axs[1].set_ylim(0,6.5)
axs[2].set_ylim(0,6.5)

axs[0].legend()

axs[0].set_xticks(x , categorias, ha='center')  
axs[1].set_xticks(x , categorias, ha='center')  
axs[2].set_xticks(x , categorias, ha='center')  

axs[0].set_xticklabels([])
axs[1].set_xticklabels([])

fig.align_ylabels(axs[:])

plt.savefig('plot_AL_EXP.png', dpi=600) 



