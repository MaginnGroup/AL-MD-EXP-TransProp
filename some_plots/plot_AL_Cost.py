import numpy as np
import matplotlib.pyplot as plt
import figstyle


fig, axs = plt.subplots(4, 1, figsize=(3.2, 5.4), dpi=300, constrained_layout=True)

axs[0].plot( [1,2,3,4] , [5,5,4,4], markeredgecolor='black', markersize=6, marker='D', linestyle='--', color='purple', label=r'CO$_2$/n-heptane')
axs[0].plot( [1,2,3,4] , [8,7,8,7], markeredgecolor='black', markersize=6, marker='o', linestyle='--', color='blue', label=r'CO$_2$/benzene') 
axs[0].plot( [1,2,3,4] , [5,4,4,4], markeredgecolor='black', markersize=6, marker='>', linestyle='--', color='gray', label=r'toluene/n-hexane') 
axs[0].plot( [1,2,3,4] , [9,6,6,4], markeredgecolor='black', markersize=6, marker='^', linestyle='--', color='green', label=r'CO$_2$/ethanol/'+'\n dibenzyl ether') 

axs[1].plot( [1,2,3,4] , [7,8,10,10], markeredgecolor='black', markersize=6, marker='D', linestyle='--', color='purple', label='')
axs[1].plot( [1,2,3,4] , [10,10,19,37], markeredgecolor='black', markersize=6, marker='o', linestyle='--', color='blue', label='') 
axs[1].plot( [1,2,3,4] , [7,8,8,10], markeredgecolor='black', markersize=6, marker='>', linestyle='--', color='gray', label='') 
axs[1].plot( [1,2,3,4] , [13,75,75,75], markeredgecolor='black', markersize=6, marker='^', linestyle='--', color='green', label='') 

axs[2].plot( [1,2,3,4] , [2.74,2.73,2.82,2.82], markeredgecolor='black', markersize=6, marker='D', linestyle='--', color='purple', label='')
axs[2].plot( [1,2,3,4] , [5.60,12.25,5.49,11.69], markeredgecolor='black', markersize=6, marker='o', linestyle='--', color='blue', label='') 
axs[2].plot( [1,2,3,4] , [0.83,1.39,1.39,1.71], markeredgecolor='black', markersize=6, marker='>', linestyle='--', color='gray', label='') 
axs[2].plot( [1,2,3,4] , [4.34,8.12,8.12,13.49], markeredgecolor='black', markersize=6, marker='^', linestyle='--', color='green', label='') 

axs[3].plot( [1,2,3,4] , [1.80,1.85,2.92,2.92], markeredgecolor='black', markersize=6, marker='D', linestyle='--', color='purple', label='')
axs[3].plot( [1,2,3,4] , [4.75,8.90,4.33,9.38], markeredgecolor='black', markersize=6, marker='o', linestyle='--', color='blue', label='') 
axs[3].plot( [1,2,3,4] , [2.02,6.90,6.90,6.46], markeredgecolor='black', markersize=6, marker='>', linestyle='--', color='gray', label='') 
axs[3].plot( [1,2,3,4] , [2.34,7.97,7.97,9.09], markeredgecolor='black', markersize=6, marker='^', linestyle='--', color='green', label='') 


axs[0].set_ylabel('Training Points EXP')
axs[1].set_ylabel('Training Points MD')
axs[2].set_ylabel(r'MRE of $\eta$ (\%)')
axs[3].set_ylabel(r'MRE of $D$ (\%)')

axs[3].set_xlabel(r'Cost$_{high}$/Cost$_{low}$')
 

axs[0].set_ylim(0,10)
axs[1].set_ylim(0,80)
axs[2].set_ylim(0,15)
axs[3].set_ylim(0,10)

axs[0].set_xticklabels([])
axs[1].set_xticklabels([])
axs[2].set_xticklabels([])

axs[0].set_yticks([0, 2, 4, 6, 8, 10])
axs[3].set_yticks([0, 2, 4, 6, 8, 10])
axs[1].set_yticks([0, 20, 40, 60, 80])
axs[2].set_yticks([0, 3, 6, 9, 12, 15])

axs[0].set_xticks([1, 2, 3, 4])
axs[1].set_xticks([1, 2, 3, 4])
axs[2].set_xticks([1, 2, 3, 4])
axs[3].set_xticks([1, 2, 3, 4])


from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0], [0], marker='D', color='purple', label='CO$_2$/n-heptane',
           markerfacecolor='purple', markeredgecolor='black', markersize=6, linestyle='--'),
    Line2D([0], [0], marker='o', color='blue', label='CO$_2$/benzene',
           markerfacecolor='blue', markeredgecolor='black', markersize=6, linestyle='--'),
    Line2D([0], [0], marker='>', color='gray', label='toluene/n-hexane',
           markerfacecolor='gray', markeredgecolor='black', markersize=6, linestyle='--'),
    Line2D([0], [0], marker='^', color='green', label='CO$_2$/ethanol/dibenzyl ether',
           markerfacecolor='green', markeredgecolor='black', markersize=6, linestyle='--')
]
fig.legend(handles=legend_elements,
           loc='upper center',
           bbox_to_anchor=(0.5, 1.09),
           ncol=2)

fig.align_ylabels(axs[:])

plt.savefig('plot_AL_Cost.png', dpi=600) 



