import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

sns.set(style='whitegrid', context='notebook', font_scale=2.25)
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['savefig.dpi'] = 300

metrics = ["MAE", "SSIM", "FRCM"]

beamshape="near-field_Xray_imaging/"

fig, axes = plt.subplots(1, 3, figsize=(21, 7))
axes = axes.flatten()

for idx, metric in enumerate(metrics):
    metric=metrics[idx]+"/"
    FourierGSNet=beamshape+metric+"FourierGSNet_WeakConstraint.npy"
    iclr=beamshape+metric+"ICLR_WeakConstraint.npy"
    SiSPRNet=beamshape+metric+"SiSPRNet.npy"
    deepCDI=beamshape+metric+"deepCDI.npy"

    FourierGSNet=np.load(FourierGSNet)
    iclr=np.load(iclr)
    SiSPRNet=np.load(SiSPRNet)
    deepCDI=np.load(deepCDI)

    values = np.concatenate([
        FourierGSNet,
        iclr,
        SiSPRNet,
        deepCDI,
    ])

    groups = ['FourierGSNet', 'GS direct unrolling', 'SiSPRNet', 'deep-CDI']
    lengths = [len(FourierGSNet), len(iclr), len(SiSPRNet), len(deepCDI)]

    group_labels = np.concatenate([
        [g] * l for g, l in zip(groups, lengths)
    ])

    df = pd.DataFrame({
        'value': values,
        'group': group_labels
    })

    palette = sns.color_palette("Paired")
    ax = axes[idx]
    
    sns.violinplot(
        x='group',
        y='value',
        data=df,
        palette=palette,
        inner=None,
        linewidth=1,
        cut=0,
        scale="width",
        width=0.8,
        ax = ax
    )
    for violin in ax.collections:
        violin.set_alpha(0.5)

    group_means = [df[df['group'] == g]['value'].mean() for g in groups]
    if(metric=="MAE/"):
        mark=min(group_means)
    if(metric=="SSIM/"):
        mark=max(group_means)
    if(metric=="FRCM/"):
        mark=min(group_means)
        
    for i, mean in enumerate(group_means):
        ax.scatter(i, mean, color='red', s=80, zorder=10, edgecolor='white', linewidth=1.2, label='Mean' if i == 0 else "")

    group_maxes = [df[df['group'] == g]['value'].max() for g in groups]
    for i, maxx in enumerate(group_maxes):
        #ax.vlines(x=i, ymin=df['value'].min(), ymax=maxx, color='black', linewidth=2.5, alpha=0.8, linestyle=(0, (2, 2)))
        mean = group_means[i]
        if(mean==mark):
            ax.text(i, group_maxes[i], f'{mean:.3f}', ha='center', va='bottom', color="mediumseagreen")
        else:
            ax.text(i, group_maxes[i], f'{mean:.3f}', ha='center', va='bottom', color='black')

    ax.set_xlabel("Benchmarked Methods")

    if(metric=="MAE/"):
        ax.set_ylabel("MAE ↓ [rad]")
    if(metric=="SSIM/"):
        ax.set_ylabel("SSIM ↑ [a.u.]")
    if(metric=="FRCM/"):
        ax.set_ylabel("FRCM ↓ [a.u.]")

    sns.despine()
    ax.set_xticks([])
    ax.legend_.remove() if ax.get_legend() else None
    plt.tight_layout()
    
legend_labels = ['FourierGSNet',
                 'GS direct unrolling',
                 'SiSPRNet',
                 'deep-CDI']

legend_patches = [
    mpatches.Patch(
        facecolor=palette[i],
        edgecolor='black',
        linewidth=1,
        alpha=0.5,
        label=legend_labels[i]
    ) for i in range(4)
]

mean_dot = Line2D(
    [0], [0],
    marker='o',
    color='red',
    label='Mean',
    markersize=10,
    linestyle='None',
    markeredgecolor='white',
    markeredgewidth=1.2
)

all_handles = legend_patches + [mean_dot]

fig.legend(
    handles=all_handles,
    loc='upper center',
    ncol=3,
    bbox_to_anchor=(0.3, 1.15),
    frameon=False,
    fontsize=30
)
    
plt.tight_layout()
plt.subplots_adjust(top=0.88)
plt.savefig(beamshape[:-1]+".png", dpi=300, bbox_inches='tight')
plt.show()