import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import matplotlib.lines as mlines
from matplotlib.legend_handler import HandlerBase

class RectDotHandler(HandlerBase):
    def __init__(self, rect_color='purple', dot_color='black', **kwargs):
        self.rect_color = rect_color
        self.dot_color = dot_color
        super().__init__(**kwargs)
    
    def create_artists(self, legend, orig_handle,
                       xdescent, ydescent, width, height, fontsize, trans):
        # Draw rectangle
        rect = mpatches.Rectangle([xdescent, ydescent], width, height,
                                  facecolor=self.rect_color,
                                  edgecolor='black', alpha=0.5, linewidth=1, transform=trans)
        # Draw dot centered inside the rectangle
        dot_radius = height / 2.7  # relative radius of dot
        cx = xdescent + width / 2
        cy = ydescent + height / 2
        dot = mpatches.Circle((cx, cy), dot_radius,
                              facecolor=self.dot_color,
                              edgecolor=self.dot_color, linewidth=1.2, transform=trans)
        return [rect, dot]

palette = sns.color_palette("Paired")
sns.set(style='whitegrid', context='notebook', font_scale=2.8)
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['savefig.dpi'] = 300

metrics = ["MAE", "SSIM", "FRCM", "ReconsErr"]

fig, axes = plt.subplots(2, 2, figsize=(32, 28))
axes = axes.flatten()

for idx, metric in enumerate(metrics):
    metric=metrics[idx]+"/"
    chair_gsnet = np.load(metric+'Chair_GSNet.npy')
    tear_gsnet = np.load(metric+'Tear_GSNet.npy')
    rec_gsnet = np.load(metric+'Rec_GSNet.npy')
    hat_gsnet = np.load(metric+'Hat_GSNet.npy')
    ring_gsnet = np.load(metric+'Ring_GSNet.npy')
    gauss_gsnet = np.load(metric+'Gauss_GSNet.npy')
    
    chair_reg = np.load(metric+'Chair_reg.npy')
    tear_reg = np.load(metric+'Tear_reg.npy')
    rec_reg = np.load(metric+'Rec_reg.npy')
    hat_reg = np.load(metric+'Hat_reg.npy')
    ring_reg = np.load(metric+'Ring_reg.npy')
    gauss_reg = np.load(metric+'Gauss_reg.npy')
    
    if(metric!='ReconsErr/'):
        raf_gsnet = np.load(metric+'RAF_GSNet.npy')
        xray_gsnet = np.load(metric+'Xray_GSNet.npy')
        
        raf_reg = np.load(metric+'RAF_reg.npy')
        xray_reg = np.load(metric+'Xray_reg.npy')
    
    if(metric!='ReconsErr/'):
        values_gsnet = np.concatenate([
            chair_gsnet,
            tear_gsnet,
            rec_gsnet,
            hat_gsnet,
            ring_gsnet,
            gauss_gsnet,
            xray_gsnet,
            raf_gsnet
        ])
        values_reg = np.concatenate([
            chair_reg,
            tear_reg,
            rec_reg,
            hat_reg,
            ring_reg,
            gauss_reg,
            xray_reg,
            raf_reg
        ])
    else:
        values_gsnet = np.concatenate([
            chair_gsnet,
            tear_gsnet,
            rec_gsnet,
            hat_gsnet,
            ring_gsnet,
            gauss_gsnet
        ])
        values_reg = np.concatenate([
            chair_reg,
            tear_reg,
            rec_reg,
            hat_reg,
            ring_reg,
            gauss_reg
        ])

    if(metric!='ReconsErr/'):
        groups_gsnet = ['Chair', 'Tear', 'Rec-\nTophat', 'Tophat', 'Ring', 'Gaussian', 'X-\nray', 'RAF-\nCDI']
        lengths_gsnet = [len(chair_gsnet), len(tear_gsnet), len(rec_gsnet), len(hat_gsnet), len(ring_gsnet), len(gauss_gsnet), len(xray_gsnet), len(raf_gsnet)]
        groups_reg = ['Chair', 'Tear', 'RecTophat', 'Tophat', 'Ring', 'Gaussian', 'X-\nray', 'RAF-\nCDI']
        lengths_reg = [len(chair_reg), len(tear_reg), len(rec_reg), len(hat_reg), len(ring_reg), len(gauss_reg), len(xray_reg), len(raf_reg)]
    else:
        groups_gsnet = ['Chair', 'Tear', 'RecTophat', 'Tophat', 'Ring', 'Gaussian']
        lengths_gsnet = [len(chair_gsnet), len(tear_gsnet), len(rec_gsnet), len(hat_gsnet), len(ring_gsnet), len(gauss_gsnet)]
        groups_reg = ['Chair', 'Tear', 'RecTophat', 'Tophat', 'Ring', 'Gaussian']
        lengths_reg = [len(chair_reg), len(tear_reg), len(rec_reg), len(hat_reg), len(ring_reg), len(gauss_reg)]
    
    group_labels_gsnet = np.concatenate([
        [g] * l for g, l in zip(groups_gsnet, lengths_gsnet)
    ])
    group_labels_reg = np.concatenate([
        [g] * l for g, l in zip(groups_reg, lengths_reg)
    ])

    df_gsnet = pd.DataFrame({
        'value': values_gsnet,
        'group': group_labels_gsnet
    })
    df_reg = pd.DataFrame({
        'value': values_reg,
        'group': group_labels_reg
    })

    ax = axes[idx]
    
    sns.violinplot(
        x='group',
        y='value',
        data=df_reg,
        color=palette[5],
        inner=None,
        linewidth=1,
        cut=0,
        scale="width",
        width=0.8,
        ax = ax,
        fontsize=35
    )
    sns.violinplot(
        x='group',
        y='value',
        data=df_gsnet,
        color=palette[2],
        inner=None,
        linewidth=1,
        cut=0,
        scale="width",
        width=0.8,
        ax = ax,
        fontsize=35
    )
    for violin in ax.collections:
        violin.set_alpha(0.5)

    group_means_gsnet = [df_gsnet[df_gsnet['group'] == g]['value'].mean() for g in groups_gsnet]
    group_means_reg = [df_reg[df_reg['group'] == g]['value'].mean() for g in groups_reg]
        
    for i, mean in enumerate(group_means_gsnet):
        ax.scatter(i, mean, color='white', s=100, zorder=10, edgecolor='white', linewidth=1.2, label='Mean' if i == 0 else "")
        if(metric!='SSIM/'):
            x0, y0 = i, mean
            if(group_means_gsnet[i]>group_means_reg[i]):
                x1, y1 = i - 0.2, 10 ** (np.log10(mean) + 0.3)
            else:
                x1, y1 = i - 0.2, 10 ** (np.log10(mean) + 0.1)
            x2, y2 = x1 - 0.3, y1
        else:
            x0, y0 = i, mean
            if(group_means_gsnet[i]>group_means_reg[i]):
                x1, y1 = i - 0.2, mean+0.1
            else:
                x1, y1 = i - 0.2, mean+0.05
            x2, y2 = x1 - 0.3, y1

        ax.plot([x0, x1], [y0, y1], color='green', linewidth=1.5)
        ax.plot([x1, x2], [y1, y2], color='green', linewidth=1.5)
        ax.text(x2, y2, f"{mean:.3f}", color='green', ha='center', va='bottom')
    for i, mean in enumerate(group_means_reg):
        ax.scatter(i, mean, color='black', s=100, zorder=10, edgecolor='white', linewidth=1.2, label='Mean' if i == 0 else "")
        if(metric!='SSIM/'):
            x0, y0 = i, mean
            if(group_means_reg[i]>group_means_gsnet[i]):
                x1, y1 = i - 0.2, 10 ** (np.log10(mean) + 0.3)
            else:
                x1, y1 = i - 0.2, 10 ** (np.log10(mean) + 0.1)
            x2, y2 = x1 - 0.3, y1
        else:
            x0, y0 = i, mean
            if(group_means_reg[i]>group_means_gsnet[i]):
                x1, y1 = i - 0.2, mean+0.1
            else:
                x1, y1 = i - 0.2, mean+0.05
            x2, y2 = x1 - 0.3, y1

        ax.plot([x0, x1], [y0, y1], color='blue', linewidth=1.5)
        ax.plot([x1, x2], [y1, y2], color='blue', linewidth=1.5)
        ax.text(x2, y2, f"{mean:.3f}", color='blue', ha='center', va='bottom')

    if(metric=="MAE/"):
        ax.set_ylabel("MAE ↓ [rad]", fontsize=35)
    if(metric=="SSIM/"):
        ax.set_ylabel("SSIM ↑ [a.u.]", fontsize=35)
    if(metric=="FRCM/"):
        ax.set_ylabel("FRCM ↓ [a.u.]", fontsize=35)
    if(metric=="ReconsErr/"):
        ax.set_ylabel("ReconsErr ↓ [a.u.]", fontsize=35)

    sns.despine()
    ax.set_xlabel('')
    ax.tick_params(axis='x', labelsize=38)
    if(metric!="SSIM/"):
        ax.set_yscale('log')
    ax.legend_.remove() if ax.get_legend() else None
    plt.tight_layout()

gsnet_handle = mpatches.Patch(
    facecolor=palette[2],
    edgecolor='black',
    linewidth=1,
    alpha=0.5,
    label='With physics knowledge injected'
    )
reg_handle = mpatches.Patch(
    facecolor=palette[5],
    edgecolor='black',
    linewidth=1,
    alpha=0.5,
    label='Without physics knowledge injected'
    )

whitedot_handle = mpatches.Patch()
blackdot_handle = mpatches.Patch()

fig.legend(
    handles=[gsnet_handle, reg_handle, whitedot_handle, blackdot_handle],
    labels=[
        'With physics knowledge injected',
        'Without physics knowledge injected',
        'Mean with physics knowledge injected',
        'Mean without physics knowledge injected'
    ],
    handler_map={
        whitedot_handle: RectDotHandler(rect_color=palette[2], dot_color='white'),
        blackdot_handle: RectDotHandler(rect_color=palette[5], dot_color='black')
    },
    loc='upper right',
    ncol=2,
    bbox_to_anchor=(0.65, 0.94),
    frameon=False,
    fontsize=38
)

plt.tight_layout()
plt.subplots_adjust(top=0.88)
plt.savefig("violin_ablation.png", dpi=300)
plt.show()