import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_auc_score,auc,precision_recall_curve,roc_curve,confusion_matrix
from sklearn.preprocessing import StandardScaler
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from collections import Counter
from matplotlib.gridspec import GridSpec
from matplotlib.colors import ListedColormap
from matplotlib.colors import LinearSegmentedColormap


def predict_visulized(y_true, y_pred, y_prob, score=0.5,
                      color='steelblue', vmin=10, vmax=20):
    
    
    fpr,tpr,thre = roc_curve(y_true, y_prob)
    auc_score = roc_auc_score(y_true, y_prob)
    
    plt.figure(figsize=(16, 4),dpi=60)
    plt.rcParams['font.sans-serif'] = 'Arial'
    
    #ROC curve
    plt.subplot(1,4,1)
    plt.plot(fpr,tpr,linewidth=3,color=color)
    plt.plot([0,1],[0,1],'k--')
    plt.xlabel('False Positive Rate', fontsize=16, labelpad=5)
    plt.ylabel('True Positive Rate', fontsize=16, labelpad=5)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.text(0.5, 0.2, f'AUC = {auc_score:.4f}', fontsize=14)
    
    #FPR-TPR curve
    index = np.where(thre > score)[0][-1]
    plt.subplot(1,4,2)
    plt.plot(thre, fpr, label='FPR', linewidth=3)
    plt.plot(thre,tpr,label='TPR',linewidth=3)
    plt.legend(fontsize=14)
    plt.xlabel('Threshold', fontsize=16,labelpad=5)
    plt.ylabel('Rate', fontsize=16,labelpad=5)
    plt.plot((score, score),(0, tpr[index]), 'k--',linewidth=2)
    plt.plot((0,score),(tpr[index],tpr[index]), 'k--',linewidth=2)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.xlim(0,1)
    plt.ylim(0,1)
    
    #PRAUC curve
    tpr,fpr,thre = precision_recall_curve(y_true, y_prob)
    prauc = auc(fpr,tpr)
    plt.subplot(1,4,3)
    plt.plot(fpr,tpr,linewidth=3,color='#274a5d')
    plt.plot([0,1],[1,0],'k--')
    plt.xlabel('Recall', fontsize=16,labelpad=5)
    plt.ylabel('Precision', fontsize=16,labelpad=5)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.text(0.1,0.2,f'AUPRC = {prauc:.4f}',fontsize=14)
    
    #Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.subplot(1,4,4)
    sns.heatmap(data=cm,annot=True,fmt='d',cmap="YlGnBu",
                vmin=vmin,vmax=vmax,annot_kws={"size": 14})
    plt.xlabel('Predicted label', fontsize=16,labelpad=5)
    plt.ylabel('True label', fontsize=16,labelpad=5)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    
    plt.tight_layout()
    plt.show()
    

def striped_heatmap(data, xlabel, interval=10, expand=0,
                    fff=False, save=False, fig_path=None, fig_name=None):
    
    fig, ax = plt.subplots(figsize=(4, 1),dpi=60)
    plt.rcParams['font.sans-serif'] = 'Arial'

    ax.plot([0-expand, data.shape[1]+1+expand], [0.7, 0.7], color="black")
    ax.plot([0-expand, data.shape[1]+1+expand], [1, 1], color="black")
    ax.plot([0-expand, 0-expand], [0.7, 1], color="black")
    ax.plot([data.shape[1]+1+expand, data.shape[1]+1+expand], [0.7, 1], color="black")
    
    #color mapping
    norm = Normalize(vmin=np.min(data), vmax=np.max(data))
    if fff:
        colors_hex = ['#f0faff','#0094fa']
        cmap_name = 'my_list'
        cmap = LinearSegmentedColormap.from_list(cmap_name, colors_hex, N=data.shape[1])
    else:
        colors_hex = ['#F4F5F6','#FFF6F4','#FFF4EC','#FFE1D8','#b20012']
        cmap_name = 'my_list'
        cmap = LinearSegmentedColormap.from_list(cmap_name, colors_hex, N=data.shape[1])
    
    sm = ScalarMappable(norm=norm, cmap=cmap)
    
    for i, value in enumerate(data.T):
        ax.plot([i+1,i+1], [0.72,0.98], color=sm.to_rgba(value), lw=2)
    
    ax.set_xticks(range(1, data.shape[1]+1, interval))
    ax.set_xticklabels(range(1, data.shape[1]+1, interval), fontsize=10)
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylim(0.68, 1.05)
    ax.get_yaxis().set_visible(False)
    ax.spines[:].set_visible(False)
    if save:
        plt.savefig(os.path.join(fig_path, fig_name), dpi=60,
                    format='pdf',bbox_inches='tight')
    plt.show()


def att_heatmap(data, col_split, annot_group, colors='Set2'):
    data = StandardScaler().fit_transform(data)
    data=pd.DataFrame(data).T
    data.insert(0, 'Group', col_split['Group'].values)
    
    width_ratios = list(Counter(data['Group']).values())
    vmin, vmax = data.iloc[:,1:].min().min(), data.iloc[:,1:].max().max()
    if type(colors) == str:
        palette = sns.color_palette(colors, n_colors=len(width_ratios))
    elif type(colors) == list:
        palette = colors
    else: print("Please input color name or color list in RGB")
    
    group2color = {}
    i = 0
    for group, _ in data.groupby(by='Group'):
        if group in annot_group:
            group2color.update({group:palette[i]})
            i += 1
        else:
            group2color.update({group:(0.827, 0.827, 0.827)})
    row_colors = data.Group.apply(lambda x:group2color[x]).to_list()
    
    fig = plt.figure(figsize=(6, 5), dpi=60)
    plt.rcParams['font.sans-serif'] = 'Arial'
    gs = GridSpec(2, len(width_ratios)+1, figure=fig, height_ratios=[0.03, 0.97],
                  width_ratios=width_ratios+[12], wspace=0.08, hspace=0.02)

    axs = []
    for i, table in data.groupby(by='Group'):
        start, end = table.index[0], table.index[-1]
        ax = fig.add_subplot(gs[1, int(i)])
        ax_color = fig.add_subplot(gs[0, int(i)])
        ax_color.imshow([row_colors[start:end+1]], aspect="auto", extent=[0,end-start+1,0,1],
                        cmap=ListedColormap(row_colors[start:end+1]))
        ax_color.set_xlim(0, end-start+1)
        ax_color.set_xticks([])
        ax_color.set_yticks([])
        
        sns.heatmap(table.iloc[:,1:].T, cmap="RdBu_r", ax=ax, cbar=False, vmin=vmin, vmax=vmax)
        if i == 0:
            ax.set_ylabel("Atom rank", fontsize=14)
            ax.set_yticks(range(1, data.shape[1], 5))
            ax.set_yticklabels(range(1, data.shape[1], 5), fontsize=10)
        else:
            ax.set_yticks([])
        if i == data['Group'].max():
            ax.set_xticks(np.array([0, ax.get_xlim()[1]]))
            ax.set_xticklabels(np.array([start, end])+1,rotation=90, fontsize=10)
        else:
            ax.set_xticks(np.array([0]))
            ax.set_xticklabels(np.array([start])+1,rotation=90, fontsize=10)
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color('black')
            spine.set_linewidth(.5)
        axs.append(ax)
    cbar_ax = fig.add_subplot(gs[1:, -1])
    cbar = fig.colorbar(axs[-1].collections[0], cax=cbar_ax,
                        location='right', pad=0.02, aspect=30)
    cbar.ax.tick_params(labelsize=10)
    cbar.set_label('Normalized attention score', size=12, labelpad=5)
    fig.text(0.45, -0.02, "Residue rank", ha='center', va='center', fontsize=14)
    plt.tight_layout()
    plt.show()


