import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import os

def plot_advanced_confusion_matrix(y_true, y_pred, labels, title, save_path):
    """
    Plots an advanced confusion matrix with counts, percentages, 
    row summaries (Precision/FDR), column summaries (Recall/FNR), and overall accuracy.
    """
    cm = confusion_matrix(y_true, y_pred)
    n_classes = cm.shape[0]
    total_samples = np.sum(cm)
    
    cm_percentages = cm / total_samples
    
    col_sums = np.sum(cm, axis=0) # Total predicted for each class
    row_sums = np.sum(cm, axis=1) # Total true for each class
    
    precision = np.zeros(n_classes)
    fdr = np.zeros(n_classes)
    for i in range(n_classes):
        if col_sums[i] > 0:
            precision[i] = cm[i, i] / col_sums[i]
            fdr[i] = 1 - precision[i]
            
    recall = np.zeros(n_classes)
    fnr = np.zeros(n_classes)
    for i in range(n_classes):
        if row_sums[i] > 0:
            recall[i] = cm[i, i] / row_sums[i]
            fnr[i] = 1 - recall[i]
            
    accuracy = np.trace(cm) / total_samples
    
    fig, ax = plt.subplots(figsize=(n_classes + 3, n_classes + 2.5))
    
    cmap_diag = sns.color_palette("Greens", as_cmap=True)
    cmap_offdiag = sns.color_palette("Reds", as_cmap=True)
    
    ax.set_xlim(0, n_classes + 1)
    ax.set_ylim(0, n_classes + 1)
    ax.invert_yaxis() 
    
    for i in range(n_classes):
        for j in range(n_classes):
            count = cm[i, j]
            percent = cm_percentages[i, j]
            
            if i == j:
                color = cmap_diag(percent * 2) 
            else:
                color = cmap_offdiag(percent * 2)
                
            rect = plt.Rectangle((j, i), 1, 1, facecolor=color, edgecolor='white', linewidth=2)
            ax.add_patch(rect)
            
            text_color = 'white' if percent > 0.3 else 'black'
            ax.text(j + 0.5, i + 0.5, f"{count}\n({percent:.1%})", 
                    ha='center', va='center', color=text_color, fontsize=10, fontweight='bold')
            
    for i in range(n_classes):
        rect = plt.Rectangle((n_classes, i), 1, 1, facecolor='#f0f0f0', edgecolor='white', linewidth=2)
        ax.add_patch(rect)
        ax.text(n_classes + 0.5, i + 0.5, f"Recall: {recall[i]:.1%}\nFNR: {fnr[i]:.1%}", 
                ha='center', va='center', color='black', fontsize=9)
                
    for j in range(n_classes):
        rect = plt.Rectangle((j, n_classes), 1, 1, facecolor='#f0f0f0', edgecolor='white', linewidth=2)
        ax.add_patch(rect)
        ax.text(j + 0.5, n_classes + 0.5, f"Prec: {precision[j]:.1%}\nFDR: {fdr[j]:.1%}", 
                ha='center', va='center', color='black', fontsize=9)
                
    rect = plt.Rectangle((n_classes, n_classes), 1, 1, facecolor='#e0e0e0', edgecolor='white', linewidth=2)
    ax.add_patch(rect)
    ax.text(n_classes + 0.5, n_classes + 0.5, f"Accuracy\n{accuracy:.1%}", 
            ha='center', va='center', color='black', fontsize=10, fontweight='bold')
            
    ax.set_xticks(np.arange(n_classes) + 0.5)
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_yticks(np.arange(n_classes) + 0.5)
    ax.set_yticklabels(labels, fontsize=11, rotation=90, va='center')
    
    ax.set_xlabel("Predicted Label", fontsize=12, fontweight='bold', labelpad=15)
    ax.xaxis.set_label_position('top')
    ax.xaxis.tick_top()
    
    ax.set_ylabel("True Label", fontsize=12, fontweight='bold', labelpad=15)
    
    plt.title(title, fontsize=14, fontweight='bold', pad=30)
    
    for spine in ax.spines.values():
        spine.set_visible(False)
        
    ax.tick_params(axis='both', which='both',length=0)
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
