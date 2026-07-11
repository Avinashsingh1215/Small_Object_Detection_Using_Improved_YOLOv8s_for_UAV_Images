import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

def plot_training_curves(csv_path, save_dir):
    df = pd.read_csv(csv_path)
    epochs = df['epoch']
    fig, axes = plt.subplots(2, 2, figsize=(15,10))
    axes[0,0].plot(epochs, df['train/box_loss'], label='Box Loss')
    axes[0,0].plot(epochs, df['train/cls_loss'], label='Cls Loss')
    axes[0,0].set_xlabel('Epoch')
    axes[0,0].set_ylabel('Loss')
    axes[0,0].legend()
    axes[0,0].set_title('Training Losses')
    
    axes[0,1].plot(epochs, df['metrics/precision(B)'], label='Precision')
    axes[0,1].plot(epochs, df['metrics/recall(B)'], label='Recall')
    axes[0,1].set_xlabel('Epoch')
    axes[0,1].set_ylabel('Score')
    axes[0,1].legend()
    axes[0,1].set_title('Precision & Recall')
    
    axes[1,0].plot(epochs, df['metrics/mAP50(B)'], label='mAP50')
    axes[1,0].plot(epochs, df['metrics/mAP50-95(B)'], label='mAP50-95')
    axes[1,0].set_xlabel('Epoch')
    axes[1,0].set_ylabel('mAP')
    axes[1,0].legend()
    axes[1,0].set_title('mAP Curves')
    
    axes[1,1].plot(epochs, df['lr/pg0'], label='Learning Rate')
    axes[1,1].set_xlabel('Epoch')
    axes[1,1].set_ylabel('LR')
    axes[1,1].legend()
    axes[1,1].set_title('Learning Rate')
    
    plt.tight_layout()
    plt.savefig(f"{save_dir}/training_curves.png")
    plt.close()
