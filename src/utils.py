# src/utils.py
import os
import numpy as np
import torch
from sklearn.metrics import cohen_kappa_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

def save_checkpoint(state, filename):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    torch.save(state, filename)

def load_checkpoint(filename, model, optimizer=None, scaler=None):
    checkpoint = torch.load(filename, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    if scaler and 'scaler_state_dict' in checkpoint:
        scaler.load_state_dict(checkpoint['scaler_state_dict'])
    return checkpoint

def compute_qwk(y_true, y_pred):
    return cohen_kappa_score(y_true, y_pred, weights='quadratic')

def plot_confusion(y_true, y_pred, out_path=None):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel("Pred")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    if out_path:
        plt.savefig(out_path)
    plt.close()
