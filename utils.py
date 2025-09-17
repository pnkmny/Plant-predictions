import torch
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

# Model save/load
def save_model(model, path='plant_model.pth'):
    torch.save(model.state_dict(), path)

def load_model(model, path='plant_model.pth', device='cpu'):
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()
    return model

# Label encoding
def encode_labels(df, column='species'):
    le = LabelEncoder()
    df[column] = le.fit_transform(df[column])
    return df, le

def decode_labels(labels, le):
    return le.inverse_transform(labels)

# Plot training curves
def plot_curves(train_losses, val_losses, train_accs=None, val_accs=None):
    plt.figure(figsize=(12,5))
    
    # Loss curve
    plt.subplot(1,2,1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss Curve')
    plt.legend()
    
    # Accuracy curve
    if train_accs and val_accs:
        plt.subplot(1,2,2)
        plt.plot(train_accs, label='Train Acc')
        plt.plot(val_accs, label='Val Acc')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.title('Accuracy Curve')
        plt.legend()
    
    plt.show()
