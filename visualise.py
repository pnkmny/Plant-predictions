import torch
import matplotlib.pyplot as plt
from utils import decode_labels
import numpy as np

def show_predictions(model, dataloader, le, device='cpu', n=5):
    model.eval()
    images_shown = 0
    plt.figure(figsize=(15,5))
    
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            
            for i in range(len(images)):
                if images_shown >= n:
                    break
                img = images[i].cpu().permute(1,2,0).numpy()
                true_label = le.inverse_transform([labels[i].cpu().item()])[0]
                pred_label = le.inverse_transform([preds[i].cpu().item()])[0]
                
                plt.subplot(1, n, images_shown+1)
                plt.imshow(img)
                plt.title(f'True: {true_label}\nPred: {pred_label}')
                plt.axis('off')
                images_shown += 1
            if images_shown >= n:
                break
    plt.show()
