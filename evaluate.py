import torch
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from preprocess import get_dataloaders
from model import get_model
from utils import load_model, decode_labels

# Config
csv_file = 'data/labels.csv'
img_dir = 'data/images'
batch_size = 32
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model_path = 'plant_model.pth'

# Load data
train_loader, val_loader = get_dataloaders(csv_file, img_dir, batch_size=batch_size)
df = pd.read_csv(csv_file)
num_classes = df['species'].nunique()

# Load model
model = get_model(num_classes)
model = load_model(model, model_path, device=device)
model.to(device)

# Evaluation
all_preds = []
all_labels = []

model.eval()
with torch.no_grad():
    for images, labels in val_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

# Classification report
print(classification_report(all_labels, all_preds))

# Confusion matrix
cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(10,8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()
