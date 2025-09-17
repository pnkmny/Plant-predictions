import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from preprocess import get_dataloaders
from model import PlantClassifier
from datetime import datetime

# Paths
csv_file = "inat_metadata.csv"
img_dir = "images"
batch_size = 32
num_epochs = 5
learning_rate = 1e-3

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load data
df = pd.read_csv(csv_file)
train_loader, val_loader, species_to_idx = get_dataloaders(csv_file, img_dir, batch_size=batch_size)
num_classes = len(species_to_idx)

# Map index to species
idx_to_species = {v: k for k, v in species_to_idx.items()}

# Encode labels function
def encode_labels(batch_labels):
    return torch.tensor([species_to_idx[label] for label in batch_labels], dtype=torch.long)

# Extract metadata tensor for a batch
def get_metadata_from_df(labels):
    meta_list = []
    for label in labels:
        row = df[df["species"] == label].iloc[0]  # take first occurrence
        lat = row["lat"]
        lon = row["lon"]
        doy = datetime.strptime(row["date"], "%Y-%m-%d").timetuple().tm_yday
        meta_list.append([lat, lon, doy])
    return torch.tensor(meta_list, dtype=torch.float).to(device)

# Model
model = PlantClassifier(num_meta_features=3, num_classes=num_classes).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        images = images.to(device)
        labels_encoded = encode_labels(labels).to(device)
        meta = get_metadata_from_df(labels)

        optimizer.zero_grad()
        outputs = model(images, meta)
        loss = criterion(outputs, labels_encoded)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * images.size(0)

    epoch_loss = running_loss / len(train_loader.dataset)
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}")
