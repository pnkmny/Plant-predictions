# run_all.py
import os
import random
import torch
import pandas as pd
from train import get_dataloaders
from model import PlantClassifier
from predict import load_model, predict
from torchvision.transforms.functional import to_pil_image

# ------------------------------
# Config
csv_file = "inat_metadata.csv"
img_dir = "images"
batch_size = 32
model_path = "plant_classifier.pth"
classes_path = "species_classes.csv"
epochs = 5

# ------------------------------
# 1️⃣ Load dataloaders
train_loader, val_loader, species_to_idx = get_dataloaders(csv_file, img_dir, batch_size=batch_size)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_classes = len(species_to_idx)
model = PlantClassifier(num_meta_features=3, num_classes=num_classes).to(device)

# ------------------------------
# 2️⃣ Train the model (skip if exists)
if os.path.exists(model_path):
    print("✅ Model already exists, skipping training.")
    model.load_state_dict(torch.load(model_path, map_location=device))
else:
    print("⏳ Training started...")
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for batch in train_loader:
            # Unpack batch correctly
            if len(batch) == 3:
                images, labels, meta = batch
                meta = meta.to(device)
            else:
                images, labels = batch
                meta = None

            images = images.to(device)
            labels = labels.to(device)

            # Forward + backward
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
            criterion = torch.nn.CrossEntropyLoss()
            optimizer.zero_grad()
            outputs = model(images, meta) if meta is not None else model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)

        epoch_loss = running_loss / len(train_loader.dataset)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}")

    torch.save(model.state_dict(), model_path)
    pd.DataFrame(list(species_to_idx.keys())).to_csv(classes_path, index=False, header=False)
    print(f"✅ Training complete and model saved to {model_path}")
    print(f"✅ Classes saved to {classes_path}")

# For validation
with torch.no_grad():
    for images, labels in val_loader:
        images = images.to(device)
        labels_encoded = torch.tensor([species_to_idx[label] for label in labels], dtype=torch.long).to(device)

        # real metadata
        meta_list = []
        for label in labels:
            row = df[df["species"] == label].iloc[0]
            lat = row["lat"]
            lon = row["lon"]
            doy = pd.to_datetime(row["date"]).dayofyear
            meta_list.append([lat, lon, doy])
        meta = torch.tensor(meta_list, dtype=torch.float).to(device)

        outputs = model(images, meta)
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels_encoded).sum().item()
        total += labels_encoded.size(0)

# For sample prediction
if len(sample_item) == 3:
    sample_image, sample_label, _ = sample_item
    row = df[df["species"] == sample_label].iloc[0]
    latitude = row["lat"]
    longitude = row["lon"]
    doy = pd.to_datetime(row["date"]).dayofyear
else:
    sample_image, sample_label = sample_item
    latitude, longitude, doy = 0.0, 0.0, 0  # fallback


# Convert meta to numpy safely
if sample_meta is not None:
    meta_tensor = sample_meta if torch.is_tensor(sample_meta) else torch.tensor(sample_meta)
    latitude, longitude, doy = meta_tensor.numpy()
else:
    latitude, longitude, doy = 0.0, 0.0, 0  # dummy

# Convert image tensor to PIL
image_pil = to_pil_image(sample_image)

# Load model and predict
model, classes, device = load_model(model_path, classes_path)
preds = predict(image_pil, latitude, longitude, doy, model, classes, device)

print("\nTop predictions for random validation sample:")
for species, prob in preds:
    print(f"{species}: {prob:.4f}")
