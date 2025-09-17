import torch
import pandas as pd
from train import model, val_loader, species_to_idx  # make sure these exist
from predict import load_model, predict  # optional for testing

# ------------------------------
# 1️⃣ Save trained model
torch.save(model.state_dict(), "plant_classifier.pth")
print("✅ Model saved as plant_classifier.pth")

# ------------------------------
# 2️⃣ Save species/classes
classes = list(species_to_idx.keys())
pd.DataFrame(classes).to_csv("species_classes.csv", index=False, header=False)
print("✅ Classes saved as species_classes.csv")

# ------------------------------
# 3️⃣ Quick validation accuracy
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.eval()
correct = 0
total = 0

with torch.no_grad():
    for images, labels, meta in val_loader:  # include meta if your model uses it
        images, labels, meta = images.to(device), labels.to(device), meta.to(device)
        outputs = model(images, meta)
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

print(f"Validation Accuracy: {correct/total:.4f}")

# ------------------------------
# 4️⃣ Optional: Test single prediction
example_image = "images/314197882.jpg"  # replace with a real path
latitude, longitude, doy = 43.5, -79.3, 120

model, classes, device = load_model()
preds = predict(example_image, latitude, longitude, doy, model, classes, device)

print("\nTop predictions for example image:")
for species, prob in preds:
    print(f"{species}: {prob:.4f}")
