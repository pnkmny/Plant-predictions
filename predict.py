# predict.py
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import pandas as pd
import numpy as np
from model import PlantClassifier  # make sure this matches your model

# Load model + label encoder
def load_model(model_path="plant_classifier.pth", classes_path="species_classes.csv", device=None):
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    classes = pd.read_csv(classes_path, header=None).squeeze().tolist()
    num_classes = len(classes)

    model = PlantClassifier(num_meta_features=3, num_classes=num_classes)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    return model, classes, device

# Predict function
def predict(image, latitude, longitude, doy, model, classes, device, topk=5):
    """
    image: either a PIL.Image.Image or a torch.Tensor [C,H,W]
    """
    # Preprocessing
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    if isinstance(image, Image.Image):
        image = transform(image)
    elif torch.is_tensor(image):
        if len(image.shape) == 3:
            # Already [C,H,W], add batch dimension
            image = image.unsqueeze(0)
        elif len(image.shape) == 4:
            pass  # batch already exists
        else:
            raise ValueError(f"Unexpected tensor shape: {image.shape}")
    else:
        raise TypeError("image must be PIL.Image or torch.Tensor")

    image = image.to(device)

    # Metadata
    meta = torch.tensor([[latitude, longitude, doy]], dtype=torch.float32).to(device)

    # Forward pass
    with torch.no_grad():
        outputs = model(image, meta)
        probs = torch.softmax(outputs, dim=1).cpu().numpy().flatten()

    # Top-k predictions
    topk_idx = np.argsort(probs)[::-1][:topk]
    predictions = [(classes[i], float(probs[i])) for i in topk_idx]

    return predictions

# Example CLI usage
if __name__ == "__main__":
    import sys
    from torchvision.transforms.functional import to_pil_image

    if len(sys.argv) != 5:
        print("Usage: python predict.py <image_path> <latitude> <longitude> <doy>")
        sys.exit(1)

    image_path = sys.argv[1]
    latitude = float(sys.argv[2])
    longitude = float(sys.argv[3])
    doy = int(sys.argv[4])

    model, classes, device = load_model()
    image = Image.open(image_path).convert("RGB")
    preds = predict(image, latitude, longitude, doy, model, classes, device)

    print("\nTop predictions:")
    for species, prob in preds:
        print(f"{species}: {prob:.4f}")
