import pandas as pd
import os
import imageio
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import requests
from PIL import Image
import numpy as np

class PlantDataset(Dataset):
    def __init__(self, df, img_dir, transform=None):
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.img_dir, f"{row['id']}.jpg")

        image = Image.open(img_path).convert("RGB")  # ensures 3 channels
        label = row['species']
        if self.transform:
            image = self.transform(image)  # returns [3,H,W] tensor already

        return image, label
    
def get_dataloaders(csv_file, img_dir, batch_size=32, test_size=0.2, random_state=42):
    df = pd.read_csv(csv_file)
    print("✅ Loaded CSV from:", csv_file)
    print("Shape of df:", df.shape)
    # Create image paths
    df['img_path'] = df['id'].apply(lambda x: os.path.join("images", f"{x}.jpg"))

    # Filter out rows with missing images
    df = df[df['img_path'].apply(os.path.exists)]
    print(f"✅ Filtered dataset, remaining samples: {len(df)}")

    # Train/val split
    try:
        train_df, val_df = train_test_split(
            df, test_size=test_size, stratify=df['species'], random_state=random_state
        )
    except ValueError:
        train_df, val_df = train_test_split(df, test_size=test_size, random_state=random_state)

    # Transforms
    transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
])

    train_dataset = PlantDataset(train_df, img_dir, transform=transform)
    val_dataset = PlantDataset(val_df, img_dir, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Species to index mapping
    all_species = sorted(df['species'].unique())
    species_to_idx = {s: i for i, s in enumerate(all_species)}

    return train_loader, val_loader, species_to_idx
