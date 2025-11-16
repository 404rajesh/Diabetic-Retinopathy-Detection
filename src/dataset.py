# src/dataset.py
import os
import torch
from torch.utils.data import Dataset
import albumentations as A
import cv2
import numpy as np

class DRDataset(Dataset):
    def __init__(self, df, image_dir, transforms=None):
        """
        df: pandas DataFrame with columns id_code and diagnosis
        image_dir: folder path where processed images exist
        transforms: albumentations pipeline
        """
        self.df = df.reset_index(drop=True)
        self.image_dir = image_dir
        self.transforms = transforms

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.image_dir, f"{row['id_code']}.png")
        img = cv2.imread(img_path)
        if img is None:
            # fallback zero image
            img = np.zeros((224,224,3), dtype=np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if self.transforms:
            img = self.transforms(image=img)['image']
        # transpose to channel-first and scale to [0,1]
        img = img.astype(np.float32) / 255.0
        img = np.transpose(img, (2,0,1))
        label = int(row['diagnosis'])
        return torch.tensor(img, dtype=torch.float32), torch.tensor(label, dtype=torch.long)

def get_transforms(train=True, size=224):
    if train:
        return A.Compose([
            A.Resize(size, size),
            A.RandomRotate90(),
            A.HorizontalFlip(),
            A.VerticalFlip(),
            A.OneOf([
                A.RandomBrightnessContrast(p=0.5),
                A.CLAHE(p=0.5),
                A.RandomGamma(p=0.5),
            ], p=0.7),
        ])
    else:
        return A.Compose([
            A.Resize(size, size),
        ])
