import os
import random
import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from sklearn.metrics import roc_auc_score


# -------------------------------
# Dataset
# -------------------------------
class SignatureDataset(Dataset):
    def __init__(self, dataset_path, transform=None):
        self.dataset_path = dataset_path
        self.transform = transform

        self.genuine_path = os.path.join(dataset_path, "full_org")
        self.forged_path = os.path.join(dataset_path, "full_forg")

        # Only accept image file extensions
        valid_exts = [".png", ".jpg", ".jpeg", ".tif", ".tiff"]

        self.genuine_images = [
            f for f in os.listdir(self.genuine_path)
            if os.path.splitext(f)[1].lower() in valid_exts
        ]
        self.forged_images = [
            f for f in os.listdir(self.forged_path)
            if os.path.splitext(f)[1].lower() in valid_exts
        ]

        self.pairs = []
        self.labels = []

        # Genuine pairs (label 1)
        for i in range(len(self.genuine_images) - 1):
            self.pairs.append((self.genuine_images[i], self.genuine_images[i + 1]))
            self.labels.append(1)

        # Forged pairs (label 0)
        for i in range(len(self.genuine_images)):
            if i < len(self.forged_images):
                self.pairs.append((self.genuine_images[i], self.forged_images[i]))
                self.labels.append(0)

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        img1_name, img2_name = self.pairs[idx]
        label = self.labels[idx]

        # Load images from correct folder
        if img1_name in self.forged_images:
            img1 = Image.open(os.path.join(self.forged_path, img1_name)).convert("L")
        else:
            img1 = Image.open(os.path.join(self.genuine_path, img1_name)).convert("L")

        if img2_name in self.forged_images:
            img2 = Image.open(os.path.join(self.forged_path, img2_name)).convert("L")
        else:
            img2 = Image.open(os.path.join(self.genuine_path, img2_name)).convert("L")

        # Apply transforms
        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)

        return img1, img2, torch.tensor([label], dtype=torch.float32)


# -------------------------------
# Siamese Network
# -------------------------------
class SiameseNetwork(nn.Module):
    def __init__(self):
        super(SiameseNetwork, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5),  # (1, 155, 220) -> (32, 151, 216)
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),              # -> (32, 75, 108)

            nn.Conv2d(32, 64, kernel_size=5),  # -> (64, 71, 104)
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),                # -> (64, 35, 52)

            nn.Conv2d(64, 128, kernel_size=3),  # -> (128, 33, 50)
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)                  # -> (128, 16, 25)
        )

        # Flattened size = 128*16*25 = 51200
        self.fc = nn.Sequential(
            nn.Linear(128 * 16 * 25, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 128)
        )

    def forward_once(self, x):
        x = self.cnn(x)
        x = x.view(x.size(0), -1)  # flatten
        x = self.fc(x)
        return x

    def forward(self, x1, x2):
        output1 = self.forward_once(x1)
        output2 = self.forward_once(x2)
        return output1, output2


# -------------------------------
# Contrastive Loss
# -------------------------------
class ContrastiveLoss(nn.Module):
    def __init__(self, margin=2.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        euclidean_distance = F.pairwise_distance(output1, output2)
        loss = torch.mean(
            (label) * torch.pow(euclidean_distance, 2) +
            (1 - label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2)
        )
        return loss


# -------------------------------
# Training function
# -------------------------------
def train_model(dataset_path, num_epochs=20, batch_size=16, lr=0.0005, device=None):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    transform = transforms.Compose([
        transforms.Resize((155, 220)),
        transforms.ToTensor()
    ])

    dataset = SignatureDataset(dataset_path, transform=transform)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = SiameseNetwork().to(device)
    criterion = ContrastiveLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        all_labels = []
        all_preds = []

        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        for img1, img2, label in loop:
            img1, img2, label = img1.to(device), img2.to(device), label.to(device)

            optimizer.zero_grad()
            output1, output2 = model(img1, img2)
            loss = criterion(output1, output2, label)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

            # Collect preds for AUC
            distances = F.pairwise_distance(output1, output2).detach().cpu().numpy()
            all_preds.extend(-distances)  # invert so higher = more similar
            all_labels.extend(label.cpu().numpy())

            loop.set_postfix(loss=loss.item())

        avg_loss = epoch_loss / len(train_loader)
        auc = roc_auc_score(all_labels, all_preds)
        print(f"Epoch [{epoch+1}/{num_epochs}] - Loss: {avg_loss:.4f}, AUC: {auc:.4f}")

        # Save model
        torch.save(model.state_dict(), f"siamese_epoch{epoch+1}.pth")


# -------------------------------
# Run Training
# -------------------------------
if __name__ == "__main__":
    dataset_path = "signatures_dataset"  # <-- adjust if needed
    train_model(dataset_path, num_epochs=20, batch_size=16, lr=0.0005)
