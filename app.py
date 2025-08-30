import os
import random
import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.transforms as transforms
from sklearn.metrics import roc_auc_score, accuracy_score


# -------------------------------
# Dataset
# -------------------------------
class SignatureDataset(Dataset):
    def __init__(self, dataset_path, transform=None):
        self.dataset_path = dataset_path
        self.transform = transform

        self.genuine_path = os.path.join(dataset_path, "full_org")
        self.forged_path = os.path.join(dataset_path, "full_forg")

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

        if img1_name in self.forged_images:
            img1 = Image.open(os.path.join(self.forged_path, img1_name)).convert("L")
        else:
            img1 = Image.open(os.path.join(self.genuine_path, img1_name)).convert("L")

        if img2_name in self.forged_images:
            img2 = Image.open(os.path.join(self.forged_path, img2_name)).convert("L")
        else:
            img2 = Image.open(os.path.join(self.genuine_path, img2_name)).convert("L")

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
            nn.Conv2d(1, 32, kernel_size=5),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(32, 64, kernel_size=5),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(64, 128, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )

        # Flattened size = 128*16*25 = 51200
        self.fc = nn.Sequential(
            nn.Linear(128 * 16 * 25, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 128)
        )

    def forward_once(self, x):
        x = self.cnn(x)
        x = x.view(x.size(0), -1)
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
# Evaluation
# -------------------------------
def evaluate(model, dataloader, device, threshold=1.0):
    model.eval()
    all_labels = []
    all_preds = []
    distances = []

    with torch.no_grad():
        for img1, img2, label in dataloader:
            img1, img2 = img1.to(device), img2.to(device)
            out1, out2 = model(img1, img2)
            dist = F.pairwise_distance(out1, out2)

            pred = (dist < threshold).float()

            all_labels.extend(label.cpu().numpy())
            all_preds.extend(pred.cpu().numpy())
            distances.extend(-dist.cpu().numpy())  # invert for AUC

    acc = accuracy_score(all_labels, np.round(all_preds))
    auc = roc_auc_score(all_labels, distances)
    return acc, auc


# -------------------------------
# Training
# -------------------------------
def train_model(dataset_path, num_epochs=20, batch_size=16, lr=0.0005, device=None):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    transform = transforms.Compose([
        transforms.Resize((155, 220)),
        transforms.ToTensor()
    ])

    dataset = SignatureDataset(dataset_path, transform=transform)

    # Train/Test split (80/20)
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    model = SiameseNetwork().to(device)
    criterion = ContrastiveLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # ✅ Make checkpoints dir
    checkpoint_dir = "checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)

    best_auc = 0.0

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0

        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        for img1, img2, label in loop:
            img1, img2, label = img1.to(device), img2.to(device), label.to(device)

            optimizer.zero_grad()
            out1, out2 = model(img1, img2)
            loss = criterion(out1, out2, label)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            loop.set_postfix(loss=loss.item())

        avg_loss = epoch_loss / len(train_loader)

        # Evaluate
        train_acc, train_auc = evaluate(model, train_loader, device)
        test_acc, test_auc = evaluate(model, test_loader, device)

        print(f"Epoch [{epoch+1}/{num_epochs}] "
              f"- Loss: {avg_loss:.4f} "
              f"- Train Acc: {train_acc:.4f}, Train AUC: {train_auc:.4f} "
              f"- Test Acc: {test_acc:.4f}, Test AUC: {test_auc:.4f}")

        # ✅ Save every epoch
        torch.save(model.state_dict(), os.path.join(checkpoint_dir, f"siamese_epoch{epoch+1}.pth"))

        # ✅ Save best model
        if test_auc > best_auc:
            best_auc = test_auc
            torch.save(model.state_dict(), os.path.join(checkpoint_dir, "siamese_best.pth"))
            print(f"✅ Saved best model at epoch {epoch+1} with Test AUC: {best_auc:.4f}")


# -------------------------------
# Run Training
# -------------------------------
if __name__ == "__main__":
    dataset_path = "signatures_dataset"  # adjust if needed
    train_model(dataset_path, num_epochs=20, batch_size=16, lr=0.0005)
