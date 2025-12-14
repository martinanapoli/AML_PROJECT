import os
import random

import numpy as np
from PIL import Image, ImageOps

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms


DATA_DIR = "/Users/martu/Desktop/dataset2"
OUTPUT_DIR = "/Users/martu/Desktop/outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")

BATCH_SIZE = 32
NUM_EPOCHS = 20
LR = 1e-4
IMG_SIZE = 256

NUM_WORKERS = 0

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


set_seed(42)


def is_image_file(path: str) -> bool:
    return path.lower().endswith(IMAGE_EXTENSIONS)


def list_images_in_dir(directory: str):
  
    paths = []
    for root, _, files in os.walk(directory):
        for f in files:
            if is_image_file(f):
                paths.append(os.path.join(root, f))
    return sorted(paths)


def train_val_split(paths, val_ratio=0.2):
  
    paths = list(paths)
    random.shuffle(paths)
    n_total = len(paths)
    n_val = int(n_total * val_ratio)
    val_paths = paths[:n_val]
    train_paths = paths[n_val:]
    return train_paths, val_paths



# TRANSFORMACIONES

base_to_tensor = transforms.Compose([
    transforms.ToTensor(), # [0,1]
    transforms.Normalize(mean=[0.5], std=[0.5])  # [-1,1]
])


def random_small_perturbation(img: Image.Image) -> Image.Image:
    angle = random.uniform(-5, 5)
    img = img.rotate(angle, resample=Image.BILINEAR)

    arr = np.array(img).astype(np.float32)
    arr = arr / 255.0
    brightness = random.uniform(0.9, 1.1)
    contrast = random.uniform(0.9, 1.1)
    mean = arr.mean()
    arr = (arr - mean) * contrast + mean
    arr = arr * brightness
    arr = np.clip(arr, 0, 1)
    arr = (arr * 255).astype(np.uint8)
    img = Image.fromarray(arr)

    return img


def structural_corruption(img: Image.Image) -> Image.Image:
    w, h = img.size
    op = random.choice([
        "flip_lr",
        "flip_ud",
        "swap_halves",
        "shuffle_quadrants",
        "rotate_90",
        "rotate_270"
    ])

    if op == "flip_lr":
        img = ImageOps.mirror(img)

    elif op == "flip_ud":
        img = ImageOps.flip(img)

    elif op == "swap_halves":
        mid = w // 2
        left = img.crop((0, 0, mid, h))
        right = img.crop((mid, 0, w, h))
        new_img = Image.new("L", (w, h))
        new_img.paste(right, (0, 0))
        new_img.paste(left, (w - mid, 0))
        img = new_img

    elif op == "shuffle_quadrants":
        mid_w = w // 2
        mid_h = h // 2
        q1 = img.crop((0, 0, mid_w, mid_h))
        q2 = img.crop((mid_w, 0, w, mid_h))
        q3 = img.crop((0, mid_h, mid_w, h))
        q4 = img.crop((mid_w, mid_h, w, h))

        quadrants = [q1, q2, q3, q4]
        random.shuffle(quadrants)

        new_img = Image.new("L", (w, h))
        new_img.paste(quadrants[0], (0, 0))
        new_img.paste(quadrants[1], (mid_w, 0))
        new_img.paste(quadrants[2], (0, mid_h))
        new_img.paste(quadrants[3], (mid_w, mid_h))
        img = new_img

    elif op == "rotate_90":
        img = img.rotate(90, expand=True)
        img = img.resize((w, h), resample=Image.BILINEAR)

    elif op == "rotate_270":
        img = img.rotate(270, expand=True)
        img = img.resize((w, h), resample=Image.BILINEAR)

    return img


# DATASET
# Input: imagen (con o sin corrupcion estructural) 
# Label: 0 = anatomia correcta, 1 = anatomia corrupta

class AnatomicalConsistencyDataset(Dataset):

    def __init__(self, image_paths, img_size=256):
        self.image_paths = image_paths
        self.img_size = img_size

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]

        img = Image.open(img_path).convert("L")  # escala de grises
        img = img.resize((self.img_size, self.img_size), resample=Image.BILINEAR)

        label = random.randint(0, 1)  # 0 = correcta, 1 = corrupta

        if label == 0:
            img = random_small_perturbation(img)
        else:
            img = structural_corruption(img)

        tensor = base_to_tensor(img)  # [1, H, W] normalizado [-1,1]

        return tensor, label


# MODELO

class AnatomicalConsistencyModel(nn.Module):
    def __init__(self):
        super().__init__()
        backbone = models.resnet18(weights=None)
        backbone.conv1 = nn.Conv2d(
            1, 64,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False
        )

        num_feats = backbone.fc.in_features
        backbone.fc = nn.Identity()

        self.encoder = backbone
        self.classifier = nn.Linear(num_feats, 2)

    def forward(self, x):
        feats = self.encoder(x)
        logits = self.classifier(feats)
        return logits

    def encode(self, x):
        with torch.no_grad():
            feats = self.encoder(x)
        return feats


# TRAINING LOOP

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

        preds = torch.argmax(logits, dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_acc = correct / total

    return epoch_loss, epoch_acc


def eval_one_epoch(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            logits = model(images)
            loss = criterion(logits, labels)

            running_loss += loss.item() * images.size(0)
            preds = torch.argmax(logits, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_acc = correct / total

    return epoch_loss, epoch_acc


# MAIN

def main():
    print(f"Usando dispositivo: {DEVICE}")
    print(f"Carpeta base (dataset1): {DATA_DIR}  -> existe: {os.path.exists(DATA_DIR)}")

    all_paths = list_images_in_dir(DATA_DIR)
    print(f"Nº total de imágenes encontradas en dataset1: {len(all_paths)}")

    if len(all_paths) == 0:
        print("⚠ No se han encontrado imágenes en DATA_DIR.")
        print("   Revisa que:")
        print(f"   - La carpeta existe: {DATA_DIR}")
        print("   - Las imágenes tienen extensión:", IMAGE_EXTENSIONS)
        print("   - Están dentro de subcarpetas con cualquier nombre.")
        return

    # Split train/val
    train_paths, val_paths = train_val_split(all_paths, val_ratio=0.2)

    print(f"Nº imágenes train: {len(train_paths)}")
    print(f"Nº imágenes val:   {len(val_paths)}")

    print("Ejemplos de imágenes train:")
    for p in train_paths[:5]:
        print("  ", p)

    train_dataset = AnatomicalConsistencyDataset(train_paths, img_size=IMG_SIZE)
    val_dataset = AnatomicalConsistencyDataset(val_paths, img_size=IMG_SIZE)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=False
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=False
    )

    model = AnatomicalConsistencyModel().to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    best_val_acc = 0.0
    best_model_path = os.path.join(OUTPUT_DIR, "pretext3_dataset1.pth")

    for epoch in range(1, NUM_EPOCHS + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, DEVICE)
        val_loss, val_acc = eval_one_epoch(model, val_loader, criterion, DEVICE)

        print(f"\nEpoch {epoch}/{NUM_EPOCHS}")
        print(f"  Train  | loss: {train_loss:.4f}  acc: {train_acc:.4f}")
        print(f"  Val    | loss: {val_loss:.4f}  acc: {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_dict = {
                "encoder_state_dict": model.encoder.state_dict(),
                "classifier_state_dict": model.classifier.state_dict(),
                "config": {
                    "img_size": IMG_SIZE,
                    "architecture": "resnet18",
                    "task": "anatomical_structure_consistency",
                }
            }
            torch.save(save_dict, best_model_path)
            print(f"  ✓ Nuevo mejor modelo guardado en: {best_model_path}")

    print("\nEntrenamiento finalizado.")
    print(f"Mejor accuracy de validación: {best_val_acc:.4f}")
    print(f"Modelo final guardado en: {best_model_path}")


if __name__ == "__main__":
    main()
