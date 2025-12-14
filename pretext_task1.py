import os
import random
from glob import glob
import shutil  
from collections import defaultdict

from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from tqdm import tqdm


DATA_DIR = "/Users/martu/Desktop/dataset3"  
OUTPUT_DIR = "/Users/martu/Desktop/OUTPUS_DEF"
os.makedirs(OUTPUT_DIR, exist_ok=True)

IMG_SIZE = 224
BATCH_SIZE = 64
EPOCHS = 20
LR = 1e-4
SEED = 42
NUM_WORKERS = 0   
VAL_SPLIT = 0.2  # 80% train / 20% val por carpeta
EARLY_STOPPING_PATIENCE = 5  

# Device: CUDA > MPS > CPU
device = (
    torch.device("cuda")
    if torch.cuda.is_available()
    else torch.device("mps")
    if torch.backends.mps.is_available()
    else torch.device("cpu")
)

random.seed(SEED)
torch.manual_seed(SEED)


# CLEAN OUTPUT_DIR
def limpiar_output_dir(path):

    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path, exist_ok=True)



# ROTATION PREDICTION
class RotationDataset(Dataset):


    def __init__(self, image_paths, transform=None):
        self.image_paths = image_paths
        self.transform = transform

        if len(self.image_paths) == 0:
            raise ValueError("No se encontraron imagenes en la lista proporcionada")

        self.angles = [0, 90, 180, 270]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]

        img = Image.open(img_path).convert("RGB")

        angle_idx = random.randint(0, 3)
        angle = self.angles[angle_idx]

        img = img.rotate(angle)

        if self.transform:
            img = self.transform(img)

        label = angle_idx
        return img, label


# TRANSFORMATIONS (data augmentation : contraste + ruido)

rotation_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ColorJitter(contrast=(0.7, 1.3)),
    transforms.ToTensor(),
    transforms.Lambda(lambda x: torch.clamp(
        x + 0.02 * torch.randn_like(x),  # gaussiano
        0.0, 1.0
    )),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# DATASET Y DATALOADER

def get_dataloaders():
    exts = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.gif")
    all_paths = []
    for ext in exts:
        all_paths.extend(glob(os.path.join(DATA_DIR, "**", ext), recursive=True))

    if len(all_paths) == 0:
        raise ValueError(f"No se encontraron imagenes en {DATA_DIR}")

    paths_por_carpeta = defaultdict(list)
    for p in all_paths:
        rel_dir = os.path.relpath(os.path.dirname(p), DATA_DIR)
        tema = rel_dir.split(os.sep)[0]
        paths_por_carpeta[tema].append(p)

    train_paths = []
    val_paths = []
    rnd = random.Random(SEED)

    print("\nSplit estratificado 80/20 por carpeta:")
    for tema, paths in paths_por_carpeta.items():
        rnd.shuffle(paths)
        n_total = len(paths)
        n_val = int(n_total * VAL_SPLIT)
        n_train = n_total - n_val

        train_paths.extend(paths[:n_train])
        val_paths.extend(paths[n_train:])

        print(f"  Tema '{tema}': total={n_total}, train={n_train}, val={n_val}")

    print(f"\nTotal imgenes train: {len(train_paths)}")
    print(f"Total imgenes val:   {len(val_paths)}")

    train_ds = RotationDataset(train_paths, transform=rotation_transform)
    val_ds = RotationDataset(val_paths, transform=rotation_transform)

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS
    )

    return train_loader, val_loader


# MODELO (ResNet18 para 4 clases)

def build_model():
    model = models.resnet18(pretrained=True)

    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 4)

    for param in model.parameters():
        param.requires_grad = False
    for param in model.fc.parameters():
        param.requires_grad = True

    return model


# LOOP DE ENTRENAMIENTO

def train_one_epoch(model, loader, criterion, optimizer, epoch):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(loader, desc=f"Epoch {epoch} [Train]", leave=False)
    for images, labels in pbar:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

        pbar.set_postfix({
            "loss": f"{loss.item():.4f}",
            "acc": f"{(correct / total) * 100:.2f}%"
        })

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


def validate_one_epoch(model, loader, criterion, epoch):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        pbar = tqdm(loader, desc=f"Epoch {epoch} [Val]", leave=False)
        for images, labels in pbar:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)

            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            pbar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "acc": f"{(correct / total) * 100:.2f}%"
            })

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


# MAIN

def main():
    print(f"Usando dispositivo: {device}")

    print(f"Limpiando carpeta de outputs: {OUTPUT_DIR}")
    limpiar_output_dir(OUTPUT_DIR)

    train_loader, val_loader = get_dataloaders()
    model = build_model().to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    warmup_epochs = 3
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LR,
        weight_decay=1e-4
    )

    best_val_acc = 0.0
    best_model_path = os.path.join(OUTPUT_DIR, "pretext1_dataset1.pth")

    # Early stopping
    epochs_without_improve = 0

    for epoch in range(1, EPOCHS + 1):

        if epoch == warmup_epochs + 1:
            print("\nDescongelando todas las capas para fine-tuning completo...")
            for param in model.parameters():
                param.requires_grad = True

            optimizer = optim.Adam(
                model.parameters(),
                lr=LR / 5,
                weight_decay=1e-4
            )

        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, epoch)
        val_loss, val_acc = validate_one_epoch(model, val_loader, criterion, epoch)

        print(f"\nEpoch {epoch}/{EPOCHS}")
        print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.2f}%")
        print(f"  Val   Loss: {val_loss:.4f} | Val   Acc: {val_acc*100:.2f}%")

        # Comprobamos mejora para early stopping
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            epochs_without_improve = 0
            torch.save(model.state_dict(), best_model_path)
            print(f"  -> Nuevo mejor modelo guardado en: {best_model_path}")
        else:
            epochs_without_improve += 1
            print(f"  -> Sin mejora en val acc. Epochs sin mejorar: {epochs_without_improve}/{EARLY_STOPPING_PATIENCE}")

        # Condición de early stopping
        if epochs_without_improve >= EARLY_STOPPING_PATIENCE:
            print(f"\nEarly stopping activado tras {epochs_without_improve} épocas sin mejora.")
            break

    print(f"\nEntrenamiento terminado. Mejor val acc: {best_val_acc*100:.2f}%")
    print(f"Mejor modelo guardado en: {best_model_path}")

if __name__ == "__main__":
    main()
 