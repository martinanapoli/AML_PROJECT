import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import os
import shutil
from glob import glob
from collections import Counter

import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import resnet18, ResNet18_Weights

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score


ROOT_DIR = r"C:\Users\clara\Desktop\4t IA\ADVANCED MACINE LEARNING\ChestX-ray14"
IMAGES_DIRS = [ os.path.join(ROOT_DIR, f"images_{i:03d}", "images") for i in range(1, 4)]

CSV_PATH = os.path.join(ROOT_DIR, "Data_Entry_2017.csv", "Data_Entry_2017.csv")
print("CSV_PATH:", CSV_PATH)

RESULTS_DIR = os.path.join(ROOT_DIR, "results_dowstream_alone")
if not os.path.exists(RESULTS_DIR):
    os.makedirs(RESULTS_DIR)

# Hyperparámetros
IMAGE_SIZE = 224
BATCH_SIZE = 32
NUM_EPOCHS = 10
LR = 1e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Usando dispositivo:", DEVICE)


df = pd.read_csv(CSV_PATH)

df = df.rename(columns={col: col.strip() for col in df.columns})  # limpiar espacios
assert "Image Index" in df.columns, "No encuentro columna 'Image Index' en el CSV"
assert "Finding Labels" in df.columns, "No encuentro columna 'Finding Labels' en el CSV"


all_image_paths = []
for d in IMAGES_DIRS:
    all_image_paths.extend(glob(os.path.join(d, "*.png")))
    all_image_paths.extend(glob(os.path.join(d, "*.jpg")))
    all_image_paths.extend(glob(os.path.join(d, "*.jpeg")))

basename_to_path = {os.path.basename(p): p for p in all_image_paths}

df = df[df["Image Index"].isin(basename_to_path.keys())].reset_index(drop=True)

# CONSTRUIR LABELS PARA LAS 3 TAREAS

all_labels = []
for text in df["Finding Labels"]:
    labs = text.split("|")
    for l in labs:
        if l != "No Finding":
            all_labels.append(l)

diseases = sorted(list(set(all_labels)))
print("Patologías (excluyendo 'No Finding'):", diseases)
num_diseases = len(diseases)

# Map disease -> index
disease_to_idx = {d: i for i, d in enumerate(diseases)}

# Crear columnas para las tareas
def get_binary_label(finding_labels: str):
    if finding_labels == "No Finding":
        return 0
    else:
        return 1

def get_three_class_label(finding_labels: str):
    if finding_labels == "No Finding":
        return 0
    labels = finding_labels.split("|")
    if "Pneumonia" in labels:
        return 1
    else:
        return 2

def get_multilabel_vector(finding_labels: str):
    vec = np.zeros(num_diseases, dtype=np.float32)
    if finding_labels == "No Finding":
        return vec
    labels = finding_labels.split("|")
    for l in labels:
        if l in disease_to_idx:
            vec[disease_to_idx[l]] = 1.0
    return vec

df["label_binary"] = df["Finding Labels"].apply(get_binary_label)
df["label_3cls"] = df["Finding Labels"].apply(get_three_class_label)
df["label_multilabel"] = df["Finding Labels"].apply(get_multilabel_vector)

# VISUALIZACIÓN 

def save_category_plots():
    # No Finding vs cualquier anomalía
    count_binary = df["label_binary"].value_counts().sort_index()
    plt.figure()
    plt.bar(["No Finding", "Anomalía"], count_binary.values)
    plt.title("Distribución binaria: sano vs anomalía")
    plt.ylabel("Frecuencia")
    plt.savefig(os.path.join(RESULTS_DIR, "binary_distribution.png"))
    plt.close()

    # Clases: sano vs neumonia vs otra
    count_3cls = df["label_3cls"].value_counts().sort_index()
    plt.figure()
    plt.bar(["Sano", "Neumonía", "Otra anomalía"], count_3cls.values)
    plt.title("Distribución 3 clases")
    plt.ylabel("Frecuencia")
    plt.savefig(os.path.join(RESULTS_DIR, "3class_distribution.png"))
    plt.close()

    # Cada patologia individual
    disease_counts = Counter()
    for labels in df["Finding Labels"]:
        if labels == "No Finding":
            continue
        for l in labels.split("|"):
            if l != "No Finding":
                disease_counts[l] += 1

    diseases_sorted = list(disease_counts.keys())
    values_sorted = [disease_counts[d] for d in diseases_sorted]

    plt.figure(figsize=(10, 5))
    plt.bar(diseases_sorted, values_sorted)
    plt.xticks(rotation=45, ha="right")
    plt.title("Frecuencia de cada patología (excluyendo 'No Finding')")
    plt.ylabel("Frecuencia")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "disease_frequencies.png"))
    plt.close()

save_category_plots()

# SPLIT TRAIN/VAL/TEST

train_df, temp_df = train_test_split(
    df,
    test_size=0.3,
    stratify=df["label_binary"],
    random_state=42,
)

val_df, test_df = train_test_split(
    temp_df,
    test_size=0.5,
    stratify=temp_df["label_binary"],
    random_state=42,
)

# DATASET Y DATALOADERS

train_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5]),
])

eval_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5]),
])

class ChestXrayDataset(Dataset):
    def __init__(self, df, label_type, transform=None):
        self.df = df.reset_index(drop=True)
        self.label_type = label_type
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_name = row["Image Index"]
        img_path = basename_to_path[img_name]

        img = Image.open(img_path).convert("L")  # imágenes de rayos X -> 1 canal

        if self.transform is not None:
            img = self.transform(img)

        # Convertimos de (1,H,W) a (3,H,W) para ResNet preentrenada
        img = img.repeat(3, 1, 1)

        if self.label_type == "binary":
            label = int(row["label_binary"])
            return img, label
        elif self.label_type == "3cls":
            label = int(row["label_3cls"])
            return img, label
        elif self.label_type == "multilabel":
            label = row["label_multilabel"]
            label = torch.tensor(label, dtype=torch.float32)
            return img, label
        else:
            raise ValueError("label_type no válido")

def make_loaders(label_type, batch_size=BATCH_SIZE):
    train_ds = ChestXrayDataset(train_df, label_type=label_type, transform=train_transform)
    val_ds   = ChestXrayDataset(val_df,   label_type=label_type, transform=eval_transform)
    test_ds  = ChestXrayDataset(test_df,  label_type=label_type, transform=eval_transform)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=0)

    return train_loader, val_loader, test_loader


# MODELO Y LOOPS DE ENTRENAMIENTO

def build_model(num_outputs, task_type):
   
    weights = ResNet18_Weights.DEFAULT
    model = resnet18(weights=weights)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_outputs)
    return model.to(DEVICE)

def train_epoch(model, loader, optimizer, criterion, task_type):
    model.train()
    total_loss = 0
    all_preds = []
    all_targets = []

    for imgs, labels in loader:
        imgs = imgs.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(imgs)

        if task_type == "ce":
            loss = criterion(outputs, labels)
            preds = outputs.argmax(dim=1).detach().cpu().numpy()
            targets = labels.detach().cpu().numpy()
        else:  # 'bce' multi-label
            loss = criterion(outputs, labels)
            preds = (torch.sigmoid(outputs) > 0.5).float().detach().cpu().numpy()
            targets = labels.detach().cpu().numpy()

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * imgs.size(0)
        all_preds.append(preds)
        all_targets.append(targets)

    avg_loss = total_loss / len(loader.dataset)
    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)

    if task_type == "ce":
        acc = accuracy_score(all_targets, all_preds)
        f1 = f1_score(all_targets, all_preds, average="weighted", zero_division=0)
    else:
        # Para multi-label calculamos F1 macro aproximado
        f1 = f1_score(all_targets, all_preds, average="macro", zero_division=0)
        acc = (all_preds == all_targets).mean()

    return avg_loss, acc, f1

def eval_epoch(model, loader, criterion, task_type):
    model.eval()
    total_loss = 0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(imgs)

            if task_type == "ce":
                loss = criterion(outputs, labels)
                preds = outputs.argmax(dim=1).detach().cpu().numpy()
                targets = labels.detach().cpu().numpy()
            else:  # 'bce'
                loss = criterion(outputs, labels)
                preds = (torch.sigmoid(outputs) > 0.5).float().detach().cpu().numpy()
                targets = labels.detach().cpu().numpy()

            total_loss += loss.item() * imgs.size(0)
            all_preds.append(preds)
            all_targets.append(targets)

    avg_loss = total_loss / len(loader.dataset)
    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)

    if task_type == "ce":
        acc = accuracy_score(all_targets, all_preds)
        f1 = f1_score(all_targets, all_preds, average="weighted", zero_division=0)
    else:
        f1 = f1_score(all_targets, all_preds, average="macro", zero_division=0)
        acc = (all_preds == all_targets).mean()

    return avg_loss, acc, f1

def train_task(label_type, num_outputs, task_type, num_epochs=NUM_EPOCHS):
    print(f"\nEntrenando tarea '{label_type}' con {num_outputs} salidas ({task_type})")

    train_loader, val_loader, test_loader = make_loaders(label_type)

    model = build_model(num_outputs, task_type=task_type)
    if task_type == "ce":
        criterion = nn.CrossEntropyLoss()
    else:
        criterion = nn.BCEWithLogitsLoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    best_val_f1 = -np.inf
    best_state = None

    for epoch in range(1, num_epochs + 1):
        train_loss, train_acc, train_f1 = train_epoch(model, train_loader, optimizer, criterion, task_type)
        val_loss, val_acc, val_f1 = eval_epoch(model, val_loader, criterion, task_type)

        print(f"Epoch {epoch:02d} | "
              f"Train loss: {train_loss:.4f} acc: {train_acc:.4f} f1: {train_f1:.4f} | "
              f"Val loss: {val_loss:.4f} acc: {val_acc:.4f} f1: {val_f1:.4f}")

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_state = model.state_dict()

    # Evaluar en test con el mejor modelo
    if best_state is not None:
        model.load_state_dict(best_state)

    test_loss, test_acc, test_f1 = eval_epoch(model, test_loader, criterion, task_type)
    print(f"\n>>> RESULTADOS TEST ({label_type}) <<<")
    print(f"Test loss: {test_loss:.4f} | acc: {test_acc:.4f} | f1: {test_f1:.4f}\n")

    return model

# ENTRENAR LAS 3 TAREAS (DOWNSTREAM FINAL)

# Tarea 1: binaria (anomalía vs no anomalía)
model_binary = train_task(
    label_type="binary",
    num_outputs=2, # 0: No Finding, 1: anomalía
    task_type="ce",
    num_epochs=NUM_EPOCHS
)

# Tarea 2: 3 clases (sano, neumonía, otra anomalía)
model_3cls = train_task(
    label_type="3cls",
    num_outputs=3,     # 0: sano, 1: neumonía, 2: otra
    task_type="ce",
    num_epochs=NUM_EPOCHS
)

print("Entrenamiento de las 2 downstream tasks completado.") 


