import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import numpy as np
import pandas as pd
from glob import glob
from collections import Counter
from PIL import Image
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import resnet18

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score, davies_bouldin_score


PROJECT_ROOT = "/Users/martu/Desktop/machine_project"

ROOT_DIR = os.path.join(PROJECT_ROOT, "ChestX")
IMAGES_DIRS = [
    os.path.join(ROOT_DIR, f"images_{i:03d}", "images") for i in range(1, 11)
]

CSV_PATH = os.path.join(ROOT_DIR, "data_test.csv")
print("CSV_PATH:", CSV_PATH)
assert os.path.exists(CSV_PATH), "CSV path does not exist, please check CSV_PATH"

RESULTS_DIR = "/Users/martu/Desktop/machine_project/RESULTS_DEFINITIVO_P1P2"
os.makedirs(RESULTS_DIR, exist_ok=True)

MODELS_DIR = "/Users/martu/Desktop/machine_project/encoders_ptx"

IMAGE_SIZE = 224
BATCH_SIZE = 32
NUM_EPOCHS_LINEAR = 10    # linear probe
NUM_EPOCHS_FINETUNE = 10   # fine-tuning
LR_LINEAR = 1e-3
LR_ENCODER = 1e-5
LR_HEAD = 1e-4

if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")

print("Using device:", DEVICE)
print("CUDA available:", torch.cuda.is_available())
print("CUDA device_count:", torch.cuda.device_count())
print("MPS available:", torch.backends.mps.is_available())


df = pd.read_csv(CSV_PATH)
df = df.rename(columns={col: col.strip() for col in df.columns})

assert "Image Index" in df.columns
assert "Finding Labels" in df.columns

all_image_paths = []
for d in IMAGES_DIRS:
    all_image_paths.extend(glob(os.path.join(d, "*.png")))
    all_image_paths.extend(glob(os.path.join(d, "*.jpg")))
    all_image_paths.extend(glob(os.path.join(d, "*.jpeg")))

basename_to_path = {os.path.basename(p): p for p in all_image_paths}
print(f"Found {len(basename_to_path)} images on disk.")

df = df[df["Image Index"].isin(basename_to_path.keys())].reset_index(drop=True)
print(f"Rows in CSV after filtering for existing images: {len(df)}")


all_labels = []
for text in df["Finding Labels"]:
    labs = text.split("|")
    for l in labs:
        if l != "No Finding":
            all_labels.append(l)

diseases = sorted(list(set(all_labels)))
print("Pathologies (excluding 'No Finding'):", diseases)
num_diseases = len(diseases)
disease_to_idx = {d: i for i, d in enumerate(diseases)}

def get_binary_label(finding_labels: str):
    return 0 if finding_labels == "No Finding" else 1

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


# 4. TRAIN/VAL/TEST SPLIT

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


# 5. DATASET AND DATALOADERS

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

        img = Image.open(img_path).convert("L")

        if self.transform is not None:
            img = self.transform(img)

        # Convert (1, H, W) → (3, H, W) for ResNet
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
            raise ValueError("Invalid label_type")

def make_loaders(label_type, batch_size=BATCH_SIZE):
    train_ds = ChestXrayDataset(train_df, label_type=label_type, transform=train_transform)
    val_ds   = ChestXrayDataset(val_df,   label_type=label_type, transform=eval_transform)
    test_ds  = ChestXrayDataset(test_df,  label_type=label_type, transform=eval_transform)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=0)

    return train_loader, val_loader, test_loader


# CLASSES AND FUNCTIONS TO LOAD PRETEXT ENCODERS

def build_resnet18_from_encoder_only(encoder_path, num_outputs, freeze_encoder=True):
    
    print(f"\nLoading PT1/PT3 encoder from: {encoder_path}")
    model = resnet18(weights=None)
    state = torch.load(encoder_path, map_location="cpu")

    missing, unexpected = model.load_state_dict(state, strict=False)
    print("   Missing keys:", missing)
    print("   Unexpected keys:", unexpected)

    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_outputs)

    if freeze_encoder:
        for name, p in model.named_parameters():
            if not name.startswith("fc."):
                p.requires_grad = False

    return model.to(DEVICE)


class PT2InpaintingEncoderClassifier(nn.Module):
   
    def __init__(self, num_outputs):
        super().__init__()
        base = resnet18(weights=None)

        # layer0 = conv1 + bn1 + relu + maxpool
        self.layer0 = nn.Sequential(
            base.conv1,
            base.bn1,
            base.relu,
            base.maxpool,
        )
        self.layer1 = base.layer1
        self.layer2 = base.layer2
        self.layer3 = base.layer3
        self.layer4 = base.layer4

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_outputs)

    def forward(self, x):
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


def build_pt2_encoder_from_state(encoder_path, num_outputs, freeze_encoder=True):
    
    print(f"\nLoading PT2 encoder (inpainting) from: {encoder_path}")
    model = PT2InpaintingEncoderClassifier(num_outputs=num_outputs)
    state = torch.load(encoder_path, map_location="cpu")

    missing, unexpected = model.load_state_dict(state, strict=False)
    print(" Missing keys:", missing)
    print(" Unexpected keys:", unexpected)

    if freeze_encoder:
        for name, p in model.named_parameters():
            if not name.startswith("fc."):
                p.requires_grad = False

    return model.to(DEVICE)


# GENERIC TRAINING LOOPS 

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

def get_predictions(model, loader, task_type):
   
    model.eval()
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(DEVICE)
            labels = labels.to(DEVICE)
            outputs = model(imgs)
            if task_type == "ce":
                preds = outputs.argmax(dim=1).detach().cpu().numpy()
                targets = labels.detach().cpu().numpy()
            else:
                preds = (torch.sigmoid(outputs) > 0.5).float().detach().cpu().numpy()
                targets = labels.detach().cpu().numpy()
            all_preds.append(preds)
            all_targets.append(targets)
    return np.concatenate(all_preds), np.concatenate(all_targets)

def save_training_history(history, filename_prefix):
    
    df_hist = pd.DataFrame(history)
    csv_path = os.path.join(RESULTS_DIR, f"{filename_prefix}_history.csv")
    df_hist.to_csv(csv_path, index=False)
    print(f"   Saved training history to: {csv_path}")

def save_confusion_matrix_plot(cm, classes, filename_prefix, title):
   
    plt.figure(figsize=(4, 4))
    plt.imshow(cm, interpolation='nearest')
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45, ha="right")
    plt.yticks(tick_marks, classes)

    thresh = cm.max() / 2.0 if cm.max() > 0 else 0.5
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(
                j, i, format(cm[i, j], 'd'),
                horizontalalignment="center",
                color="white" if cm[i, j] > thresh else "black",
            )

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, f"{filename_prefix}_confusion_matrix.png")
    plt.savefig(path)
    plt.close()
    print(f" Saved confusion matrix plot to: {path}")


# LINEAR PROBE AND FINE-TUNING FOR A SINGLE ENCODER

def _build_model_for_encoder(encoder_name, encoder_path, num_outputs, freeze_encoder):
   
    if encoder_name.startswith("PT2"):
        return build_pt2_encoder_from_state(
            encoder_path=encoder_path,
            num_outputs=num_outputs,
            freeze_encoder=freeze_encoder,
        )
    else:
        return build_resnet18_from_encoder_only(
            encoder_path=encoder_path,
            num_outputs=num_outputs,
            freeze_encoder=freeze_encoder,
        )

def _classes_for_label_type(label_type, num_outputs):
    if label_type == "binary":
        return ["No Finding", "Anomaly"]
    elif label_type == "3cls":
        return ["No Finding", "Pneumonia", "Other"]
    else:
        # For generic multi class (not used here)
        return [str(i) for i in range(num_outputs)]

def train_linear_probe(encoder_name, encoder_path, label_type, num_outputs, task_type):
    print(f"\n LINEAR PROBE | Encoder: {encoder_name} | Task: {label_type}")

    train_loader, val_loader, test_loader = make_loaders(label_type)
    model = _build_model_for_encoder(
        encoder_name=encoder_name,
        encoder_path=encoder_path,
        num_outputs=num_outputs,
        freeze_encoder=True,   # freeze encoder
    )

    criterion = nn.CrossEntropyLoss() if task_type == "ce" else nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LR_LINEAR
    )

    history = []

    best_val_f1 = -np.inf
    best_state = None

    for epoch in range(1, NUM_EPOCHS_LINEAR + 1):
        train_loss, train_acc, train_f1 = train_epoch(model, train_loader, optimizer, criterion, task_type)
        val_loss, val_acc, val_f1 = eval_epoch(model, val_loader, criterion, task_type)

        print(f"[Linear] Epoch {epoch:02d} | "
              f"Train loss: {train_loss:.4f} acc: {train_acc:.4f} f1: {train_f1:.4f} | "
              f"Val loss: {val_loss:.4f} acc: {val_acc:.4f} f1: {val_f1:.4f}")

        history.append({
            "epoch": epoch, "split": "train",
            "loss": train_loss, "acc": train_acc, "f1": train_f1,
        })
        history.append({
            "epoch": epoch, "split": "val",
            "loss": val_loss, "acc": val_acc, "f1": val_f1,
        })

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_state = model.state_dict()

    if best_state is not None:
        model.load_state_dict(best_state)

    # historial de entrenamiento
    save_training_history(history, filename_prefix=f"{encoder_name}_{label_type}_linear")

    test_loss, test_acc, test_f1 = eval_epoch(model, test_loader, criterion, task_type)
    print(f"\n>>> [Linear] TEST RESULTS ({encoder_name} - {label_type}) <<<")
    print(f"Test loss: {test_loss:.4f} | acc: {test_acc:.4f} | f1: {test_f1:.4f}\n")

    # confusion matrix
    if task_type == "ce":
        preds, targets = get_predictions(model, test_loader, task_type)
        cm = confusion_matrix(targets, preds)
        print("Confusion matrix (test):")
        print(cm)
        class_names = _classes_for_label_type(label_type, num_outputs)
        save_confusion_matrix_plot(
            cm,
            classes=class_names,
            filename_prefix=f"{encoder_name}_{label_type}_linear",
            title=f"Confusion Matrix - {encoder_name} (Linear, {label_type})",
        )

    return model


def train_finetune(encoder_name, encoder_path, label_type, num_outputs, task_type):
    print(f"\n FINE-TUNING | Encoder: {encoder_name} | Task: {label_type}")

    train_loader, val_loader, test_loader = make_loaders(label_type)
    model = _build_model_for_encoder(
        encoder_name=encoder_name,
        encoder_path=encoder_path,
        num_outputs=num_outputs,
        freeze_encoder=False,  # encoder trainable
    )

    criterion = nn.CrossEntropyLoss() if task_type == "ce" else nn.BCEWithLogitsLoss()

    # Two parameter groups: encoder and head
    encoder_params = []
    head_params = []

    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if name.startswith("fc."):
            head_params.append(p)
        else:
            encoder_params.append(p)

    optimizer = torch.optim.Adam([
        {"params": encoder_params, "lr": LR_ENCODER},
        {"params": head_params, "lr": LR_HEAD},
    ])

    history = []

    best_val_f1 = -np.inf
    best_state = None

    for epoch in range(1, NUM_EPOCHS_FINETUNE + 1):
        train_loss, train_acc, train_f1 = train_epoch(model, train_loader, optimizer, criterion, task_type)
        val_loss, val_acc, val_f1 = eval_epoch(model, val_loader, criterion, task_type)

        print(f"[FT] Epoch {epoch:02d} | "
              f"Train loss: {train_loss:.4f} acc: {train_acc:.4f} f1: {train_f1:.4f} | "
              f"Val loss: {val_loss:.4f} acc: {val_acc:.4f} f1: {val_f1:.4f}")

        history.append({
            "epoch": epoch, "split": "train",
            "loss": train_loss, "acc": train_acc, "f1": train_f1,
        })
        history.append({
            "epoch": epoch, "split": "val",
            "loss": val_loss, "acc": val_acc, "f1": val_f1,
        })

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_state = model.state_dict()

    if best_state is not None:
        model.load_state_dict(best_state)

    # historial
    save_training_history(history, filename_prefix=f"{encoder_name}_{label_type}_finetune")

    test_loss, test_acc, test_f1 = eval_epoch(model, test_loader, criterion, task_type)
    print(f"\n>>> [FT] TEST RESULTS ({encoder_name} - {label_type}) <<<")
    print(f"Test loss: {test_loss:.4f} | acc: {test_acc:.4f} | f1: {test_f1:.4f}\n")

    # matriz de confusion
    if task_type == "ce":
        preds, targets = get_predictions(model, test_loader, task_type)
        cm = confusion_matrix(targets, preds)
        print("Confusion matrix (test):")
        print(cm)
        class_names = _classes_for_label_type(label_type, num_outputs)
        save_confusion_matrix_plot(
            cm,
            classes=class_names,
            filename_prefix=f"{encoder_name}_{label_type}_finetune",
            title=f"Confusion Matrix - {encoder_name} (FT, {label_type})",
        )

    return model

# 9. MULTI-ENCODER FUSION (H4 / O4)

class MultiEncoderFusion(nn.Module):
   
    def __init__(self, encoder_specs, num_outputs, freeze_encoders=True):
       
        super().__init__()
        self.encoder_names = [name for (name, _) in encoder_specs]
        self.encoders = nn.ModuleList()

        for enc_name, enc_path in encoder_specs:
            enc = build_feature_extractor_for_encoder(enc_name, enc_path)  # 512-d output
            if freeze_encoders:
                for p in enc.parameters():
                    p.requires_grad = False
            self.encoders.append(enc)

        in_features = 512 * len(self.encoders)
        self.head = nn.Linear(in_features, num_outputs)

    def forward(self, x):
        feats_list = []
        for enc in self.encoders:
            f = enc(x)           # (B, 512)
            feats_list.append(f)
        z = torch.cat(feats_list, dim=1)  # (B, 512 * n_encoders)
        out = self.head(z)
        return out


def train_linear_probe_fusion(fusion_name, encoder_specs, label_type, num_outputs, task_type):
    
    print(f"\n LINEAR PROBE FUSION | {fusion_name} | Task: {label_type}")

    train_loader, val_loader, test_loader = make_loaders(label_type)
    model = MultiEncoderFusion(
        encoder_specs=encoder_specs,
        num_outputs=num_outputs,
        freeze_encoders=True,   # freeze encoders, train only head
    ).to(DEVICE)

    criterion = nn.CrossEntropyLoss() if task_type == "ce" else nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LR_LINEAR
    )

    history = []

    best_val_f1 = -np.inf
    best_state = None

    for epoch in range(1, NUM_EPOCHS_LINEAR + 1):
        train_loss, train_acc, train_f1 = train_epoch(model, train_loader, optimizer, criterion, task_type)
        val_loss, val_acc, val_f1 = eval_epoch(model, val_loader, criterion, task_type)

        print(f"[Linear-Fusion] Epoch {epoch:02d} | "
              f"Train loss: {train_loss:.4f} acc: {train_acc:.4f} f1: {train_f1:.4f} | "
              f"Val loss: {val_loss:.4f} acc: {val_acc:.4f} f1: {val_f1:.4f}")

        history.append({
            "epoch": epoch, "split": "train",
            "loss": train_loss, "acc": train_acc, "f1": train_f1,
        })
        history.append({
            "epoch": epoch, "split": "val",
            "loss": val_loss, "acc": val_acc, "f1": val_f1,
        })

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_state = model.state_dict()

    if best_state is not None:
        model.load_state_dict(best_state)

    # guardar historial
    save_training_history(history, filename_prefix=f"{fusion_name}_{label_type}_linear_fusion")

    test_loss, test_acc, test_f1 = eval_epoch(model, test_loader, criterion, task_type)
    print(f"\n>>> [Linear-Fusion] TEST RESULTS ({fusion_name} - {label_type}) <<<")
    print(f"Test loss: {test_loss:.4f} | acc: {test_acc:.4f} | f1: {test_f1:.4f}\n")

    # matriz
    if task_type == "ce":
        preds, targets = get_predictions(model, test_loader, task_type)
        cm = confusion_matrix(targets, preds)
        print("Confusion matrix (test):")
        print(cm)
        class_names = _classes_for_label_type(label_type, num_outputs)
        save_confusion_matrix_plot(
            cm,
            classes=class_names,
            filename_prefix=f"{fusion_name}_{label_type}_linear_fusion",
            title=f"Confusion Matrix - {fusion_name} (Linear Fusion, {label_type})",
        )

    return model


def train_finetune_fusion(fusion_name, encoder_specs, label_type, num_outputs, task_type):
   
    print(f"\n FINE-TUNING FUSION | {fusion_name} | Task: {label_type}")

    train_loader, val_loader, test_loader = make_loaders(label_type)
    model = MultiEncoderFusion(
        encoder_specs=encoder_specs,
        num_outputs=num_outputs,
        freeze_encoders=False,  # encoders are trainable
    ).to(DEVICE)

    criterion = nn.CrossEntropyLoss() if task_type == "ce" else nn.BCEWithLogitsLoss()

    encoder_params = []
    head_params = []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if name.startswith("head."):
            head_params.append(p)
        else:
            encoder_params.append(p)

    optimizer = torch.optim.Adam([
        {"params": encoder_params, "lr": LR_ENCODER},
        {"params": head_params, "lr": LR_HEAD},
    ])

    history = []

    best_val_f1 = -np.inf
    best_state = None

    for epoch in range(1, NUM_EPOCHS_FINETUNE + 1):
        train_loss, train_acc, train_f1 = train_epoch(model, train_loader, optimizer, criterion, task_type)
        val_loss, val_acc, val_f1 = eval_epoch(model, val_loader, criterion, task_type)

        print(f"[FT-Fusion] Epoch {epoch:02d} | "
              f"Train loss: {train_loss:.4f} acc: {train_acc:.4f} f1: {train_f1:.4f} | "
              f"Val loss: {val_loss:.4f} acc: {val_acc:.4f} f1: {val_f1:.4f}")

        history.append({
            "epoch": epoch, "split": "train",
            "loss": train_loss, "acc": train_acc, "f1": train_f1,
        })
        history.append({
            "epoch": epoch, "split": "val",
            "loss": val_loss, "acc": val_acc, "f1": val_f1,
        })

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_state = model.state_dict()

    if best_state is not None:
        model.load_state_dict(best_state)

    # guardar historial
    save_training_history(history, filename_prefix=f"{fusion_name}_{label_type}_finetune_fusion")

    test_loss, test_acc, test_f1 = eval_epoch(model, test_loader, criterion, task_type)
    print(f"\n>>> [FT-Fusion] TEST RESULTS ({fusion_name} - {label_type}) <<<")
    print(f"Test loss: {test_loss:.4f} | acc: {test_acc:.4f} | f1: {test_f1:.4f}\n")

    # matriz
    if task_type == "ce":
        preds, targets = get_predictions(model, test_loader, task_type)
        cm = confusion_matrix(targets, preds)
        print("Confusion matrix (test):")
        print(cm)
        class_names = _classes_for_label_type(label_type, num_outputs)
        save_confusion_matrix_plot(
            cm,
            classes=class_names,
            filename_prefix=f"{fusion_name}_{label_type}_finetune_fusion",
            title=f"Confusion Matrix - {fusion_name} (FT Fusion, {label_type})",
        )

    return model


def run_fusion_experiments(fusion_configs):
   
    for fusion_name, encoder_specs in fusion_configs:

        label_type = "binary"
        num_outputs = 2
        task_type = "ce"

        # Linear probe fusion
        _ = train_linear_probe_fusion(
            fusion_name=fusion_name,
            encoder_specs=encoder_specs,
            label_type=label_type,
            num_outputs=num_outputs,
            task_type=task_type,
        )

        # Fine-tuning fusion
        _ = train_finetune_fusion(
            fusion_name=fusion_name,
            encoder_specs=encoder_specs,
            label_type=label_type,
            num_outputs=num_outputs,
            task_type=task_type,
        )
