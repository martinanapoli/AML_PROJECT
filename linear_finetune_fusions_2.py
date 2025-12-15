import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import numpy as np
import pandas as pd
from glob import glob
from PIL import Image
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import resnet18

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix



PROJECT_ROOT = "/Users/martu/Desktop/machine_project"

ROOT_DIR = os.path.join(PROJECT_ROOT, "ChestX")
IMAGES_DIRS = [os.path.join(ROOT_DIR, f"images_{i:03d}", "images") for i in range(1, 9)]
CSV_PATH = "/Users/martu/Desktop/machine_project/ChestX/Data_Entry_2017.csv"

RESULTS_DIR = "/Users/martu/Desktop/machine_project/RESULTS_fusiones_2"
os.makedirs(RESULTS_DIR, exist_ok=True)

MODELS_DIR = "/Users/martu/Desktop/machine_project/encoders_ptx"

IMAGE_SIZE = 224
BATCH_SIZE = 32
NUM_EPOCHS_LINEAR = 5
NUM_EPOCHS_FINETUNE = 5
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

assert os.path.exists(CSV_PATH), f"CSV path does not exist: {CSV_PATH}"


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

# BINARY LABELS ONLY

def get_binary_label(finding_labels: str):
    return 0 if finding_labels == "No Finding" else 1

df["label_binary"] = df["Finding Labels"].apply(get_binary_label)

# SPLIT

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

# DATASET + LOADERS


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

class ChestXrayBinaryDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df.reset_index(drop=True)
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

        # (1,H,W) -> (3,H,W)
        img = img.repeat(3, 1, 1)

        label = int(row["label_binary"])
        return img, label

def make_binary_loaders(batch_size=BATCH_SIZE):
    train_ds = ChestXrayBinaryDataset(train_df, transform=train_transform)
    val_ds   = ChestXrayBinaryDataset(val_df,   transform=eval_transform)
    test_ds  = ChestXrayBinaryDataset(test_df,  transform=eval_transform)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=0)
    return train_loader, val_loader, test_loader

# PT2 encoder class (needed to load PT2 checkpoints)

class PT2InpaintingEncoderClassifier(nn.Module):
    def __init__(self, num_outputs):
        super().__init__()
        base = resnet18(weights=None)

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

# ADAPT conv1 1-channel -> 3-channel (PT3 issue)


def _adapt_conv1_weight_if_needed(state_dict, model, conv_key_candidates=("conv1.weight", "layer0.0.weight")):
   
    key = None
    named_params = dict(model.named_parameters())
    for k in conv_key_candidates:
        if k in state_dict and k in named_params:
            key = k
            break

    if key is None:
        return state_dict

    w_ckpt = state_dict[key]
    w_model = model.state_dict().get(key, None)
    if w_model is None:
        return state_dict

    if w_ckpt.ndim == 4 and w_model.ndim == 4:
        if w_ckpt.shape[1] == 1 and w_model.shape[1] == 3:
            state_dict[key] = w_ckpt.repeat(1, 3, 1, 1) / 3.0

    return state_dict

def build_feature_extractor_for_encoder(encoder_name, encoder_path):
    
    print(f"[Fusion] Loading feature extractor: {encoder_name} from {encoder_path}")
    assert os.path.exists(encoder_path), f"Missing checkpoint: {encoder_path}"

    state = torch.load(encoder_path, map_location="cpu")

    if encoder_name.startswith("PT2"):
        model = PT2InpaintingEncoderClassifier(num_outputs=2)  # dummy head
        state = _adapt_conv1_weight_if_needed(state, model, conv_key_candidates=("layer0.0.weight", "conv1.weight"))
        missing, unexpected = model.load_state_dict(state, strict=False)
        print(" Missing keys:", missing)
        print(" Unexpected keys:", unexpected)
        model.fc = nn.Identity()
    else:
        model = resnet18(weights=None)
        state = _adapt_conv1_weight_if_needed(state, model, conv_key_candidates=("conv1.weight",))
        missing, unexpected = model.load_state_dict(state, strict=False)
        print(" Missing keys:", missing)
        print(" Unexpected keys:", unexpected)
        model.fc = nn.Identity()

    return model.to(DEVICE)

# METRICS 

def train_epoch_ce(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0.0
    all_preds, all_targets = [], []

    for imgs, labels in loader:
        imgs = imgs.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * imgs.size(0)
        preds = outputs.argmax(dim=1).detach().cpu().numpy()
        targets = labels.detach().cpu().numpy()
        all_preds.append(preds)
        all_targets.append(targets)

    avg_loss = total_loss / len(loader.dataset)
    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)

    acc = accuracy_score(all_targets, all_preds)
    f1 = f1_score(all_targets, all_preds, average="weighted", zero_division=0)
    return avg_loss, acc, f1

def eval_epoch_ce(model, loader, criterion):
    model.eval()
    total_loss = 0.0
    all_preds, all_targets = [], []

    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(imgs)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * imgs.size(0)
            preds = outputs.argmax(dim=1).detach().cpu().numpy()
            targets = labels.detach().cpu().numpy()
            all_preds.append(preds)
            all_targets.append(targets)

    avg_loss = total_loss / len(loader.dataset)
    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)

    acc = accuracy_score(all_targets, all_preds)
    f1 = f1_score(all_targets, all_preds, average="weighted", zero_division=0)
    return avg_loss, acc, f1

def get_predictions_ce(model, loader):
    model.eval()
    all_preds, all_targets = [], []
    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(DEVICE)
            labels = labels.to(DEVICE)
            outputs = model(imgs)
            preds = outputs.argmax(dim=1).detach().cpu().numpy()
            targets = labels.detach().cpu().numpy()
            all_preds.append(preds)
            all_targets.append(targets)
    return np.concatenate(all_preds), np.concatenate(all_targets)

def save_training_history(history, filename_prefix):
    df_hist = pd.DataFrame(history)
    csv_path = os.path.join(RESULTS_DIR, f"{filename_prefix}_history.csv")
    df_hist.to_csv(csv_path, index=False)
    print(f"Saved history: {csv_path}")

def save_confusion_matrix_plot(cm, classes, filename_prefix, title):
    plt.figure(figsize=(4, 4))
    plt.imshow(cm, interpolation="nearest")
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45, ha="right")
    plt.yticks(tick_marks, classes)

    thresh = cm.max() / 2.0 if cm.max() > 0 else 0.5
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(
                j, i, format(cm[i, j], "d"),
                horizontalalignment="center",
                color="white" if cm[i, j] > thresh else "black",
            )
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.tight_layout()

    path = os.path.join(RESULTS_DIR, f"{filename_prefix}_confusion_matrix.png")
    plt.savefig(path)
    plt.close()
    print(f"Saved confusion matrix: {path}")

# FUSION MODEL

class MultiEncoderFusion(nn.Module):
    def __init__(self, encoder_specs, num_outputs=2, freeze_encoders=True):
        super().__init__()
        self.encoders = nn.ModuleList()
        for enc_name, enc_path in encoder_specs:
            enc = build_feature_extractor_for_encoder(enc_name, enc_path)  # (B,512)
            if freeze_encoders:
                for p in enc.parameters():
                    p.requires_grad = False
            self.encoders.append(enc)

        self.head = nn.Linear(512 * len(self.encoders), num_outputs)

    def forward(self, x):
        feats = [enc(x) for enc in self.encoders]
        z = torch.cat(feats, dim=1)
        return self.head(z)

# TRAIN: LINEAR-FUSION + FINETUNE-FUSION + SAVE 

def run_one_fusion(fusion_name, encoder_specs):
    label_type = "binary"
    classes = ["No Finding", "Anomaly"]

    train_loader, val_loader, test_loader = make_binary_loaders()
    criterion = nn.CrossEntropyLoss()

    # LINEAR FUSION (frozen encoders)
    print(f"\nLINEAR FUSION: {fusion_name}")
    model = MultiEncoderFusion(encoder_specs, num_outputs=2, freeze_encoders=True).to(DEVICE)
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LR_LINEAR)

    history = []
    best_val_f1 = -1e9
    best_state = None

    for epoch in range(1, NUM_EPOCHS_LINEAR + 1):
        tr_loss, tr_acc, tr_f1 = train_epoch_ce(model, train_loader, optimizer, criterion)
        va_loss, va_acc, va_f1 = eval_epoch_ce(model, val_loader, criterion)

        print(f"[LinearFusion] Epoch {epoch:02d} | "
              f"Train loss {tr_loss:.4f} acc {tr_acc:.4f} f1 {tr_f1:.4f} | "
              f"Val loss {va_loss:.4f} acc {va_acc:.4f} f1 {va_f1:.4f}")

        history += [
            {"epoch": epoch, "split": "train", "loss": tr_loss, "acc": tr_acc, "f1": tr_f1},
            {"epoch": epoch, "split": "val",   "loss": va_loss, "acc": va_acc, "f1": va_f1},
        ]

        if va_f1 > best_val_f1:
            best_val_f1 = va_f1
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    save_training_history(history, f"{fusion_name}_{label_type}_linear_fusion")

    te_loss, te_acc, te_f1 = eval_epoch_ce(model, test_loader, criterion)
    print(f"[LinearFusion TEST] loss {te_loss:.4f} acc {te_acc:.4f} f1 {te_f1:.4f}")

    preds, targets = get_predictions_ce(model, test_loader)
    cm = confusion_matrix(targets, preds)
    save_confusion_matrix_plot(
        cm, classes,
        f"{fusion_name}_{label_type}_linear_fusion",
        f"Confusion Matrix - {fusion_name} (Linear Fusion)"
    )

    ckpt_path_linear = os.path.join(RESULTS_DIR, f"{fusion_name}_linear_fusion_best.pth")
    torch.save(model.state_dict(), ckpt_path_linear)
    print(f"Saved best linear fusion checkpoint: {ckpt_path_linear}")

    linear_test_row = {
        "fusion_name": fusion_name,
        "mode": "linear_fusion",
        "test_loss": te_loss,
        "test_acc": te_acc,
        "test_f1": te_f1,
    }

    # FINETUNE FUSION (train encoders + head)
    print(f"\nFINETUNE FUSION: {fusion_name}")
    model = MultiEncoderFusion(encoder_specs, num_outputs=2, freeze_encoders=False).to(DEVICE)

    encoder_params, head_params = [], []
    for name, p in model.named_parameters():
        if name.startswith("head."):
            head_params.append(p)
        else:
            encoder_params.append(p)

    optimizer = torch.optim.Adam([
        {"params": encoder_params, "lr": LR_ENCODER},
        {"params": head_params, "lr": LR_HEAD},
    ])

    history = []
    best_val_f1 = -1e9
    best_state = None

    for epoch in range(1, NUM_EPOCHS_FINETUNE + 1):
        tr_loss, tr_acc, tr_f1 = train_epoch_ce(model, train_loader, optimizer, criterion)
        va_loss, va_acc, va_f1 = eval_epoch_ce(model, val_loader, criterion)

        print(f"[FinetuneFusion] Epoch {epoch:02d} | "
              f"Train loss {tr_loss:.4f} acc {tr_acc:.4f} f1 {tr_f1:.4f} | "
              f"Val loss {va_loss:.4f} acc {va_acc:.4f} f1 {va_f1:.4f}")

        history += [
            {"epoch": epoch, "split": "train", "loss": tr_loss, "acc": tr_acc, "f1": tr_f1},
            {"epoch": epoch, "split": "val",   "loss": va_loss, "acc": va_acc, "f1": va_f1},
        ]

        if va_f1 > best_val_f1:
            best_val_f1 = va_f1
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    save_training_history(history, f"{fusion_name}_{label_type}_finetune_fusion")

    te_loss, te_acc, te_f1 = eval_epoch_ce(model, test_loader, criterion)
    print(f"[FinetuneFusion TEST] loss {te_loss:.4f} acc {te_acc:.4f} f1 {te_f1:.4f}")

    preds, targets = get_predictions_ce(model, test_loader)
    cm = confusion_matrix(targets, preds)
    save_confusion_matrix_plot(
        cm, classes,
        f"{fusion_name}_{label_type}_finetune_fusion",
        f"Confusion Matrix - {fusion_name} (Finetune Fusion)"
    )

    ckpt_path_ft = os.path.join(RESULTS_DIR, f"{fusion_name}_finetune_fusion_best.pth")
    torch.save(model.state_dict(), ckpt_path_ft)
    print(f"Saved best finetune fusion checkpoint: {ckpt_path_ft}")

    finetune_test_row = {
        "fusion_name": fusion_name,
        "mode": "finetune_fusion",
        "test_loss": te_loss,
        "test_acc": te_acc,
        "test_f1": te_f1,
    }

    return [linear_test_row, finetune_test_row]

def run_all_fusions(fusion_configs):
    all_rows = []
    for fusion_name, encoder_specs in fusion_configs:
        rows = run_one_fusion(fusion_name, encoder_specs)
        all_rows.extend(rows)

    summary_path = os.path.join(RESULTS_DIR, "fusion_test_summary.csv")
    pd.DataFrame(all_rows).to_csv(summary_path, index=False)
    print(f"\nSaved TEST summary for all fusions: {summary_path}")

# MAIN: ONLY THE TWO LAST FUSIONS (PT2+PT3 and PT1+PT2+PT3)

if __name__ == "__main__":
    fusion_configs = [
        (
            "FUSION_PT2_PT3_D3",
            [
                ("PT2_D3", os.path.join(MODELS_DIR, "PT2_D3_encoder_only.pth")),
                ("PT3_D3", os.path.join(MODELS_DIR, "PT3_D3_encoder_only.pth")),
            ],
        ),
        (
            "FUSION_PT1_PT2_PT3_D3",
            [
                ("PT1_D3", os.path.join(MODELS_DIR, "PT1_D3_encoder_only.pth")),
                ("PT2_D3", os.path.join(MODELS_DIR, "PT2_D3_encoder_only.pth")),
                ("PT3_D3", os.path.join(MODELS_DIR, "PT3_D3_encoder_only.pth")),
            ],
        ),
    ]

    run_all_fusions(fusion_configs)
    print("Fusion-only (binary, D3) completed (ONLY last 2 fusions).")
