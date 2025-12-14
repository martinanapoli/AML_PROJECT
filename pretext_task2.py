import os
import random
from glob import glob

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from torchvision import models


data_dir = "/Users/martu/Desktop/dataset3"
PRETEXT1_WEIGHTS = None
# PRETEXT1_WEIGHTS = "/Users/martu/Desktop/outputs/rotation_resnet18.pth"
OUTPUT_DIR = "/Users/martu/Desktop/outputs"

image_size = 224
batch_size = 4
num_epochs = 10
learning_rate = 1e-4

device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)

num_workers = 0  

print("Using device:", device)

ALLOWED_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


FOLDER_NAME_FILTER = None  
USE_CLASS_SUBFOLDERS = True

CLASSES = []  


# FUNCIONES AUXILIARES 

def is_image_file(filename):
    ext = os.path.splitext(filename)[1].lower()
    return ext in ALLOWED_EXTENSIONS


def collect_image_paths(root_dir, classes=None, folder_name_filter=None):
   
    all_paths = []

    if classes is not None and len(classes) > 0:
        for cls in classes:
            cls_root = os.path.join(root_dir, cls)
            if not os.path.isdir(cls_root):
                continue

            for current_root, dirs, files in os.walk(cls_root):
                if folder_name_filter is not None and os.path.basename(current_root) != folder_name_filter:
                    continue

                for fname in files:
                    if is_image_file(fname):
                        all_paths.append(os.path.join(current_root, fname))
    else:
        for current_root, dirs, files in os.walk(root_dir):
            if folder_name_filter is not None and os.path.basename(current_root) != folder_name_filter:
                continue

            for fname in files:
                if is_image_file(fname):
                    all_paths.append(os.path.join(current_root, fname))

    return all_paths


# RECOGER TODAS LAS IMÁGENES (PNG/JPG/JPEG/...)

if USE_CLASS_SUBFOLDERS:
    clases = CLASSES
    all_paths = collect_image_paths(
        root_dir=data_dir,
        classes=clases,
        folder_name_filter=FOLDER_NAME_FILTER
    )
else:
    clases = None
    all_paths = collect_image_paths(
        root_dir=data_dir,
        classes=None,
        folder_name_filter=FOLDER_NAME_FILTER
    )

random.seed(42)
random.shuffle(all_paths)

num_total = len(all_paths)
print("Total image files found:", num_total)

if num_total == 0:
    raise RuntimeError("No image files found! Revisa la ruta, extensiones y estructura del dataset.")

# SPLIT TRAIN / VAL / TEST (70% / 15% / 15%)

train_end = int(0.7 * num_total)
val_end = int(0.85 * num_total)

train_paths = all_paths[:train_end]
val_paths   = all_paths[train_end:val_end]
test_paths  = all_paths[val_end:]

print(f"Train: {len(train_paths)}, Val: {len(val_paths)}, Test: {len(test_paths)}")


# DATASET

class CTInpaintingDataset(Dataset):
    def __init__(self, image_paths, image_size=224):
        self.image_paths = image_paths
        self.image_size = image_size

        if len(self.image_paths) == 0:
            raise RuntimeError("No image files found in dataset paths!")

        # Transform to tensor and resize
        self.transform = T.Compose([
            T.Resize((image_size, image_size)),
            T.ToTensor(),  # [0,1], CxHxW
        ])

        # Normalización tipo ImageNet
        self.normalize = T.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )

    def __len__(self):
        return len(self.image_paths)

    def _random_rect_mask(self, h, w, min_frac=0.2, max_frac=0.5):
        """
        Crea una máscara rectangular (1 dentro del hueco, 0 fuera).
        """
        mask = np.zeros((h, w), dtype=np.float32)

        mask_h = int(random.uniform(min_frac, max_frac) * h)
        mask_w = int(random.uniform(min_frac, max_frac) * w)

        top = random.randint(0, h - mask_h)
        left = random.randint(0, w - mask_w)

        mask[top:top + mask_h, left:left + mask_w] = 1.0
        return mask  # HxW

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]

        img = Image.open(img_path).convert("L")  # grayscale
        img = self.transform(img)  # [1, H, W], values [0,1]
        # Replicar a 3 canales para ResNet
        img_3c = img.repeat(3, 1, 1)  # [3, H, W]


        _, H, W = img_3c.shape

        # Create mask: 1 = hole, 0 = visible
        mask_np = self._random_rect_mask(H, W)
        mask = torch.from_numpy(mask_np).unsqueeze(0)  # [1, H, W]

        # Apply mask: set hole to 0 (negro)
        masked_img = img_3c * (1.0 - mask)

        # Normalize input (masked image) for ResNet
        masked_img_norm = self.normalize(masked_img)

        # Target es la imagen original (sin normalizar, en [0,1])
        target = img_3c  # [3, H, W]

        return masked_img_norm, mask, target


# MODELO: ResNet18 Encoder + U-Net-like Decoder

class ResNetUNet(nn.Module):
    def __init__(self, n_channels=3, n_classes=3, encoder_weights_path=None):
        super().__init__()

        if encoder_weights_path is not None:
            # Creamos una ResNet18 "vacía" con la estructura (fc de 1000 clases)
            base_model = models.resnet18()

            # Cargamos el state_dict de la Pretext Task 1
            state_dict = torch.load(encoder_weights_path, map_location="cpu")

            # Eliminar pesos de la capa fc para evitar size mismatch
            state_dict = {
                k: v for k, v in state_dict.items()
                if not k.startswith("fc.")
            }

            # Cargamos solo las capas que coinciden 
            missing, unexpected = base_model.load_state_dict(state_dict, strict=False)
            print(">> Cargados pesos del encoder desde Pretext Task 1")
            if missing:
                print("   Missing keys (normal, suele incluir 'fc.*'):", missing)
            if unexpected:
                print("   Unexpected keys:", unexpected)
        else:
            try:
                base_model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
                print(">> Usando ResNet18 con pesos de ImageNet")
            except AttributeError:
                base_model = models.resnet18(pretrained=True)
                print(">> Usando ResNet18(pretrained=True)")

        self.base_layers = list(base_model.children())

        # Encoder (ResNet18)
        self.layer0 = nn.Sequential(
            self.base_layers[0],  # conv1
            self.base_layers[1],  # bn1
            self.base_layers[2],  # relu
        )  # 64, 112x112 para input 224x224

        self.maxpool = self.base_layers[3]  # maxpool 112->56

        self.layer1 = self.base_layers[4]  # -> 64, 56x56
        self.layer2 = self.base_layers[5]  # -> 128, 28x28
        self.layer3 = self.base_layers[6]  # -> 256, 14x14
        self.layer4 = self.base_layers[7]  # -> 512, 7x7

        # Decoder (U-Net-like, upsampling + convs)
        self.upsample = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)

        self.conv_up4 = self._conv_block(512 + 256, 256)  # x4 + x3
        self.conv_up3 = self._conv_block(256 + 128, 128)  # d4 + x2
        self.conv_up2 = self._conv_block(128 + 64, 64)    # d3 + x1
        self.conv_up1 = self._conv_block(64 + 64, 64)     # d2 + x0

        # ultima capa para volver al tamaño original (224x224) y 3 canales
        self.conv_last = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),  # 112 -> 224
            nn.Conv2d(64, n_classes, kernel_size=3, padding=1),
            nn.Sigmoid()  # salida en [0,1]
        )

    def _conv_block(self, in_channels, out_channels):
        block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        return block

    def forward(self, x):
        # Encoder
        x0 = self.layer0(x)                        # 64, 112x112
        x1 = self.layer1(self.maxpool(x0))         # 64, 56x56
        x2 = self.layer2(x1)                       # 128, 28x28
        x3 = self.layer3(x2)                       # 256, 14x14
        x4 = self.layer4(x3)                       # 512, 7x7

        # Decoder con skip connections
        d4 = self.upsample(x4)                     # 512, 14x14
        d4 = torch.cat([d4, x3], 1)                # 512+256
        d4 = self.conv_up4(d4)                     # 256, 14x14

        d3 = self.upsample(d4)                     # 256, 28x28
        d3 = torch.cat([d3, x2], 1)                # 256+128
        d3 = self.conv_up3(d3)                     # 128, 28x28

        d2 = self.upsample(d3)                     # 128, 56x56
        d2 = torch.cat([d2, x1], 1)                # 128+64
        d2 = self.conv_up2(d2)                     # 64, 56x56

        d1 = self.upsample(d2)                     # 64, 112x112
        d1 = torch.cat([d1, x0], 1)                # 64+64
        d1 = self.conv_up1(d1)                     # 64, 112x112

        out = self.conv_last(d1)                   # 3, 224x224

        return out


# VISUALIZACIÓN

def evaluate(model, dataloader, device):
    
    model.eval()
    l1_loss = nn.L1Loss(reduction="none")
    total_loss = 0.0
    total_acc = 0.0
    num_batches = 0

    with torch.no_grad():
        for masked_img, mask, target in dataloader:
            masked_img = masked_img.to(device)
            mask = mask.to(device)
            target = target.to(device)

            output = model(masked_img)

            loss_map = l1_loss(output, target)
            loss_masked = loss_map * mask
            denom = mask.sum() * 3.0 + 1e-8
            loss = loss_masked.sum() / denom  

            acc = 1.0 - loss.item()
            if acc < 0:
                acc = 0.0

            total_loss += loss.item()
            total_acc += acc
            num_batches += 1

    return total_loss / num_batches, total_acc / num_batches


def show_examples(model, dataset, device, num_examples=5):
    model.eval()
    indices = random.sample(range(len(dataset)), k=min(num_examples, len(dataset)))

    for idx in indices:
        masked_img_norm, mask, target = dataset[idx]

        inp = masked_img_norm.unsqueeze(0).to(device)

        with torch.no_grad():
            out = model(inp).cpu().squeeze(0)  # [3,H,W]

        # Convertir a numpy para plot (usamos solo el primer canal)
        target_np = target[0].numpy()
        mask_np = mask.squeeze(0).numpy()
        masked_vis = target_np * (1 - mask_np)
        recon_np = out[0].numpy()

        fig, axs = plt.subplots(1, 3, figsize=(9, 3))
        axs[0].imshow(target_np, cmap="gray")
        axs[0].set_title("Original")
        axs[0].axis("off")

        axs[1].imshow(masked_vis, cmap="gray")
        axs[1].set_title("Masked")
        axs[1].axis("off")

        axs[2].imshow(recon_np, cmap="gray")
        axs[2].set_title("Reconstruction")
        axs[2].axis("off")

        plt.tight_layout()
        plt.show()


# TRAIN LOOP CON TRAIN/VAL/TEST

def main():
    train_dataset = CTInpaintingDataset(train_paths, image_size=image_size)
    val_dataset   = CTInpaintingDataset(val_paths,   image_size=image_size)
    test_dataset  = CTInpaintingDataset(test_paths,  image_size=image_size)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers)
    val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False,
                              num_workers=num_workers)
    test_loader  = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False,
                              num_workers=num_workers)

    model = ResNetUNet(
        n_channels=3,
        n_classes=3,
        encoder_weights_path=PRETEXT1_WEIGHTS
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    l1_loss = nn.L1Loss(reduction="none")

    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []

    print("Train size:", len(train_dataset),
          "Val size:", len(val_dataset),
          "Test size:", len(test_dataset))
    print("Starting training...")

    plt.figure(figsize=(12, 6))

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        running_acc = 0.0
        num_batches = 0

        for i, (masked_img, mask, target) in enumerate(train_loader):
            masked_img = masked_img.to(device)
            mask = mask.to(device)
            target = target.to(device)

            optimizer.zero_grad()

            output = model(masked_img)

            # L1 sobre la region enmascarada
            loss_map = l1_loss(output, target)
            loss_masked = loss_map * mask
            denom = mask.sum() * 3.0 + 1e-8
            loss = loss_masked.sum() / denom     

            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            running_acc += max(0.0, 1.0 - loss.item())
            num_batches += 1

        train_loss = running_loss / num_batches
        train_acc = running_acc / num_batches
        train_losses.append(train_loss)
        train_accuracies.append(train_acc)

        val_loss, val_acc = evaluate(model, val_loader, device)
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)

        print(f"Epoch [{epoch+1}/{num_epochs}] "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

    # Resultados finales
    final_train_loss = np.mean(train_losses)
    final_train_acc = np.mean(train_accuracies)
    print(f"\nFinal Training Loss: {final_train_loss:.4f}")
    print(f"Final Training Accuracy: {final_train_acc:.4f}")

    plt.subplot(1, 2, 1)
    plt.plot(range(1, num_epochs+1), train_losses, label="Train Loss")
    plt.plot(range(1, num_epochs+1), val_losses, label="Val Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Train and Validation Loss")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(range(1, num_epochs+1), train_accuracies, label="Train Accuracy")
    plt.plot(range(1, num_epochs+1), val_accuracies, label="Val Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.title("Train and Validation Accuracy")
    plt.legend()

    plt.tight_layout()
    plt.show()

    print("Training finished.")

    # Evaluacion en test
    test_loss, test_acc = evaluate(model, test_loader, device)
    print(f"[TEST] Loss: {test_loss:.4f}, Accuracy: {test_acc:.4f}")

    # Guardar modelo en OUTPUT_DIR
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    save_path = os.path.join(OUTPUT_DIR, "pretext2_dataset1.pth")
    torch.save(model.state_dict(), save_path)
    print(f"Model saved at: {save_path}")

    # Visualizar ejemplos
    show_examples(model, test_dataset, device, num_examples=5)


if __name__ == "__main__":
    main()
