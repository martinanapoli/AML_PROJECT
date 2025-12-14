# Script to generate the 9 encoder_only.pth files from the pretext models

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
from torchvision.models import resnet18


# CONFIGURATION

# Project root
PROJECT_ROOT = "/Users/martu/Desktop/aml project"

# Folder that contains the original .pth files from the pretext tasks
RAW_MODELS_DIR = os.path.join(PROJECT_ROOT, "MODELOS")

# Folder where encoder-only files will be saved
ENCODERS_DIR = os.path.join(PROJECT_ROOT, "encoders")
os.makedirs(ENCODERS_DIR, exist_ok=True)

# File mappings: input → output
# PT1: texture/color (rotation), full ResNet18
PT1_FILES = [
    ("pretext1_dataset1.pth", "PT1_D1_encoder_only.pth"),
    ("pretext1_dataset2.pth", "PT1_D2_encoder_only.pth"),
    ("pretext1_dataset3.pth", "PT1_D3_encoder_only.pth"),
]

# PT2: inpainting (ResNet + UNet); encoder is in layer0..layer4
PT2_FILES = [
    ("pretext2_dataset1.pth", "PT2_D1_encoder_only.pth"),
    ("pretext2_dataset2.pth", "PT2_D2_encoder_only.pth"),
    ("pretext2_dataset3.pth", "PT2_D3_encoder_only.pth"),
]

# PT3: structure/composition; encoder is inside state["encoder_state_dict"]
PT3_FILES = [
    ("pretext3_dataset1.pth", "PT3_D1_encoder_only.pth"),
    ("pretext3_dataset2.pth", "PT3_D2_encoder_only.pth"),
    ("pretext3_dataset3.pth", "PT3_D3_encoder_only.pth"),
]


# HELPER FUNCTIONS

def make_pt1_encoder_only(in_path, out_path):
   
    print(f"[PT1] Cleaning {in_path} -> {out_path}")
    state = torch.load(in_path, map_location="cpu")

    if not isinstance(state, dict):
        raise ValueError(f"Expected a dict state_dict in {in_path}, got {type(state)}.")

    # Remove classifier head weights to avoid size mismatch (your fc has 4 outputs
    encoder_state = {k: v for k, v in state.items() if not k.startswith("fc.")}
    print(f"   Total keys in original state_dict: {len(state)}")
    print(f"   Keys kept after removing fc.*: {len(encoder_state)}")

    model = resnet18(weights=None)
    missing, unexpected = model.load_state_dict(encoder_state, strict=False)
    print("   Missing keys:", missing)
    print("   Unexpected keys:", unexpected)

    # Replace classifier head 
    model.fc = torch.nn.Identity()

    torch.save(model.state_dict(), out_path)
    print("Saved encoder-only (PT1).\n")


def make_pt2_encoder_only(in_path, out_path):
    
    print(f"[PT2] Cleaning {in_path} -> {out_path}")
    state = torch.load(in_path, map_location="cpu")

    if not isinstance(state, dict):
        raise ValueError(f"Expected a dict in {in_path}, got {type(state)}.")

    encoder_state = {
        k: v for k, v in state.items()
        if k.startswith("layer0.")
        or k.startswith("layer1.")
        or k.startswith("layer2.")
        or k.startswith("layer3.")
        or k.startswith("layer4.")
    }

    print(f"Total keys in original state_dict: {len(state)}")
    print(f"Encoder keys kept (layer0–layer4): {len(encoder_state)}")

    torch.save(encoder_state, out_path)
    print("Saved encoder-only (PT2).\n")


def make_pt3_encoder_only(in_path, out_path):
 
    print(f"[PT3] Cleaning {in_path} -> {out_path}")
    state = torch.load(in_path, map_location="cpu")

    if not isinstance(state, dict):
        raise ValueError(f"Expected a dict in {in_path}, got {type(state)}.")

    if "encoder_state_dict" not in state:
        raise KeyError(
            f"Key 'encoder_state_dict' not found in {in_path}. "
            f"Available keys: {list(state.keys())}"
        )

    encoder_state = state["encoder_state_dict"]

    if not isinstance(encoder_state, dict):
        raise ValueError(
            f"'encoder_state_dict' in {in_path} is not a dict, got {type(encoder_state)}."
        )

    print(f"   Encoder keys found (PT3): {len(encoder_state)}")

    torch.save(encoder_state, out_path)
    print("   Saved encoder-only (PT3).\n")

# PROCESS THE 9 MODELS

if __name__ == "__main__":

    # PT1: rotation / texture-color
    for in_name, out_name in PT1_FILES:
        in_path = os.path.join(RAW_MODELS_DIR, in_name)
        out_path = os.path.join(ENCODERS_DIR, out_name)
        assert os.path.exists(in_path), f"Input file not found: {in_path}"
        make_pt1_encoder_only(in_path, out_path)

    # PT2: inpainting
    for in_name, out_name in PT2_FILES:
        in_path = os.path.join(RAW_MODELS_DIR, in_name)
        out_path = os.path.join(ENCODERS_DIR, out_name)
        assert os.path.exists(in_path), f"Input file not found: {in_path}"
        make_pt2_encoder_only(in_path, out_path)

    # PT3: structure / composition
    for in_name, out_name in PT3_FILES:
        in_path = os.path.join(RAW_MODELS_DIR, in_name)
        out_path = os.path.join(ENCODERS_DIR, out_name)
        assert os.path.exists(in_path), f"Input file not found: {in_path}"
        make_pt3_encoder_only(in_path, out_path)

    print("All encoder-only files generated successfully.")