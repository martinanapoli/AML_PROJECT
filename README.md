
# SELF SUPERVISED X-RAY ANOMLAY CLASSIFICATION
_Analyzing the role of pretraining data, induced invariances and pretext-task fusions._

Self-supervised learning (SSL) learns visual representations without manual labels using pretext tasks. However, different pretext tasks impose different inductive biases and invariances (texture/appearance, spatial context, global structure) which can strongly impact transfer performance.
This project systematically evaluates which factors make a pretext task useful for downstream anomaly-related classification in chest X-rays:
- **(H1)** Is visual similarity between pretraining and downstream data enough to explain transfer performance, or does the induced invariance matter equally (or more)?
- **(H2)** Can we improve downstream results by combining complementary invariances, using multi-encoder feature fusion?
  
We implement 3 pretext tasks × 3 pretraining datasets (factorial design), extract encoder-only checkpoints and evaluate them on downstream tasks with :
-  **Linear evaluation (frozen encoder + linear head)**
-  **Fine-tuning (train encoder + head with different LRs)**
We also include a downstream-from-scratch baseline.

## DATASETS

This work uses **four datasets**: three for self-supervised pretraining and one for downstream evaluation.

### Pretraining datasets
- **D1: ImageNet (subset)**  
  Generic natural images used as a non-medical baseline.

- **D2: Brain Stroke (CT / brain imaging)**  
  Medical images with anatomical structure, but significantly different from chest X-rays.

- **D3: Mammograms**  
  Grayscale radiographic images that are visually closer to chest X-rays and represent the most domain-aligned pretraining dataset.

### Downstream dataset
- **ChestX-ray14**  
  Used exclusively for supervised downstream evaluation:
  - **Binary classification**: *No Finding* vs *Anomaly* (any pathology)
  - **3-class classification**: *No Finding* vs *Pneumonia* vs *Other pathology*

> **Note** : Chest X-ray images are grayscale. To maintain compatibility with ResNet-based models, images are replicated to **3 channels** before being passed to the network.

## METHODS

### Pretext tasks
All pretext tasks use a **ResNet18** backbone and are trained **independently** on each pretraining dataset.

- **PT1 — Rotation prediction (4-way classification)**  
  Predict the rotation angle applied to the input image {0°, 90°, 180°, 270°}.  
  This task is intended to emphasize **appearance-level and orientation cues**.

- **PT2 — Inpainting (masked reconstruction)**  
  Encoder–decoder architecture composed of a ResNet18 encoder and a U-Net–like decoder.  
  A **masked L1 loss** is computed only inside the missing region.  
  This task is designed to promote learning of **spatial context** and **local/global consistency**.

- **PT3 — Structural consistency (binary classification)**  
  Binary classification of whether an image is **structurally consistent** or **structurally corrupted**  
  (e.g., flips, half swaps, quadrant shuffling, strong rotations).  
  This task emphasizes **global structure** and **anatomical composition**.

### Encoder-only checkpoint extraction
Because the three pretext-task implementations store model weights differently, all pretrained models are standardized into **encoder-only checkpoints** compatible with a unified downstream pipeline:

- **PT1**: remove `fc.*` weights and set `fc = Identity`
- **PT2**: extract only the encoder layers from the encoder–decoder architecture
- **PT3**: load and directly save the `encoder_state_dict`

All extracted checkpoints are saved using the following naming convention:

### Downstream evaluation protocols
Each encoder is evaluated using two standard downstream protocols:

- **Linear evaluation**  
  The encoder is frozen and only a linear classification head is trained.

- **Fine-tuning**  
  The encoder and classification head are trained jointly, using:
  - a smaller learning rate for encoder parameters
  - a larger learning rate for the classification head

**Evaluation metrics**:
- Loss
- Accuracy
- Weighted F1-score
- Confusion matrices  
- *(Optional)* ROC-AUC and balanced accuracy, when implemented


### Multi-encoder feature fusion
To test **H2**, a **MultiEncoderFusion** strategy is implemented to integrate complementary invariances learned by different pretext tasks.

**Fusion procedure**:
- Load multiple encoder-only checkpoints
- Extract a **512-dimensional** feature vector from each encoder
- Concatenate feature vectors:
  - 2 encoders → 1024 dimensions
  - 3 encoders → 1536 dimensions
- Train a shared classification head on top of the fused representation

**Fusion training modes**:
- **Linear fusion**: all encoders are frozen
- **Fine-tuned fusion**: encoders and fusion head are jointly optimized using separate learning rates

**Fusion configurations (D3-aligned setting)**:
- PT1 + PT2
- PT1 + PT3
- PT2 + PT3
- PT1 + PT2 + PT3


## IMPLEMENTATION

The full implementation consists of the following steps, which **must be executed in order**.

## Step 1: Pretext-task training (factorial design)

Each pretext task is trained **independently** on each pretraining dataset, following a full factorial design.

### Pretext tasks
- **PT1**: Rotation prediction  
- **PT2**: Inpainting  
- **PT3**: Structural consistency  

### Pretraining datasets
- **D1**: ImageNet (subset)  
- **D2**: Brain Stroke  
- **D3**: Mammograms  

This step produces **9 pretext-task checkpoints** (3 tasks × 3 datasets), saved as .pth files.

- One checkpoint per *(pretext task, dataset)* pair  
- Models correspond to the complete pretext-task architectures  

> **Output**: 9 pretext-task .pth files.


## Step 2: Encoder-only checkpoint extraction

Because each pretext-task implementation stores model weights in different formats, all pretrained models are standardized into a common **encoder-only representation** before downstream evaluation.

### Standardization rules
- **PT1**: remove `fc.*` weights and set `fc = Identity`  
- **PT2**: extract only the encoder layers from the encoder–decoder architecture  
- **PT3**: directly save the `encoder_state_dict`  

## Step 3: Downstream evaluation (linear evaluation and fine-tuning)

Each encoder-only checkpoint is evaluated using **two downstream protocols** to assess both representation quality and adaptability.

### Downstream evaluation protocols
- **Linear evaluation**  
  - Encoder weights are frozen  
  - A linear classification head is trained on top of the encoder  

- **Fine-tuning**  
  - Encoder and classification head are trained jointly  
  - A smaller learning rate is used for the encoder  
  - A larger learning rate is used for the classification head  

All evaluations are repeated for the **9 encoder-only checkpoints** under identical training conditions.

> **Output**: Downstream performance metrics and logs for all single-pretext encoders under linear evaluation and fine-tuning.

## Step 4: Multi-encoder feature fusion (linear and fine-tuned fusion)

To evaluate whether complementary invariances can be integrated, multi-encoder fusion experiments are conducted using both **frozen** and **fine-tuned** encoders.

### Fusion procedure
- Load multiple encoder-only checkpoints  
- Extract a **512-dimensional** feature vector from each encoder  
- Concatenate feature vectors  
- Train a shared classification head on top of the fused representation  

### Fusion training modes
- **Linear fusion**: all encoders are frozen  
- **Fine-tuned fusion**: encoders and fusion head are jointly optimized using separate learning rates  

### Fusion configurations
- PT1 + PT2  
- PT1 + PT3  
- PT2 + PT3  
- PT1 + PT2 + PT3  

> **Output**: Downstream performance metrics for all fusion configurations under linear and fine-tuned settings.

## Execution Summary

1. Train all pretext tasks on all pretraining datasets  
2. Extract encoder-only checkpoints  
3. Run downstream evaluation for all single-pretext encoders  
   - Linear evaluation (frozen encoder)  
   - Fine-tuning (joint encoder + head training)  
4. Run multi-encoder fusion experiments  
   - Linear fusion (frozen encoders)  
   - Fine-tuned fusion (jointly optimized encoders and fusion head)  

