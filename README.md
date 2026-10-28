# UTAE with Meta DINO-v3 Backbone

A deep learning framework for Spatio-Temporal Satellite Image Segmentation tailored for Wildfire Spread Prediction. This project upgrades the original [UTAE](utae.py) (U-Net with Temporal Attention Encoder) architecture by replacing traditional CNN downsampling blocks with **Meta DINO-v3** vision backbones ([dinov3/](dinov3/)), coupled with Lightweight Temporal Attention (L-TAE) and specialized loss functions designed for extreme class imbalance.

Scope of this version:
- Replaces traditional CNN encoder blocks with official Meta DINO-v3 backbones (supporting both ViT with RoPE and ConvNeXt).
- Multi-spectral Channel Adapter mapping $C$-channel Sentinel-2 optical satellite imagery to 3 standard DINO-v3 channels.
- Multi-Scale Feature Neck projecting intermediate representations to 4 hierarchical resolution stages for the UTAE Decoder.
- Specialized loss functions ([WeightedBCEWithLogitsLoss](losses.py#L6-L20), [BinaryFocalLoss](losses.py#L22-L48), [DiceLoss](losses.py#L50-L68)) addressing severe wildfire foreground-background class imbalance.
- Full training and evaluation workflow integrated with [PyTorch Lightning](utae_lightning.py).

---

## 1. Project Overview

- **Problem Statement**: Predict burned scars and active wildfire propagation patterns from multi-temporal optical satellite imagery (Sentinel-2 time series with Day-of-Year positional inputs).
- **Target Audience**: Remote sensing researchers, wildfire hazard analysts, and machine learning engineers developing environmental monitoring solutions.
- **Key Value**: Leverages self-supervised spatial representations from Meta DINO-v3 to capture complex land cover and terrain textures far beyond conventional CNNs, while retaining the temporal sequence modeling capability of L-TAE.

---

## 2. System Overview

### 2.1. End-to-End Architecture Dataflow

```
[Multi-Temporal Sentinel-2 Imagery]
Shape: (B, T, C, H, W) + DOY (B, T)
        │
        ▼
┌────────────────────────────────────────────────────────┐
│ 1. Multi-spectral Channel Adapter                      │
│    Conv2d(C -> 3, 1x1) + BatchNorm + ReLU              │
└───────────────────────┬────────────────────────────────┘
                        │ Shape: (B * T, 3, H, W)
                        ▼
┌────────────────────────────────────────────────────────┐
│ 2. Official Meta DINO-v3 Backbone                      │
│    (DinoVisionTransformer with RoPE / ConvNeXt)        │
│    Extracts Intermediate Layers                        │
└───────────────────────┬────────────────────────────────┘
                        │ 4 intermediate feature maps
                        ▼
┌────────────────────────────────────────────────────────┐
│ 3. DINOv3FeatureNeck                                   │
│    Projects & resizes features for UTAE Decoder:       │
│    - Stage 0: 1x  (H, W)       -> channels[0] (64)     │
│    - Stage 1: 1/2x (H/2, W/2)   -> channels[1] (64)     │
│    - Stage 2: 1/4x (H/4, W/4)   -> channels[2] (64)     │
│    - Stage 3: 1/8x (H/8, W/8)   -> channels[3] (128)    │
└───────────┬────────────────────────────────┬───────────┘
            │ Skip connections               │ Bottleneck features
            │ (Stage 0, 1, 2)                │ (Stage 3: B, T, 128, H/8, W/8)
            │                                ▼
            │                   ┌────────────────────────┐
            │                   │ 4. L-TAE Temporal      │
            │                   │    Attention Bottleneck│
            │                   │    with DOY Encodings  │
            │                   └────────────┬───────────┘
            ▼                                │ Output (B, 128, H/8, W/8)
┌────────────────────────────────────────────▼───────────┐
│ 5. UTAE Multi-Scale Spatial Decoder                    │
│    Upsampling + Skip Connection Concatenation          │
└───────────────────────┬────────────────────────────────┘
                        │
                        ▼ Logits (B, 1, H, W)
┌────────────────────────────────────────────────────────┐
│ 6. Specialized Loss Functions (Weighted BCE / Focal)   │
│    Loss computation against Ground Truth (B, H, W)     │
│    ==> Full Backward Gradient Propagation              │
└────────────────────────────────────────────────────────┘
```

### 2.2. Component Execution
- **DINO-v3 Encoder**: Loaded directly from the local cloned repository [dinov3/](dinov3/), running on GPU/CPU.
- **Training & Validation Engine**: Encapsulated in [UTAELightning](utae_lightning.py) using AdamW optimizer and Cosine Annealing learning rate schedule.

---

## 3. Environment

Recommended operating system and dependency requirements:

| Component | Requirement / Recommendation | Notes |
| :--- | :--- | :--- |
| **Operating System** | Windows 10/11 (64-bit) or Ubuntu 20.04 / 22.04 LTS | Verified and tested on both Windows and Linux |
| **Python** | `3.10.x` | Recommended Python 3.10 |
| **PyTorch** | `>= 2.0.0` (with TorchVision) | Compatible with CPU and CUDA 11.8 / 12.1 |
| **PyTorch Lightning** | `>= 2.0.0` | Powers [UTAELightning](utae_lightning.py) |
| **System Memory (RAM)** | $\ge$ 16 GB | Required for multi-dimensional spatio-temporal tensors |
| **GPU Hardware** | NVIDIA GPU $\ge$ 8 GB VRAM recommended | Recommended for training or fine-tuning DINO-v3 |
| **Model Weights** | Official `.pth` checkpoint from Meta (optional) | Supports random initialization or loading pretrained weights |

---

## 4. Setup

Follow these setup steps in order:

### Step 1: Clone the repository and navigate to the project directory
```bash
git clone https://github.com/your-username/utae_dinov3.git
cd utae_dinov3
```

### Step 2: Create and activate a virtual environment
On Windows:
```bash
python -m venv .venv
.venv\Scripts\activate
```
On Linux / macOS:
```bash
python3 -m venv .venv
source .venv/bin/activate
```
*Expected Result*: The command prompt displays the `(.venv)` prefix.

### Step 3: Install required dependencies
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install pytorch-lightning timm packaging
```
*(If using an NVIDIA GPU with CUDA 11.8, replace the PyTorch index URL with `https://download.pytorch.org/whl/cu118`)*

### Step 4: Verify DINO-v3 availability
The official Meta DINO-v3 code is included in the subfolder [dinov3/](dinov3/). Test the import:
```bash
python -c "import sys; sys.path.insert(0, 'dinov3'); from dinov3.models.vision_transformer import DinoVisionTransformer; print('DINOv3 Meta Imported Successfully!')"
```
*Expected Output*:
```text
DINOv3 Meta Imported Successfully!
```

---

## 5. Run & Verify

Run the verification commands below to validate the pipeline from forward pass to loss calculation and gradient flow.

### 5.1. Forward Pass Verification (UTAE + DINO-v3 ViT)
Execute:
```bash
python -c "import torch; from utae import UTAE; m = UTAE(input_dim=12, encoder_type='dinov3', dinov3_size='small'); x = torch.randn(2, 4, 12, 128, 128); doys = torch.tensor([[10, 20, 30, 40], [15, 25, 35, 45]]); out, att = m(x, doys, return_att=True); print('Output shape:', out.shape); print('Temporal Attention shape:', att.shape)"
```
*Expected Output*:
```text
Output shape: torch.Size([2, 1, 128, 128])
Temporal Attention shape: torch.Size([16, 2, 4, 16, 16])
```

### 5.2. Loss Function & Backward Gradient Flow Verification
Execute:
```bash
python -c "import torch; from utae import UTAE; from losses import get_loss_function; m = UTAE(input_dim=12, encoder_type='dinov3', dinov3_size='small'); x = torch.randn(2, 4, 12, 128, 128); doys = torch.tensor([[10, 20, 30, 40], [15, 25, 35, 45]]); y = torch.randint(0, 2, (2, 128, 128)).float(); out = m(x, doys).squeeze(1); criterion = get_loss_function('BCE', pos_class_weight=10.0); loss = criterion(out, y); loss.backward(); print('Loss value computed:', round(loss.item(), 4)); print('DINOv3 Attention Grad Status:', m.encoder.backbone.blocks[0].attn.qkv.weight.grad is not None)"
```
*Expected Output*:
```text
Loss value computed: 3.2732
DINOv3 Attention Grad Status: True
```

---

## 6. Configuration & Hyperparameters

Key parameters for initializing the model and training modules:

| Parameter | Type | Default | Description |
| :--- | :---: | :---: | :--- |
| `input_dim` / `n_channels` | `int` | `12` | Number of input satellite spectral bands (e.g., Sentinel-2 B02–B12). |
| `encoder_type` | `str` | `"dinov3"` | Spatial encoder type: `"dinov3"` (ViT), `"dinov3_convnext"`, or `"cnn"` (original UTAE). |
| `dinov3_size` | `str` | `"small"` | DINO-v3 model size: `"small"`, `"base"`, `"large"` (or `"tiny"` for ConvNeXt). |
| `dinov3_weights` | `str` / `None` | `None` | Path to official Meta `.pth` pretrained checkpoint. |
| `freeze_encoder` | `bool` | `False` | `True` to freeze DINO-v3 backbone; `False` for end-to-end fine-tuning. |
| `loss_function` | `str` | `"BCE"` | Selected loss function: `"BCE"`, `"FOCAL"`, or `"DICE"`. |
| `pos_class_weight` | `float` | `10.0` | Positive class penalty weight in [WeightedBCEWithLogitsLoss](losses.py#L6-L20). |
| `lr` | `float` | `0.001` | Initial learning rate for AdamW optimizer. |
| `weight_decay` | `float` | `1e-4` | L2 regularization coefficient to prevent overfitting. |

---

## 7. AI Model: DINO-v3 Updates & Loss Function Implementation

### 7.1. Model Specifications
- **Model Name**: UTAE-DINOv3 (Spatio-Temporal Vision Transformer U-Net).
- **Task**: Pixel-level binary segmentation for wildfire burn scar detection and spread boundary prediction.
- **Input**:
  - Spatio-temporal image tensor $X \in \mathbb{R}^{B \times T \times C \times H \times W}$ ($C=12$ Sentinel-2 bands, $H=W=128$, across $T$ observation dates).
  - Day-of-Year tensor $DOY \in \mathbb{R}^{B \times T}$ for harmonic temporal positional encoding.
- **Output**: Spatial segmentation logits $\hat{Y} \in \mathbb{R}^{B \times 1 \times H \times W}$.
- **Decision Threshold**: Apply Sigmoid $\sigma(\hat{Y}) \ge 0.50$ to generate the binary wildfire mask.

---

### 7.2. Code Locations for DINO-v3 Integration

All modifications introducing DINO-v3 are encapsulated in specific files and methods:

#### 1. In [dinov3_encoder.py](dinov3_encoder.py) (New Module)
- **Class `DINOv3Encoder` (Lines 72–226)**:
  - **Multi-spectral Channel Adapter (Lines 104–111)**:
    ```python
    if self.input_dim != 3:
        self.adapter = nn.Sequential(
            nn.Conv2d(self.input_dim, 3, kernel_size=1, bias=False),
            nn.BatchNorm2d(3),
            nn.ReLU(inplace=True),
        )
    ```
    Projects $C=12$ Sentinel-2 channels into 3 channels for DINO-v3.
  - **Meta DINO-v3 Backbone Initialization (Lines 114–148)**:
    Imports and instantiates official ViT (`vit_small`, `vit_base`, `vit_large`) or ConvNeXt models from [dinov3/](dinov3/).
  - **Intermediate Feature Extraction (Lines 197–201)**:
    ```python
    intermediate_features = self.backbone.get_intermediate_layers(
        x_3ch, n=self.layer_indices, reshape=True
    )
    ```
  - **Time-Series Handling via `smart_forward` (Lines 206–226)**:
    Folds batch and sequence dimensions `(B, T, C, H, W) -> (B * T, C, H, W)`, runs DINO-v3 feature extraction, and unfolds back into 4 hierarchical tensors `(B, T, C_stage, H_stage, W_stage)`.
- **Class `DINOv3FeatureNeck` (Lines 35–65)**:
  Projects and interpolates the 4 DINO-v3 feature stages into resolution levels $[1\times, 1/2\times, 1/4\times, 1/8\times]$ required by the UTAE Decoder skip connections.

#### 2. In [utae.py](utae.py) (Updated Architecture)
- **Import Statements (Lines 5–10)**: Imports [DINOv3Encoder](dinov3_encoder.py).
- **`__init__` Method (Lines 242–270)**:
  Adds `encoder_type="dinov3"`, `dinov3_size="small"`, `dinov3_weights=None`, `freeze_encoder=False`. Instantiates `self.encoder = DINOv3Encoder(...)` in place of the CNN encoder blocks.
- **`forward` Method (Lines 342–351)**:
  ```python
  if self.encoder is not None:
      feature_maps = self.encoder.smart_forward(input)
  else:
      # Legacy CNN branch
      ...
  ```

---

### 7.3. Loss Function Implementation & Training Mechanics

Wildfire segmentation suffers from **severe class imbalance**: burned pixels (positive class) typically account for only **1% to 5%** of the entire scene, while unburned land (background) dominates the rest. The loss system is implemented in [losses.py](losses.py) and wired directly into [utae_lightning.py](utae_lightning.py):

#### 1. In [losses.py](losses.py)

- **Class `WeightedBCEWithLogitsLoss` (Lines 6–20)**:
  ```python
  class WeightedBCEWithLogitsLoss(nn.Module):
      def __init__(self, pos_class_weight: float = 10.0):
          super(WeightedBCEWithLogitsLoss, self).__init__()
          self.pos_class_weight = pos_class_weight

      def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
          if logits.ndim == targets.ndim + 1:
              logits = logits.squeeze(1)
          pos_weight = torch.tensor([self.pos_class_weight], device=logits.device, dtype=logits.dtype)
          return F.binary_cross_entropy_with_logits(logits, targets.float(), pos_weight=pos_weight)
  ```
  - **Key Role in Code**: Applies PyTorch's `pos_weight` to multiply the penalty on false negative burn predictions by 10x (`pos_class_weight=10.0`). This prevents DINO-v3 from collapsing into predicting only background zeros.
  - Squeezes `logits` from `(B, 1, H, W)` to match target mask dimensions `(B, H, W)`.

- **Class `BinaryFocalLoss` (Lines 22–48)**:
  ```python
  class BinaryFocalLoss(nn.Module):
      def __init__(self, alpha: float = 0.25, gamma: float = 2.0, reduction: str = "mean"):
          ...
      def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
          ...
          p = torch.sigmoid(logits)
          ce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
          p_t = p * targets + (1 - p) * (1 - targets)
          alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
          loss = alpha_t * ((1 - p_t) ** self.gamma) * ce_loss
          return loss.mean()
  ```
  - **Key Role in Code**: The focusing parameter $\gamma = 2.0$ downweights easy background predictions, driving the loss gradients toward hard ambiguous pixels along active fire perimeters.

- **Class `DiceLoss` (Lines 50–68)**:
  Directly optimizes the spatial overlap score `(2 * intersection + smooth) / (cardinality + smooth)`, making it insensitive to large background areas.

- **Factory `get_loss_function` (Lines 70–85)**:
  Provides a unified factory returning the requested loss module (`"BCE"`, `"FOCAL"`, or `"DICE"`).

---

#### 2. In [utae_lightning.py](utae_lightning.py)

- **Criterion Initialization in `__init__` (Line 70)**:
  ```python
  self.criterion = get_loss_function(name=loss_function, pos_class_weight=pos_class_weight)
  ```
- **Training Loss in `training_step` (Lines 82–86)**:
  ```python
  logits = self(x, doys).squeeze(1)
  loss = self.criterion(logits, y)
  self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True)
  return loss
  ```
- **Evaluation in `validation_step` (Lines 95–102)**:
  Evaluates `loss = self.criterion(logits, y)` alongside performance metrics (`AveragePrecision`, `F1`) from [metrics.py](metrics.py).

---

#### 3. Gradient Flow Between Loss and DINO-v3 in Code

When `loss.backward()` executes during training:
- **Full Fine-Tuning (`freeze_encoder=False`)**:
  Gradients flow backward from `self.criterion` through Decoder $\to$ L-TAE $\to$ Neck $\to$ and directly update Meta DINO-v3's attention weight tensors:
  ```python
  # Verification: gradient exists on DINO-v3 Attention QKV matrix
  m.encoder.backbone.blocks[0].attn.qkv.weight.grad is not None  # -> True
  ```
- **Feature Extraction (`freeze_encoder=True`)**:
  In [dinov3_encoder.py](dinov3_encoder.py#L182-L186), the backbone parameters are frozen:
  ```python
  for param in self.backbone.parameters():
      param.requires_grad = False
  ```
  `loss.backward()` updates only the Adapter, Feature Neck, L-TAE, and Decoder parameters, reducing GPU memory footprint during training.

---

### 7.4. Known Limitations
- Spatial dimensions ($H, W$) should be divisible by 16 due to DINO-v3 ViT's $16 \times 16$ patch projection.
- High GPU memory usage when using DINO-v3 ViT-Large with long temporal sequences ($T > 8$). The `small` or `base` variants are recommended for real-time applications.

---

## 8. Usage & API

### 8.1. Standalone PyTorch Inference

```python
import torch
from utae import UTAE

# 1. Simulate Sentinel-2 input: Batch=2, T=4 timestamps, 12 spectral channels, 128x128
x = torch.randn(2, 4, 12, 128, 128)
doys = torch.tensor([[15, 30, 45, 60], [10, 25, 40, 55]])  # Day-of-year

# 2. Instantiate UTAE with DINO-v3 ViT-Small
model = UTAE(
    input_dim=12,
    encoder_type="dinov3",     # "dinov3" (ViT) or "dinov3_convnext"
    dinov3_size="small",       # "small" (21M params), "base", or "large"
    dinov3_weights=None,       # Or path to 'models/dinov3_vits16.pth'
    freeze_encoder=False,
)

# 3. Perform inference
model.eval()
with torch.no_grad():
    logits, attention_weights = model(x, batch_positions=doys, return_att=True)
    probs = torch.sigmoid(logits)

print("Output Logits Shape:", logits.shape)                  # torch.Size([2, 1, 128, 128])
print("L-TAE Attention Shape:", attention_weights.shape)      # torch.Size([16, 2, 4, 16, 16])
```

### 8.2. Training with PyTorch Lightning and Specialized Loss

```python
import pytorch_lightning as pl
from torch.utils.data import DataLoader, TensorDataset
from utae_lightning import UTAELightning

# 1. Initialize Lightning Module with Weighted BCE Loss
lightning_module = UTAELightning(
    n_channels=12,
    encoder_type="dinov3",
    dinov3_size="small",
    loss_function="BCE",          # "BCE", "FOCAL", or "DICE"
    pos_class_weight=10.0,        # Penalizes burned scar false negatives
    lr=1e-4,
    weight_decay=1e-4,
)

# 2. Create sample DataLoader
dummy_x = torch.randn(8, 4, 12, 128, 128)
dummy_y = torch.randint(0, 2, (8, 128, 128)).float()
dummy_doys = torch.tensor([[10, 20, 30, 40]] * 8)
train_loader = DataLoader(TensorDataset(dummy_x, dummy_y, dummy_doys), batch_size=2)

# 3. Fit using PyTorch Lightning Trainer
trainer = pl.Trainer(max_epochs=5, accelerator="auto")
trainer.fit(lightning_module, train_loader)
```

---

## 9. Troubleshooting

Common issues and recommended resolutions:

| Error / Symptom | Potential Cause | Resolution |
| :--- | :--- | :--- |
| `ImportError: Failed to import official DINO-v3 models` | The `dinov3` subfolder is missing or `sys.path` is incorrect. | Ensure the [dinov3/](dinov3/) directory exists at the root of the repository. |
| `RuntimeError: CUDA out of memory` | DINO-v3 ViT requires high VRAM when coupled with long time series ($T$). | Reduce `batch_size`, reduce input spatial size $H, W$, use `dinov3_size="small"`, or set `freeze_encoder=True`. |
| `AssertionError: Spatial dimensions must be divisible by 16` | Input dimensions $H$ or $W$ are not divisible by 16 for ViT patch division. | Resize or pad input images to multiples of 16 (e.g., $128 \times 128$, $256 \times 256$). |
| `KeyError / Unexpected key in state_dict` | Checkpoint architecture does not match the configured ViT / ConvNeXt variant. | Ensure `arch` and `dinov3_size` match the checkpoint file specified in `dinov3_weights`. |
| `Loss value is NaN / Inf` | Learning rate is too high or input satellite imagery contains invalid NoData values. | Reduce learning rate to `1e-4` or `5e-5`; check input normalization (e.g., divide by 10,000 for Sentinel-2 L2A). |
