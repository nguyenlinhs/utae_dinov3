import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

# Ensure the cloned dinov3 repository is accessible on sys.path
_DINOV3_REPO_DIR = Path(__file__).resolve().parent / "dinov3"
if str(_DINOV3_REPO_DIR) not in sys.path:
    sys.path.insert(0, str(_DINOV3_REPO_DIR))

# Import official Meta DINOv3 models directly from the cloned repository
try:
    from dinov3.models.vision_transformer import (
        DinoVisionTransformer,
        vit_small,
        vit_base,
        vit_large,
    )
    from dinov3.models.convnext import ConvNeXt, get_convnext_arch
except ImportError as err:
    raise ImportError(
        f"Failed to import official DINO-v3 models from {_DINOV3_REPO_DIR}. "
        "Please ensure the repository is properly cloned: 'git clone https://github.com/facebookresearch/dinov3.git'"
    ) from err


# ============================================================================
# Multi-Scale Feature Neck for UTAE Skip Connections
# ============================================================================

class DINOv3FeatureNeck(nn.Module):
    """
    Projects intermediate feature maps from official Meta DINOv3 backbones into
    the 4 hierarchical multi-scale levels required by UTAE Decoder:
      - Stage 0: 1x  resolution (H, W),         channels = out_channels[0]
      - Stage 1: 1/2x resolution (H/2, W/2),     channels = out_channels[1]
      - Stage 2: 1/4x resolution (H/4, W/4),     channels = out_channels[2]
      - Stage 3: 1/8x resolution (H/8, W/8),     channels = out_channels[3] (Bottleneck -> LTAE)
    """
    def __init__(self, in_dims: List[int], out_channels: List[int] = [64, 64, 64, 128]):
        super().__init__()
        self.out_channels = out_channels
        self.projs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_dim, out_ch, kernel_size=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
            )
            for in_dim, out_ch in zip(in_dims, out_channels)
        ])

    def forward(self, features: List[torch.Tensor], target_size: Tuple[int, int]) -> List[torch.Tensor]:
        H, W = target_size
        f0, f1, f2, f3 = features

        m0 = F.interpolate(self.projs[0](f0), size=(H, W), mode="bilinear", align_corners=False)
        m1 = F.interpolate(self.projs[1](f1), size=(H // 2, W // 2), mode="bilinear", align_corners=False)
        m2 = F.interpolate(self.projs[2](f2), size=(H // 4, W // 4), mode="bilinear", align_corners=False)
        m3 = F.interpolate(self.projs[3](f3), size=(H // 8, W // 8), mode="bilinear", align_corners=False)

        return [m0, m1, m2, m3]


# ============================================================================
# Unified DINOv3 Encoder wrapping the Official Cloned Meta DINOv3 Repository
# ============================================================================

class DINOv3Encoder(nn.Module):
    """
    Spatial Encoder for UTAE powered directly by official Meta DINO-v3 models
    from the cloned `dinov3` repository (github.com/facebookresearch/dinov3).

    Supported Backbones:
      - ViT variants:     'dinov3_vits16', 'dinov3_vitb16', 'dinov3_vitl16'
      - ConvNeXt variants: 'dinov3_convnext_tiny', 'dinov3_convnext_small', 'dinov3_convnext_base'

    Key Features:
      - Direct instantiation of official DinoVisionTransformer or ConvNeXt
      - Multi-spectral input channel adapter (supports any input_dim: e.g. 10 or 12 Sentinel-2 bands)
      - get_intermediate_layers extraction natively supported by official Meta models
      - smart_forward for handling time-series batches (B, T, C, H, W) -> (B, T, C_stage, H_stage, W_stage)
      - Official weight loading support (.pth file)
    """
    def __init__(
        self,
        input_dim: int = 12,
        arch: str = "vit",           # 'vit' or 'convnext'
        model_size: str = "small",   # 'small', 'base', 'large' (or 'tiny')
        out_channels: List[int] = [64, 64, 64, 128],
        pretrained_weights: Optional[str] = None,
        freeze_backbone: bool = False,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.arch = arch.lower()
        self.model_size = model_size.lower()
        self.out_channels = out_channels

        # 1. Multi-spectral Channel Adapter (projects input_dim to 3 channels for Meta's backbone)
        if input_dim != 3:
            self.input_adapter = nn.Sequential(
                nn.Conv2d(input_dim, 3, kernel_size=1, bias=False),
                nn.BatchNorm2d(3),
                nn.ReLU(inplace=True),
            )
        else:
            self.input_adapter = nn.Identity()

        # 2. Instantiate Official Meta Model from dinov3 repo
        if "vit" in self.arch:
            if self.model_size == "small":
                self.backbone: DinoVisionTransformer = vit_small(patch_size=16)
                embed_dim = 384
            elif self.model_size == "large":
                self.backbone: DinoVisionTransformer = vit_large(patch_size=16)
                embed_dim = 1024
            else:  # base
                self.backbone: DinoVisionTransformer = vit_base(patch_size=16)
                embed_dim = 768

            # Take 4 intermediate layers evenly spaced across depth
            depth = len(self.backbone.blocks)
            step = depth // 4
            self.layer_indices = [step - 1, 2 * step - 1, 3 * step - 1, depth - 1]
            in_dims = [embed_dim, embed_dim, embed_dim, embed_dim]

        elif "convnext" in self.arch:
            if self.model_size in ["tiny", "small"]:
                arch_name = "convnext_tiny" if self.model_size == "tiny" else "convnext_small"
                dims = [96, 192, 384, 768]
            elif self.model_size == "large":
                arch_name = "convnext_large"
                dims = [192, 384, 768, 1536]
            else:  # base
                arch_name = "convnext_base"
                dims = [128, 256, 512, 1024]

            convnext_fn = get_convnext_arch(arch_name)
            self.backbone: ConvNeXt = convnext_fn()
            self.layer_indices = [0, 1, 2, 3]
            in_dims = dims
        else:
            raise ValueError(f"Unsupported DINO-v3 arch '{arch}'. Must be 'vit' or 'convnext'.")

        # 3. Initialize official Meta backbone weights (DINOv3 models use uninitialized memory by default)
        if hasattr(self.backbone, "init_weights"):
            self.backbone.init_weights()

        # 4. Multi-scale Feature Neck to feed UTAE Decoder skip connections
        self.neck = DINOv3FeatureNeck(in_dims=in_dims, out_channels=out_channels)

        # 4. Optional Pretrained Checkpoint Loading
        if pretrained_weights is not None:
            self.load_pretrained_weights(pretrained_weights)

        # 5. Optional Freezing
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

    def load_pretrained_weights(self, weights_path: str):
        """
        Loads official Meta DINOv3 pre-trained weights from local .pth file.
        """
        if not os.path.isfile(weights_path):
            print(f"[DINOv3Encoder] Warning: Weights file not found: {weights_path}")
            return

        try:
            state_dict = torch.load(weights_path, map_location="cpu")
            if "model" in state_dict:
                state_dict = state_dict["model"]
            elif "teacher" in state_dict:
                state_dict = state_dict["teacher"]

            # Load into the official backbone
            missing, unexpected = self.backbone.load_state_dict(state_dict, strict=False)
            print(f"[DINOv3Encoder] Successfully loaded official Meta weights from: {weights_path}")
            print(f"  Missing keys: {len(missing)}, Unexpected keys: {len(unexpected)}")
        except Exception as e:
            print(f"[DINOv3Encoder] Error loading checkpoint: {e}")

    def forward_features_2d(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        Forward pass for a 4D tensor (B_total, C, H, W).
        Returns list of 4 multi-scale feature maps at scales: [1x, 1/2x, 1/4x, 1/8x].
        """
        B_total, C, H, W = x.shape
        # Adapt input channels if multi-spectral (e.g. 10/12 channels -> 3 channels)
        x_3ch = self.input_adapter(x)

        # Call official Meta get_intermediate_layers method natively!
        intermediate_features = self.backbone.get_intermediate_layers(
            x_3ch,
            n=self.layer_indices,
            reshape=True,
        )

        # Reassemble to 4 hierarchical multi-scale feature maps for UTAE
        return self.neck(list(intermediate_features), target_size=(H, W))

    def smart_forward(self, input: torch.Tensor) -> List[torch.Tensor]:
        """
        Processes either:
          - (B, T, C, H, W) for temporal sequence
          - (B, C, H, W) for single image
        Returns:
          List of 4 feature maps, each reshaped back to (B, T, C_stage, H_stage, W_stage)
        """
        if len(input.shape) == 4:
            return self.forward_features_2d(input)

        b, t, c, h, w = input.shape
        x_flat = input.view(b * t, c, h, w)
        multi_scale_features = self.forward_features_2d(x_flat)

        outputs = []
        for feat in multi_scale_features:
            _, c_out, h_out, w_out = feat.shape
            outputs.append(feat.view(b, t, c_out, h_out, w_out))

        return outputs
