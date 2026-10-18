"""
UTAE Model Package for Wildfire Spread Time-Series Prediction
"""

from .positional_encoding import PositionalEncoder
from .ltae import LTAE2d, MultiHeadAttention, ScaledDotProductAttention
from .utae import (
    UTAE,
    Temporal_Aggregator,
    UpConvBlock,
    DownConvBlock,
    ConvBlock,
    ConvLayer,
    TemporallySharedBlock,
)
from .dinov3_encoder import DINOv3Encoder
from .losses import (
    WeightedBCEWithLogitsLoss,
    BinaryFocalLoss,
    DiceLoss,
    get_loss_function,
)
from .metrics import (
    compute_average_precision,
    compute_metrics,
)
from .utae_lightning import UTAELightning

__all__ = [
    "PositionalEncoder",
    "LTAE2d",
    "MultiHeadAttention",
    "ScaledDotProductAttention",
    "DINOv3Encoder",
    "UTAE",
    "Temporal_Aggregator",
    "UpConvBlock",
    "DownConvBlock",
    "ConvBlock",
    "ConvLayer",
    "TemporallySharedBlock",
    "WeightedBCEWithLogitsLoss",
    "BinaryFocalLoss",
    "DiceLoss",
    "get_loss_function",
    "compute_average_precision",
    "compute_metrics",
    "UTAELightning",
]
