import torch
import torch.nn as nn
try:
    import pytorch_lightning as pl
    PL_AVAILABLE = True
except ImportError:
    PL_AVAILABLE = False
    pl = object

try:
    from .utae import UTAE
    from .losses import get_loss_function
    from .metrics import compute_metrics
except ImportError:
    from utae import UTAE
    from losses import get_loss_function
    from metrics import compute_metrics


class UTAELightning(pl.LightningModule if PL_AVAILABLE else nn.Module):
    """
    Lightning Module wrapper for UTAE model.
    Matches the author's original BaseModel + UTAELightning implementation.
    """
    def __init__(
        self,
        n_channels: int = 12,
        pos_class_weight: float = 10.0,
        loss_function: str = "BCE",
        lr: float = 0.001,
        weight_decay: float = 1e-4,
        encoder_widths=[64, 64, 64, 128],
        decoder_widths=[32, 32, 64, 128],
        out_conv=[32, 1],
        n_head: int = 16,
        d_model: int = 256,
        d_k: int = 4,
        encoder_type: str = "dinov3",
        dinov3_size: str = "small",
        dinov3_weights: str = None,
        freeze_encoder: bool = False,
        *args,
        **kwargs,
    ):
        super().__init__()
        if PL_AVAILABLE:
            self.save_hyperparameters()

        self.lr = lr
        self.weight_decay = weight_decay
        self.pos_class_weight = pos_class_weight
        self.loss_name = loss_function

        # 1. Model architecture
        self.model = UTAE(
            input_dim=n_channels,
            encoder_widths=encoder_widths,
            decoder_widths=decoder_widths,
            out_conv=out_conv,
            n_head=n_head,
            d_model=d_model,
            d_k=d_k,
            encoder_type=encoder_type,
            dinov3_size=dinov3_size,
            dinov3_weights=dinov3_weights,
            freeze_encoder=freeze_encoder,
        )

        # 2. Loss function
        self.criterion = get_loss_function(name=loss_function, pos_class_weight=pos_class_weight)

    def forward(self, x: torch.Tensor, doys: torch.Tensor = None, return_att: bool = False):
        return self.model(x, batch_positions=doys, return_att=return_att)

    def training_step(self, batch, batch_idx):
        if len(batch) == 3:
            x, y, doys = batch
        else:
            x, y = batch
            doys = None

        logits = self(x, doys).squeeze(1)
        loss = self.criterion(logits, y)
        if PL_AVAILABLE:
            self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        if len(batch) == 3:
            x, y, doys = batch
        else:
            x, y = batch
            doys = None

        logits = self(x, doys).squeeze(1)
        loss = self.criterion(logits, y)
        metrics = compute_metrics(logits, y)
        if PL_AVAILABLE:
            self.log("val_loss", loss, prog_bar=True, on_epoch=True)
            self.log("val_AP", metrics["AveragePrecision"], prog_bar=True, on_epoch=True)
            self.log("val_F1", metrics["F1"], prog_bar=True, on_epoch=True)
        return {"val_loss": loss, **metrics}

    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=50, eta_min=1e-6
        )
        return {"optimizer": optimizer, "lr_scheduler": scheduler}

    def load_checkpoint(self, checkpoint_path: str, strict: bool = False):
        """
        Load pretrained weights (.pth or .pth.tar) from author's repo or HuggingFace.
        """
        ckpt = torch.load(checkpoint_path, map_location="cpu")
        state_dict = ckpt.get("state_dict", ckpt)
        prefix = "model."
        cleaned = {}
        for k, v in state_dict.items():
            if k.startswith(prefix):
                cleaned[k[len(prefix):]] = v
            else:
                cleaned[k] = v
        self.model.load_state_dict(cleaned, strict=strict)
        print(f"Loaded weights successfully from {checkpoint_path}")
