# UTAE với Meta DINO-v3 Backbone: Hướng Dẫn & Tài Liệu Thay Đổi Cập Nhật

Tài liệu này tổng hợp chi tiết toàn bộ các điểm **thay đổi, cập nhật và bổ sung** trong kiến trúc mô hình **UTAE (U-Net with Temporal Attention Encoder)** khi nâng cấp bộ mã hóa không gian (Spatial Encoder) từ các khối CNN truyền thống sang **Meta DINO-v3** chính thức (`facebookresearch/dinov3`).

---

## 1. Tổng Quan Mục Tiêu Nâng Cấp

Mô hình UTAE nguyên bản sử dụng mạng tích chập tuần tự (CNN DownConvBlocks) để trích xuất đặc trưng không gian trên từng ảnh vệ tinh đa phổ (Sentinel-2) trước khi đưa vào khối chú ý thời gian L-TAE (Lightweight Temporal Attention Encoder).

Trong bản cập nhật này, kiến trúc được nâng cấp toàn diện:
- **Tích hợp trực tiếp mã nguồn DINO-v3 chính thức từ Meta Research** (thư mục [dinov3/](file:///d:/THChuyenDe/utae_model/dinov3)).
- **Hỗ trợ 2 họ backbone hiện đại nhất**:
  - **DINO-v3 ViT** (Vision Transformer với Rotary Positional Embeddings - RoPE): `small` (21M), `base` (86M), `large` (300M).
  - **DINO-v3 ConvNeXt**: `tiny` (28M), `small` (50M), `base` (88M), `large` (198M).
- **Bộ điều hợp đa phổ (Multi-spectral Channel Adapter)**: Tự động chuyển đổi tensor đa phổ của vệ tinh ($C$ kênh, ví dụ 10 hoặc 12 kênh quang học Sentinel-2) sang 3 kênh chuẩn cho DINO-v3 mà vẫn giữ nguyên thông tin ngữ cảnh.
- **Multi-Scale Feature Neck**: Trích xuất các tầng đặc trưng trung gian (`intermediate layers`) từ DINO-v3 và ánh xạ chuẩn xác vào 4 tầng tỷ lệ độ phân giải $[1\times, 1/2\times, 1/4\times, 1/8\times]$ của Decoder UTAE.
- **Tương thích hoàn toàn ngược**: Vẫn giữ tùy chọn chạy kiến trúc CNN gốc (`encoder_type="cnn"`).

---

## 2. Bảng Đối Chiếu Các Vị Trí File Thay Đổi & Tạo Mới

| Tên File | Trạng Thái | Mô Tả & Chi Tiết Thay Đổi |
| :--- | :---: | :--- |
| [dinov3/](file:///d:/THChuyenDe/utae_model/dinov3) | **Thêm mới** | Mã nguồn chính thức từ repo Meta Research (`facebookresearch/dinov3`), chứa các module độc quyền: `DinoVisionTransformer`, `ConvNeXt`, `RopePositionEmbedding`, Attention blocks. |
| [dinov3_encoder.py](file:///d:/THChuyenDe/utae_model/dinov3_encoder.py) | **Tạo mới** | Module cầu nối không gian (`DINOv3Encoder` & `DINOv3FeatureNeck`): kết nối backbone DINO-v3 với decoder UTAE, xử lý đầu vào đa kênh $C \to 3$, nạp pretrained weights và trích xuất đặc trưng 4 tầng. |
| [utae.py](file:///d:/THChuyenDe/utae_model/utae.py) | **Cập nhật** | Tích hợp nhánh `DINOv3Encoder` vào class `UTAE`: bổ sung các tham số `encoder_type`, `dinov3_size`, `dinov3_weights`, `freeze_encoder`; cập nhật hàm `forward()` để chuyển tiếp dữ liệu qua DINO-v3. |
| [utae_lightning.py](file:///d:/THChuyenDe/utae_model/utae_lightning.py) | **Cập nhật** | Bổ sung các hyperparameter của DINO-v3 vào `UTAELightning` để huấn luyện tự động với PyTorch Lightning. |
| [\_\_init\_\_.py](file:///d:/THChuyenDe/utae_model/__init__.py) | **Cập nhật** | Export `DINOv3Encoder` ra ngoài package namespace chung. |
| [verify_proof.py](file:///d:/THChuyenDe/utae_model/verify_proof.py) | **Tạo mới** | Script chứng minh thực nghiệm 100% sử dụng repo DINO-v3 chính chủ của Meta (kiểm tra nguồn gốc class, RoPE, số lượng tham số, gradient flow lan truyền ngược). |
| [test_suite.py](file:///d:/THChuyenDe/utae_model/test_suite.py) | **Tạo mới** | Bộ kiểm thử tự động (Unit test) toàn diện từ từng khối backbone đến toàn bộ luồng UTAE. |
| [test_demo.py](file:///d:/THChuyenDe/utae_model/test_demo.py) | **Tạo mới** | Script demo chạy mẫu toàn bộ quy trình: ViT, ConvNeXt, Backward Loss, Metrics và Lightning wrapper. |

---

## 3. Chi Tiết Từng Điểm Thay Đổi Trong Code

### 3.1. File Tạo Mới: `dinov3_encoder.py`
Toàn bộ logic kết nối DINO-v3 được đóng gói tại [dinov3_encoder.py](file:///d:/THChuyenDe/utae_model/dinov3_encoder.py):

1. **Khởi tạo và thiết lập đường dẫn repo Meta** (Dòng 10–28):
   - Đưa thư mục `dinov3` vào `sys.path`.
   - Import trực tiếp `DinoVisionTransformer`, `vit_small`, `vit_base`, `vit_large`, `ConvNeXt`, `get_convnext_arch`.
2. **Lớp `DINOv3FeatureNeck`** (Dòng 35–65):
   - Tiếp nhận danh sách các intermediate feature maps từ DINO-v3.
   - Dùng các lớp Conv 1x1 + BatchNorm + Bilinear Interpolation để chuẩn hóa kích thước ra 4 tầng độ phân giải tương thích với Decoder UTAE:
     - Stage 0: Tỷ lệ $1\times$ (kích thước gốc $H \times W$) $\to$ `out_channels[0]`
     - Stage 1: Tỷ lệ $1/2\times$ ($H/2 \times W/2$) $\to$ `out_channels[1]`
     - Stage 2: Tỷ lệ $1/4\times$ ($H/4 \times W/4$) $\to$ `out_channels[2]`
     - Stage 3: Tỷ lệ $1/8\times$ ($H/8 \times W/8$) $\to$ `out_channels[3]` (Bottleneck đưa vào L-TAE)
3. **Lớp `DINOv3Encoder`** (Dòng 72–226):
   - **Multi-spectral Channel Adapter** (Dòng 104–111): Conv2d($C \to 3$, $1\times 1$) + BatchNorm + ReLU. Nếu đầu vào đã là 3 kênh, sử dụng `nn.Identity()`.
   - **Khởi tạo Backbone DINO-v3** (Dòng 114–148):
     - Lựa chọn model ViT hoặc ConvNeXt theo kích cỡ chỉ định (`small`, `base`, `large`, `tiny`).
     - Xác định danh sách chỉ số tầng trung gian `self.layer_indices` trích xuất đều qua độ sâu mạng.
   - **Trích xuất đặc trưng qua hàm gốc của Meta** (Dòng 197–201):
     ```python
     intermediate_features = self.backbone.get_intermediate_layers(
         x_3ch,
         n=self.layer_indices,
         reshape=True,
     )
     ```
   - **Xử lý chuỗi thời gian `smart_forward`** (Dòng 206–226): Nhận tensor $(B, T, C, H, W)$, gộp thành $(B \times T, C, H, W)$ để đưa qua backbone DINO-v3, sau đó reshape ngược lại thành $(B, T, C_i, H_i, W_i)$ cho từng stage.
   - **Hỗ trợ Pretrained Weights & Freeze** (Dòng 156–164, 165–186): Cho phép tải file `.pth` của Meta và tùy chọn đóng băng (`freeze_backbone=True`).

---

### 3.2. Cập Nhật Trong `utae.py`

Xem tại [utae.py](file:///d:/THChuyenDe/utae_model/utae.py):

1. **Import module** (Dòng 5–10):
   ```python
   try:
       from .ltae import LTAE2d
       from .dinov3_encoder import DINOv3Encoder
   except ImportError:
       from ltae import LTAE2d
       from dinov3_encoder import DINOv3Encoder
   ```

2. **Tham số khởi tạo của `UTAE.__init__`** (Dòng 242–246):
   ```python
   encoder_type="dinov3",  # "dinov3" (ViT), "dinov3_convnext", hoặc "cnn"
   dinov3_size="small",    # "small", "base", "large"
   dinov3_weights=None,    # Đường dẫn file checkpoint .pth của Meta (nếu có)
   freeze_encoder=False,   # Đóng băng trọng số DINO-v3 hay finetune
   ```

3. **Khởi tạo bộ Encoder** (Dòng 258–270):
   - Nếu `encoder_type` là `"dinov3"` hoặc `"dinov3_convnext"`:
     ```python
     arch = "convnext" if "convnext" in self.encoder_type else "vit"
     self.encoder = DINOv3Encoder(
         input_dim=input_dim,
         arch=arch,
         model_size=dinov3_size,
         out_channels=encoder_widths,
         pretrained_weights=dinov3_weights,
         freeze_backbone=freeze_encoder,
     )
     self.in_conv = None
     self.down_blocks = None
     ```
   - Nếu là `"cnn"`: Giữ nguyên `self.in_conv` và `self.down_blocks` ban đầu.

4. **Hàm `forward`** (Dòng 342–351):
   ```python
   # 1. SPATIAL ENCODER (DINO-v3 or CNN)
   if self.encoder is not None:
       feature_maps = self.encoder.smart_forward(input)
   else:
       out = self.in_conv.smart_forward(input)
       feature_maps = [out]
       for i in range(self.n_stages - 1):
           out = self.down_blocks[i].smart_forward(feature_maps[-1])
           feature_maps.append(out)
   ```

---

### 3.3. Cập Nhật Trong `utae_lightning.py`

Xem tại [utae_lightning.py](file:///d:/THChuyenDe/utae_model/utae_lightning.py):

- Bổ sung các tham số DINO-v3 vào `UTAELightning.__init__` (Dòng 38–41):
  ```python
  encoder_type: str = "dinov3",
  dinov3_size: str = "small",
  dinov3_weights: str = None,
  freeze_encoder: bool = False,
  ```
- Chuyển tiếp các đối số này khi khởi tạo `self.model = UTAE(...)` (Dòng 63–66).

---

### 3.4. Cập Nhật Trong `__init__.py`

Xem tại [__init__.py](file:///d:/THChuyenDe/utae_model/__init__.py):

- Thêm import:
  ```python
  from .dinov3_encoder import DINOv3Encoder
  ```
- Đưa `"DINOv3Encoder"` vào danh sách `__all__`.

---

## 4. So Sánh Kiến Trúc Trước & Sau Khi Cập Nhật

```
[TRƯỚC ĐÂY: CNN Encoder]
Input (B, T, C, H, W)
  │
  ├──> ConvBlock (input_dim -> 64) ────────────> Feature Map 0 (H, W)      ─┐
  │                                                                         │
  ├──> DownConvBlock 1 (64 -> 64, stride 2) ───> Feature Map 1 (H/2, W/2)   │ Skip
  │                                                                         ├─ Connections
  ├──> DownConvBlock 2 (64 -> 64, stride 2) ───> Feature Map 2 (H/4, W/4)   │ vào Decoder
  │                                                                         │
  └──> DownConvBlock 3 (64 -> 128, stride 2) ──> Feature Map 3 (H/8, W/8)  ─┘
                                                       │
                                                       ▼
                                            L-TAE Temporal Bottleneck


[HIỆN TẠI: Meta DINO-v3 Encoder]
Input (B, T, C, H, W)
  │
  ▼
Input Adapter (C -> 3 channels)
  │
  ▼
Meta DINO-v3 Backbone (ViT / ConvNeXt từ cloned 'dinov3' repo)
  │  (Sử dụng RoPE, Self-Attention blocks, trích xuất intermediate layers)
  ▼
DINOv3FeatureNeck
  │
  ├──> Proj 0 + Interp (1x)  ──────────────────> Feature Map 0 (H, W)      ─┐
  ├──> Proj 1 + Interp (1/2x) ─────────────────> Feature Map 1 (H/2, W/2)   │ Skip
  ├──> Proj 2 + Interp (1/4x) ─────────────────> Feature Map 2 (H/4, W/4)   │ Connections
  └──> Proj 3 + Interp (1/8x) ─────────────────> Feature Map 3 (H/8, W/8)  ─┘
                                                       │
                                                       ▼
                                            L-TAE Temporal Bottleneck
```

---

## 5. Bằng Chứng Thực Nghiệm (Empirical Verification)

Script [verify_proof.py](file:///d:/THChuyenDe/utae_model/verify_proof.py) cung cấp 4 bằng chứng thực nghiệm kiểm tra tính nguyên bản của DINO-v3:

1. **Vị trí file nguồn**: Backbone được nạp trực tiếp từ `dinov3/dinov3/models/vision_transformer.py`.
2. **Linh kiện độc quyền của Meta**: Có sự xuất hiện của `SelfAttentionBlock` và `RopePositionEmbedding` (RoPE). Số lượng tham số của ViT-Small chuẩn xác $21,590,400$ tham số.
3. **Gradient Flow**: Lan truyền ngược qua hàm mất mát tính ra gradient khác 0 trực tiếp trên ma trận trọng số Attention: `backbone.blocks[0].attn.qkv.weight.grad`.
4. **Kiểm chứng biến thể ConvNeXt**: Khởi tạo và lan truyền chính xác trên `ConvNeXt` của Meta với chuẩn $49,454,688$ tham số.

Chạy kiểm chứng:
```bash
python verify_proof.py
```

---

## 6. Hướng Dẫn Sử Dụng Nhanh (Quick Start)

### 6.1. Khởi tạo UTAE với DINO-v3 ViT (Mặc định)

```python
import torch
from utae import UTAE

# Giả lập đầu vào Sentinel-2: Batch=2, Time=4, Kênh=12, H=128, W=128
x = torch.randn(2, 4, 12, 128, 128)
doys = torch.tensor([[10, 20, 30, 40], [15, 25, 35, 45]])  # Ngày trong năm (Day of Year)

# Khởi tạo UTAE với DINO-v3 ViT Small
model = UTAE(
    input_dim=12,
    encoder_type="dinov3",     # DINO-v3 ViT
    dinov3_size="small",       # "small", "base", hoặc "large"
    dinov3_weights=None,       # Hoặc đường dẫn tới file .pth pretrained
    freeze_encoder=False,      # False để finetune toàn mạng
)

output, att = model(x, batch_positions=doys, return_att=True)
print("Output shape:", output.shape)  # torch.Size([2, 1, 128, 128])
```

### 6.2. Khởi tạo UTAE với DINO-v3 ConvNeXt

```python
model_convnext = UTAE(
    input_dim=12,
    encoder_type="dinov3_convnext",  # DINO-v3 ConvNeXt
    dinov3_size="small",
)
output = model_convnext(x, batch_positions=doys)
```

### 6.3. Huấn luyện với PyTorch Lightning

```python
from utae_lightning import UTAELightning

lightning_module = UTAELightning(
    n_channels=12,
    encoder_type="dinov3",
    dinov3_size="small",
    loss_function="BCE",
    lr=1e-4,
    pos_class_weight=10.0,
)
```
