# Backbones

## 1. TableResNetExtra

**Chức năng:** Backbone network để trích xuất đặc trưng không gian từ ảnh bảng. Kế thừa từ ResNet architecture và được tối ưu hóa cho table recognition.

**Đặc điểm:**
- Sử dụng ResNet-based architecture với BasicBlock
- Tích hợp Global Context Block (GCB) để cải thiện context modeling
- Optimized cho table structure recognition
- Multi-scale feature extraction
- Configurable layers và GCB placement

**Input:**
- `inputs`: Tensor ảnh đã chuẩn hóa (torch.Tensor, shape: [N, C, H, W])
- Thường là output từ @import "../../datasets/transforms/pack_inputs/README.md"

**Output:**
- `feature_map`: List of feature maps từ multiple scales
  - f[0]: Feature map từ layer 1 (shape: [N, 256, H/4, W/4])
  - f[1]: Feature map từ layer 2 (shape: [N, 256, H/8, W/8])
  - f[2]: Feature map từ layer 4 (shape: [N, 512, H/16, W/16])

**Tham số cấu hình:**
- `input_dim`: Số channels của input image. Mặc định 3
- `layers`: Danh sách số blocks cho mỗi layer. Ví dụ [1, 2, 5, 3]
- `gcb_config`: Cấu hình Global Context Block:
  - `ratio`: Reduction ratio cho GCB. Mặc định 0.0625
  - `headers`: Số attention heads. Mặc định 1
  - `att_scale`: Có scale attention hay không. Mặc định False
  - `fusion_type`: Fusion type ('channel_add'). Mặc định 'channel_add'
  - `layers`: Boolean list chỉ định layers nào có GCB. Ví dụ [False, True, True, True]

**Kiến trúc chi tiết:**

1. **Initial Convolution Layers:**
   - Conv1: 3→64, 3x3, stride=1, padding=1
   - Conv2: 64→128, 3x3, stride=1, padding=1
   - MaxPool1: 2x2, stride=2

2. **ResNet Layers:**
   - Layer1: 128→256, với GCB tùy chọn
   - Layer2: 256→256, với GCB tùy chọn
   - Layer3: 256→512, với GCB tùy chọn
   - Layer4: 512→512, với GCB tùy chọn

3. **Feature Extraction Points:**
   - Extract features tại 3 scales khác nhau
   - Multi-scale features cho better localization

**Ví dụ cấu hình:**
```python
backbone = dict(
    type='TableResNetExtra',
    input_dim=3,
    gcb_config=dict(
        ratio=0.0625,
        headers=1,
        att_scale=False,
        fusion_type="channel_add",
        layers=[False, True, True, True],
    ),
    layers=[1, 2, 5, 3]
)
```

**Quan hệ với pipeline:**
- Nhận input từ [Pack Inputs](../../datasets/transforms/pack_inputs/README.md)
- Truyền feature tới [Encoders](../encoders/README.md)

**Lưu ý đặc biệt:**
- GCB giúp model capture long-range dependencies trong table structure
- Multi-scale features essential cho table structure recognition
- Architecture được optimize cho table-specific patterns
- Khác biệt với standard ResNet ở maxpool stride và GCB integration
