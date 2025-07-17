
# Backbones

## 1. TableResNetExtra

**Chức năng:** Mạng backbone để trích xuất đặc trưng không gian từ ảnh bảng. Kế thừa từ kiến trúc ResNet và được tối ưu hóa cho nhận diện bảng.

**Đặc điểm:**
- Sử dụng kiến trúc ResNet với BasicBlock
- Tích hợp Global Context Block (GCB) để cải thiện khả năng mô hình hóa ngữ cảnh
- Tối ưu hóa cho nhận diện cấu trúc bảng
- Trích xuất đặc trưng đa tỷ lệ (multi-scale)
- Các layer và vị trí GCB có thể cấu hình

**Input:**
- `inputs`: Tensor ảnh đã chuẩn hóa (torch.Tensor, shape: [N, C, H, W])
- Thường là output từ @import "../../datasets/transforms/pack_inputs/README.md"

**Output:**
- `feature_map`: Danh sách các feature map từ nhiều tỷ lệ
  - f[0]: Feature map từ layer 1 (shape: [N, 256, H/4, W/4])
  - f[1]: Feature map từ layer 2 (shape: [N, 256, H/8, W/8])
  - f[2]: Feature map từ layer 4 (shape: [N, 512, H/16, W/16])

**Tham số cấu hình:**
- `input_dim`: Số lượng channels của ảnh đầu vào. Mặc định là 3
- `layers`: Danh sách số lượng block cho mỗi layer. Ví dụ [1, 2, 5, 3]
- `gcb_config`: Cấu hình cho Global Context Block:
  - `ratio`: Tỉ lệ giảm chiều cho GCB. Mặc định 0.0625
  - `headers`: Số lượng attention heads. Mặc định 1
  - `att_scale`: Có scale attention hay không. Mặc định False
  - `fusion_type`: Kiểu kết hợp ('channel_add'). Mặc định 'channel_add'
  - `layers`: Danh sách boolean chỉ định layer nào có GCB. Ví dụ [False, True, True, True]

**Kiến trúc chi tiết:**

1. **Các lớp Convolution ban đầu:**
   - Conv1: 3→64, kernel 3x3, stride=1, padding=1
   - Conv2: 64→128, kernel 3x3, stride=1, padding=1
   - MaxPool1: 2x2, stride=2

2. **Các lớp ResNet:**
   - Layer1: 128→256, có thể tích hợp GCB
   - Layer2: 256→256, có thể tích hợp GCB
   - Layer3: 256→512, có thể tích hợp GCB
   - Layer4: 512→512, có thể tích hợp GCB

3. **Điểm trích xuất đặc trưng:**
   - Trích xuất đặc trưng tại 3 tỷ lệ khác nhau
   - Đặc trưng đa tỷ lệ giúp định vị tốt hơn

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
- GCB giúp mô hình bắt được các phụ thuộc dài trong cấu trúc bảng
- Đặc trưng đa tỷ lệ rất quan trọng cho nhận diện cấu trúc bảng
- Kiến trúc được tối ưu hóa cho các mẫu đặc trưng của bảng
- Khác biệt với ResNet tiêu chuẩn ở stride của maxpool và tích hợp GCB
