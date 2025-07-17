# Backbones

#### 1. TableResNetExtra (Backbone)
**Chức năng:** Trích xuất đặc trưng không gian từ ảnh đầu vào.
**Kiến trúc:**
- Nhận tensor ảnh chuẩn hóa từ pipeline (`inputs`).
- Áp dụng các block ResNet, GCB (Global Context Block) để tạo ra feature map.
- Output: `feature_map` (Tensor, shape [N, C, H, W]) truyền sang encoder.

**Input:**
- `inputs` (torch.Tensor hoặc numpy.ndarray): Ảnh đầu vào đã chuẩn hóa.
**Output:**
- `feature_map` (torch.Tensor): Đặc trưng không gian của ảnh.
