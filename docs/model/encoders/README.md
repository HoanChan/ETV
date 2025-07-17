# Encoders

#### 2. PositionalEncoding (Encoder)
**Chức năng:** Mã hóa vị trí cho feature map để bổ sung thông tin vị trí cho từng token.
**Kiến trúc:**
- Nhận feature map từ backbone.
- Áp dụng encoding sinusoidal hoặc learnable để tạo ra `encoded_features`.
- Output: `encoded_features` (Tensor, shape [N, C, H, W]) truyền sang decoder.

**Input:**
- `feature_map` (torch.Tensor): Đặc trưng không gian của ảnh.
**Output:**
- `encoded_features` (torch.Tensor): Đặc trưng đã bổ sung thông tin vị trí.
