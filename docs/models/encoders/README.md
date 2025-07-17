# Encoders

## 2. PositionalEncoding

**Chức năng:** Encoder module để thêm positional encoding vào feature maps từ backbone. Chuyển đổi 2D spatial features thành 1D sequence với positional information.

**Đặc điểm:**
- Chuyển đổi 2D feature maps thành 1D sequences
- Thêm sinusoidal positional encoding
- Hỗ trợ dropout cho regularization
- Tương thích với Transformer architecture
- Learnable positional embeddings

**Input:**
- `feat`: Feature maps từ backbone (torch.Tensor, shape: [N, C, H, W])
- `data_samples`: Batch data samples (optional)

**Output:**
- `encoded_features`: Feature sequences với positional encoding (torch.Tensor, shape: [N, H*W, C])

**Tham số cấu hình:**
- `d_model`: Dimension của model. Mặc định 512
- `dropout`: Dropout probability. Mặc định 0.0
- `max_len`: Maximum sequence length. Mặc định 5000

**Quy trình xử lý:**

1. **Feature Reshaping:**
   - Reshape feature maps từ (N, C, H, W) thành (N, H*W, C)
   - Flatten spatial dimensions thành sequence dimension

2. **Positional Encoding:**
   - Tạo sinusoidal positional encoding matrix
   - PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
   - PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))

3. **Encoding Addition:**
   - Thêm positional encoding vào feature sequence
   - Áp dụng dropout nếu được cấu hình

**Ví dụ cấu hình:**
```python
encoder = dict(
    type='PositionalEncoding',
    d_model=512,
    dropout=0.2,
    max_len=5000
)
```

**Positional Encoding Formula:**
- Position encoding được tính sử dụng sine và cosine functions
- Cho phép model học relative positions trong sequence
- Stable across different sequence lengths

**Quan hệ với pipeline:**
- Nhận feature maps từ @import "../backbones/README.md"
- Truyền encoded features tới @import "../decoders/README.md"

**Lưu ý đặc biệt:**
- Essential cho Transformer-based architectures
- Positional information crucial cho table structure understanding
- Sinusoidal encoding allows extrapolation to longer sequences
- Dropout helps prevent overfitting to positional patterns
