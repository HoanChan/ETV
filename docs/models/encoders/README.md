
# Encoders

## 2. PositionalEncoding

**Chức năng:** Module encoder để thêm positional encoding vào feature maps từ backbone. Chuyển đổi đặc trưng không gian 2D thành chuỗi 1D với thông tin vị trí.

**Đặc điểm:**
- Chuyển đổi feature maps 2D thành chuỗi 1D
- Thêm sinusoidal positional encoding
- Hỗ trợ dropout để regularization
- Tương thích với kiến trúc Transformer
- Có thể học positional embeddings

**Input:**
- `feat`: Feature maps từ backbone (torch.Tensor, shape: [N, C, H, W])
- `data_samples`: Batch dữ liệu (tùy chọn)

**Output:**
- `encoded_features`: Chuỗi đặc trưng với positional encoding (torch.Tensor, shape: [N, H*W, C])

**Tham số cấu hình:**
- `d_model`: Kích thước mô hình. Mặc định 512
- `dropout`: Xác suất dropout. Mặc định 0.0
- `max_len`: Độ dài chuỗi tối đa. Mặc định 5000

**Quy trình xử lý:**

1. **Chuyển đổi đặc trưng:**
   - Reshape feature maps từ (N, C, H, W) thành (N, H*W, C)
   - Làm phẳng các chiều không gian thành chiều chuỗi

2. **Positional Encoding:**
   - Tạo ma trận sinusoidal positional encoding
   - PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
   - PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))

3. **Thêm Encoding:**
   - Thêm positional encoding vào chuỗi đặc trưng
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

**Công thức Positional Encoding:**
- Position encoding được tính bằng hàm sine và cosine
- Cho phép mô hình học vị trí tương đối trong chuỗi
- Ổn định với các độ dài chuỗi khác nhau

**Quan hệ với pipeline:**
- Nhận feature maps từ [Backbones](../backbones/README.md)
- Truyền encoded features tới [Decoders](../decoders/README.md)

**Lưu ý đặc biệt:**
- Rất quan trọng cho kiến trúc Transformer
- Thông tin vị trí rất quan trọng cho nhận diện cấu trúc bảng
- Sinusoidal encoding cho phép mô hình hóa chuỗi dài hơn
- Dropout giúp tránh overfitting vào vị trí
