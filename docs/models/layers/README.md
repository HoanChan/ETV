
# Layers

## 9. Các lớp mạng nơ-ron

### 9.1 DecoderLayer

**Chức năng:** Lớp Transformer decoder được sử dụng trong TableMasterDecoder. Tích hợp self-attention, cross-attention và feed-forward networks.

**Đặc điểm:**
- Multi-head self-attention cho modeling chuỗi
- Cross-attention với đặc trưng từ encoder
- Feed-forward network để biến đổi đặc trưng
- Kết nối residual và layer normalization
- Tham số attention có thể cấu hình

**Input:**
- `tgt`: Chuỗi mục tiêu (torch.Tensor, shape: [N, T, C])
- `memory`: Bộ nhớ encoder (torch.Tensor, shape: [N, S, C])
- `tgt_mask`: Mask attention cho chuỗi mục tiêu (torch.Tensor)
- `memory_mask`: Mask attention cho bộ nhớ encoder (torch.Tensor)

**Output:**
- `output`: Chuỗi mục tiêu đã biến đổi (torch.Tensor, shape: [N, T, C])

**Tham số cấu hình:**
- `d_model`: Kích thước mô hình. Mặc định 512
- `n_head`: Số lượng attention heads. Mặc định 8
- `d_ff`: Kích thước feed-forward. Mặc định 2048
- `dropout`: Tỉ lệ dropout. Mặc định 0.1
- `activation`: Hàm kích hoạt. Mặc định 'relu'

**Thành phần layer:**

1. **Multi-Head Self-Attention:**
   - Cơ chế attention trong chuỗi mục tiêu
   - Causal masking cho sinh chuỗi
   - Xử lý song song nhiều head

2. **Multi-Head Cross-Attention:**
   - Attention giữa chuỗi mục tiêu và đặc trưng encoder
   - Spatial attention cho cấu trúc bảng
   - Cơ chế kết hợp đặc trưng

3. **Feed-Forward Network:**
   - MLP hai lớp với activation
   - Biến đổi đặc trưng và phi tuyến tính
   - Kết nối residual

**Ví dụ cấu hình:**
```python
decoder_layer = dict(
    self_attn=dict(
        headers=8,
        d_model=512,
        dropout=0.0
    ),
    src_attn=dict(
        headers=8,
        d_model=512,
        dropout=0.0
    ),
    feed_forward=dict(
        d_model=512,
        d_ff=2024,
        dropout=0.0
    ),
    size=512,
    dropout=0.0
)
```

### 9.2 TMD (TableMaster Decoder) Layer

**Chức năng:** Lớp decoder chuyên biệt cho TableMaster với các tối ưu hóa cho nhận diện cấu trúc bảng.

**Đặc điểm:**
- Attention patterns đặc thù cho bảng
- Tối ưu hóa cho quan hệ không gian
- Sử dụng bộ nhớ hiệu quả
- Tương thích với các lớp transformer tiêu chuẩn

**Tối ưu hóa:**
- Spatial attention bias cho cấu trúc bảng
- Tính attention hiệu quả
- Triển khai tiết kiệm bộ nhớ
- Hỗ trợ gradient checkpointing

**Quan hệ với pipeline:**
- Được sử dụng trong [Decoders](../decoders/README.md)
- Tích hợp trong [Recognizer](../recognizer/README.md)
- Tối ưu hóa cho các bài toán nhận diện bảng

**Lưu ý đặc biệt:**
- Attention heads và kích thước phải nhất quán giữa các layer
- Tỉ lệ dropout ảnh hưởng đến độ ổn định huấn luyện
- Memory mask quan trọng cho chuỗi có độ dài thay đổi
- Vị trí layer normalization ảnh hưởng đến luồng gradient
