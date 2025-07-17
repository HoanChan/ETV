
# Decoders

## 3. TableMasterConcatDecoder

**Chức năng:** Module decoder chính của TableMaster với kiến trúc dual-head cho nhận diện cấu trúc bảng và hồi quy bbox. Sử dụng chiến lược nối (concatenation) để kết hợp output từ nhiều layer.

**Đặc điểm:**
- Kiến trúc dual-head: classification và bbox regression
- Chiến lược nối (concatenation) giúp kết hợp đặc trưng tốt hơn
- Decoder dựa trên Transformer với attention mechanism
- Tích hợp postprocessor để xử lý output
- Hỗ trợ nhiều hàm loss

**Input:**
- `feat`: Đặc trưng đã mã hóa từ encoder (torch.Tensor, shape: [N, H*W, C])
- `out_enc`: Đặc trưng đã mã hóa (tùy chọn)
- `data_samples`: Batch dữ liệu với ground truth

**Output:**
- `token_logits`: Logits phân loại cho structure tokens (torch.Tensor)
- `bbox_logits`: Output hồi quy bbox (torch.Tensor)

**Tham số cấu hình:**
- `n_layers`: Số lượng layer trong decoder. Mặc định 3
- `n_head`: Số lượng attention heads. Mặc định 8
- `d_model`: Kích thước mô hình. Mặc định 512
- `max_seq_len`: Độ dài chuỗi tối đa. Mặc định 600
- `dictionary`: Cấu hình dictionary cho token mapping
- `decoder`: Cấu hình layer decoder
- `postprocessor`: Cấu hình postprocessor
- `tokens_loss`: Hàm loss cho classification head
- `bboxes_loss`: Hàm loss cho bbox regression head

**Kiến trúc chi tiết:**

1. **Lớp Embedding:**
   - Token embedding cho chuỗi đầu vào
   - Tích hợp positional encoding

2. **Các lớp Decoder dùng chung:**
   - Nhiều lớp transformer decoder
   - Cơ chế self-attention và cross-attention

3. **Dual Heads:**
   - **Classification Head:** Dự đoán structure tokens
   - **Bbox Head:** Dự đoán tọa độ bounding box

4. **Chiến lược nối (Concatenation):**
   - Nối output từ nhiều layer
   - Cải thiện biểu diễn đặc trưng

**Ví dụ cấu hình:**
```python
decoder = dict(
    type='TableMasterConcatDecoder',
    n_layers=3,
    n_head=8,
    d_model=512,
    max_seq_len=600,
    dictionary=dictionary,
    decoder=dict(
        self_attn=dict(headers=8, d_model=512, dropout=0.0),
        src_attn=dict(headers=8, d_model=512, dropout=0.0),
        feed_forward=dict(d_model=512, d_ff=2024, dropout=0.0),
        size=512,
        dropout=0.0
    ),
    postprocessor=dict(
        type='TableMasterPostprocessor',
        dictionary=dictionary,
        max_seq_len=600,
        start_end_same=False
    ),
    tokens_loss=dict(
        type='MasterTFLoss',
        ignore_index=PAD,
        reduction='mean',
        flatten=True
    ),
    bboxes_loss=dict(
        type='TableL1Loss',
        reduction='sum',
        lambda_horizon=1.0,
        lambda_vertical=1.0,
        eps=1e-9
    )
)
```

**Loss Functions:**
- **tokens_loss:** Cross-entropy loss cho classification
- **bboxes_loss:** L1 loss cho bbox regression

**Quan hệ với pipeline:**
- Nhận đặc trưng từ [Encoders](../encoders/README.md)
- Sử dụng [Postprocessors](../postprocessors/README.md) để xử lý output
- Sử dụng [Dictionaries](../dictionaries/README.md) cho token mapping

**Lưu ý đặc biệt:**
- Thiết kế dual-head rất quan trọng cho nhận diện cấu trúc bảng và dự đoán bbox
- Chiến lược nối giúp tận dụng đặc trưng tốt hơn
- Attention mechanism giúp mô hình hóa quan hệ không gian
- Postprocessor rất quan trọng để tạo ra output có ý nghĩa
