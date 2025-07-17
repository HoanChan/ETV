# Decoders

## 3. TableMasterConcatDecoder

**Chức năng:** Decoder module chính của TableMaster với dual-head architecture cho table structure recognition và bbox regression. Sử dụng concatenation strategy để combine multiple layer outputs.

**Đặc điểm:**
- Dual-head architecture: classification và bbox regression
- Concatenation strategy cho better feature fusion
- Transformer-based decoder với attention mechanism
- Tích hợp postprocessor cho output processing
- Hỗ trợ multiple loss functions

**Input:**
- `feat`: Encoded features từ encoder (torch.Tensor, shape: [N, H*W, C])
- `out_enc`: Encoded features (optional)
- `data_samples`: Batch data samples với ground truth

**Output:**
- `token_logits`: Classification logits cho structure tokens (torch.Tensor)
- `bbox_logits`: Bbox regression outputs (torch.Tensor)

**Tham số cấu hình:**
- `n_layers`: Số layers trong decoder. Mặc định 3
- `n_head`: Số attention heads. Mặc định 8
- `d_model`: Model dimension. Mặc định 512
- `max_seq_len`: Maximum sequence length. Mặc định 600
- `dictionary`: Dictionary config cho token mapping
- `decoder`: Decoder layer configuration
- `postprocessor`: Postprocessor configuration
- `tokens_loss`: Loss function cho classification head
- `bboxes_loss`: Loss function cho bbox regression head

**Kiến trúc chi tiết:**

1. **Embedding Layer:**
   - Token embedding cho input sequences
   - Positional encoding integration

2. **Shared Decoder Layers:**
   - Multiple transformer decoder layers
   - Self-attention và cross-attention mechanisms

3. **Dual Heads:**
   - **Classification Head:** Predict structure tokens
   - **Bbox Head:** Predict bounding box coordinates

4. **Concatenation Strategy:**
   - Concatenate outputs từ multiple layers
   - Improved feature representation

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
- Nhận features từ [Encoders](../encoders/README.md)
- Sử dụng [Postprocessors](../postprocessors/README.md) cho output processing
- Sử dụng [Dictionaries](../dictionaries/README.md) cho token mapping

**Lưu ý đặc biệt:**
- Dual-head design essential cho table structure + bbox prediction
- Concatenation strategy improves feature utilization
- Attention mechanism captures spatial relationships
- Postprocessor crucial cho meaningful outputs
