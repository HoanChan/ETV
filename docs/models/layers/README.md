# Layers

## 9. Neural Network Layers

### 9.1 DecoderLayer

**Chức năng:** Transformer decoder layer được sử dụng trong TableMasterDecoder. Tích hợp self-attention, cross-attention, và feed-forward networks.

**Đặc điểm:**
- Multi-head self-attention cho sequence modeling
- Cross-attention với encoder features
- Feed-forward network cho feature transformation
- Residual connections và layer normalization
- Configurable attention parameters

**Input:**
- `tgt`: Target sequence (torch.Tensor, shape: [N, T, C])
- `memory`: Encoder memory (torch.Tensor, shape: [N, S, C])
- `tgt_mask`: Target attention mask (torch.Tensor)
- `memory_mask`: Memory attention mask (torch.Tensor)

**Output:**
- `output`: Transformed target sequence (torch.Tensor, shape: [N, T, C])

**Tham số cấu hình:**
- `d_model`: Model dimension. Mặc định 512
- `n_head`: Number of attention heads. Mặc định 8
- `d_ff`: Feed-forward dimension. Mặc định 2048
- `dropout`: Dropout rate. Mặc định 0.1
- `activation`: Activation function. Mặc định 'relu'

**Layer Components:**

1. **Multi-Head Self-Attention:**
   - Attention mechanism trong target sequence
   - Causal masking cho sequence generation
   - Multi-head parallel processing

2. **Multi-Head Cross-Attention:**
   - Attention giữa target sequence và encoder features
   - Spatial attention cho table structure
   - Feature fusion mechanism

3. **Feed-Forward Network:**
   - Two-layer MLP với activation
   - Feature transformation và non-linearity
   - Residual connection

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

**Chức năng:** Specialized decoder layer cho TableMaster với optimizations cho table structure recognition.

**Đặc điểm:**
- Table-specific attention patterns
- Optimized cho spatial relationships
- Efficient memory usage
- Compatible với standard transformer layers

**Optimizations:**
- Spatial attention bias cho table structure
- Efficient attention computation
- Memory-efficient implementation
- Gradient checkpointing support

**Quan hệ với pipeline:**
- Được sử dụng trong [Decoders](../decoders/README.md)
- Tích hợp trong [Recognizer](../recognizer/README.md)
- Optimized cho table recognition tasks

**Lưu ý đặc biệt:**
- Attention heads và dimensions phải consistent across layers
- Dropout rates affect training stability
- Memory masks important cho variable sequence lengths
- Layer normalization placement affects gradient flow
