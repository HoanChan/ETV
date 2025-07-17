# Losses

## 7. Loss Functions

### 7.1 MasterTFLoss

**Chức năng:** Cross-entropy loss function cho classification head của TableMaster. Xử lý sequence prediction với padding và unknown tokens.

**Đặc điểm:**
- Cross-entropy loss cho multi-class classification
- Hỗ trợ ignore_index cho padding tokens
- Flatten option cho sequence processing
- Configurable reduction methods

**Input:**
- `predictions`: Logits từ classification head (torch.Tensor)
- `targets`: Ground truth token indices (torch.Tensor)

**Output:**
- `loss`: Scalar loss value (torch.Tensor)

**Tham số cấu hình:**
- `ignore_index`: Index to ignore trong loss calculation (thường là PAD token)
- `reduction`: Reduction method ('mean', 'sum', 'none'). Mặc định 'mean'
- `flatten`: Có flatten inputs hay không. Mặc định True

**Ví dụ cấu hình:**
```python
tokens_loss = dict(
    type='MasterTFLoss',
    ignore_index=PAD,
    reduction='mean',
    flatten=True
)
```

### 7.2 TableL1Loss

**Chức năng:** L1 loss function cho bbox regression head của TableMaster. Xử lý bounding box prediction với separate weights cho horizontal và vertical coordinates.

**Đặc điểm:**
- L1 loss cho bbox regression
- Separate weights cho horizontal (x, w) và vertical (y, h) coordinates
- Epsilon smoothing cho numerical stability
- Configurable reduction methods

**Input:**
- `bbox_predictions`: Predicted bbox coordinates (torch.Tensor)
- `bbox_targets`: Ground truth bbox coordinates (torch.Tensor)

**Output:**
- `loss`: Scalar loss value (torch.Tensor)

**Tham số cấu hình:**
- `reduction`: Reduction method ('mean', 'sum', 'none'). Mặc định 'sum'
- `lambda_horizon`: Weight cho horizontal coordinates (x, w). Mặc định 1.0
- `lambda_vertical`: Weight cho vertical coordinates (y, h). Mặc định 1.0
- `eps`: Epsilon cho numerical stability. Mặc định 1e-9

**Ví dụ cấu hình:**
```python
bboxes_loss = dict(
    type='TableL1Loss',
    reduction='sum',
    lambda_horizon=1.0,
    lambda_vertical=1.0,
    eps=1e-9
)
```

**Loss Calculation:**
```python
# Separate horizontal and vertical components
horizontal_loss = L1(pred_x, target_x) + L1(pred_w, target_w)
vertical_loss = L1(pred_y, target_y) + L1(pred_h, target_h)

# Weighted combination
total_loss = lambda_horizon * horizontal_loss + lambda_vertical * vertical_loss
```

**Quan hệ với pipeline:**
- Được sử dụng trong [Decoders](../decoders/README.md)
- Nhận predictions và targets từ training data
- Tích hợp trong [Recognizer](../recognizer/README.md)

**Lưu ý đặc biệt:**
- ignore_index trong MasterTFLoss phải match với PAD token trong dictionary
- TableL1Loss weights có thể được tune để balance horizontal vs vertical accuracy
- Epsilon value important cho numerical stability với very small coordinates
- Reduction methods affect gradient scaling và learning dynamics
