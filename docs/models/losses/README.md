
# Losses

## 7. Các hàm Loss

### 7.1 MasterTFLoss

**Chức năng:** Hàm loss cross-entropy cho classification head của TableMaster. Xử lý dự đoán chuỗi với padding và unknown tokens.

**Đặc điểm:**
- Cross-entropy loss cho phân loại đa lớp
- Hỗ trợ ignore_index cho padding tokens
- Tùy chọn flatten cho xử lý chuỗi
- Phương pháp giảm (reduction) có thể cấu hình

**Input:**
- `predictions`: Logits từ classification head (torch.Tensor)
- `targets`: Chỉ số token ground truth (torch.Tensor)

**Output:**
- `loss`: Giá trị loss dạng scalar (torch.Tensor)

**Tham số cấu hình:**
- `ignore_index`: Chỉ số cần bỏ qua khi tính loss (thường là PAD token)
- `reduction`: Phương pháp giảm ('mean', 'sum', 'none'). Mặc định 'mean'
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

**Chức năng:** Hàm loss L1 cho bbox regression head của TableMaster. Xử lý dự đoán bounding box với trọng số riêng cho tọa độ ngang và dọc.

**Đặc điểm:**
- L1 loss cho hồi quy bbox
- Trọng số riêng cho tọa độ ngang (x, w) và dọc (y, h)
- Epsilon smoothing cho ổn định số học
- Phương pháp giảm (reduction) có thể cấu hình

**Input:**
- `bbox_predictions`: Tọa độ bbox dự đoán (torch.Tensor)
- `bbox_targets`: Tọa độ bbox ground truth (torch.Tensor)

**Output:**
- `loss`: Giá trị loss dạng scalar (torch.Tensor)

**Tham số cấu hình:**
- `reduction`: Phương pháp giảm ('mean', 'sum', 'none'). Mặc định 'sum'
- `lambda_horizon`: Trọng số cho tọa độ ngang (x, w). Mặc định 1.0
- `lambda_vertical`: Trọng số cho tọa độ dọc (y, h). Mặc định 1.0
- `eps`: Epsilon cho ổn định số học. Mặc định 1e-9

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

**Cách tính Loss:**
```python
# Tách riêng thành phần ngang và dọc
horizontal_loss = L1(pred_x, target_x) + L1(pred_w, target_w)
vertical_loss = L1(pred_y, target_y) + L1(pred_h, target_h)

# Kết hợp có trọng số
total_loss = lambda_horizon * horizontal_loss + lambda_vertical * vertical_loss
```

**Quan hệ với pipeline:**
- Được sử dụng trong [Decoders](../decoders/README.md)
- Nhận predictions và targets từ dữ liệu huấn luyện
- Tích hợp trong [Recognizer](../recognizer/README.md)

**Lưu ý đặc biệt:**
- ignore_index trong MasterTFLoss phải khớp với PAD token trong dictionary
- Trọng số TableL1Loss có thể điều chỉnh để cân bằng độ chính xác ngang và dọc
- Giá trị epsilon quan trọng cho ổn định số học với tọa độ rất nhỏ
- Phương pháp giảm ảnh hưởng đến gradient scaling và động học học
