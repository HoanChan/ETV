# Postprocessors

## 5. TableMasterPostprocessor

**Chức năng:** Postprocessor để xử lý raw outputs từ decoder thành meaningful predictions. Chuyển đổi logits thành tokens và bbox coordinates thành actual bounding boxes.

**Đặc điểm:**
- Chuyển đổi logits thành token predictions
- Decode bbox coordinates với denormalization
- Áp dụng confidence thresholding
- Xử lý sequence length mismatches
- Tích hợp với dictionary cho token mapping

**Input:**
- `structure_outputs`: Classification logits (torch.Tensor)
- `bbox_outputs`: Bbox regression outputs (torch.Tensor)
- `data_samples`: Batch data samples với meta information

**Output:**
- `final_tokens`: Predicted token strings (List[str])
- `final_bboxes`: Predicted bounding boxes (List[np.ndarray])
- `scores`: Confidence scores cho predictions

**Tham số cấu hình:**
- `dictionary`: Dictionary instance cho token mapping
- `max_seq_len`: Maximum sequence length. Mặc định 600
- `start_end_same`: Start và end tokens có giống nhau hay không. Mặc định False

**Quy trình xử lý:**

1. **Token Prediction:**
   - Chuyển đổi logits thành token indices
   - Áp dụng confidence thresholding
   - Map indices thành token strings

2. **Bbox Prediction:**
   - Extract bbox coordinates từ regression outputs
   - Áp dụng bbox masks cho valid predictions
   - Denormalize coordinates về original image space

3. **Coordinate Denormalization:**
   - Scale coordinates từ [0,1] về pixel coordinates
   - Áp dụng scale_factor từ image preprocessing
   - Adjust cho padding offsets

4. **Sequence Alignment:**
   - Align bbox predictions với token predictions
   - Handle sequence length mismatches
   - Filter invalid predictions

**Ví dụ cấu hình:**
```python
postprocessor = dict(
    type='TableMasterPostprocessor',
    dictionary=dictionary,
    max_seq_len=600,
    start_end_same=False
)
```

**Bbox Denormalization Process:**
1. **Padding Space:** Bbox coordinates trong [0,1] range của padded image
2. **Scale to Pad Shape:** Multiply với pad_shape để get pixel coordinates
3. **Adjust Scale Factor:** Divide bằng scale_factor để get original coordinates
4. **Clamp to Image:** Clamp coordinates within original image boundaries

**Output Format:**
```python
# Token predictions
final_tokens = ["<table>", "<tr>", "<td>", "</td>", "</tr>", "</table>"]

# Bbox predictions (denormalized)
final_bboxes = [
    np.array([[10, 20, 100, 200],    # <td> bbox
              [15, 25, 95, 180]]),    # another <td> bbox
    ...
]

# Confidence scores
scores = [0.95, 0.87, 0.92, ...]
```

**Quan hệ với pipeline:**
- Nhận outputs từ @import "../decoders/README.md"
- Sử dụng @import "../dictionaries/README.md" cho token mapping
- Sử dụng meta information từ @import "../../datasets/transforms/pack_inputs/README.md"

**Lưu ý đặc biệt:**
- Denormalization process phải chính xác để get correct bbox coordinates
- Sequence alignment critical cho structure-bbox correspondence
- Confidence thresholding helps filter low-quality predictions
- Meta information từ preprocessing essential cho accurate denormalization
