
# Metrics

## 8. Các chỉ số đánh giá

### 8.1 TEDS (Tree Edit Distance based Similarity)

**Chức năng:** Chỉ số chính để đánh giá độ chính xác của nhận diện cấu trúc bảng. Tính toán độ tương đồng giữa cấu trúc bảng dự đoán và ground truth dựa trên tree edit distance.

**Đặc điểm:**
- Tính toán độ tương đồng dựa trên Tree Edit Distance
- Đánh giá cả độ chính xác cấu trúc và không gian
- Điểm số similarity được chuẩn hóa từ 0 đến 1
- Ổn định với các khác biệt cấu trúc nhỏ
- Hỗ trợ đánh giá theo batch

**Input:**
- `pred_structures`: Cấu trúc bảng dự đoán (chuỗi HTML hoặc chuỗi token)
- `gt_structures`: Cấu trúc bảng ground truth
- `pred_bboxes`: Bounding boxes dự đoán (tùy chọn)
- `gt_bboxes`: Bounding boxes ground truth (tùy chọn)

**Output:**
- `teds_score`: Điểm số similarity TEDS (float, 0.0 - 1.0)
- `detailed_scores`: Phân tích chi tiết các thành phần độ chính xác

**Tham số cấu hình:**
- `structure_only`: Chỉ đánh giá cấu trúc, không xét bbox. Mặc định False
- `normalize`: Có chuẩn hóa điểm số hay không. Mặc định True
- `ignore_nodes`: Danh sách node cần bỏ qua khi đánh giá

**Cách tính TEDS:**
1. **Phân tích cấu trúc:** Chuyển HTML/tokens thành cây cấu trúc
2. **Tree Edit Distance:** Tính số phép chỉnh sửa tối thiểu cần thiết
3. **Chuẩn hóa:** Chuẩn hóa theo kích thước cây lớn nhất
4. **Điểm số similarity:** score = 1 - (edit_distance / max_tree_size)

**Ví dụ cấu hình:**
```python
metric = dict(
    type='TEDSMetric',
    structure_only=False,
    normalize=True,
    ignore_nodes=['<pad>', '<unk>']
)
```

### 8.2 Các chỉ số hậu xử lý (Post-processing Metrics)

**Chức năng:** Các chỉ số bổ sung để đánh giá chi tiết hiệu năng nhận diện bảng.

**Đặc điểm:**
- Độ chính xác từng token
- Tính toán IoU cho bbox
- Kiểm tra tính nhất quán cấu trúc
- Phân tích lỗi

**Các chỉ số bao gồm:**

1. **Độ chính xác token:**
   - Độ chính xác khớp hoàn toàn cho chuỗi token
   - Khoảng cách chỉnh sửa cho chuỗi token
   - Độ chính xác phân loại từng token

2. **Chỉ số bbox:**
   - IoU (Intersection over Union) cho bbox dự đoán
   - Độ chính xác trung bình cho phát hiện bbox
   - Chỉ số độ chính xác tọa độ

3. **Chỉ số cấu trúc:**
   - Tính nhất quán cấu trúc bảng
   - Độ chính xác đếm hàng/cột
   - Độ chính xác cấu trúc lồng nhau

**Ví dụ sử dụng:**
```python
# Đánh giá
results = evaluator.evaluate(predictions, ground_truth)

# Định dạng kết quả
{
    'teds_score': 0.85,
    'token_accuracy': 0.92,
    'bbox_iou': 0.78,
    'structure_consistency': 0.88,
    'detailed_breakdown': {
        'table_accuracy': 0.90,
        'row_accuracy': 0.87,
        'cell_accuracy': 0.85
    }
}
```

**Quan hệ với pipeline:**
- Nhận predictions từ [Recognizer](../recognizer/README.md)
- Sử dụng ground truth từ [Table Dataset](../../datasets/table_dataset/README.md)
- Tích hợp trong các script đánh giá

**Lưu ý đặc biệt:**
- TEDS là chỉ số chính cho đánh giá nhận diện bảng
- Tính nhất quán cấu trúc rất quan trọng cho ứng dụng thực tế
- Độ chính xác bbox ảnh hưởng đến hiệu năng OCR phía sau
- Cấu hình chỉ số nên phù hợp với đặc điểm dataset
