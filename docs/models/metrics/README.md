# Metrics

## 8. Evaluation Metrics

### 8.1 TEDS (Tree Edit Distance based Similarity)

**Chức năng:** Metric chính để đánh giá accuracy của table structure recognition. Tính toán similarity giữa predicted table structure và ground truth dựa trên tree edit distance.

**Đặc điểm:**
- Tree Edit Distance based similarity calculation
- Xử lý cả structure accuracy và spatial accuracy
- Normalized similarity score từ 0 đến 1
- Robust đối với minor structural differences
- Hỗ trợ batch evaluation

**Input:**
- `pred_structures`: Predicted table structures (HTML strings hoặc token sequences)
- `gt_structures`: Ground truth table structures
- `pred_bboxes`: Predicted bounding boxes (optional)
- `gt_bboxes`: Ground truth bounding boxes (optional)

**Output:**
- `teds_score`: TEDS similarity score (float, 0.0 - 1.0)
- `detailed_scores`: Detailed breakdown của accuracy components

**Tham số cấu hình:**
- `structure_only`: Chỉ evaluate structure mà không xét bbox. Mặc định False
- `normalize`: Có normalize score hay không. Mặc định True
- `ignore_nodes`: Danh sách nodes to ignore trong evaluation

**TEDS Calculation:**
1. **Structure Parsing:** Convert HTML/tokens thành tree structure
2. **Tree Edit Distance:** Calculate minimum edit operations needed
3. **Normalization:** Normalize bằng maximum tree size
4. **Similarity Score:** score = 1 - (edit_distance / max_tree_size)

**Ví dụ cấu hình:**
```python
metric = dict(
    type='TEDSMetric',
    structure_only=False,
    normalize=True,
    ignore_nodes=['<pad>', '<unk>']
)
```

### 8.2 Post-processing Metrics

**Chức năng:** Additional metrics cho detailed evaluation của table recognition performance.

**Đặc điểm:**
- Token-level accuracy
- Bbox IoU calculation
- Structure consistency checking
- Error analysis capabilities

**Metrics bao gồm:**

1. **Token Accuracy:**
   - Exact match accuracy cho token sequences
   - Edit distance cho token sequences
   - Per-token classification accuracy

2. **Bbox Metrics:**
   - IoU (Intersection over Union) cho bbox predictions
   - Average precision cho bbox detection
   - Coordinate accuracy metrics

3. **Structure Metrics:**
   - Table structure consistency
   - Row/column counting accuracy
   - Nested structure correctness

**Ví dụ sử dụng:**
```python
# Evaluation
results = evaluator.evaluate(predictions, ground_truth)

# Results format
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
- Tích hợp trong evaluation scripts

**Lưu ý đặc biệt:**
- TEDS là metric chính cho table recognition evaluation
- Structure consistency important cho practical applications
- Bbox accuracy affects downstream OCR performance
- Metric configuration should match dataset characteristics
