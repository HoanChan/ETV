# Bbox Encode

## 5. BboxEncode

**Chức năng:** Transform để encode và normalize bounding boxes cho training. Chuyển đổi từ format (x1, y1, x2, y2) sang format (x, y, w, h) và normalize về range [0, 1].

**Đặc điểm:**
- Chuyển đổi bbox format từ xyxy sang xywh
- Normalize coordinates về range [0, 1]
- Validation bbox coordinates
- Xử lý cả list và numpy array formats
- Đảm bảo data type consistency

**Input:**
- `bboxes`: Bounding boxes ở format xyxy (list hoặc numpy.ndarray)
- `img_shape`: Shape của ảnh để normalize

**Output:**
- `bboxes`: Bounding boxes đã được encode và normalize ở format xywh
- `have_normalized_bboxes`: Flag indicating bboxes đã được normalize

**Quy trình xử lý:**

1. **Format Conversion:**
   - Kiểm tra và chuyển đổi input thành numpy array
   - Đảm bảo data type là float32
   - Validate input format

2. **Coordinate Transformation:**
   - Chuyển đổi từ (x1, y1, x2, y2) sang (x, y, w, h) format
   - Sử dụng utility function `xyxy2xywh()`

3. **Normalization:**
   - Normalize coordinates về range [0, 1] dựa trên img_shape
   - Sử dụng utility function `normalize_bbox()`

4. **Validation:**
   - Kiểm tra bbox coordinates có trong range [0, 1]
   - Validate bbox format (4 coordinates per box)
   - Raise assertion error nếu invalid

**Utility Functions:**
- `xyxy2xywh()`: Chuyển đổi bbox format
- `normalize_bbox()`: Normalize bbox coordinates
- `_check_bbox_valid()`: Validation function

**Ví dụ cấu hình:**
```python
dict(type='BboxEncode')
```

**Ví dụ Input/Output:**
```python
# Input
bboxes = [[10, 20, 50, 80]]  # xyxy format
img_shape = (100, 100)

# Output  
bboxes = [[0.3, 0.5, 0.4, 0.6]]  # xywh format, normalized
```

**Quan hệ với pipeline:**
- Nhận dữ liệu từ @import "../table_pad/README.md"
- Truyền dữ liệu tới @import "../pad_data/README.md"

**Lưu ý đặc biệt:**
- Transform này essential cho training stability
- Normalized coordinates giúp model convergence tốt hơn
- Validation strict để phát hiện data issues sớm
- Tương thích với cả list và numpy array inputs
