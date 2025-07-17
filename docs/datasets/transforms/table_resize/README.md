# Table Resize

## 3. TableResize

**Chức năng:** Transform để resize ảnh bảng với các constraints về min_size và long_size. Kế thừa từ MMOCR's Resize class và thêm các tính năng đặc biệt cho table recognition.

**Đặc điểm:**
- Kế thừa từ MMOCR's Resize class
- Hỗ trợ min_size constraint cho cạnh ngắn hơn
- Hỗ trợ long_size constraint cho cạnh dài hơn
- Tự động tính toán scale factor
- Giữ nguyên aspect ratio
- Xử lý đặc biệt cho bboxes structure

**Input:**
- `img`: Ảnh bảng gốc (numpy.ndarray)
- `bboxes` (optional): Danh sách bounding boxes cần resize
- Các thông tin meta khác từ pipeline

**Output:**
- `img`: Ảnh đã được resize
- `img_shape`: Shape mới của ảnh
- `scale_factor`: Tỷ lệ scale được áp dụng
- `bboxes`: Bounding boxes đã được resize (nếu có)
- Các thông tin meta khác được cập nhật

**Tham số cấu hình:**
- `min_size` (int, optional): Kích thước tối thiểu cho cạnh ngắn hơn
- `long_size` (int, optional): Kích thước mục tiêu cho cạnh dài hơn. Nếu có, min_size sẽ bị bỏ qua
- `keep_ratio` (bool): Giữ nguyên aspect ratio. Mặc định True
- Các tham số khác kế thừa từ MMOCR's Resize

**Quy trình xử lý:**

1. **Size Calculation:**
   - Lấy kích thước gốc của ảnh (h, w)
   - Tính toán scale factor dựa trên constraints:
     - Nếu có `long_size`: scale = long_size / max(w, h)
     - Nếu có `min_size`: scale = min_size / min(w, h)
   - Tính toán kích thước mới: new_w = w * scale, new_h = h * scale

2. **Image Resizing:**
   - Sử dụng logic resize của parent class
   - Cập nhật `img_shape` và `scale_factor`

3. **Bbox Resizing:**
   - Xử lý đặc biệt cho structure bboxes
   - Tạm thời chuyển `bboxes` thành `gt_bboxes` để sử dụng logic parent
   - Áp dụng scale factor cho các coordinates
   - Chuyển ngược lại thành `bboxes`

**Ví dụ cấu hình:**
```python
# Resize với long_size constraint
dict(
    type='TableResize',
    keep_ratio=True,
    long_size=480
)

# Resize với min_size constraint
dict(
    type='TableResize',
    keep_ratio=True,
    min_size=320
)
```

**Quan hệ với pipeline:**
- Nhận dữ liệu từ @import "../load_tokens/README.md"  
- Truyền dữ liệu tới @import "../table_pad/README.md"

**Lưu ý đặc biệt:**
- Transform này được thiết kế đặc biệt cho table structure recognition
- Xử lý cả ảnh và bboxes đồng thời để đảm bảo consistency
- Tự động đặt `keep_ratio=True` để tránh deformation của bảng
- Tương thích với format bboxes của table dataset
