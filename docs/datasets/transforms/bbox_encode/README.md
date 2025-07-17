
# Bbox Encode

## 5. BboxEncode

**Chức năng:** Transform dùng để encode và chuẩn hoá bounding boxes cho quá trình huấn luyện. Chuyển đổi từ định dạng (x1, y1, x2, y2) sang (x, y, w, h) và chuẩn hoá về khoảng [0, 1].

**Đặc điểm:**
- Chuyển đổi định dạng bbox từ xyxy sang xywh
- Chuẩn hoá toạ độ về khoảng [0, 1]
- Kiểm tra hợp lệ toạ độ bbox
- Xử lý cả hai kiểu list và numpy array
- Đảm bảo tính nhất quán kiểu dữ liệu

**Input:**
- `bboxes`: Bounding boxes ở định dạng xyxy (list hoặc numpy.ndarray)
- `img_shape`: Kích thước ảnh để chuẩn hoá

**Output:**
- `bboxes`: Bounding boxes đã được encode và chuẩn hoá ở định dạng xywh
- `have_normalized_bboxes`: Cờ báo hiệu bboxes đã được chuẩn hoá

**Quy trình xử lý:**

1. **Chuyển đổi định dạng:**
   - Kiểm tra và chuyển đổi input thành numpy array
   - Đảm bảo kiểu dữ liệu là float32
   - Kiểm tra hợp lệ định dạng input

2. **Chuyển đổi toạ độ:**
   - Chuyển từ (x1, y1, x2, y2) sang (x, y, w, h)
   - Sử dụng hàm tiện ích `xyxy2xywh()`

3. **Chuẩn hoá:**
   - Chuẩn hoá toạ độ về khoảng [0, 1] dựa trên img_shape
   - Sử dụng hàm tiện ích `normalize_bbox()`

4. **Kiểm tra hợp lệ:**
   - Kiểm tra toạ độ bbox nằm trong khoảng [0, 1]
   - Kiểm tra định dạng bbox (4 toạ độ mỗi box)
   - Báo lỗi nếu không hợp lệ

**Các hàm tiện ích:**
- `xyxy2xywh()`: Chuyển đổi định dạng bbox
- `normalize_bbox()`: Chuẩn hoá toạ độ bbox
- `_check_bbox_valid()`: Hàm kiểm tra hợp lệ

**Ví dụ cấu hình:**
```python
dict(type='BboxEncode')
```

**Ví dụ Input/Output:**
```python
# Input
bboxes = [[10, 20, 50, 80]]  # định dạng xyxy
img_shape = (100, 100)

# Output  
bboxes = [[0.3, 0.5, 0.4, 0.6]]  # định dạng xywh, đã chuẩn hoá
```

**Quan hệ với pipeline:**
- Nhận dữ liệu từ [Pad](../pad/README.md)
- Truyền dữ liệu tới [Pad Data](../pad_data/README.md)

**Lưu ý đặc biệt:**
- Transform này rất quan trọng cho sự ổn định khi huấn luyện
- Toạ độ đã chuẩn hoá giúp model hội tụ tốt hơn
- Kiểm tra nghiêm ngặt để phát hiện lỗi dữ liệu sớm
- Tương thích với cả kiểu list và numpy array
