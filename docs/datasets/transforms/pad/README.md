# Table Pad

## 4. TablePad

**Chức năng:** Transform để pad ảnh bảng và tạo attention mask. Kế thừa từ MMDetection's Pad class và thêm tính năng tạo mask cho table recognition.

**Đặc điểm:**
- Kế thừa từ MMDetection's Pad class
- Hỗ trợ padding tới fixed size hoặc size divisible
- Tạo attention mask cho vùng hợp lệ
- Hỗ trợ mask downsampling với configurable ratio
- Xử lý cả ảnh và bboxes đồng thời

**Input:**
- `img`: Ảnh đã được resize (numpy.ndarray)
- `bboxes` (optional): Danh sách bounding boxes
- `gt_masks` (optional): Ground truth masks
- `gt_seg_map` (optional): Segmentation maps

**Output:**
- `img`: Ảnh đã được pad
- `img_shape`: Shape của ảnh sau khi pad
- `pad_shape`: Shape được pad tới
- `pad_fixed_size`: Fixed size được sử dụng (nếu có)
- `pad_size_divisor`: Size divisor được sử dụng (nếu có)
- `mask` (optional): Attention mask cho vùng hợp lệ (nếu `return_mask=True`)
- Các fields khác được cập nhật tương ứng

**Tham số cấu hình:**
- `size` (tuple, optional): Fixed padding size (width, height)
- `size_divisor` (int, optional): Divisor cho padded size
- `pad_val` (int): Giá trị padding. Mặc định 0
- `pad_to_square` (bool): Có pad thành hình vuông không. Mặc định False
- `padding_mode` (str): Chế độ padding ('constant', 'edge', 'reflect', 'symmetric'). Mặc định 'constant'
- `return_mask` (bool): Có trả về attention mask không. Mặc định False
- `mask_ratio` (int|tuple): Tỷ lệ downsample mask. Mặc định 1

**Quy trình xử lý:**

1. **Padding Process:**
   - Sử dụng logic padding của parent class
   - Cập nhật `img_shape`, `pad_shape`
   - Xử lý bboxes và masks tương ứng

2. **Mask Generation (nếu `return_mask=True`):**
   - Tạo mask với kích thước = `size`
   - Fill mask với 0 (invalid region)
   - Đánh dấu vùng hợp lệ (original image area) với 1
   - Áp dụng mask downsampling theo `mask_ratio`

3. **Meta Information Update:**
   - Cập nhật `pad_fixed_size` và `pad_size_divisor`
   - Thêm channel dimension cho mask

**Ví dụ cấu hình:**
```python
# Pad tới fixed size với mask
dict(
    type='TablePad',
    size=(480, 480),
    pad_val=0,
    return_mask=True,
    mask_ratio=(8, 8)
)

# Pad tới size divisible
dict(
    type='TablePad',
    size_divisor=32,
    pad_val=0,
    return_mask=False
)
```

**Mask Downsampling:**
- Nếu `mask_ratio` là int: downsample theo cùng ratio cho cả 2 dimensions
- Nếu `mask_ratio` là tuple: downsample theo ratio khác nhau cho height và width
- Ví dụ: `mask_ratio=(8, 8)` sẽ downsample mask từ (480, 480) xuống (60, 60)

**Quan hệ với pipeline:**
- Nhận dữ liệu từ [Resize](../resize/README.md)
- Truyền dữ liệu tới [BBox Encode](../bbox_encode/README.md)

**Lưu ý đặc biệt:**
- Mask được tạo để hỗ trợ attention mechanism trong model
- Mask ratio thường được set match với stride của backbone feature extraction
- Transform này đảm bảo all images có cùng kích thước sau khi pad
- Tương thích với MMDetection's data format
