
# Pack Inputs

## 7. PackInputs

**Chức năng:** Transform cuối cùng trong pipeline để đóng gói toàn bộ dữ liệu đã xử lý thành định dạng chuẩn cho model TableMaster. Tạo ra `inputs` và `data_samples` theo chuẩn mmOCR.

**Đặc điểm:**
- Đóng gói ảnh thành tensor với chuẩn hóa
- Tạo TableMasterDataSample với gt_instances và gt_tokens
- Xử lý meta information cho training/inference
- Hỗ trợ tham số chuẩn hóa tuỳ chỉnh
- Tương thích với định dạng dữ liệu mmOCR

**Input:**
- `img`: Ảnh đã được pad (numpy.ndarray)
- `tokens`: Chuỗi token
- `bboxes`: Chuỗi bounding box
- `masks`: Chuỗi mask
- `padded_indexes`: Chỉ số token đã pad
- `padded_bboxes`: Chuỗi bbox đã pad
- `padded_masks`: Chuỗi mask đã pad
- Các meta information khác

**Output:**
- `inputs`: Tensor ảnh đã chuẩn hóa (torch.Tensor), input cho backbone
- `data_samples`: Đối tượng TableMasterDataSample gồm:
  - `gt_instances`: Dữ liệu instance với bboxes, masks
  - `gt_tokens`: Dữ liệu token với chuỗi
  - `metainfo`: Meta information cho training/inference

**Tham số cấu hình:**
- `keys`: Danh sách keys để pack thêm. Mặc định ()
- `meta_keys`: Danh sách meta keys để extract. Mặc định ('img_path', 'ori_shape', 'img_shape', 'pad_shape', 'valid_ratio')
- `mean`: Giá trị mean cho chuẩn hóa. Mặc định None
- `std`: Giá trị std cho chuẩn hóa. Mặc định None

**Quy trình xử lý:**

1. **Xử lý ảnh:**
   - Chuyển ảnh sang tensor (CHW)
   - Áp dụng chuẩn hóa với mean/std nếu có
   - Đảm bảo memory layout liên tục

2. **Tạo Data Sample:**
   - Tạo instance TableMasterDataSample
   - Pack gt_instances với bboxes, masks, dữ liệu đã pad
   - Pack gt_tokens với chuỗi token và chỉ số

3. **Meta Information:**
   - Extract meta keys từ kết quả
   - Thêm cấu hình chuẩn hóa nếu có
   - Set metainfo cho data sample

4. **Đóng gói cuối cùng:**
   - Tạo dict packed_results
   - Thêm inputs và data_samples
   - Pack thêm các keys nếu được chỉ định

**Ví dụ cấu hình:**
```python
dict(
    type='PackInputs',
    keys=['img'],
    mean=[0.5, 0.5, 0.5],
    std=[0.5, 0.5, 0.5],
    meta_keys=('ori_shape', 'img_shape', 'scale_factor', 'pad_shape', 'valid_ratio')
)
```


**Cấu trúc Data Sample:**
```python
data_samples = TableMasterDataSample(
    gt_instances=InstanceData(
        bboxes=original_bboxes,
        masks=original_masks,
        padded_bboxes=padded_bboxes,
        padded_masks=padded_masks
    ),
    gt_tokens=LabelData(
        item=tokens,
        indexes=indexes,
        padded_indexes=padded_indexes,
        have_padded_indexes=True
    ),
    metainfo={
        'ori_shape': (H, W),
        'img_shape': (H', W'),
        'scale_factor': [sx, sy],
        'pad_shape': (H'', W''),
        'valid_ratio': ratio,
        'img_norm_cfg': {'mean': [...], 'std': [...]}
    }
)
```

**Quan hệ với pipeline:**
- Nhận dữ liệu từ [Pad Data](../pad_data/README.md)
- Kết thúc pipeline, truyền tới model

**Lưu ý đặc biệt:**
- Transform này là giao diện giữa pipeline và model
- Cấu trúc TableMasterDataSample phải khớp với yêu cầu của model
- Tham số chuẩn hóa phải nhất quán với quá trình huấn luyện model
- Meta information rất quan trọng cho suy luận và đánh giá
