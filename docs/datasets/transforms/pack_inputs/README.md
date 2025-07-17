# Pack Inputs

## 7. PackInputs

**Chức năng:** Transform cuối cùng trong pipeline để đóng gói toàn bộ dữ liệu đã xử lý thành format chuẩn cho model TableMaster. Tạo ra `inputs` và `data_samples` theo chuẩn mmOCR.

**Đặc điểm:**
- Đóng gói ảnh thành tensor format với normalization
- Tạo TableMasterDataSample với gt_instances và gt_tokens
- Xử lý meta information cho training/inference
- Hỗ trợ custom normalization parameters
- Tương thích với mmOCR data format

**Input:**
- `img`: Ảnh đã được pad (numpy.ndarray)
- `tokens`: Token sequences
- `bboxes`: Bounding box sequences
- `masks`: Mask sequences
- `padded_indexes`: Padded token indexes
- `padded_bboxes`: Padded bbox sequences
- `padded_masks`: Padded mask sequences
- Các meta information khác

**Output:**
- `inputs`: Tensor ảnh đã chuẩn hóa (torch.Tensor), input cho backbone
- `data_samples`: TableMasterDataSample object chứa:
  - `gt_instances`: Instance data với bboxes, masks
  - `gt_tokens`: Token data với sequences
  - `metainfo`: Meta information cho training/inference

**Tham số cấu hình:**
- `keys`: Danh sách keys để pack thêm. Mặc định ()
- `meta_keys`: Danh sách meta keys để extract. Mặc định ('img_path', 'ori_shape', 'img_shape', 'pad_shape', 'valid_ratio')
- `mean`: Mean values cho normalization. Mặc định None
- `std`: Std values cho normalization. Mặc định None

**Quy trình xử lý:**

1. **Image Processing:**
   - Convert ảnh thành tensor format (CHW)
   - Áp dụng normalization với mean/std nếu có
   - Đảm bảo contiguous memory layout

2. **Data Sample Creation:**
   - Tạo TableMasterDataSample instance
   - Pack gt_instances với bboxes, masks, padded data
   - Pack gt_tokens với token sequences và indexes

3. **Meta Information:**
   - Extract meta keys từ results
   - Thêm normalization config nếu có
   - Set metainfo cho data sample

4. **Final Packaging:**
   - Tạo packed_results dict
   - Thêm inputs và data_samples
   - Pack additional keys nếu được chỉ định

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

**Data Sample Structure:**
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
- Transform này là interface giữa pipeline và model
- TableMasterDataSample structure phải match với model expectations
- Normalization parameters phải consistent với model training
- Meta information essential cho inference và evaluation
