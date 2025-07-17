
# Postprocessors

## 5. TableMasterPostprocessor

**Chức năng:** Postprocessor để xử lý output thô từ decoder thành dự đoán có ý nghĩa. Chuyển đổi logits thành tokens và tọa độ bbox thành bounding boxes thực tế.

**Đặc điểm:**
- Chuyển đổi logits thành dự đoán token
- Giải mã tọa độ bbox với khử chuẩn hóa
- Áp dụng ngưỡng độ tin cậy (confidence thresholding)
- Xử lý sự không khớp độ dài chuỗi
- Tích hợp với dictionary để ánh xạ token

**Input:**
- `structure_outputs`: Logits phân loại (torch.Tensor)
- `bbox_outputs`: Output hồi quy bbox (torch.Tensor)
- `data_samples`: Batch dữ liệu với meta information

**Output:**
- `final_tokens`: Chuỗi token dự đoán (List[str])
- `final_bboxes`: Bounding boxes dự đoán (List[np.ndarray])
- `scores`: Độ tin cậy cho các dự đoán

**Tham số cấu hình:**
- `dictionary`: Đối tượng Dictionary để ánh xạ token
- `max_seq_len`: Độ dài chuỗi tối đa. Mặc định 600
- `start_end_same`: Start và end token có giống nhau không. Mặc định False

**Quy trình xử lý:**

1. **Dự đoán token:**
   - Chuyển logits thành chỉ số token
   - Áp dụng ngưỡng độ tin cậy
   - Ánh xạ chỉ số thành chuỗi token

2. **Dự đoán bbox:**
   - Trích xuất tọa độ bbox từ output hồi quy
   - Áp dụng mask bbox cho các dự đoán hợp lệ
   - Khử chuẩn hóa tọa độ về không gian ảnh gốc

3. **Khử chuẩn hóa tọa độ:**
   - Scale tọa độ từ [0,1] về pixel
   - Áp dụng scale_factor từ bước tiền xử lý ảnh
   - Điều chỉnh theo padding offsets

4. **Căn chỉnh chuỗi:**
   - Căn chỉnh bbox với chuỗi token dự đoán
   - Xử lý sự không khớp độ dài chuỗi
   - Lọc các dự đoán không hợp lệ

**Ví dụ cấu hình:**
```python
postprocessor = dict(
    type='TableMasterPostprocessor',
    dictionary=dictionary,
    max_seq_len=600,
    start_end_same=False
)
```

**Quy trình khử chuẩn hóa bbox:**
1. **Không gian padding:** Tọa độ bbox nằm trong khoảng [0,1] của ảnh đã padding
2. **Scale theo pad_shape:** Nhân với pad_shape để lấy tọa độ pixel
3. **Điều chỉnh scale_factor:** Chia cho scale_factor để lấy tọa độ gốc
4. **Giới hạn trong ảnh:** Clamp tọa độ trong phạm vi ảnh gốc

**Định dạng output:**
```python
# Dự đoán token
final_tokens = ["<table>", "<tr>", "<td>", "</td>", "</tr>", "</table>"]

# Dự đoán bbox (đã khử chuẩn hóa)
final_bboxes = [
    np.array([[10, 20, 100, 200],    # bbox cho <td>
              [15, 25, 95, 180]]),    # bbox khác cho <td>
    ...
]

# Độ tin cậy
scores = [0.95, 0.87, 0.92, ...]
```

**Quan hệ với pipeline:**
- Nhận output từ [Decoders](../decoders/README.md)
- Sử dụng [Dictionaries](../dictionaries/README.md) để ánh xạ token
- Sử dụng meta information từ [Pack Inputs](../../datasets/transforms/pack_inputs/README.md)

**Lưu ý đặc biệt:**
- Quy trình khử chuẩn hóa phải chính xác để lấy tọa độ bbox đúng
- Căn chỉnh chuỗi rất quan trọng cho sự tương ứng giữa cấu trúc và bbox
- Ngưỡng độ tin cậy giúp lọc dự đoán chất lượng thấp
- Meta information từ tiền xử lý rất quan trọng cho khử chuẩn hóa chính xác
