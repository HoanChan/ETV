# Postprocessors

#### 4. TableMasterPostprocessor (Postprocessor)
**Chức năng:** Giải mã kết quả dự đoán về dạng cuối cùng (token, bbox) và chuẩn hóa theo meta info.
**Kiến trúc:**
- Nhận đầu vào là tuple (`token_logits`, `bbox_logits`) và meta info từ `data_samples`.
- Áp dụng giải mã token, chuẩn hóa bbox về kích thước gốc ảnh.
- Output: `final_tokens` (list[str]), `final_bboxes` (numpy.ndarray hoặc list).

**Input:**
- `token_logits` (torch.Tensor): Dự đoán nhãn cho từng token.
- `bbox_logits` (torch.Tensor): Dự đoán vị trí bounding box cho từng ô.
- `data_samples` (meta info): Thông tin ảnh, scale, pad...
**Output:**
- `final_tokens` (list[str]): Chuỗi token đã giải mã.
- `final_bboxes` (numpy.ndarray hoặc list): Vị trí các ô bảng đã chuẩn hóa.
