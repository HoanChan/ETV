# Recognizer

1. **TableMaster**
   - Chức năng: Mô hình tổng thể thực hiện nhận diện cấu trúc bảng và vị trí các ô, gồm các bước: tiền xử lý, backbone, encoder, decoder, tính loss, dự đoán và hậu xử lý.
   - Các phương thức chính:
       - `extract_feat`: Trích xuất đặc trưng từ ảnh đầu vào (qua preprocessor, backbone).
       - `loss`: Tính toán loss cho token và bbox (gọi decoder, bbox_loss).
       - `predict`: Dự đoán token, bbox, trả về kết quả đã giải mã.
       - `_forward`: Forward qua backbone, encoder, decoder (không hậu xử lý).
   - Input:
       - `inputs` (torch.Tensor): Ảnh đầu vào đã chuẩn hóa.
       - `data_samples` (List[TableMasterDataSample]): Ground truth, meta info.
   - Output:
       - `token logits` (torch.Tensor): Dự đoán nhãn cho từng token.
       - `bbox logits` (torch.Tensor): Dự đoán vị trí các bounding box.
       - `final tokens, bboxes` (list, numpy.ndarray): Kết quả cuối cùng sau hậu xử lý.
       - `loss` (dict): Giá trị mất mát cho token, bbox.
