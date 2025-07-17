# Pack Inputs

7. **PackInputs**
   - Chức năng: Đóng gói toàn bộ dữ liệu đã xử lý thành các trường đầu vào chuẩn cho mô hình TableMaster.
   - Input:
       - Tất cả các trường đã xử lý ở các bước trên
   - Output:
       - `inputs`: Tensor ảnh đã chuẩn hóa (torch.Tensor hoặc numpy.ndarray), dùng làm đầu vào cho backbone của mô hình.
       - `data_samples`: Đối tượng `TableMasterDataSample` (class), chứa thông tin ground truth (`gt_instances`, `gt_tokens`) và các meta info cần thiết cho quá trình huấn luyện và đánh giá.
