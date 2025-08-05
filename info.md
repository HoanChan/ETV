# Dự án nhận diện cấu trúc bảng TableMaster

Code được viết trên mmOCR 1.0.1, nâng cấp từ phiên bản mmOCR 0.2.0.

# Thông tin chung

- Thư mục `old` chứa project TableMaster ở phiên bản `mmOCR 0.2.0`.
- Thư mục `src` chứa project TableMaster ở phiên bản `mmOCR 1.0.1`.
- Tôi đang cố gắng chuyển đổi code từ `old` sang `src` để sử dụng các tính năng mới của `mmOCR 1.0.1`.

# Lưu ý điểm khác biệt và nâng cấp của phiên bản mới so với phiên bản cũ:

1. **Cấu trúc dataset**:

- Phiên bản cũ sử dụng dataset dựa trên text, dataset cho nhận diện cấu trúc bảng riêng, cho nhận diện vị trí và nội dung văn bản riêng. Nó sử dụng code preprocess để chuyển đổi dữ liệu từ định dạng JSON sang định dạng text. Sinh ra rất nhiều file nhỏ cho từng cell trong bảng.
- Phiên bản mới sử dụng dataset dựa trên JSON theo chuẩn của `PubTabNet`, bao gồm thông tin về cấu trúc bảng, vị trí và nội dung văn bản trong một dòng JSON và toàn bộ dữ liệu được lưu trong một file duy nhất.
- Nhờ kiến trúc dataset mới, các dữ liệu cần cho huấn luyện đề đọc trực tiếp từ dataset gốc mà không cần preprocess phức tạp. 

2. **Thuật ngữ**:

- Phiên bản cũ sử dụng thuật ngữ `text` là các `tokens` được ghép lại cho đúng kiến trúc Master gốc. Khi xử lý cần `split` thành các `tokens` và `join` lại sau khi xử lý để tạo thành `text` hoàn chỉnh.
- Phiên bản mới sử dụng thuật ngữ `tokens` cho phù hợp với ý tưởng thuật toán, giảm các bước trung gian không cần thiết.

3. **Cấu trúc mô hình**:

- Để đảm bảo được kết quả kết quả là như nhau ở 2 phiên bản, cần giữ nguyên toàn bộ cấu trúc mô hình, từ việc nhận input tới việc sinh output.
- Các hàm loss và metric cũng cần giữ nguyên để đảm bảo tính tương thích với các mô hình đã huấn luyện trước đó.