### Kiến trúc của hệ thống nhận diện cấu trúc tài liệu
- **PSENet**: Sử dụng để phát hiện các vùng văn bản trong tài liệu.
- **MASTER**: Sử dụng để nhận diện văn bản trong các vùng đã phát hiện.
- **Custom Model**: Fork từ MASTER, được điều chỉnh để dự đoán cấu trúc bảng biểu.
- **Custom Metric (TEDS)**: Được sử dụng để đánh giá độ chính xác của mô hình trong việc nhận diện cấu trúc tài liệu.

