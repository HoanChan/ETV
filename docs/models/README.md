# Model Documentation

## Tổng quan về Architecture TableMaster

TableMaster là một end-to-end model cho table structure recognition, sử dụng dual-head architecture để predict cả table structure tokens và bounding box coordinates.

## Cấu trúc tài liệu

### Architecture Overview
[Xem chi tiết: Overview](overview/README.md)

### Core Components

#### 1. Backbone Networks
[Xem chi tiết: Backbone Networks](backbones/README.md)

#### 2. Encoders
[Xem chi tiết: Encoders](encoders/README.md)

#### 3. Decoders
[Xem chi tiết: Decoders](decoders/README.md)

#### 4. Dictionaries
[Xem chi tiết: Dictionaries](dictionaries/README.md)

#### 5. Postprocessors
[Xem chi tiết: Postprocessors](postprocessors/README.md)

#### 6. Recognizers
[Xem chi tiết: Recognizers](recognizer/README.md)

### Training Components

#### 7. Loss Functions
[Xem chi tiết: Loss Functions](losses/README.md)

#### 8. Evaluation Metrics
[Xem chi tiết: Evaluation Metrics](metrics/README.md)

#### 9. Neural Network Layers
[Xem chi tiết: Neural Network Layers](layers/README.md)

## Kiến trúc tổng thể


Model TableMaster gồm các thành phần chính:

1. **Trích xuất đặc trưng:** Backbone TableResNetExtra với Global Context Blocks
2. **Mã hóa chuỗi:** PositionalEncoding để thêm thông tin vị trí không gian
3. **Giải mã hai đầu:** TableMasterConcatDecoder cho dự đoán token và bbox
4. **Quản lý token:** TableMasterDictionary cho ánh xạ token cấu trúc
5. **Xử lý đầu ra:** TableMasterPostprocessor để chuyển đổi dự đoán thô

## Huấn luyện và Suy luận

### Quy trình huấn luyện
- Học đa nhiệm với mục tiêu phân loại và hồi quy
- MasterTFLoss cho dự đoán token
- TableL1Loss cho hồi quy bbox
- Cân bằng gradient giữa hai đầu

### Quy trình suy luận
- Forward pass qua toàn bộ pipeline
- Hậu xử lý để tạo output có ý nghĩa
- Ngưỡng độ tin cậy và khử chuẩn hóa tọa độ

## Đánh giá

Model được đánh giá bằng:
- Chỉ số TEDS (Tree Edit Distance based Similarity)
- Độ chính xác token và bbox IoU
- Kiểm tra tính nhất quán cấu trúc

## Lưu ý quan trọng

- **Thiết kế hai đầu** rất quan trọng cho bài toán nhận diện bảng
- **Đặc trưng đa tỷ lệ** quan trọng cho hiểu không gian
- **Attention mechanism** giúp mô hình hóa quan hệ cấu trúc bảng
- **Thiết kế dictionary** ảnh hưởng đến từ vựng và hiệu năng mô hình
- **Hậu xử lý** quyết định ứng dụng thực tế
