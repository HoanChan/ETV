# 🧾 ETV: End-to-End Table Vision

ETV (End-to-End Table Vision) là dự án Deep Learning giúp chuyển đổi bảng từ giấy (hình ảnh, scan) sang định dạng Excel hoặc HTML một cách tự động.

Mô hình tham khảo từ [TableMASTER-mmocr](https://github.com/JiaquanYe/TableMASTER-mmocr/tree/master)
Sử dụng kiến trúc chính của [mmOCR](https://github.com/open-mmlab/mmocr/tree/main)

## 🎯 Mục tiêu

- 📄 Nhận diện và trích xuất bảng từ hình ảnh tài liệu, hóa đơn, biểu mẫu, v.v.
- 🔄 Chuyển đổi bảng đã nhận diện sang file Excel (.xlsx) hoặc HTML table.
- 🧩 Hỗ trợ nhiều loại bảng với cấu trúc đa dạng.

## ✨ Tính năng chính

- 🕵️‍♂️ Phát hiện vị trí bảng trong ảnh.
- 🏗️ Nhận diện cấu trúc bảng (hàng, cột, ô).
- 🔍 Nhận diện và trích xuất nội dung từng ô.
- 💾 Xuất kết quả sang Excel hoặc HTML.

## 📚 Các thư viện được sử dụng:

- 🤖 `PyTorch`, `torchvision`: Nền tảng deep learning.
- 🦾 `MMDetection`, `MMOCR`: Nhận diện bảng, nhận diện ký tự quang học (OCR).
- 🖼️ `OpenCV`: Xử lý ảnh.
- 📝 `pandas`, `openpyxl`: Xử lý dữ liệu bảng, xuất file Excel.
- 🌐 `BeautifulSoup4`: Tạo bảng HTML.
- 🧪 `pytest`: Kiểm thử tự động.

## ⚙️ Cài đặt

Cài đặt các thư viện phụ thuộc:
```bash
pip install -r requirements.txt
```

## 🚀 Sử dụng

Ví dụ sử dụng script để chuyển đổi ảnh bảng sang Excel:
```bash
python src/table_inference.py --input path/to/image.jpg --output result.xlsx
```

## 📁 Thư mục chính

- `src/`: Mã nguồn chính cho inference và các module xử lý bảng.
- `configs/`: Cấu hình mô hình.
- `notebooks/`: Notebook hướng dẫn và thử nghiệm.
- `tests/`: Unit test.
- `mmdetection/`, `mmocr/`: Thư viện phụ trợ cho nhận diện bảng và ký tự.

## 📜 License

MIT License
