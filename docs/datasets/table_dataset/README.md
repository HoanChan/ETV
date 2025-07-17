# Table Dataset

## 1.1 PubTabNetDataset

**Chức năng:** Dataset chính cho nhận diện cấu trúc và nội dung bảng từ PubTabNet dataset. Tuân thủ chuẩn MMEngine Dataset và format 'instances' của mmOCR 1.x.

**Đặc điểm:**
- Load dữ liệu từ file annotation (.jsonl hoặc .bz2)
- Parse dữ liệu thành format chuẩn cho table recognition
- Hỗ trợ filter theo split (train/val/test)
- Hỗ trợ giới hạn số lượng dữ liệu và random sampling
- Xử lý cả structure tokens và cell content tokens

**Input:**
- `ann_file`: Đường dẫn file annotation (hỗ trợ .jsonl và .bz2)
- `data_prefix`: Dictionary chứa đường dẫn root của images
- Các tham số cấu hình (split_filter, max_structure_len, etc.)

**Output:** Mỗi sample trả về dictionary với cấu trúc:
```python
{
    "img_path": str,          # Đường dẫn file ảnh
    "sample_idx": int,        # Index của sample
    "instances": [            # Danh sách instances
        {
            "tokens": [str],        # Danh sách tokens
            "type": str,           # 'structure' hoặc 'content'
            "cell_id": int,        # ID của cell (chỉ cho 'content')
            "bbox": [int, int, int, int]  # Bounding box (chỉ cho 'content')
        }
    ],
    "img_info": {             # Thông tin về ảnh
        "height": int,        # Chiều cao ảnh
        "width": int,         # Chiều rộng ảnh
        "split": str          # Split của dữ liệu
    }
}
