# Table Dataset

### 1.1 Dataset

**Loại:** `PubTabNetDataset`

**Chức năng:**
  - Dataset cho nhận diện cấu trúc và nội dung bảng, tuân thủ chuẩn mmOCR 1.x.
  - Mỗi sample trả về dict với key 'instances' chứa thông tin cho từng nhiệm vụ (structure/content).

**Định dạng annotation:**
```json
{
    "filename": str,
    "split": str,  // "train", "val", hoặc "test"
    "imgid": int,
    "html": {
        "structure": {"tokens": [str]},
        "cells": [
            {
                "tokens": [str],
                "bbox": [x0, y0, x1, y1] // chỉ có với ô không rỗng
            }
        ]
    }
}
```

**Định dạng output:**
```json
{
    "img_path": str,  // đường dẫn ảnh
    "sample_idx": int,  // chỉ số mẫu
    "instances": [
        {
            "tokens": [str],
            "type": "structure" | "content",
            "cell_id": int,           // chỉ với content
            "bbox": [x0, y0, x1, y1]  // chỉ với content
        }
    ],
