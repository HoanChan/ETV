# Load Tokens

## 2. LoadTokens

**Chức năng:** Transform để load và xử lý token annotations từ instances trong table dataset. Thực hiện chuẩn hóa structure tokens và xử lý cell content tokens.

**Đặc điểm:**
- Loại bỏ bold tags (`<b>`, `</b>`) từ table header cells
- Merge common tokens và chèn empty bbox tokens
- Xử lý cả structure tokens và cell content tokens
- Tính toán bbox và mask cho structure tokens
- Hỗ trợ giới hạn độ dài token sequences

**Input:**
- `instances`: Danh sách instances từ dataset với cấu trúc:
  ```python
  [
      {
          "tokens": [str],     # Danh sách tokens
          "type": str,         # 'structure' hoặc 'content'
          "cell_id": int,      # ID của cell (chỉ cho 'content')
          "bbox": [int, int, int, int]  # Bounding box (chỉ cho 'content')
      }
  ]
  ```

**Output:**
- Nếu `with_structure=True`:
  - `tokens`: Danh sách structure tokens đã được chuẩn hóa
  - `bboxes`: Danh sách bounding boxes tương ứng với tokens
  - `masks`: Mask cho bboxes (1 cho valid bbox, 0 cho empty bbox)
- Nếu `with_cell=True`:
  - `cells`: Danh sách thông tin cell:
    ```python
    [
        {
            "tokens": [str],     # Tokens của cell
            "bbox": [int, int, int, int],  # Bounding box của cell
            "id": int            # ID của cell
        }
    ]
    ```

**Tham số cấu hình:**
- `with_structure` (bool): Có load structure tokens hay không. Mặc định True
- `with_cell` (bool): Có load cell tokens hay không. Mặc định True  
- `max_structure_token_len` (int, optional): Giới hạn số lượng structure tokens
- `max_cell_token_len` (int, optional): Giới hạn số lượng cell tokens

**Quy trình xử lý:**

1. **Token Normalization:**
   - Tách instances thành structure và content
   - Loại bỏ `<b>` và `</b>` tags từ header cells
   - Merge common tokens và chèn empty bbox tokens

2. **Structure Token Processing:**
   - Trích xuất structure tokens từ instances
   - Áp dụng giới hạn độ dài nếu được cấu hình
   - Tính toán số lượng bbox cần thiết

3. **Cell Information Processing:**
   - Tạo danh sách cell information
   - Xử lý empty cells với bbox [0, 0, 0, 0]
   - Đảm bảo cell_id là duy nhất

4. **Bbox và Mask Calculation:**
   - Tính toán bbox alignment với structure tokens
   - Tạo mask cho valid/invalid bboxes
   - Áp dụng empty bbox mask

**Ví dụ cấu hình:**
```python
dict(
    type='LoadTokens',
    with_structure=True,
    with_cell=True,
    max_structure_token_len=600,
    max_cell_token_len=150
)
```

**Quan hệ với pipeline:**
- Nhận dữ liệu từ @import "../../../datasets/table_dataset/README.md"
- Truyền dữ liệu tới @import "../table_resize/README.md"

**Utilities Functions:**
Transform này sử dụng các utility functions từ `transforms_utils.py`:
- `remove_thead_Bb()`: Loại bỏ bold tags
- `process_token()`: Xử lý token normalization
- `get_bbox_nums()`: Tính số lượng bbox cần thiết
- `build_empty_bbox_mask()`: Tạo mask cho empty bbox
- `align_bbox_mask()`: Align bbox với tokens
- `build_bbox_mask()`: Tạo bbox mask cho structure tokens
