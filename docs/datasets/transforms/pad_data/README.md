# Pad Data

6. **PadData**
   - Chức năng: Padding các trường dữ liệu để đảm bảo đầu vào đồng nhất, phù hợp với batch và kiến trúc mô hình.
   - Input:
       - `tokens` (list[str])
       - `bboxes` (numpy.ndarray hoặc list)
       - `masks` (numpy.ndarray hoặc list)
   - Output:
       - padded indexes (numpy.ndarray hoặc list)
       - padded bboxes/masks (numpy.ndarray hoặc list)
