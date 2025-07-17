# Load Tokens

2. **LoadTokens**
   - Chức năng: Trích xuất thông tin cấu trúc bảng, vị trí ô, và mask từ annotation để phục vụ nhận diện và xử lý bảng.
   - Input:
       - Annotation JSON (`str` hoặc dict)
   - Output:
       - `tokens` (list[str])
       - `bboxes` (list[list[int]])
       - `masks` (list hoặc numpy.ndarray)
