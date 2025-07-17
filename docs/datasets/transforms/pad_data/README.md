
# Pad Data

## 6. PadData

**Chức năng:** Transform để pad các chuỗi (tokens, bboxes, masks) tới độ dài cố định nhằm tạo batch đồng nhất. Xử lý đồng thời cả chuỗi token và chuỗi bbox.

**Đặc điểm:**
- Pad chuỗi token với padding dựa trên dictionary
- Pad chuỗi bbox và chuỗi mask
- Hỗ trợ nhiều chiến lược padding
- Tự động thêm token bắt đầu/kết thúc cho chuỗi
- Chuyển đổi sang tensor để huấn luyện

**Input:**
- `tokens`: Chuỗi token (dạng list hoặc tensor)
- `bboxes`: Chuỗi bounding box (dạng list)
- `masks`: Chuỗi mask (dạng list)
- `dictionary`: Đối tượng Dictionary dùng để chuyển đổi token

**Output:**
- `indexes`: Chỉ số token gốc
- `padded_indexes`: Chuỗi token đã pad (torch.Tensor)
- `padded_bboxes`: Chuỗi bbox đã pad (torch.Tensor)
- `padded_masks`: Chuỗi mask đã pad (torch.Tensor)
- `have_padded_indexes`, `have_padded_bboxes`: Cờ trạng thái

**Tham số cấu hình:**
- `dictionary`: Đối tượng Dictionary hoặc dict cấu hình
- `max_seq_len`: Độ dài tối đa cho chuỗi token. Mặc định 500
- `max_bbox_len`: Độ dài tối đa cho chuỗi bbox. Mặc định 500
- `pad_with`: Chiến lược padding ('auto', 'padding', 'end', 'none'). Mặc định 'auto'

**Quy trình xử lý:**

1. **Dictionary Setup:**
   - Tạo dictionary từ cấu hình nếu cần
   - Xác định padding index dựa trên chiến lược
   - Kiểm tra dictionary có đủ các chỉ số cần thiết

2. **Token Processing:**
   - Chuyển token thành chỉ số nếu cần
   - Tạo chuỗi target với token bắt đầu/kết thúc
   - Pad chuỗi tới max_seq_len

3. **Bbox và Mask Processing:**
   - Chuyển bbox sang dạng tensor
   - Pad chuỗi tới max_bbox_len
   - Đồng bộ độ dài bbox và mask

4. **Tensor Conversion:**
   - Chuyển tất cả chuỗi sang torch.Tensor
   - Đảm bảo kiểu dữ liệu đúng (LongTensor cho token, FloatTensor cho bbox)

**Padding Strategies:**
- `'auto'`: Sử dụng padding_idx hoặc end_idx
- `'padding'`: Sử dụng dictionary.padding_idx
- `'end'`: Sử dụng dictionary.end_idx
- `'none'`: Không pad

**Ví dụ cấu hình:**
```python
dict(
    type='PadData',
    dictionary=dictionary,
    max_seq_len=600,
    max_bbox_len=600,
    pad_with='auto'
)
```

**Ví dụ Input/Output:**
```python
# Input
tokens = ["<table>", "<td>", "</td>", "</table>"]
bboxes = [[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8]]
masks = [1, 1]

# Output (với max_seq_len=600, max_bbox_len=600)
padded_indexes = torch.LongTensor([start_idx, tok1_idx, tok2_idx, tok3_idx, tok4_idx, end_idx, pad_idx, ...])  # độ dài 600
padded_bboxes = torch.FloatTensor([[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8], [0, 0, 0, 0], ...])  # shape (600, 4)
padded_masks = torch.FloatTensor([1, 1, 0, 0, ...])  # độ dài 600
```


**Quan hệ với pipeline:**
- Nhận dữ liệu từ [BBox Encode](../bbox_encode/README.md)
- Truyền dữ liệu tới [Pack Inputs](../pack_inputs/README.md)

**Lưu ý đặc biệt:**
- Transform này rất quan trọng cho xử lý batch
- Chiến lược padding phải khớp với thiết lập dictionary
- Token bắt đầu/kết thúc được thêm tự động nếu dictionary hỗ trợ
- Kiểu tensor phải nhất quán với yêu cầu của model
