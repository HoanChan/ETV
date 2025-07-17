# Pad Data

## 6. PadData

**Chức năng:** Transform để pad sequences (tokens, bboxes, masks) tới fixed length để tạo batch uniform. Xử lý cả token sequences và bbox sequences đồng thời.

**Đặc điểm:**
- Pad token sequences với dictionary-based padding
- Pad bbox sequences và mask sequences
- Hỗ trợ multiple padding strategies
- Tạo start/end tokens cho sequences
- Tensor conversion cho training

**Input:**
- `tokens`: Token sequences (list hoặc tensor)
- `bboxes`: Bounding box sequences (list)
- `masks`: Mask sequences (list)
- `dictionary`: Dictionary instance cho token conversion

**Output:**
- `indexes`: Original token indexes
- `padded_indexes`: Padded token sequences (torch.Tensor)
- `padded_bboxes`: Padded bbox sequences (torch.Tensor)
- `padded_masks`: Padded mask sequences (torch.Tensor)
- `have_padded_indexes`, `have_padded_bboxes`: Status flags

**Tham số cấu hình:**
- `dictionary`: Dictionary instance hoặc config dict
- `max_seq_len`: Maximum sequence length cho tokens. Mặc định 500
- `max_bbox_len`: Maximum sequence length cho bboxes. Mặc định 500
- `pad_with`: Padding strategy ('auto', 'padding', 'end', 'none'). Mặc định 'auto'

**Quy trình xử lý:**

1. **Dictionary Setup:**
   - Build dictionary từ config nếu cần
   - Determine padding index dựa trên strategy
   - Validate dictionary có required indices

2. **Token Processing:**
   - Convert tokens thành indices nếu cần
   - Create target sequence với start/end tokens
   - Apply padding tới max_seq_len

3. **Bbox và Mask Processing:**
   - Convert bboxes thành tensor format
   - Pad sequences tới max_bbox_len
   - Sync bbox và mask lengths

4. **Tensor Conversion:**
   - Convert tất cả sequences thành torch.Tensor
   - Ensure correct data types (LongTensor cho tokens, FloatTensor cho bboxes)

**Padding Strategies:**
- `'auto'`: Sử dụng padding_idx hoặc end_idx
- `'padding'`: Sử dụng dictionary.padding_idx
- `'end'`: Sử dụng dictionary.end_idx
- `'none'`: Không padding

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
padded_indexes = torch.LongTensor([start_idx, tok1_idx, tok2_idx, tok3_idx, tok4_idx, end_idx, pad_idx, ...])  # length 600
padded_bboxes = torch.FloatTensor([[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8], [0, 0, 0, 0], ...])  # shape (600, 4)
padded_masks = torch.FloatTensor([1, 1, 0, 0, ...])  # length 600
```

**Quan hệ với pipeline:**
- Nhận dữ liệu từ @import "../bbox_encode/README.md"
- Truyền dữ liệu tới @import "../pack_inputs/README.md"

**Lưu ý đặc biệt:**
- Transform này critical cho batch processing
- Padding strategy phải match với dictionary setup
- Start/end tokens được add tự động nếu dictionary support
- Tensor types phải consistent với model expectations
