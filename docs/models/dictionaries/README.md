# Dictionaries

## 4. TableMasterDictionary

**Chức năng:** Dictionary class để quản lý table structure tokens và special tokens cho TableMaster model. Xử lý multi-character tokens khác với standard character-based dictionaries.

**Đặc điểm:**
- Quản lý table structure tokens (`<td>`, `<tr>`, `<tbody>`, etc.)
- Hỗ trợ special tokens (`<BOS>`, `<EOS>`, `<PAD>`, `<UKN>`)
- Multi-character token handling
- Flexible token configuration
- Tương thích với sequence-to-sequence models

**Input:**
- `dict_file`: Đường dẫn tới structure alphabet file
- Các token configuration parameters

**Output:**
- Dictionary instance với token mapping functions:
  - `str2idx()`: Convert tokens thành indices
  - `idx2str()`: Convert indices thành tokens
  - `num_classes`: Tổng số classes/tokens

**Tham số cấu hình:**
- `dict_file`: Path tới structure alphabet file
- `with_start`: Có thêm start token hay không. Mặc định False
- `with_end`: Có thêm end token hay không. Mặc định False
- `same_start_end`: Start và end tokens có giống nhau hay không. Mặc định False
- `with_padding`: Có thêm padding token hay không. Mặc định False
- `with_unknown`: Có thêm unknown token hay không. Mặc định False
- `start_token`: Start token string. Mặc định '<BOS>'
- `end_token`: End token string. Mặc định '<EOS>'
- `start_end_token`: Combined start/end token. Mặc định '<BOS/EOS>'
- `padding_token`: Padding token string. Mặc định '<PAD>'
- `unknown_token`: Unknown token string. Mặc định '<UKN>'

**Structure Alphabet Format:**
File chứa danh sách structure tokens, mỗi token trên một dòng:
```
<table>
</table>
<tbody>
</tbody>
<thead>
</thead>
<tr>
</tr>
<td>
</td>
<eb></eb>
<eb1></eb1>
...
```

**Token Categories:**
1. **Table Structure Tokens:**
   - `<table>`, `</table>`: Table boundaries
   - `<tr>`, `</tr>`: Table rows
   - `<td>`, `</td>`: Table cells
   - `<thead>`, `<tbody>`: Table sections

2. **Special Tokens:**
   - `<BOS>`: Beginning of sequence
   - `<EOS>`: End of sequence
   - `<PAD>`: Padding token
   - `<UKN>`: Unknown token

3. **Empty Box Tokens:**
   - `<eb></eb>`: Empty box token
   - `<eb1></eb1>`: Empty box variant

**Ví dụ cấu hình:**
```python
dictionary = dict(
    type='TableMasterDictionary',
    dict_file='src/data/structure_vocab.txt',
    with_padding=True,
    with_unknown=True,
    same_start_end=True,
    with_start=True,
    with_end=True
)
```

**Token Mapping Functions:**
- `str2idx(tokens)`: Convert token strings thành indices
- `idx2str(indices)`: Convert indices thành token strings
- `num_classes`: Property trả về tổng số tokens

**Quan hệ với pipeline:**
- Được sử dụng trong [Pad Data](../../datasets/transforms/pad_data/README.md)
- Được sử dụng trong [Decoders](../decoders/README.md)
- Được sử dụng trong [Postprocessors](../postprocessors/README.md)

**Lưu ý đặc biệt:**
- Multi-character tokens khác với character-based dictionaries
- Token ordering trong dict_file affects model performance
- Special tokens cần consistent với model architecture
- Dictionary size affects model complexity và memory usage
