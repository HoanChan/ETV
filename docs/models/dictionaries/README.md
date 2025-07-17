
# Dictionaries

## 4. TableMasterDictionary

**Chức năng:** Lớp Dictionary để quản lý các table structure tokens và special tokens cho mô hình TableMaster. Xử lý các token nhiều ký tự khác với dictionary dạng ký tự thông thường.

**Đặc điểm:**
- Quản lý các table structure tokens (`<td>`, `<tr>`, `<tbody>`, v.v.)
- Hỗ trợ các special tokens (`<BOS>`, `<EOS>`, `<PAD>`, `<UKN>`)
- Xử lý token nhiều ký tự
- Cấu hình token linh hoạt
- Tương thích với các mô hình sequence-to-sequence

**Input:**
- `dict_file`: Đường dẫn tới file structure alphabet
- Các tham số cấu hình token

**Output:**
- Đối tượng Dictionary với các hàm ánh xạ token:
  - `str2idx()`: Chuyển token thành chỉ số
  - `idx2str()`: Chuyển chỉ số thành token
  - `num_classes`: Tổng số lớp/tokens

**Tham số cấu hình:**
- `dict_file`: Đường dẫn tới file structure alphabet
- `with_start`: Có thêm start token hay không. Mặc định False
- `with_end`: Có thêm end token hay không. Mặc định False
- `same_start_end`: Start và end token có giống nhau không. Mặc định False
- `with_padding`: Có thêm padding token hay không. Mặc định False
- `with_unknown`: Có thêm unknown token hay không. Mặc định False
- `start_token`: Chuỗi start token. Mặc định '<BOS>'
- `end_token`: Chuỗi end token. Mặc định '<EOS>'
- `start_end_token`: Token kết hợp start/end. Mặc định '<BOS/EOS>'
- `padding_token`: Chuỗi padding token. Mặc định '<PAD>'
- `unknown_token`: Chuỗi unknown token. Mặc định '<UKN>'

**Định dạng Structure Alphabet:**
File chứa danh sách các structure token, mỗi token một dòng:
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

**Phân loại Token:**
1. **Table Structure Tokens:**
   - `<table>`, `</table>`: Ranh giới bảng
   - `<tr>`, `</tr>`: Hàng của bảng
   - `<td>`, `</td>`: Ô của bảng
   - `<thead>`, `<tbody>`: Các phần của bảng

2. **Special Tokens:**
   - `<BOS>`: Bắt đầu chuỗi
   - `<EOS>`: Kết thúc chuỗi
   - `<PAD>`: Token padding
   - `<UKN>`: Token không xác định

3. **Empty Box Tokens:**
   - `<eb></eb>`: Token ô trống
   - `<eb1></eb1>`: Biến thể ô trống

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

**Hàm ánh xạ Token:**
- `str2idx(tokens)`: Chuyển chuỗi token thành chỉ số
- `idx2str(indices)`: Chuyển chỉ số thành chuỗi token
- `num_classes`: Thuộc tính trả về tổng số token

**Quan hệ với pipeline:**
- Được sử dụng trong [Pad Data](../../datasets/transforms/pad_data/README.md)
- Được sử dụng trong [Decoders](../decoders/README.md)
- Được sử dụng trong [Postprocessors](../postprocessors/README.md)

**Lưu ý đặc biệt:**
- Token nhiều ký tự khác với dictionary dạng ký tự
- Thứ tự token trong dict_file ảnh hưởng đến hiệu năng mô hình
- Special tokens cần nhất quán với kiến trúc mô hình
- Kích thước dictionary ảnh hưởng đến độ phức tạp và bộ nhớ của mô hình
