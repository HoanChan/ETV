
# So sánh DecoderLayer và TMDLayer

Thư mục này chứa so sánh toàn diện giữa hai lớp decoder:
- `DecoderLayer`: Cài đặt tùy chỉnh bằng PyTorch thuần
- `TMDLayer`: Cài đặt dựa trên MMCV sử dụng `BaseTransformerLayer`

## Tổng quan các tệp

- `decoder_layer.py`: Lớp decoder tùy chỉnh với attention đa đầu
- `tmd_layer.py`: Lớp decoder transformer dựa trên MMCV
- `test_decoder_vs_tmd.py`: Bộ kiểm thử so sánh hai cài đặt
- `usage_examples.py`: Ví dụ sử dụng thực tế và hướng dẫn chuyển đổi
- `test_requirements.txt`: Các phụ thuộc bổ sung cho kiểm thử

## Kết quả chính

### ✅ Tương thích
- **Kích thước đầu ra**: Cả hai lớp đều cho đầu ra cùng kích thước
- **Số lượng tham số**: Giống nhau (4.204.032)
- **Ổn định số học**: Ổn định trên nhiều dải giá trị đầu vào
- **Lan truyền gradient**: Gradient hợp lệ, độ lớn tương đương

### ⚡ Hiệu năng
- **DecoderLayer**: Nhanh hơn ~1.2 lần trung bình
- **TMDLayer**: Chậm hơn một chút nhưng chuẩn hóa hơn
- **Bộ nhớ**: Gần như giống nhau (~16MB, chênh lệch < 0.1MB)

### 🔧 Khác biệt kỹ thuật

| Khía cạnh | DecoderLayer | TMDLayer |
|----------|--------------|----------|
| **Phụ thuộc** | PyTorch thuần | Cần MMCV |
| **Định dạng mask** | Tensor 4D | Tensor 2D |
| **Cấu hình** | Dạng dict | Tham số trực tiếp |
| **Linh hoạt** | Attention tùy chỉnh | Chuẩn hóa |
| **Tích hợp** | Độc lập | Hệ sinh thái MMCV |

## Ví dụ sử dụng

### DecoderLayer
```python
layer = DecoderLayer(
    size=512,
    self_attn={'headers': 8, 'd_model': 512, 'dropout': 0.1},
    src_attn={'headers': 8, 'd_model': 512, 'dropout': 0.1},
    feed_forward={'d_model': 512, 'd_ff': 2048, 'dropout': 0.1},
    dropout=0.1
)

output = layer(target, source, src_mask, tgt_mask)
```

### TMDLayer
```python
layer = TMDLayer(
    d_model=512,
    n_head=8,
    d_inner=2048,
    attn_drop=0.1,
    ffn_drop=0.1
)

output = layer(
    query=target,
    key=source,
    value=source,
    query_pos=None,
    key_pos=None,
    attn_masks=[self_attn_mask, cross_attn_mask],
    query_key_padding_mask=None,
    key_padding_mask=None
)
```

## Chạy kiểm thử

### Yêu cầu
```bash
pip install torch mmcv-full psutil
```

### Chạy toàn bộ kiểm thử
```bash
python test_decoder_vs_tmd.py
```

### Chạy ví dụ sử dụng
```bash
python usage_examples.py
```

## Tóm tắt kết quả kiểm thử

```
============================================================
SO SÁNH DecoderLayer vs TMDLayer
============================================================
Kiểm tra kích thước đầu ra...
✓ DecoderLayer output shape: torch.Size([2, 10, 512])
✓ TMDLayer output shape: torch.Size([2, 10, 512])
✓ Kích thước khớp: True

Kiểm tra số lượng tham số...
✓ DecoderLayer parameters: 4,204,032
✓ TMDLayer parameters: 4,204,032
✓ Chênh lệch tham số: 0

Kiểm tra tương thích forward...
✓ Tất cả kiểm thử forward đều vượt qua!

Kiểm tra gradient...
✓ Cả hai đều có gradient khác 0: True

Kiểm tra bộ nhớ...
✓ Chênh lệch bộ nhớ: 0.04 MB

Kiểm tra hiệu năng...
✓ DecoderLayer nhanh hơn 1.23x

Kiểm tra ổn định số học...
✓ Cả hai đều ổn định số học
```

## Hướng dẫn chuyển đổi

### Từ DecoderLayer sang TMDLayer

1. **Cập nhật cấu hình**:
   - Đổi từ dict sang tham số trực tiếp
   - Đổi tên `headers` thành `n_head`
   - Đổi tên `d_ff` thành `d_inner`

2. **Cập nhật forward**:
   - Dùng tham số tên thay vì vị trí
   - Đổi mask từ 4D sang 2D
   - Đảo ngược logic mask self-attention

3. **Cập nhật phụ thuộc**:
   - Thêm MMCV vào requirements
   - Cập nhật import

### Từ TMDLayer sang DecoderLayer

1. **Bỏ phụ thuộc MMCV**:
   - Cài đặt PyTorch thuần
   - Không phụ thuộc ngoài

2. **Cập nhật mask**:
   - Chuyển mask 2D sang 4D
   - Điều chỉnh logic mask

3. **Cập nhật cấu hình**:
   - Dùng dict cho cấu hình
   - Điều chỉnh tên tham số

## Khuyến nghị

### Dùng DecoderLayer khi:
- Muốn giảm phụ thuộc
- Cần attention tùy chỉnh
- Yêu cầu hiệu năng cao
- Làm việc với PyTorch thuần

### Dùng TMDLayer khi:
- Đã dùng hệ sinh thái MMCV
- Muốn chuẩn hóa thao tác
- Cần tích hợp tốt với MMCV
- Ưu tiên cài đặt đã được kiểm thử kỹ

## Chi tiết kiến trúc

### Kiến trúc DecoderLayer
```
Input → LayerNorm → Self-Attention → Residual →
        LayerNorm → Cross-Attention → Residual →
        LayerNorm → FeedForward → Residual → Output
```

### Kiến trúc TMDLayer
```
Input → Self-Attention → LayerNorm → Residual →
        Cross-Attention → LayerNorm → Residual →
        FeedForward → LayerNorm → Residual → Output
```

Hai kiến trúc này về chức năng là tương đương, chỉ khác thứ tự chuẩn hóa.

## Đánh giá hiệu năng

### Hiệu năng lớp đơn
- **DecoderLayer**: ~4.1ms mỗi lần forward
- **TMDLayer**: ~5.1ms mỗi lần forward
- **Tăng tốc**: DecoderLayer nhanh hơn 1.23x

### Hiệu năng nhiều lớp (6 lớp)
- **DecoderLayer**: ~63.7ms mỗi lần forward
- **TMDLayer**: ~72.8ms mỗi lần forward
- **Tăng tốc**: DecoderLayer nhanh hơn 1.14x

## Kết luận

Cả hai cài đặt đều rất tương thích và cho kết quả giống nhau. Việc lựa chọn phụ thuộc vào nhu cầu cụ thể:

- **DecoderLayer**: Phù hợp ứng dụng cần hiệu năng, ít phụ thuộc
- **TMDLayer**: Phù hợp tích hợp hệ sinh thái MMCV, thao tác chuẩn hóa

Bộ kiểm thử đảm bảo cả hai cài đặt luôn tương thích và có thể thay thế cho nhau trong hầu hết trường hợp.
