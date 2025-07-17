# Decoders

#### 3. TableMasterConcatDecoder (Decoder)
**Chức năng:** Dự đoán chuỗi token cấu trúc bảng và vị trí các bounding box cho từng ô.
**Kiến trúc chi tiết:**
- **Embedding Layer:** Nhận chuỗi target token (tgt_seq), chuyển thành embedding vector.
- **Positional Encoding:** Bổ sung thông tin vị trí cho embedding.
- **Shared Decoder Layers:**
    - Gồm N-1 layer (mặc định 2 nếu n_layers=3), mỗi layer gồm các block:
        - Self-Attention: Tương tác giữa các token trong chuỗi.
        - Source-Attention: Tương tác giữa token và feature map từ encoder.
        - Feed-Forward: Biến đổi phi tuyến.
    - Output: tensor x (shape [N, T, d_model])
- **Phân nhánh cuối:**
    - **Classification Branch:**
        - 1 layer decoder riêng cho classification (cls_layer).
        - Có thể dùng nhiều layer, kết quả các layer sẽ được concat lại (ở bản concat).
        - Sau khi concat, áp dụng LayerNorm và Linear để ra `token_logits` (shape [N, T, num_classes]).
    - **BBox Branch:**
        - 1 layer decoder riêng cho bbox (bbox_layer).
        - Tương tự, concat các layer, LayerNorm, Linear + Sigmoid để ra `bbox_logits` (shape [N, T, 4]).
- **Loss Head:**
    - `tokens_loss`: MasterTFLoss (CrossEntropy, ignore_index, flatten...)
    - `bboxes_loss`: TableL1Loss (L1, mask, lambda_horizon, lambda_vertical...)
- **Các phương thức chính:**
    - `decode`: Xử lý forward qua các layer, phân nhánh, concat, heads.
    - `forward_train`, `forward_test`: Xây dựng input, mask, gọi decode.
    - `predict`: Dự đoán và hậu xử lý.
    - `loss`: Tính toán loss từ output và ground truth.

**Input:**
- `encoded_features` (torch.Tensor): Đặc trưng đã mã hóa vị trí.
- `tgt_seq` (torch.Tensor): Chuỗi token target (huấn luyện).
- `data_samples` (List[TableMasterDataSample]): Ground truth, meta info.
**Output:**
- `token_logits` (torch.Tensor): Dự đoán nhãn cho từng token.
- `bbox_logits` (torch.Tensor): Dự đoán vị trí bounding box cho từng ô.
- `loss` (dict): Giá trị mất mát cho token, bbox.
