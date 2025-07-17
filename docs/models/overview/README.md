# Model Overview

## 2.1 Kiến trúc TableMaster

### Tổng quan
TableMaster là một end-to-end model cho table structure recognition, sử dụng dual-head architecture để predict cả table structure tokens và bounding box coordinates. Model được thiết kế để xử lý images của bảng và tạo ra structured output bao gồm HTML-like tokens và spatial information.

### Thành phần chính

#### Sơ đồ Model Architecture
```mermaid
flowchart TD
    %% Input
    subgraph pack_inputs.py
        Inputs["<b>inputs</b><br/>ảnh tensor (N,C,H,W)"]
        DataSamples["<b>data_samples</b><br/>TableMasterDataSample"]
    end
    
    %% Backbone
    subgraph backbones/table_resnet_extra.py
        TableResNetExtra["<b>TableResNetExtra</b><br/>Feature Extraction"]
        TableResNetExtra.feature_map{{"feature_map<br/>List[Tensor]"}}
        TableResNetExtra --> TableResNetExtra.feature_map
    end
    
    %% Encoder
    subgraph encoders/positional_encoding.py
        PositionalEncoding["<b>PositionalEncoding</b><br/>Sequence Encoding"]
        PositionalEncoding.encoded_features{{"encoded_features<br/>(N,H*W,C)"}}
        PositionalEncoding --> PositionalEncoding.encoded_features
    end
    
    %% Decoder
    subgraph decoders/table_master_concat_decoder.py
        TableMasterConcatDecoder["<b>TableMasterConcatDecoder</b><br/>Dual-Head Decoder"]
        TableMasterConcatDecoder.token_logits{{"token_logits<br/>Classification"}}
        TableMasterConcatDecoder.bbox_logits{{"bbox_logits<br/>Regression"}}
        TableMasterConcatDecoder --> TableMasterConcatDecoder.token_logits
        TableMasterConcatDecoder --> TableMasterConcatDecoder.bbox_logits
    end
    
    %% Postprocessor
    subgraph postprocessors/table_master_postprocessor.py
        TableMasterPostprocessor["<b>TableMasterPostprocessor</b><br/>Output Processing"]
        TableMasterPostprocessor.final_tokens{{"final_tokens<br/>List[str]"}}
        TableMasterPostprocessor.final_bboxes{{"final_bboxes<br/>List[np.ndarray]"}}
        TableMasterPostprocessor --> TableMasterPostprocessor.final_tokens
        TableMasterPostprocessor --> TableMasterPostprocessor.final_bboxes
    end
    
    %% Dictionary
    subgraph dictionaries/table_master_dictionary.py
        TableMasterDictionary["<b>TableMasterDictionary</b><br/>Token Mapping"]
        TableMasterDictionary.mapping{{"str2idx/idx2str<br/>Token Conversion"}}
        TableMasterDictionary --> TableMasterDictionary.mapping
    end
    
    %% Loss Functions
    subgraph losses/
        MasterTFLoss["<b>MasterTFLoss</b><br/>Classification Loss"]
        TableL1Loss["<b>TableL1Loss</b><br/>Bbox Regression Loss"]
    end
    
    %% Connections
    Inputs --> TableResNetExtra
    TableResNetExtra.feature_map --> PositionalEncoding
    PositionalEncoding.encoded_features --> TableMasterConcatDecoder
    DataSamples --> TableMasterConcatDecoder
    
    TableMasterConcatDecoder.token_logits --> TableMasterPostprocessor
    TableMasterConcatDecoder.bbox_logits --> TableMasterPostprocessor
    DataSamples --> TableMasterPostprocessor
    
    TableMasterDictionary.mapping --> TableMasterConcatDecoder
    TableMasterDictionary.mapping --> TableMasterPostprocessor
    
    TableMasterConcatDecoder.token_logits --> MasterTFLoss
    TableMasterConcatDecoder.bbox_logits --> TableL1Loss
    
    %% Styling
    style Inputs fill:#e1f5fe,stroke:#0277bd,stroke-width:2px
    style TableResNetExtra.feature_map fill:#e8f5e8,stroke:#2e7d32,stroke-width:2px
    style PositionalEncoding.encoded_features fill:#e8f5e8,stroke:#2e7d32,stroke-width:2px
    style TableMasterConcatDecoder.token_logits fill:#fff3e0,stroke:#f57c00,stroke-width:2px
    style TableMasterConcatDecoder.bbox_logits fill:#fff3e0,stroke:#f57c00,stroke-width:2px
    style TableMasterPostprocessor.final_tokens fill:#f3e5f5,stroke:#7b1fa2,stroke-width:2px
    style TableMasterPostprocessor.final_bboxes fill:#f3e5f5,stroke:#7b1fa2,stroke-width:2px
```


### Chi tiết các thành phần

#### 1. Backbone (TableResNetExtra)
- **Chức năng:** Trích xuất đặc trưng đa tỷ lệ từ ảnh bảng
- **Input:** Ảnh bảng (N, C, H, W)
- **Output:** Feature maps đa tỷ lệ
- **Đặc điểm:** Dựa trên ResNet, tích hợp Global Context Blocks

#### 2. Encoder (PositionalEncoding)
- **Chức năng:** Thêm thông tin vị trí vào đặc trưng không gian
- **Input:** Feature maps 2D
- **Output:** Chuỗi 1D với positional encoding
- **Đặc điểm:** Sinusoidal positional encoding

#### 3. Decoder (TableMasterConcatDecoder)
- **Chức năng:** Dự đoán hai đầu cho tokens và bboxes
- **Input:** Đặc trưng đã mã hóa + chuỗi mục tiêu
- **Output:** Logits phân loại + tọa độ bbox
- **Đặc điểm:** Dựa trên Transformer với chiến lược nối đặc trưng

#### 4. Dictionary (TableMasterDictionary)
- **Chức năng:** Ánh xạ token cho các phần tử cấu trúc bảng
- **Đặc điểm:** Token nhiều ký tự, special tokens
- **Tokens:** `<table>`, `<tr>`, `<td>`, `<eb></eb>`, v.v.

#### 5. Postprocessor (TableMasterPostprocessor)
- **Chức năng:** Chuyển đổi output thô thành dự đoán có ý nghĩa
- **Input:** Logits + tọa độ bbox
- **Output:** Chuỗi token + bbox đã khử chuẩn hóa
- **Đặc điểm:** Ngưỡng độ tin cậy, khử chuẩn hóa tọa độ

### Quy trình huấn luyện

1. **Luồng dữ liệu:**
   - Ảnh → Backbone → Encoder → Decoder
   - Ground truth tokens/bboxes → Tính loss

2. **Hàm loss:**
   - **MasterTFLoss:** Cross-entropy cho phân loại token
   - **TableL1Loss:** L1 loss cho hồi quy bbox

3. **Tối ưu hóa:**
   - Học đa nhiệm với mục tiêu token + bbox
   - Cân bằng gradient giữa phân loại và hồi quy

### Quy trình suy luận

1. **Forward Pass:**
   - Ảnh → Đặc trưng → Đặc trưng đã mã hóa → Dự đoán

2. **Hậu xử lý:**
   - Logits → Chuỗi token
   - Tọa độ bbox → Bbox đã khử chuẩn hóa

3. **Output:**
   - Biểu diễn bảng có cấu trúc
   - Thông tin không gian cho các ô

### Quan hệ với Dataset Pipeline

Model nhận input từ pipeline dữ liệu:
- [Pack Inputs](../../datasets/transforms/pack_inputs/README.md)
- Định dạng TableMasterDataSample
- Ảnh đã chuẩn hóa và ground truth labels

### Đánh giá

Model được đánh giá bằng:
- [Metrics](metrics/README.md)
- TEDS (Tree Edit Distance based Similarity)
- Độ chính xác token, bbox IoU, tính nhất quán cấu trúc

### Lưu ý đặc biệt

- **Kiến trúc dual-head** rất quan trọng cho nhận diện bảng
- **Đặc trưng đa tỷ lệ** quan trọng cho hiểu không gian
- **Attention mechanism** giúp mô hình hóa quan hệ cấu trúc bảng
- **Hậu xử lý** quyết định output có ý nghĩa
- **Thiết kế dictionary** ảnh hưởng đến từ vựng và hiệu năng mô hình
