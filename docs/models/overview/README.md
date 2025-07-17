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

### Component Details

#### 1. Backbone (TableResNetExtra)
- **Chức năng:** Trích xuất multi-scale features từ table images
- **Input:** Table images (N, C, H, W)
- **Output:** Multi-scale feature maps
- **Đặc điểm:** ResNet-based với Global Context Blocks

#### 2. Encoder (PositionalEncoding)
- **Chức năng:** Thêm positional information vào spatial features
- **Input:** 2D feature maps
- **Output:** 1D sequence với positional encoding
- **Đặc điểm:** Sinusoidal positional encoding

#### 3. Decoder (TableMasterConcatDecoder)
- **Chức năng:** Dual-head prediction cho tokens và bboxes
- **Input:** Encoded features + target sequences
- **Output:** Classification logits + bbox coordinates
- **Đặc điểm:** Transformer-based với concatenation strategy

#### 4. Dictionary (TableMasterDictionary)
- **Chức năng:** Token mapping cho table structure elements
- **Features:** Multi-character tokens, special tokens
- **Tokens:** `<table>`, `<tr>`, `<td>`, `<eb></eb>`, etc.

#### 5. Postprocessor (TableMasterPostprocessor)
- **Chức năng:** Convert raw outputs thành meaningful predictions
- **Input:** Logits + bbox coordinates
- **Output:** Token strings + denormalized bboxes
- **Đặc điểm:** Confidence thresholding, coordinate denormalization

### Training Process

1. **Data Flow:**
   - Images → Backbone → Encoder → Decoder
   - Ground truth tokens/bboxes → Loss calculation

2. **Loss Functions:**
   - **MasterTFLoss:** Cross-entropy cho token classification
   - **TableL1Loss:** L1 loss cho bbox regression

3. **Optimization:**
   - Multi-task learning với token + bbox objectives
   - Gradient balancing giữa classification và regression

### Inference Process

1. **Forward Pass:**
   - Image → Features → Encoded features → Predictions

2. **Postprocessing:**
   - Logits → Token strings
   - Bbox coordinates → Denormalized bounding boxes

3. **Output:**
   - Structured table representation
   - Spatial information for cells

### Quan hệ với Dataset Pipeline

Model nhận input từ dataset pipeline:
- [Pack Inputs](../../datasets/transforms/pack_inputs/README.md)
- TableMasterDataSample format
- Normalized images và ground truth labels

### Evaluation

Model được evaluate bằng:
- [Metrics](metrics/README.md)
- TEDS (Tree Edit Distance based Similarity)
- Token accuracy, bbox IoU, structure consistency

### Lưu ý đặc biệt

- **Dual-head architecture** essential cho table recognition
- **Multi-scale features** important cho spatial understanding
- **Attention mechanism** captures table structure relationships
- **Postprocessing** critical cho meaningful outputs
- **Dictionary design** affects model vocabulary và performance
