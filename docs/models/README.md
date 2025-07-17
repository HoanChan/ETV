# Model Documentation

## Tổng quan về Architecture TableMaster

TableMaster là một end-to-end model cho table structure recognition, sử dụng dual-head architecture để predict cả table structure tokens và bounding box coordinates.

## Cấu trúc tài liệu

### Architecture Overview
[Xem chi tiết: Overview](overview/README.md)

### Core Components

#### 1. Backbone Networks
[Xem chi tiết: Backbone Networks](backbones/README.md)

#### 2. Encoders
[Xem chi tiết: Encoders](encoders/README.md)

#### 3. Decoders
[Xem chi tiết: Decoders](decoders/README.md)

#### 4. Dictionaries
[Xem chi tiết: Dictionaries](dictionaries/README.md)

#### 5. Postprocessors
[Xem chi tiết: Postprocessors](postprocessors/README.md)

#### 6. Recognizers
[Xem chi tiết: Recognizers](recognizer/README.md)

### Training Components

#### 7. Loss Functions
[Xem chi tiết: Loss Functions](losses/README.md)

#### 8. Evaluation Metrics
[Xem chi tiết: Evaluation Metrics](metrics/README.md)

#### 9. Neural Network Layers
[Xem chi tiết: Neural Network Layers](layers/README.md)

## Kiến trúc tổng thể

Model TableMaster bao gồm các thành phần chính:

1. **Feature Extraction:** TableResNetExtra backbone với Global Context Blocks
2. **Sequence Encoding:** PositionalEncoding để thêm spatial information
3. **Dual-Head Decoding:** TableMasterConcatDecoder cho token và bbox prediction
4. **Token Management:** TableMasterDictionary cho structure token mapping
5. **Output Processing:** TableMasterPostprocessor để convert raw predictions

## Training và Inference

### Training Process
- Multi-task learning với classification và regression objectives
- MasterTFLoss cho token prediction
- TableL1Loss cho bbox regression
- Gradient balancing giữa hai heads

### Inference Process
- Forward pass qua toàn bộ pipeline
- Postprocessing để tạo meaningful outputs
- Confidence thresholding và coordinate denormalization

## Evaluation

Model được evaluate bằng:
- TEDS (Tree Edit Distance based Similarity) metric
- Token accuracy và bbox IoU
- Structure consistency checking

## Lưu ý quan trọng

- **Dual-head design** essential cho table recognition task
- **Multi-scale features** important cho spatial understanding
- **Attention mechanism** captures table structure relationships
- **Dictionary design** affects model vocabulary và performance
- **Postprocessing** critical cho practical applications
