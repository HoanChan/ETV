# Model Documentation

## Tổng quan về Architecture TableMaster

TableMaster là một end-to-end model cho table structure recognition, sử dụng dual-head architecture để predict cả table structure tokens và bounding box coordinates.

## Cấu trúc tài liệu

### Architecture Overview
@import "overview/README.md"

### Core Components

#### 1. Backbone Networks
@import "backbones/README.md"

#### 2. Encoders
@import "encoders/README.md"

#### 3. Decoders
@import "decoders/README.md"

#### 4. Dictionaries
@import "dictionaries/README.md"

#### 5. Postprocessors
@import "postprocessors/README.md"

#### 6. Recognizers
@import "recognizer/README.md"

### Training Components

#### 7. Loss Functions
@import "losses/README.md"

#### 8. Evaluation Metrics
@import "metrics/README.md"

#### 9. Neural Network Layers
@import "layers/README.md"

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
