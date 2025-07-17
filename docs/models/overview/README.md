# Overview

### 2.1 Thành phần chính
#### Sơ đồ Model (định dạng đồng bộ với pipeline, mỗi thành phần là subgraph, output tách riêng, tên subgraph là file python)
```mermaid
flowchart TD
    %% Inputs
    subgraph pack_inputs.py
        Inputs["<b>inputs</b>"]
        DataSamples["<b>data_samples</b>"]
    end
    %% Backbone
    subgraph backbones/table_resnet_extra.py
        TableResNetExtra["<b>TableResNetExtra</b>"]
        TableResNetExtra.feature_map{{"feature_map"}}
        TableResNetExtra --> TableResNetExtra.feature_map
    end
    %% Encoder
    subgraph encoders/positional_encoding.py
        PositionalEncoding["<b>PositionalEncoding</b>"]
        PositionalEncoding.encoded_features{{"encoded_features"}}
        PositionalEncoding --> PositionalEncoding.encoded_features
    end
    %% Decoder
    subgraph decoders/table_master_concat_decoder.py
        TableMasterConcatDecoder["<b>TableMasterConcatDecoder</b>"]
        TableMasterConcatDecoder.token_logits{{"token_logits"}}
        TableMasterConcatDecoder.bbox_logits{{"bbox_logits"}}
        TableMasterConcatDecoder --> TableMasterConcatDecoder.token_logits
        TableMasterConcatDecoder --> TableMasterConcatDecoder.bbox_logits
    end
    %% Postprocessor
    subgraph postprocessors/table_master_postprocessor.py
        TableMasterPostprocessor["<b>TableMasterPostprocessor</b>"]
        TableMasterPostprocessor.final_tokens{{"final_tokens"}}
        TableMasterPostprocessor.final_bboxes{{"final_bboxes"}}
        TableMasterPostprocessor --> TableMasterPostprocessor.final_tokens
        TableMasterPostprocessor --> TableMasterPostprocessor.final_bboxes
    end
    %% Kết nối
    Inputs --> TableResNetExtra
