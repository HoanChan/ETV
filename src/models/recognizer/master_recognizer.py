# Copyright (c) Lê Hoàn Chân. All rights reserved.
import torch
import torch.nn as nn
from mmocr.registry import MODELS
from typing import Dict, Optional, List
from mmocr.utils.typing_utils import ConfigType, InitConfigType, OptConfigType
from mmocr.models.textrecog.recognizers.base import BaseRecognizer
from mmocr.structures import TextRecogDataSample

@MODELS.register_module()
class MasterTextRecognizer(BaseRecognizer):
    """Master recognizer for text recognition.
    
    This is a simple text recognizer based on Master decoder.
    
    Args:
        preprocessor (dict, optional): Config dict for preprocessor. Defaults
            to None.
        backbone (dict, optional): Backbone config. Defaults to None.
        encoder (dict, optional): Encoder config. If None, the output from
            backbone will be directly fed into ``decoder``. Defaults to None.
        decoder (dict, optional): Decoder config. Defaults to None.
        data_preprocessor (dict, optional): Model preprocessing config
            for processing the input image data. Defaults to None.
        init_cfg (dict or list[dict], optional): Initialization configs.
    """

    def __init__(self,
                 preprocessor: OptConfigType = None,
                 backbone: OptConfigType = None,
                 encoder: OptConfigType = None,
                 decoder: OptConfigType = None,
                 data_preprocessor: ConfigType = None,
                 init_cfg: InitConfigType = None) -> None:

        super().__init__(
            init_cfg=init_cfg, data_preprocessor=data_preprocessor)

        # Preprocessor module, e.g., TPS
        if preprocessor is not None:
            self.preprocessor = MODELS.build(preprocessor)

        # Backbone
        if backbone is not None:
            self.backbone = MODELS.build(backbone)

        # Encoder module
        if encoder is not None:
            self.encoder = MODELS.build(encoder)

        # Decoder module
        assert decoder is not None
        self.decoder = MODELS.build(decoder)

    def init_weights(self) -> None:
        """Initialize weights."""
        super().init_weights()
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def extract_feat(self, inputs: torch.Tensor) -> torch.Tensor:
        """Directly extract features from the backbone."""
        if self.with_preprocessor:
            inputs = self.preprocessor(inputs)
        if self.with_backbone:
            inputs = self.backbone(inputs)
        return inputs

    def loss(self, inputs: torch.Tensor, data_samples: List[TextRecogDataSample], **kwargs) -> Dict:
        """Calculate losses from a batch of inputs and data samples."""
        feat = self.extract_feat(inputs)
        
        out_enc = None
        if self.with_encoder:
            out_enc = self.encoder(feat)
        
        # Get decoder losses
        decoder_losses = self.decoder.loss(feat, out_enc, data_samples)
        
        return decoder_losses

    def predict(self, inputs: torch.Tensor, data_samples: List[TextRecogDataSample],
                **kwargs) -> List[TextRecogDataSample]:
        """Predict text from a batch of inputs."""
        feat = self.extract_feat(inputs)
        
        out_enc = None
        if self.with_encoder:
            out_enc = self.encoder(feat)
        
        # Get predictions from decoder
        predictions = self.decoder.predict(feat, out_enc, data_samples)
        
        return predictions

    def _forward(self,
                 inputs: torch.Tensor,
                 data_samples: Optional[List[TextRecogDataSample]] = None,
                 **kwargs):
        """Network forward process."""
        feat = self.extract_feat(inputs)
        
        out_enc = None
        if self.with_encoder:
            out_enc = self.encoder(feat)
        
        return self.decoder(feat, out_enc, data_samples, **kwargs)
