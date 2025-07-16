# Copyright (c) Lê Hoàn Chân. All rights reserved.
import torch
import torch.nn as nn
from mmocr.registry import MODELS
from typing import Dict, Optional, Tuple, Union
from mmocr.utils.typing_utils import ConfigType, InitConfigType, OptConfigType
from mmocr.models.textrecog.recognizers.base import BaseRecognizer
from traitlets import List
from structures.table_master_data_sample import TableMasterDataSample

@MODELS.register_module()
class TableMaster(BaseRecognizer):
    """TableMaster recognizer for table structure recognition.

    Args:
        preprocessor (dict, optional): Config dict for preprocessor. Defaults
            to None.
        backbone (dict, optional): Backbone config. Defaults to None.
        encoder (dict, optional): Encoder config. If None, the output from
            backbone will be directly fed into ``decoder``. Defaults to None.
        decoder (dict, optional): Decoder config. Defaults to None.
        bbox_loss (dict, optional): Config for bbox loss. Defaults to None.
        data_preprocessor (dict, optional): Model preprocessing config
            for processing the input image data. Keys allowed are
            ``to_rgb``(bool), ``pad_size_divisor``(int), ``pad_value``(int or
            float), ``mean``(int or float) and ``std``(int or float).
            Preprcessing order: 1. to rgb; 2. normalization 3. pad.
            Defaults to None.
        init_cfg (dict or list[dict], optional): Initialization configs.
            Defaults to None.
    """

    def __init__(self,
                 preprocessor: OptConfigType = None,
                 backbone: OptConfigType = None,
                 encoder: OptConfigType = None,
                 decoder: OptConfigType = None,
                 bbox_loss: OptConfigType = None,
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

        # BBox loss module
        if bbox_loss is not None:
            self.bbox_loss = MODELS.build(bbox_loss)

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

    def loss(self, inputs: torch.Tensor, data_samples: List[TableMasterDataSample], **kwargs) -> Dict:
        """Calculate losses from a batch of inputs and data samples.
        
        Args:
            inputs (tensor): Input images of shape (N, C, H, W).
                Typically these should be mean centered and std scaled.
            data_samples (list[TableMasterDataSample]): A list of N
                datasamples, containing meta information and gold
                annotations for each of the images.

        Returns:
            dict[str, tensor]: A dictionary of loss components.
        """
        feat = self.extract_feat(inputs)
        if isinstance(feat, (list, tuple)):
            feat = feat[-1]

        out_enc = None
        if self.with_encoder:
            out_enc = self.encoder(feat, data_samples)

        # Get decoder losses
        decoder_losses = self.decoder.loss(feat, out_enc, data_samples)

        # Get bbox losses if bbox_loss is defined
        if hasattr(self, 'bbox_loss') and self.bbox_loss is not None:
            # Get bbox predictions from decoder
            bbox_predictions = self.decoder.predict_bbox(feat, out_enc, data_samples)
            bbox_losses = self.bbox_loss(bbox_predictions, data_samples)
            decoder_losses.update(bbox_losses)

        return decoder_losses

    def predict(self, inputs: torch.Tensor, data_samples: List[TableMasterDataSample],
                **kwargs) -> List[TableMasterDataSample]:
        """Predict results from a batch of inputs and data samples with post-
        processing.

        Args:
            inputs (torch.Tensor): Image input tensor.
            data_samples (list[TableMasterDataSample]): A list of N datasamples,
                containing meta information and gold annotations for each of
                the images.

        Returns:
            list[TableMasterDataSample]:  A list of N datasamples of prediction
            results. Results are stored in ``pred_instances``.
        """
        feat = self.extract_feat(inputs)
        if isinstance(feat, (list, tuple)):
            feat = feat[-1]

        out_enc = None
        if self.with_encoder:
            out_enc = self.encoder(feat, data_samples)

        # Get predictions from decoder
        predictions = self.decoder.predict(feat, out_enc, data_samples)

        # Process predictions and add bbox results
        results = []
        for i, (pred_sample, data_sample) in enumerate(zip(predictions, data_samples)):
            # Extract token and score from prediction tokens (recognition head)
            token = ""
            score = 0.0
            if hasattr(pred_sample, 'pred_tokens'):
                if hasattr(pred_sample.pred_tokens, 'item'):
                    token = pred_sample.pred_tokens.item
                score = getattr(pred_sample.pred_tokens, 'scores', 0.0)
                if hasattr(score, 'item'):
                    score = score.item()
            
            # Extract bbox from prediction instances (detection head)
            bbox = None
            if hasattr(pred_sample, 'pred_instances') and hasattr(pred_sample.pred_instances, 'bboxes'):
                bbox = pred_sample.pred_instances.bboxes
                if bbox is not None:
                    bbox = bbox.cpu().numpy() if hasattr(bbox, 'cpu') else bbox

            result = dict(token=token, score=score, bbox=bbox)
            results.append(result)

        # Optional: visualize predicted bboxes
        # visual_pred_bboxes(data_samples, results)

        return predictions

    def _forward(self,
                 inputs: torch.Tensor,
                 data_samples: Optional[List[TableMasterDataSample]] = None,
                 **kwargs) -> Union[Dict[str, torch.Tensor], List[TableMasterDataSample], Tuple[torch.Tensor], torch.Tensor]:
        """Network forward process. Usually includes backbone, encoder and
        decoder forward without any post-processing.

         Args:
            inputs (Tensor): Inputs with shape (N, C, H, W).
            data_samples (list[TableMasterDataSample]): A list of N
                datasamples, containing meta information and gold
                annotations for each of the images.

        Returns:
            Tensor: A tuple of features from ``decoder`` forward.
        """
        feat = self.extract_feat(inputs)
        if isinstance(feat, (list, tuple)):
            feat = feat[-1]

        out_enc = None
        if self.with_encoder:
            out_enc = self.encoder(feat, data_samples)
        return self.decoder(feat, out_enc, data_samples)