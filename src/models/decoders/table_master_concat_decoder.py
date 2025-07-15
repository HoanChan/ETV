from typing import Dict, Optional, Sequence, Tuple, Union
import torch
import torch.nn as nn
from mmocr.registry import MODELS
from mmocr.models.common.dictionary import Dictionary
from .table_master_decoder import TableMasterDecoder

@MODELS.register_module()
class TableMasterConcatDecoder(TableMasterDecoder):
    """TableMaster Concat Decoder module.
    This version concatenates the outputs from multiple layers for classification and bbox heads.
    Inherits all logic from TableMasterDecoder except for the decode method and fully connected layers.
    """

    def __init__(self, d_model: int, **kwargs):
        # Initialize parent class first
        super().__init__(d_model=d_model, **kwargs)

        # Override classification and bbox heads for concatenation
        # For concat version, we concatenate one layer output (so concat_dim = d_model * 1 = d_model)
        # If you have multiple layers to concatenate, adjust accordingly
        concat_dim = d_model
        
        # Classification head (adjusted for concatenation)
        self.cls_fc = nn.Linear(concat_dim, self.dictionary.num_classes)
        
        # Bbox regression head (adjusted for concatenation)
        self.bbox_fc = nn.Sequential(nn.Linear(concat_dim, 4), nn.Sigmoid())

    def decode(self, tgt_seq: torch.Tensor, feature: torch.Tensor,
               src_mask: torch.BoolTensor,
               tgt_mask: torch.BoolTensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Decode the input sequence with concatenation strategy (override).

        Args:
            tgt_seq (Tensor): Target sequence of shape: (N, T, C).
            feature (Tensor): Input feature map from encoder of
                shape: (N, C, H, W)
            src_mask (BoolTensor): The source mask of shape: (N, H*W).
            tgt_mask (BoolTensor): The target mask of shape: (N, T, T).

        Returns:
            Tuple[Tensor, Tensor]: The decoded classification and bbox outputs.
        """
        # Main process of transformer decoder
        x = self.embedding(tgt_seq)
        x = self.positional_encoding(x)
        
        # Shared decoder layers (N-1 layers)
        for layer in self.decoder_layers:
            x = layer(x, feature, src_mask, tgt_mask)
        
        # Classification head with concatenation
        cls_x_list = []
        cls_x = x
        for layer in self.cls_layer:
            cls_x = layer(cls_x, feature, src_mask, tgt_mask)
            cls_x_list.append(cls_x)
        cls_x = torch.cat(cls_x_list, dim=-1)
        cls_x = self.norm(cls_x)
        cls_output = self.cls_fc(cls_x)
        
        # Bbox head with concatenation
        bbox_x_list = []
        bbox_x = x
        for layer in self.bbox_layer:
            bbox_x = layer(bbox_x, feature, src_mask, tgt_mask)
            bbox_x_list.append(bbox_x)
        bbox_x = torch.cat(bbox_x_list, dim=-1)
        bbox_x = self.norm(bbox_x)
        bbox_output = self.bbox_fc(bbox_x)
        
        return cls_output, bbox_output