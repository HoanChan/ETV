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

    def __init__(
        self,
        n_layers: int = 3,
        n_head: int = 8,
        d_model: int = 512,
        decoder: Optional[Dict] = None,
        module_loss: Optional[Dict] = None,
        postprocessor: Optional[Dict] = None,
        dictionary: Optional[Union[Dict, Dictionary]] = None,
        max_seq_len: int = 30,
        init_cfg: Optional[Union[Dict, Sequence[Dict]]] = None,
    ):
        # Initialize parent class first
        super().__init__(
            n_layers=n_layers,
            n_head=n_head,
            d_model=d_model,
            decoder=decoder,
            module_loss=module_loss,
            postprocessor=postprocessor,
            dictionary=dictionary,
            max_seq_len=max_seq_len,
            init_cfg=init_cfg,
        )
        
        # Override classification and bbox heads for concatenation
        # For concat version, we concatenate one layer output (so concat_dim = d_model * 1 = d_model)
        # If you have multiple layers to concatenate, adjust accordingly
        concat_dim = d_model
        
        # Classification head (adjusted for concatenation)
        self.cls_fc = nn.Linear(concat_dim, self.dictionary.num_classes)
        
        # Bbox regression head (adjusted for concatenation)
        self.bbox_fc = nn.Sequential(
            nn.Linear(concat_dim, 4),
            nn.Sigmoid()
        )

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
        tgt_seq = self.embedding(tgt_seq)
        x = self.positional_encoding(tgt_seq)
        attn_masks = [tgt_mask, src_mask]
        # Shared decoder layers
        for layer in self.decoder_layers:
            x = layer(
                query=x, key=feature, value=feature, attn_masks=attn_masks)
        # Classification head with concatenation
        cls_x_list = []
        cls_x = x
        for layer in self.cls_layer:
            cls_x = layer(
                query=cls_x, key=feature, value=feature, attn_masks=attn_masks)
            cls_x_list.append(cls_x)
        cls_x = torch.cat(cls_x_list, dim=-1)
        cls_x = self.norm(cls_x)
        cls_output = self.cls_fc(cls_x)
        # Bbox head with concatenation
        bbox_x_list = []
        bbox_x = x
        for layer in self.bbox_layer:
            bbox_x = layer(
                query=bbox_x, key=feature, value=feature, attn_masks=attn_masks)
            bbox_x_list.append(bbox_x)
        bbox_x = torch.cat(bbox_x_list, dim=-1)
        bbox_x = self.norm(bbox_x)
        bbox_output = self.bbox_fc(bbox_x)
        return cls_output, bbox_output