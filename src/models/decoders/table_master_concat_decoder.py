from typing import Tuple
import torch
from mmocr.registry import MODELS
from .table_master_decoder import TableMasterDecoder

@MODELS.register_module()
class TableMasterConcatDecoder(TableMasterDecoder):
    """TableMaster Concat Decoder module.
    This version concatenates the outputs from multiple layers for classification and bbox heads.
    Inherits all logic from TableMasterDecoder except for the decode method.
    """

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