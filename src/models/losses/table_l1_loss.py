# Copyright (c) Lê Hoàn Chân. All rights reserved.
from typing import Dict
import torch
import torch.nn as nn
from mmocr.registry import MODELS
from structures.token_recog_data_sample import TokenRecogDataSample

@MODELS.register_module()
class TableL1Loss(nn.Module):
    """Implementation of L1 loss module for table master bbox regression.
    
    This loss function is designed for table structure recognition models
    that predict bounding boxes for table cells and structure elements.
    
    Args:
        reduction (str): Specifies the reduction to apply to the output.
            Should be 'sum' for proper normalization. Defaults to 'sum'.
        lambda_horizon (float): Weight for horizontal bbox loss (x, width).
            Defaults to 1.0.
        lambda_vertical (float): Weight for vertical bbox loss (y, height).
            Defaults to 1.0.
        eps (float): Small epsilon value to avoid division by zero.
            Defaults to 1e-9.
        **kwargs: Other keyword arguments.
    """
    
    def __init__(self,
                 reduction: str = 'sum',
                 lambda_horizon: float = 1.0,
                 lambda_vertical: float = 1.0,
                 eps: float = 1e-9,
                 **kwargs) -> None:
        super().__init__()
        assert isinstance(reduction, str)
        assert reduction == 'sum', 'TableL1Loss should use reduction="sum" for proper normalization.'
        
        self.reduction = reduction
        self.lambda_horizon = lambda_horizon
        self.lambda_vertical = lambda_vertical
        self.eps = eps
        
        # Build L1 loss with sum reduction
        self.l1_loss = nn.L1Loss(reduction=reduction)

    def forward(self, outputs: torch.Tensor, data_samples: list[TokenRecogDataSample], **kwargs) -> Dict[str, torch.Tensor]:
        """
        Compute the L1 loss for table bounding boxes.
        
        Args:
            outputs (torch.Tensor): Predicted bounding boxes with shape (B, L, 4),
                where B is the batch size, L is the sequence length, and 4 represents (x, y, width, height).
            data_samples (list[TokenRecogDataSample]): List of samples containing ground truth information,
                including 'bboxes' and 'masks' fields.
            **kwargs: Additional arguments (not used).
        
        Returns:
            Dict[str, torch.Tensor]:
                - 'loss_horizon_bbox': L1 loss for horizontal coordinates (x, width)
                - 'loss_vertical_bbox': L1 loss for vertical coordinates (y, height)
        """
        # Extract ground truth tokens and bounding boxes from data samples
        bboxes = torch.stack([s.metainfo['padded_bboxes'] for s in data_samples])
        masks = torch.stack([s.metainfo['padded_masks'] for s in data_samples])
        # Extract targets starting from index 1 to align with predictions
        bboxes = bboxes[:, 1:, :].to(outputs.device)  # B x L x 4
        masks = masks[:, 1:].unsqueeze(-1).to(outputs.device)  # B x L x 1
        # Apply masks to filter valid bounding boxes
        masked_outputs = outputs * masks
        masked_targets = bboxes * masks
        # Compute horizontal loss (x and width coordinates: indices 0, 2)
        horizon_sum_loss = self.l1_loss(
            masked_outputs[:, :, 0::2].contiguous(), 
            masked_targets[:, :, 0::2].contiguous()
        )
        horizon_loss = self.lambda_horizon * horizon_sum_loss / (masks.sum() + self.eps)
        # Compute vertical loss (y and height coordinates: indices 1, 3)
        vertical_sum_loss = self.l1_loss(
            masked_outputs[:, :, 1::2].contiguous(), 
            masked_targets[:, :, 1::2].contiguous()
        )
        vertical_loss = self.lambda_vertical * vertical_sum_loss / (masks.sum() + self.eps)
        return {'loss_horizon_bbox': horizon_loss, 'loss_vertical_bbox': vertical_loss}