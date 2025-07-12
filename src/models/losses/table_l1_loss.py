
# Copyright (c) Lê Hoàn Chân. All rights reserved.
from typing import Dict, Optional

import torch
import torch.nn as nn

from mmocr.registry import MODELS


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
        assert reduction == 'sum', (
            'TableL1Loss should use reduction="sum" for proper normalization.')
        
        self.reduction = reduction
        self.lambda_horizon = lambda_horizon
        self.lambda_vertical = lambda_vertical
        self.eps = eps
        
        # Build L1 loss with sum reduction
        self.l1_loss = nn.L1Loss(reduction=reduction)

    def _format_inputs(self, 
                      outputs: torch.Tensor, 
                      targets_dict: Dict[str, torch.Tensor]) -> tuple:
        """Format inputs for loss computation.
        
        Args:
            outputs (torch.Tensor): Predicted bounding boxes with shape (B, L, 4).
            targets_dict (Dict[str, torch.Tensor]): Dictionary containing target
                information including 'padded_bbox' and 'padded_masks'.
                
        Returns:
            tuple: Formatted (masked_outputs, masked_targets, masks).
        """
        # Extract targets starting from index 1 to align with predictions
        # bboxes = targets_dict['padded_bboxes'][:, 1:, :].to(outputs.device)  # B x L x 4
        # masks = targets_dict['padded_masks'][:, 1:].unsqueeze(-1).to(outputs.device)  # B x L x 1
        
        bboxes = targets_dict['padded_bboxes'].to(outputs.device)  # B x L x 4
        masks = targets_dict['padded_masks'].unsqueeze(-1).to(outputs.device)  # B x L x 1

        # Apply masks to filter valid bounding boxes
        masked_outputs = outputs * masks
        masked_targets = bboxes * masks
        
        return masked_outputs, masked_targets, masks

    def forward(self, 
                outputs: torch.Tensor, 
                targets_dict: Dict[str, torch.Tensor], 
                img_metas: Optional[list] = None) -> Dict[str, torch.Tensor]:
        """Forward function to compute L1 loss for table bounding boxes.
        
        Args:
            outputs (torch.Tensor): Predicted bounding boxes with shape (B, L, 4)
                where B is batch size, L is sequence length, and 4 represents
                (x, y, width, height).
            targets_dict (Dict[str, torch.Tensor]): Dictionary containing target
                information with keys 'padded_bboxes' and 'padded_masks'.
            img_metas (Optional[list]): Image meta information. Defaults to None.
                
        Returns:
            Dict[str, torch.Tensor]: Dictionary containing computed losses:
                - 'horizon_bbox_loss': L1 loss for horizontal coordinates (x, width)
                - 'vertical_bbox_loss': L1 loss for vertical coordinates (y, height)
        """
        # Format inputs
        masked_outputs, masked_targets, masks = self._format_inputs(outputs, targets_dict)
        
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

        return [horizon_loss, vertical_loss]