# Copyright (c) Lê Hoàn Chân. All rights reserved.
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
from torch.nn.modules.linear import Linear
from torch.nn.modules.container import Sequential
from mmocr.registry import MODELS


@MODELS.register_module()
class TableLoss(nn.Module):
    """Implementation of composite loss for table structure recognition.
    
    This loss combines text recognition loss and bounding box regression loss
    for table structure models.
    
    Args:
        loss_token (dict): Config for text loss module.
        loss_bbox (dict): Config for bbox loss module.
    """
    def __init__(self, loss_token, loss_bbox):
        super().__init__()
        self.loss_token = MODELS.build(loss_token)
        self.loss_bbox = MODELS.build(loss_bbox)

    def forward(self, 
                outputs: tuple[torch.Tensor, torch.Tensor], 
                data_samples: list, 
                **kwargs) -> Dict[str, torch.Tensor]:
        """Forward function to compute composite loss.
        
        Args:
            outputs (tuple): Model outputs, each element is a tensor.
            data_samples (list): List of ground truth samples, each with attributes
                'gt_token' and 'gt_bbox'.
        
        Returns:
            dict: Dictionary with computed losses 'loss_token' and 'loss_bbox'.
        """
        gt_tokens = [s.gt_token for s in data_samples]
        gt_bboxes = [s.gt_bbox for s in data_samples]

        loss1 = self.loss_token(outputs[0], gt_tokens)
        loss2 = self.loss_bbox(outputs[1], gt_bboxes)
        return dict(loss_token=loss1, loss_bbox=loss2)
