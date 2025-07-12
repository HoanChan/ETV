# Copyright (c) Lê Hoàn Chân. All rights reserved.
from typing import Dict, Optional

import torch
import torch.nn as nn

from mmocr.registry import MODELS


@MODELS.register_module()
class MASTERTFLoss(nn.CrossEntropyLoss):
    """Implementation of Cross Entropy loss module for MASTER transformer.
    
    This loss function is specifically designed for MASTER (Multi-Aspect 
    Structure Table Recognition) model's sequence prediction task.
    
    Args:
        ignore_index (int): The index to be ignored in loss computation.
            Defaults to -1.
        reduction (str): The reduction method for the output. 
            Options are 'none', 'mean' and 'sum'. Defaults to 'none'.
        flatten (bool): Whether to flatten the output and target tensors.
            If True, the output will be flattened to (N*L, C) and target
            to (N*L,). If False, output will be permuted to (N, C, L).
            Defaults to True.
        **kwargs: Other keyword arguments passed to nn.CrossEntropyLoss.
    """

    def __init__(self,
                 ignore_index: int = -1,
                 reduction: str = 'none',
                 flatten: bool = True,
                 **kwargs) -> None:
        super().__init__(ignore_index=ignore_index, reduction=reduction, **kwargs)
        assert isinstance(flatten, bool)
        self.flatten = flatten

    def _format_inputs(self, 
                      outputs: torch.Tensor, 
                      targets_dict: Dict[str, torch.Tensor]) -> tuple:
        """Format inputs for loss computation.
        
        Args:
            outputs (torch.Tensor): The prediction logits with shape 
                (N, L, C) where N is batch size, L is sequence length, 
                and C is number of classes.
            targets_dict (Dict[str, torch.Tensor]): Dictionary containing
                target information. Must contain 'padded_targets' key.
                
        Returns:
            tuple: Formatted (outputs, targets) tensors ready for loss computation.
        """
        # Extract targets from dictionary
        targets = targets_dict['padded_targets'].to(outputs.device)
        
        # MASTER decoder already handles sequence shifting internally
        # We take targets starting from index 1 to align with predictions
        # targets = targets[:, 1:].contiguous()
        
        if self.flatten:
            # Flatten for standard CrossEntropyLoss: (N*L, C) and (N*L,)
            outputs = outputs.contiguous().view(-1, outputs.size(-1))
            targets = targets.contiguous().view(-1)
        else:
            # Permute to (N, C, L) format expected by CrossEntropyLoss
            outputs = outputs.permute(0, 2, 1).contiguous()
            
        return outputs, targets

    def forward(self, 
                outputs: torch.Tensor, 
                targets_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Forward function to compute the cross entropy loss.
        
        Args:
            outputs (torch.Tensor): The prediction logits with shape 
                (N, L, C) where N is batch size, L is sequence length, 
                and C is number of classes.
            targets_dict (Dict[str, torch.Tensor]): Dictionary containing
                target information. Must contain 'padded_targets' key.
                
        Returns:
            torch.Tensor: The computed cross entropy loss.
        """
        # Format inputs for loss computation
        formatted_outputs, formatted_targets = self._format_inputs(outputs, targets_dict)

        return super().forward(formatted_outputs, formatted_targets)