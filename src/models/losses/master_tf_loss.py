# Copyright (c) Lê Hoàn Chân. All rights reserved.
from typing import Dict, Union
from pyparsing import Sequence
import torch
import torch.nn as nn
from mmocr.registry import MODELS
from mmocr.registry import TASK_UTILS
from mmocr.models.common.dictionary import Dictionary
from structures.token_recog_data_sample import TokenRecogDataSample

@MODELS.register_module()
class MASTERTFLoss(nn.Module):
    """
    Cross-Entropy loss module for the MASTER transformer model's sequence prediction task.

    This loss is tailored for the MASTER (Multi-Aspect Structure Table Recognition) model, handling sequence-to-sequence prediction for table recognition tasks.

    Args:
        ignore_index (int): Index to ignore in the loss computation. Default is -1.
        reduction (str): Specifies the reduction to apply to the output: 'none', 'mean', or 'sum'. Default is 'none'.
        flatten (bool): If True, flattens outputs and targets to (N*L, C) and (N*L,) for loss computation. If False, permutes outputs to (N, C, L). Default is True.
        **kwargs: Additional keyword arguments for nn.Module.
    """

    def __init__(self, # https://github.com/open-mmlab/mmocr/blob/main/mmocr/models/textrecog/module_losses/base.py
                 ignore_index: int = -1,
                 reduction: str = 'none',
                 flatten: bool = True) -> None:
        super().__init__()
        self.ctc_loss = nn.CrossEntropyLoss(
            ignore_index=ignore_index,
            reduction=reduction)
        assert isinstance(flatten, bool)
        self.flatten = flatten
            
    def forward(self, outputs: torch.Tensor, data_samples: list[TokenRecogDataSample], **kwargs) -> Dict[str, torch.Tensor]:
        """
        Compute the cross-entropy loss for sequence prediction.

        Args:
            outputs (torch.Tensor): Prediction logits of shape (N, L, C), where N is the batch size, L is the sequence length, and C is the number of classes.
            data_samples (list[TokenRecogDataSample]): List of data samples containing ground truth token information.
            **kwargs: Additional keyword arguments.

        Returns:
            Dict[str, torch.Tensor]: Dictionary containing the computed loss tensor.
        """
        # Extract ground truth tokens and bounding boxes from data samples
        gt_tokens = torch.stack([s.gt_tokens.padded_indexes for s in data_samples])
        # Format inputs for loss computation
        targets = gt_tokens.to(outputs.device)
        
        # MASTER decoder already handles sequence shifting internally
        # We take targets starting from index 1 to align with predictions
        targets = targets[:, 1:].contiguous()
        
        if self.flatten:
            # Flatten for standard CrossEntropyLoss: (N*L, C) and (N*L,)
            outputs = outputs.contiguous().view(-1, outputs.size(-1))
            targets = targets.contiguous().view(-1)
        else:
            # Permute to (N, C, L) format expected by CrossEntropyLoss
            outputs = outputs.permute(0, 2, 1).contiguous()

        return {'loss_tokens': self.ctc_loss(outputs, targets)}