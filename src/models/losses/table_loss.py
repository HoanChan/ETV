# Copyright (c) Lê Hoàn Chân. All rights reserved.
from typing import Dict, Union, Sequence
import torch
import torch.nn as nn
from mmocr.registry import MODELS
from mmocr.models.common.dictionary import Dictionary
from mmocr.models.textrecog.module_losses.base import BaseTextRecogModuleLoss
from structures.token_recog_data_sample import TokenRecogDataSample

@MODELS.register_module()
class TableLoss(BaseTextRecogModuleLoss):
    """Implementation of composite loss for table structure recognition.
    
    This loss combines text recognition loss and bounding box regression loss
    for table structure models.
    
    Args:
        loss_token (dict): Config for text loss module.
        loss_bbox (dict): Config for bbox loss module.
    """
    def __init__(self, 
                 loss_token: dict, 
                 loss_bbox: dict,
                 dictionary: Union[Dict, Dictionary],
                 max_seq_len: int = 40,
                 max_bbox_len: int = 100,
                 **kwargs) -> None:
        super().__init__(
            dictionary=dictionary,
            max_seq_len=max_seq_len,
            **kwargs
        )
        self.max_bbox_len = max_bbox_len
        self.loss_token = MODELS.build(loss_token)
        self.loss_bbox = MODELS.build(loss_bbox)

    def forward(self, outputs: tuple[torch.Tensor, torch.Tensor], data_samples: list[TokenRecogDataSample], **kwargs) -> Dict[str, torch.Tensor]:
        """Forward function to compute composite loss.
        
        Args:
            outputs (tuple): Model outputs, each element is a tensor.
            data_samples (list): List of ground truth samples, each with attributes
                'gt_token' and 'gt_bbox'.
        
        Returns:
            dict: Dictionary with computed losses 'loss_token' and 'loss_bbox'.
        """
        # add padded_indexes to data_samples
        data_samples = self.get_targets(data_samples)
        # Extract ground truth tokens and bounding boxes from data samples
        gt_tokens = torch.stack([s.gt_tokens.padded_indexes for s in data_samples])
        bboxes = torch.stack([s.metainfo['padded_bboxes'] for s in data_samples])
        masks = torch.stack([s.metainfo['padded_masks'] for s in data_samples])
        loss1 = self.loss_token(outputs[0], {'padded_targets': gt_tokens})
        loss2 = self.loss_bbox(outputs[1], {'padded_bboxes': bboxes, 'padded_masks': masks})
        return dict(loss_token=loss1, loss_bbox=loss2)
    
    def _process_tokens(self, data_samples: Sequence[TokenRecogDataSample]):
        """Xử lý tokens: thêm indexes và padded_indexes cho mỗi sample."""
        for data_sample in data_samples:
            if data_sample.get('have_target', False):
                continue

            # gt_tokens should already be token indices, not strings
            if isinstance(data_sample.gt_tokens, torch.Tensor):
                # If already tensor, squeeze to 1D and convert to list for processing
                tokens = data_sample.gt_tokens.squeeze().tolist()
                if isinstance(tokens, int):  # Single token case
                    tokens = [tokens]
            elif isinstance(data_sample.gt_tokens, (list, tuple)):
                # If list/tuple of token indices
                tokens = list(data_sample.gt_tokens)
            else:
                # If it's a string of tokens (comma-separated), convert to indices
                tokens = self.dictionary.str2idx(str(data_sample.gt_tokens))

            indexes = torch.LongTensor(tokens)
            # Create target sequence with start/end tokens
            src_target = torch.LongTensor(indexes.size(0) + 2).fill_(0)
            src_target[1:-1] = indexes
            
            if self.dictionary.start_idx is not None:
                src_target[0] = self.dictionary.start_idx
                slice_start = 0
            else:
                slice_start = 1
                
            if self.dictionary.end_idx is not None:
                src_target[-1] = self.dictionary.end_idx
                slice_end = src_target.size(0)
            else:
                slice_end = src_target.size(0) - 1
                
            src_target = src_target[slice_start:slice_end]
            
            # Apply padding if needed
            if self.pad_idx is not None:
                padded_indexes = (torch.ones(self.max_seq_len) * self.pad_idx).long()
                char_num = min(src_target.size(0), self.max_seq_len)
                padded_indexes[:char_num] = src_target[:char_num]
            else:
                padded_indexes = src_target

            # Store processed targets in gt_tokens object
            data_sample.gt_tokens.indexes = indexes
            data_sample.gt_tokens.padded_indexes = padded_indexes
            # Mark as processed
            data_sample.set_metainfo({'have_target':True})
    
    def _process_bboxes_and_masks(self, data_samples: Sequence[TokenRecogDataSample]):
        """Xử lý padding cho bboxes và masks, lưu vào metainfo."""
        for data_sample in data_samples:
            if data_sample.get('have_bboxes', False):
                continue
            bboxes = data_sample.get('bboxes')
            masks = data_sample.get('masks')
            bbox_tensor = torch.zeros((self.max_bbox_len, 4), dtype=torch.float32)
            bbox_tensor[:min(len(bboxes), self.max_bbox_len)] = torch.tensor(bboxes[:self.max_bbox_len], dtype=torch.float32)
            mask_tensor = torch.zeros(self.max_bbox_len, dtype=torch.float32)
            mask_tensor[:min(len(masks), self.max_bbox_len)] = torch.tensor(masks[:self.max_bbox_len], dtype=torch.float32)
            data_sample.set_metainfo({'padded_bboxes':bbox_tensor})
            data_sample.set_metainfo({'padded_masks':mask_tensor})
            # Mark as processed
            data_sample.set_metainfo({'have_bboxes':True})

    def get_targets(self, data_samples: Sequence[TokenRecogDataSample]) -> Sequence[TokenRecogDataSample]:
        """Target generator for table structure recognition.
        
        Override base class method to handle tokens instead of characters.
        The base class processes gt_text.item character by character using str2idx,
        but TableLoss expects gt_tokens which contains pre-processed token sequences.

        Args:
            data_samples (list): It usually includes ``gt_tokens`` 
            information for table structure recognition.

        Returns:
            For gt_tokens, two keys will be added:

            - indexes (torch.LongTensor): Token indexes representing gt tokens.
              All special tokens are excluded, except for UKN.
            - padded_indexes (torch.LongTensor): Token indexes representing
              gt_tokens with BOS and EOS if applicable, following several padding
              indexes until the length reaches `max_seq_len`.
            For bboxes and masks, two keys will be added:
            - padded_bboxes (torch.Tensor): Padded bounding boxes tensor.
            - padded_masks (torch.Tensor): Padded masks tensor.

        """
        self._process_tokens(data_samples)
        self._process_bboxes_and_masks(data_samples)
        return data_samples