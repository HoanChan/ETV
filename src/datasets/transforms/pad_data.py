# Copyright (c) Lê Hoàn Chân. All rights reserved.
from typing import Dict, Optional, Sequence, Union
import torch
from mmcv.transforms import BaseTransform
from mmocr.registry import TRANSFORMS
from mmocr.registry import TASK_UTILS
from mmocr.models.common.dictionary import Dictionary

from structures.table_master_data_sample import TableMasterDataSample

@TRANSFORMS.register_module()
class PadData(BaseTransform):
    """
    Pad tokens and bboxes for each sample in a batch.
    - For tokens: add 'indexes' and 'padded_indexes' attributes to each data sample for loss computation.
    - For bboxes: pad bboxes and masks to max_bbox_len.

    Args:
        - dictionary (Union[Dict, Dictionary]): The dictionary object or configuration for token-to-index conversion.
        - max_seq_len (int): Maximum length for token sequence padding. Sequences longer than this will be truncated.
        - max_bbox_len (int): Maximum number of bounding boxes to pad to. Extra boxes will be truncated.
        - pad_with (str): Padding mode for tokens. Can be 'auto', 'padding', 'end', or 'none'.
            + 'auto': Use padding_idx if available, otherwise end_idx.
            + 'padding': Use padding_idx.
            + 'end': Use end_idx.
            + 'none': No padding will be applied.
        **kwargs: Additional keyword arguments.
    """
    def __init__(self,
                 dictionary: Union[Dict, Dictionary],
                 max_seq_len: int = 500,
                 max_bbox_len: int = 500,
                 pad_with: str = 'auto',
                 **kwargs) -> None:
        super().__init__()
        self.dictionary = dictionary
        self.max_seq_len = max_seq_len
        self.max_bbox_len = max_bbox_len
        self.pad_with = pad_with
        if isinstance(dictionary, dict):
            self.dictionary = TASK_UTILS.build(dictionary)
        elif isinstance(dictionary, Dictionary):
            self.dictionary = dictionary
        else:
            raise TypeError(f'The type of dictionary should be `Dictionary` or dict, but got {type(dictionary)}')
        
        assert pad_with in ['auto', 'padding', 'end', 'none']
        if pad_with == 'auto':
            self.pad_idx = self.dictionary.padding_idx or self.dictionary.end_idx
        elif pad_with == 'padding':
            self.pad_idx = self.dictionary.padding_idx
        elif pad_with == 'end':
            self.pad_idx = self.dictionary.end_idx
        else:
            self.pad_idx = None
        if self.pad_idx is None and pad_with != 'none':
            if pad_with == 'auto':
                raise ValueError('pad_with="auto", but dictionary.end_idx and dictionary.padding_idx are both None')
            else:
                raise ValueError(f'pad_with="{pad_with}", but dictionary.{pad_with}_idx is None')
        

    def transform(self, results: dict) -> dict:
        """
        Pad the token and bounding box data in a TableMasterDataSample.

        This method processes the input data sample by:
            - Padding the token indices to the specified max_seq_len, adding start/end tokens if needed, and storing both the original and padded indices.
            - Padding the bounding boxes and masks to max_bbox_len, filling with zeros.
            - Marking the sample as processed to avoid redundant work.

        Args:
            data_sample (TableMasterDataSample): The data sample to be transformed. Should contain 'gt_tokens', and 'gt_instances'.

        Returns:
            TableMasterDataSample: The transformed data sample with padded tokens and bboxes/masks.
        """
        # Pad tokens if present
        data_sample = results.get('data_samples', None)
        assert isinstance(data_sample, TableMasterDataSample), f"data_sample should be an instance of TableMasterDataSample, but got {type(data_sample)}"
        if data_sample.get('have_padded_indexes', False):
            pass
        else:
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
                # If it's a list of tokens, convert to indices
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
            data_sample.set_metainfo({'have_padded_indexes':True})

        # Pad bboxes and masks if present
        if data_sample.get('have_padded_bboxes', False):
            pass
        else:
            bboxes = data_sample.get('bboxes')
            masks = data_sample.get('masks')
            bbox_tensor = torch.zeros((self.max_bbox_len, 4), dtype=torch.float32)
            bbox_tensor[:min(len(bboxes), self.max_bbox_len)] = torch.tensor(bboxes[:self.max_bbox_len], dtype=torch.float32)
            mask_tensor = torch.zeros(self.max_bbox_len, dtype=torch.float32)
            mask_tensor[:min(len(masks), self.max_bbox_len)] = torch.tensor(masks[:self.max_bbox_len], dtype=torch.float32)
            data_sample.set_metainfo({'padded_bboxes':bbox_tensor})
            data_sample.set_metainfo({'padded_masks':mask_tensor})
            # Mark as processed
            data_sample.set_metainfo({'have_padded_bboxes':True})

        return results

    def __repr__(self) -> str:
        repr_str = self.__class__.__name__
        repr_str += f'(max_seq_len={self.max_seq_len}, max_bbox_len={self.max_bbox_len}, pad_with={self.pad_with})'
        return repr_str
