from typing import Dict, List, Optional, Sequence, Tuple, Union
import torch
import numpy as np

from mmocr.registry import MODELS, TASK_UTILS
from mmocr.models.common.dictionary import Dictionary
from mmocr.structures import TextRecogDataSample
from structures.token_recog_data_sample import TokenRecogDataSample


@MODELS.register_module()
class TableMasterPostprocessor:
    """Custom postprocessor for TableMaster that handles both classification and bbox outputs.
    
    This postprocessor is designed to work with TableMaster decoder which returns
    tuple of (cls_output, bbox_output) and processes them together similar to
    the original mmOCR 0.x TableMasterConvertor.
    
    Args:
        dictionary (dict): Structure dictionary config for table tags
        max_seq_len (int): Maximum sequence length for structure tokens. Defaults to 500.
        start_end_same (bool): Whether to use same token for start and end. Defaults to False.
    """
    
    def __init__(self,
                 dictionary: Dict,
                 max_seq_len: int = 500,
                 start_end_same: bool = False,
                 **kwargs) -> None:
        
        # Initialize dictionary
        if isinstance(dictionary, dict):
            self.dictionary = TASK_UTILS.build(dictionary)
        else:
            self.dictionary = dictionary
            
        self.max_seq_len = max_seq_len
        self.start_end_same = start_end_same
        
        # Validate configuration
        if self.start_end_same:
            raise AssertionError("TableMaster requires start_end_same=False")
        
        # Set up ignore indexes (similar to BaseTextRecogPostprocessor)
        self.ignore_indexes = [self.dictionary.padding_idx]
        
    def __call__(self, 
                 outputs: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
                 data_samples: Optional[Sequence[TokenRecogDataSample]] = None
                 ) -> Sequence[TokenRecogDataSample]:
        """Process outputs from TableMaster decoder.
        
        Args:
            outputs: Can be either:
                - torch.Tensor: Only cls_output (for backward compatibility)
                - Tuple[torch.Tensor, torch.Tensor]: (cls_output, bbox_output)
            data_samples: List of data samples with metadata
            
        Returns:
            List of TokenRecogDataSample with prediction results
        """
        # Handle different input formats
        if isinstance(outputs, tuple):
            cls_output, bbox_output = outputs
        else:
            cls_output = outputs
            bbox_output = None
            
        # Process classification output
        str_indexes, str_scores = self._tensor2idx(cls_output)
        strings = []
        for str_idx in str_indexes:
            strings.append(self.dictionary.idx2str(str_idx))
        scores = self._get_avg_scores(str_scores)
        
        # Process bbox output if available
        if bbox_output is not None:
            pred_bbox_masks = self._get_pred_bbox_mask(strings)
            pred_bboxes = self._decode_bboxes(bbox_output, pred_bbox_masks, data_samples)
            pred_bboxes = self._adjust_bboxes_len(pred_bboxes, strings)
        else:
            pred_bboxes = [None] * len(strings)
        
        # Create result data samples
        results = []
        for i, (string, score, pred_bbox) in enumerate(zip(strings, scores, pred_bboxes)):
            if data_samples is not None:
                data_sample = data_samples[i]
            else:
                data_sample = TokenRecogDataSample()
                
            # Set prediction results
            data_sample.pred_text = string
            data_sample.pred_score = score
            
            # Set bbox if available
            if pred_bbox is not None:
                data_sample.pred_bbox = pred_bbox
                
            results.append(data_sample)
            
        return results
    
    def _tensor2idx(self, outputs: torch.Tensor) -> Tuple[List[List[int]], List[List[float]]]:
        """Convert tensor outputs to indexes and scores."""
        batch_size = outputs.size(0)
        indexes, scores = [], []
        
        for idx in range(batch_size):
            seq = outputs[idx, :, :].softmax(-1)
            max_value, max_idx = torch.max(seq, -1)
            
            str_index, str_score = [], []
            output_index = max_idx.cpu().detach().numpy().tolist()
            output_score = max_value.cpu().detach().numpy().tolist()
            
            for char_index, char_score in zip(output_index, output_score):
                if char_index in self.ignore_indexes:
                    continue
                if char_index == self.dictionary.end_idx:
                    break
                str_index.append(char_index)
                str_score.append(char_score)
                
            indexes.append(str_index)
            scores.append(str_score)
            
        return indexes, scores
    
    def _get_pred_bbox_mask(self, strings: List[str]) -> List[List[int]]:
        """Get bbox mask for structure tokens."""
        pred_bbox_masks = []
        
        # Get special tokens
        sos_token = self.dictionary.idx2str([self.dictionary.start_idx])[0] if hasattr(self.dictionary, 'start_idx') else '<BOS>'
        eos_token = self.dictionary.idx2str([self.dictionary.end_idx])[0] if hasattr(self.dictionary, 'end_idx') else '<EOS>'
        pad_token = self.dictionary.idx2str([self.dictionary.padding_idx])[0] if hasattr(self.dictionary, 'padding_idx') else '<PAD>'
        
        for string in strings:
            pred_bbox_mask = []
            char_list = string.split(',') if string else []
            
            for char in char_list:
                char = char.strip()
                if char == eos_token:
                    pred_bbox_mask.append(0)
                    break
                elif char in [pad_token, sos_token]:
                    pred_bbox_mask.append(0)
                    continue
                else:
                    # Only cells should have bbox predictions
                    if char in ['<td></td>', '<td']:
                        pred_bbox_mask.append(1)
                    else:
                        pred_bbox_mask.append(0)
                        
            pred_bbox_masks.append(pred_bbox_mask)
            
        return pred_bbox_masks
    
    def _decode_bboxes(self, 
                      outputs_bbox: torch.Tensor, 
                      pred_bbox_masks: List[List[int]], 
                      data_samples: Sequence[TokenRecogDataSample]) -> List[np.ndarray]:
        """Decode bbox predictions."""
        pred_bboxes = []
        
        for output_bbox, pred_bbox_mask, data_sample in zip(outputs_bbox, pred_bbox_masks, data_samples):
            output_bbox = output_bbox.cpu().numpy()
            
            # Get image metadata
            img_meta = data_sample.metainfo
            scale_factor = img_meta.get('scale_factor', [1.0, 1.0])
            pad_shape = img_meta.get('pad_shape', img_meta.get('img_shape', [1, 1]))
            
            # Filter invalid bboxes
            pred_bbox_mask_array = np.array(pred_bbox_mask)
            output_bbox = self._filter_invalid_bbox(output_bbox, pred_bbox_mask_array)
            
            # De-normalize to pad shape
            output_bbox[:, 0::2] = output_bbox[:, 0::2] * pad_shape[1]
            output_bbox[:, 1::2] = output_bbox[:, 1::2] * pad_shape[0]
            
            # Scale to origin shape
            output_bbox[:, 0::2] = output_bbox[:, 0::2] / scale_factor[1]
            output_bbox[:, 1::2] = output_bbox[:, 1::2] / scale_factor[0]
            
            pred_bboxes.append(output_bbox)
            
        return pred_bboxes
    
    def _filter_invalid_bbox(self, output_bbox: np.ndarray, pred_bbox_mask: np.ndarray) -> np.ndarray:
        """Filter invalid bbox predictions."""
        # Check if bbox coordinates are in valid range [0, 1]
        low_mask = (output_bbox >= 0.) * 1
        high_mask = (output_bbox <= 1.) * 1
        mask = np.sum((low_mask + high_mask), axis=1)
        value_mask = np.where(mask == 2*4, 1, 0)  # All 4 coordinates valid
        
        # Pad bbox mask to match output length
        output_bbox_len = output_bbox.shape[0]
        pred_bbox_mask_len = pred_bbox_mask.shape[0]
        padded_pred_bbox_mask = np.zeros(output_bbox_len, dtype='int64')
        padded_pred_bbox_mask[:pred_bbox_mask_len] = pred_bbox_mask
        
        # Apply both masks
        filtered_output_bbox = (output_bbox * 
                              np.expand_dims(value_mask, 1) * 
                              np.expand_dims(padded_pred_bbox_mask, 1))
        
        return filtered_output_bbox
    
    def _adjust_bboxes_len(self, bboxes: List[np.ndarray], strings: List[str]) -> List[np.ndarray]:
        """Adjust bbox length to match string length."""
        new_bboxes = []
        for bbox, string in zip(bboxes, strings):
            string_tokens = string.split(',')
            string_len = len(string_tokens)
            bbox = bbox[:string_len, :]
            new_bboxes.append(bbox)
        return new_bboxes
    
    def _get_avg_scores(self, str_scores: List[List[float]]) -> List[float]:
        """Calculate average scores for each string."""
        avg_scores = []
        for str_score in str_scores:
            if len(str_score) > 0:
                score = sum(str_score) / len(str_score)
            else:
                score = 0.0
            avg_scores.append(score)
        return avg_scores
