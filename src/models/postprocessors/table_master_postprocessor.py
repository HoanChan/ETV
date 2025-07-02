# Converted to mmOCR 1.x standard with BaseTextRecogPostprocessor inheritance
# This implementation leverages BaseTextRecogPostprocessor features:
# - Dictionary handling and initialization
# - Base __call__ method for batch processing
# - Ignore indexes configuration
# - Standard get_single_prediction interface

from typing import Dict, List, Optional, Sequence, Tuple, Union
import torch
import numpy as np

from mmocr.registry import MODELS
from mmocr.models.textrecog.postprocessors.base import BaseTextRecogPostprocessor
from mmocr.structures import TextRecogDataSample


@MODELS.register_module()
class TableMasterPostprocessor(BaseTextRecogPostprocessor):
    """Postprocessor for TableMaster model supporting table structure recognition 
    with text content, bounding boxes, and cell content prediction.
    
    This postprocessor handles:
    1. Structure token prediction (table tags like <td>, <tr>, etc.)
    2. Cell content text recognition
    3. Bounding box prediction and filtering
    4. Multi-branch output coordination
    
    Args:
        dictionary (dict): Structure dictionary config for table tags
        cell_dictionary (dict): Cell content dictionary config
        max_seq_len (int): Maximum sequence length for structure tokens. Defaults to 500.
        max_seq_len_cell (int): Maximum sequence length for cell content. Defaults to 100.
        start_end_same (bool): Whether to use same token for start and end. Defaults to False.
    """
    
    def __init__(self,
                 dictionary: Dict,
                 cell_dictionary: Dict,
                 max_seq_len: int = 500,
                 max_seq_len_cell: int = 100,
                 start_end_same: bool = False,
                 **kwargs) -> None:
        
        # Initialize base class with dictionary
        super().__init__(dictionary=dictionary, max_seq_len=max_seq_len, **kwargs)
        
        # Initialize cell dictionary
        if isinstance(cell_dictionary, dict):
            self.cell_dictionary = MODELS.build(cell_dictionary)
        else:
            self.cell_dictionary = cell_dictionary
        
        self.max_seq_len_cell = max_seq_len_cell
        self.start_end_same = start_end_same
        
        # Validation
        self._validate_config()
        
    def _validate_config(self) -> None:
        """Validate configuration parameters."""
        if self.start_end_same:
            raise AssertionError("TableMaster requires start_end_same=False")
    
    def get_single_prediction(self, 
                             probs: torch.Tensor,
                             data_sample: Optional[TextRecogDataSample] = None) -> Tuple[List[int], List[float]]:
        """Get prediction from single probability tensor.
        
        Args:
            probs (torch.Tensor): Character probabilities with shape (T, C)
            data_sample (TextRecogDataSample): Data sample (optional)
            
        Returns:
            Tuple[List[int], List[float]]: Character indices and scores
        """
        # Convert tensor to character indices and scores
        char_indexes, char_scores = self._tensor2idx(probs.unsqueeze(0))
        return char_indexes[0], char_scores[0]
    
    def format_table_outputs(self,
                            structure_outputs: torch.Tensor,
                            bbox_outputs: torch.Tensor, 
                            cell_outputs: List[torch.Tensor],
                            data_samples: Sequence[TextRecogDataSample]) -> List[Dict]:
        """Format complete table recognition outputs including structure, bbox and cells.
        
        Args:
            structure_outputs (torch.Tensor): Structure prediction outputs
            bbox_outputs (torch.Tensor): Bounding box prediction outputs  
            cell_outputs (List[torch.Tensor]): Cell content prediction outputs
            data_samples (Sequence[TextRecogDataSample]): Data samples with metadata
            
        Returns:
            List[Dict]: Formatted results for each sample
        """
        results = []
        
        # Process structure branch
        structure_indexes, structure_scores = self._tensor2idx(structure_outputs)
        structure_strings = self.dictionary.idx2str(structure_indexes)
        structure_avg_scores = self._get_avg_scores(structure_scores)
        
        # Process bbox branch
        pred_bbox_masks = self._get_pred_bbox_mask(structure_strings)
        pred_bboxes = self._decode_bboxes(bbox_outputs, pred_bbox_masks, data_samples)
        pred_bboxes = self._adjust_bboxes_len(pred_bboxes, structure_strings)
        
        # Process cell content branch
        cell_strings = []
        cell_scores = []
        
        for idx, cell_output in enumerate(cell_outputs):
            if cell_output.size(0) == 1:
                # Empty cell case
                cell_strings_i = []
                cell_scores_i = []
            else:
                cell_indexes, cell_score_list = self._tensor2idx_cell(cell_output)
                cell_strings_i = self.cell_dictionary.idx2str(cell_indexes)
                cell_scores_i = self._get_avg_scores(cell_score_list)
                
            cell_strings.append(cell_strings_i)
            cell_scores.append(cell_scores_i)
        
        # Combine results
        for i in range(len(structure_strings)):
            result = {
                'structure_text': structure_strings[i],
                'structure_score': structure_avg_scores[i],
                'bboxes': pred_bboxes[i],
                'cell_texts': cell_strings[i],
                'cell_scores': cell_scores[i]
            }
            results.append(result)
            
        return results
    
    def _tensor2idx(self, outputs: torch.Tensor) -> Tuple[List[List[int]], List[List[float]]]:
        """Convert output tensor to character indices and scores.
        
        Args:
            outputs (torch.Tensor): Model outputs with shape (N, T, C)
            
        Returns:
            Tuple[List[List[int]], List[List[float]]]: Character indices and scores
        """
        batch_size = outputs.size(0)
        
        indexes, scores = [], []
        
        for idx in range(batch_size):
            seq = outputs[idx, :, :]
            seq = seq.softmax(-1)
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
    
    def _tensor2idx_cell(self, outputs: torch.Tensor) -> Tuple[List[List[int]], List[List[float]]]:
        """Convert cell content output tensor to character indices and scores.
        
        Args:
            outputs (torch.Tensor): Cell content outputs
            
        Returns:
            Tuple[List[List[int]], List[List[float]]]: Character indices and scores
        """
        batch_size = outputs.size(0)  
        # Use cell dictionary's padding index if available, otherwise use main dictionary
        cell_ignore_indexes = [getattr(self.cell_dictionary, 'padding_idx', self.dictionary.padding_idx)]
        
        indexes, scores = [], []
        
        for idx in range(batch_size):
            seq = outputs[idx, :, :]
            seq = seq.softmax(-1)
            max_value, max_idx = torch.max(seq, -1)
            
            str_index, str_score = [], []
            output_index = max_idx.cpu().detach().numpy().tolist()
            output_score = max_value.cpu().detach().numpy().tolist()
            
            for char_index, char_score in zip(output_index, output_score):
                if char_index in cell_ignore_indexes:
                    continue
                if char_index == getattr(self.cell_dictionary, 'end_idx', self.dictionary.end_idx):
                    break
                str_index.append(char_index)
                str_score.append(char_score)
                
            indexes.append(str_index)
            scores.append(str_score)
            
        return indexes, scores
    
    def _get_pred_bbox_mask(self, strings: List[str]) -> np.ndarray:
        """Generate bbox mask from predicted structure strings.
        
        Args:
            strings (List[str]): Predicted structure strings
            
        Returns:
            np.ndarray: Bbox masks
        """
        pred_bbox_masks = []
        sos_token = self.dictionary.idx2str([self.dictionary.start_idx])[0] if hasattr(self.dictionary, 'start_idx') else '<BOS>'
        eos_token = self.dictionary.idx2str([self.dictionary.end_idx])[0] if hasattr(self.dictionary, 'end_idx') else '<EOS>'  
        pad_token = self.dictionary.idx2str([self.dictionary.padding_idx])[0] if hasattr(self.dictionary, 'padding_idx') else '<PAD>'
        
        for string in strings:
            pred_bbox_mask = []
            char_list = string.split(',')
            
            for char in char_list:
                if char == eos_token:
                    pred_bbox_mask.append(0)
                    break
                elif char in [pad_token, sos_token]:
                    pred_bbox_mask.append(0)
                    continue
                else:
                    # Mark cells that should have bboxes
                    if char in ['<td></td>', '<td']:
                        pred_bbox_mask.append(1)
                    else:
                        pred_bbox_mask.append(0)
                        
            pred_bbox_masks.append(pred_bbox_mask)
            
        return np.array(pred_bbox_masks)
    
    def _decode_bboxes(self, 
                      outputs_bbox: torch.Tensor,
                      pred_bbox_masks: np.ndarray, 
                      data_samples: Sequence[TextRecogDataSample]) -> List[np.ndarray]:
        """Decode and denormalize bounding boxes.
        
        Args:
            outputs_bbox (torch.Tensor): Raw bbox outputs
            pred_bbox_masks (np.ndarray): Bbox masks
            data_samples (Sequence[TextRecogDataSample]): Data samples with metadata
            
        Returns:
            List[np.ndarray]: Decoded bboxes
        """
        pred_bboxes = []
        
        for output_bbox, pred_bbox_mask, data_sample in zip(outputs_bbox, pred_bbox_masks, data_samples):
            output_bbox = output_bbox.cpu().numpy()
            
            # Get image metadata
            img_meta = data_sample.metainfo
            scale_factor = img_meta.get('scale_factor', [1.0, 1.0])
            pad_shape = img_meta.get('pad_shape', img_meta.get('img_shape', [1, 1]))
            
            # Filter invalid bboxes
            output_bbox = self._filter_invalid_bbox(output_bbox, pred_bbox_mask)
            
            # Denormalize to pad shape
            output_bbox[:, 0::2] = output_bbox[:, 0::2] * pad_shape[1]
            output_bbox[:, 1::2] = output_bbox[:, 1::2] * pad_shape[0]
            
            # Scale to original shape
            output_bbox[:, 0::2] = output_bbox[:, 0::2] / scale_factor[1]
            output_bbox[:, 1::2] = output_bbox[:, 1::2] / scale_factor[0]
            
            pred_bboxes.append(output_bbox)
            
        return pred_bboxes
    
    def _filter_invalid_bbox(self, output_bbox: np.ndarray, pred_bbox_mask: np.ndarray) -> np.ndarray:
        """Filter invalid bounding boxes.
        
        Args:
            output_bbox (np.ndarray): Raw bbox coordinates
            pred_bbox_mask (np.ndarray): Bbox mask
            
        Returns:
            np.ndarray: Filtered bboxes
        """
        # Filter bboxes with coordinates outside [0,1]
        low_mask = (output_bbox >= 0.) * 1
        high_mask = (output_bbox <= 1.) * 1
        mask = np.sum((low_mask + high_mask), axis=1)
        value_mask = np.where(mask == 2*4, 1, 0)
        
        # Pad bbox mask to match output length
        output_bbox_len = output_bbox.shape[0]
        pred_bbox_mask_len = pred_bbox_mask.shape[0]
        padded_pred_bbox_mask = np.zeros(output_bbox_len, dtype='int64')
        padded_pred_bbox_mask[:pred_bbox_mask_len] = pred_bbox_mask
        
        # Apply filters
        filtered_output_bbox = (output_bbox * 
                               np.expand_dims(value_mask, 1) * 
                               np.expand_dims(padded_pred_bbox_mask, 1))
        
        return filtered_output_bbox
    
    def _adjust_bboxes_len(self, bboxes: List[np.ndarray], strings: List[str]) -> List[np.ndarray]:
        """Adjust bbox length to match string length.
        
        Args:
            bboxes (List[np.ndarray]): Bboxes
            strings (List[str]): Structure strings
            
        Returns:
            List[np.ndarray]: Adjusted bboxes
        """
        new_bboxes = []
        for bbox, string in zip(bboxes, strings):
            string_tokens = string.split(',')
            string_len = len(string_tokens)
            bbox = bbox[:string_len, :]
            new_bboxes.append(bbox)
        return new_bboxes
    
    def _get_avg_scores(self, str_scores: List[List[float]]) -> List[float]:
        """Calculate average scores for strings.
        
        Args:
            str_scores (List[List[float]]): Character scores for each string
            
        Returns:
            List[float]: Average scores
        """
        avg_scores = []
        for str_score in str_scores:
            if len(str_score) > 0:
                score = sum(str_score) / len(str_score)
            else:
                score = 0.0
            avg_scores.append(score)
        return avg_scores

