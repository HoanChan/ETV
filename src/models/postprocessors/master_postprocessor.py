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
class MasterPostprocessor(BaseTextRecogPostprocessor):
    """Basic Master postprocessor for simpler recognition tasks.
    
    Args:
        dictionary (dict): Dictionary config
        max_seq_len (int): Maximum sequence length. Defaults to 40.
        start_end_same (bool): Whether to use same start/end token. Defaults to True.
    """
    
    def __init__(self,
                 dictionary: Dict,
                 max_seq_len: int = 40,
                 start_end_same: bool = True,
                 **kwargs) -> None:
        
        # Initialize base class
        super().__init__(dictionary=dictionary, max_seq_len=max_seq_len, **kwargs)
        self.start_end_same = start_end_same
    
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
        char_indexes, char_scores = self._tensor2idx(probs)
        return char_indexes, char_scores
    
    def _tensor2idx(self, outputs: torch.Tensor) -> Tuple[List[int], List[float]]:
        """Convert tensor to indices and scores."""
        if outputs.dim() == 3:
            outputs = outputs.squeeze(0)
            
        seq = outputs.softmax(-1)
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
            
        return str_index, str_score
