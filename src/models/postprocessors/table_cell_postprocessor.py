# TableCellPostprocessor: chỉ xử lý cell content (text trong ô)
from typing import Dict, List, Optional, Tuple
import torch
from mmocr.registry import MODELS
from mmocr.models.textrecog.postprocessors.base import BaseTextRecogPostprocessor

@MODELS.register_module()
class TableCellPostprocessor(BaseTextRecogPostprocessor):
    """Postprocessor for TableMaster cell content branch only."""
    def __init__(self, dictionary: Dict, max_seq_len: int = 100, start_end_same: bool = False, **kwargs) -> None:
        super().__init__(dictionary=dictionary, max_seq_len=max_seq_len, **kwargs)
        self.start_end_same = start_end_same
        if self.start_end_same:
            raise AssertionError("TableCellPostprocessor requires start_end_same=False")

    def get_single_prediction(self, probs: torch.Tensor) -> Tuple[List[int], List[float]]:
        char_indexes, char_scores = self._tensor2idx(probs.unsqueeze(0))
        return char_indexes[0], char_scores[0]

    def _tensor2idx(self, outputs: torch.Tensor) -> Tuple[List[List[int]], List[List[float]]]:
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
