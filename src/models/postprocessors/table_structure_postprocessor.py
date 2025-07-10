# TableStructurePostprocessor: chỉ xử lý structure (table tags, bbox)
from typing import Dict, List, Optional, Sequence, Tuple
import torch
import numpy as np
from mmocr.registry import MODELS
from mmocr.models.textrecog.postprocessors.base import BaseTextRecogPostprocessor
from mmocr.structures import TextRecogDataSample

@MODELS.register_module()
class TableStructurePostprocessor(BaseTextRecogPostprocessor):
    """Postprocessor for TableMaster structure branch: table tags & bbox only."""
    def __init__(self, dictionary: Dict, max_seq_len: int = 500, start_end_same: bool = False, **kwargs) -> None:
        super().__init__(dictionary=dictionary, max_seq_len=max_seq_len, **kwargs)
        self.start_end_same = start_end_same
        if self.start_end_same:
            raise AssertionError("TableMaster requires start_end_same=False")

    def get_single_prediction(self, probs: torch.Tensor, data_sample: Optional[TextRecogDataSample] = None) -> Tuple[List[int], List[float]]:
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

    def _get_pred_bbox_mask(self, strings: List[str]) -> List[List[int]]:
        pred_bbox_masks = []
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
                    if char in ['<td></td>', '<td']:
                        pred_bbox_mask.append(1)
                    else:
                        pred_bbox_mask.append(0)
            pred_bbox_masks.append(pred_bbox_mask)
        return pred_bbox_masks

    def _decode_bboxes(self, outputs_bbox: torch.Tensor, pred_bbox_masks: List[List[int]], data_samples: Sequence[TextRecogDataSample]) -> List[np.ndarray]:
        pred_bboxes = []
        for output_bbox, pred_bbox_mask, data_sample in zip(outputs_bbox, pred_bbox_masks, data_samples):
            output_bbox = output_bbox.cpu().numpy()
            img_meta = data_sample.metainfo
            scale_factor = img_meta.get('scale_factor', [1.0, 1.0])
            pad_shape = img_meta.get('pad_shape', img_meta.get('img_shape', [1, 1]))
            pred_bbox_mask_array = np.array(pred_bbox_mask)
            output_bbox = self._filter_invalid_bbox(output_bbox, pred_bbox_mask_array)
            output_bbox[:, 0::2] = output_bbox[:, 0::2] * pad_shape[1]
            output_bbox[:, 1::2] = output_bbox[:, 1::2] * pad_shape[0]
            output_bbox[:, 0::2] = output_bbox[:, 0::2] / scale_factor[1]
            output_bbox[:, 1::2] = output_bbox[:, 1::2] / scale_factor[0]
            pred_bboxes.append(output_bbox)
        return pred_bboxes

    def _filter_invalid_bbox(self, output_bbox: np.ndarray, pred_bbox_mask: np.ndarray) -> np.ndarray:
        low_mask = (output_bbox >= 0.) * 1
        high_mask = (output_bbox <= 1.) * 1
        mask = np.sum((low_mask + high_mask), axis=1)
        value_mask = np.where(mask == 2*4, 1, 0)
        output_bbox_len = output_bbox.shape[0]
        pred_bbox_mask_len = pred_bbox_mask.shape[0]
        padded_pred_bbox_mask = np.zeros(output_bbox_len, dtype='int64')
        padded_pred_bbox_mask[:pred_bbox_mask_len] = pred_bbox_mask
        filtered_output_bbox = (output_bbox * np.expand_dims(value_mask, 1) * np.expand_dims(padded_pred_bbox_mask, 1))
        return filtered_output_bbox

    def _adjust_bboxes_len(self, bboxes: List[np.ndarray], strings: List[str]) -> List[np.ndarray]:
        new_bboxes = []
        for bbox, string in zip(bboxes, strings):
            string_tokens = string.split(',')
            string_len = len(string_tokens)
            bbox = bbox[:string_len, :]
            new_bboxes.append(bbox)
        return new_bboxes

    def _get_avg_scores(self, str_scores: List[List[float]]) -> List[float]:
        avg_scores = []
        for str_score in str_scores:
            if len(str_score) > 0:
                score = sum(str_score) / len(str_score)
            else:
                score = 0.0
            avg_scores.append(score)
        return avg_scores
