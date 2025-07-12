# Copyright (c) Lê Hoàn Chân. All rights reserved.
from typing import Optional
import numpy as np
from mmcv.transforms import BaseTransform
from mmocr.registry import TRANSFORMS
from datasets.transforms.transforms_utils import align_bbox_mask, build_bbox_mask, build_empty_bbox_mask, get_bbox_nums, remove_thead_Bb, process_token

@TRANSFORMS.register_module()
class LoadTokens(BaseTransform):
    """
    This transform normalizes the raw structure tokens in the annotation through the following steps:
        - It removes bold tags (<b>, </b>) from table header cells.
        - It merges common tokens and inserts empty bounding box tokens.
    After normalization, it loads and processes the token annotations from instances in the table dataset.

    - If with_structure: returns 
        + 'tokens': list of structure tokens.
        + 'bboxes': list of all token bboxes (calculated from structure tokens and cell bboxes).
        + 'masks': mask for bboxes (1 for valid bbox, 0 for empty bbox).

    - If with_cell: returns 'cells' as a list of dicts, each dict contains:
        {
            'tokens': ...,    # list of tokens for the cell
            'bboxes': ...,    # bboxes of the cell (one bbox)
            'id': ...         # id of the cell
        }

    Full output annotation format (if with_structure and with_cell):
        {
            'img': ...        # original table image
            'tokens': [...],  # list of structure tokens if with_structure
            'bboxes': [...],  # list of all token bboxes if with_structure
            'masks': [...],   # list of all masks for bboxes if with_structure
            'cells': [        # list of cell information if with_cell
                {
                    'tokens': ...,    # list of tokens for the cell
                    'bboxes': ...,    # bboxes of the cell (one bbox)
                    'id': ...         # id of the cell
                }
                ...
            ],
        }

    Input annotation format:
        {
            'img': ...  # original table image
            'instances': [
                {
                    'tokens': [...],     # list of tokens for the instance (table or cell)
                    'task_type': ...     # 'structure' or 'content',
                    'cell_id': ...,      # only for content
                    'bbox': [...],       # only for content
                    'img': ...           # only for content if already cropped
                },
                ...
            ],
        }

    Args:
        with_structure (bool): Whether to load structure tokens. If True, will return 'img', 'tokens', 'bboxes', 'masks'.
        with_cell (bool): Whether to load cell tokens. If True, will return 'cells'.
        max_structure_token_len (int, optional): Limit the number of structure tokens.
        max_cell_token_len (int, optional): Limit the number of cell tokens.
    """

    def __init__(self,
                 with_structure: bool = True,
                 with_cell: bool = True,
                 max_structure_token_len: Optional[int] = None,
                 max_cell_token_len: Optional[int] = None,
                 **kwargs) -> None:
        super().__init__()
        assert with_structure or with_cell, "At least one of with_structure or with_cell must be True"
        self.with_structure = with_structure
        self.with_cell = with_cell
        self.max_structure_token_len = max_structure_token_len
        self.max_cell_token_len = max_cell_token_len

    def transform(self, results: dict) -> dict:
        """
        Load token annotations for structure or cell.
        - If with_structure: returns 'img' and 'tokens'.
        - If with_cell: returns 'cells' as a list of dicts and 'bboxes' as a list of bboxes.
        """
        if 'instances' not in results:
            if self.with_structure:
                results['tokens'] = []
                results['bboxes'] = []
            if self.with_cell:
                results['cells'] = []
            return results
        
        # Normalize Tokens
        structures = [inst for inst in results['instances'] if inst.get('task_type') == 'structure']
        cells = [cell for cell in results['instances'] if cell.get('task_type') == 'content']
        for instance in structures:
            tokens = instance.get('tokens', [])
            tokens = remove_thead_Bb(tokens)
            tokens = process_token(tokens, cells)
            instance['tokens'] = tokens
        
        # Get information for each cell
        cells = []
        bboxes = []
        for instance in results['instances']:
            if instance.get('task_type') == 'content':
                tokens = instance.get('tokens', [])
                if self.max_cell_token_len is not None:
                    tokens = tokens[:self.max_cell_token_len]
                bbox = instance.get('bbox', [0, 0, 0, 0])
                cell_id = instance.get('cell_id', 0)
                cells.append({
                    'tokens': tokens,
                    'bboxes': [bbox],
                    'id': cell_id
                })
                bboxes.append(bbox)

        if self.with_cell:
            results['cells'] = cells

        if self.with_structure:
            # Get structure tokens
            structure_tokens = []
            for instance in results['instances']:
                if instance.get('task_type') == 'structure':
                    tokens = instance.get('tokens', [])
                    if self.max_structure_token_len is not None:
                        tokens = tokens[:self.max_structure_token_len]
                    structure_tokens.extend(tokens)
            results['tokens'] = structure_tokens
            
            # Advanced bbox parsing - only if we have structure tokens and matching bboxes
            if structure_tokens and bboxes and len(bboxes) == get_bbox_nums(structure_tokens):
                empty_bbox_mask = build_empty_bbox_mask(bboxes)
                aligned_bboxes, empty_bbox_mask = align_bbox_mask(bboxes, empty_bbox_mask, structure_tokens)
                empty_bbox_mask = np.array(empty_bbox_mask)
                bbox_masks = build_bbox_mask(structure_tokens)
                bbox_masks = bbox_masks * empty_bbox_mask
                results['bboxes'] = np.array(aligned_bboxes)
                results['masks'] = bbox_masks
            else: # No structure tokens or bboxes or matching count, return empty arrays
                results['bboxes'] = np.array([])
                results['masks'] = np.array([])
            # Keep 'img' as the original table image
            
        return results

    def __repr__(self) -> str:
        repr_str = self.__class__.__name__
        repr_str += f'(with_structure={self.with_structure}, '
        repr_str += f'with_cell={self.with_cell}, '
        repr_str += f'max_structure_token_len={self.max_structure_token_len}, '
        repr_str += f'max_cell_token_len={self.max_cell_token_len})'
        return repr_str