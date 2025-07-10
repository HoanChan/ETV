# Copyright (c) Lê Hoàn Chân. All rights reserved.
from typing import Optional
import numpy as np
from mmcv.transforms import BaseTransform
from mmocr.registry import TRANSFORMS

@TRANSFORMS.register_module()
class LoadTokens(BaseTransform):
    """
    Load and process token annotations from table dataset instances.
    
    - If with_structure: returns 'img' (table image) and 'gt_tokens' (list of structure tokens).
    - If with_cell: returns 'gt_cells' as a list of dicts, each dict contains:
        {
            'img': ...,        # cell image
            'gt_tokens': ...,  # list of tokens for the cell
            'gt_bbox': ...,    # bbox of the cell
            'id': ...          # id of the cell
        }

    Full output annotation format:
        {
            'gt_tokens': [...],  # list of structure tokens if with_structure
            'gt_cells': [        # list of cell information if with_cell
                {
                    'img': ...,        # cell image
                    'gt_tokens': ...,  # list of tokens for the cell
                    'gt_bbox': ...,    # bbox of the cell
                    'id': ...          # id of the cell
                }
                ...
            ],
            'img': ...           # original table image
        }
    
    Input annotation format:
        {
            'instances': [
                {
                    'tokens': [...],
                    'task_type': 'structure' or 'content',
                    'cell_id': ...,      # only for content
                    'bbox': [...],       # only for content
                    'img': ...           # only for content if already cropped
                },
                ...
            ],
            'img': ...  # original table image
        }

    Args:
        with_structure (bool): Whether to load structure tokens. If True, will return 'img' and 'gt_tokens'.
        with_cell (bool): Whether to load cell tokens. If True, will return 'gt_cells'.
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
        - If with_structure: returns 'img' and 'gt_tokens'.
        - If with_cell: returns 'gt_cells' as a list of dicts.
        """
        if 'instances' not in results:
            if self.with_structure:
                results['gt_tokens'] = []
            if self.with_cell:
                results['gt_cells'] = []
            return results

        if self.with_structure:
            # Get structure tokens
            structure_tokens = []
            for instance in results['instances']:
                if instance.get('task_type') == 'structure':
                    tokens = instance.get('tokens', [])
                    if self.max_structure_token_len is not None:
                        tokens = tokens[:self.max_structure_token_len]
                    structure_tokens.extend(tokens)
            results['gt_tokens'] = structure_tokens
            # Keep 'img' as the original table image

        if self.with_cell:
            # Get information for each cell
            gt_cells = []
            for instance in results['instances']:
                if instance.get('task_type') == 'content':
                    tokens = instance.get('tokens', [])
                    if self.max_cell_token_len is not None:
                        tokens = tokens[:self.max_cell_token_len]
                    cell_img = instance.get('img', None)  # May be None if not cropped yet
                    bbox = instance.get('bbox', [0, 0, 0, 0])
                    cell_id = instance.get('cell_id', 0)
                    gt_cells.append({
                        'img': cell_img,
                        'gt_tokens': tokens,
                        'gt_bbox': bbox,
                        'id': cell_id
                    })
            results['gt_cells'] = gt_cells
        return results

    def __repr__(self) -> str:
        repr_str = self.__class__.__name__
        repr_str += f'(with_structure={self.with_structure}, '
        repr_str += f'with_cell={self.with_cell}, '
        repr_str += f'max_structure_token_len={self.max_structure_token_len}, '
        repr_str += f'max_cell_token_len={self.max_cell_token_len})'
        return repr_str
