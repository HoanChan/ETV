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

    Full output annotation format (if with_structure and with_cell):
        {
            'img': ...               # original table image
            'tokens': [...],         # list of structure tokens if with_structure
            'bboxes': [...],         # list of all token bboxes if with_structure
            'masks': [...],          # list of all masks for bboxes if with_structure
            'cells': [               # list of cell information if with_cell
                {
                    'tokens': ...,   # list of tokens for the cell
                    'bbox': ...,     # bbox of the cell
                    'id': ...        # id of the cell
                }
                ...
            ],
        }

    Input annotation format:
        {
            'img': ...               # original table image
            'instances': [
                {
                    'tokens': [...], # list of tokens for the instance (table or cell)
                    'task_type': ... # 'structure' or 'content',
                    'cell_id': ...,  # only for content
                    'bbox': [...],   # only for content
                    'img': ...       # only for content if already cropped
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
        structures = [inst for inst in results['instances'] if inst.get('type') == 'structure']
        cells = [inst for inst in results['instances'] if inst.get('type') == 'content']
        for instance in structures:
            tokens = instance.get('tokens', [])
            tokens = remove_thead_Bb(tokens) # Remove <b> and </b> tags from header cells
            tokens = process_token(tokens, cells) # Merge the common tokens, insert empty bbox tokens if cell dont have bbox
            instance['tokens'] = tokens
        
        # Get structure tokens
        structure_tokens = []
        for instance in structures:
            tokens = instance.get('tokens', [])
            if self.max_structure_token_len is not None:
                tokens = tokens[:self.max_structure_token_len]
            structure_tokens.extend(tokens)
            
        if self.with_structure:
            results['tokens'] = structure_tokens

        bbox_count_in_structure = get_bbox_nums(structure_tokens)

        # Get information for each cell and build bboxes for structure, empty cells will have bbox [0, 0, 0, 0]
        cells_result = []
        bboxes = []
        idxs = [s.get('cell_id') for s in cells if 'cell_id' in s]
        assert len(set(idxs)) == len(idxs), f"cell_id must be unique in cells, id: {idxs}"
        for i in range(bbox_count_in_structure):
            if i in idxs: # If cell_id exists in cells, use it
                # Find the cell instance with cell_id == i
                cell_instance = None
                for cell in cells:
                    if cell.get('cell_id') == i:
                        cell_instance = cell
                        break
                
                if cell_instance is not None:
                    tokens = cell_instance.get('tokens', [])
                    if self.max_cell_token_len is not None:
                        tokens = tokens[:self.max_cell_token_len]
                    bbox = cell_instance.get('bbox', [0, 0, 0, 0])
                    cell_id = cell_instance.get('cell_id', 0)
                    cells_result.append({
                        'tokens': tokens,
                        'bbox': bbox,
                        'id': cell_id
                    })
            else: # If cell_id does not exist, use empty bbox
                bbox = [0, 0, 0, 0]
            bboxes.append(bbox)
        
        if self.with_cell:
            results['cells'] = cells_result

        if self.with_structure:
            
            # Advanced bbox parsing - only if we have structure tokens and matching bboxes

            if bbox_count_in_structure > 0:
                empty_bbox_mask = build_empty_bbox_mask(bboxes)
                aligned_bboxes, empty_bbox_mask = align_bbox_mask(bboxes, empty_bbox_mask, structure_tokens)
                empty_bbox_mask = empty_bbox_mask
                bbox_masks = build_bbox_mask(structure_tokens)
                bbox_masks = bbox_masks * empty_bbox_mask
                results['bboxes'] = aligned_bboxes
                results['masks'] = bbox_masks
            else: # No structure tokens or bboxes or matching count, return empty arrays
                results['bboxes'] = []
                results['masks'] = []
            # Keep 'img' as the original table image
            
        return results

    def __repr__(self) -> str:
        repr_str = self.__class__.__name__
        repr_str += f'(with_structure={self.with_structure}, '
        repr_str += f'with_cell={self.with_cell}, '
        repr_str += f'max_structure_token_len={self.max_structure_token_len}, '
        repr_str += f'max_cell_token_len={self.max_cell_token_len})'
        return repr_str