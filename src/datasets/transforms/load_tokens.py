# Copyright (c) Lê Hoàn Chân. All rights reserved.
from typing import Optional
import numpy as np
from mmcv.transforms import BaseTransform

# from mmocr.registry import TRANSFORMS
from mmengine.registry import TRANSFORMS


@TRANSFORMS.register_module()
class LoadTokens(BaseTransform):
    """Load and process token annotations from table dataset instances.
    
    This transform extracts tokens from the 'instances' annotation provided by
    PubTabNetDataset. It handles both structure tokens and cell content tokens.

    The annotation format is as the following:

    .. code-block:: python

        {
            'instances': [
                {
                    'tokens': ['<thead>', 'tr', 'td', '</td>', ...],  # Structure or cell tokens
                    'task_type': 'structure',  # or 'content'
                    'cell_id': 0,  # Only for content instances
                    'bbox': [x1, y1, x2, y2],  # Only for content instances with bbox
                },
                ...
            ]
        }

    After this module, the annotation has been changed to the format below:

    .. code-block:: python

        {
            # Structure tokens (if present)
            'gt_structure_tokens': list[str],
            
            # Cell content tokens (if present)
            'gt_cell_tokens': list[list[str]],
            'gt_cell_ids': list[int],
            'gt_cell_bboxes': np.ndarray(N, 4),  # Optional, only if bbox present
            
            # Task types for each instance
            'gt_task_types': list[str],
            
            # Combined tokens for backward compatibility
            'gt_tokens': list[str] or list[list[str]],
        }

    Required Keys:

    - instances
      - tokens
      - task_type (optional, defaults to 'both')
      - cell_id (optional, for content instances)
      - bbox (optional, for content instances)

    Added Keys:

    - gt_structure_tokens (list[str])
    - gt_cell_tokens (list[list[str]])
    - gt_cell_ids (list[int])
    - gt_cell_bboxes (np.ndarray, optional)
    - gt_task_types (list[str])
    - gt_tokens (list[str] or list[list[str]])

    Args:
        with_structure (bool): Whether to load structure tokens.
            Defaults to True.
        with_content (bool): Whether to load cell content tokens.
            Defaults to True.
        with_bbox (bool): Whether to load bounding box information for cells.
            Defaults to False.
        max_structure_len (int, optional): Maximum length for structure tokens.
            If None, no truncation is applied. Defaults to None.
        max_cell_len (int, optional): Maximum length for cell tokens.
            If None, no truncation is applied. Defaults to None.
        flatten_tokens (bool): Whether to flatten all tokens into a single list.
            If False, maintains separation between structure and cell tokens.
            Defaults to False.
    """

    def __init__(self,
                 with_structure: bool = True,
                 with_content: bool = True,
                 with_bbox: bool = False,
                 max_structure_len: Optional[int] = None,
                 max_cell_len: Optional[int] = None,
                 flatten_tokens: bool = False,
                 **kwargs) -> None:
        super().__init__()
        
        assert with_structure or with_content, "At least one of with_structure or with_content must be True"
        
        self.with_structure = with_structure
        self.with_content = with_content
        self.with_bbox = with_bbox
        self.max_structure_len = max_structure_len
        self.max_cell_len = max_cell_len
        self.flatten_tokens = flatten_tokens

    def _load_structure_tokens(self, results: dict) -> None:
        """Private function to load structure token annotations.

        Args:
            results (dict): Result dict from table dataset.
        """
        structure_tokens = []
        
        for instance in results['instances']:
            if instance.get('task_type') == 'structure':
                tokens = instance.get('tokens', [])
                if self.max_structure_len is not None:
                    tokens = tokens[:self.max_structure_len]
                structure_tokens.extend(tokens)
        
        results['gt_structure_tokens'] = structure_tokens

    def _load_content_tokens(self, results: dict) -> None:
        """Private function to load cell content token annotations.

        Args:
            results (dict): Result dict from table dataset.
        """
        cell_tokens = []
        cell_ids = []
        cell_bboxes = []
        
        for instance in results['instances']:
            if instance.get('task_type') == 'content':
                tokens = instance.get('tokens', [])
                if self.max_cell_len is not None:
                    tokens = tokens[:self.max_cell_len]
                
                cell_tokens.append(tokens)
                cell_ids.append(instance.get('cell_id', 0))
                
                if self.with_bbox and 'bbox' in instance:
                    bbox = instance['bbox']
                    if len(bbox) == 4:
                        cell_bboxes.append(bbox)
                    else:
                        # Add empty bbox if invalid
                        cell_bboxes.append([0, 0, 0, 0])
                elif self.with_bbox:
                    # Add empty bbox if not present but required
                    cell_bboxes.append([0, 0, 0, 0])
        
        results['gt_cell_tokens'] = cell_tokens
        results['gt_cell_ids'] = cell_ids
        
        if self.with_bbox and cell_bboxes:
            results['gt_cell_bboxes'] = np.array(cell_bboxes, dtype=np.float32)

    def _load_task_types(self, results: dict) -> None:
        """Private function to load task type annotations.

        Args:
            results (dict): Result dict from table dataset.
        """
        task_types = []
        
        for instance in results['instances']:
            task_type = instance.get('task_type', 'both')
            task_types.append(task_type)
        
        results['gt_task_types'] = task_types

    def _combine_tokens(self, results: dict) -> None:
        """Private function to combine tokens for backward compatibility.

        Args:
            results (dict): Result dict from table dataset.
        """
        combined_tokens = []
        
        if self.flatten_tokens:
            # Flatten all tokens into a single list
            if 'gt_structure_tokens' in results:
                combined_tokens.extend(results['gt_structure_tokens'])
            
            if 'gt_cell_tokens' in results:
                for cell_tokens in results['gt_cell_tokens']:
                    combined_tokens.extend(cell_tokens)
            
            results['gt_tokens'] = combined_tokens
        else:
            # Keep structure separate from content
            if self.with_structure and self.with_content:
                results['gt_tokens'] = {
                    'structure': results.get('gt_structure_tokens', []),
                    'content': results.get('gt_cell_tokens', [])
                }
            elif self.with_structure:
                results['gt_tokens'] = results.get('gt_structure_tokens', [])
            elif self.with_content:
                results['gt_tokens'] = results.get('gt_cell_tokens', [])

    def transform(self, results: dict) -> dict:
        """Function to load token annotations.

        Args:
            results (dict): Result dict from table dataset.

        Returns:
            dict: The dict contains loaded token annotations.
        """
        if 'instances' not in results:
            # Initialize empty annotations if no instances
            if self.with_structure:
                results['gt_structure_tokens'] = []
            if self.with_content:
                results['gt_cell_tokens'] = []
                results['gt_cell_ids'] = []
                if self.with_bbox:
                    results['gt_cell_bboxes'] = np.array([], dtype=np.float32).reshape(0, 4)
            results['gt_task_types'] = []
            results['gt_tokens'] = []
            return results

        # Load different types of annotations
        if self.with_structure:
            self._load_structure_tokens(results)
        
        if self.with_content:
            self._load_content_tokens(results)
        
        # Always load task types for reference
        self._load_task_types(results)
        
        # Combine tokens for backward compatibility
        self._combine_tokens(results)
        
        return results

    def __repr__(self) -> str:
        """String representation of the transform."""
        repr_str = self.__class__.__name__
        repr_str += f'(with_structure={self.with_structure}, '
        repr_str += f'with_content={self.with_content}, '
        repr_str += f'with_bbox={self.with_bbox}, '
        repr_str += f'max_structure_len={self.max_structure_len}, '
        repr_str += f'max_cell_len={self.max_cell_len}, '
        repr_str += f'flatten_tokens={self.flatten_tokens})'
        return repr_str
