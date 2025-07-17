# Copyright (c) Lê Hoàn Chân. All rights reserved.
from typing import Dict, Optional
from mmocr.registry import TRANSFORMS
from mmocr.datasets.transforms import Resize  # Import MMOCR's Resize class

@TRANSFORMS.register_module()
class TableResize(Resize):
    """Image resizing transform for Table Recognition with min_size and long_size constraints.
    
    Extends MMOCR's Resize to add table-specific size constraints.
    
    Args:
        min_size (int, optional): Minimum size for the shorter side.
        long_size (int, optional): Target size for the longer side. if provided, the min_size will be ignored.
        **kwargs: Arguments passed to parent Resize class.
    """

    def __init__(self,
                 min_size: Optional[int] = None,
                 long_size: Optional[int] = None,
                 **kwargs):
        # Nếu không có scale/scale_factor thì truyền scale=(1,1) cho lớp cha để tránh lỗi
        if 'scale' not in kwargs and 'scale_factor' not in kwargs:
            kwargs['scale'] = (1, 1)
        kwargs['keep_ratio'] = True  # Ensure keep_ratio is set to True
        super().__init__(**kwargs)
        self.min_size = min_size
        self.long_size = long_size

    def _resize_img(self, results: Dict) -> None:
        """Apply size constraints then use parent's resize logic."""
        if results.get('img', None) is not None:
            img = results['img']
            h, w = img.shape[:2]
            scale = 1.0

            # Apply long_size constraint
            if self.long_size is not None:
                max_side = max(w, h)
                if max_side != self.long_size:
                    scale = self.long_size / max_side
            elif self.min_size is not None:  # Apply min_size constraint
                min_side = min(w, h)
                if min_side < self.min_size:
                    scale = self.min_size / min_side

            new_w = int(w * scale)
            new_h = int(h * scale)

            # Set scale and use parent's logic
            results['scale'] = (new_w, new_h)
            
        super()._resize_img(results)

    def _resize_bboxes(self, results: Dict) -> None:
        """Resize bboxes using parent's logic, with special handling for img_info structure."""
        # Handle special case for img_info structure (from old implementation)
        if 'bboxes' in results.keys() and results['bboxes'] is not None:
            results['gt_bboxes'] = results['bboxes'] # Move bboxes to gt_bboxes temporarily for parent processing
            super()._resize_bboxes(results) # Use parent's resize logic
            results['bboxes'] = results['gt_bboxes'] # Move back to structure
            results.pop('gt_bboxes', None) # Clean up temporary gt_bboxes
        else:
            super()._resize_bboxes(results)

    def __repr__(self) -> str:
        repr_str = self.__class__.__name__
        repr_str += f'(min_size={self.min_size}, long_size={self.long_size})'
        return repr_str