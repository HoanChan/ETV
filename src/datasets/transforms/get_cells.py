from typing import Dict, List, Optional
import numpy as np
from PIL import Image
from mmcv.transforms.base import BaseTransform

# from mmocr.registry import TRANSFORMS
from mmengine.registry import TRANSFORMS


@TRANSFORMS.register_module()
class GetCell(BaseTransform):
    """Transform để cắt các cell từ ảnh bảng dựa vào bbox của từng cell.
    
    Cắt từng cell từ ảnh bảng dựa vào bbox, trả về list ảnh cell tương ứng.
    Đảm bảo tính tương thích với pipeline mmOCR và xử lý edge cases.
    
    Required Keys:
        - img (ndarray): Ảnh gốc
        - instances (list[dict]): List các instance chứa bbox và tokens

    Modified Keys:
        - None
        
    Added Keys:
        - cell_imgs (list[ndarray]): List các ảnh cell đã cắt
        - cell_tokens (list[str]): List ground truth tokens tương ứng
        - cell_bboxes (list[list]): List bbox đã sử dụng
    
    Args:
        img_key (str): Key của ảnh trong results dict. Defaults to 'img'.
        instances_key (str): Key của instances trong results dict. Defaults to 'instances'.
        task_filter (str, optional): Chỉ xử lý instances có task_type này. 
            Nếu None, xử lý tất cả instances. Defaults to 'content'.
        min_cell_size (int): Kích thước tối thiểu của cell (width hoặc height).
            Defaults to 5.
    """

    def __init__(self,
                 img_key: str = 'img',
                 instances_key: str = 'instances', 
                 task_filter: Optional[str] = 'content',
                 min_cell_size: int = 5):
        super().__init__()
        self.img_key = img_key
        self.instances_key = instances_key
        self.task_filter = task_filter
        self.min_cell_size = min_cell_size

    def transform(self, results: Dict) -> Dict:
        """Transform function để cắt cell từ ảnh.
        
        Args:
            results (dict): Result dict từ loading pipeline.
            
        Returns:
            dict: Updated result dict với các cell đã cắt.
        """
        img = results[self.img_key]
        if isinstance(img, Image.Image):
            img = np.array(img)
            
        # Validate image
        if len(img.shape) < 2:
            raise ValueError(f"Invalid image shape: {img.shape}")
            
        img_h, img_w = img.shape[:2]
        
        cell_imgs = []
        cell_tokens = []
        cell_bboxes = []
        
        instances = results.get(self.instances_key, [])
        
        for inst in instances:
            # Filter by task type if specified
            if self.task_filter is not None:
                if inst.get('task_type') != self.task_filter:
                    continue
                    
            bbox = inst.get('bbox', None)
            tokens = inst.get('tokens', None)
            
            # Skip if missing required fields
            if bbox is None or tokens is None:
                continue
                
            # Validate bbox format
            if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
                continue
                
            try:
                x0, y0, x1, y1 = map(int, bbox)
            except (ValueError, TypeError):
                continue
                
            # Validate bbox coordinates
            x0, x1 = sorted([x0, x1])  # Ensure x0 <= x1
            y0, y1 = sorted([y0, y1])  # Ensure y0 <= y1
            
            # Clip to image boundaries
            x0 = max(0, min(x0, img_w - 1))
            y0 = max(0, min(y0, img_h - 1))
            x1 = max(x0 + 1, min(x1, img_w))
            y1 = max(y0 + 1, min(y1, img_h))
            
            # Check minimum cell size and zero area
            cell_w = x1 - x0
            cell_h = y1 - y0
            
            # Filter out zero-area cells (original dimensions were zero)
            orig_w = abs(bbox[2] - bbox[0]) 
            orig_h = abs(bbox[3] - bbox[1])
            if orig_w == 0 or orig_h == 0:
                continue
                
            # Filter out cells where both dimensions are too small
            if cell_w < self.min_cell_size and cell_h < self.min_cell_size:
                continue
                
            # Extract cell image
            cell_img = img[y0:y1, x0:x1].copy()
            
            cell_imgs.append(cell_img)
            cell_tokens.append(tokens)
            cell_bboxes.append([x0, y0, x1, y1])
        
        # Add results
        results['cell_imgs'] = cell_imgs
        results['cell_tokens'] = cell_tokens
        results['cell_bboxes'] = cell_bboxes
        
        return results
        
    def __repr__(self) -> str:
        repr_str = self.__class__.__name__
        repr_str += f'(img_key={self.img_key}, '
        repr_str += f'instances_key={self.instances_key}, '
        repr_str += f'task_filter={self.task_filter}, '
        repr_str += f'min_cell_size={self.min_cell_size})'
        return repr_str