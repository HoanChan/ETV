# Copyright (c) Lê Hoàn Chân. All rights reserved.
from typing import Dict, Optional
import numpy as np
from PIL import Image
from mmcv.transforms.base import BaseTransform
from mmocr.registry import TRANSFORMS

@TRANSFORMS.register_module()
class GetCells(BaseTransform):
    """Transform để cắt các cell từ ảnh bảng dựa vào bbox của từng cell.
    
    Cắt từng cell từ ảnh bảng dựa vào bbox, trả về list ảnh cell tương ứng.
    Đảm bảo tính tương thích với pipeline mmOCR và xử lý edge cases.
    
    Transform này có thể hoạt động ở hai chế độ:
    1. Sử dụng output chuẩn hóa từ LoadTokens (gt_cell_bboxes) - ưu tiên
    2. Fallback về raw instances nếu chưa có LoadTokens
    
    Required Keys:
        - img (ndarray): Ảnh gốc
        - gt_cell_bboxes (ndarray, optional): Bbox đã chuẩn hóa từ LoadTokens
        - instances (list[dict], optional): Raw instances chứa bbox và tokens

    Modified Keys:
        - None
        
    Added Keys:
        - cell_imgs (list[ndarray]): List các ảnh cell đã cắt
    
    Args:
        img_key (str): Key của ảnh trong results dict. Defaults to 'img'.
        instances_key (str): Key của instances trong results dict. Defaults to 'instances'.
        task_filter (str, optional): Chỉ xử lý instances có task_type này khi fallback.
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
        
        # Try to use processed annotations from LoadTokens first
        if 'gt_cell_bboxes' in results and len(results['gt_cell_bboxes']) > 0:
            # Use standardized output from LoadTokens
            bboxes = results['gt_cell_bboxes']
            for bbox in bboxes:
                self._extract_cell_from_bbox(bbox, img, img_w, img_h, cell_imgs)
        else:
            # Fallback to raw instances
            instances = results.get(self.instances_key, [])
            
            for inst in instances:
                # Filter by task type if specified
                if self.task_filter is not None:
                    if inst.get('task_type') != self.task_filter:
                        continue
                        
                bbox = inst.get('bbox', None)
                
                # Skip if missing required fields
                if bbox is None:
                    continue
                    
                # Validate bbox format
                if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
                    continue
                    
                self._extract_cell_from_bbox(bbox, img, img_w, img_h, cell_imgs)
        
        # Add results
        results['cell_imgs'] = cell_imgs
        
        return results

    def _extract_cell_from_bbox(self, bbox, img, img_w, img_h, cell_imgs):
        """Extract cell image from bbox and add to cell_imgs list.
        
        Args:
            bbox: Bounding box coordinates [x0, y0, x1, y1]
            img: Source image array
            img_w: Image width
            img_h: Image height
            cell_imgs: List to append extracted cell image to
        """
        try:
            x0, y0, x1, y1 = map(int, bbox)
        except (ValueError, TypeError):
            return
            
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
            return
            
        # Filter out cells where both dimensions are too small
        if cell_w < self.min_cell_size and cell_h < self.min_cell_size:
            return
            
        # Extract cell image
        cell_img = img[y0:y1, x0:x1].copy()
        
        cell_imgs.append(cell_img)
        
    def __repr__(self) -> str:
        repr_str = self.__class__.__name__
        repr_str += f'(img_key={self.img_key}, '
        repr_str += f'instances_key={self.instances_key}, '
        repr_str += f'task_filter={self.task_filter}, '
        repr_str += f'min_cell_size={self.min_cell_size})'
        return repr_str