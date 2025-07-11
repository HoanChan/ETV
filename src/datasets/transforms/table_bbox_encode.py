# Copyright (c) Lê Hoàn Chân. All rights reserved.
from typing import Dict
import os
import cv2
import numpy as np
from mmcv.transforms.base import BaseTransform
from mmocr.registry import TRANSFORMS
from .transforms_utils import xyxy2xywh, normalize_bbox, xywh2xyxy


@TRANSFORMS.register_module()
class TableBboxEncode(BaseTransform):
    """Encode table bboxes for training.
    
    This transform:
    1. Converts bbox coordinates from (x1,y1,x2,y2) to (x,y,w,h) format
    2. Normalizes coordinates to [0,1] range
    3. Adjusts key locations in the results dict for compatibility
    
    Required Keys:
        - img (ndarray): Input image for shape reference
        - img_info (dict): Dictionary containing bbox information
            - bbox (ndarray): Bounding boxes in xyxy format
            - bbox_masks (ndarray): Bbox masks
            
    Modified Keys:
        - img_info (dict): bbox and bbox_masks keys are removed
        
    Added Keys:
        - bbox (ndarray): Normalized bboxes in xywh format
        - bbox_masks (ndarray): Bbox masks moved from img_info
    """

    def __init__(self):
        super().__init__()

    def transform(self, results: Dict) -> Dict:
        """Transform function to encode bboxes.
        
        Args:
            results (dict): Result dict from loading pipeline.
            
        Returns:
            dict: Updated result dict with encoded bboxes.
        """
        # Get bboxes from img_info
        bboxes = results['img_info']['bbox']
        
        # Ensure bboxes are float32
        if bboxes.dtype != np.float32:
            bboxes = bboxes.astype(np.float32)
        
        # Convert from xyxy to xywh format
        bboxes = xyxy2xywh(bboxes)
        
        # Normalize to [0,1] range
        img_shape = results['img'].shape
        bboxes = normalize_bbox(bboxes, img_shape)
        
        # Validate bboxes are in valid range
        is_valid = self._check_bbox_valid(bboxes)
        if not is_valid:
            filename = results.get('filename', 'unknown')
            print(f'Box invalid in {filename}')
        
        # Update results
        results['img_info']['bbox'] = bboxes
        self._adjust_key_locations(results)
        
        return results

    def _check_bbox_valid(self, bboxes: np.ndarray) -> bool:
        """Check if all bboxes are within valid range [0,1].
        
        Args:
            bboxes (ndarray): Normalized bboxes to validate
            
        Returns:
            bool: True if all bboxes are valid, False otherwise
        """
        # Check if all values are between 0 and 1
        low_valid = (bboxes >= 0.).astype(int)
        high_valid = (bboxes <= 1.).astype(int)
        validity_matrix = low_valid + high_valid
        
        for idx, bbox_validity in enumerate(validity_matrix):
            # Each bbox should have 8 valid checks (4 coords * 2 bounds)
            if bbox_validity.sum() != 8:
                return False
        return True

    def _adjust_key_locations(self, results: Dict) -> None:
        """Adjust key locations in results dict for compatibility.
        
        Move 'bbox' and 'bbox_masks' from 'img_info' to top level.
        
        Args:
            results (dict): Results dict to modify
        """
        # Move bbox and bbox_masks to top level
        bboxes = results['img_info'].pop('bbox')
        bbox_masks = results['img_info'].pop('bbox_masks')
        
        results['bbox'] = bboxes
        results['bbox_masks'] = bbox_masks

    def _visualize_normalized_bbox(self, results: Dict, save_dir: str = '/data_0/cache') -> None:
        """Visualize normalized bboxes on image for debugging.
        
        Args:
            results (dict): Results dict containing normalized bboxes
            save_dir (str): Directory to save visualization
        """
        filename = results.get('filename', 'unknown')
        base_name = os.path.basename(filename).split('.')[0]
        save_path = os.path.join(save_dir, f'{base_name}_normalized.jpg')
        
        img = results['img'].copy()
        img_shape = img.shape
        
        # Get normalized bboxes (x,y,w,h format)
        bboxes = results['bbox'].copy()
        
        # Denormalize to pixel coordinates
        bboxes[..., 0::2] = bboxes[..., 0::2] * img_shape[1]  # x, w
        bboxes[..., 1::2] = bboxes[..., 1::2] * img_shape[0]  # y, h
        
        # Convert xywh to xyxy for drawing
        xyxy_bboxes = xywh2xyxy(bboxes)
        
        # Draw bboxes
        for bbox in xyxy_bboxes:
            x1, y1, x2, y2 = bbox.astype(int)
            img = cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), thickness=1)
        
        # Ensure save directory exists
        os.makedirs(save_dir, exist_ok=True)
        cv2.imwrite(save_path, img)

    def __repr__(self) -> str:
        repr_str = self.__class__.__name__
        return repr_str
