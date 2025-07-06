# Copyright (c) Lê Hoàn Chân. All rights reserved.
from typing import Dict
import numpy as np


def xyxy2xywh(bboxes: np.ndarray) -> np.ndarray:
    """Convert coordinate format from (x1,y1,x2,y2) to (x,y,w,h).
    
    Where (x1,y1) is top-left, (x2,y2) is bottom-right.
    (x,y) is bbox center and (w,h) is width and height.
    
    Args:
        bboxes (ndarray): Bounding boxes in xyxy format (..., 4)
        
    Returns:
        ndarray: Bounding boxes in xywh format (..., 4)
    """
    new_bboxes = np.empty_like(bboxes)
    new_bboxes[..., 0] = (bboxes[..., 0] + bboxes[..., 2]) / 2  # x center
    new_bboxes[..., 1] = (bboxes[..., 1] + bboxes[..., 3]) / 2  # y center
    new_bboxes[..., 2] = bboxes[..., 2] - bboxes[..., 0]        # width
    new_bboxes[..., 3] = bboxes[..., 3] - bboxes[..., 1]        # height
    return new_bboxes


def normalize_bbox(bboxes: np.ndarray, img_shape: tuple) -> np.ndarray:
    """Normalize bounding boxes to [0, 1] range.
    
    Args:
        bboxes (ndarray): Bounding boxes to normalize
        img_shape (tuple): Image shape (height, width, channels)
        
    Returns:
        ndarray: Normalized bounding boxes
    """
    bboxes = bboxes.copy()
    bboxes[..., 0] = bboxes[..., 0] / img_shape[1]  # normalize x
    bboxes[..., 2] = bboxes[..., 2] / img_shape[1]  # normalize width
    bboxes[..., 1] = bboxes[..., 1] / img_shape[0]  # normalize y
    bboxes[..., 3] = bboxes[..., 3] / img_shape[0]  # normalize height
    return bboxes


def xywh2xyxy(bboxes: np.ndarray) -> np.ndarray:
    """Convert coordinate format from (x,y,w,h) to (x1,y1,x2,y2).
    
    Where (x,y) is bbox center and (w,h) is width and height.
    (x1,y1) is top-left, (x2,y2) is bottom-right.
    
    Args:
        bboxes (ndarray): Bounding boxes in xywh format (..., 4)
        
    Returns:
        ndarray: Bounding boxes in xyxy format (..., 4)
    """
    new_bboxes = np.empty_like(bboxes)
    new_bboxes[..., 0] = bboxes[..., 0] - bboxes[..., 2] / 2  # x1
    new_bboxes[..., 1] = bboxes[..., 1] - bboxes[..., 3] / 2  # y1
    new_bboxes[..., 2] = bboxes[..., 0] + bboxes[..., 2] / 2  # x2
    new_bboxes[..., 3] = bboxes[..., 1] + bboxes[..., 3] / 2  # y2
    return new_bboxes
