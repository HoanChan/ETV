# Copyright (c) Lê Hoàn Chân. All rights reserved.
from typing import Dict
import numpy as np
from mmcv.transforms.base import BaseTransform
from mmocr.registry import TRANSFORMS
from .transforms_utils import xyxy2xywh, normalize_bbox


@TRANSFORMS.register_module()
class BboxEncode(BaseTransform):
    """
    Transform to encode and normalize table bounding boxes for training.

    This transform performs the following:
        1. Converts bounding box coordinates from (x1, y1, x2, y2) format to (x, y, w, h) format.
        2. Normalizes the bounding box coordinates to the [0, 1] range based on the image shape.

    Expected input:
        - data_samples (TableMasterDataSample): Contains 'bboxes' (ndarray, in xyxy format) and 'img_shape'.

    Output/Modifications:
        - Updates 'bboxes' in data_samples to normalized xywh format.
        - Sets 'have_normalized_bboxes' flag in data_samples metainfo.
        - Prints a warning if any bounding box is out of the [0, 1] range.
    """

    def __init__(self):
        super().__init__()

    def transform(self, results: Dict) -> Dict:
        """
        Encode and normalize bounding boxes in the input results.

        Args:
            results (dict): Dictionary containing bounding box and image shape information.

        Returns:
            dict: Updated results dictionary with normalized bounding boxes in xywh format.
        """
            
        bboxes = results.get('bboxes')
        assert bboxes is not None, "Input 'bboxes' is missing in results."
            
        # If bboxes is List of Lists, convert to numpy array
        if isinstance(bboxes, list) and all(isinstance(bbox, list) for bbox in bboxes):
            bboxes = np.array(bboxes, dtype=np.float32)
        # Ensure bboxes are float32
        if bboxes.dtype != np.float32:
            bboxes = bboxes.astype(np.float32)
        # Convert from xyxy to xywh format
        bboxes = xyxy2xywh(bboxes)
        # Normalize to [0,1] range
        img_shape = results.get('img_shape')
        bboxes = normalize_bbox(bboxes, img_shape)
        
        # Validate bboxes are in valid range
        assert self._check_bbox_valid(bboxes), f"Box invalid in {results.get('filename', 'unknown')}:{bboxes}"
        
        # Update results
        results['bboxes'] = bboxes

        return results

    def _check_bbox_valid(self, bboxes: np.ndarray) -> bool:
        """
        Check if all bounding box coordinates are within the valid [0, 1] range and each bbox has 4 values.

        Args:
            bboxes (ndarray): Normalized bounding boxes to validate.

        Returns:
            bool: True if all bounding boxes are valid, False otherwise.
        """
        if bboxes.ndim != 2 or bboxes.shape[1] != 4:
            return False
        return np.all((bboxes >= 0) & (bboxes <= 1))

    def __repr__(self) -> str:
        repr_str = self.__class__.__name__
        return repr_str
