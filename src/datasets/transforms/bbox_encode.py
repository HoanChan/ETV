# Copyright (c) Lê Hoàn Chân. All rights reserved.
from typing import Dict
import os
import cv2
import numpy as np
from mmcv.transforms.base import BaseTransform
from mmocr.registry import TRANSFORMS

from structures.table_master_data_sample import TableMasterDataSample
from .transforms_utils import xyxy2xywh, normalize_bbox, xywh2xyxy


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
            results (dict): Dictionary containing 'data_samples' with bounding box and image shape information.

        Returns:
            dict: Updated results dictionary with normalized bounding boxes in xywh format.
        """
        data_sample = results.get('data_samples', None)
        assert isinstance(data_sample, TableMasterDataSample), f"data_sample should be an instance of TableMasterDataSample, but got {type(data_sample)}"
        if data_sample.get('have_normalized_bboxes', False):
            pass
        else:
            bboxes = data_sample.get('bboxes')
            # If bboxes is List of Lists, convert to numpy array
            if isinstance(bboxes, list) and all(isinstance(bbox, list) for bbox in bboxes):
                bboxes = np.array(bboxes, dtype=np.float32)
            # Ensure bboxes are float32
            if bboxes.dtype != np.float32:
                bboxes = bboxes.astype(np.float32)
            # Convert from xyxy to xywh format
            bboxes = xyxy2xywh(bboxes)
            # Normalize to [0,1] range
            img_shape = data_sample.get('img_shape')
            bboxes = normalize_bbox(bboxes, img_shape)
            
            # Validate bboxes are in valid range
            is_valid = self._check_bbox_valid(bboxes)
            if not is_valid:
                filename = results.get('filename', 'unknown')
                print(f'Box invalid in {filename}')
            
            # Update results
            data_sample.set_metainfo({'bboxes': bboxes})
            data_sample.set_metainfo({'have_normalized_bboxes':True})

        return results

    def _check_bbox_valid(self, bboxes: np.ndarray) -> bool:
        """
        Check if all bounding box coordinates are within the valid [0, 1] range.

        Args:
            bboxes (ndarray): Normalized bounding boxes to validate.

        Returns:
            bool: True if all bounding boxes are valid, False otherwise.
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

    def __repr__(self) -> str:
        repr_str = self.__class__.__name__
        return repr_str
