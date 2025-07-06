# Copyright (c) Lê Hoàn Chân. All rights reserved.
from typing import Dict, Optional, Union, Tuple
import random
import cv2
import numpy as np
from mmcv.transforms.base import BaseTransform
from mmocr.registry import TRANSFORMS


@TRANSFORMS.register_module()
class TableResize(BaseTransform):
    """Image resizing transform for Table Recognition OCR and Table Structure Recognition.
    
    This transform resizes images while optionally maintaining aspect ratio and handling
    bounding box coordinates for table structure recognition tasks.
    
    Required Keys:
        - img (ndarray): Input image
        
    Modified Keys:
        - img (ndarray): Resized image
        - img_shape (tuple): Shape of resized image
        - pad_shape (tuple): Same as img_shape after resize
        - scale_factor (tuple): Scale factor applied (height_scale, width_scale)
        
    Added Keys:
        - keep_ratio (bool): Whether aspect ratio was kept
        
    Args:
        img_scale (tuple, optional): Target image scale (width, height).
            If -1 is used for width or height, it will be calculated based on aspect ratio.
        min_size (int, optional): Minimum size for the shorter side.
        ratio_range (list, optional): Random ratio range for augmentation [min_ratio, max_ratio].
        interpolation (int): Interpolation method. Defaults to cv2.INTER_LINEAR.
        keep_ratio (bool): Whether to keep aspect ratio. Defaults to True.
        long_size (int, optional): Target size for the longer side.
    """

    def __init__(self,
                 img_scale: Optional[Tuple[int, int]] = None,
                 min_size: Optional[int] = None,
                 ratio_range: Optional[list] = None,
                 interpolation: int = cv2.INTER_LINEAR,
                 keep_ratio: bool = True,
                 long_size: Optional[int] = None):
        super().__init__()
        self.img_scale = img_scale
        self.min_size = min_size
        self.ratio_range = ratio_range
        self.interpolation = interpolation
        self.long_size = long_size
        self.keep_ratio = keep_ratio

    def _get_resize_scale(self, w: int, h: int) -> Tuple[int, int]:
        """Calculate target resize scale based on current dimensions and parameters.
        
        Args:
            w (int): Current width
            h (int): Current height
            
        Returns:
            tuple: Target (width, height) for resize
        """
        if self.keep_ratio:
            if self.img_scale is None and isinstance(self.ratio_range, list):
                choice_ratio = random.uniform(self.ratio_range[0], self.ratio_range[1])
                return (int(w * choice_ratio), int(h * choice_ratio))
            elif isinstance(self.img_scale, tuple) and -1 in self.img_scale:
                if self.img_scale[0] == -1:
                    resize_w = w / h * self.img_scale[1]
                    return (int(resize_w), self.img_scale[1])
                else:
                    resize_h = h / w * self.img_scale[0]
                    return (self.img_scale[0], int(resize_h))
            else:
                return (int(w), int(h))
        else:
            if isinstance(self.img_scale, tuple):
                return self.img_scale
            else:
                raise NotImplementedError("img_scale must be a tuple when keep_ratio=False")

    def _resize_bboxes(self, results: Dict) -> None:
        """Resize bounding boxes according to the scale factor.
        
        Args:
            results (dict): Results dict containing bboxes and scale factor
        """
        img_shape = results['img_shape']
        if 'img_info' in results.keys():
            # train and validate phase
            if results['img_info'].get('bbox', None) is not None:
                bboxes = results['img_info']['bbox']
                scale_factor = results['scale_factor']
                # Apply scale and clip to image boundaries
                bboxes[..., 0::2] = np.clip(bboxes[..., 0::2] * scale_factor[1], 0, img_shape[1]-1)
                bboxes[..., 1::2] = np.clip(bboxes[..., 1::2] * scale_factor[0], 0, img_shape[0]-1)
                results['img_info']['bbox'] = bboxes
            else:
                raise ValueError('results should have bbox keys.')
        # testing phase - no bbox to resize

    def _resize_img(self, results: Dict) -> None:
        """Resize the image according to the specified parameters.
        
        Args:
            results (dict): Results dict containing the image
        """
        img = results['img']
        h, w = img.shape[:2]

        # Apply min_size constraint
        if self.min_size is not None:
            if w > h:
                w = self.min_size / h * w
                h = self.min_size
            else:
                h = self.min_size / w * h
                w = self.min_size

        # Apply long_size constraint
        if self.long_size is not None:
            if w < h:
                w = self.long_size / h * w
                h = self.long_size
            else:
                h = self.long_size / w * h
                w = self.long_size

        img_scale = self._get_resize_scale(w, h)
        resize_img = cv2.resize(img, img_scale, interpolation=self.interpolation)
        scale_factor = (resize_img.shape[0] / img.shape[0], resize_img.shape[1] / img.shape[1])

        results['img'] = resize_img
        results['img_shape'] = resize_img.shape
        results['pad_shape'] = resize_img.shape
        results['scale_factor'] = scale_factor
        results['keep_ratio'] = self.keep_ratio

    def transform(self, results: Dict) -> Dict:
        """Transform function to resize image and bboxes.
        
        Args:
            results (dict): Result dict from loading pipeline.
            
        Returns:
            dict: Updated result dict with resized image and bboxes.
        """
        self._resize_img(results)
        self._resize_bboxes(results)
        return results

    def __repr__(self) -> str:
        repr_str = self.__class__.__name__
        repr_str += f'(img_scale={self.img_scale}, '
        repr_str += f'min_size={self.min_size}, '
        repr_str += f'ratio_range={self.ratio_range}, '
        repr_str += f'keep_ratio={self.keep_ratio}, '
        repr_str += f'long_size={self.long_size})'
        return repr_str
