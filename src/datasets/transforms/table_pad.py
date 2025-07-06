# Copyright (c) Lê Hoàn Chân. All rights reserved.
from typing import Dict, Optional, Union, Tuple
import cv2
import numpy as np
from mmcv.transforms.base import BaseTransform
from mmocr.registry import TRANSFORMS


@TRANSFORMS.register_module()
class TablePad(BaseTransform):
    """Pad the image to a fixed size or divisible size for table recognition.
    
    This transform supports two padding modes:
    1. Pad to fixed size
    2. Pad to minimum size that is divisible by some number
    
    Required Keys:
        - img (ndarray): Input image
        
    Modified Keys:
        - img (ndarray): Padded image
        - pad_shape (tuple): Shape after padding
        
    Added Keys:
        - mask (ndarray, optional): Padding mask if return_mask=True
        - pad_fixed_size (tuple): Fixed padding size used
        - pad_size_divisor (int): Size divisor used
    
    Args:
        size (tuple, optional): Fixed size to pad to (width, height).
        size_divisor (int, optional): Pad to size divisible by this number.
        pad_val (int): Value to use for padding. Defaults to 0.
        keep_ratio (bool): Whether to keep aspect ratio when resizing before padding.
            Defaults to False.
        return_mask (bool): Whether to return padding mask. Defaults to False.
        mask_ratio (int | tuple): Stride ratio for mask downsampling. Defaults to 2.
        train_state (bool): Whether in training state. Defaults to True.
    """

    def __init__(self,
                 size: Optional[Tuple[int, int]] = None,
                 size_divisor: Optional[int] = None,
                 pad_val: int = 0,
                 keep_ratio: bool = False,
                 return_mask: bool = False,
                 mask_ratio: Union[int, Tuple[int, int]] = 2,
                 train_state: bool = True):
        super().__init__()
        
        # Validate size type before reversing
        if size is not None and not isinstance(size, (tuple, list)):
            raise TypeError("size must be a tuple or list, not {}".format(type(size)))
            
        # Reverse size to match (width, height) format internally
        self.size = size[::-1] if size is not None else None
        self.size_divisor = size_divisor
        self.pad_val = pad_val
        self.keep_ratio = keep_ratio
        self.return_mask = return_mask
        self.mask_ratio = mask_ratio
        self.training = train_state
        
        # Validation: only one of size or size_divisor should be specified
        assert size is not None or size_divisor is not None, \
            "Either size or size_divisor must be specified"
        assert size is None or size_divisor is None, \
            "Only one of size or size_divisor can be specified"

    def _pad(self, img: np.ndarray, size: Tuple[int, int], pad_val: int) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Pad image to specified size.
        
        Args:
            img (ndarray): Input image
            size (tuple): Target size (width, height)
            pad_val (int): Padding value
            
        Returns:
            tuple: (padded_image, mask) where mask is None if return_mask=False
        """
        if not isinstance(size, tuple):
            raise NotImplementedError("Size must be a tuple")

        # Determine output shape
        if len(size) < len(img.shape):
            shape = size + (img.shape[-1], )
        else:
            shape = size

        # Create padded array
        pad = np.empty(shape, dtype=img.dtype)
        pad[...] = pad_val

        h, w = img.shape[:2]
        size_w, size_h = size[:2]
        
        # Resize if image is larger than target size
        if h > size_h or w > size_w:
            if self.keep_ratio:
                if h / size_h > w / size_w:
                    new_size = (int(w / h * size_h), size_h)
                else:
                    new_size = (size_w, int(h / w * size_w))
            else:
                new_size = size
            img = cv2.resize(img, new_size[::-1], cv2.INTER_LINEAR)
        
        # Place image in padded array
        pad[:img.shape[0], :img.shape[1], ...] = img
        
        # Create mask if requested
        mask = None
        if self.return_mask:
            mask = np.empty(size, dtype=img.dtype)
            mask[...] = 0
            mask[:img.shape[0], :img.shape[1]] = 1

            # Downsample mask according to mask_ratio (backbone stride)
            if isinstance(self.mask_ratio, int):
                mask = mask[::self.mask_ratio, ::self.mask_ratio]
            elif isinstance(self.mask_ratio, tuple):
                mask = mask[::self.mask_ratio[0], ::self.mask_ratio[1]]
            else:
                raise NotImplementedError("mask_ratio must be int or tuple")

            mask = np.expand_dims(mask, axis=0)

        return pad, mask

    def _pad_img(self, results: Dict) -> None:
        """Pad image in results dict.
        
        Args:
            results (dict): Results dict containing image
        """
        if self.size is not None:
            padded_img, mask = self._pad(results['img'], self.size, self.pad_val)
        elif self.size_divisor is not None:
            raise NotImplementedError("size_divisor padding not implemented yet")
        
        results['img'] = padded_img
        results['mask'] = mask
        results['pad_shape'] = padded_img.shape
        results['pad_fixed_size'] = self.size
        results['pad_size_divisor'] = self.size_divisor

    def transform(self, results: Dict) -> Dict:
        """Transform function to pad image.
        
        Args:
            results (dict): Result dict from loading pipeline.
            
        Returns:
            dict: Updated result dict with padded image.
        """
        self._pad_img(results)
        return results

    def __repr__(self) -> str:
        repr_str = self.__class__.__name__
        repr_str += f'(size={self.size}, '
        repr_str += f'size_divisor={self.size_divisor}, '
        repr_str += f'pad_val={self.pad_val}, '
        repr_str += f'keep_ratio={self.keep_ratio}, '
        repr_str += f'return_mask={self.return_mask})'
        return repr_str
