# Copyright (c) Lê Hoàn Chân. All rights reserved.
from typing import Dict, Optional, Tuple, Union
import numpy as np
from mmdet.datasets.transforms import Pad
from mmocr.registry import TRANSFORMS


@TRANSFORMS.register_module()
class TablePad(Pad):
    """TablePad kế thừa Pad từ mmdetection, thêm một số metadata đặc thù cho bài toán nhận diện bảng.
    
    Thêm các trường:
        - return_mask: Trả về mask của vùng hợp lệ (nếu cần)
        - mask_ratio: Tỷ lệ downsample mask (nếu cần dùng)
    """

    def __init__(self,
                 size: Optional[Tuple[int, int]] = None,
                 size_divisor: Optional[int] = None,
                 pad_val: int = 0,
                 pad_to_square: bool = False,
                 return_mask: bool = False,
                 mask_ratio: Union[int, Tuple[int, int]] = 1):
        super().__init__(
            size=size,
            size_divisor=size_divisor,
            pad_val=pad_val,
            pad_to_square=pad_to_square,
            # return_mask=return_mask
        )
        self.mask_ratio = mask_ratio
        self.return_mask = return_mask

    def transform(self, results: Dict) -> Dict:
        img = results['img'] # original image
        results = super().transform(results)

        # Gán thêm các trường bổ sung
        results['pad_fixed_size'] = self.size
        results['pad_size_divisor'] = self.size_divisor

        # Optional: Nếu muốn thay đổi mask theo mask_ratio
        if self.return_mask:
            mask = np.empty(self.size, dtype=img.dtype)
            mask[...] = 0 # fill mask
            mask[:img.shape[0], :img.shape[1]] = 1 # fill valid area
            if isinstance(self.mask_ratio, int):
                mask = mask[::self.mask_ratio, ::self.mask_ratio]
            elif isinstance(self.mask_ratio, tuple):
                mask = mask[::self.mask_ratio[0], ::self.mask_ratio[1]]
            else:
                raise NotImplementedError("mask_ratio must be int or tuple")
            results['mask'] = np.expand_dims(mask, axis=2) # add channel dimension

        return results
