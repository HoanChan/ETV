# Copyright (c) Lê Hoàn Chân. All rights reserved.
from typing import Dict, Optional, Tuple, Union
import numpy as np
from mmdet.datasets.transforms import Pad
from mmocr.registry import TRANSFORMS


@TRANSFORMS.register_module()
class TablePad(Pad):
    """Pad the image & segmentation map.

    There are three padding modes: (1) pad to a fixed size and (2) pad to the
    minimum size that is divisible by some number. and (3)pad to square. Also,
    pad to square and pad to the minimum size can be used as the same time.

    Required Keys:

    - img
    - gt_bboxes (BaseBoxes[torch.float32]) (optional)
    - gt_masks (BitmapMasks | PolygonMasks) (optional)
    - gt_seg_map (np.uint8) (optional)

    Modified Keys:

    - img
    - img_shape
    - gt_masks
    - gt_seg_map

    Added Keys:

    - pad_shape
    - pad_fixed_size
    - pad_size_divisor
    - mask (optional, if return_mask is True)

    Args:
        size (tuple, optional): Fixed padding size.
            Expected padding shape (width, height). Defaults to None.
        size_divisor (int, optional): The divisor of padded size. Defaults to
            None.
        pad_to_square (bool): Whether to pad the image into a square.
            Currently only used for YOLOX. Defaults to False.
        pad_val (Number | dict[str, Number], optional) - Padding value for if
            the pad_mode is "constant".  If it is a single number, the value
            to pad the image is the number and to pad the semantic
            segmentation map is 255. If it is a dict, it should have the
            following keys:

            - img: The value to pad the image.
            - seg: The value to pad the semantic segmentation map.
            Defaults to dict(img=0, seg=255).
        padding_mode (str): Type of padding. Should be: constant, edge,
            reflect or symmetric. Defaults to 'constant'.

            - constant: pads with a constant value, this value is specified
              with pad_val.
            - edge: pads with the last value at the edge of the image.
            - reflect: pads with reflection of image without repeating the last
              value on the edge. For example, padding [1, 2, 3, 4] with 2
              elements on both sides in reflect mode will result in
              [3, 2, 1, 2, 3, 4, 3, 2].
            - symmetric: pads with reflection of image repeating the last value
              on the edge. For example, padding [1, 2, 3, 4] with 2 elements on
              both sides in symmetric mode will result in
              [2, 1, 1, 2, 3, 4, 4, 3]
        return_mask (bool): Whether to return a mask indicating the valid area
            after padding. Defaults to False.
        mask_ratio (int | tuple[int, int]): The ratio to downsample the mask.
            If it is an int, the mask will be downsampled by that factor.
    """

    def __init__(self,
                 size: Optional[Tuple[int, int]] = None,
                 size_divisor: Optional[int] = None,
                 pad_val: int = 0,
                 pad_to_square: bool = False,
                 padding_mode: str = 'constant',
                 return_mask: bool = False,
                 mask_ratio: Union[int, Tuple[int, int]] = 1):
        super().__init__(
            size=size,
            size_divisor=size_divisor,
            pad_val=pad_val,
            pad_to_square=pad_to_square,
            padding_mode=padding_mode,
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
    
    def __repr__(self):
        return (f"{self.__class__.__name__}(size={self.size}, size_divisor={self.size_divisor}, "
                f"pad_val={self.pad_val}, pad_to_square={self.pad_to_square}, "
                f"padding_mode='{self.padding_mode}', return_mask={self.return_mask}, "
                f"mask_ratio={self.mask_ratio})")
