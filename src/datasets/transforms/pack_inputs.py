import numpy as np
from mmcv.transforms import to_tensor
from mmcv.transforms.base import BaseTransform
from mmengine.structures import LabelData
import torchvision.transforms.functional as TF

from mmocr.registry import TRANSFORMS
from mmocr.structures import (TextRecogDataSample)

@TRANSFORMS.register_module()
class PackInputs(BaseTransform):
    """Pack the inputs data for text recognition or other tasks, collecting flexible keys.

    Args:
        keys (Sequence[str]): Keys of results to be collected in output (besides image and annotation).
        meta_keys (Sequence[str], optional): Meta keys to be converted to metainfo. Defaults to
            ('img_path', 'ori_shape', 'img_shape', 'pad_shape', 'valid_ratio').
        mean (Sequence[float], optional): Mean values for each channel for normalization.
        std (Sequence[float], optional): Std values for each channel for normalization.
    """

    def __init__(self,
                 keys=(),
                 meta_keys=(
                     'img_path', 'ori_shape', 'img_shape', 'pad_shape',
                     'valid_ratio'),
                 mean=None,
                 std=None):
        self.keys = keys
        self.meta_keys = meta_keys
        self.mean = mean
        self.std = std

    def transform(self, results: dict) -> dict:
        """Method to pack the input data and collect flexible keys."""
        packed_results = dict()
        # Pack image
        if 'img' in results:
            img = results['img']
            if len(img.shape) < 3:
                img = np.expand_dims(img, -1)
            if img.flags.c_contiguous:
                img = to_tensor(img)
                img = img.permute(2, 0, 1).contiguous()
            else:
                img = np.ascontiguousarray(img.transpose(2, 0, 1))
                img = to_tensor(img)
            # Normalize if mean and std are provided
            if self.mean is not None and self.std is not None:
                img = TF.normalize(img, self.mean, self.std)
                packed_results['img_norm_cfg'] = dict(mean=self.mean, std=self.std)
            packed_results['inputs'] = img

        # Pack annotation (text recog)
        data_sample = TextRecogDataSample()
        gt_text = LabelData()
        if results.get('gt_texts', None):
            assert len(
                results['gt_texts']
            ) == 1, 'Each image sample should have one text annotation only'
            gt_text.item = results['gt_texts'][0]
        data_sample.gt_text = gt_text

        # Pack meta info
        img_meta = {}
        for key in self.meta_keys:
            if key == 'valid_ratio':
                img_meta[key] = results.get('valid_ratio', 1)
            else:
                img_meta[key] = results.get(key, None)
        data_sample.set_metainfo(img_meta)
        packed_results['data_samples'] = data_sample

        # Pack other keys
        for key in self.keys:
            if key in results and key not in ['img', 'gt_texts']:
                packed_results[key] = results[key]
        return packed_results

    def __repr__(self) -> str:
        repr_str = self.__class__.__name__
        repr_str += f'(keys={self.keys}, meta_keys={self.meta_keys}'
        if self.mean is not None and self.std is not None:
            repr_str += f', mean={self.mean}, std={self.std}'
        repr_str += ')'
        return repr_str