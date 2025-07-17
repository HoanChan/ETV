import numpy as np
from mmcv.transforms import to_tensor
from mmcv.transforms.base import BaseTransform
from mmengine.structures import InstanceData, LabelData
import torchvision.transforms.functional as TF

from mmocr.registry import TRANSFORMS
from structures.table_master_data_sample import TableMasterDataSample

@TRANSFORMS.register_module()
class PackInputs(BaseTransform):
    """Pack the inputs data for text recognition or other tasks, collecting flexible keys.

    requires keys:
        - 'img': the input image tensor.
        - 'tokens': list of tokens for text recognition.

    optional keys:
        - 'valid_ratio': ratio of valid pixels in the image. Defaults to 1 if not found.

    Args:
        keys (Sequence[str]): Keys of results to be collected in output (besides image and annotation).
        meta_keys (Sequence[str], optional): Meta keys to be converted to metainfo. Defaults to
            ('img_path', 'ori_shape', 'img_shape', 'pad_shape', 'valid_ratio').
        mean (Sequence[float], optional): Mean values for each channel for image normalization.
        std (Sequence[float], optional): Std values for each channel for image normalization.
    """

    def __init__(self,
                 keys=(),
                 meta_keys=( 'img_path', 'ori_shape', 'img_shape', 'pad_shape', 'valid_ratio'),
                 mean=None,
                 std=None):
        self.keys = keys
        self.meta_keys = meta_keys
        self.mean = mean
        self.std = std

    def transform(self, results: dict) -> dict:
        """Method to pack the input data and collect flexible keys."""
        packed_results = dict()
        img_meta = {}
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
                img = img.float()  # Ensure tensor is float before normalization
                img = TF.normalize(img, self.mean, self.std)
                img_meta['img_norm_cfg'] = dict(mean=self.mean, std=self.std)
            packed_results['inputs'] = img

        # Pack annotation (token recog)
        data_sample = TableMasterDataSample()
        
        # Pack gt_instances for detection head (bboxes, masks, etc)
        gt_instances = InstanceData()
        
        # Pack bboxes if available
        bboxes = results.get('bboxes', None)
        if bboxes is not None:
            gt_instances.set_metainfo({'bboxes': bboxes})
                
        # Pack padded bboxes/masks directly into gt_instances (for loss computation)
        if 'padded_bboxes' in results:
            gt_instances.padded_bboxes = results['padded_bboxes']
        if 'padded_masks' in results:
            gt_instances.padded_masks = results['padded_masks']

        data_sample.gt_instances = gt_instances
        
        # Pack gt_tokens for recognition head (tokens)
        gt_tokens = LabelData()
        tokens = results.get('tokens', [])
        if tokens:
            assert isinstance(tokens, list), "tokens should be a list of tokens."
            assert all(isinstance(token, str) for token in tokens), "All tokens should be strings."
            gt_tokens.item = tokens
            
        # Pack token-level fields
        for key in ['padded_indexes', 'indexes', 'have_padded_indexes']:
            if key in results:
                setattr(gt_tokens, key, results[key])
                
        data_sample.gt_tokens = gt_tokens

        # Pack meta info
        for key in self.meta_keys:
            if key == 'valid_ratio':
                img_meta[key] = results.get('valid_ratio', 1)
            else:
                img_meta[key] = results.get(key, None)
        
        # Pack additional meta info from transforms
        for key in ['have_normalized_bboxes', 'have_padded_indexes', 'have_padded_bboxes']:
            if key in results:
                img_meta[key] = results[key]
                
        data_sample.set_metainfo(img_meta)

        packed_results['data_samples'] = data_sample

        # Pack other keys
        for key in self.keys:
            if key in results and key not in ['img', 'tokens', 'bboxes', 'valid_ratio']:
                packed_results[key] = results[key]
                
        return packed_results

    def __repr__(self) -> str:
        repr_str = self.__class__.__name__
        repr_str += f'(keys={self.keys}, meta_keys={self.meta_keys}'
        if self.mean is not None and self.std is not None:
            repr_str += f', mean={self.mean}, std={self.std}'
        repr_str += ')'
        return repr_str