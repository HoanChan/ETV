import cv2
import random
import numpy as np
from mmdet.datasets.builder import PIPELINES

@PIPELINES.register_module()
class TableResize:
    """Image resizing and padding for Table Recognition OCR, Table Structure Recognition.

    Args:
        height (int | tuple(int)): Image height after resizing.
        min_width (none | int | tuple(int)): Image minimum width
            after resizing.
        max_width (none | int | tuple(int)): Image maximum width
            after resizing.
        keep_aspect_ratio (bool): Keep image aspect ratio if True
            during resizing, Otherwise resize to the size height *
            max_width.
        img_pad_value (int): Scalar to fill padding area.
        width_downsample_ratio (float): Downsample ratio in horizontal
            direction from input image to output feature.
        backend (str | None): The image resize backend type. Options are `cv2`,
            `pillow`, `None`. If backend is None, the global imread_backend
            specified by ``mmcv.use_backend()`` will be used. Default: None.
    """

    def __init__(self,
                 img_scale=None,
                 min_size=None,
                 ratio_range=None,
                 interpolation=None,
                 keep_ratio=True,
                 long_size=None):
        self.img_scale = img_scale
        self.min_size = min_size
        self.ratio_range = ratio_range
        self.interpolation = cv2.INTER_LINEAR
        self.long_size = long_size
        self.keep_ratio = keep_ratio

    def _get_resize_scale(self, w, h):
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
                raise NotImplementedError

    def _resize_bboxes(self, results):
        img_shape = results['img_shape']
        if 'img_info' in results.keys():
            # train and validate phase
            if results['img_info'].get('bbox', None) is not None:
                bboxes = results['img_info']['bbox']
                scale_factor = results['scale_factor']
                # bboxes[..., 0::2], bboxes[..., 1::2] = \
                #     bboxes[..., 0::2] * scale_factor[1], bboxes[..., 1::2] * scale_factor[0]
                bboxes[..., 0::2] = np.clip(bboxes[..., 0::2] * scale_factor[1], 0, img_shape[1]-1)
                bboxes[..., 1::2] = np.clip(bboxes[..., 1::2] * scale_factor[0], 0, img_shape[0]-1)
                results['img_info']['bbox'] = bboxes
            else:
                raise ValueError('results should have bbox keys.')
        else:
            # testing phase
            pass

    def _resize_img(self, results):
        img = results['img']
        h, w, _ = img.shape

        if self.min_size is not None:
            if w > h:
                w = self.min_size / h * w
                h = self.min_size
            else:
                h = self.min_size / w * h
                w = self.min_size

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

    def __call__(self, results):
        self._resize_img(results)
        self._resize_bboxes(results)
        return results
