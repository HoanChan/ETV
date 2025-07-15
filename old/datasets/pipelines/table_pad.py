import cv2
import numpy as np
from mmdet.datasets.builder import PIPELINES

@PIPELINES.register_module()
class TablePad:
    """Pad the image & mask.
    Two padding modes:
    (1) pad to fixed size.
    (2) pad to the minium size that is divisible by some number.
    """
    def __init__(self,
                 size=None,
                 size_divisor=None,
                 pad_val=None,
                 keep_ratio=False,
                 return_mask=False,
                 mask_ratio=2,
                 train_state=True,
                 ):
        self.size = size[::-1]
        self.size_divisor = size_divisor
        self.pad_val = pad_val
        self.keep_ratio = keep_ratio
        self.return_mask = return_mask
        self.mask_ratio = mask_ratio
        self.training = train_state
        # only one of size or size_divisor is valid.
        assert size is not None or size_divisor is not None
        assert size is None or size_divisor is None

    def _pad(self, img, size, pad_val):
        if not isinstance(size, tuple):
            raise NotImplementedError

        if len(size) < len(img.shape):
            shape = size + (img.shape[-1], )
        else:
            shape = size

        pad = np.empty(shape, dtype=img.dtype)
        pad[...] = pad_val

        h, w = img.shape[:2]
        size_w, size_h = size[:2]
        if h > size_h or w > size_w:
            if self.keep_ratio:
                if h / size_h > w / size_w:
                    size = (int(w / h * size_h), size_h)
                else:
                    size = (size_w, int(h / w * size_w))
            img = cv2.resize(img, size[::-1], cv2.INTER_LINEAR)
        pad[:img.shape[0], :img.shape[1], ...] = img
        if self.return_mask:
            mask = np.empty(size, dtype=img.dtype)
            mask[...] = 0
            mask[:img.shape[0], :img.shape[1]] = 1

            # mask_ratio is mean stride of backbone in (height, width)
            if isinstance(self.mask_ratio, int):
                mask = mask[::self.mask_ratio, ::self.mask_ratio]
            elif isinstance(self.mask_ratio, tuple):
                mask = mask[::self.mask_ratio[0], ::self.mask_ratio[1]]
            else:
                raise NotImplementedError

            mask = np.expand_dims(mask, axis=0)
        else:
            mask = None
        return pad, mask

    def _divisor(self, img, size_divisor, pad_val):
        pass

    def _pad_img(self, results):
        if self.size is not None:
            padded_img, mask = self._pad(results['img'], self.size, self.pad_val)
        elif self.size_divisor is not None:
            raise NotImplementedError
        results['img'] = padded_img
        results['mask'] = mask
        results['pad_shape'] = padded_img.shape
        results['pad_fixed_size'] = self.size
        results['pad_size_divisor'] = self.size_divisor

    def __call__(self, results):
        self._pad_img(results)
        #visual_img = visual_table_resized_bbox(results)
        #cv2.imwrite('/data_0/cache/{}_visual.jpg'.format(os.path.basename(results['filename']).split('.')[0]), visual_img)
        # if self.training:
            # scaleBbox(results)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += '(size={}, size_divisor={}, pad_val={})'.format(
            self.size, self.size_divisor, self.pad_val)
        return repr_str