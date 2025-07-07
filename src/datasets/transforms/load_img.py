from mmcv.transforms import LoadImageFromFile
from mmocr.registry import TRANSFORMS

@TRANSFORMS.register_module()
class LoadImage(LoadImageFromFile):
    """Custom transform to load images."""

    def __init__(self, with_bbox=True, **kwargs):
        super().__init__(**kwargs)
        self.with_bbox = with_bbox

    def transform(self, results):
        results = super().transform(results)
        if self.with_bbox:
            img_shape = results['img'].shape
            height, width = img_shape[0], img_shape[1]
            results['gt_bboxes'] = [[0, 0, width, height]]
        return results