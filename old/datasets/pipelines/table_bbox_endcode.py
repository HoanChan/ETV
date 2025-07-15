import os
import cv2
import numpy as np
from mmdet.datasets.builder import PIPELINES
from .pipelines_utils import xyxy2xywh, normalize_bbox

@PIPELINES.register_module()
class TableBboxEncode:
    """Encode table bbox for training.
    convert coord (x1,y1,x2,y2) to (x,y,w,h)
    normalize to (0,1)
    adjust key 'bbox' and 'bbox_mask' location in dictionary 'results'
    """
    def __init__(self):
        pass

    def __call__(self, results):
        bboxes = results['img_info']['bbox']
        bboxes = xyxy2xywh(bboxes)
        img_shape = results['img'].shape
        bboxes = normalize_bbox(bboxes, img_shape)
        flag = self.check_bbox_valid(bboxes)
        if not flag:
            print('Box invalid in {}'.format(results['filename']))
        results['img_info']['bbox'] = bboxes
        self.adjust_key(results)
        # self.visual_normalized_bbox(results)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        return repr_str

    def check_bbox_valid(self, bboxes):
        low = (bboxes >= 0.) * 1
        high = (bboxes <= 1.) * 1
        matrix = low + high
        for idx, m in enumerate(matrix):
            if m.sum() != 8:
                return False
        return True

    def visual_normalized_bbox(self, results):
        """
        visual after normalized bbox in results.
        :param results:
        :return:
        """
        save_path = '/data_0/cache/{}_normalized.jpg'.\
            format(os.path.basename(results['filename']).split('.')[0])
        img = results['img']
        img_shape = img.shape
        # x,y,w,h
        bboxes = results['img_info']['bbox']
        bboxes[..., 0::2] = bboxes[..., 0::2] * img_shape[1]
        bboxes[..., 1::2] = bboxes[..., 1::2] * img_shape[0]
        # x,y,x,y
        new_bboxes = np.empty_like(bboxes)
        new_bboxes[..., 0] = bboxes[..., 0] - bboxes[..., 2] / 2
        new_bboxes[..., 1] = bboxes[..., 1] - bboxes[..., 3] / 2
        new_bboxes[..., 2] = bboxes[..., 0] + bboxes[..., 2] / 2
        new_bboxes[..., 3] = bboxes[..., 1] + bboxes[..., 3] / 2
        # draw
        for new_bbox in new_bboxes:
            img = cv2.rectangle(img, (int(new_bbox[0]), int(new_bbox[1])),
                                   (int(new_bbox[2]), int(new_bbox[3])), (0, 255, 0), thickness=1)
        cv2.imwrite(save_path, img)

    def adjust_key(self, results):
        """
        Adjust key 'bbox' and 'bbox_mask' location in dictionary 'results'.
        :param results:
        :return:
        """
        bboxes = results['img_info'].pop('bbox')
        bboxes_masks = results['img_info'].pop('bbox_masks')
        results['bbox'] = bboxes
        results['bbox_masks'] = bboxes_masks
        return results