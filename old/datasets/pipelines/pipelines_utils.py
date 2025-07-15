import os
import cv2
import random
import numpy as np

def visual_table_resized_bbox(results):
    bboxes = results['img_info']['bbox']
    img = results['img']
    for bbox in bboxes:
        img = cv2.rectangle(img, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0, 255, 0), thickness=1)
    return img

def visual_table_xywh_bbox(results):
    img = results['img']
    bboxes = results['img_info']['bbox']
    for bbox in bboxes:
        draw_bbox = np.empty_like(bbox)
        draw_bbox[0] = bbox[0] - bbox[2] / 2
        draw_bbox[1] = bbox[1] - bbox[3] / 2
        draw_bbox[2] = bbox[0] + bbox[2] / 2
        draw_bbox[3] = bbox[1] + bbox[3] / 2
        img = cv2.rectangle(img, (int(draw_bbox[0]), int(draw_bbox[1])), (int(draw_bbox[2]), int(draw_bbox[3])), (0, 255, 0), thickness=1)
    return img

def xyxy2xywh(bboxes):
    """
    Convert coord (x1,y1,x2,y2) to (x,y,w,h).
    where (x1,y1) is top-left, (x2,y2) is bottom-right.
    (x,y) is bbox center and (w,h) is width and height.
    :param bboxes: (x1, y1, x2, y2)
    :return:
    """
    new_bboxes = np.empty_like(bboxes)
    new_bboxes[..., 0] = (bboxes[..., 0] + bboxes[..., 2]) / 2 # x center
    new_bboxes[..., 1] = (bboxes[..., 1] + bboxes[..., 3]) / 2 # y center
    new_bboxes[..., 2] = bboxes[..., 2] - bboxes[..., 0] # width
    new_bboxes[..., 3] = bboxes[..., 3] - bboxes[..., 1] # height
    return new_bboxes


def normalize_bbox(bboxes, img_shape):
    bboxes[..., 0], bboxes[..., 2] = bboxes[..., 0] / img_shape[1], bboxes[..., 2] / img_shape[1]
    bboxes[..., 1], bboxes[..., 3] = bboxes[..., 1] / img_shape[0], bboxes[..., 3] / img_shape[0]
    return bboxes