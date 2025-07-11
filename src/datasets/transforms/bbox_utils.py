# Copyright (c) Lê Hoàn Chân. All rights reserved.
from typing import Dict
import numpy as np


def xyxy2xywh(bboxes: np.ndarray) -> np.ndarray:
    """Convert coordinate format from (x1,y1,x2,y2) to (x,y,w,h).
    
    Where (x1,y1) is top-left, (x2,y2) is bottom-right.
    (x,y) is bbox center and (w,h) is width and height.
    
    Args:
        bboxes (ndarray): Bounding boxes in xyxy format (..., 4)
        
    Returns:
        ndarray: Bounding boxes in xywh format (..., 4)
    """
    new_bboxes = np.empty_like(bboxes)
    new_bboxes[..., 0] = (bboxes[..., 0] + bboxes[..., 2]) / 2  # x center
    new_bboxes[..., 1] = (bboxes[..., 1] + bboxes[..., 3]) / 2  # y center
    new_bboxes[..., 2] = bboxes[..., 2] - bboxes[..., 0]        # width
    new_bboxes[..., 3] = bboxes[..., 3] - bboxes[..., 1]        # height
    return new_bboxes


def normalize_bbox(bboxes: np.ndarray, img_shape: tuple) -> np.ndarray:
    """Normalize bounding boxes to [0, 1] range.
    
    Args:
        bboxes (ndarray): Bounding boxes to normalize
        img_shape (tuple): Image shape (height, width, channels)
        
    Returns:
        ndarray: Normalized bounding boxes
    """
    bboxes = bboxes.copy()
    bboxes[..., 0] = bboxes[..., 0] / img_shape[1]  # normalize x
    bboxes[..., 2] = bboxes[..., 2] / img_shape[1]  # normalize width
    bboxes[..., 1] = bboxes[..., 1] / img_shape[0]  # normalize y
    bboxes[..., 3] = bboxes[..., 3] / img_shape[0]  # normalize height
    return bboxes


def xywh2xyxy(bboxes: np.ndarray) -> np.ndarray:
    """Convert coordinate format from (x,y,w,h) to (x1,y1,x2,y2).
    
    Where (x,y) is bbox center and (w,h) is width and height.
    (x1,y1) is top-left, (x2,y2) is bottom-right.
    
    Args:
        bboxes (ndarray): Bounding boxes in xywh format (..., 4)
        
    Returns:
        ndarray: Bounding boxes in xyxy format (..., 4)
    """
    new_bboxes = np.empty_like(bboxes)
    new_bboxes[..., 0] = bboxes[..., 0] - bboxes[..., 2] / 2  # x1
    new_bboxes[..., 1] = bboxes[..., 1] - bboxes[..., 3] / 2  # y1
    new_bboxes[..., 2] = bboxes[..., 0] + bboxes[..., 2] / 2  # x2
    new_bboxes[..., 3] = bboxes[..., 1] + bboxes[..., 3] / 2  # y2
    return new_bboxes

# some functions for table structure label parse.
def build_empty_bbox_mask(bboxes):
    """
    Generate a mask, 0 means empty bbox, 1 means non-empty bbox.
    :param bboxes: list[list] bboxes list
    :return: flag matrix.
    """
    flag = [1 for _ in range(len(bboxes))]
    for i, bbox in enumerate(bboxes):
        # empty bbox coord in label files
        if bbox == [0,0,0,0]:
            flag[i] = 0
    return flag

def get_bbox_nums(tokens):
    pattern = ['<td></td>', '<td', '<eb></eb>',
               '<eb1></eb1>', '<eb2></eb2>', '<eb3></eb3>',
               '<eb4></eb4>', '<eb5></eb5>', '<eb6></eb6>',
               '<eb7></eb7>', '<eb8></eb8>', '<eb9></eb9>',
               '<eb10></eb10>']
    count = 0
    for t in tokens:
        if t in pattern:
            count += 1
    return count

def align_bbox_mask(bboxes, empty_bbox_mask, tokens):
    """
    This function is used to in insert [0,0,0,0] in the location, which corresponding
    structure tokens is non-bbox tokens(not <td> style structure token, eg. <thead>, <tr>)
    in raw tokens file. This function will not insert [0,0,0,0] in the empty bbox location,
    which is done in tokens-preprocess.

    :param bboxes: list[list] bboxes list
    :param empty_bboxes_mask: the empty bbox mask
    :param tokens: table structure tokens
    :return: aligned bbox structure tokens
    """
    pattern = ['<td></td>', '<td', '<eb></eb>',
               '<eb1></eb1>', '<eb2></eb2>', '<eb3></eb3>',
               '<eb4></eb4>', '<eb5></eb5>', '<eb6></eb6>',
               '<eb7></eb7>', '<eb8></eb8>', '<eb9></eb9>',
               '<eb10></eb10>']
    assert len(bboxes) == get_bbox_nums(tokens) == len(empty_bbox_mask)
    bbox_count = 0
    structure_token_nums = len(tokens)
    # init with [0,0,0,0], and change the real bbox to corresponding value
    aligned_bbox = [[0., 0., 0., 0.] for _ in range(structure_token_nums)]
    aligned_empty_bbox_mask = [1 for _ in range(structure_token_nums)]
    for idx, l in enumerate(tokens):
        if l in pattern:
            aligned_bbox[idx] = bboxes[bbox_count]
            aligned_empty_bbox_mask[idx] = empty_bbox_mask[bbox_count]
            bbox_count += 1
    return aligned_bbox, aligned_empty_bbox_mask

def build_bbox_mask(tokens):
    #TODO : need to debug to keep <eb></eb> or not.
    structure_token_nums = len(tokens)
    pattern = ['<td></td>', '<td', '<eb></eb>']
    mask = [0 for _ in range(structure_token_nums)]
    for idx, l in enumerate(tokens):
        if l in pattern:
           mask[idx] = 1
    return np.array(mask)