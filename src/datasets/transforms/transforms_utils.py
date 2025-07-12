# Copyright (c) Lê Hoàn Chân. All rights reserved.

import numpy as np

# empty box token dict, encoding for the token which is showed in image is blank.
empty_bbox_token_dict = {
    "[]": '<eb></eb>',
    "[' ']": '<eb1></eb1>',
    "['<b>', ' ', '</b>']": '<eb2></eb2>',
    "['\\u2028', '\\u2028']": '<eb3></eb3>',
    "['<sup>', ' ', '</sup>']": '<eb4></eb4>',
    "['<b>', '</b>']": '<eb5></eb5>',
    "['<i>', ' ', '</i>']": '<eb6></eb6>',
    "['<b>', '<i>', '</i>', '</b>']": '<eb7></eb7>',
    "['<b>', '<i>', ' ', '</i>', '</b>']": '<eb8></eb8>',
    "['<i>', '</i>']": '<eb9></eb9>',
    "['<b>', ' ', '\\u2028', ' ', '\\u2028', ' ', '</b>']": '<eb10></eb10>',
}

def merge_token(token_list: list) -> list:
    """
    This function used to merge the common tokens of raw tokens, and reduce the max length.
    eg. merge '<td>' and '</td>' to '<td></td>' which are always appear together.
    :param token_list: [list]. the raw tokens from the json line file.
    :return: merged tokens.
    """
    pointer = 0
    merge_token_list = []
    while pointer < len(token_list) and token_list[pointer] != '</tbody>':
        if token_list[pointer] == '<td>':
            tmp = token_list[pointer] + token_list[pointer+1]
            merge_token_list.append(tmp)
            pointer += 2
        else:
            merge_token_list.append(token_list[pointer])
            pointer += 1
    if pointer < len(token_list) and token_list[pointer] == '</tbody>':
        merge_token_list.append('</tbody>')
    return merge_token_list

def insert_empty_bbox_token(token_list: list, cells: list) -> list:
    """
    This function used to insert the empty bbox token(from empty_bbox_token_dict) to token_list.
    check every '<td></td>' and '<td'(table cell token), if 'bbox' not in cell dict, is a empty bbox.
    :param token_list: [list]. merged tokens.
    :param cells: [list]. list of table cell dict, each dict include cell's content and coord.
    :return: tokens add empty bbox str.
    """
    bbox_idx = 0
    add_empty_bbox_token_list = []
    for token in token_list:
        if token == '<td></td>' or token == '<td':
            if bbox_idx < len(cells) and 'bbox' not in cells[bbox_idx].keys():
                content = str(cells[bbox_idx]['tokens'])
                empty_bbox_token = empty_bbox_token_dict[content]
                add_empty_bbox_token_list.append(empty_bbox_token)
            else:
                add_empty_bbox_token_list.append(token)
            bbox_idx += 1
        else:
            add_empty_bbox_token_list.append(token)
    return add_empty_bbox_token_list

def get_thead_item_idx(token_list: list) -> list:
    """
    This function will return the index (start from 0) of cell, which is belong to table head.
    :param token_list: [list]. the raw tokens from the json line file.
    :return: list of index.
    """
    if '</thead>' not in token_list:
        return []
    count = 0
    while token_list[count] != '</thead>':
        count += 1
    thead_tokens = token_list[:count+1]
    cell_nums_in_thead = thead_tokens.count('</td>')
    return [i for i in range(cell_nums_in_thead)]

def remove_Bb(content: list) -> list:
    """
    This function will remove the '<b>' and '</b>' of the content.
    :param content: [list]. text content of each cell.
    :return: text content without '<b>' and '</b>'.
    """
    if '<b>' in content:
        content.remove('<b>')
    if '</b>' in content:
        content.remove('</b>')
    return content

def remove_thead_Bb(token_list: list):
    """
    This function will remove the '<b>' and '</b>' of the table head's content.
    :param token_list: [list]. the raw tokens from the json line file.
    :return: text content without '<b>' and '</b>'.
    """
    for idx in get_thead_item_idx(token_list):
        if '<b>' in token_list[idx]:
            token_list[idx] = remove_Bb(token_list[idx])
    return token_list

def process_token(token_list: list, cells: list) -> list:
    """
    This function will process the token list, merge the common tokens, insert empty bbox token.
    :param token_list: [list]. the raw tokens from the json line file.
    :param cells: [list]. list of table cell dict, each dict include cell's content and coord.
    :return: processed token list.
    """
    merged_token = merge_token(token_list)
    encoded_token = insert_empty_bbox_token(merged_token, cells)
    return encoded_token

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
def build_empty_bbox_mask(bboxes: list) -> list:
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

def get_bbox_nums(tokens: list) -> int:
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

def align_bbox_mask(bboxes: list, empty_bbox_mask: list, tokens: list) -> tuple:
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

def build_bbox_mask(tokens: list) -> np.ndarray:
    #TODO : need to debug to keep <eb></eb> or not.
    structure_token_nums = len(tokens)
    pattern = ['<td></td>', '<td', '<eb></eb>']
    mask = [0 for _ in range(structure_token_nums)]
    for idx, l in enumerate(tokens):
        if l in pattern:
           mask[idx] = 1
    return np.array(mask)