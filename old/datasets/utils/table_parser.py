import os
from mmocr.datasets.builder import PARSERS
import numpy as np

@PARSERS.register_module()
class TableParser:
    """PubTabNet Dataset parser for table structure and content recognition.
    This parser processes the raw annotation data into a structured format suitable for training and evaluation.

    The annotation format for PubTabNet:
    {
        'filename': str,
        'split': str,  # 'train', 'val', or 'test'
        'imgid': int,
        'html': {
            'structure': {'tokens': [str]},
            'cells': [
                {
                    'tokens': [str],
                    'bbox': [x0, y0, x1, y1]  # only non-empty cells have this
                }
            ]
        }
    }

    Output format:
    {
        'filename': str,
        'tokens': [str],
        'bbox': [[float]],
        'bbox_masks': [bool]
    }

    Args:
        max_structure_len (int): Maximum length of structure tokens.
        max_cell_len (int): Maximum length of cell tokens.

    """
    def __init__(self, max_structure_len: int = 500):
        self.max_structure_len = max_structure_len

    def get_item(self, data_ret, index):
        """Get item from data_ret at index."""
        raw_data_info = data_ret[index%len(data_ret)]

        try:
            html_data = raw_data_info.get('html', {})
            tokens = html_data.get('structure', {}).get('tokens', [])
            cells = html_data.get('cells', [])
            bboxes = [cell.get('bbox', []) for cell in cells]
            # advance parse bbox
            empty_bbox_mask = build_empty_bbox_mask(bboxes)
            bboxes, empty_bbox_mask = align_bbox_mask(bboxes, empty_bbox_mask, tokens)
            bboxes = np.array(bboxes)
            empty_bbox_mask = np.array(empty_bbox_mask)

            bbox_masks = build_bbox_mask(tokens)
            bbox_masks = bbox_masks * empty_bbox_mask

        
            return {
                'filename': raw_data_info.get('filename', ''),
                'tokens': tokens[:self.max_structure_len],
                'bbox': bboxes,
                'bbox_masks': bbox_masks
            }
        
        except Exception as e:
            print(f"Error parsing data info for {raw_data_info.get('filename', 'unknown')}: {e}")
            return None
        

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
    pattern = ['<td></td>', '<td', '<eb></eb>'] + [f'<eb{i}></eb{i}>' for i in range(1, 11)]
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
    pattern = ['<td></td>', '<td', '<eb></eb>'] + [f'<eb{i}></eb{i}>' for i in range(1, 11)]
    assert len(bboxes) == get_bbox_nums(tokens) == len(empty_bbox_mask)
    bbox_count = 0
    structure_token_nums = len(tokens)
    # init with [0,0,0,0], and change the real bbox to corresponding value
    aligned_bbox = [[0., 0., 0., 0.] for _ in range(structure_token_nums)]
    aligned_empty_bbox_mask = [1 for _ in range(structure_token_nums)]
    for idx, t in enumerate(tokens):
        if t in pattern:
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