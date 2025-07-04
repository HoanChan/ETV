import os
import cv2
import numpy as np
from typing import Dict, List, Optional, Tuple
from PIL import Image
from mmengine.fileio import get_local_path

from .table_dataset import PubTabNetDataset

try:
    from mmocr.registry import DATASETS
except ImportError:
    from mmengine.registry import Registry
    DATASETS = Registry('dataset')


@DATASETS.register_module()
class CellDataset(PubTabNetDataset):
    """Cell Dataset for individual cell content recognition.
    Inherits from PubTabNetDataset, but each sample is a cell (not a table).
    """

    METAINFO = dict(
        dataset_name='CellDataset',
        task_name='cell_recognition',
        paper_info=dict(
            title='Cell-level Dataset for Table Content Recognition',
            authors='Custom Implementation',
            url=''
        )
    )

    def __init__(self,
                 padding: int = 5,
                 resize_to: Optional[Tuple[int, int]] = None,
                 filter_empty_cells: bool = True,
                 **kwargs):
        kwargs['task_type'] = 'content'
        kwargs['ignore_empty_cells'] = filter_empty_cells
        self.padding = padding
        self.resize_to = resize_to
        self.filter_empty_cells = filter_empty_cells
        super().__init__(**kwargs)

    def load_data_list(self) -> List[Dict]:
        """Load data list from annotation file and create cell-level samples."""
        table_data_list = super().load_data_list()
        data_list = []
        for table_info in table_data_list:
            img_path = table_info['img_path']
            imgid = table_info.get('sample_idx', 0)
            split = table_info['img_info'].get('split', 'train')
            for idx, inst in enumerate(table_info['instances']):
                if inst.get('task_type') != 'content':
                    continue
                bbox = inst.get('bbox', [])
                if len(bbox) != 4:
                    continue
                if self.filter_empty_cells and not inst.get('text'):
                    continue
                data_info = {
                    'img_path': img_path,
                    'cell_bbox': bbox,
                    'sample_idx': f"{imgid}_{idx}",
                    'original_imgid': imgid,
                    'cell_id': idx,
                    'instances': [inst],
                    'img_info': {
                        'height': None,
                        'width': None,
                        'split': split,
                        'original_bbox': bbox,
                        'padding': self.padding
                    }
                }
                data_list.append(data_info)
        return data_list

    def crop_cell_image(self, img_path: str, bbox: List[int]) -> Optional[np.ndarray]:
        try:
            image = cv2.imread(img_path)
            if image is None:
                pil_image = Image.open(img_path)
                image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
            if image is None:
                return None
            h, w = image.shape[:2]
            x0, y0, x1, y1 = bbox
            x0 = max(0, x0 - self.padding)
            y0 = max(0, y0 - self.padding)
            x1 = min(w, x1 + self.padding)
            y1 = min(h, y1 + self.padding)
            cell_image = image[y0:y1, x0:x1]
            if self.resize_to is not None:
                cell_image = cv2.resize(cell_image, self.resize_to)
            return cell_image
        except Exception as e:
            print(f"Error cropping cell from {img_path}: {e}")
            return None

    def get_data_info(self, idx: int) -> Dict:
        data_info = self.data_list[idx].copy()
        cell_image = self.crop_cell_image(data_info['img_path'], data_info['cell_bbox'])
        if cell_image is not None:
            h, w = cell_image.shape[:2]
            data_info['img_info']['height'] = h
            data_info['img_info']['width'] = w
            data_info['img'] = cell_image
        data_info['task_type'] = self.task_type
        return data_info

    def __repr__(self) -> str:
        repr_str = (f'{self.__class__.__name__}('
                   f'task_type={self.task_type}, '
                   f'split_filter={self.split_filter}, '
                   f'num_samples={len(self)}, '
                   f'padding={self.padding}, '
                   f'resize_to={self.resize_to}, '
                   f'ann_file={self.ann_file})')
        return repr_str