import copy
import cv2
import numpy as np
from typing import Dict, List, Optional
from PIL import Image

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

    def __init__(self, ignore_empty_cells: bool = True, max_cell_len: int = 150, **kwargs):
        kwargs['task_type'] = 'content' # load only cell data in super class
        kwargs['ignore_empty_cells'] = ignore_empty_cells
        kwargs['max_cell_len'] = max_cell_len
        super().__init__(**kwargs)

    def load_data_list(self) -> List[Dict]:
        """Load data list from annotation file and create cell-level samples."""
        table_data_list = super().load_data_list()
        data_list = []
        for table_info in table_data_list:
            img_path = table_info['img_path']
            imgid = table_info.get('sample_idx', 0)
            # Filter only content instances (skip structure instances)
            content_instances = [inst for inst in table_info['instances'] 
                               if inst.get('task_type', 'content') == 'content']
            
            for inst in content_instances:
                bbox = inst.get('bbox', [])
                if self.ignore_empty_cells and not inst.get('text'): continue
                if len(bbox) != 4: continue
                cell_id = inst.get('cell_id', 0)
                data_info = {
                    'img_path': img_path,
                    'height': None,
                    'width': None,
                    'bbox': bbox,
                    'sample_idx': f"{imgid}_{cell_id}",
                    'original_imgid': imgid,
                    'instances': [inst],
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
            x0, y0, x1, y1 = bbox
            cell_image = image[y0:y1, x0:x1]
            return cell_image
        except Exception as e:
            print(f"Error cropping cell from {img_path}: {e}")
            return None

    def get_data_info(self, idx: int) -> Dict:
        # Force initialize data_list if needed
        if len(self.data_list) == 0: # This might be a lazy init issue, try to force load
            try: self.data_list = self.load_data_list()
            except: pass
        
        if idx >= len(self.data_list):
            raise IndexError(f"Index {idx} out of range for data_list of length {len(self.data_list)}")
        data_info = copy.deepcopy(self.data_list[idx])
        cell_image = self.crop_cell_image(data_info['img_path'], data_info['bbox'])
        if cell_image is not None:
            h, w = cell_image.shape[:2]
            data_info['height'] = h
            data_info['width'] = w
            data_info['img'] = cell_image
        return data_info

    def __repr__(self) -> str:
        repr_str = (f'{self.__class__.__name__}('
                   f'task_type={self.task_type}, '
                   f'split_filter={self.split_filter}, '
                   f'num_samples={len(self)}, '
                   f'ann_file={self.ann_file})')
        return repr_str